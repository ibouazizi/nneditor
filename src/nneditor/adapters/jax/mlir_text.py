"""A focused reader for textual MLIR / StableHLO (P7.1-P7.2).

Only the subset ``jax.export`` emits is interpreted, and the grammar is
narrow by design:

* ``module @name attributes {...} { ... }``
* ``func.func visibility @name(%arg: type loc(...), ...) -> (type {attrs}) { ... }``
* ``%r = dialect.op %operands ... : (in-types) -> out-type``
* ``return %values : types``

Tokenization respects strings, nested brackets, and ``//`` comments; regions
are captured verbatim so control-flow bodies survive as source extensions
even before their internals are modelled. Anything unrecognized becomes an
opaque operation with a diagnostic rather than a parse failure — a module
must open even when it uses a construct this build has not learned yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "MlirFunction",
    "MlirModule",
    "MlirOperation",
    "MlirParseError",
    "MlirType",
    "parse_module",
    "parse_type",
]


class MlirParseError(ValueError):
    """Raised when the text is not a readable MLIR module."""


@dataclass(frozen=True, slots=True)
class MlirType:
    """A parsed tensor type: element type plus dimensions."""

    element_type: str
    shape: tuple[int | str | None, ...] | None
    text: str

    @property
    def is_tensor(self) -> bool:
        return self.shape is not None


@dataclass(frozen=True, slots=True)
class MlirOperation:
    """One operation inside a function body."""

    results: tuple[str, ...]
    dialect: str
    name: str
    operands: tuple[str, ...]
    result_types: tuple[MlirType, ...]
    attributes: tuple[tuple[str, str], ...]
    regions: tuple[str, ...]
    text: str
    opaque: bool = False

    @property
    def qualified(self) -> str:
        return f"{self.dialect}.{self.name}" if self.dialect else self.name


@dataclass(frozen=True, slots=True)
class MlirFunction:
    """One ``func.func`` with its signature and body."""

    name: str
    visibility: str
    arguments: tuple[tuple[str, MlirType], ...]
    results: tuple[MlirType, ...]
    operations: tuple[MlirOperation, ...]
    returns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MlirModule:
    """A parsed module: its name, attributes, and functions."""

    name: str
    attributes: str
    functions: tuple[MlirFunction, ...] = field(default_factory=tuple)


_ELEMENT_TYPES: dict[str, str] = {
    "f64": "float64",
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "i64": "int64",
    "i32": "int32",
    "i16": "int16",
    "i8": "int8",
    "i4": "int4",
    "i1": "bool",
    "ui64": "uint64",
    "ui32": "uint32",
    "ui16": "uint16",
    "ui8": "uint8",
    "complex<f32>": "complex64",
    "complex<f64>": "complex128",
    "f8E4M3FN": "float8e4m3fn",
    "f8E5M2": "float8e5m2",
}

_TENSOR = re.compile(r"^tensor<(.*)>$", re.DOTALL)
# One dimension token and its `x` separator: `?` (dynamic), `*` (unranked),
# or an ASCII decimal run. `[0-9]` rather than `\d` — Unicode "digits" such
# as `²` and arabic-indic numerals are not MLIR dimension syntax, and
# ``str.isdigit`` would have accepted them straight into a failing ``int()``.
_DIM_TOKEN = re.compile(r"\s*(\?|\*|[0-9]+)\s*x")
_MAX_DIM_DIGITS = 18
# An element type never begins with an `x` separator, and the only things
# that may precede one are dimension characters. Anything else before a
# leading `x` is a dimension this grammar does not accept (`tensor<xxf32>`,
# `tensor<²xf32>`) rather than part of the element type (`index`).
_BAD_DIMENSION = re.compile(r"^[^A-Za-z_!#<(\[]*x")


def parse_type(text: str) -> MlirType:
    """Parse one type; non-tensor types keep their text and no shape."""
    stripped = text.strip()
    match = _TENSOR.match(stripped)
    if match is None:
        return MlirType(
            element_type=_ELEMENT_TYPES.get(stripped, stripped),
            shape=None,
            text=stripped,
        )
    body = match.group(1)
    # Drop an encoding attribute (`tensor<2x3xf32, #enc>`).
    depth = 0
    for index, character in enumerate(body):
        if character in "<[({":
            depth += 1
        elif character in ">])}":
            depth -= 1
        elif character == "," and depth == 0:
            body = body[:index]
            break
    # Dimensions are consumed left to right; the first token that is not a
    # dimension starts the element type. Splitting on every `x` would tear
    # element types apart (`complex<f32>` contains one).
    dims: list[int | str | None] = []
    cursor = 0
    while True:
        token_match = _DIM_TOKEN.match(body, cursor)
        if token_match is None:
            break
        token = token_match.group(1)
        if token in ("?", "*"):
            dims.append(None)
        else:
            if len(token) > _MAX_DIM_DIGITS:
                raise MlirParseError(
                    f"dimension {token[:24]}… in {stripped[:64]!r} is too "
                    "large to be a tensor extent"
                )
            dims.append(int(token))
        cursor = token_match.end()
    element = body[cursor:].strip()
    if not element:
        raise MlirParseError(f"tensor type {stripped[:64]!r} has no element type")
    if _BAD_DIMENSION.match(element):
        raise MlirParseError(
            f"tensor type {stripped[:64]!r} holds an empty or malformed dimension"
        )
    # `tensor<f32>` (rank 0) consumes no dimension tokens at all.
    return MlirType(
        element_type=_ELEMENT_TYPES.get(element, element),
        shape=tuple(dims),
        text=stripped,
    )


def _strip_comments(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        depth = 0
        cut = len(line)
        index = 0
        while index < len(line):
            character = line[index]
            if character == '"':
                index += 1
                while index < len(line) and line[index] != '"':
                    index += 2 if line[index] == "\\" else 1
            elif character == "/" and line[index : index + 2] == "//" and depth == 0:
                cut = index
                break
            index += 1
        out.append(line[:cut])
    return "\n".join(out)


def _split_top_level(text: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character == '"':
            current.append(character)
            index += 1
            while index < len(text) and text[index] != '"':
                current.append(text[index])
                if text[index] == "\\" and index + 1 < len(text):
                    current.append(text[index + 1])
                    index += 1
                index += 1
            current.append('"')
        elif character in "<[({":
            depth += 1
            current.append(character)
        elif character in ">])}":
            depth -= 1
            current.append(character)
        elif character == separator and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(character)
        index += 1
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _matching(text: str, start: int, opening: str, closing: str) -> int:
    """Index just past the bracket matching the one at ``start``."""
    depth = 0
    index = start
    while index < len(text):
        character = text[index]
        if character == '"':
            index += 1
            while index < len(text) and text[index] != '"':
                index += 2 if text[index] == "\\" else 1
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise MlirParseError(f"unbalanced {opening!r} starting at offset {start}")


_MODULE = re.compile(r"\bmodule\b(\s+@(?P<name>[\w.$-]+))?", re.DOTALL)
# Possessive quantifiers (`*+`): the adjacent optional groups would
# otherwise re-split whitespace runs quadratically on hostile input.
_FUNC = re.compile(
    r"\bfunc\.func\b\s*+(?:(?P<visibility>public|private|nested)\b\s*+)?"
    r"@(?P<name>[\w.$-]+)\s*+\(",
)
_OPERATION = re.compile(
    r"^(?P<results>%[\w#:.$-]+(?:\s*,\s*%[\w#:.$-]+)*\s*=\s*)?"
    # Both the pretty form (`stablehlo.add`), the generic quoted form
    # (`"stablehlo.case"(...)`), and bare builtin ops (`call @callee`).
    r'"?(?:(?P<dialect>[\w_]+)\.)?(?P<name>[\w_.]+)"?',
)


def _string_spans(text: str) -> list[tuple[int, int]]:
    """Half-open spans of double-quoted strings, so keyword searches can
    ignore matches that live inside them (MLIR location strings routinely
    contain the word ``module``)."""
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text[index] == '"':
            start = index
            index += 1
            while index < len(text) and text[index] != '"':
                index += 2 if text[index] == "\\" else 1
            spans.append((start, min(index + 1, len(text))))
        index += 1
    return spans


def _outside_strings(text: str, pattern: re.Pattern[str]) -> re.Match[str] | None:
    # Spans and matches are both in ascending order, so a single merged walk
    # replaces the per-match span scan that was quadratic in hostile input.
    spans = _string_spans(text)
    span_index = 0
    for match in pattern.finditer(text):
        position = match.start()
        while span_index < len(spans) and spans[span_index][1] <= position:
            span_index += 1
        if (
            span_index < len(spans)
            and spans[span_index][0] <= position < spans[span_index][1]
        ):
            continue
        return match
    return None


def _has_ssa_assignment(text: str) -> bool:
    """Whether ``%name =`` (an SSA assignment) occurs, via one linear pass.

    Equivalent to searching for ``%\\S+\\s*=`` but without that pattern's
    quadratic backtracking on hostile operand runs.
    """
    percent = False  # a '%' opened the current non-space run
    chars_after = 0  # non-space characters seen since that '%'
    pending = False  # a completed "%token" run awaiting optional space + '='
    for character in text:
        if character.isspace():
            if percent and chars_after >= 1:
                pending = True
            percent, chars_after = False, 0
            continue
        if character == "=" and (pending or (percent and chars_after >= 1)):
            return True
        pending = False
        if percent:
            chars_after += 1
        elif character == "%":
            percent, chars_after = True, 0
    return False


_ARGUMENT = re.compile(r"^(?P<name>%[\w.$-]+)\s*:\s*(?P<type>.+?)(?:\s+loc\(.*\))?$")


def _parse_operation(statement: str) -> MlirOperation:
    text = statement.strip()
    regions: list[str] = []
    body = text
    while True:
        brace = body.find("{")
        if brace < 0:
            break
        # An attribute dictionary directly after the operands is not a region;
        # regions are brace blocks that contain operations (a nested `=` or
        # `return`), which is what distinguishes them textually.
        end = _matching(body, brace, "{", "}")
        inner = body[brace + 1 : end - 1]
        if "return" in inner or _has_ssa_assignment(inner):
            regions.append(inner.strip())
            body = body[:brace] + body[end:]
            continue
        break

    match = _OPERATION.match(body)
    if match is None:
        return MlirOperation(
            results=(),
            dialect="",
            name="unknown",
            operands=(),
            result_types=(),
            attributes=(),
            regions=tuple(regions),
            text=text,
            opaque=True,
        )
    results = tuple(
        item.strip()
        # The capture keeps the assignment (`%0 = `); strip whitespace before
        # the `=` or the SSA name would carry it into every value id.
        for item in (match.group("results") or "").strip().rstrip("=").split(",")
        if item.strip()
    )
    remainder = body[match.end() :]

    signature = ""
    colon = -1
    depth = 0
    for index, character in enumerate(remainder):
        if character in "<[({":
            depth += 1
        elif character in ">])}":
            depth -= 1
        elif character == ":" and depth == 0:
            colon = index
            break
    head = remainder if colon < 0 else remainder[:colon]
    if colon >= 0:
        signature = remainder[colon + 1 :].strip()

    operands = tuple(token for token in re.findall(r"%[\w.$-]+", head.split("loc(")[0]))
    attributes: list[tuple[str, str]] = []
    brace = head.find("{")
    if brace >= 0:
        try:
            end = _matching(head, brace, "{", "}")
        except MlirParseError:
            end = len(head)
        for entry in _split_top_level(head[brace + 1 : end - 1]):
            key, _, value = entry.partition("=")
            if key.strip():
                attributes.append((key.strip(), value.strip()))
    for entry in re.findall(r"(\b[\w_]+)\s*=\s*(\[[^\]]*\]|[^,\s]+)", head):
        if entry[0] not in {key for key, _ in attributes}:
            attributes.append((entry[0], entry[1]))

    result_types: list[MlirType] = []
    if signature:
        arrow = signature.rfind("->")
        tail = signature[arrow + 2 :].strip() if arrow >= 0 else signature
        tail = tail.split("loc(")[0].strip()
        if tail.startswith("("):
            tail = tail[1 : _matching(tail, 0, "(", ")") - 1]
        for item in _split_top_level(tail):
            if item:
                result_types.append(parse_type(item))

    return MlirOperation(
        results=results,
        dialect=match.group("dialect") or "builtin",
        name=match.group("name"),
        operands=operands,
        result_types=tuple(result_types),
        attributes=tuple(attributes),
        regions=tuple(regions),
        text=text,
    )


def _statements(body: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    depth = 0
    index = 0
    while index < len(body):
        character = body[index]
        if character == '"':
            current.append(character)
            index += 1
            while index < len(body) and body[index] != '"':
                current.append(body[index])
                if body[index] == "\\" and index + 1 < len(body):
                    current.append(body[index + 1])
                    index += 1
                index += 1
            current.append('"')
        elif character == "-" and body[index : index + 2] == "->":
            # `->` is an arrow, not a bracket (see _parse_function).
            current.append("->")
            index += 2
            continue
        elif character in "<[({":
            depth += 1
            current.append(character)
        elif character in ">])}":
            depth -= 1
            current.append(character)
        elif character == "\n" and depth == 0:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
        else:
            current.append(character)
        index += 1
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def _parse_function(text: str, start: int) -> tuple[MlirFunction, int]:
    match = _FUNC.search(text, start)
    if match is None:
        raise MlirParseError("no function found")
    open_paren = match.end() - 1
    close_paren = _matching(text, open_paren, "(", ")")
    argument_text = text[open_paren + 1 : close_paren - 1]
    arguments: list[tuple[str, MlirType]] = []
    for entry in _split_top_level(argument_text):
        argument = _ARGUMENT.match(entry.strip())
        if argument is None:
            continue
        arguments.append((argument.group("name"), parse_type(argument.group("type"))))

    # A body-less declaration (`func.func private @callee(...) -> ...`) is
    # legal MLIR; scanning for its body past the next `func.func` would
    # swallow that function's body and mis-attribute its operations, so the
    # brace search never crosses the next function header.
    next_func = _FUNC.search(text, close_paren)
    limit = next_func.start() if next_func is not None else len(text)

    # The body brace is the first `{` at bracket depth zero: result lists can
    # carry their own attribute dictionaries (`-> (tensor<...> {attr})`), and
    # taking the first `{` blindly would cut the signature in half.
    brace = -1
    depth = 0
    index = close_paren
    while index < limit:
        character = text[index]
        if character == '"':
            index += 1
            while index < len(text) and text[index] != '"':
                index += 2 if text[index] == "\\" else 1
        elif character == "-" and text[index : index + 2] == "->":
            # The result arrow is not a bracket; counting its '>' would
            # unbalance the scan and mistake an attribute dict for the body.
            index += 1
        elif character in "(<[":
            depth += 1
        elif character in ")>]":
            depth -= 1
        elif character == "{":
            if depth == 0:
                brace = index
                break
            index = _matching(text, index, "{", "}") - 1
        index += 1

    def signature_results(signature_text: str) -> tuple[MlirType, ...]:
        results: list[MlirType] = []
        arrow = signature_text.find("->")
        if arrow >= 0:
            tail = signature_text[arrow + 2 :].strip()
            if tail.startswith("("):
                tail = tail[1 : _matching(tail, 0, "(", ")") - 1]
            for item in _split_top_level(tail):
                cleaned = item.split("{")[0].strip()
                if cleaned:
                    results.append(parse_type(cleaned))
        return tuple(results)

    if brace < 0:
        # A declaration: emit it signature-only and hand the cursor to the
        # next function without consuming anything beyond the signature.
        return (
            MlirFunction(
                name=match.group("name"),
                visibility=match.group("visibility") or "public",
                arguments=tuple(arguments),
                results=signature_results(text[close_paren:limit]),
                operations=(),
                returns=(),
            ),
            limit,
        )
    results = signature_results(text[close_paren:brace])

    end = _matching(text, brace, "{", "}")
    body = text[brace + 1 : end - 1]

    operations: list[MlirOperation] = []
    returns: tuple[str, ...] = ()
    for statement in _statements(body):
        if statement.startswith("return") or statement.startswith("func.return"):
            returns = tuple(
                token for token in re.findall(r"%[\w.$-]+", statement.split(":")[0])
            )
            continue
        operations.append(_parse_operation(statement))

    return (
        MlirFunction(
            name=match.group("name"),
            visibility=match.group("visibility") or "public",
            arguments=tuple(arguments),
            results=tuple(results),
            operations=tuple(operations),
            returns=returns,
        ),
        end,
    )


def parse_module(text: str) -> MlirModule:
    """Parse a textual MLIR module holding StableHLO functions."""
    cleaned = _strip_comments(text)
    module = _outside_strings(cleaned, _MODULE)
    if module is None:
        raise MlirParseError("the text declares no module")
    # `module @name attributes {...} { body }` — the attribute dictionary
    # opens a brace that is *not* the body, so skip it before locating one.
    cursor = module.end()
    attributes = ""
    tail = cleaned[cursor:]
    stripped = tail.lstrip()
    if stripped.startswith("attributes"):
        offset = cursor + (len(tail) - len(stripped)) + len("attributes")
        attribute_brace = cleaned.find("{", offset)
        if attribute_brace < 0:
            raise MlirParseError("module attributes are malformed")
        attribute_end = _matching(cleaned, attribute_brace, "{", "}")
        attributes = cleaned[attribute_brace + 1 : attribute_end - 1].strip()
        cursor = attribute_end
    brace = cleaned.find("{", cursor)
    if brace < 0:
        raise MlirParseError("the module has no body")
    end = _matching(cleaned, brace, "{", "}")
    body = cleaned[brace + 1 : end - 1]

    functions: list[MlirFunction] = []
    cursor = 0
    while True:
        if _FUNC.search(body, cursor) is None:
            break
        function, cursor = _parse_function(body, cursor)
        functions.append(function)
    if not functions:
        raise MlirParseError("the module declares no functions")
    return MlirModule(
        name=module.group("name") or "module",
        attributes=attributes,
        functions=tuple(functions),
    )
