"""Constrained FX graph-module ingestion (P6.3) — secondary, version-sensitive.

An FX ``GraphModule`` has no stable serialized contract: pickling it stores
arbitrary module state plus the *generated Python source* of ``forward``.
Safe mode therefore never rebuilds the module. Instead:

* ``pickletools.genops`` — a pure parser that executes nothing — walks the
  stream, harvesting the imports the pickle *would* perform (reported, never
  resolved) and every string literal;
* the string containing ``def forward(`` is parsed with :mod:`ast`, and the
  assignment/call structure of that generated code is rebuilt as nodes.

What cannot be recovered this way imports as *opaque* nodes with a
diagnostic, matching the artifact contract: topology partial, weights in a
companion artifact, editing and export unavailable, execution trusted-mode
only.
"""

from __future__ import annotations

import ast
import pickletools
from pathlib import Path
from typing import Final

from nneditor import __version__
from nneditor.adapters.pytorch.zip_store import ZipStoreError, read_member, zip_members
from nneditor.diagnostics import DiagnosticLog, Severity
from nneditor.ir.capabilities import ArtifactKind, contract_for
from nneditor.ir.core import (
    ArtifactRef,
    Document,
    Graph,
    JsonValue,
    Node,
    ProvenanceEntry,
    Value,
)
from nneditor.ir.identity import ROOT_GRAPH_ID, node_id, value_id
from nneditor.storage.reader import hash_file

__all__ = ["FxError", "open_fx_graph_module"]

_MAX_PICKLE_BYTES: Final = 256 * 1024 * 1024


class FxError(ValueError):
    """Raised when an FX artifact yields nothing that can be shown."""


def _read_direct(source: Path) -> bytes:
    """The raw file, refused before allocation when it exceeds the cap."""
    size = source.stat().st_size
    if size > _MAX_PICKLE_BYTES:
        raise FxError(
            f"{source.name} is {size} bytes; the pickle limit is {_MAX_PICKLE_BYTES}"
        )
    return source.read_bytes()


def _pickle_payload(source: Path) -> bytes:
    try:
        members = zip_members(source)
    except ZipStoreError:
        return _read_direct(source)
    for name, member in members.items():
        if name.endswith("data.pkl"):
            try:
                return read_member(source, member, limit=_MAX_PICKLE_BYTES)
            except ZipStoreError as error:
                raise FxError(
                    f"{source.name} holds an unreadable pickle: {error}"
                ) from error
    return _read_direct(source)


def _harvest(payload: bytes) -> tuple[list[str], list[str]]:
    """Imports referenced and string literals present — nothing is executed."""
    imports: list[str] = []
    strings: list[str] = []
    pending: list[str] = []
    try:
        for opcode, argument, _position in pickletools.genops(payload):
            if opcode.name in {"GLOBAL", "INST"} and isinstance(argument, str):
                imports.append(argument.replace(" ", "."))
            elif opcode.name == "STACK_GLOBAL" and len(pending) >= 2:
                imports.append(f"{pending[-2]}.{pending[-1]}")
            if opcode.name in {
                "BINUNICODE",
                "SHORT_BINUNICODE",
                "BINUNICODE8",
                "UNICODE",
                "STRING",
                "BINSTRING",
                "SHORT_BINSTRING",
            } and isinstance(argument, str):
                strings.append(argument)
                pending.append(argument)
                del pending[:-2]
    except ValueError as error:
        # pickletools.genops parses without executing, but a malformed
        # stream still makes it raise mid-iteration.
        raise FxError(f"the pickle stream could not be walked: {error}") from error
    return imports, strings


def _call_label(expression: ast.expr) -> str:
    if isinstance(expression, ast.Attribute):
        parent = _call_label(expression.value)
        return f"{parent}.{expression.attr}" if parent else expression.attr
    if isinstance(expression, ast.Name):
        return expression.id
    return ""


def _forward_nodes(
    code: str, log: DiagnosticLog
) -> tuple[list[Node], dict[str, Value], list[str], list[str]]:
    module = ast.parse(code)
    forward = next(
        (
            item
            for item in ast.walk(module)
            if isinstance(item, ast.FunctionDef) and item.name == "forward"
        ),
        None,
    )
    if forward is None:
        raise FxError("the recovered source holds no forward definition")

    values: dict[str, Value] = {}

    def declare(name: str) -> str:
        vid = value_id(ROOT_GRAPH_ID, name)
        if vid not in values:
            values[vid] = Value(id=vid, name=name)
        return vid

    inputs = [
        declare(argument.arg)
        for argument in forward.args.args
        if argument.arg != "self"
    ]

    nodes: list[Node] = []
    outputs: list[str] = []
    for ordinal, statement in enumerate(forward.body):
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if not isinstance(target, ast.Name):
                continue
            expression = statement.value
            # FX emits `name = None` after a value's last use to free it.
            # Those are liveness hints, not operations, and reusing the same
            # target name would otherwise mint a duplicate node.
            if isinstance(expression, ast.Constant) and expression.value is None:
                continue
            label = ""
            argument_ids: list[str] = []
            opaque = False
            if isinstance(expression, ast.Call):
                label = _call_label(expression.func)
                for argument in expression.args:
                    if isinstance(argument, ast.Name):
                        argument_ids.append(declare(argument.id))
                    elif not isinstance(argument, ast.Constant):
                        opaque = True
            else:
                opaque = True
            if not label:
                label = "opaque"
                opaque = True
            if opaque:
                log.add(
                    "pytorch.fx-opaque-node",
                    Severity.WARNING,
                    f"statement defining {target.id!r} could not be fully "
                    "interpreted from generated source; it imports as opaque",
                    target.id,
                )
            output_id = declare(target.id)
            derived_id, stability = node_id(
                ROOT_GRAPH_ID, name="", first_output=target.id, ordinal=ordinal
            )
            prefix = "opaque:" if opaque and label != "opaque" else ""
            nodes.append(
                Node(
                    id=derived_id,
                    id_stability=stability,
                    op_type=f"{prefix}{label}",
                    domain="pytorch.fx",
                    inputs=tuple(argument_ids),
                    outputs=(output_id,),
                )
            )
        elif isinstance(statement, ast.Return) and statement.value is not None:
            returned = statement.value
            elements = (
                list(returned.elts)
                if isinstance(returned, (ast.Tuple, ast.List))
                else [returned]
            )
            for element in elements:
                if isinstance(element, ast.Name):
                    outputs.append(declare(element.id))
    return nodes, values, inputs, outputs


def open_fx_graph_module(path: Path | str) -> Document:
    """Open a serialized FX graph module as a partial-topology document.

    Every failure — including unexpected ones from hostile input — surfaces
    as :class:`FxError`; the layered error contract in the session depends
    on adapters never leaking raw exceptions.
    """
    source = Path(path)
    try:
        return _open_fx_graph_module(source)
    except FxError:
        raise
    except Exception as error:
        raise FxError(
            f"{source.name} could not be opened as an FX graph module: "
            f"{type(error).__name__}: {error}"
        ) from error


def _open_fx_graph_module(source: Path) -> Document:
    payload = _pickle_payload(source)
    imports, strings = _harvest(payload)
    log = DiagnosticLog()

    code = next((item for item in strings if "def forward(" in item), None)
    nodes: list[Node] = []
    values: dict[str, Value] = {}
    inputs: list[str] = []
    outputs: list[str] = []
    if code is None:
        log.add(
            "pytorch.fx-no-source",
            Severity.ERROR,
            "no generated forward source was found in the artifact; the "
            "topology cannot be recovered in safe mode",
            source.name,
        )
    else:
        try:
            nodes, values, inputs, outputs = _forward_nodes(code, log)
        except (SyntaxError, ValueError, RecursionError) as error:
            log.add(
                "pytorch.fx-no-source",
                Severity.ERROR,
                f"the recovered source could not be parsed: {error}",
                source.name,
            )

    harvested: dict[str, JsonValue] = {
        "pickle_imports": [str(item) for item in sorted(set(imports))]
    }
    extensions: list[tuple[str, JsonValue]] = [("x-torch.fx", harvested)]
    if code is not None:
        extensions.append(("x-torch.fx_source", code))

    contract = contract_for(ArtifactKind.PYTORCH_FX_GRAPH_MODULE)
    return Document(
        source=ArtifactRef(
            path=str(source),
            content_hash=hash_file(source),
            byte_size=source.stat().st_size,
        ),
        artifact_kind=ArtifactKind.PYTORCH_FX_GRAPH_MODULE,
        capabilities=contract.statuses,
        graphs=[
            Graph(
                id=ROOT_GRAPH_ID,
                name="fx_forward",
                nodes=nodes,
                values=values.values(),
                inputs=inputs,
                outputs=outputs,
            )
        ],
        provenance=[
            ProvenanceEntry(
                operation="import",
                tool_version=f"nneditor {__version__}",
                target=ROOT_GRAPH_ID,
                parameters=(
                    ("loading_mode", "safe artifact"),
                    ("reader", "non-executing opcode walk plus ast recovery"),
                ),
            )
        ],
        diagnostics=tuple(log),
        extensions=extensions,
    )
