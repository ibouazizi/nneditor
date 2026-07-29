"""StableHLO ingestion into the canonical IR (P7.2).

Maps a parsed textual module onto the IR: the entry function becomes the
entry graph, other functions become their own graphs, operations become
nodes in the ``stablehlo`` (or originating dialect's) domain, SSA results
become values carrying their tensor types, and dynamic dimensions (``?`` or
named) survive as symbolic dimensions.

Constants and custom calls are *identified*, not interpreted: a
``stablehlo.constant`` keeps its literal in an attribute when small and is
otherwise reported, and every ``custom_call`` raises a diagnostic plus a
capability note, because its semantics live outside the artifact.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

from nneditor import __version__
from nneditor.adapters.jax.mlir_text import (
    MlirFunction,
    MlirModule,
    MlirOperation,
    MlirParseError,
    MlirType,
    parse_module,
)
from nneditor.diagnostics import DiagnosticLog, Severity
from nneditor.ir.capabilities import (
    ArtifactKind,
    Availability,
    Capability,
    contract_for,
)
from nneditor.ir.core import (
    ArtifactRef,
    Attribute,
    AttrKind,
    CapabilityNote,
    Document,
    Graph,
    JsonValue,
    Node,
    ProvenanceEntry,
    Value,
)
from nneditor.ir.identity import ROOT_GRAPH_ID, encode_segment, node_id, value_id
from nneditor.storage.reader import hash_file

__all__ = ["StableHloError", "open_stablehlo", "parse_stablehlo"]

_MAX_TEXT_BYTES: Final = 256 * 1024 * 1024
_MAX_ATTRIBUTE_CHARS: Final = 512


class StableHloError(ValueError):
    """Raised when a StableHLO artifact cannot be read."""


def _graph_id_for(function: MlirFunction, entry: str) -> str:
    if function.name == entry:
        return ROOT_GRAPH_ID
    return f"g:fn:stablehlo/{encode_segment(function.name)}"


def _value(graph_id: str, ssa: str, mlir_type: MlirType | None) -> Value:
    return Value(
        id=value_id(graph_id, ssa),
        name=ssa,
        element_type=mlir_type.element_type if mlir_type is not None else None,
        shape=mlir_type.shape if mlir_type is not None else None,
    )


def _attributes(
    operation: MlirOperation, log: DiagnosticLog, owner: str
) -> tuple[Attribute, ...]:
    attributes: list[Attribute] = []
    for name, raw in operation.attributes:
        text = raw.strip()
        if len(text) > _MAX_ATTRIBUTE_CHARS:
            log.add(
                "jax.attribute-truncated",
                Severity.INFO,
                f"attribute {name!r} of {operation.qualified} is "
                f"{len(text)} characters and is shown truncated",
                owner,
            )
            text = text[:_MAX_ATTRIBUTE_CHARS] + "…"
        try:
            attributes.append(Attribute(name, AttrKind.STRING, text))
        except Exception:  # pragma: no cover - defensive
            continue
    return tuple(attributes)


def _convert_function(
    function: MlirFunction,
    graph_id: str,
    log: DiagnosticLog,
    notes: list[CapabilityNote],
) -> Graph:
    values: dict[str, Value] = {}

    def declare(ssa: str, mlir_type: MlirType | None = None) -> str:
        vid = value_id(graph_id, ssa)
        existing = values.get(vid)
        if existing is None or (
            mlir_type is not None and existing.element_type is None
        ):
            values[vid] = _value(graph_id, ssa, mlir_type)
        return vid

    inputs = [
        declare(name, argument_type) for name, argument_type in function.arguments
    ]

    nodes: list[Node] = []
    for ordinal, operation in enumerate(function.operations):
        operand_ids = [declare(operand) for operand in operation.operands]
        result_ids: list[str] = []
        for index, result in enumerate(operation.results):
            result_type = (
                operation.result_types[index]
                if index < len(operation.result_types)
                else None
            )
            result_ids.append(declare(result, result_type))
        derived_id, stability = node_id(
            graph_id,
            name="",
            first_output=operation.results[0] if operation.results else "",
            ordinal=ordinal,
        )
        if operation.opaque:
            log.add(
                "jax.opaque-operation",
                Severity.WARNING,
                "an operation could not be interpreted and is shown opaquely: "
                + operation.text[:120],
                derived_id,
            )
        if operation.name.startswith("custom_call"):
            log.add(
                "jax.custom-call",
                Severity.INFO,
                f"{operation.qualified} calls out to a target defined outside "
                "the artifact; its semantics are unknown here",
                derived_id,
            )
            notes.append(
                CapabilityNote(
                    entity_id=derived_id,
                    capability=Capability.EDITING,
                    availability=Availability.UNAVAILABLE,
                    reason="custom calls have no schema in the artifact, so "
                    "edits cannot be validated",
                )
            )
        if operation.regions:
            log.add(
                "jax.region-not-modelled",
                Severity.INFO,
                f"{operation.qualified} carries {len(operation.regions)} "
                "nested region(s), preserved verbatim as source text",
                derived_id,
            )
        attributes = list(_attributes(operation, log, derived_id))
        for index, region in enumerate(operation.regions):
            attributes.append(
                Attribute(
                    f"region_{index}", AttrKind.STRING, region[:_MAX_ATTRIBUTE_CHARS]
                )
            )
        nodes.append(
            Node(
                id=derived_id,
                id_stability=stability,
                op_type=operation.name,
                domain=operation.dialect,
                inputs=tuple(operand_ids),
                outputs=tuple(result_ids),
                attributes=tuple(attributes),
            )
        )

    outputs = [declare(item) for item in function.returns]
    return Graph(
        id=graph_id,
        name=function.name,
        nodes=nodes,
        values=values.values(),
        inputs=inputs,
        outputs=outputs,
    )


def parse_stablehlo(text: str) -> MlirModule:
    """Parse StableHLO text, raising :class:`StableHloError` on failure."""
    try:
        return parse_module(text)
    except MlirParseError as error:
        raise StableHloError(str(error)) from error


def open_stablehlo(path: Path | str) -> Document:
    """Open a textual StableHLO module as an IR document.

    Every failure — including unexpected ones from hostile input — surfaces
    as :class:`StableHloError`; the layered error contract in the session
    depends on adapters never leaking raw exceptions.
    """
    source = Path(path)
    try:
        return _open_stablehlo(source)
    except StableHloError:
        raise
    except Exception as error:
        raise StableHloError(
            f"{source.name} could not be opened as a StableHLO module: "
            f"{type(error).__name__}: {error}"
        ) from error


def _open_stablehlo(source: Path) -> Document:
    size = source.stat().st_size
    if size > _MAX_TEXT_BYTES:
        raise StableHloError(
            f"{source.name} is {size} bytes; the textual-module limit is "
            f"{_MAX_TEXT_BYTES}"
        )
    raw = source.read_bytes()
    if b"\x00" in raw[:4096]:
        raise StableHloError(
            f"{source.name} looks like StableHLO *bytecode*; only the textual "
            "form is supported, so re-export with `mlir_module()` text"
        )
    module = parse_stablehlo(raw.decode("utf-8", errors="replace"))

    counts = Counter(function.name for function in module.functions)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        # Two functions with one name would collapse onto one graph id and
        # silently mis-attribute operations; the module is ambiguous.
        raise StableHloError(
            "the module defines duplicate function name(s): "
            + ", ".join(f"@{name}" for name in duplicates)
        )

    log = DiagnosticLog()
    notes: list[CapabilityNote] = []
    entry = next(
        (
            function.name
            for function in module.functions
            if function.name == "main" or function.visibility == "public"
        ),
        module.functions[0].name,
    )
    graphs = [
        _convert_function(function, _graph_id_for(function, entry), log, notes)
        for function in module.functions
    ]
    for graph in graphs:
        for value in graph.values:
            if value.shape is not None and any(
                isinstance(dim, str) for dim in value.shape
            ):
                log.add(
                    "jax.symbolic-dimension",
                    Severity.INFO,
                    f"value {value.name} has a symbolic extent; shape-dependent "
                    "validation stays conservative for paths through it",
                    value.id,
                )
                break

    contract = contract_for(ArtifactKind.JAX_STABLEHLO)
    extensions: list[tuple[str, JsonValue]] = [
        (
            "x-stablehlo.module",
            {"name": module.name, "attributes": module.attributes},
        )
    ]
    return Document(
        source=ArtifactRef(
            path=str(source), content_hash=hash_file(source), byte_size=size
        ),
        artifact_kind=ArtifactKind.JAX_STABLEHLO,
        capabilities=contract.statuses,
        graphs=graphs,
        entry_graph=ROOT_GRAPH_ID,
        provenance=[
            ProvenanceEntry(
                operation="import",
                tool_version=f"nneditor {__version__}",
                target=ROOT_GRAPH_ID,
                parameters=(
                    ("loading_mode", "safe artifact"),
                    ("reader", "native textual MLIR parser"),
                ),
            )
        ],
        diagnostics=tuple(log),
        capability_notes=notes,
        extensions=extensions,
    )
