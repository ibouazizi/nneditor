"""Convert a GGUF index into a canonical IR document.

Tensor payloads stay locations. Topology is a synthetic LLM graph derived
from tensor names so architecture/block navigation has something to show.
Identity is the indexer's ``gguf-index:`` digest — the converter never
re-reads the file.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from nneditor import __version__
from nneditor.adapters.gguf.architecture import (
    ArchitecturePlan,
    LayerSpec,
    plan_architecture,
)
from nneditor.adapters.gguf.index import GgufIndex, GgufTensorIndex
from nneditor.adapters.gguf.indexer import GgufIndexError, index_gguf
from nneditor.adapters.gguf.media import media_graph
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.ir.capabilities import (
    ArtifactKind,
    Availability,
    Capability,
    CapabilityStatus,
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
    PayloadRange,
    ProvenanceEntry,
    Storage,
    TensorRef,
    Value,
)
from nneditor.ir.identity import ROOT_GRAPH_ID, node_id, value_id

__all__ = ["index_to_document", "open_gguf"]

_CHAIN_OPS = frozenset(
    {
        "Embedding",
        "Attention",
        "FFN",
        "SSM",
        "RMSNorm",
        "Linear",
        "VisionBlock",
        "EncoderBlock",
        "DecoderBlock",
    }
)
_DOMAINS = {
    "VisionBlock": "nneditor.vision",
    "EncoderBlock": "nneditor.audio",
    "DecoderBlock": "nneditor.audio",
}


def _ir_dims(file_dims: tuple[int, ...]) -> tuple[int, ...]:
    """Numpy/PyTorch order: GGUF writes ggml ``ne[]``, innermost first."""
    return tuple(reversed(file_dims))


def _tensor_ref(tensor: GgufTensorIndex) -> TensorRef:
    payload = None
    storage = tensor.storage
    if storage is Storage.EMBEDDED_RAW:
        if tensor.payload is None:
            storage = Storage.ABSENT
        else:
            payload = PayloadRange(tensor.payload.offset, tensor.payload.length)
    return TensorRef(
        id=tensor.id,
        element_type=tensor.element_type,
        dims=_ir_dims(tensor.dims),
        storage=storage,
        payload=payload,
    )


def _json_kv(index: GgufIndex) -> dict[str, JsonValue]:
    return {entry.key: entry.value for entry in index.kv}


def _kv_records(index: GgufIndex) -> list[JsonValue]:
    return [
        {
            "key": entry.key,
            "type_id": entry.type_id,
            "offset": entry.offset,
            "length": entry.length,
            "elided": entry.elided,
            "value": entry.value,
        }
        for entry in index.kv
    ]


def _model_extension(index: GgufIndex, plan: ArchitecturePlan) -> dict[str, JsonValue]:
    return {
        "version": index.version,
        "endian": index.endian,
        "alignment": index.alignment,
        "n_tensors": index.n_tensors,
        "n_kv": index.n_kv,
        "kv_end": index.kv_end,
        "tensor_table_end": index.tensor_table_end,
        "data_start": index.data_start,
        "architecture": plan.architecture,
        "name": plan.name,
        "block_count": plan.block_count,
        "vocab_size": plan.vocab_size,
        "embedding_length": plan.embedding_length,
        "tied_embeddings": plan.tied_embeddings,
    }


def _narrow_capabilities(
    tensors: list[TensorRef],
) -> tuple[tuple[CapabilityStatus, ...], tuple[CapabilityNote, ...]]:
    contract = contract_for(ArtifactKind.GGUF_MODEL)
    statuses = {status.capability: status for status in contract.statuses}
    notes: list[CapabilityNote] = []
    absent = [tensor for tensor in tensors if tensor.storage is Storage.ABSENT]
    for tensor in absent:
        notes.append(
            CapabilityNote(
                entity_id=tensor.id,
                capability=Capability.WEIGHTS,
                availability=Availability.UNAVAILABLE,
                reason="the tensor's bytes could not be located",
            )
        )
    readable = [tensor for tensor in tensors if tensor.storage is not Storage.ABSENT]
    if not readable:
        statuses[Capability.WEIGHTS] = CapabilityStatus(
            Capability.WEIGHTS,
            Availability.UNAVAILABLE,
            "no tensor payloads could be located, so weights cannot be read.",
        )
        statuses[Capability.EXECUTION] = CapabilityStatus(
            Capability.EXECUTION,
            Availability.UNAVAILABLE,
            "the numpy LLM worker needs at least one readable weight tensor.",
        )
        statuses[Capability.TRACING] = CapabilityStatus(
            Capability.TRACING,
            Availability.UNAVAILABLE,
            "activation capture needs readable weights to run the isolated "
            "numpy worker.",
        )
    elif absent:
        statuses[Capability.WEIGHTS] = CapabilityStatus(
            Capability.WEIGHTS,
            Availability.PARTIAL,
            f"{len(absent)} of {len(tensors)} tensors have no readable bytes; "
            "see the capability notes on each tensor.",
        )
    return tuple(statuses.values()), tuple(notes)


def _hidden_shape(plan: ArchitecturePlan) -> tuple[str | int, ...]:
    width = plan.embedding_length if plan.embedding_length else "hidden"
    return ("batch", "seq", width)


def _output_shape(spec: LayerSpec, plan: ArchitecturePlan) -> tuple[str | int, ...]:
    if spec.op_type == "Linear":
        vocab = plan.vocab_size if plan.vocab_size else "vocab"
        return ("batch", "seq", vocab)
    return _hidden_shape(plan)


def _weight_attributes(
    spec: LayerSpec, name_to_id: dict[str, str]
) -> tuple[Attribute, ...]:
    attributes: list[Attribute] = []
    for binding in spec.weights:
        tensor_id = name_to_id.get(binding.tensor_name)
        if tensor_id is None:
            continue
        attributes.append(Attribute(binding.attribute, AttrKind.TENSOR, tensor_id))
    return tuple(attributes)


def _append_layer(
    spec: LayerSpec,
    ordinal: int,
    previous: str | None,
    plan: ArchitecturePlan,
    values: dict[str, Value],
    nodes: list[Node],
    name_to_id: dict[str, str],
) -> str | None:
    nid, stability = node_id(ROOT_GRAPH_ID, name=spec.name, ordinal=ordinal)
    out_id = value_id(ROOT_GRAPH_ID, spec.name)
    chained = spec.op_type in _CHAIN_OPS
    values[out_id] = Value(
        id=out_id,
        name=spec.name,
        element_type="float32" if chained else None,
        shape=_output_shape(spec, plan) if chained else None,
    )
    if chained:
        if previous is None:
            raise GgufIndexError(f"synthetic GGUF node {spec.name!r} has no producer")
        inputs: tuple[str, ...] = (previous,)
    else:
        inputs = ()
    nodes.append(
        Node(
            id=nid,
            id_stability=stability,
            op_type=spec.op_type,
            domain=_DOMAINS.get(spec.op_type, "nneditor.llm"),
            inputs=inputs,
            outputs=(out_id,),
            attributes=_weight_attributes(spec, name_to_id),
            source_name=spec.source_name,
        )
    )
    return out_id if chained else previous


def _build_graph(index: GgufIndex, plan: ArchitecturePlan) -> Graph:
    name_to_id = {tensor.name: tensor.id for tensor in index.tensors}
    values: dict[str, Value] = {}
    nodes: list[Node] = []
    chained = any(spec.op_type in _CHAIN_OPS for spec in plan.layers)
    previous: str | None = None
    inputs: tuple[str, ...] = ()
    if chained:
        kinds = {spec.op_type for spec in plan.layers}
        first = next(spec.op_type for spec in plan.layers if spec.op_type in _CHAIN_OPS)
        if first == "VisionBlock":
            feed_name, feed_type = "pixel-values", "float32"
            feed_shape: tuple[int | str, ...] = ("batch", "channels", "height", "width")
        elif first == "EncoderBlock":
            feed_name, feed_type = "input-features", "float32"
            feed_shape = ("batch", "mel", "frames")
        else:
            feed_name, feed_type = "tokens", "int64"
            feed_shape = ("batch", "seq")
        tokens = Value(
            id=value_id(ROOT_GRAPH_ID, feed_name),
            name=feed_name,
            element_type=feed_type,
            shape=feed_shape,
        )
        values[tokens.id] = tokens
        previous = tokens.id
        inputs = (tokens.id,)
        if "EncoderBlock" in kinds and "DecoderBlock" in kinds:
            decoder_tokens = Value(
                id=value_id(ROOT_GRAPH_ID, "decoder-tokens"),
                name="decoder-tokens",
                element_type="int64",
                shape=("batch", "target"),
            )
            values[decoder_tokens.id] = decoder_tokens
            inputs = (*inputs, decoder_tokens.id)
    for ordinal, spec in enumerate(plan.layers):
        previous = _append_layer(
            spec, ordinal, previous, plan, values, nodes, name_to_id
        )
    outputs = (previous,) if chained and previous is not None else ()
    return Graph(
        id=ROOT_GRAPH_ID,
        name=plan.name or plan.architecture or "llm",
        nodes=nodes,
        values=values.values(),
        inputs=inputs,
        outputs=outputs,
        initializers=[tensor.id for tensor in index.tensors],
    )


def index_to_document(index: GgufIndex) -> Document:
    """Build the canonical IR document for an indexed GGUF artifact."""
    from nneditor.tracing.llm_support import with_trace_support

    plan = plan_architecture(index)
    tensors = [_tensor_ref(tensor) for tensor in index.tensors]
    statuses, notes = _narrow_capabilities(tensors)
    graph_error = None
    synthesis = "synthesize-llm-architecture"
    try:
        media = media_graph(index)
        graph = media or _build_graph(index, plan)
        if media is not None:
            synthesis = "synthesize-modality-architecture"
            plan = replace(
                plan,
                block_count=sum(
                    node.op_type in {"VisionBlock", "EncoderBlock", "DecoderBlock"}
                    for node in graph.nodes
                ),
            )
    except ValueError as error:
        graph_error = str(error)
        graph = _build_graph(index, plan)
    tool_version = f"nneditor {__version__}"
    document = Document(
        source=ArtifactRef(
            path=str(index.path),
            content_hash=index.content_hash,
            byte_size=index.file_size,
        ),
        artifact_kind=ArtifactKind.GGUF_MODEL,
        capabilities=statuses,
        graphs=[graph],
        tensors=tensors,
        provenance=[
            ProvenanceEntry(
                operation="import",
                tool_version=tool_version,
                target=ROOT_GRAPH_ID,
                parameters=(
                    ("loading_mode", "safe artifact"),
                    ("gguf_version", str(index.version)),
                    ("endian", index.endian),
                ),
                source_artifact=index.content_hash,
            ),
            ProvenanceEntry(
                operation=synthesis,
                tool_version=tool_version,
                target=ROOT_GRAPH_ID,
                parameters=(
                    ("architecture", plan.architecture or ""),
                    ("block_count", str(plan.block_count)),
                    ("tied_embeddings", "true" if plan.tied_embeddings else "false"),
                ),
            ),
        ],
        diagnostics=tuple(index.diagnostics),
        capability_notes=notes,
        extensions=[
            ("x-gguf.model", _model_extension(index, plan)),
            ("x-gguf.metadata", _json_kv(index)),
            ("x-gguf.kv", _kv_records(index)),
            *([("x-gguf.trace-graph-error", graph_error)] if graph_error else []),
        ],
    )
    return with_trace_support(document)


def open_gguf(
    path: Path | str,
    token: CancellationToken | None = None,
) -> Document:
    """Index a GGUF file and convert it to an IR document."""
    source = Path(path)
    try:
        return index_to_document(index_gguf(source, token=token))
    except (GgufIndexError, OperationCancelled):
        raise
    except Exception as error:
        raise GgufIndexError(
            f"{source.name} could not be opened as GGUF: "
            f"{type(error).__name__}: {error}"
        ) from error
