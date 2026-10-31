"""Convert a Hugging Face snapshot index into a canonical IR document.

Tensor payloads stay locations in their shards. Topology is a synthetic LLM
graph derived from tensor names so architecture/block navigation has something
to show. Identity is the indexer's ``hf-index:`` digest.
"""

from __future__ import annotations

from pathlib import Path

from nneditor import __version__
from nneditor.adapters.huggingface.architecture import (
    ArchitecturePlan,
    LayerSpec,
    plan_architecture,
)
from nneditor.adapters.huggingface.snapshot import (
    HfIndex,
    HfTensorIndex,
    HuggingFaceError,
    index_huggingface,
)
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
    ExternalRef,
    Graph,
    JsonValue,
    Node,
    ProvenanceEntry,
    Storage,
    TensorRef,
    Value,
)
from nneditor.ir.identity import ROOT_GRAPH_ID, node_id, value_id

__all__ = ["index_to_document", "open_huggingface"]

_CHAIN_OPS = frozenset({"Embedding", "Attention", "FFN", "SSM", "RMSNorm", "Linear"})


def _tensor_from_index(tensor: HfTensorIndex) -> TensorRef:
    if tensor.storage is Storage.EXTERNAL:
        return TensorRef(
            id=tensor.id,
            element_type=tensor.element_type,
            dims=tensor.dims,
            storage=Storage.EXTERNAL,
            external=ExternalRef(tensor.shard, tensor.offset, tensor.length),
        )
    return TensorRef(
        id=tensor.id,
        element_type=tensor.element_type,
        dims=tensor.dims,
        storage=Storage.ABSENT,
    )


def _model_extension(index: HfIndex, plan: ArchitecturePlan) -> dict[str, JsonValue]:
    return {
        "architecture": plan.architecture,
        "name": plan.name,
        "block_count": plan.block_count,
        "vocab_size": plan.vocab_size,
        "embedding_length": plan.embedding_length,
        "context_length": plan.context_length,
        "tied_embeddings": plan.tied_embeddings,
        "n_tensors": len(index.tensors),
        "n_shards": len(index.shards),
        "n_members": len(index.members),
        "byte_size": index.byte_size,
    }


def _member_records(index: HfIndex) -> list[JsonValue]:
    return [
        {"name": member.name, "size": member.size, "kind": member.kind}
        for member in index.members
    ]


def _shard_records(index: HfIndex) -> list[JsonValue]:
    counts: dict[str, int] = {}
    for tensor in index.tensors:
        counts[tensor.shard] = counts.get(tensor.shard, 0) + 1
    sizes = {member.name: member.size for member in index.members}
    return [
        {
            "name": shard,
            "size": sizes.get(shard, 0),
            "tensors": counts.get(shard, 0),
        }
        for shard in index.shards
    ]


def _narrow_capabilities(
    tensors: list[TensorRef],
) -> tuple[tuple[CapabilityStatus, ...], tuple[CapabilityNote, ...]]:
    contract = contract_for(ArtifactKind.HUGGINGFACE_MODEL)
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
    if not tensors:
        statuses[Capability.WEIGHTS] = CapabilityStatus(
            Capability.WEIGHTS,
            Availability.UNAVAILABLE,
            "the snapshot lists no range-readable tensors; pickle shards "
            "are not unpickled in safe artifact mode.",
        )
        statuses[Capability.TOPOLOGY] = CapabilityStatus(
            Capability.TOPOLOGY,
            Availability.UNAVAILABLE,
            "no tensor names were found from which a synthetic graph could be derived.",
        )
        statuses[Capability.EXECUTION] = CapabilityStatus(
            Capability.EXECUTION,
            Availability.UNAVAILABLE,
            "the numpy LLM worker needs at least one readable weight tensor.",
        )
        statuses[Capability.TRACING] = CapabilityStatus(
            Capability.TRACING,
            Availability.UNAVAILABLE,
            "activation capture needs readable safetensors weights to run "
            "the isolated numpy worker.",
        )
    elif not readable:
        statuses[Capability.WEIGHTS] = CapabilityStatus(
            Capability.WEIGHTS,
            Availability.UNAVAILABLE,
            "listed tensors have no readable bytes; pickle shards are not "
            "unpickled in safe artifact mode.",
        )
        statuses[Capability.EXECUTION] = CapabilityStatus(
            Capability.EXECUTION,
            Availability.UNAVAILABLE,
            "the numpy LLM worker needs at least one readable weight tensor.",
        )
        statuses[Capability.TRACING] = CapabilityStatus(
            Capability.TRACING,
            Availability.UNAVAILABLE,
            "activation capture needs readable safetensors weights to run "
            "the isolated numpy worker.",
        )
    elif absent:
        statuses[Capability.WEIGHTS] = CapabilityStatus(
            Capability.WEIGHTS,
            Availability.PARTIAL,
            f"{len(absent)} of {len(tensors)} tensors have no readable "
            "bytes; see the capability notes on each tensor.",
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
            raise HuggingFaceError(
                f"synthetic Hugging Face node {spec.name!r} has no producer"
            )
        inputs: tuple[str, ...] = (previous,)
    else:
        inputs = ()
    nodes.append(
        Node(
            id=nid,
            id_stability=stability,
            op_type=spec.op_type,
            domain="nneditor.llm",
            inputs=inputs,
            outputs=(out_id,),
            attributes=_weight_attributes(spec, name_to_id),
            source_name=spec.source_name,
        )
    )
    return out_id if chained else previous


def _build_graph(index: HfIndex, plan: ArchitecturePlan) -> Graph:
    name_to_id = {tensor.name: tensor.id for tensor in index.tensors}
    values: dict[str, Value] = {}
    nodes: list[Node] = []
    chained = any(spec.op_type in _CHAIN_OPS for spec in plan.layers)
    previous: str | None = None
    inputs: tuple[str, ...] = ()
    if chained:
        tokens = Value(
            id=value_id(ROOT_GRAPH_ID, "tokens"),
            name="tokens",
            element_type="int64",
            shape=("batch", "seq"),
        )
        values[tokens.id] = tokens
        previous = tokens.id
        inputs = (tokens.id,)
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


def index_to_document(index: HfIndex) -> Document:
    """Build the canonical IR document for an indexed Hugging Face snapshot."""
    from nneditor.tracing.llm_support import with_trace_support

    names = tuple(tensor.name for tensor in index.tensors)
    dims = {tensor.name: tensor.dims for tensor in index.tensors}
    plan = plan_architecture(index.config, names, dims)
    tensors = [_tensor_from_index(tensor) for tensor in index.tensors]
    statuses, notes = _narrow_capabilities(tensors)
    graph = _build_graph(index, plan)
    tool_version = f"nneditor {__version__}"
    document = Document(
        source=ArtifactRef(
            path=str(index.path),
            content_hash=index.content_hash,
            byte_size=index.byte_size,
        ),
        artifact_kind=ArtifactKind.HUGGINGFACE_MODEL,
        capabilities=statuses,
        graphs=[graph],
        tensors=tensors,
        provenance=[
            ProvenanceEntry(
                operation="import",
                tool_version=tool_version,
                target=ROOT_GRAPH_ID,
                parameters=(("loading_mode", "safe artifact"),),
                source_artifact=index.content_hash,
            ),
            ProvenanceEntry(
                operation="synthesize-llm-architecture",
                tool_version=tool_version,
                target=ROOT_GRAPH_ID,
                parameters=(
                    ("architecture", plan.architecture),
                    ("block_count", str(plan.block_count)),
                    ("tied_embeddings", "true" if plan.tied_embeddings else "false"),
                ),
            ),
        ],
        diagnostics=tuple(index.diagnostics),
        capability_notes=notes,
        extensions=[
            ("x-huggingface.model", _model_extension(index, plan)),
            ("x-huggingface.config", dict(index.config)),
            ("x-huggingface.members", _member_records(index)),
            ("x-huggingface.shards", _shard_records(index)),
            ("x-huggingface.tokenizer", dict(index.tokenizer)),
        ],
    )
    return with_trace_support(document)


def open_huggingface(
    path: Path | str,
    token: CancellationToken | None = None,
) -> Document:
    """Index a Hugging Face snapshot directory and convert it to IR."""
    source = Path(path)
    try:
        return index_to_document(index_huggingface(source, token=token))
    except (HuggingFaceError, OperationCancelled):
        raise
    except Exception as error:
        raise HuggingFaceError(
            f"{source.name} could not be opened as a Hugging Face snapshot: "
            f"{type(error).__name__}: {error}"
        ) from error
