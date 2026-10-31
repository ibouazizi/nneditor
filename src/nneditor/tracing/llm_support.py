"""Metadata-only preflight for the explicitly supported LLM execution paths."""

from __future__ import annotations

import math

from nneditor.adapters.gguf.dequant import supports_dequantization
from nneditor.ir.capabilities import (
    ArtifactKind,
    Availability,
    Capability,
    CapabilityStatus,
)
from nneditor.ir.core import AttrKind, Document, Node, Storage
from nneditor.tracing.llm_runtime import (
    LlmTraceError,
    _attention_role,
    _embed_role,
    _ffn_role,
    _tensor_name,
    hyperparams_from_document,
)

LLM_KINDS = frozenset({ArtifactKind.GGUF_MODEL, ArtifactKind.HUGGINGFACE_MODEL})


def _special_runtime_requirements(document: Document) -> tuple[str, ...]:
    """Identify incompatible execution semantics from metadata, without reads.

    Prism's type IDs are fork-specific. Name them only when the GGUF also
    declares Prism metadata; keep unknown IDs opaque in the general reader.
    """
    if document.artifact_kind is not ArtifactKind.GGUF_MODEL:
        return ()
    metadata = dict(document.extensions).get("x-gguf.metadata")
    if not isinstance(metadata, dict):
        return ()
    requirements: list[str] = []
    if metadata.get("general.architecture") == "qwen35":
        requirements.append("Qwen3.5 hybrid decoder with GatedDeltaNet layers")
    if any(key.startswith("prism.hadamard.") for key in metadata):
        types = {tensor.element_type for tensor in document.tensors.values()}
        for dtype, name in (("gguf.type_142", "PQ2_0"), ("gguf.type_143", "PTQ1_0")):
            if dtype in types:
                requirements.append(f"Prism {name} packed-weight decoding")
        requirements.append("Prism Hadamard activation transforms")
    return tuple(requirements)


def _roles(document: Document, node: Node, *, layer_norm: bool) -> set[str]:
    classifier = {
        "Embedding": _embed_role,
        "Attention": _attention_role,
        "FFN": _ffn_role,
    }.get(node.op_type, lambda _name: "weight")
    roles: set[str] = set()
    for attribute in node.attributes:
        if attribute.kind is not AttrKind.TENSOR:
            continue
        tensor = document.tensors[str(attribute.value)]
        name = _tensor_name(tensor.id)
        role = classifier(name)
        if role is None:
            if not name.endswith("rotary_emb.inv_freq"):
                raise LlmTraceError(f"{node.source_name}: unsupported weight {name}")
            continue
        if tensor.storage is Storage.ABSENT:
            raise LlmTraceError(f"Weight {name} has no readable bytes.")
        if not supports_dequantization(tensor.element_type):
            raise LlmTraceError(
                f"Weight {name} uses unsupported type {tensor.element_type}."
            )
        if name.endswith(".bias"):
            if (
                not layer_norm
                and role in {"norm", "q_norm", "k_norm", "weight"}
                and node.op_type != "Linear"
            ):
                raise LlmTraceError(f"RMS normalization bias {name} cannot be traced.")
            continue
        roles.add(role)
    return roles


def validate_llm_trace(document: Document) -> None:
    """Reject unsupported semantics before allocating or reading any weights."""
    if requirements := _special_runtime_requirements(document):
        raise LlmTraceError(
            "NNEditor cannot trace this model yet. Missing runtime support:\n"
            + "\n".join(f"• {requirement}" for requirement in requirements)
            + "\nMetadata, the inferred graph, and weight bytes can still be inspected."
        )
    params = hyperparams_from_document(document)
    if params.architecture not in {"llama", "qwen2", "qwen3", "gpt2"}:
        raise LlmTraceError(
            f"Tracing {params.architecture} is not supported. "
            "You can inspect its graph and weights."
        )
    if min(params.n_embd, params.n_head, params.n_kv_head, params.vocab_size) <= 0:
        raise LlmTraceError(
            "Hidden size, vocabulary size and head counts must be positive."
        )
    if params.n_head % params.n_kv_head:
        raise LlmTraceError(
            "Query head count must be a multiple of key/value head count."
        )
    if (params.head_dim is not None and params.head_dim <= 0) or not (
        math.isfinite(params.rms_eps)
        and params.rms_eps > 0
        and math.isfinite(params.rope_theta)
        and params.rope_theta > 0
    ):
        raise LlmTraceError(
            "Head dimension, normalization epsilon and RoPE base "
            "must be positive finite values."
        )
    config = dict(document.extensions).get("x-huggingface.config", {})
    if isinstance(config, dict):
        if config.get("rope_scaling") or (
            config.get("sliding_window")
            and config.get("use_sliding_window") is not False
        ):
            raise LlmTraceError(
                "Scaled RoPE and sliding-window attention cannot be traced. "
                "Weight inspection is available."
            )
        if (
            config.get("scale_attn_by_inverse_layer_idx")
            or config.get("scale_attn_weights") is False
        ):
            raise LlmTraceError(
                "This GPT-2 attention scaling configuration cannot be traced."
            )
    metadata = dict(document.extensions).get("x-gguf.metadata", {})
    if isinstance(metadata, dict):
        scaling = metadata.get(f"{params.architecture}.rope.scaling.type")
        if scaling is not None and scaling != "none":
            raise LlmTraceError(
                "Scaled GGUF RoPE cannot be traced. Weight inspection is available."
            )
    ops = {node.op_type for node in document.main_graph.nodes}
    if not {"Embedding", "Attention", "FFN", "Linear"} <= ops:
        raise LlmTraceError(
            "The snapshot is missing weights for a complete decoder. "
            "You can inspect the available weights."
        )
    for node in document.main_graph.nodes:
        if node.op_type in {"SSM", "Parameters"}:
            raise LlmTraceError(
                f"{node.source_name}: this block cannot be traced. "
                "Weight inspection is available."
            )
        roles = _roles(document, node, layer_norm=params.layer_norm)
        required = {
            "Embedding": {"token"},
            "FFN": {"up", "down", "norm"},
            "RMSNorm": {"weight"},
            "Linear": {"weight"},
        }.get(node.op_type, set())
        if node.op_type == "Attention":
            required = {"norm", "o"} | ({"qkv"} if "qkv" in roles else {"q", "k", "v"})
            if params.architecture == "qwen3":
                required |= {"q_norm", "k_norm"}
        if node.op_type == "FFN" and params.hidden_act == "silu":
            required |= {"gate"}
        if missing := required - roles:
            raise LlmTraceError(
                f"{node.source_name}: missing required weights "
                f"{', '.join(sorted(missing))}."
            )


def trace_unavailable_reason(document: Document) -> str | None:
    """A user-facing explanation, or None for a supported source artifact."""
    try:
        validate_llm_trace(document)
    except (LlmTraceError, ValueError) as error:
        return str(error)
    return None


def with_trace_support(document: Document) -> Document:
    """Narrow the format contract to the actual architecture and weight set."""
    reason = trace_unavailable_reason(document)
    if reason is None:
        return document
    statuses = dict(document.capabilities)
    for capability in (Capability.TRACING, Capability.EXECUTION):
        statuses[capability] = CapabilityStatus(
            capability, Availability.UNAVAILABLE, reason
        )
    return Document(
        source=document.source,
        artifact_kind=document.artifact_kind,
        capabilities=statuses.values(),
        graphs=document.graphs.values(),
        tensors=document.tensors.values(),
        entry_graph=document.entry_graph,
        provenance=document.provenance,
        diagnostics=document.diagnostics,
        capability_notes=document.capability_notes,
        extensions=document.extensions,
        schema_version=document.schema_version,
    )
