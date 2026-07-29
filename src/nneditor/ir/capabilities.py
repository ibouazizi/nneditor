"""Artifact and capability contracts (task P0.2).

The viewer never advertises support by file extension. Every accepted artifact
declares, for each capability, whether the capability is available and *why*.
The reason is mandatory so the UI can always explain a disabled action instead
of silently omitting it.

The registry in this module is the single source of truth for
``docs/artifact-capabilities.md``; the document embeds generated sections and a
test fails when the two drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

__all__ = [
    "ARTIFACT_CONTRACTS",
    "ArtifactContract",
    "ArtifactKind",
    "Availability",
    "Capability",
    "CapabilityStatus",
    "ExportFidelity",
    "LoadingMode",
    "capability_matrix_markdown",
    "contract_for",
]


class ArtifactKind(StrEnum):
    """A concrete artifact contract, not a file extension."""

    ONNX_MODEL = "onnx_model"
    PYTORCH_EXPORTED_PROGRAM = "pytorch_exported_program"
    PYTORCH_FX_GRAPH_MODULE = "pytorch_fx_graph_module"
    PYTORCH_STATE_DICT = "pytorch_state_dict"
    SAFETENSORS = "safetensors"
    JAX_STABLEHLO = "jax_stablehlo"
    JAX_TRACED_FUNCTION = "jax_traced_function"
    ORBAX_CHECKPOINT = "orbax_checkpoint"


class Capability(StrEnum):
    """The seven capabilities every artifact contract must answer for."""

    TOPOLOGY = "topology"
    WEIGHTS = "weights"
    METADATA = "metadata"
    EDITING = "editing"
    EXECUTION = "execution"
    TRACING = "tracing"
    EXPORT = "export"


class Availability(StrEnum):
    """How far a capability is supported for a given artifact."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    REQUIRES_TRUSTED_MODE = "requires trusted mode"
    REQUIRES_COMPANION_ARTIFACT = "requires companion artifact"


class LoadingMode(StrEnum):
    """The isolation boundary an artifact is loaded behind."""

    SAFE_ARTIFACT = "safe artifact"
    TRUSTED_CODE = "trusted code"
    REMOTE_REPOSITORY = "remote repository"


class ExportFidelity(StrEnum):
    """The strongest export outcome an artifact contract can reach.

    Mirrors architectural rule 8. An individual export reports its own achieved
    fidelity, which is never better than the contract's ceiling.
    """

    LOSSLESS = "lossless"
    SEMANTICALLY_EQUIVALENT = "semantically equivalent"
    LOSSY = "lossy"
    WEIGHTS_ONLY = "weights only"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CapabilityStatus:
    """One capability answer plus the reason the UI shows to the user."""

    capability: Capability
    availability: Availability
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError(
                f"capability {self.capability.value!r} needs a non-empty reason"
            )

    @property
    def is_usable(self) -> bool:
        """Whether the capability can be offered without further conditions."""
        return self.availability is Availability.AVAILABLE


@dataclass(frozen=True, slots=True)
class ArtifactContract:
    """What the viewer promises for one kind of accepted artifact."""

    kind: ArtifactKind
    title: str
    preferred_inputs: tuple[str, ...]
    loading_modes: tuple[LoadingMode, ...]
    best_export_fidelity: ExportFidelity
    statuses: tuple[CapabilityStatus, ...]

    def __post_init__(self) -> None:
        declared = [status.capability for status in self.statuses]
        if len(set(declared)) != len(declared):
            raise ValueError(f"{self.kind.value}: duplicate capability status")
        missing = set(Capability) - set(declared)
        if missing:
            names = ", ".join(sorted(capability.value for capability in missing))
            raise ValueError(f"{self.kind.value}: missing capability status: {names}")
        if not self.preferred_inputs:
            raise ValueError(f"{self.kind.value}: needs at least one preferred input")
        if not self.loading_modes:
            raise ValueError(f"{self.kind.value}: needs at least one loading mode")

    def status(self, capability: Capability) -> CapabilityStatus:
        """Return the declared status for ``capability``."""
        for status in self.statuses:
            if status.capability is capability:
                return status
        raise KeyError(capability)  # pragma: no cover - __post_init__ prevents this

    def availability(self, capability: Capability) -> Availability:
        """Return only the availability for ``capability``."""
        return self.status(capability).availability


def _contract(
    kind: ArtifactKind,
    title: str,
    preferred_inputs: tuple[str, ...],
    loading_modes: tuple[LoadingMode, ...],
    best_export_fidelity: ExportFidelity,
    answers: dict[Capability, tuple[Availability, str]],
) -> ArtifactContract:
    return ArtifactContract(
        kind=kind,
        title=title,
        preferred_inputs=preferred_inputs,
        loading_modes=loading_modes,
        best_export_fidelity=best_export_fidelity,
        statuses=tuple(
            CapabilityStatus(capability, availability, reason)
            for capability, (availability, reason) in answers.items()
        ),
    )


_CONTRACTS: tuple[ArtifactContract, ...] = (
    _contract(
        ArtifactKind.ONNX_MODEL,
        "ONNX model",
        ("model.onnx", "model.onnx plus external tensor files"),
        (LoadingMode.SAFE_ARTIFACT, LoadingMode.REMOTE_REPOSITORY),
        ExportFidelity.LOSSLESS,
        {
            Capability.TOPOLOGY: (
                Availability.AVAILABLE,
                "The artifact stores the full graph, including nested control-flow "
                "subgraphs and local functions.",
            ),
            Capability.WEIGHTS: (
                Availability.AVAILABLE,
                "Initializers are addressable as embedded byte ranges or as "
                "external files with an offset and length.",
            ),
            Capability.METADATA: (
                Availability.AVAILABLE,
                "Producer, opset imports, doc strings, and metadata properties are "
                "part of the serialized model.",
            ),
            Capability.EDITING: (
                Availability.PARTIAL,
                "Entry-graph rename, scalar/list attribute edits, compatible "
                "operator replacement, unary insertion/removal, edge "
                "reconnection, same-size tensor edits, previewed 8-bit weight "
                "conversion, portable Q/DQ insertion, logical pruning, and one "
                "terminal-MatMul channel-pruning pattern are supported only "
                "when validation can prove their preconditions.",
            ),
            Capability.EXECUTION: (
                Availability.REQUIRES_COMPANION_ARTIFACT,
                "Running the graph needs an approved runtime and valid example "
                "inputs, which the artifact does not carry.",
            ),
            Capability.TRACING: (
                Availability.AVAILABLE,
                "Intermediate values can be captured by the ONNX reference "
                "evaluator only in an isolated, resource-limited worker after "
                "explicit approval of each run.",
            ),
            Capability.EXPORT: (
                Availability.AVAILABLE,
                "A committed revision is checked and written as a new ONNX "
                "artifact with command and transformation provenance. No-op "
                "output is byte-identical when external references do not need "
                "relocation; quantization and pruning exports are reported as "
                "lossy.",
            ),
        },
    ),
    _contract(
        ArtifactKind.PYTORCH_EXPORTED_PROGRAM,
        "PyTorch exported program",
        ("torch.export.save output",),
        (LoadingMode.SAFE_ARTIFACT,),
        ExportFidelity.WEIGHTS_ONLY,
        {
            Capability.TOPOLOGY: (
                Availability.AVAILABLE,
                "The exported program serializes a complete graph with a call "
                "signature and symbolic shape constraints.",
            ),
            Capability.WEIGHTS: (
                Availability.AVAILABLE,
                "Parameters and buffers ship with the artifact and are addressable "
                "without executing Python.",
            ),
            Capability.METADATA: (
                Availability.PARTIAL,
                "Module paths and stack traces survive when the exporter recorded "
                "them; original source code never does.",
            ),
            Capability.EDITING: (
                Availability.PARTIAL,
                "Readable parameter and buffer bytes can be edited. PT2 graph "
                "surgery is unavailable until a versioned PT2 writer exists.",
            ),
            Capability.EXECUTION: (
                Availability.REQUIRES_TRUSTED_MODE,
                "Replaying the program runs framework code, so it is confined to "
                "an isolated worker the user explicitly enables.",
            ),
            Capability.TRACING: (
                Availability.UNAVAILABLE,
                "No isolated PyTorch ExportedProgram activation-capture adapter "
                "is implemented.",
            ),
            Capability.EXPORT: (
                Availability.PARTIAL,
                "Readable parameters and buffers export as a labelled "
                "weights-only safetensors artifact. PT2 graph re-serialization "
                "and ONNX conversion are not implemented.",
            ),
        },
    ),
    _contract(
        ArtifactKind.PYTORCH_FX_GRAPH_MODULE,
        "PyTorch FX graph module",
        ("serialized GraphModule plus a weights file",),
        (LoadingMode.SAFE_ARTIFACT, LoadingMode.TRUSTED_CODE),
        ExportFidelity.LOSSY,
        {
            Capability.TOPOLOGY: (
                Availability.PARTIAL,
                "FX graphs carry Python-level call targets that have no stable "
                "serialized contract, so some nodes import as opaque.",
            ),
            Capability.WEIGHTS: (
                Availability.REQUIRES_COMPANION_ARTIFACT,
                "The graph references parameters by name; the values live in a "
                "separate state dictionary or safetensors file.",
            ),
            Capability.METADATA: (
                Availability.PARTIAL,
                "Node metadata is version sensitive and is preserved verbatim in a "
                "source extension namespace rather than interpreted.",
            ),
            Capability.EDITING: (
                Availability.UNAVAILABLE,
                "Preconditions cannot be validated against a stable schema, so no "
                "edit can be proven safe.",
            ),
            Capability.EXECUTION: (
                Availability.REQUIRES_TRUSTED_MODE,
                "Reconstructing a callable module executes Python from the artifact.",
            ),
            Capability.TRACING: (
                Availability.UNAVAILABLE,
                "FX call targets do not have a stable safe execution contract, "
                "and no isolated activation-capture adapter is implemented.",
            ),
            Capability.EXPORT: (
                Availability.UNAVAILABLE,
                "Re-serializing an FX graph would silently drop the parts that did "
                "not import; original Python is not recoverable.",
            ),
        },
    ),
    _contract(
        ArtifactKind.PYTORCH_STATE_DICT,
        "PyTorch checkpoint (state dict only)",
        ("checkpoint.pt", "checkpoint.pth"),
        (LoadingMode.SAFE_ARTIFACT,),
        ExportFidelity.WEIGHTS_ONLY,
        {
            Capability.TOPOLOGY: (
                Availability.UNAVAILABLE,
                "A state dictionary stores tensors keyed by name; it contains no "
                "operators, edges, or module structure to display.",
            ),
            Capability.WEIGHTS: (
                Availability.AVAILABLE,
                "Tensors are read with weights-only deserialization, which never "
                "unpickles arbitrary objects.",
            ),
            Capability.METADATA: (
                Availability.PARTIAL,
                "Only tensor names, shapes, and dtypes are trustworthy; any other "
                "checkpoint payload is not interpreted.",
            ),
            Capability.EDITING: (
                Availability.PARTIAL,
                "Tensor values can be replaced, quantized, or pruned; nothing "
                "structural exists to edit.",
            ),
            Capability.EXECUTION: (
                Availability.UNAVAILABLE,
                "There is no graph to execute without a separately supplied model "
                "definition.",
            ),
            Capability.TRACING: (
                Availability.UNAVAILABLE,
                "A state dictionary has no graph whose intermediate values could "
                "be captured.",
            ),
            Capability.EXPORT: (
                Availability.PARTIAL,
                "Edited tensors export as a new weights-only artifact and are "
                "labelled as such.",
            ),
        },
    ),
    _contract(
        ArtifactKind.SAFETENSORS,
        "Safetensors weights",
        ("model.safetensors", "sharded safetensors plus an index file"),
        (LoadingMode.SAFE_ARTIFACT, LoadingMode.REMOTE_REPOSITORY),
        ExportFidelity.WEIGHTS_ONLY,
        {
            Capability.TOPOLOGY: (
                Availability.UNAVAILABLE,
                "The container is a tensor dictionary with a JSON header; it "
                "describes no computation.",
            ),
            Capability.WEIGHTS: (
                Availability.AVAILABLE,
                "The header gives an exact offset and length per tensor, which is "
                "the ideal case for range reads.",
            ),
            Capability.METADATA: (
                Availability.PARTIAL,
                "Only the free-form header metadata block is available and it is "
                "not standardized across producers.",
            ),
            Capability.EDITING: (
                Availability.PARTIAL,
                "Tensor-level edits apply; there is no graph to validate against.",
            ),
            Capability.EXECUTION: (
                Availability.UNAVAILABLE,
                "No computation is described by the artifact.",
            ),
            Capability.TRACING: (
                Availability.UNAVAILABLE,
                "Safetensors describes weights but no executable graph or "
                "intermediate values.",
            ),
            Capability.EXPORT: (
                Availability.AVAILABLE,
                "A new safetensors file is written from the edited tensor set "
                "without loss.",
            ),
        },
    ),
    _contract(
        ArtifactKind.JAX_STABLEHLO,
        "JAX export / StableHLO artifact",
        ("jax.export serialization", "StableHLO bytecode or textual module"),
        (LoadingMode.SAFE_ARTIFACT,),
        ExportFidelity.UNAVAILABLE,
        {
            Capability.TOPOLOGY: (
                Availability.AVAILABLE,
                "Functions, regions, and operations are fully described by the "
                "serialized module.",
            ),
            Capability.WEIGHTS: (
                Availability.UNAVAILABLE,
                "The textual StableHLO reader preserves constants as operation "
                "syntax but does not expose them as addressable tensor payloads; "
                "parameters passed as arguments live in a separate checkpoint.",
            ),
            Capability.METADATA: (
                Availability.PARTIAL,
                "Locations and attributes are preserved verbatim; Python-level "
                "names survive only when the producer emitted them.",
            ),
            Capability.EDITING: (
                Availability.UNAVAILABLE,
                "No StableHLO writer or dialect-level validation pipeline is "
                "implemented, so edits cannot be round-tripped safely.",
            ),
            Capability.EXECUTION: (
                Availability.REQUIRES_TRUSTED_MODE,
                "Evaluation needs a compiler and runtime, executed only in an "
                "isolated worker.",
            ),
            Capability.TRACING: (
                Availability.UNAVAILABLE,
                "No isolated StableHLO intermediate-value capture adapter is "
                "implemented.",
            ),
            Capability.EXPORT: (
                Availability.UNAVAILABLE,
                "NNEditor has no StableHLO writer. The imported module can be "
                "inspected but cannot be re-serialized after edits.",
            ),
        },
    ),
    _contract(
        ArtifactKind.JAX_TRACED_FUNCTION,
        "JAX function traced from user code",
        ("Python callable plus an input specification",),
        (LoadingMode.TRUSTED_CODE,),
        ExportFidelity.SEMANTICALLY_EQUIVALENT,
        {
            Capability.TOPOLOGY: (
                Availability.REQUIRES_TRUSTED_MODE,
                "The graph exists only after tracing the user's code, which runs "
                "in a consented, resource-limited worker.",
            ),
            Capability.WEIGHTS: (
                Availability.REQUIRES_COMPANION_ARTIFACT,
                "Values come from whatever checkpoint the traced function is given; "
                "tracing alone produces no weights.",
            ),
            Capability.METADATA: (
                Availability.AVAILABLE,
                "Tracing preserves source locations and function names because the "
                "producing code is present.",
            ),
            Capability.EDITING: (
                Availability.PARTIAL,
                "Edits apply to the traced artifact, never to the user's source code.",
            ),
            Capability.EXECUTION: (
                Availability.REQUIRES_TRUSTED_MODE,
                "Execution is the tracing mechanism itself and stays inside the "
                "isolated worker.",
            ),
            Capability.TRACING: (
                Availability.UNAVAILABLE,
                "User-code tracing remains out of scope until the trusted-code "
                "worker boundary is implemented.",
            ),
            Capability.EXPORT: (
                Availability.PARTIAL,
                "The traced module exports as StableHLO; the original Python is "
                "never regenerated.",
            ),
        },
    ),
    _contract(
        ArtifactKind.ORBAX_CHECKPOINT,
        "Flax / Orbax checkpoint",
        ("Orbax checkpoint directory",),
        (LoadingMode.SAFE_ARTIFACT,),
        ExportFidelity.WEIGHTS_ONLY,
        {
            Capability.TOPOLOGY: (
                Availability.UNAVAILABLE,
                "A checkpoint stores a nested parameter tree, not a computation.",
            ),
            Capability.WEIGHTS: (
                Availability.AVAILABLE,
                "Array shards are addressable through the checkpoint metadata "
                "without executing model code.",
            ),
            Capability.METADATA: (
                Availability.PARTIAL,
                "Tree structure, shapes, dtypes, and sharding are available; layer "
                "semantics are not.",
            ),
            Capability.EDITING: (
                Availability.PARTIAL,
                "Readable array bytes can be edited in the working revision. "
                "The source checkpoint remains immutable.",
            ),
            Capability.EXECUTION: (
                Availability.UNAVAILABLE,
                "No computation is described by the checkpoint.",
            ),
            Capability.TRACING: (
                Availability.UNAVAILABLE,
                "An Orbax checkpoint has parameters but no executable graph whose "
                "intermediate values could be captured.",
            ),
            Capability.EXPORT: (
                Availability.PARTIAL,
                "Readable arrays export as a labelled weights-only safetensors "
                "artifact. No Orbax writer is implemented, so checkpoint tree "
                "and sharding metadata are not re-serialized.",
            ),
        },
    ),
)


ARTIFACT_CONTRACTS: Mapping[ArtifactKind, ArtifactContract] = MappingProxyType(
    {contract.kind: contract for contract in _CONTRACTS}
)


def contract_for(kind: ArtifactKind) -> ArtifactContract:
    """Return the declared contract for ``kind``."""
    return ARTIFACT_CONTRACTS[kind]


_SUMMARY_SYMBOLS: Mapping[Availability, str] = MappingProxyType(
    {
        Availability.AVAILABLE: "yes",
        Availability.PARTIAL: "partial",
        Availability.UNAVAILABLE: "no",
        Availability.REQUIRES_TRUSTED_MODE: "trusted",
        Availability.REQUIRES_COMPANION_ARTIFACT: "companion",
    }
)


def capability_matrix_markdown() -> str:
    """Render the registry as the generated part of the capability document."""
    lines: list[str] = ["## Capability Summary", ""]
    headers = ["Artifact", *(capability.value.title() for capability in Capability)]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|:--" + "|:--" * len(Capability) + "|")
    for contract in _CONTRACTS:
        cells = [
            _SUMMARY_SYMBOLS[contract.availability(capability)]
            for capability in Capability
        ]
        lines.append(f"| {contract.title} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "`yes` available, `partial` limited to modelled cases, `no` "
            "unavailable, `trusted` requires the isolated trusted-code worker, "
            "`companion` requires a second artifact.",
            "",
            "## Contracts",
        ]
    )
    for contract in _CONTRACTS:
        lines.extend(
            [
                "",
                f"### {contract.title}",
                "",
                f"- Artifact kind: `{contract.kind.value}`",
                "- Preferred inputs: "
                + ", ".join(f"`{item}`" for item in contract.preferred_inputs),
                "- Loading modes: "
                + ", ".join(mode.value for mode in contract.loading_modes),
                f"- Best export fidelity: {contract.best_export_fidelity.value}",
                "",
                "| Capability | Availability | Reason |",
                "|:--|:--|:--|",
            ]
        )
        for capability in Capability:
            status = contract.status(capability)
            lines.append(
                f"| {capability.value} | {status.availability.value} | "
                f"{status.reason} |"
            )
    return "\n".join(lines)
