"""Import and validation diagnostics.

A diagnostic is a *coded* finding: the code identifies the class of problem,
the registry entry for that code carries the category, human-readable title,
and guidance the UI shows, and the individual diagnostic adds the specifics
(message and target). Codes are registered here (task P1.3) so every importer
draws from one vocabulary and `docs/import-diagnostics.md` is generated from
it — the list and the behavior cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

__all__ = [
    "DIAGNOSTIC_CODES",
    "CodeDescriptor",
    "Diagnostic",
    "DiagnosticCategory",
    "DiagnosticLog",
    "Severity",
    "describe",
    "diagnostics_markdown",
]


class Severity(StrEnum):
    """How badly a diagnostic affects the opened model."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One reportable finding, always attributable to something in the model."""

    code: str
    severity: Severity
    message: str
    target: str | None = None

    def __str__(self) -> str:
        where = f" [{self.target}]" if self.target else ""
        return f"{self.severity.value}: {self.code}{where}: {self.message}"


class DiagnosticLog:
    """An append-only collection of diagnostics gathered during one import."""

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[Diagnostic] = ()) -> None:
        self._items: list[Diagnostic] = list(items)

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        target: str | None = None,
    ) -> Diagnostic:
        """Record a diagnostic and return it."""
        diagnostic = Diagnostic(code, severity, message, target)
        self._items.append(diagnostic)
        return diagnostic

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def of_severity(self, severity: Severity) -> tuple[Diagnostic, ...]:
        """Return every diagnostic at exactly ``severity``."""
        return tuple(item for item in self._items if item.severity is severity)

    def codes(self) -> tuple[str, ...]:
        """Return the recorded codes in order, which keeps tests readable."""
        return tuple(item.code for item in self._items)

    @property
    def has_errors(self) -> bool:
        """Whether anything was recorded at :attr:`Severity.ERROR`."""
        return any(item.severity is Severity.ERROR for item in self._items)

    def counts(self) -> Mapping[Severity, int]:
        """How many diagnostics were recorded at each severity."""
        totals = dict.fromkeys(Severity, 0)
        for item in self._items:
            totals[item.severity] += 1
        return MappingProxyType(totals)


class DiagnosticCategory(StrEnum):
    """The problem classes the plan requires imports to report (P1.3)."""

    UNSUPPORTED = "unsupported construct"
    TENSOR_REFERENCE = "tensor reference"
    EXTERNAL_DATA = "external data"
    SHAPE = "shape"
    IDENTITY = "identity"
    METADATA_LOSS = "metadata loss"


@dataclass(frozen=True, slots=True)
class CodeDescriptor:
    """What the UI knows about a diagnostic code, independent of any finding."""

    code: str
    category: DiagnosticCategory
    title: str
    guidance: str

    def __post_init__(self) -> None:
        if not self.code or not self.title or not self.guidance.strip():
            raise ValueError("code descriptors need a code, title, and guidance")


_PYTORCH_DESCRIPTORS: tuple[CodeDescriptor, ...] = (
    CodeDescriptor(
        "pytorch.malformed-tensor-entry",
        DiagnosticCategory.TENSOR_REFERENCE,
        "Safetensors header entry malformed",
        "The header entry is missing or mistypes dtype, shape, or offsets; "
        "the tensor is listed without data.",
    ),
    CodeDescriptor(
        "pytorch.tensor-range-out-of-bounds",
        DiagnosticCategory.TENSOR_REFERENCE,
        "Tensor range exceeds its container",
        "The declared byte range does not fit inside the data section or "
        "storage, so the tensor is treated as unreadable.",
    ),
    CodeDescriptor(
        "pytorch.tensor-length-mismatch",
        DiagnosticCategory.TENSOR_REFERENCE,
        "Tensor bytes disagree with declared shape",
        "The stored length does not match the length implied by shape and "
        "dtype; the tensor is treated as unreadable.",
    ),
    CodeDescriptor(
        "pytorch.no-tensors",
        DiagnosticCategory.METADATA_LOSS,
        "Checkpoint holds no tensors",
        "The mapping deserialized safely but contains nothing tensor-shaped.",
    ),
    CodeDescriptor(
        "pytorch.unsupported-storage-dtype",
        DiagnosticCategory.UNSUPPORTED,
        "Unsupported storage dtype",
        "The tensor's storage class is not in this build's vocabulary; its "
        "values are not readable.",
    ),
    CodeDescriptor(
        "pytorch.storage-missing",
        DiagnosticCategory.TENSOR_REFERENCE,
        "Referenced storage missing from archive",
        "The pickle references a storage member the archive does not hold.",
    ),
    CodeDescriptor(
        "pytorch.compressed-storage",
        DiagnosticCategory.TENSOR_REFERENCE,
        "Storage is compressed",
        "The storage member is deflated, so it is not range-readable; the "
        "tensor is listed without data in safe mode.",
    ),
    CodeDescriptor(
        "pytorch.non-contiguous-tensor",
        DiagnosticCategory.TENSOR_REFERENCE,
        "Tensor is a strided view",
        "The tensor's strides do not describe one contiguous range; its "
        "bytes are not materialized in safe mode.",
    ),
    CodeDescriptor(
        "pytorch.shared-storage",
        DiagnosticCategory.TENSOR_REFERENCE,
        "Tensors share one storage",
        "Several tensors alias the same bytes; editing one edits them all.",
    ),
    CodeDescriptor(
        "pytorch.unsupported-dimension",
        DiagnosticCategory.SHAPE,
        "Dimension could not be interpreted",
        "A size entry was neither an integer nor a recognizable symbolic "
        "expression; the extent is unknown.",
    ),
    CodeDescriptor(
        "pytorch.unsupported-argument",
        DiagnosticCategory.METADATA_LOSS,
        "Node argument kind not modelled",
        "The argument's serialized kind has no IR representation yet; it is "
        "reported rather than silently dropped.",
    ),
    CodeDescriptor(
        "pytorch.custom-operator",
        DiagnosticCategory.UNSUPPORTED,
        "Operators outside torch.ops.aten",
        "Custom operators are shown, but their schemas are unknown, so "
        "validation and editing are disabled for them.",
    ),
    CodeDescriptor(
        "pytorch.weight-not-range-readable",
        DiagnosticCategory.TENSOR_REFERENCE,
        "Weight is pickled or compressed",
        "The parameter's bytes are not a plain range; safe mode lists it without data.",
    ),
    CodeDescriptor(
        "pytorch.fx-opaque-node",
        DiagnosticCategory.UNSUPPORTED,
        "FX statement imports as opaque",
        "The generated source statement could not be fully interpreted; the "
        "node is shown without semantics.",
    ),
    CodeDescriptor(
        "pytorch.fx-no-source",
        DiagnosticCategory.UNSUPPORTED,
        "FX topology not recoverable",
        "No parseable generated forward source was found; only the artifact's "
        "import list is reported.",
    ),
    CodeDescriptor(
        "pytorch.invalid-tensor-geometry",
        DiagnosticCategory.TENSOR_REFERENCE,
        "Tensor geometry is invalid",
        "A stored offset or dimension is negative, so the tensor's byte range "
        "cannot be trusted; it is listed without data.",
    ),
    CodeDescriptor(
        "pytorch.unknown-user-input",
        DiagnosticCategory.METADATA_LOSS,
        "Signature names an absent input",
        "The exported signature references a placeholder that no value "
        "declares; the input is reported rather than invented.",
    ),
    CodeDescriptor(
        "pytorch.tensor-range-overlap",
        DiagnosticCategory.TENSOR_REFERENCE,
        "Tensor byte ranges overlap",
        "Two tensors claim overlapping bytes, which the format forbids; both "
        "are listed without data so an edit cannot corrupt a neighbor.",
    ),
    CodeDescriptor(
        "pytorch.tensor-ranges-unordered",
        DiagnosticCategory.METADATA_LOSS,
        "Tensor byte ranges out of order",
        "The header lists disjoint ranges out of ascending order; reads work, "
        "but the deviation from the specification is disclosed.",
    ),
)

_HIERARCHY_DESCRIPTORS: tuple[CodeDescriptor, ...] = (
    CodeDescriptor(
        "hierarchy.structural-analysis-skipped",
        DiagnosticCategory.UNSUPPORTED,
        "Structural block detection was skipped",
        "The graph has more nodes than the exact dominator analysis is allowed "
        "to sweep, so branch/merge and residual regions were not detected. "
        "Blocks still come from exporter scopes, repeated structure, and "
        "operator motifs; group manually where a region matters.",
    ),
    CodeDescriptor(
        "hierarchy.repeat-analysis-skipped",
        DiagnosticCategory.UNSUPPORTED,
        "Repeated block detection was skipped",
        "The graph has more nodes than the repeated-structure scan is allowed "
        "to examine, so identical blocks were not grouped by structure. Blocks "
        "still come from exporter scopes and operator motifs.",
    ),
)


_JAX_DESCRIPTORS: tuple[CodeDescriptor, ...] = (
    CodeDescriptor(
        "jax.opaque-operation",
        DiagnosticCategory.UNSUPPORTED,
        "StableHLO operation not interpreted",
        "The operation's textual form is outside the modelled grammar; it is "
        "shown opaquely with its source text preserved.",
    ),
    CodeDescriptor(
        "jax.custom-call",
        DiagnosticCategory.UNSUPPORTED,
        "Custom call target is external",
        "The callee is defined outside the artifact, so its semantics are "
        "unknown here and editing is disabled for the node.",
    ),
    CodeDescriptor(
        "jax.region-not-modelled",
        DiagnosticCategory.UNSUPPORTED,
        "Nested region carried as text",
        "Control-flow bodies are preserved verbatim as source text; their "
        "internals are not yet modelled as graphs.",
    ),
    CodeDescriptor(
        "jax.attribute-truncated",
        DiagnosticCategory.METADATA_LOSS,
        "Attribute shown truncated",
        "The attribute's text exceeds the display ceiling; the artifact still "
        "holds it in full.",
    ),
    CodeDescriptor(
        "jax.symbolic-dimension",
        DiagnosticCategory.SHAPE,
        "Symbolic extent present",
        "A dimension is polymorphic, so shape-dependent validation stays "
        "conservative for paths through it.",
    ),
    CodeDescriptor(
        "jax.sharded-checkpoint",
        DiagnosticCategory.UNSUPPORTED,
        "Sharded checkpoint partially opened",
        "Several shards were found; this build opens the first one only.",
    ),
    CodeDescriptor(
        "jax.unreadable-tree-metadata",
        DiagnosticCategory.METADATA_LOSS,
        "Checkpoint tree metadata unreadable",
        "A metadata file was not valid JSON and is skipped; the weights are "
        "unaffected.",
    ),
    CodeDescriptor(
        "jax.tree-metadata-truncated",
        DiagnosticCategory.METADATA_LOSS,
        "Checkpoint tree metadata truncated",
        "The checkpoint holds more metadata files than the reading ceiling; "
        "the excess files are skipped and counted.",
    ),
    CodeDescriptor(
        "jax.tree-metadata-too-large",
        DiagnosticCategory.METADATA_LOSS,
        "Checkpoint metadata file too large",
        "A metadata file exceeds the per-file size ceiling and is skipped; "
        "the weights are unaffected.",
    ),
)

_DESCRIPTORS: tuple[CodeDescriptor, ...] = (
    _PYTORCH_DESCRIPTORS
    + _HIERARCHY_DESCRIPTORS
    + _JAX_DESCRIPTORS
    + (
        CodeDescriptor(
            "onnx.custom-domain",
            DiagnosticCategory.UNSUPPORTED,
            "Operator from a custom domain",
            "The model uses operators outside the standard ONNX set. They are "
            "shown, but their schemas are unknown, so validation and editing are "
            "disabled for them.",
        ),
        CodeDescriptor(
            "onnx.sparse-initializer-unsupported",
            DiagnosticCategory.UNSUPPORTED,
            "Sparse initializer not modelled",
            "Sparse tensor storage is not supported yet; the tensor is listed "
            "without its values.",
        ),
        CodeDescriptor(
            "onnx.unknown-element-type",
            DiagnosticCategory.UNSUPPORTED,
            "Unknown tensor element type",
            "The tensor declares an element type this build does not know, "
            "usually from a newer ONNX release. Its byte length cannot be "
            "validated.",
        ),
        CodeDescriptor(
            "onnx.payload-length-mismatch",
            DiagnosticCategory.TENSOR_REFERENCE,
            "Payload disagrees with declared shape",
            "The embedded data length does not match the length implied by the "
            "tensor's shape and type. The tensor is treated as unreadable.",
        ),
        CodeDescriptor(
            "onnx.implausible-shape",
            DiagnosticCategory.TENSOR_REFERENCE,
            "Implausible tensor shape",
            "The declared shape implies an absurd element count; the tensor's "
            "data is not trusted.",
        ),
        CodeDescriptor(
            "onnx.negative-dimension",
            DiagnosticCategory.TENSOR_REFERENCE,
            "Negative tensor dimension",
            "A tensor declares a negative extent, which is not valid; its data "
            "is not trusted.",
        ),
        CodeDescriptor(
            "onnx.typed-tensor-field",
            DiagnosticCategory.TENSOR_REFERENCE,
            "Tensor stored in a typed field",
            "The tensor's values live in a typed protobuf field rather than raw "
            "bytes, so reading it requires a full parse instead of a byte range. "
            "Inspection may be slower for large tensors.",
        ),
        CodeDescriptor(
            "onnx.external-file-missing",
            DiagnosticCategory.EXTERNAL_DATA,
            "External data file missing",
            "The referenced data file was not found next to the model. Copy the "
            "missing file into the model's directory and reopen.",
        ),
        CodeDescriptor(
            "onnx.external-path-unsafe",
            DiagnosticCategory.EXTERNAL_DATA,
            "External path escapes the model directory",
            "The reference points outside the approved directory and was "
            "refused. This protects against malicious artifacts.",
        ),
        CodeDescriptor(
            "onnx.external-range-out-of-bounds",
            DiagnosticCategory.EXTERNAL_DATA,
            "External range exceeds the data file",
            "The declared offset and length do not fit inside the external "
            "file, so the tensor is treated as unreadable.",
        ),
        CodeDescriptor(
            "onnx.external-length-mismatch",
            DiagnosticCategory.EXTERNAL_DATA,
            "External length disagrees with shape",
            "The declared external length does not match the length implied by "
            "the tensor's shape and type.",
        ),
        CodeDescriptor(
            "onnx.external-metadata-invalid",
            DiagnosticCategory.EXTERNAL_DATA,
            "External metadata malformed",
            "The external_data entry is missing required keys or holds "
            "non-numeric offsets or lengths.",
        ),
        CodeDescriptor(
            "onnx.checksum-mismatch",
            DiagnosticCategory.EXTERNAL_DATA,
            "External data checksum mismatch",
            "The external file's contents do not match the checksum recorded in "
            "the model. The file may be corrupted or substituted.",
        ),
        CodeDescriptor(
            "onnx.incomplete-io-shape",
            DiagnosticCategory.SHAPE,
            "Graph input or output lacks a complete shape",
            "The value's element type or extents are not fully declared. "
            "Navigation works normally; shape-dependent validation will be "
            "conservative.",
        ),
        CodeDescriptor(
            "onnx.positional-node-id",
            DiagnosticCategory.IDENTITY,
            "Node identity depends on its position",
            "The node has neither a name nor a named output, so selections and "
            "manual groups attached to it may not survive edits elsewhere in "
            "the graph.",
        ),
        CodeDescriptor(
            "onnx.duplicate-node-id",
            DiagnosticCategory.IDENTITY,
            "Two nodes derive the same identifier",
            "A later node fell back to a positional identity to stay unique; "
            "the same caveat as positional identity applies.",
        ),
        CodeDescriptor(
            "onnx.duplicate-tensor-id",
            DiagnosticCategory.IDENTITY,
            "Two tensors derive the same identifier",
            "Both tensors stay listed in their graph, but lookups and edits "
            "resolve to the later definition; rename one tensor to address "
            "them separately.",
        ),
        CodeDescriptor(
            "onnx.attribute-too-large",
            DiagnosticCategory.METADATA_LOSS,
            "Attribute value exceeds the import limit",
            "The attribute's value is larger than the importer buffers; it is "
            "listed by name only and survives unchanged in the source "
            "artifact.",
        ),
        CodeDescriptor(
            "onnx.unsupported-attribute",
            DiagnosticCategory.METADATA_LOSS,
            "Attribute value not imported",
            "The attribute uses a sparse-tensor, type-proto, or function-"
            "reference value the IR does not model; the attribute is listed by "
            "name only and survives unchanged in the source artifact.",
        ),
        CodeDescriptor(
            "onnx.unknown-attribute-type",
            DiagnosticCategory.METADATA_LOSS,
            "Attribute type not representable",
            "The attribute declares a type code this build cannot represent; "
            "its value is not imported.",
        ),
        CodeDescriptor(
            "onnx.malformed-attribute",
            DiagnosticCategory.METADATA_LOSS,
            "Attribute is malformed",
            "The attribute's declared kind and its stored value disagree; the "
            "value is not imported.",
        ),
        CodeDescriptor(
            "onnx.type-annotation-conflict",
            DiagnosticCategory.IDENTITY,
            "Type annotation contradicts its initializer",
            "A value_info entry declares a different element type from the "
            "initializer of the same name. The annotation is redundant — the "
            "initializer already carries its type — but ONNX type inference "
            "treats the disagreement as fatal, so export and tracing are "
            "blocked until the annotation is removed or corrected. Exporters "
            "sometimes emit this; the initializer is normally the truth.",
        ),
        CodeDescriptor(
            "onnx.shape-annotation-conflict",
            DiagnosticCategory.SHAPE,
            "Shape annotation contradicts its initializer",
            "A value_info entry declares a rank or extent the initializer of "
            "the same name does not have. ONNX shape inference rejects the "
            "model, so export and tracing cannot run until the annotation is "
            "removed or corrected.",
        ),
        CodeDescriptor(
            "onnx.duplicate-initializer",
            DiagnosticCategory.IDENTITY,
            "Initializer name declared more than once",
            "Two initializers in one graph share a name, so which tensor a "
            "consumer receives depends on the runtime's resolution order. "
            "Rename or remove one before relying on this model.",
        ),
        CodeDescriptor(
            "onnx.unresolved-input",
            DiagnosticCategory.IDENTITY,
            "Node input has no producer in scope",
            "A node reads a value that no initializer, graph input, or node "
            "output supplies, counting enclosing scopes for subgraphs. The "
            "graph is incomplete and will not execute.",
        ),
        CodeDescriptor(
            "onnx.nameless-attribute",
            DiagnosticCategory.METADATA_LOSS,
            "Attribute without a name",
            "An unnamed attribute cannot be addressed and is not imported.",
        ),
    )
)

DIAGNOSTIC_CODES: Mapping[str, CodeDescriptor] = MappingProxyType(
    {descriptor.code: descriptor for descriptor in _DESCRIPTORS}
)

_FALLBACK = CodeDescriptor(
    "unregistered",
    DiagnosticCategory.UNSUPPORTED,
    "Unregistered diagnostic",
    "This finding's code has no registered description; report it as a bug.",
)


def describe(code: str) -> CodeDescriptor:
    """The registered descriptor for ``code``, or a labelled fallback.

    The fallback keeps display code total: an unregistered code renders as a
    visible bug instead of crashing the diagnostics panel.
    """
    return DIAGNOSTIC_CODES.get(code, _FALLBACK)


def diagnostics_markdown() -> str:
    """Render the code registry as the generated documentation table."""
    lines = [
        "| Code | Category | Title | Guidance |",
        "|:--|:--|:--|:--|",
    ]
    for descriptor in sorted(
        DIAGNOSTIC_CODES.values(), key=lambda item: (item.category.value, item.code)
    ):
        lines.append(
            f"| `{descriptor.code}` | {descriptor.category.value} "
            f"| {descriptor.title} | {descriptor.guidance} |"
        )
    return "\n".join(lines)
