"""Model sessions and the application service (task P1.5).

A :class:`ModelSession` is one opened artifact: its IR document, its tensor
store, and the slice queries the renderer consumes. The
:class:`ApplicationService` owns every session plus the shared job manager and
layout cache. The UI holds session ids and calls these services; it never
holds a document or store reference of its own, which is what keeps UI state
disposable and the IR authoritative.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Self, cast

import numpy as np
from numpy.typing import NDArray

from nneditor.adapters.onnx import (
    ExportReport,
    export_revision,
)
from nneditor.adapters.pytorch.safetensors import (
    SafetensorsError,
    write_safetensors,
)
from nneditor.adapters.registry import (
    ArtifactAdapterError,
    ArtifactAdapterRegistry,
    default_artifact_adapter_registry,
)
from nneditor.analysis.hierarchy import Hierarchy
from nneditor.analysis.layout import LayoutSettings
from nneditor.analysis.lod import DetailLevel
from nneditor.analysis.statistics import TensorStatistics, compute_statistics
from nneditor.application.editing import EditingController, SidecarPersistenceError
from nneditor.application.edits_store import RevisionSidecarStore
from nneditor.application.hierarchy import (
    HierarchyController,
    HierarchySidecarStore,
)
from nneditor.application.jobs import Job, JobListener, JobManager
from nneditor.application.navigation import (
    Breadcrumb,
    Direction,
    MiniMap,
    NavigationModel,
    SearchHit,
)
from nneditor.application.persistence import SessionStateStore
from nneditor.application.slices import GraphSlice, GraphSlicer
from nneditor.application.statistics_store import StatisticsSidecarStore
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.editing.revisions import Revision
from nneditor.editing.validation import EditRequest, EditTransaction
from nneditor.ir.capabilities import (
    ArtifactKind,
    Availability,
    Capability,
    CapabilityStatus,
    ExportFidelity,
)
from nneditor.ir.core import Document, Storage, TensorRef
from nneditor.rendering.scene import Bounds, Scene, Viewport
from nneditor.storage.cache import BudgetedCache
from nneditor.storage.store import TensorStore, TensorUnavailableError
from nneditor.tracing.comparison import (
    TraceComparison,
    compare_traces,
    comparison_scene_patch,
    trace_scene_patch,
)
from nneditor.tracing.contracts import (
    ActivationRecord,
    InputBinding,
    InputSpecification,
    TraceKey,
    TraceRequest,
    TraceResult,
)
from nneditor.tracing.inputs import (
    InputSpecificationStore,
    default_input_specification,
    tensor_file_binding_for_input,
)
from nneditor.tracing.runner import run_onnx_trace
from nneditor.tracing.store import ActivationStore, ActivationView
from nneditor.tracing.visualization import (
    ActivationVisualization,
    build_activation_visualizations,
)
from nneditor.transformations.calibration import (
    CalibrationLimits,
    CalibrationProvider,
    CalibrationResult,
    run_calibration,
)
from nneditor.transformations.engine import (
    TransformationProposal,
    TransformationRequest,
)

__all__ = [
    "ApplicationService",
    "ExportOutcome",
    "ExportPlan",
    "ModelSession",
    "SessionError",
]


class SessionError(ValueError):
    """Raised when a session operation cannot be performed."""


_PREWARM_DETAIL_LEVELS: Final = (
    DetailLevel.BLOCK,
    DetailLevel.LAYER,
    DetailLevel.OPERATOR,
)
"""Deeper levels warmed in the background after open. Their first interactive
switch otherwise pays the full base-layout-plus-semantic-slice cost cold —
hundreds of milliseconds on large models, well past the interaction budget."""


@dataclass(frozen=True, slots=True)
class ExportPlan:
    """How the current artifact can be exported by the application."""

    dialog_title: str
    file_name: str
    allowed_extensions: tuple[str, ...]
    fidelity: ExportFidelity
    available: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ExportOutcome:
    """Format-neutral result consumed by the shell."""

    written_files: tuple[Path, ...]
    fidelity: ExportFidelity
    report_path: Path | None = None


class _EditedBytesView:
    """A ``TensorStore``-shaped facade serving edit-overlaid tensor bytes.

    :func:`compute_statistics` only needs ``metadata``, ``byte_length``, and
    ``read``; routing them through the editing chain makes statistics reflect
    committed byte edits and generated tensors (quantization scales and zero
    points exist only in the chain, never in the base store).
    """

    __slots__ = ("_document", "_editing")

    def __init__(self, document: Document, editing: EditingController) -> None:
        self._document = document
        self._editing = editing

    def metadata(self, tensor_id: str) -> TensorRef:
        try:
            return self._document.tensors[tensor_id]
        except KeyError:
            raise TensorUnavailableError(
                f"tensor {tensor_id!r} is not in the document"
            ) from None

    def byte_length(self, tensor_id: str) -> int | None:
        return self._editing.byte_length(tensor_id)

    def read(
        self,
        tensor_id: str,
        *,
        offset: int = 0,
        length: int | None = None,
        token: CancellationToken | None = None,
    ) -> bytes:
        if token is not None:
            token.raise_if_cancelled()
        return self._editing.read(tensor_id, offset=offset, length=length)


class ModelSession:
    """One opened model: document, tensor access, and renderer queries."""

    __slots__ = (
        "_activation_store",
        "_closed",
        "_edited_stats",
        "_hierarchy_store",
        "_input_store",
        "_jobs",
        "_navigation",
        "_prewarm_job",
        "_slicer",
        "_source_document",
        "_state_lock",
        "_stats",
        "_stats_lock",
        "_stats_sidecar",
        "editing",
        "hierarchy",
        "id",
        "store",
    )

    def __init__(
        self,
        session_id: int,
        document: Document,
        slicer: GraphSlicer,
        hierarchy: HierarchyController,
        *,
        jobs: JobManager | None = None,
        stats_sidecar: StatisticsSidecarStore | None = None,
        edits_sidecar: RevisionSidecarStore | None = None,
        hierarchy_store: HierarchySidecarStore | None = None,
        activation_store: ActivationStore | None = None,
        input_store: InputSpecificationStore | None = None,
    ) -> None:
        self.id = session_id
        self._source_document = document
        self.store = TensorStore(document)
        try:
            self._state_lock = threading.RLock()
            self._slicer = slicer
            self.hierarchy = hierarchy
            self._hierarchy_store = hierarchy_store
            self._navigation = NavigationModel()
            self._jobs = jobs
            self._prewarm_job: Job[tuple[GraphSlice, ...]] | None = None
            self._activation_store = activation_store
            self._input_store = input_store
            self._stats_sidecar = stats_sidecar
            self._stats: dict[str, TensorStatistics] | None = None
            self._edited_stats: BudgetedCache[
                tuple[str | None, str], TensorStatistics
            ] = BudgetedCache("edited-statistics", 128)
            self._stats_lock = threading.Lock()
            self.editing = EditingController(
                document, self.store, sidecar=edits_sidecar
            )
            if self.editing.preview().graph_changes:
                self.hierarchy = HierarchyController(
                    self.document,
                    store=self._hierarchy_store,
                )
            self._closed = False
        except BaseException:
            # A failure after this point (sidecar recovery opens file
            # handles) would otherwise leak the store and keep the artifact
            # locked on Windows.
            self.store.close()
            raise

    @property
    def document(self) -> Document:
        """The current working document, including committed graph edits."""
        with self._state_lock:
            return self.editing.document

    # -- capability and diagnostic access ---------------------------------

    def capability(self, capability: Capability) -> CapabilityStatus:
        return self.document.capability(capability)

    @property
    def title(self) -> str:
        return Path(self.document.source.path).name

    # -- renderer-facing queries ------------------------------------------

    def scene(
        self,
        graph_id: str | None = None,
        settings: LayoutSettings | None = None,
        *,
        detail_level: DetailLevel = DetailLevel.OPERATOR,
        root_group: str | None = None,
        viewport: Bounds | None = None,
    ) -> GraphSlice:
        self._ensure_open()
        with self._state_lock:
            document = self.editing.document
            resolved = graph_id or document.entry_graph
            controller = self.hierarchy
            return self._slicer.slice_graph(
                document,
                resolved,
                settings,
                revision_id=self.editing.current_revision_id,
                hierarchy=controller.hierarchy(resolved),
                detail_level=detail_level,
                root_group=root_group,
                viewport=viewport,
            )

    def _schedule_layout_prewarm(self) -> None:
        """Warm the deeper detail levels' shared slicer caches on the pool.

        Walks BLOCK -> LAYER -> OPERATOR for the entry graph (root group
        ``None``) through the same slicer call :meth:`scene` makes with the
        same resolved arguments, so the populated ``SliceKey``/base-layout
        keys are exactly the ones the shell's first detail switch asks for.
        The shell needs no changes: it keeps calling :meth:`scene` and finds
        the entries warm.
        """
        if self._jobs is None or self._closed:
            return

        def work(token: CancellationToken) -> tuple[GraphSlice, ...]:
            # Capture once, briefly. The walk below runs without
            # _state_lock: holding it across multi-hundred-millisecond
            # layouts would block UI reads and close().
            with self._state_lock, self.editing.stable_view():
                if self._closed:
                    raise OperationCancelled
                document = self.editing.document
                revision_id = self.editing.current_revision_id
                graph_id = document.entry_graph
                hierarchy = self.hierarchy.hierarchy(graph_id)
            warmed: list[GraphSlice] = []
            for level in _PREWARM_DETAIL_LEVELS:
                # Checkpoint between levels: closing the session cancels
                # this job, and the walk stops before its next level.
                token.raise_if_cancelled()
                if self._closed:
                    raise OperationCancelled
                # Every key ingredient (content hash, editing revision,
                # hierarchy revision, settings) was captured above, so a
                # hierarchy edit or organization-mode change mid-walk only
                # changes *future* keys — entries computed here are at
                # worst unused until evicted, never served stale, and need
                # no invalidation.
                warmed.append(
                    self._slicer.slice_graph(
                        document,
                        graph_id,
                        revision_id=revision_id,
                        hierarchy=hierarchy,
                        detail_level=level,
                        root_group=None,
                    )
                )
            return tuple(warmed)

        # Best-effort: a service already shutting down skips the warmup
        # rather than failing the open.
        with contextlib.suppress(RuntimeError):
            self._prewarm_job = self._jobs.submit(f"prewarm layouts {self.title}", work)

    def search(self, text: str) -> tuple[str, ...]:
        self._ensure_open()
        return self._slicer.search(self.document, text)

    def search_hits(self, text: str) -> tuple[SearchHit, ...]:
        self._ensure_open()
        return self._slicer.search_hits(self.document, text)

    def neighborhood(
        self, graph_id: str, node_id: str, *, hops: int = 1
    ) -> frozenset[str]:
        self._ensure_open()
        return self._slicer.neighborhood(self.document, graph_id, node_id, hops=hops)

    def graph_hierarchy(self, graph_id: str | None = None) -> Hierarchy:
        self._ensure_open()
        with self._state_lock:
            return self.hierarchy.hierarchy(
                graph_id or self.editing.document.entry_graph
            )

    def breadcrumbs(
        self, graph_id: str, group_id: str | None = None
    ) -> tuple[Breadcrumb, ...]:
        self._ensure_open()
        with self._state_lock:
            return self._navigation.breadcrumbs(
                self.editing.document,
                {
                    hierarchy.graph_id: hierarchy
                    for hierarchy in self.hierarchy.hierarchies
                },
                graph_id,
                group_id,
            )

    def directional(
        self, scene: Scene, current_id: str, direction: Direction
    ) -> str | None:
        self._ensure_open()
        return self._navigation.directional(scene, current_id, direction)

    def minimap(
        self,
        scene: Scene,
        *,
        width: float,
        height: float,
        viewport: Viewport | None = None,
    ) -> MiniMap:
        self._ensure_open()
        return self._navigation.minimap(
            scene, width=width, height=height, viewport=viewport
        )

    # -- tensor statistics (task P3.3) -------------------------------------

    def _loaded_stats(self) -> dict[str, TensorStatistics]:
        with self._stats_lock:
            if self._stats is None:
                # The content hash comes from the immutable source ref, not
                # the live document: reading the live document here would
                # take _state_lock while holding _stats_lock, inverting the
                # _state_lock -> _stats_lock order used by statistics().
                self._stats = (
                    self._stats_sidecar.load(self._source_document.source.content_hash)
                    if self._stats_sidecar is not None
                    else {}
                )
            return self._stats

    def _serves_edited_bytes(self, tensor_id: str) -> bool:
        """Whether this tensor's bytes come from the editing chain.

        True for tensors with committed byte edits and for GENERATED tensors
        (quantization scales and zero points), whose bytes exist only in the
        chain; the base store either serves stale data or fails outright.
        """
        tensor = self.document.tensors.get(tensor_id)
        if tensor is not None and tensor.storage is Storage.GENERATED:
            return True
        return tensor_id in self.editing.edited_tensor_ids()

    def statistics(self, tensor_id: str) -> TensorStatistics | None:
        """A cached or persisted summary, or ``None`` — never a computation."""
        self._ensure_open()
        with self._state_lock, self.editing.stable_view():
            if self._serves_edited_bytes(tensor_id):
                return self._edited_stats.get(
                    (self.editing.current_revision_id, tensor_id)
                )
            return self._loaded_stats().get(tensor_id)

    def compute_statistics(
        self, tensor_id: str, *, token: CancellationToken | None = None
    ) -> TensorStatistics:
        """Stream (or fetch) one tensor's statistics, persisting the result.

        Base-artifact tensors are persisted in the content-hash-keyed sidecar
        and shared. Edited and generated tensors are served through the
        editing chain and cached only in this session, keyed by the current
        revision id — their bytes do not belong to the content hash, so
        persisting them would poison the shared sidecar.

        The session lock is held only to capture the revision at the start
        and to commit the result at the end; the pass itself runs unlocked,
        so UI reads and ``close()`` never wait behind it.
        """
        self._ensure_open()
        # Capture once, briefly. The streaming pass below runs without
        # _state_lock or the controller lock: holding either for the whole
        # pass would freeze document/scene reads and block close().
        with self._state_lock, self.editing.stable_view():
            document = self.editing.document
            revision_id = self.editing.current_revision_id
            edited = self._serves_edited_bytes(tensor_id)
            cached = (
                self._edited_stats.get((revision_id, tensor_id))
                if edited
                else self._loaded_stats().get(tensor_id)
            )
        if cached is not None:
            return cached
        if edited:
            return self._stream_edited_statistics(
                document, revision_id, tensor_id, token
            )
        return self._stream_base_statistics(tensor_id, token)

    def _stream_edited_statistics(
        self,
        document: Document,
        revision_id: str | None,
        tensor_id: str,
        token: CancellationToken | None,
    ) -> TensorStatistics:
        """Stream chain-served bytes unlocked; commit only if still current.

        Each chunk read takes the controller lock for just that read, so a
        commit landing mid-pass makes later chunks serve a newer revision's
        bytes. That torn pass is acceptable *only because* the revision
        check below discards the result on mismatch instead of caching it
        under the captured key — a stale pass reports itself and is rerun.
        """
        view = _EditedBytesView(document, self.editing)
        stats = compute_statistics(cast(TensorStore, view), tensor_id, token=token)
        with self._state_lock:
            if self.editing.current_revision_id != revision_id:
                raise SessionError(
                    f"the revision changed while streaming statistics for "
                    f"{tensor_id!r}; the stale result was discarded — compute "
                    "again on the new revision"
                )
            self._edited_stats.put((revision_id, tensor_id), stats)
        return stats

    def _stream_base_statistics(
        self, tensor_id: str, token: CancellationToken | None
    ) -> TensorStatistics:
        """Stream immutable base-artifact bytes and persist them shared.

        Base bytes never change while the artifact is open, so the result
        stays valid for the content hash no matter which revisions commit
        mid-pass; no staleness check is needed. A session closed mid-pass
        makes the store raise, which surfaces through the job-error path.
        """
        stats = compute_statistics(self.store, tensor_id, token=token)
        known = self._loaded_stats()
        with self._stats_lock:
            known[tensor_id] = stats
            if self._stats_sidecar is not None:
                self._stats_sidecar.save(
                    self._source_document.source.content_hash,
                    dict(known),
                )
        return stats

    def compute_statistics_async(self, tensor_id: str) -> Job[TensorStatistics]:
        """Statistics as a cancellable background job."""
        self._ensure_open()
        if self._jobs is None:
            raise SessionError("this session has no job manager")
        return self._jobs.submit(
            f"statistics {tensor_id}",
            lambda token: self.compute_statistics(tensor_id, token=token),
        )

    # -- inference tracing (Phase 9) --------------------------------------

    def default_trace_inputs(
        self,
        *,
        seed: int = 0,
        shapes: Mapping[str, tuple[int, ...]] | None = None,
        bindings: Mapping[str, InputBinding] | None = None,
    ) -> InputSpecification:
        """Create a reproducible random input spec without guessing symbols."""
        self._ensure_open()
        return default_input_specification(
            self.document,
            seed=seed,
            shapes=shapes,
            bindings=bindings,
        )

    def trace_tensor_input(self, input_name: str, path: Path | str) -> InputBinding:
        """Validate one selected ``.npy`` file against a model input."""
        self._ensure_open()
        return tensor_file_binding_for_input(self.document, input_name, path)

    def save_trace_inputs(self, specification: InputSpecification) -> None:
        self._ensure_open()
        if self._input_store is None:
            raise SessionError("this session has no trace input sidecar")
        if specification.artifact_hash != self.document.source.content_hash:
            raise SessionError("input specification belongs to another artifact")
        self._input_store.save(specification)

    def trace_input_specifications(self) -> tuple[InputSpecification, ...]:
        self._ensure_open()
        if self._input_store is None:
            return ()
        return self._input_store.specifications(self.document.source.content_hash)

    def trace_async(self, request: TraceRequest) -> Job[TraceResult]:
        """Run an explicitly approved ONNX trace outside the UI process."""
        self._ensure_open()
        status = self.capability(Capability.TRACING)
        if status.availability is not Availability.AVAILABLE:
            raise SessionError(status.reason)
        if self.document.artifact_kind is not ArtifactKind.ONNX_MODEL:
            raise SessionError("only ONNX reference-evaluator tracing is implemented")
        if self._jobs is None or self._activation_store is None:
            raise SessionError("this session has no trace worker/store")
        self.save_trace_inputs(request.specification)
        # Capture once, briefly: the export below runs on a job thread and
        # must never hold _state_lock or the controller lock while it
        # materializes a full model to disk.
        with self._state_lock, self.editing.stable_view():
            snapshot = self.editing.document
            revision_id = self.editing.current_revision_id
            revisions = self.editing.applied_revisions
            read_snapshot = self.editing.reader_at(revisions)
        key = TraceKey(
            snapshot.source.content_hash,
            revision_id,
            request.specification.hash,
            request.backend,
            request.device,
        )

        def work(token: CancellationToken) -> TraceResult:
            def build_model(destination: Path, build_token: CancellationToken) -> None:
                # The result is keyed by the captured revision id, so the
                # captured snapshot and its pinned reader stay the correct
                # source even while the user keeps editing — no lock and no
                # staleness check. A session closed mid-export fails the
                # base read, which surfaces through the job-error path.
                export_revision(
                    snapshot,
                    revisions,
                    destination,
                    tensor_reader=read_snapshot,
                    token=build_token,
                )

            assert self._activation_store is not None
            return run_onnx_trace(
                snapshot,
                key,
                request,
                self._activation_store,
                build_model,
                token=token,
            )

        return self._jobs.submit(f"trace {self.title}", work)

    def traces(self) -> tuple[TraceResult, ...]:
        self._ensure_open()
        if self._activation_store is None:
            return ()
        return self._activation_store.results(self.document.source.content_hash)

    def trace(self, trace_id: str) -> TraceResult:
        self._ensure_open()
        if self._activation_store is None:
            raise SessionError("this session has no activation store")
        result = self._activation_store.result(trace_id)
        if result.key.artifact_hash != self.document.source.content_hash:
            raise SessionError("trace belongs to another artifact")
        return result

    def activations(self, trace_id: str) -> ActivationView:
        self.trace(trace_id)
        assert self._activation_store is not None
        return self._activation_store.open(trace_id)

    def node_activations(
        self,
        trace_id: str,
        node_id: str,
    ) -> tuple[ActivationRecord, ...]:
        """Captured inputs and outputs for one selected operator."""
        result = self.trace(trace_id)
        node = self.document.main_graph.node(node_id)
        value_ids = frozenset((*node.inputs, *node.outputs))
        return tuple(
            record for record in result.records if record.value_id in value_ids
        )

    def compute_activation_statistics_async(
        self,
        trace_id: str,
        value_id: str,
    ) -> Job[TensorStatistics]:
        self._ensure_open()
        if self._jobs is None:
            raise SessionError("this session has no job manager")
        view = self.activations(trace_id)
        return self._jobs.submit(
            f"activation statistics {value_id}",
            lambda token: view.statistics(value_id, token=token),
        )

    def activation_visualizations_async(
        self,
        trace_id: str,
        value_id: str,
        *,
        attention: bool = False,
    ) -> Job[tuple[ActivationVisualization, ...]]:
        self._ensure_open()
        if self._jobs is None:
            raise SessionError("this session has no job manager")
        view = self.activations(trace_id)
        record = view.metadata(value_id)

        def work(token: CancellationToken) -> tuple[ActivationVisualization, ...]:
            raw = view.read(value_id, token=token)
            token.raise_if_cancelled()
            return build_activation_visualizations(record, raw, attention=attention)

        return self._jobs.submit(f"visualize activation {value_id}", work)

    def compare_traces_async(
        self,
        left_trace_id: str,
        right_trace_id: str,
    ) -> Job[TraceComparison]:
        self._ensure_open()
        if self._jobs is None or self._activation_store is None:
            raise SessionError("this session has no trace job/store")
        left = self.trace(left_trace_id)
        right = self.trace(right_trace_id)
        store = self._activation_store
        return self._jobs.submit(
            "compare traces",
            lambda token: self._compare_traces(store, left, right, token),
        )

    @staticmethod
    def _compare_traces(
        store: ActivationStore,
        left: TraceResult,
        right: TraceResult,
        token: CancellationToken,
    ) -> TraceComparison:
        token.raise_if_cancelled()
        result = compare_traces(store, left, right, token=token)
        token.raise_if_cancelled()
        return result

    @staticmethod
    def trace_overlay(graph_slice: GraphSlice, result: TraceResult) -> Scene:
        return graph_slice.scene.apply(trace_scene_patch(graph_slice, result))

    @staticmethod
    def comparison_overlay(
        graph_slice: GraphSlice,
        comparison: TraceComparison,
    ) -> Scene:
        return graph_slice.scene.apply(comparison_scene_patch(graph_slice, comparison))

    # -- transactional editing (Phase 4) ----------------------------------

    def prepare_edit(self, request: EditRequest) -> EditTransaction:
        self._ensure_open()
        return self.editing.prepare(request)

    def commit_edit(self, transaction: EditTransaction) -> Revision:
        self._ensure_open()
        with self._state_lock:
            with self._refresh_even_on_persistence_failure():
                revision = self.editing.commit(transaction)
            self._refresh_after_graph_edit()
            return revision

    def reject_edit(self, transaction: EditTransaction) -> None:
        self._ensure_open()
        self.editing.reject(transaction)

    def prepare_transformation(
        self,
        request: TransformationRequest,
        *,
        token: CancellationToken | None = None,
    ) -> TransformationProposal:
        self._ensure_open()
        return self.editing.prepare_transformation(request, token=token)

    def commit_transformation(
        self,
        proposal: TransformationProposal,
    ) -> Revision:
        self._ensure_open()
        with self._state_lock:
            with self._refresh_even_on_persistence_failure():
                revision = self.editing.commit_transformation(proposal)
            self._refresh_after_graph_edit()
            return revision

    def reject_transformation(self, proposal: TransformationProposal) -> None:
        self._ensure_open()
        self.editing.reject_transformation(proposal)

    def calibrate_async(
        self,
        provider: CalibrationProvider,
        samples: Iterable[Mapping[str, NDArray[np.generic]]],
        *,
        limits: CalibrationLimits | None = None,
    ) -> Job[CalibrationResult]:
        """Run an explicitly supplied calibration provider outside commands."""
        self._ensure_open()
        if self._jobs is None:
            raise SessionError("this session has no job manager")
        return self._jobs.submit(
            f"calibrate {provider.provider_id}",
            lambda token: run_calibration(
                provider,
                samples,
                token=token,
                limits=limits,
            ),
        )

    def undo_edit(self) -> Revision:
        self._ensure_open()
        with self._state_lock:
            with self._refresh_even_on_persistence_failure():
                revision = self.editing.undo()
            self._refresh_after_graph_edit()
            return revision

    def redo_edit(self) -> Revision:
        self._ensure_open()
        with self._state_lock:
            with self._refresh_even_on_persistence_failure():
                revision = self.editing.redo()
            self._refresh_after_graph_edit()
            return revision

    @contextlib.contextmanager
    def _refresh_even_on_persistence_failure(self) -> Iterator[None]:
        """Keep caches aligned with the chain when only persistence failed.

        A :class:`SidecarPersistenceError` means the revision *is* applied;
        the slicer and hierarchy must follow the in-memory truth before the
        warning propagates, exactly as they would on success.
        """
        try:
            yield
        except SidecarPersistenceError:
            self._refresh_after_graph_edit()
            raise

    def _refresh_after_graph_edit(self) -> None:
        self.hierarchy = HierarchyController(
            self.document,
            store=self._hierarchy_store,
        )

    def export_plan(self) -> ExportPlan:
        """Describe the concrete export path the shell should offer."""
        status = self.capability(Capability.EXPORT)
        stem = Path(self.title).stem
        if self.document.artifact_kind is ArtifactKind.ONNX_MODEL:
            return ExportPlan(
                dialog_title="Export committed ONNX revision",
                file_name=f"{stem}-edited.onnx",
                allowed_extensions=("onnx",),
                fidelity=ExportFidelity.LOSSLESS,
                available=status.availability
                in {Availability.AVAILABLE, Availability.PARTIAL},
                reason=status.reason,
            )
        weights_only = {
            ArtifactKind.PYTORCH_EXPORTED_PROGRAM,
            ArtifactKind.PYTORCH_STATE_DICT,
            ArtifactKind.SAFETENSORS,
            ArtifactKind.ORBAX_CHECKPOINT,
        }
        if self.document.artifact_kind in weights_only:
            return ExportPlan(
                dialog_title="Export readable tensors as safetensors",
                file_name=f"{stem}-edited.safetensors",
                allowed_extensions=("safetensors",),
                fidelity=ExportFidelity.WEIGHTS_ONLY,
                available=status.availability
                in {Availability.AVAILABLE, Availability.PARTIAL},
                reason=status.reason,
            )
        return ExportPlan(
            dialog_title="Export unavailable",
            file_name=f"{stem}-edited",
            allowed_extensions=(),
            fidelity=ExportFidelity.UNAVAILABLE,
            available=False,
            reason=status.reason,
        )

    def export(
        self,
        destination: Path | str,
        *,
        external_data_threshold: int = 64 * 1024 * 1024,
        smoke_inputs: dict[str, NDArray[np.generic]] | None = None,
        approve_numerical_execution: bool = False,
        token: CancellationToken | None = None,
    ) -> ExportReport:
        self._ensure_open()
        return export_revision(
            self.document,
            self.editing.applied_revisions,
            destination,
            tensor_reader=self.editing.read,
            external_data_threshold=external_data_threshold,
            smoke_inputs=smoke_inputs,
            approve_numerical_execution=approve_numerical_execution,
            token=token,
        )

    def export_weights_only(
        self,
        destination: Path | str,
        *,
        token: CancellationToken | None = None,
    ) -> Path:
        """Write every readable tensor, with edits applied, as safetensors.

        The supported export path for weights-only artifacts (P6.5/P7.5):
        checkpoints, safetensors, and Orbax trees have no graph to
        re-serialize, so their fidelity ceiling *is* weights-only and the
        result is labelled as such. Tensors whose bytes could not be located
        are skipped rather than written as zeros — inventing data would be
        worse than an incomplete, honestly reported export.
        """
        self._ensure_open()
        payload: list[tuple[str, str, tuple[int, ...], bytes]] = []
        for tensor_id, tensor in self.document.tensors.items():
            if token is not None:
                token.raise_if_cancelled()
            if tensor.storage is Storage.ABSENT:
                continue
            name = tensor_id.split("#", 1)[-1]
            payload.append(
                (
                    name,
                    tensor.element_type,
                    tensor.dims,
                    self.editing.read(tensor_id),
                )
            )
        if not payload:
            raise SessionError("this artifact holds no readable tensors to export")
        try:
            return write_safetensors(
                destination,
                payload,
                metadata={
                    "producer": "nneditor",
                    "source_hash": self.document.source.content_hash,
                    "source_kind": self.document.artifact_kind.value,
                    "fidelity": "weights only",
                },
            )
        except SafetensorsError as error:
            raise SessionError(str(error)) from error

    def export_artifact(
        self,
        destination: Path | str,
        *,
        external_data_threshold: int = 64 * 1024 * 1024,
        token: CancellationToken | None = None,
    ) -> ExportOutcome:
        """Dispatch export according to the artifact's implemented lifecycle."""
        self._ensure_open()
        plan = self.export_plan()
        if not plan.available:
            raise SessionError(plan.reason)
        if self.document.artifact_kind is ArtifactKind.ONNX_MODEL:
            report = self.export(
                destination,
                external_data_threshold=external_data_threshold,
                token=token,
            )
            return ExportOutcome(
                written_files=tuple(report.written_files),
                fidelity=report.fidelity,
                report_path=report.report_path,
            )
        written = self.export_weights_only(destination, token=token)
        return ExportOutcome(
            written_files=(written,),
            fidelity=ExportFidelity.WEIGHTS_ONLY,
        )

    def export_artifact_async(
        self,
        destination: Path | str,
        *,
        external_data_threshold: int = 64 * 1024 * 1024,
    ) -> Job[ExportOutcome]:
        """Run the format-routed export through the shared job boundary."""
        self._ensure_open()
        if self._jobs is None:
            raise SessionError("this session has no job manager")
        return self._jobs.submit(
            f"export {Path(destination).name}",
            lambda token: self.export_artifact(
                destination,
                external_data_threshold=external_data_threshold,
                token=token,
            ),
        )

    def export_async(
        self,
        destination: Path | str,
        *,
        external_data_threshold: int = 64 * 1024 * 1024,
    ) -> Job[ExportReport]:
        self._ensure_open()
        if self._jobs is None:
            raise SessionError("this session has no job manager")
        return self._jobs.submit(
            f"export {Path(destination).name}",
            lambda token: self._export_with_token(
                destination,
                external_data_threshold,
                token,
            ),
        )

    def _export_with_token(
        self,
        destination: Path | str,
        external_data_threshold: int,
        token: CancellationToken,
    ) -> ExportReport:
        token.raise_if_cancelled()
        report = self.export(
            destination,
            external_data_threshold=external_data_threshold,
            token=token,
        )
        return report

    # -- lifecycle ---------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise SessionError("the session is closed")

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Release this session's tensor store.

        Layout entries are immutable and content/revision keyed, so they stay
        available to sibling sessions and future reopens until the shared
        bounded cache evicts them.
        """
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            if self._prewarm_job is not None:
                # cancel() only flips the token; the walk stops at its next
                # between-level checkpoint, so close never waits on it.
                self._prewarm_job.cancel()
            self.store.close()


class ApplicationService:
    """The root object the UI shell talks to."""

    __slots__ = (
        "_activation_store",
        "_adapters",
        "_counter",
        "_edits_store",
        "_hierarchy_store",
        "_input_store",
        "_lock",
        "_prewarm",
        "_sessions",
        "_slicer",
        "_stats_store",
        "_trace_temp",
        "_uploads",
        "jobs",
        "state_store",
    )

    def __init__(
        self,
        *,
        job_listener: JobListener | None = None,
        state_store: SessionStateStore | None = None,
        hierarchy_store: HierarchySidecarStore | None = None,
        statistics_store: StatisticsSidecarStore | None = None,
        edits_store: RevisionSidecarStore | None = None,
        adapter_registry: ArtifactAdapterRegistry | None = None,
        activation_store: ActivationStore | None = None,
        input_store: InputSpecificationStore | None = None,
        prewarm_layouts: bool = True,
    ) -> None:
        self.jobs = JobManager(listener=job_listener)
        self._prewarm = prewarm_layouts
        self._adapters = adapter_registry or default_artifact_adapter_registry()
        self._slicer = GraphSlicer()
        self._sessions: dict[int, ModelSession] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self.state_store = state_store
        self._hierarchy_store = hierarchy_store
        if self._hierarchy_store is None and state_store is not None:
            self._hierarchy_store = HierarchySidecarStore(state_store.path.parent)
        self._stats_store = statistics_store
        if self._stats_store is None and state_store is not None:
            self._stats_store = StatisticsSidecarStore(state_store.path.parent)
        self._edits_store = edits_store
        if self._edits_store is None and state_store is not None:
            self._edits_store = RevisionSidecarStore(state_store.path.parent)
        self._trace_temp: tempfile.TemporaryDirectory[str] | None = None
        if activation_store is None or input_store is None:
            if state_store is not None:
                trace_root = state_store.path.parent
            else:
                self._trace_temp = tempfile.TemporaryDirectory(
                    prefix="nneditor-traces-"
                )
                trace_root = Path(self._trace_temp.name)
            if activation_store is None:
                activation_store = ActivationStore(trace_root / "activations")
            if input_store is None:
                input_store = InputSpecificationStore(trace_root)
        self._activation_store = activation_store
        self._input_store = input_store
        self._uploads = tempfile.TemporaryDirectory(prefix="nneditor-upload-")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # -- opening models ----------------------------------------------------

    def open_model(self, path: Path | str) -> ModelSession:
        """Open an artifact synchronously; raises on unreadable artifacts."""
        document = self._import(Path(path), None)
        hierarchy = HierarchyController(document, store=self._hierarchy_store)
        with self._lock:
            self._counter += 1
            session_id = self._counter
        # Session construction (sidecar recovery, edit replay) is expensive;
        # holding the service lock through it would stall every other open.
        session = ModelSession(
            session_id,
            document,
            self._slicer,
            hierarchy,
            jobs=self.jobs,
            stats_sidecar=self._stats_store,
            edits_sidecar=self._edits_store,
            hierarchy_store=self._hierarchy_store,
            activation_store=self._activation_store,
            input_store=self._input_store,
        )
        with self._lock:
            self._sessions[session.id] = session
        if self._prewarm:
            # Scheduled only after the session is registered and fully
            # constructed, so the job never observes a half-built session.
            session._schedule_layout_prewarm()
        return session

    def open_model_async(self, path: Path | str) -> Job[ModelSession]:
        """Open and prepare an artifact on the job pool.

        Warming the first architecture slice here is important: graph layout is
        often more expensive than parsing, and doing it from the shell's
        completion callback would still freeze a web client's event loop.
        """
        target = Path(path)

        def work(token: CancellationToken) -> ModelSession:
            return self._open_and_prepare(target, token)

        return self.jobs.submit(f"open {target.name}", work)

    def open_upload_async(
        self,
        name: str,
        data: bytes,
        *,
        max_bytes: int = 256 * 1024 * 1024,
    ) -> Job[ModelSession]:
        """Stage a browser upload and prepare it entirely on the worker pool."""

        def work(token: CancellationToken) -> ModelSession:
            token.raise_if_cancelled()
            target = self.stage_upload(name, data, max_bytes=max_bytes)
            token.raise_if_cancelled()
            return self._open_and_prepare(target, token)

        return self.jobs.submit(f"upload and open {Path(name).name}", work)

    def open_uploaded_file_async(
        self,
        name: str,
        uploaded_path: Path | str,
        *,
        max_bytes: int = 256 * 1024 * 1024,
    ) -> Job[ModelSession]:
        """Adopt a streamed Flet upload and prepare it on the worker pool."""
        source = Path(uploaded_path)

        def work(token: CancellationToken) -> ModelSession:
            token.raise_if_cancelled()
            safe_name = Path(name).name or "uploaded-artifact"
            size = source.stat().st_size
            if size > max_bytes:
                raise SessionError(
                    f"upload is {size:,} bytes; the current limit is "
                    f"{max_bytes:,} bytes"
                )
            upload_directory = Path(
                tempfile.mkdtemp(prefix="artifact-", dir=self._uploads.name)
            )
            target = upload_directory / safe_name
            shutil.copyfile(source, target)
            token.raise_if_cancelled()
            return self._open_and_prepare(target, token)

        return self.jobs.submit(f"open uploaded {Path(name).name}", work)

    def _open_and_prepare(self, target: Path, token: CancellationToken) -> ModelSession:
        document = self._import(target, token)
        token.raise_if_cancelled()
        hierarchy = HierarchyController(
            document, store=self._hierarchy_store, token=token
        )
        token.raise_if_cancelled()
        with self._lock:
            self._counter += 1
            session_id = self._counter
        session = ModelSession(
            session_id,
            document,
            self._slicer,
            hierarchy,
            jobs=self.jobs,
            stats_sidecar=self._stats_store,
            edits_sidecar=self._edits_store,
            hierarchy_store=self._hierarchy_store,
            activation_store=self._activation_store,
            input_store=self._input_store,
        )
        try:
            # Layout and semantic aggregation are part of opening, not
            # presenting. The shell receives a cached slice and only creates
            # lightweight Flet controls after the worker has finished.
            session.scene(
                document.entry_graph,
                detail_level=DetailLevel.ARCHITECTURE,
            )
            token.raise_if_cancelled()
        except BaseException:
            session.close()
            raise
        with self._lock:
            self._sessions[session.id] = session
        if self._prewarm:
            # Scheduled only after the session is registered and fully
            # constructed, so the job never observes a half-built session.
            session._schedule_layout_prewarm()
        return session

    def stage_upload(
        self,
        name: str,
        data: bytes,
        *,
        max_bytes: int = 256 * 1024 * 1024,
    ) -> Path:
        """Stage a browser upload in this service's private temporary root."""
        if len(data) > max_bytes:
            raise SessionError(
                f"upload is {len(data):,} bytes; the current limit is "
                f"{max_bytes:,} bytes"
            )
        safe_name = Path(name).name
        if not safe_name:
            safe_name = "uploaded-artifact"
        upload_directory = Path(
            tempfile.mkdtemp(prefix="artifact-", dir=self._uploads.name)
        )
        target = upload_directory / safe_name
        target.write_bytes(data)
        return target

    def _import(self, path: Path, token: CancellationToken | None) -> Document:
        self._validate_source(path)
        if token is not None:
            token.raise_if_cancelled()
        document = self._read_document(path, token)
        if token is not None:
            token.raise_if_cancelled()
        if self.state_store is not None:
            # record_open is a read-modify-write on the recents list; two
            # concurrent open jobs would lose one entry without the lock.
            with self._lock:
                self.state_store.state.record_open(
                    str(path), document.source.content_hash, path.name
                )
                self.state_store.save()
        return document

    def _read_document(
        self,
        path: Path,
        token: CancellationToken | None,
    ) -> Document:
        """Open through the content-detected artifact adapter registry."""
        try:
            return self._adapters.open(path, token=token)
        except ArtifactAdapterError as error:
            raise SessionError(str(error)) from error

    @staticmethod
    def _validate_source(path: Path) -> None:
        """Reject unusable open targets with a user-explainable error.

        The path is user-chosen and therefore untrusted: it may point at
        nothing, at a directory, or at something unreadable, and each case
        must surface as a session error instead of a raw OS traceback.
        """
        if not path.exists():
            raise SessionError(f"{path} does not exist")
        if not path.is_file():
            raise SessionError(f"{path} is not a file")
        try:
            with open(path, "rb") as handle:
                handle.read(1)
        except OSError as error:
            raise SessionError(f"{path} cannot be read: {error}") from error

    # -- session registry --------------------------------------------------

    def session(self, session_id: int) -> ModelSession:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionError(f"no session {session_id}") from None

    @property
    def open_sessions(self) -> tuple[ModelSession, ...]:
        with self._lock:
            return tuple(
                session for session in self._sessions.values() if not session.closed
            )

    def close_session(self, session_id: int) -> None:
        session = self.session(session_id)
        session.close()
        with self._lock:
            self._sessions.pop(session_id, None)

    def close(self) -> None:
        """Close every session, persist state, and stop the job pool.

        Shutdown must be complete rather than fail-fast: one session failing
        to close must not keep the remaining stores open, the state unsaved,
        or the staged uploads on disk.
        """
        self.jobs.close()
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        try:
            for session in sessions:
                with contextlib.suppress(Exception):
                    session.close()
        finally:
            try:
                if self.state_store is not None:
                    with contextlib.suppress(Exception):
                        self.state_store.save()
            finally:
                self._uploads.cleanup()
                if self._trace_temp is not None:
                    self._trace_temp.cleanup()
