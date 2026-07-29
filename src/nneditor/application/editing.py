"""The session's editing surface (task P3.6).

One :class:`EditingController` per session owns the revision chain, its
sidecar persistence, and edited-view reads. The rules it enforces:

* The source artifact is never written; edits exist only as deltas.
* Every committed mutation persists atomically before the call returns, so a
  process crash loses at most the edit that was mid-validation.
* A sidecar that fails verification on open is quarantined and reported —
  the session recovers at the base revision instead of refusing to open.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from functools import wraps
from typing import cast

from nneditor.application.edits_store import RevisionSidecarStore
from nneditor.cancellation import CancellationToken
from nneditor.editing.cow import EditError
from nneditor.editing.diff import DiffPreview, preview_diff
from nneditor.editing.revisions import Revision, RevisionChain
from nneditor.editing.validation import EditRequest, EditTransaction, ValidationPipeline
from nneditor.ir.core import Document
from nneditor.storage.store import TensorStore, TensorUnavailableError
from nneditor.transformations.engine import (
    TransformationEngine,
    TransformationProposal,
    TransformationRequest,
)

__all__ = ["EditingController", "SidecarPersistenceError"]


def _editing_locked[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    """Serialize one editing operation, including sidecar persistence."""

    @wraps(method)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> R:
        controller = cast("EditingController", args[0])
        with controller._lock:
            return method(*args, **kwargs)

    return guarded


class SidecarPersistenceError(RuntimeError):
    """The mutation applied, but its sidecar persistence failed.

    Raised *after* the revision chain mutated, so the in-memory truth already
    includes ``revision``; callers must treat this as a warning about durability
    (a crash may lose the sidecar copy), not as a failed edit, and must refresh
    any state derived from the document exactly as on success.
    """

    def __init__(self, revision: Revision, cause: BaseException) -> None:
        super().__init__(
            f"revision {revision.id} is applied, but saving the edits sidecar "
            f"failed: {cause}"
        )
        self.revision = revision


class EditingController:
    """Reversible byte-span editing over one opened artifact."""

    __slots__ = (
        "_chain",
        "_content_hash",
        "_document",
        "_lock",
        "_pipeline",
        "_read_base",
        "_read_range",
        "_sidecar",
        "_tensor_length",
        "_transformations",
        "recovered_revisions",
        "sidecar_was_quarantined",
    )

    def __init__(
        self,
        document: Document,
        store: TensorStore,
        *,
        sidecar: RevisionSidecarStore | None = None,
    ) -> None:
        self._document = document
        self._lock = threading.RLock()
        self._content_hash = document.source.content_hash
        self._sidecar = sidecar
        self._read_base = lambda tensor_id: store.read(tensor_id)
        self._read_range = lambda tensor_id, offset, length: store.read(
            tensor_id, offset=offset, length=length
        )
        self._tensor_length = store.byte_length
        self._pipeline = ValidationPipeline()
        self._chain = RevisionChain(
            self._read_base,
            read_range=self._read_range,
            tensor_length=self._tensor_length,
            base_document=document,
        )
        self._transformations = TransformationEngine(
            lambda tensor_id: self.read(tensor_id)
        )
        self.recovered_revisions = 0
        self.sidecar_was_quarantined = False
        if sidecar is not None:
            self._recover(sidecar)

    def _recover(self, sidecar: RevisionSidecarStore) -> None:
        loaded = sidecar.load(self._content_hash)
        if loaded is None:
            # Either nothing was persisted or the store quarantined a corrupt
            # file; distinguishing the two is the store's log, not ours.
            return
        revisions, cursor = loaded
        try:
            self._chain.restore(revisions, cursor)
        except (EditError, TensorUnavailableError):
            sidecar.quarantine(self._content_hash)
            self.sidecar_was_quarantined = True
            return
        except Exception:
            # A transient source read or filesystem failure says nothing
            # about the sidecar's integrity. Leave it in place so a later
            # reopen can recover it.
            return
        self.recovered_revisions = len(revisions)

    # -- observable state --------------------------------------------------

    @property
    @_editing_locked
    def is_dirty(self) -> bool:
        return self._chain.is_dirty

    @property
    @_editing_locked
    def can_undo(self) -> bool:
        return self._chain.can_undo

    @property
    @_editing_locked
    def can_redo(self) -> bool:
        return self._chain.can_redo

    @property
    @_editing_locked
    def current_revision_id(self) -> str | None:
        return self._chain.current_revision_id

    @property
    @_editing_locked
    def document(self) -> Document:
        current = self._chain.document
        assert current is not None
        return current

    @property
    @_editing_locked
    def applied_revisions(self) -> tuple[Revision, ...]:
        return self._chain.applied

    @property
    @_editing_locked
    def delta_byte_cost(self) -> int:
        return self._chain.delta_byte_cost

    @_editing_locked
    def edited_tensor_ids(self) -> tuple[str, ...]:
        return self._chain.edited_tensor_ids()

    # -- reads -------------------------------------------------------------

    @_editing_locked
    def read(
        self, tensor_id: str, *, offset: int = 0, length: int | None = None
    ) -> bytes:
        """The tensor's bytes with every applied edit overlaid."""
        return self._chain.read(tensor_id, offset=offset, length=length)

    @_editing_locked
    def byte_length(self, tensor_id: str) -> int | None:
        return self._chain.byte_length(tensor_id)

    @_editing_locked
    def preview(self) -> DiffPreview:
        """A read-only diff of the applied chain against the base artifact."""
        return preview_diff(self._document, self._chain.applied)

    @_editing_locked
    def prepare(self, request: EditRequest) -> EditTransaction:
        """Validate a request without mutating the revision chain."""
        return self._pipeline.prepare(
            self.document,
            request,
            base_revision_id=self.current_revision_id,
        )

    @_editing_locked
    def commit(self, transaction: EditTransaction) -> Revision:
        """Atomically commit a still-current, successful transaction."""
        if transaction.base_revision_id != self.current_revision_id:
            raise EditError(
                "the working revision changed after validation; validate again"
            )
        if not transaction.ok:
            raise EditError(
                "the transaction has validation errors: "
                + "; ".join(str(item) for item in transaction.findings)
            )
        revision = self._chain.apply_commands(
            transaction.commands,
            transaction.validation_state,
        )
        return self._persist(revision)

    @_editing_locked
    def prepare_transformation(
        self,
        request: TransformationRequest,
        *,
        token: CancellationToken | None = None,
    ) -> TransformationProposal:
        return self._transformations.prepare(
            self.document,
            request,
            base_revision_id=self.current_revision_id,
            token=token,
        )

    @_editing_locked
    def commit_transformation(self, proposal: TransformationProposal) -> Revision:
        if proposal.base_revision_id != self.current_revision_id:
            raise EditError(
                "the working revision changed after transformation preview; "
                "preview again"
            )
        if not proposal.ok:
            raise EditError(
                "the transformation has validation errors: "
                + "; ".join(str(item) for item in proposal.findings)
            )
        revision = self._chain.apply_commands(
            proposal.commands,
            proposal.validation_state,
        )
        return self._persist(revision)

    @staticmethod
    def reject_transformation(_proposal: TransformationProposal) -> None:
        """Rejecting a preview is intentionally a no-op."""

    @staticmethod
    def reject(_transaction: EditTransaction) -> None:
        """Rejecting a prepared transaction is intentionally a no-op."""

    # -- mutation ----------------------------------------------------------

    @_editing_locked
    def replace_bytes(
        self, tensor_id: str, offset: int, replacement: bytes
    ) -> Revision:
        """Commit one validated replacement and persist the chain."""
        revision = self._chain.apply_replace(tensor_id, offset, replacement)
        return self._persist(revision)

    @_editing_locked
    def undo(self) -> Revision:
        return self._persist(self._chain.undo())

    @_editing_locked
    def redo(self) -> Revision:
        return self._persist(self._chain.redo())

    @_editing_locked
    def discard_all(self) -> None:
        """Drop every revision and the sidecar; back to the source artifact."""
        self._chain = RevisionChain(
            self._read_base,
            read_range=self._read_range,
            tensor_length=self._tensor_length,
            base_document=self._document,
        )
        if self._sidecar is not None:
            self._sidecar.discard(self._content_hash)

    def _persist(self, revision: Revision) -> Revision:
        """Persist the chain; a failure is reported, never a rollback.

        The chain has already mutated when this runs, so raising a plain
        exception would present an applied revision as a failed one. A
        persistence failure therefore surfaces as the distinct
        :class:`SidecarPersistenceError` carrying the applied revision, which
        callers can treat as a durability warning.
        """
        if self._sidecar is not None:
            try:
                self._sidecar.save(
                    self._content_hash,
                    list(self._chain.revisions),
                    len(self._chain.applied),
                )
            except Exception as error:
                raise SidecarPersistenceError(revision, error) from error
        return revision

    @contextlib.contextmanager
    def stable_view(self) -> Iterator[EditingController]:
        """Prevent a revision change across a multi-read background job."""
        with self._lock:
            yield self
