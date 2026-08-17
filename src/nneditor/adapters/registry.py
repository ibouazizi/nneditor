"""Artifact adapter protocol and registry.

Opening an artifact is an application-wide extension point, not a chain of
``if`` statements owned by the session service.  Each adapter receives the
same cancellation boundary and must translate format-specific failures into
one stable error type before they reach the UI.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from nneditor.adapters.coreml.package import CoremlError, open_coreml_package
from nneditor.adapters.detect import DetectionError, detect_artifact_kind
from nneditor.adapters.jax.stablehlo import StableHloError, open_stablehlo
from nneditor.adapters.onnx import OnnxIndexError, index_model, index_to_document
from nneditor.adapters.pytorch.checkpoint import CheckpointError, open_checkpoint
from nneditor.adapters.pytorch.fx import FxError, open_fx_graph_module
from nneditor.adapters.pytorch.pt2 import Pt2Error, open_pt2
from nneditor.adapters.pytorch.safetensors import SafetensorsError, open_safetensors
from nneditor.adapters.qualcomm.dlc import DlcError, open_dlc
from nneditor.cancellation import CancellationToken, OperationCancelled
from nneditor.ir.capabilities import ArtifactKind
from nneditor.ir.core import Document

__all__ = [
    "DEFAULT_ARTIFACT_ADAPTERS",
    "ArtifactAdapter",
    "ArtifactAdapterError",
    "ArtifactAdapterRegistry",
    "default_artifact_adapter_registry",
]


class ArtifactAdapterError(ValueError):
    """A supported artifact could not be opened through its adapter."""


@runtime_checkable
class ArtifactAdapter(Protocol):
    """One safe-artifact importer registered under a concrete artifact kind."""

    @property
    def kind(self) -> ArtifactKind:
        """The artifact contract this adapter implements."""

    def open(
        self,
        path: Path,
        *,
        token: CancellationToken | None = None,
    ) -> Document:
        """Open ``path`` without executing artifact-provided code."""


AdapterReader = Callable[[Path, CancellationToken | None], Document]


@dataclass(frozen=True, slots=True)
class _FunctionAdapter:
    kind: ArtifactKind
    reader: AdapterReader

    def open(
        self,
        path: Path,
        *,
        token: CancellationToken | None = None,
    ) -> Document:
        if token is not None:
            token.raise_if_cancelled()
        try:
            document = self.reader(path, token)
        except OperationCancelled:
            raise
        except ArtifactAdapterError:
            raise
        except Exception as error:
            raise ArtifactAdapterError(
                f"{path.name} could not be opened as {self.kind.value}: {error}"
            ) from error
        if token is not None:
            token.raise_if_cancelled()
        return document


class ArtifactAdapterRegistry:
    """Immutable lookup table for content-detected artifact readers."""

    __slots__ = ("_adapters",)

    def __init__(self, adapters: Iterable[ArtifactAdapter]) -> None:
        registered: dict[ArtifactKind, ArtifactAdapter] = {}
        for adapter in adapters:
            if adapter.kind in registered:
                raise ValueError(
                    f"artifact adapter {adapter.kind.value!r} is registered twice"
                )
            registered[adapter.kind] = adapter
        self._adapters = MappingProxyType(registered)

    @property
    def kinds(self) -> frozenset[ArtifactKind]:
        return frozenset(self._adapters)

    def adapter_for(self, kind: ArtifactKind) -> ArtifactAdapter:
        try:
            return self._adapters[kind]
        except KeyError:
            raise ArtifactAdapterError(
                f"no reader is registered for artifact kind {kind.value}"
            ) from None

    def open(
        self,
        path: Path,
        *,
        token: CancellationToken | None = None,
    ) -> Document:
        try:
            kind = detect_artifact_kind(path)
        except DetectionError as error:
            raise ArtifactAdapterError(str(error)) from error
        return self.adapter_for(kind).open(path, token=token)


def _open_onnx(path: Path, token: CancellationToken | None) -> Document:
    try:
        return index_to_document(index_model(path, token=token))
    except OnnxIndexError:
        raise


def _open_pt2(path: Path, _token: CancellationToken | None) -> Document:
    try:
        return open_pt2(path)
    except Pt2Error:
        raise


def _open_safetensors(path: Path, _token: CancellationToken | None) -> Document:
    try:
        return open_safetensors(path)
    except SafetensorsError:
        raise


def _open_checkpoint_or_fx(
    path: Path,
    _token: CancellationToken | None,
) -> Document:
    try:
        return open_checkpoint(path)
    except CheckpointError as checkpoint_error:
        try:
            return open_fx_graph_module(path)
        except FxError as fx_error:
            raise ArtifactAdapterError(
                f"{path.name} is neither a supported state dictionary nor a "
                f"readable FX graph module: checkpoint: {checkpoint_error}; "
                f"FX: {fx_error}"
            ) from fx_error


def _open_stablehlo(path: Path, _token: CancellationToken | None) -> Document:
    try:
        return open_stablehlo(path)
    except StableHloError:
        raise


def _open_dlc(path: Path, _token: CancellationToken | None) -> Document:
    try:
        return open_dlc(path)
    except DlcError:
        raise


def _open_coreml(path: Path, _token: CancellationToken | None) -> Document:
    try:
        return open_coreml_package(path)
    except CoremlError:
        raise


DEFAULT_ARTIFACT_ADAPTERS: tuple[ArtifactAdapter, ...] = (
    _FunctionAdapter(ArtifactKind.ONNX_MODEL, _open_onnx),
    _FunctionAdapter(ArtifactKind.PYTORCH_EXPORTED_PROGRAM, _open_pt2),
    _FunctionAdapter(ArtifactKind.SAFETENSORS, _open_safetensors),
    _FunctionAdapter(ArtifactKind.PYTORCH_STATE_DICT, _open_checkpoint_or_fx),
    _FunctionAdapter(ArtifactKind.JAX_STABLEHLO, _open_stablehlo),
    _FunctionAdapter(ArtifactKind.QUALCOMM_DLC, _open_dlc),
    _FunctionAdapter(ArtifactKind.COREML_PACKAGE, _open_coreml),
)


def default_artifact_adapter_registry() -> ArtifactAdapterRegistry:
    """Return a fresh registry containing the built-in safe readers."""
    return ArtifactAdapterRegistry(DEFAULT_ARTIFACT_ADAPTERS)
