"""Tests for the artifact and capability contracts (P0.2)."""

from __future__ import annotations

import pytest

from nneditor.ir.capabilities import (
    ARTIFACT_CONTRACTS,
    ArtifactContract,
    ArtifactKind,
    Availability,
    Capability,
    CapabilityStatus,
    ExportFidelity,
    LoadingMode,
    capability_matrix_markdown,
    contract_for,
)


def test_every_artifact_kind_has_a_contract() -> None:
    assert set(ARTIFACT_CONTRACTS) == set(ArtifactKind)


@pytest.mark.parametrize("kind", list(ArtifactKind))
def test_every_capability_has_a_reason(kind: ArtifactKind) -> None:
    contract = contract_for(kind)
    for capability in Capability:
        status = contract.status(capability)
        assert status.reason.strip(), f"{kind} {capability} has no reason"
        assert status.reason.endswith("."), "reasons are shown as sentences"


def test_status_lookup_matches_availability() -> None:
    contract = contract_for(ArtifactKind.ONNX_MODEL)
    assert contract.availability(Capability.TOPOLOGY) is Availability.AVAILABLE
    assert contract.status(Capability.TOPOLOGY).is_usable
    assert not contract.status(Capability.EDITING).is_usable


def test_artifacts_without_topology_cannot_promise_a_graph() -> None:
    """An artifact without a graph must never advertise executable topology.

    Its export ceiling is weights-only at best — or unavailable when, as for
    a DLC container, not even individual weights can be located.
    """
    for contract in ARTIFACT_CONTRACTS.values():
        if contract.availability(Capability.TOPOLOGY) is Availability.UNAVAILABLE:
            assert contract.availability(Capability.EXECUTION) is (
                Availability.UNAVAILABLE
            )
            assert contract.best_export_fidelity in (
                ExportFidelity.WEIGHTS_ONLY,
                ExportFidelity.UNAVAILABLE,
            )


def test_topology_behind_trust_requires_the_trusted_loading_mode() -> None:
    """If a graph only exists after running user code, say so in both places."""
    for contract in ARTIFACT_CONTRACTS.values():
        if contract.availability(Capability.TOPOLOGY) is (
            Availability.REQUIRES_TRUSTED_MODE
        ):
            assert LoadingMode.TRUSTED_CODE in contract.loading_modes


def test_only_onnx_claims_lossless_export() -> None:
    lossless = [
        contract.kind
        for contract in ARTIFACT_CONTRACTS.values()
        if contract.best_export_fidelity is ExportFidelity.LOSSLESS
    ]
    assert lossless == [ArtifactKind.ONNX_MODEL]


def test_capability_status_rejects_an_empty_reason() -> None:
    with pytest.raises(ValueError, match="non-empty reason"):
        CapabilityStatus(Capability.EDITING, Availability.UNAVAILABLE, "   ")


def _minimal_statuses() -> tuple[CapabilityStatus, ...]:
    return tuple(
        CapabilityStatus(capability, Availability.UNAVAILABLE, "Because.")
        for capability in Capability
    )


def test_contract_rejects_a_missing_capability() -> None:
    with pytest.raises(ValueError, match="missing capability status"):
        ArtifactContract(
            kind=ArtifactKind.SAFETENSORS,
            title="broken",
            preferred_inputs=("x",),
            loading_modes=(LoadingMode.SAFE_ARTIFACT,),
            best_export_fidelity=ExportFidelity.UNAVAILABLE,
            statuses=_minimal_statuses()[:-1],
        )


def test_contract_rejects_a_duplicate_capability() -> None:
    duplicated = (
        *_minimal_statuses(),
        CapabilityStatus(Capability.EXPORT, Availability.AVAILABLE, "Again."),
    )
    with pytest.raises(ValueError, match="duplicate capability status"):
        ArtifactContract(
            kind=ArtifactKind.SAFETENSORS,
            title="broken",
            preferred_inputs=("x",),
            loading_modes=(LoadingMode.SAFE_ARTIFACT,),
            best_export_fidelity=ExportFidelity.UNAVAILABLE,
            statuses=duplicated,
        )


@pytest.mark.parametrize(
    ("inputs", "modes", "message"),
    [
        ((), (LoadingMode.SAFE_ARTIFACT,), "preferred input"),
        (("x",), (), "loading mode"),
    ],
)
def test_contract_requires_inputs_and_modes(
    inputs: tuple[str, ...],
    modes: tuple[LoadingMode, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ArtifactContract(
            kind=ArtifactKind.SAFETENSORS,
            title="broken",
            preferred_inputs=inputs,
            loading_modes=modes,
            best_export_fidelity=ExportFidelity.UNAVAILABLE,
            statuses=_minimal_statuses(),
        )


def test_generated_capability_matrix_covers_the_registry() -> None:
    """The generated matrix is complete without relying on internal docs."""
    matrix = capability_matrix_markdown()
    for contract in ARTIFACT_CONTRACTS.values():
        assert f"### {contract.title}" in matrix
        for status in contract.statuses:
            assert (
                f"| {status.capability.value} | {status.availability.value} | "
                f"{status.reason} |"
            ) in matrix
