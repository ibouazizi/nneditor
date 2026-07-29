"""Resource-bounded plugin boundary for activation-aware calibration.

Core transformation commands never contain a backend quantizer or dataset
callback. A caller may run an explicitly selected provider here, then pass the
provider's serializable provenance and parameters into a later proposal.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from nneditor.cancellation import CancellationToken
from nneditor.editing.cow import EditError

__all__ = [
    "CalibrationLimits",
    "CalibrationProvider",
    "CalibrationResult",
    "run_calibration",
]


class CalibrationProvider(Protocol):
    """External provider contract; implementations are never persisted."""

    @property
    def provider_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    def observe(
        self,
        sample: Mapping[str, NDArray[np.generic]],
        token: CancellationToken,
    ) -> None: ...

    def finish(self, token: CancellationToken) -> Mapping[str, float]: ...


@dataclass(frozen=True, slots=True)
class CalibrationLimits:
    max_samples: int = 128
    max_total_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.max_samples <= 0 or self.max_total_bytes <= 0:
            raise EditError("calibration limits must be positive")


_DEFAULT_LIMITS = CalibrationLimits()


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    provider_id: str
    provider_version: str
    sample_count: int
    total_bytes: int
    parameters: tuple[tuple[str, float], ...]

    @property
    def provenance(self) -> tuple[tuple[str, str], ...]:
        return (
            ("provider", self.provider_id),
            ("provider_version", self.provider_version),
            ("samples", str(self.sample_count)),
            ("bytes", str(self.total_bytes)),
        )


def run_calibration(
    provider: CalibrationProvider,
    samples: Iterable[Mapping[str, NDArray[np.generic]]],
    *,
    token: CancellationToken,
    limits: CalibrationLimits | None = None,
) -> CalibrationResult:
    """Run a provider without allowing it into the revision command model."""
    if not provider.provider_id.strip() or not provider.version.strip():
        raise EditError("calibration provider needs an id and version")
    resolved_limits = limits or _DEFAULT_LIMITS
    count = 0
    total_bytes = 0
    for sample in samples:
        token.raise_if_cancelled()
        if count >= resolved_limits.max_samples:
            raise EditError(
                f"calibration exceeds the {resolved_limits.max_samples}-sample limit"
            )
        sample_bytes = sum(array.nbytes for array in sample.values())
        if total_bytes + sample_bytes > resolved_limits.max_total_bytes:
            raise EditError(
                f"calibration exceeds the {resolved_limits.max_total_bytes}-byte limit"
            )
        provider.observe(sample, token)
        count += 1
        total_bytes += sample_bytes
    token.raise_if_cancelled()
    parameters = provider.finish(token)
    token.raise_if_cancelled()
    normalized = tuple(
        sorted((str(key), float(value)) for key, value in parameters.items())
    )
    if any(not np.isfinite(value) for _key, value in normalized):
        raise EditError("calibration parameters must be finite")
    return CalibrationResult(
        provider_id=provider.provider_id,
        provider_version=provider.version,
        sample_count=count,
        total_bytes=total_bytes,
        parameters=normalized,
    )
