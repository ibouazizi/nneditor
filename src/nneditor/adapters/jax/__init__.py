"""JAX / StableHLO artifact adapters (Phase 7).

Textual StableHLO modules — what ``jax.export`` and ``jax.jit(...).lower()``
emit — are parsed natively: MLIR's textual form is regular enough that a
focused reader recovers functions, operations, values, and types without
linking MLIR or executing any JAX code. Orbax/Flax checkpoint trees open
weights-only. ``jax`` itself is a dev-only dependency used to generate
fixtures and validate this reader.
"""

from nneditor.adapters.jax.checkpoint import open_orbax_checkpoint
from nneditor.adapters.jax.stablehlo import (
    StableHloError,
    open_stablehlo,
    parse_stablehlo,
)

__all__ = [
    "StableHloError",
    "open_orbax_checkpoint",
    "open_stablehlo",
    "parse_stablehlo",
]
