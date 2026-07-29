"""Deterministic JAX / StableHLO fixtures (Phase 7).

``jax`` is a dev-only dependency: it emits the StableHLO text these tests
feed to the native reader, so the reader is validated against real exporter
output rather than hand-written approximations.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from jax import export as jexport

__all__ = [
    "build_control_flow_module",
    "build_custom_call_module",
    "build_mlp_module",
    "build_polymorphic_module",
]


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def build_mlp_module(path: Path) -> Path:
    """A dot_general + tanh function with fully static shapes."""

    def forward(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(jnp.dot(x, w))

    exported = jexport.export(jax.jit(forward))(
        jnp.zeros((2, 3), jnp.float32), jnp.zeros((3, 4), jnp.float32)
    )
    return _write(path, exported.mlir_module())


def build_polymorphic_module(path: Path) -> Path:
    """The same function with a symbolic batch dimension."""
    batch = jexport.symbolic_shape("batch, 3")

    def forward(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(jnp.dot(x, w))

    exported = jexport.export(jax.jit(forward))(
        jax.ShapeDtypeStruct(batch, jnp.float32),  # type: ignore[no-untyped-call]
        jnp.zeros((3, 4), jnp.float32),
    )
    return _write(path, exported.mlir_module())


def build_control_flow_module(path: Path) -> Path:
    """A function whose body contains a nested region (while/cond)."""

    def forward(x: jnp.ndarray) -> jnp.ndarray:
        result = jax.lax.cond(jnp.sum(x) > 0, lambda v: v * 2.0, lambda v: v - 1.0, x)
        return result  # type: ignore[no-any-return]

    exported = jexport.export(jax.jit(forward))(jnp.zeros((4,), jnp.float32))
    return _write(path, exported.mlir_module())


def build_custom_call_module(path: Path) -> Path:
    """A module using a custom call (top-k lowers to one)."""

    def forward(x: jnp.ndarray) -> jnp.ndarray:
        values, _indices = jax.lax.top_k(x, 2)
        return values

    exported = jexport.export(jax.jit(forward))(jnp.zeros((8,), jnp.float32))
    return _write(path, exported.mlir_module())
