"""Regression tests for the JAX/StableHLO adapter security fixes.

The textual MLIR reader parses untrusted input, so each test here pins one
of three contracts: bounded work on hostile text, a typed diagnostic instead
of a raw exception, and no silently-wrong model.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from nneditor.adapters.jax import checkpoint as checkpoint_module
from nneditor.adapters.jax.checkpoint import OrbaxError, open_orbax_checkpoint
from nneditor.adapters.jax.mlir_text import (
    MlirParseError,
    _has_ssa_assignment,
    _outside_strings,
    _parse_operation,
    parse_module,
    parse_type,
)
from nneditor.adapters.jax.stablehlo import StableHloError, open_stablehlo
from nneditor.adapters.pytorch.safetensors import write_safetensors

# A generous ceiling: the quadratic originals took seconds on inputs this
# size, the linear replacements take milliseconds.
_PERF_BUDGET_SECONDS = 2.0


def elapsed(work: Callable[[], object]) -> float:
    start = time.perf_counter()
    work()
    return time.perf_counter() - start


# --- type parsing -----------------------------------------------------------


class TestTypeParsing:
    def test_well_formed_types_are_unchanged(self) -> None:
        assert parse_type("tensor<2x3xf32>").shape == (2, 3)
        assert parse_type("tensor<?x4xf32>").shape == (None, 4)
        assert parse_type("tensor<f32>").shape == ()
        assert parse_type("tensor<2xindex>").element_type == "index"
        assert parse_type("tensor<2x3xf32, #enc>").shape == (2, 3)

    def test_complex_element_types_survive(self) -> None:
        # `body.split("x")` used to tear `complex<f32>` apart, yielding a
        # bogus 'comple' dimension and element type '<f32>'.
        parsed = parse_type("tensor<2x3xcomplex<f32>>")
        assert parsed.element_type == "complex64"
        assert parsed.shape == (2, 3)
        assert parse_type("tensor<complex<f64>>").element_type == "complex128"

    def test_unicode_digits_are_not_dimensions(self) -> None:
        # '²'.isdigit() is True but int('²') raises; the old guard let it
        # straight through to a ValueError.
        with pytest.raises(MlirParseError):
            parse_type("tensor<²xf32>")

    def test_absurdly_long_dimensions_are_refused(self) -> None:
        with pytest.raises(MlirParseError):
            parse_type("tensor<" + "9" * 5000 + "xf32>")

    def test_empty_dimensions_are_refused(self) -> None:
        with pytest.raises(MlirParseError):
            parse_type("tensor<xxf32>")
        with pytest.raises(MlirParseError):
            parse_type("tensor<2xxf32>")

    def test_a_module_using_a_malformed_type_is_refused_cleanly(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "bad.mlir"
        path.write_text(
            "module @m {\n"
            "  func.func public @main(%arg0: tensor<xxf32>) -> tensor<xxf32> {\n"
            "    return %arg0 : tensor<xxf32>\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
        with pytest.raises(StableHloError):
            open_stablehlo(path)


# --- function declarations --------------------------------------------------


DECLARATION_MODULE = """
module @m {
  func.func private @callee(%arg0: tensor<4xf32>) -> tensor<4xf32>
  func.func public @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.tanh %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""


class TestDeclarations:
    def test_a_body_less_declaration_does_not_swallow_the_next_body(self) -> None:
        module = parse_module(DECLARATION_MODULE)
        names = [function.name for function in module.functions]
        assert names == ["callee", "main"], "@main must survive the declaration"
        declaration = module.functions[0]
        assert declaration.operations == (), "a declaration has no operations"
        assert declaration.arguments and declaration.results
        body = module.functions[1]
        assert [operation.name for operation in body.operations] == ["tanh"]
        assert body.returns == ("%0",)

    def test_a_declaration_only_module_still_opens(self, tmp_path: Path) -> None:
        path = tmp_path / "decl.mlir"
        path.write_text(DECLARATION_MODULE, encoding="utf-8")
        document = open_stablehlo(path)
        graphs = list(document.graphs.values())
        assert {graph.name for graph in graphs} == {"callee", "main"}
        main = next(graph for graph in graphs if graph.name == "main")
        assert [node.op_type for node in main.nodes] == ["tanh"]
        callee = next(graph for graph in graphs if graph.name == "callee")
        assert callee.nodes == ()


# --- duplicate function names -----------------------------------------------


class TestDuplicateNames:
    def test_duplicate_function_names_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "dup.mlir"
        path.write_text(
            "module @m {\n"
            "  func.func public @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {\n"
            "    return %arg0 : tensor<2xf32>\n"
            "  }\n"
            "  func.func private @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {\n"
            "    return %arg0 : tensor<2xf32>\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
        with pytest.raises(StableHloError, match="duplicate function name"):
            open_stablehlo(path)


# --- performance guards -----------------------------------------------------


class TestPerformanceGuards:
    def test_func_header_scan_is_linear(self) -> None:
        # `\\s*(optional)?\\s*` used to backtrack quadratically: 80 KB of
        # whitespace after `func.func` took 5.9 s.
        hostile = "module @m {\n func.func" + " " * 100_000 + "\n}\n"
        start = time.perf_counter()
        with pytest.raises(MlirParseError):
            parse_module(hostile)
        duration = time.perf_counter() - start
        assert duration < _PERF_BUDGET_SECONDS, duration

    def test_region_detection_is_linear(self) -> None:
        # `%\\S+\\s*=` used to backtrack on a long operand run with no `=`.
        hostile = "foo.bar {" + "%" + "a" * 100_000 + "}"
        duration = elapsed(lambda: _parse_operation(hostile))
        assert duration < _PERF_BUDGET_SECONDS, duration

    def test_string_span_filtering_is_linear(self) -> None:
        # `_outside_strings` compared every match against every span.
        hostile = '"module" ' * 10_000
        pattern = re.compile(r"\bmodule\b")
        duration = elapsed(lambda: _outside_strings(hostile, pattern))
        assert duration < _PERF_BUDGET_SECONDS, duration

    def test_hostile_modules_open_or_refuse_quickly(self, tmp_path: Path) -> None:
        path = tmp_path / "hostile.mlir"
        path.write_text(
            "module @m {\n  func.func" + " " * 50_000 + '\n  "loc"' * 5_000 + "\n}\n",
            encoding="utf-8",
        )
        start = time.perf_counter()
        with pytest.raises(StableHloError):
            open_stablehlo(path)
        assert time.perf_counter() - start < _PERF_BUDGET_SECONDS


class TestSsaAssignmentScan:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("%0 = stablehlo.add", True),
            ("%0=stablehlo.add", True),
            ("%a.b$c   =  x", True),
            ("stablehlo.add %0, %1", False),
            ("% = x", False),
            ("", False),
            ("return %0", False),
            ("foo = 1", False),
            ("%0", False),
            ("%0 %1 = x", True),
        ],
    )
    def test_matches_the_regex_it_replaces(self, text: str, expected: bool) -> None:
        assert _has_ssa_assignment(text) is expected
        assert bool(re.search(r"%\S+\s*=", text)) is expected


# --- orbax checkpoint metadata bounds ---------------------------------------


class TestOrbaxBounds:
    def _checkpoint(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        write_safetensors(
            root / "params.safetensors",
            [("dense.kernel", "float32", (2, 2), b"\x00" * 16)],
        )
        return root

    def test_oversized_metadata_is_skipped_with_a_diagnostic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(checkpoint_module, "_MAX_TREE_FILE_BYTES", 128)
        root = self._checkpoint(tmp_path / "ckpt")
        (root / "small.json").write_text(json.dumps({"ok": 1}), encoding="utf-8")
        (root / "huge.json").write_text(
            json.dumps({"pad": "x" * 4096}), encoding="utf-8"
        )
        document = open_orbax_checkpoint(root)
        codes = {item.code for item in document.diagnostics}
        assert "jax.tree-metadata-too-large" in codes
        tree = dict(document.extensions)["x-orbax.tree"]
        assert isinstance(tree, dict)
        assert set(tree) == {"small.json"}

    def test_too_many_metadata_files_are_capped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(checkpoint_module, "_MAX_TREE_FILES", 3)
        root = self._checkpoint(tmp_path / "many")
        for index in range(10):
            (root / f"m{index}.json").write_text("{}", encoding="utf-8")
        document = open_orbax_checkpoint(root)
        codes = {item.code for item in document.diagnostics}
        assert "jax.tree-metadata-truncated" in codes
        tree = dict(document.extensions)["x-orbax.tree"]
        assert isinstance(tree, dict) and len(tree) == 3

    def test_unexpected_failures_become_orbax_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = self._checkpoint(tmp_path / "boom")

        def explode(path: Path) -> None:
            raise RecursionError("synthetic blow-up")

        monkeypatch.setattr(checkpoint_module, "open_safetensors", explode)
        with pytest.raises(OrbaxError, match="RecursionError"):
            open_orbax_checkpoint(root)


# --- entry-point contract ---------------------------------------------------


class TestEntryPointContract:
    def test_unexpected_failures_become_stablehlo_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nneditor.adapters.jax import stablehlo as stablehlo_module

        path = tmp_path / "m.mlir"
        path.write_text(
            "module @m {\n"
            "  func.func public @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {\n"
            "    return %arg0 : tensor<2xf32>\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )

        def explode(*_args: object, **_kwargs: object) -> None:
            raise RecursionError("synthetic blow-up")

        monkeypatch.setattr(stablehlo_module, "_convert_function", explode)
        with pytest.raises(StableHloError, match="RecursionError"):
            open_stablehlo(path)
