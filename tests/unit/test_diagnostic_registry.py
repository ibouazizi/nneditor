"""Tests for the diagnostic code registry (P1.3)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nneditor.diagnostics import (
    DIAGNOSTIC_CODES,
    CodeDescriptor,
    DiagnosticCategory,
    DiagnosticLog,
    Severity,
    describe,
    diagnostics_markdown,
)

SRC = Path(__file__).resolve().parents[2] / "src"


def emitted_codes() -> set[str]:
    """Every diagnostic code string the product source actually emits.

    Matched at the call sites that create findings — ``log.add(...)`` and
    ``Diagnostic(...)`` — so IR node domains like ``"pytorch.aten"``, which
    look like codes but are not, stay out of the comparison.
    """
    pattern = re.compile(
        r"(?:log\.add|DiagnosticLog\(\)\.add|Diagnostic)\(\s*\n?\s*"
        r'"((?:onnx|pytorch|jax|hierarchy|dlc|coreml)\.[a-z-]+)"'
    )
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        found.update(pattern.findall(path.read_text(encoding="utf-8")))
    return found


def test_every_emitted_code_is_registered() -> None:
    unregistered = emitted_codes() - set(DIAGNOSTIC_CODES)
    assert not unregistered, (
        f"codes emitted but not registered: {sorted(unregistered)}; add them "
        "to nneditor.diagnostics._DESCRIPTORS"
    )


def test_every_registered_code_is_emitted_somewhere() -> None:
    dead = set(DIAGNOSTIC_CODES) - emitted_codes()
    assert not dead, f"registered codes never emitted: {sorted(dead)}"


def test_describe_returns_registered_descriptors() -> None:
    descriptor = describe("onnx.external-file-missing")
    assert descriptor.category is DiagnosticCategory.EXTERNAL_DATA
    assert "missing" in descriptor.title.lower()


def test_describe_falls_back_visibly_for_unknown_codes() -> None:
    descriptor = describe("onnx.never-heard-of-it")
    assert descriptor.code == "unregistered"
    assert "bug" in descriptor.guidance


def test_descriptors_are_validated() -> None:
    with pytest.raises(ValueError, match="need"):
        CodeDescriptor("", DiagnosticCategory.SHAPE, "t", "g")
    with pytest.raises(ValueError, match="need"):
        CodeDescriptor("x", DiagnosticCategory.SHAPE, "t", "   ")


def test_every_category_has_at_least_one_code() -> None:
    used = {descriptor.category for descriptor in DIAGNOSTIC_CODES.values()}
    assert used == set(DiagnosticCategory)


def test_markdown_lists_every_code_once() -> None:
    rendered = diagnostics_markdown()
    for code in DIAGNOSTIC_CODES:
        assert rendered.count(f"`{code}`") == 1


def test_log_counts_by_severity() -> None:
    log = DiagnosticLog()
    log.add("onnx.custom-domain", Severity.INFO, "info")
    log.add("onnx.external-file-missing", Severity.ERROR, "gone")
    log.add("onnx.incomplete-io-shape", Severity.INFO, "loose")
    counts = log.counts()
    assert counts[Severity.INFO] == 2
    assert counts[Severity.ERROR] == 1
    assert counts[Severity.WARNING] == 0
