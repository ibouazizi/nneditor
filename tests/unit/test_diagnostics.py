"""Tests for the diagnostic log."""

from __future__ import annotations

from nneditor.diagnostics import Diagnostic, DiagnosticLog, Severity


def test_an_empty_log_is_falsy() -> None:
    log = DiagnosticLog()
    assert not log
    assert len(log) == 0
    assert not log.has_errors


def test_diagnostics_are_recorded_in_order() -> None:
    log = DiagnosticLog()
    log.add("a.first", Severity.INFO, "one")
    log.add("a.second", Severity.WARNING, "two", "target")
    assert log.codes() == ("a.first", "a.second")
    assert len(log) == 2
    assert bool(log)


def test_add_returns_the_recorded_diagnostic() -> None:
    log = DiagnosticLog()
    diagnostic = log.add("a.code", Severity.ERROR, "boom", "node")
    assert diagnostic.target == "node"
    assert log.has_errors


def test_filtering_by_severity() -> None:
    log = DiagnosticLog(
        [
            Diagnostic("a", Severity.INFO, "one"),
            Diagnostic("b", Severity.ERROR, "two"),
            Diagnostic("c", Severity.ERROR, "three"),
        ]
    )
    assert [item.code for item in log.of_severity(Severity.ERROR)] == ["b", "c"]
    assert [item.code for item in log.of_severity(Severity.WARNING)] == []
    assert [item.code for item in log] == ["a", "b", "c"]


def test_diagnostics_render_with_and_without_a_target() -> None:
    assert str(Diagnostic("a.code", Severity.WARNING, "careful")) == (
        "warning: a.code: careful"
    )
    assert str(Diagnostic("a.code", Severity.ERROR, "broken", "n:g:main#name:x")) == (
        "error: a.code [n:g:main#name:x]: broken"
    )
