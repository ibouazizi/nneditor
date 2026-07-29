"""Tests for IR schema versioning and migration planning (P0.3)."""

from __future__ import annotations

import pytest

from nneditor.ir.schema import (
    IR_SCHEMA_VERSION,
    UNKNOWN_FIELD_NAMESPACE,
    ReadStrategy,
    SchemaVersion,
    SchemaVersionError,
    plan_read,
    split_extensions,
    validate_extension_namespace,
)


def test_current_version_is_one_one() -> None:
    # 1.1 added TensorRef.typed_span (P3.2).
    assert str(IR_SCHEMA_VERSION) == "1.1"


@pytest.mark.parametrize(
    ("text", "expected"), [("1.0", (1, 0)), ("2.13", (2, 13)), (" 3.4 ", (3, 4))]
)
def test_parse_accepts_well_formed_versions(
    text: str, expected: tuple[int, int]
) -> None:
    version = SchemaVersion.parse(text)
    assert (version.major, version.minor) == expected


@pytest.mark.parametrize("text", ["1", "1.0.0", "01.2", "1.02", "v1.0", "", "a.b"])
def test_parse_rejects_malformed_versions(text: str) -> None:
    with pytest.raises(SchemaVersionError, match="malformed schema version"):
        SchemaVersion.parse(text)


@pytest.mark.parametrize(("major", "minor"), [(0, 1), (-1, 0), (1, -1)])
def test_construction_rejects_out_of_range_components(major: int, minor: int) -> None:
    with pytest.raises(SchemaVersionError):
        SchemaVersion(major, minor)


def test_versions_order_by_major_then_minor() -> None:
    assert SchemaVersion(1, 9) < SchemaVersion(2, 0)
    assert SchemaVersion(1, 2) < SchemaVersion(1, 10)


def test_same_version_reads_directly() -> None:
    plan = plan_read(SchemaVersion(1, 0), SchemaVersion(1, 0))
    assert plan.strategy is ReadStrategy.DIRECT
    assert plan.is_readable


def test_older_minor_reads_directly() -> None:
    assert plan_read(SchemaVersion(1, 2), SchemaVersion(1, 7)).strategy is (
        ReadStrategy.DIRECT
    )


def test_newer_minor_reads_in_degraded_mode() -> None:
    plan = plan_read(SchemaVersion(1, 9), SchemaVersion(1, 3))
    assert plan.strategy is ReadStrategy.DEGRADED
    assert plan.is_readable
    assert UNKNOWN_FIELD_NAMESPACE in plan.reason


def test_newer_major_is_unsupported() -> None:
    plan = plan_read(SchemaVersion(3, 0), SchemaVersion(2, 4))
    assert plan.strategy is ReadStrategy.UNSUPPORTED
    assert not plan.is_readable
    assert "never inferred" in plan.reason


def test_older_major_migrates_through_every_registered_step() -> None:
    plan = plan_read(SchemaVersion(1, 4), SchemaVersion(3, 0), {1, 2})
    assert plan.strategy is ReadStrategy.MIGRATE
    assert plan.migration_path == (1, 2)


def test_older_major_without_a_migration_is_unsupported() -> None:
    plan = plan_read(SchemaVersion(1, 0), SchemaVersion(3, 0), {1})
    assert plan.strategy is ReadStrategy.UNSUPPORTED
    assert "2 to 3" in plan.reason


def test_plan_read_defaults_to_the_current_reader() -> None:
    assert plan_read(IR_SCHEMA_VERSION).strategy is ReadStrategy.DIRECT


@pytest.mark.parametrize(
    "namespace",
    ["x-onnx.metadata_props", "x-torch.node_meta", "x-nneditor.unknown", "x-a.b"],
)
def test_valid_extension_namespaces(namespace: str) -> None:
    assert validate_extension_namespace(namespace) == namespace


@pytest.mark.parametrize(
    "namespace", ["onnx.meta", "x-ONNX.meta", "x-onnx", "x-.meta", "x-onnx.", "xonnx.a"]
)
def test_invalid_extension_namespaces(namespace: str) -> None:
    with pytest.raises(SchemaVersionError, match="extension namespace"):
        validate_extension_namespace(namespace)


def test_split_extensions_separates_core_and_namespaced_fields() -> None:
    core, extensions = split_extensions(
        {"op_type": "Conv", "x-onnx.doc_string": "hi", "domain": ""}
    )
    assert core == {"op_type": "Conv", "domain": ""}
    assert extensions == {"x-onnx.doc_string": "hi"}


def test_split_extensions_rejects_a_malformed_namespace() -> None:
    with pytest.raises(SchemaVersionError):
        split_extensions({"x-Bad": 1})
