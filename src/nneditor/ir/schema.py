"""IR schema versioning and migration rules (task P0.3).

The IR is a versioned document format, not an incidental in-memory shape. Every
serialized document, sidecar view document, and cached layout records the schema
version that produced it, and every reader decides what to do with that version
through :func:`plan_read` rather than by guessing.

Policy
------

* ``major`` changes when a document can no longer be read field-for-field: a
  field is removed, retyped, or given different meaning. Reading requires a
  registered migration.
* ``minor`` changes when fields are added and older documents stay readable.
  A reader that meets a *newer* minor version reads it in degraded mode: known
  fields are used, unknown fields are preserved verbatim under the
  ``x-nneditor.unknown`` extension so a later write does not silently drop them.
* Extension namespaces carry source-specific data that the core IR does not
  interpret. They are never migrated, only carried.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Set
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

__all__ = [
    "EXTENSION_NAMESPACE_PATTERN",
    "IR_SCHEMA_VERSION",
    "UNKNOWN_FIELD_NAMESPACE",
    "ReadPlan",
    "ReadStrategy",
    "SchemaVersion",
    "SchemaVersionError",
    "plan_read",
    "split_extensions",
    "validate_extension_namespace",
]


class SchemaVersionError(ValueError):
    """Raised when a schema version cannot be parsed or is not usable."""


@dataclass(frozen=True, slots=True, order=True)
class SchemaVersion:
    """A ``major.minor`` IR document version."""

    major: int
    minor: int

    def __post_init__(self) -> None:
        if self.major < 1:
            raise SchemaVersionError(f"major version must be >= 1, got {self.major}")
        if self.minor < 0:
            raise SchemaVersionError(f"minor version must be >= 0, got {self.minor}")

    @classmethod
    def parse(cls, text: str) -> SchemaVersion:
        """Parse ``"1.4"`` into a :class:`SchemaVersion`."""
        match = re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", text.strip())
        if match is None:
            raise SchemaVersionError(f"malformed schema version: {text!r}")
        return cls(int(match.group(1)), int(match.group(2)))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


IR_SCHEMA_VERSION: Final = SchemaVersion(1, 1)
"""The version this build writes.

1.1 added ``TensorRef.typed_span`` (task P3.2); 1.0 documents read directly
and simply lack the field, so typed tensors from them stay metadata-only.
"""

UNKNOWN_FIELD_NAMESPACE: Final = "x-nneditor.unknown"
"""Where a degraded read parks fields it did not recognize."""

EXTENSION_NAMESPACE_PATTERN: Final = re.compile(
    r"x-[a-z0-9]+(?:[-_][a-z0-9]+)*\.[a-z0-9]+(?:[-_.][a-z0-9]+)*"
)
"""``x-<vendor>.<name>``; core IR fields never start with ``x-``."""


class ReadStrategy(StrEnum):
    """What a reader must do with a document of a given version."""

    DIRECT = "direct"
    DEGRADED = "degraded"
    MIGRATE = "migrate"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ReadPlan:
    """The decision :func:`plan_read` reached, with its justification."""

    strategy: ReadStrategy
    document_version: SchemaVersion
    reader_version: SchemaVersion
    reason: str
    migration_path: tuple[int, ...] = ()

    @property
    def is_readable(self) -> bool:
        """Whether the document can be opened at all."""
        return self.strategy is not ReadStrategy.UNSUPPORTED


def plan_read(
    document_version: SchemaVersion,
    reader_version: SchemaVersion = IR_SCHEMA_VERSION,
    available_migrations: Set[int] = frozenset(),
) -> ReadPlan:
    """Decide how ``reader_version`` should read ``document_version``.

    ``available_migrations`` holds the source major versions for which an
    upgrade to ``source + 1`` is registered. A chain is followed only when every
    intermediate step exists.
    """
    if document_version.major == reader_version.major:
        if document_version.minor <= reader_version.minor:
            return ReadPlan(
                ReadStrategy.DIRECT,
                document_version,
                reader_version,
                "Same major version and no newer fields are present.",
            )
        return ReadPlan(
            ReadStrategy.DEGRADED,
            document_version,
            reader_version,
            f"Document minor version {document_version.minor} is newer than the "
            f"reader's {reader_version.minor}; unknown fields are preserved under "
            f"{UNKNOWN_FIELD_NAMESPACE!r}.",
        )
    if document_version.major > reader_version.major:
        return ReadPlan(
            ReadStrategy.UNSUPPORTED,
            document_version,
            reader_version,
            f"Document major version {document_version.major} is newer than the "
            f"reader's {reader_version.major}; downgrading is never inferred.",
        )
    path = tuple(range(document_version.major, reader_version.major))
    missing = [step for step in path if step not in available_migrations]
    if missing:
        steps = ", ".join(f"{step} to {step + 1}" for step in missing)
        return ReadPlan(
            ReadStrategy.UNSUPPORTED,
            document_version,
            reader_version,
            f"No registered migration for major version step(s): {steps}.",
        )
    return ReadPlan(
        ReadStrategy.MIGRATE,
        document_version,
        reader_version,
        "A complete migration chain upgrades the document before reading.",
        migration_path=path,
    )


def validate_extension_namespace(namespace: str) -> str:
    """Return ``namespace`` if it is a legal extension namespace."""
    if EXTENSION_NAMESPACE_PATTERN.fullmatch(namespace) is None:
        raise SchemaVersionError(
            f"extension namespace {namespace!r} must match "
            f"{EXTENSION_NAMESPACE_PATTERN.pattern!r}"
        )
    return namespace


def split_extensions(
    fields: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Split a raw field mapping into core fields and extension fields."""
    core: dict[str, object] = {}
    extensions: dict[str, object] = {}
    for key, value in fields.items():
        if key.startswith("x-"):
            extensions[validate_extension_namespace(key)] = value
        else:
            core[key] = value
    return core, extensions
