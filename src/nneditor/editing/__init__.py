"""Transactional, revision-based editing.

Architectural rule 7 makes every edit a reversible command applied to a
working revision, and rule 1 keeps source artifacts immutable. This package
holds the format-agnostic parts: commands, deltas, and undo/redo. Writing an
edited artifact back out is format-specific and lives with each adapter.

Phase 3 supplies compact persistent tensor deltas and Phase 4 adds reversible
graph commands plus a schema-aware preparation pipeline. Phase 5 adds
manifested, previewable quantization and pruning commands. A failed or
rejected transaction never creates a revision.
"""
