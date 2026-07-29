"""Tests for the instrumented, bounds-checked artifact reader (P0.4)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nneditor.storage.reader import (
    MAX_SINGLE_READ,
    ArtifactReader,
    ByteRange,
    RangeReadError,
    ReadStats,
    hash_file,
)

PAYLOAD = bytes(range(256)) * 64  # 16 KiB of predictable bytes


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    path = tmp_path / "artifact.bin"
    path.write_bytes(PAYLOAD)
    return path


def test_reads_return_the_requested_bytes(artifact: Path) -> None:
    with ArtifactReader(artifact) as reader:
        assert reader.size == len(PAYLOAD)
        assert reader.read(0, 4) == PAYLOAD[:4]
        assert reader.read(1000, 32) == PAYLOAD[1000:1032]
        assert reader.read(len(PAYLOAD) - 1, 1) == PAYLOAD[-1:]


def test_a_read_spanning_blocks_is_reassembled(artifact: Path) -> None:
    with ArtifactReader(artifact, block_size=64) as reader:
        assert reader.read(50, 200) == PAYLOAD[50:250]


def test_zero_length_reads_are_allowed_and_recorded(artifact: Path) -> None:
    with ArtifactReader(artifact) as reader:
        assert reader.read(10, 0) == b""
        assert reader.stats.logical_ranges == [ByteRange(10, 0)]
        assert reader.stats.physical_bytes == 0


@pytest.mark.parametrize(
    ("offset", "length"),
    [(-1, 4), (0, -1), (len(PAYLOAD), 1), (len(PAYLOAD) - 2, 4), (0, len(PAYLOAD) + 1)],
)
def test_out_of_bounds_ranges_are_rejected(
    artifact: Path, offset: int, length: int
) -> None:
    with ArtifactReader(artifact) as reader, pytest.raises(RangeReadError):
        reader.read(offset, length)


def test_an_absurd_length_is_refused_before_allocation(artifact: Path) -> None:
    with ArtifactReader(artifact) as reader:
        with pytest.raises(RangeReadError, match="refusing"):
            reader.read(0, MAX_SINGLE_READ + 1)


@pytest.mark.parametrize(("offset", "length"), [(True, 4), (0, False), ("0", 4)])
def test_non_integer_ranges_are_rejected(
    artifact: Path, offset: object, length: object
) -> None:
    with ArtifactReader(artifact) as reader, pytest.raises(RangeReadError):
        reader.read(offset, length)  # type: ignore[arg-type]


def test_validate_does_not_read_anything(artifact: Path) -> None:
    with ArtifactReader(artifact) as reader:
        assert reader.validate(4, 8) == ByteRange(4, 8)
        assert reader.stats.logical_ranges == []
        assert reader.stats.physical_ranges == []


def test_reading_after_close_is_refused(artifact: Path) -> None:
    reader = ArtifactReader(artifact)
    reader.close()
    assert reader.closed
    reader.close()  # idempotent
    with pytest.raises(RangeReadError, match="closed"):
        reader.read(0, 1)


def test_blocks_are_cached_so_repeated_reads_hit_the_file_once(
    artifact: Path,
) -> None:
    with ArtifactReader(artifact, block_size=4096) as reader:
        for _ in range(5):
            reader.read(10, 20)
        assert len(reader.stats.physical_ranges) == 1
        assert len(reader.stats.logical_ranges) == 5


def test_the_block_cache_is_bounded(artifact: Path) -> None:
    with ArtifactReader(artifact, block_size=512, max_cached_blocks=2) as reader:
        for block in range(8):
            reader.read(block * 512, 16)
        for block in range(8):
            reader.read(block * 512, 16)
        # Evicted blocks are fetched again rather than growing the cache.
        assert len(reader.stats.physical_ranges) > 8


@pytest.mark.parametrize(
    ("block_size", "max_blocks"), [(0, 4), (-1, 4), (64, 0), (64, -1)]
)
def test_reader_construction_validates_its_own_settings(
    artifact: Path, block_size: int, max_blocks: int
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ArtifactReader(artifact, block_size=block_size, max_cached_blocks=max_blocks)


def test_byte_range_rejects_negative_components() -> None:
    with pytest.raises(RangeReadError, match="non-negative"):
        ByteRange(-1, 4)


def test_byte_range_overlap_and_containment() -> None:
    span = ByteRange(10, 10)
    assert span.end == 20
    assert span.overlaps(ByteRange(19, 5))
    assert not span.overlaps(ByteRange(20, 5))
    assert not span.overlaps(ByteRange(5, 5))
    assert span.contains(ByteRange(12, 4))
    assert not span.contains(ByteRange(12, 20))


def test_stats_summarize_and_reset() -> None:
    stats = ReadStats()
    stats.logical_ranges.extend([ByteRange(0, 4), ByteRange(4, 6), ByteRange(20, 2)])
    stats.physical_ranges.append(ByteRange(0, 64))
    assert stats.logical_bytes == 12
    assert stats.physical_bytes == 64
    assert stats.coalesced_logical() == (ByteRange(0, 10), ByteRange(20, 2))
    assert stats.coalesced_physical() == (ByteRange(0, 64),)
    assert stats.touched_logically(ByteRange(3, 1))
    assert not stats.touched_logically(ByteRange(10, 10))
    stats.reset()
    assert stats.logical_bytes == 0


def test_coalescing_drops_empty_ranges() -> None:
    stats = ReadStats()
    stats.logical_ranges.extend([ByteRange(5, 0), ByteRange(0, 2)])
    assert stats.coalesced_logical() == (ByteRange(0, 2),)


def test_hash_file_matches_hashlib(artifact: Path) -> None:
    expected = hashlib.sha256(PAYLOAD).hexdigest()
    assert hash_file(artifact) == f"sha256:{expected}"


def test_hash_file_supports_other_algorithms(artifact: Path) -> None:
    digest = hash_file(artifact, algorithm="blake2b")
    assert digest.startswith("blake2b:")
