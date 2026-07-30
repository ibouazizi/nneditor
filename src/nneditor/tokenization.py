"""Text codebooks for generated language-model test inputs.

Tracing a language model needs token ids, not pixels, so the input generator
needs a way to turn text into ids. This module provides that without adding a
tokenizer dependency, the same way the ONNX and safetensors readers avoid
depending on their reference packages.

Two families are available:

* **Self-contained codebooks** — byte, code point, and hashed-word — always
  work with no external files. The hashed-word codebook is the one to reach for
  when a model's real vocabulary is unavailable but its *vocabulary size* is
  known: it produces deterministic, in-range ids of the right shape, which is
  what exercising a graph actually requires.
* **File-backed byte-level BPE** — the GPT-2/RoBERTa scheme — loaded from a
  HuggingFace ``vocab.json`` plus ``merges.txt``, or from a single
  ``tokenizer.json``. Vocabularies are data, never code: files are parsed as
  JSON/text under a size ceiling and nothing is ever unpickled.

The BPE pre-tokenizer is an approximation of GPT-2's, because the exact pattern
needs variable-width Unicode property escapes that :mod:`re` does not support.
Every codebook therefore reports an :attr:`Codebook.fidelity` string, and the
generator records it, so a caller is never left guessing whether ids match
their model's own tokenizer byte for byte.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

__all__ = [
    "MAX_VOCABULARY_BYTES",
    "TOKENIZER_CHOICES",
    "BpeCodebook",
    "ByteCodebook",
    "Codebook",
    "CodepointCodebook",
    "TokenizationError",
    "TokenizerChoice",
    "WordHashCodebook",
    "choice_for",
    "load_codebook",
]

MAX_VOCABULARY_BYTES: Final = 32 * 1024 * 1024
"""Ceiling for a vocabulary or merge file, applied before it is read."""

_MAX_MERGES: Final = 200_000
_MAX_TEXT_CHARACTERS: Final = 1_000_000

# GPT-2's pre-tokenizer without variable-width Unicode property escapes:
# ``[^\W\d_]`` stands in for ``\p{L}`` and ``\d`` for ``\p{N}``. Contractions,
# leading spaces, punctuation runs, and trailing whitespace are all preserved,
# so the split matches GPT-2 on ordinary prose and can differ on unusual
# scripts or symbol runs.
_PRETOKEN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?[^\s\w]+"
    r"|\s+(?!\S)"
    r"|\s+",
    re.UNICODE,
)


class TokenizationError(ValueError):
    """A codebook could not be built, or text could not be encoded."""


@runtime_checkable
class Codebook(Protocol):
    """Turns text into model-ready token ids."""

    @property
    def name(self) -> str:
        """Identifier of the codebook that produced the ids."""

    @property
    def vocab_size(self) -> int:
        """Exclusive upper bound for every id this codebook emits."""

    @property
    def fidelity(self) -> str:
        """How faithful these ids are to the model's own tokenizer."""

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode ``text`` into ids, all inside ``range(vocab_size)``."""


@dataclass(frozen=True, slots=True)
class TokenizerChoice:
    """One selectable entry for the input generator's codebook drop-down."""

    id: str
    label: str
    needs_vocabulary: bool
    needs_merges: bool
    note: str

    @property
    def needs_files(self) -> bool:
        return self.needs_vocabulary or self.needs_merges


TOKENIZER_CHOICES: Final[tuple[TokenizerChoice, ...]] = (
    TokenizerChoice(
        "byte",
        "Byte level (UTF-8)",
        False,
        False,
        "Each UTF-8 byte is one id in 0-255. Exact for byte-level models.",
    ),
    TokenizerChoice(
        "codepoint",
        "Unicode code points",
        False,
        False,
        "Each character is its code point. Exact for character-level models.",
    ),
    TokenizerChoice(
        "word-hash",
        "Hashed words (set vocabulary size)",
        False,
        False,
        "Deterministic in-range ids without a real vocabulary — right shape "
        "and range for smoke-tracing, not the model's own token ids.",
    ),
    TokenizerChoice(
        "gpt2-bpe",
        "GPT-2 / RoBERTa BPE (vocab.json + merges.txt)",
        True,
        True,
        "Byte-level BPE from HuggingFace files, e.g. GPT-2's vocab.json and "
        "merges.txt.",
    ),
    TokenizerChoice(
        "tokenizer-json",
        "HuggingFace tokenizer.json",
        True,
        False,
        "Byte-level BPE read from a single tokenizer.json.",
    ),
)


def choice_for(choice_id: str) -> TokenizerChoice:
    """The drop-down entry registered under ``choice_id``."""
    for choice in TOKENIZER_CHOICES:
        if choice.id == choice_id:
            return choice
    raise TokenizationError(f"unknown tokenizer choice: {choice_id!r}")


def _checked_text(text: str) -> str:
    if len(text) > _MAX_TEXT_CHARACTERS:
        raise TokenizationError(
            f"text is {len(text)} characters, above the "
            f"{_MAX_TEXT_CHARACTERS} character limit"
        )
    return text


@dataclass(frozen=True, slots=True)
class ByteCodebook:
    """UTF-8 bytes as ids."""

    name: str = "byte"
    vocab_size: int = 256
    fidelity: str = "exact for byte-level models: one id per UTF-8 byte"

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(_checked_text(text).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class CodepointCodebook:
    """Unicode code points as ids, folded into ``vocab_size``."""

    vocab_size: int = 0x110000
    name: str = "codepoint"

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise TokenizationError("vocab_size must be positive")

    @property
    def fidelity(self) -> str:
        if self.vocab_size >= 0x110000:
            return "exact for character-level models: one id per code point"
        return (
            "code points folded into the vocabulary by remainder, so ids "
            "above the vocabulary size wrap"
        )

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(char) % self.vocab_size for char in _checked_text(text))


@dataclass(frozen=True, slots=True)
class WordHashCodebook:
    """Deterministic hashed word ids for a known vocabulary size.

    Useful when a model's vocabulary is unavailable: the ids are stable across
    runs and machines (a content hash, not :func:`hash`) and always in range,
    which is what tracing a graph needs. They are *not* the model's own ids, so
    output text is meaningless — the fidelity string says so.
    """

    vocab_size: int = 50257
    reserved_ids: int = 1
    name: str = "word-hash"

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise TokenizationError("vocab_size must be at least two")
        if not 0 <= self.reserved_ids < self.vocab_size:
            raise TokenizationError("reserved_ids must be inside the vocabulary")

    @property
    def fidelity(self) -> str:
        return (
            "not the model's own ids: words are hashed into "
            f"[{self.reserved_ids}, {self.vocab_size}) deterministically, so "
            "shapes and ranges are valid but decoded text is meaningless"
        )

    def encode(self, text: str) -> tuple[int, ...]:
        span = self.vocab_size - self.reserved_ids
        ids: list[int] = []
        for match in _PRETOKEN.finditer(_checked_text(text)):
            piece = match.group().strip()
            if not piece:
                continue
            digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest()
            ids.append(self.reserved_ids + int.from_bytes(digest, "big") % span)
        return tuple(ids)


def _byte_encoder() -> dict[int, str]:
    """GPT-2's reversible byte-to-character map for byte-level BPE."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapped = list(printable)
    spare = 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            mapped.append(256 + spare)
            spare += 1
    return {value: chr(code) for value, code in zip(printable, mapped, strict=True)}


class BpeCodebook:
    """Byte-level byte-pair encoding loaded from vocabulary and merge data."""

    __slots__ = ("_bytes", "_merges", "_name", "_vocab", "fidelity")

    def __init__(
        self,
        vocabulary: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        name: str = "gpt2-bpe",
        fidelity: str | None = None,
    ) -> None:
        if not vocabulary:
            raise TokenizationError("the vocabulary is empty")
        if len(merges) > _MAX_MERGES:
            raise TokenizationError(
                f"{len(merges)} merges exceed the {_MAX_MERGES} merge limit"
            )
        self._vocab = vocabulary
        self._merges = {pair: rank for rank, pair in enumerate(merges)}
        self._bytes = _byte_encoder()
        self._name = name
        self.fidelity = fidelity or (
            "byte-level BPE from the supplied vocabulary; the pre-tokenizer "
            "approximates GPT-2's and can differ on unusual scripts or "
            "symbol runs"
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def vocab_size(self) -> int:
        return max(self._vocab.values()) + 1

    def _merge(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        word = symbols
        while len(word) > 1:
            pairs = set(itertools.pairwise(word))
            ranked = [pair for pair in pairs if pair in self._merges]
            if not ranked:
                break
            first, second = min(ranked, key=self._merges.__getitem__)
            merged: list[str] = []
            index = 0
            while index < len(word):
                if (
                    index < len(word) - 1
                    and word[index] == first
                    and word[index + 1] == second
                ):
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
        return word

    def encode(self, text: str) -> tuple[int, ...]:
        ids: list[int] = []
        for match in _PRETOKEN.finditer(_checked_text(text)):
            piece = match.group()
            if not piece:
                continue
            symbols = tuple(self._bytes[byte] for byte in piece.encode("utf-8"))
            for token in self._merge(symbols):
                identifier = self._vocab.get(token)
                if identifier is None:
                    # A byte-level vocabulary contains every single byte, so a
                    # miss means the files are inconsistent rather than that
                    # the text is unusual. Say which token failed.
                    raise TokenizationError(
                        f"token {token!r} is absent from the vocabulary; the "
                        "vocabulary and merge files may not belong together"
                    )
                ids.append(identifier)
        return tuple(ids)


def _read_bounded(path: Path) -> str:
    resolved = Path(path).expanduser()
    try:
        size = resolved.stat().st_size
    except OSError as error:
        raise TokenizationError(f"cannot read {resolved.name}: {error}") from error
    if not resolved.is_file():
        raise TokenizationError(f"not a readable file: {resolved}")
    if size > MAX_VOCABULARY_BYTES:
        raise TokenizationError(
            f"{resolved.name} is {size} bytes, above the "
            f"{MAX_VOCABULARY_BYTES} byte limit"
        )
    try:
        return resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise TokenizationError(f"cannot read {resolved.name}: {error}") from error


def _vocabulary_from(payload: object, where: str) -> dict[str, int]:
    if not isinstance(payload, dict) or not payload:
        raise TokenizationError(f"{where} does not contain a vocabulary object")
    vocabulary: dict[str, int] = {}
    for token, identifier in payload.items():
        if not isinstance(token, str) or not isinstance(identifier, int):
            raise TokenizationError(
                f"{where} has a non string-to-integer vocabulary entry"
            )
        vocabulary[token] = identifier
    return vocabulary


def _merges_from(entries: object, where: str) -> list[tuple[str, str]]:
    if not isinstance(entries, list):
        raise TokenizationError(f"{where} does not contain a merge list")
    merges: list[tuple[str, str]] = []
    for entry in entries:
        if isinstance(entry, str):
            parts = entry.split()
        elif isinstance(entry, list) and len(entry) == 2:
            parts = [str(item) for item in entry]
        else:
            raise TokenizationError(f"{where} has a malformed merge entry")
        if len(parts) != 2:
            raise TokenizationError(f"{where} has a merge that is not a pair")
        merges.append((parts[0], parts[1]))
    return merges


def _parse_json(text: str, where: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise TokenizationError(f"{where} is not valid JSON: {error}") from error


def load_codebook(
    choice_id: str,
    *,
    vocabulary_path: Path | str | None = None,
    merges_path: Path | str | None = None,
    vocab_size: int | None = None,
) -> Codebook:
    """Build the codebook a drop-down selection names.

    ``vocab_size`` applies to the self-contained codebooks whose range is a
    caller's choice; the file-backed ones take their size from the vocabulary.
    """
    choice = choice_for(choice_id)
    if choice.needs_vocabulary and vocabulary_path is None:
        raise TokenizationError(f"{choice.label} needs a vocabulary file")
    if choice.needs_merges and merges_path is None:
        raise TokenizationError(f"{choice.label} needs a merges file")

    if choice.id == "byte":
        return ByteCodebook()
    if choice.id == "codepoint":
        return (
            CodepointCodebook()
            if vocab_size is None
            else CodepointCodebook(vocab_size=vocab_size)
        )
    if choice.id == "word-hash":
        return (
            WordHashCodebook()
            if vocab_size is None
            else WordHashCodebook(vocab_size=vocab_size)
        )
    if choice.id == "gpt2-bpe":
        assert vocabulary_path is not None and merges_path is not None
        vocabulary = _vocabulary_from(
            _parse_json(_read_bounded(Path(vocabulary_path)), "vocab.json"),
            "vocab.json",
        )
        lines = [
            line
            for line in _read_bounded(Path(merges_path)).splitlines()
            if line.strip() and not line.startswith("#version")
        ]
        return BpeCodebook(vocabulary, _merges_from(lines, "merges.txt"))
    if choice.id == "tokenizer-json":
        assert vocabulary_path is not None
        payload = _parse_json(_read_bounded(Path(vocabulary_path)), "tokenizer.json")
        if not isinstance(payload, dict):
            raise TokenizationError("tokenizer.json is not an object")
        model = payload.get("model")
        if not isinstance(model, dict):
            raise TokenizationError("tokenizer.json has no model object")
        if str(model.get("type", "BPE")) != "BPE":
            raise TokenizationError(
                f"tokenizer.json model type {model.get('type')!r} is not "
                "supported; only byte-level BPE is implemented"
            )
        return BpeCodebook(
            _vocabulary_from(model.get("vocab"), "tokenizer.json"),
            _merges_from(model.get("merges"), "tokenizer.json"),
            name="tokenizer-json",
        )
    raise TokenizationError(f"unsupported tokenizer choice: {choice_id!r}")
