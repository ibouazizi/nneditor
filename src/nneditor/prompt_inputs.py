"""Prompt-text catalogs and model-bound token-id generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nneditor.input_generation import GeneratedTensor, generate_token_ids_tensor
from nneditor.ir.capabilities import ArtifactKind
from nneditor.ir.core import Document, JsonValue
from nneditor.tokenization import (
    Codebook,
    TokenizationError,
    WordHashCodebook,
    load_codebook,
)

__all__ = [
    "DEFAULT_PROMPT",
    "SAMPLE_PROMPTS",
    "PromptSettings",
    "SamplePrompt",
    "codebook_for_document",
    "generate_prompt_token_tensor",
    "prompt_by_id",
    "prompt_settings_for_document",
]


@dataclass(frozen=True, slots=True)
class SamplePrompt:
    """One catalog entry the input generator can insert as prompt text."""

    id: str
    title: str
    text: str


SAMPLE_PROMPTS: tuple[SamplePrompt, ...] = (
    SamplePrompt(
        "pangram",
        "Pangram",
        "The quick brown fox jumps over the lazy dog.",
    ),
    SamplePrompt(
        "photosynthesis",
        "Photosynthesis",
        "Write a one-sentence summary of photosynthesis.",
    ),
    SamplePrompt(
        "primes",
        "Primes",
        "List three prime numbers greater than twenty.",
    ),
    SamplePrompt(
        "village",
        "Village story",
        "In a quiet village at dawn, a baker lights the oven.",
    ),
    SamplePrompt(
        "attention",
        "Attention",
        "Explain attention in one paragraph.",
    ),
)
DEFAULT_PROMPT = SAMPLE_PROMPTS[0].text


@dataclass(frozen=True, slots=True)
class PromptSettings:
    """Tokenizer and special-token defaults inferred from an opened model."""

    text: str
    tokenizer_choice: str
    vocab_size: int | None = None
    bos_id: int | None = None
    eos_id: int | None = None
    pad_id: int = 0
    sequence_length: int | None = None
    vocabulary_path: str | None = None
    merges_path: str | None = None
    note: str = ""


def prompt_by_id(prompt_id: str) -> SamplePrompt:
    """Return the catalog entry registered under ``prompt_id``."""
    for prompt in SAMPLE_PROMPTS:
        if prompt.id == prompt_id:
            return prompt
    raise KeyError(prompt_id)


def _extension_map(document: Document, key: str) -> dict[str, JsonValue]:
    value = dict(document.extensions).get(key)
    return dict(value) if isinstance(value, dict) else {}


def _as_int(value: JsonValue | None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return None
    return value


def _document_vocab_size(document: Document) -> int | None:
    if document.artifact_kind is ArtifactKind.GGUF_MODEL:
        model = _extension_map(document, "x-gguf.model")
        return _as_int(model.get("vocab_size"))
    if document.artifact_kind is ArtifactKind.HUGGINGFACE_MODEL:
        model = _extension_map(document, "x-huggingface.model")
        config = _extension_map(document, "x-huggingface.config")
        return _as_int(config.get("vocab_size")) or _as_int(model.get("vocab_size"))
    return None


def _word_hash(document: Document) -> WordHashCodebook:
    vocab = _document_vocab_size(document) or 50257
    return WordHashCodebook(vocab_size=max(vocab, 2))


def _hf_tokenizer_path(document: Document) -> Path | None:
    if document.artifact_kind is not ArtifactKind.HUGGINGFACE_MODEL:
        return None
    path = Path(document.source.path) / "tokenizer.json"
    return path if path.is_file() else None


def codebook_for_document(document: Document) -> Codebook:
    """Bind the snapshot tokenizer when it is a usable BPE, else hashed words.

    Hugging Face ``tokenizer.json`` files that are byte-level BPE with a
    complete vocabulary are used as-is. Tiny or incomplete vocabularies, GGUF
    SentencePiece tables, and missing tokenizer files fall back to
    :class:`WordHashCodebook` sized to the model's vocabulary.
    """
    tokenizer_json = _hf_tokenizer_path(document)
    if tokenizer_json is not None:
        try:
            codebook = load_codebook("tokenizer-json", vocabulary_path=tokenizer_json)
        except TokenizationError:
            pass
        else:
            vocab = _document_vocab_size(document)
            if codebook.vocab_size >= 256 and (
                vocab is None or codebook.vocab_size <= vocab
            ):
                return codebook
    return _word_hash(document)


def _special_ids(document: Document) -> tuple[int | None, int | None, int]:
    if document.artifact_kind is ArtifactKind.GGUF_MODEL:
        metadata = _extension_map(document, "x-gguf.metadata")
        bos = _as_int(metadata.get("tokenizer.ggml.bos_token_id"))
        eos = _as_int(metadata.get("tokenizer.ggml.eos_token_id"))
        pad = _as_int(metadata.get("tokenizer.ggml.padding_token_id")) or 0
        return bos, eos, pad
    if document.artifact_kind is ArtifactKind.HUGGINGFACE_MODEL:
        config = _extension_map(document, "x-huggingface.config")
        bos = _as_int(config.get("bos_token_id"))
        eos = _as_int(config.get("eos_token_id"))
        pad = _as_int(config.get("pad_token_id")) or 0
        return bos, eos, pad
    return None, None, 0


def prompt_settings_for_document(document: Document) -> PromptSettings:
    """Tokenizer drop-down defaults for a GGUF file or Hugging Face snapshot."""
    codebook = codebook_for_document(document)
    bos, eos, pad = _special_ids(document)
    tokenizer_json = _hf_tokenizer_path(document)
    if getattr(codebook, "name", "") == "tokenizer-json" and tokenizer_json is not None:
        return PromptSettings(
            text=DEFAULT_PROMPT,
            tokenizer_choice="tokenizer-json",
            vocab_size=codebook.vocab_size,
            bos_id=bos,
            eos_id=eos,
            pad_id=pad,
            vocabulary_path=str(tokenizer_json),
            note=codebook.fidelity,
        )
    return PromptSettings(
        text=DEFAULT_PROMPT,
        tokenizer_choice="word-hash",
        vocab_size=codebook.vocab_size,
        bos_id=bos,
        eos_id=eos,
        pad_id=pad,
        note=codebook.fidelity,
    )


def generate_prompt_token_tensor(
    destination: Path | str,
    *,
    document: Document,
    text: str | None = None,
    prompt_id: str | None = None,
    sequence_length: int | None = None,
) -> GeneratedTensor:
    """Encode a catalog or custom prompt with the model's bound codebook."""
    if prompt_id is not None:
        text = prompt_by_id(prompt_id).text
    settings = prompt_settings_for_document(document)
    return generate_token_ids_tensor(
        destination,
        text=text or settings.text,
        codebook=codebook_for_document(document),
        sequence_length=sequence_length,
        dtype="int64",
        pad_id=settings.pad_id,
        bos_id=settings.bos_id,
        eos_id=settings.eos_id,
    )
