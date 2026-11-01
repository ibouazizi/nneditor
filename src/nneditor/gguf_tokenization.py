"""Text encoding from a GGUF's own vocabulary, merges, and tokenizer rules."""

from __future__ import annotations

from pathlib import Path

from nneditor.adapters.gguf.to_ir import open_gguf
from nneditor.detokenization import DetokenizationError, _gguf_array, _map
from nneditor.ir.core import Document
from nneditor.tokenization import TokenizationError, _byte_encoder, _checked_text

# Match the named pre-tokenizers in llama.cpp/src/llama-vocab.cpp. These
# expressions split text; all token IDs and merge ranks come from this GGUF.
_CONTRACTIONS = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
_QWEN_SUFFIX = r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
_PATTERNS = {
    "gpt-2": (
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+"
        r"| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    ),
    "qwen2": _CONTRACTIONS + r"|[^\r\n\p{L}\p{N}]?\p{L}+" + _QWEN_SUFFIX,
    "qwen35": _CONTRACTIONS
    + r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    + r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "llama3": _CONTRACTIONS
    + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    + r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}
_PATTERNS["llama-bpe"] = _PATTERNS["llama3"]
_PATTERNS["llama-v3"] = _PATTERNS["llama3"]


class GgufCodebook:
    """A bounded byte-BPE tokenizer reconstructed entirely from model data."""

    name = "gguf-embedded"

    def __init__(self, source: Document | Path | str) -> None:
        from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers

        document = source if isinstance(source, Document) else open_gguf(Path(source))
        metadata = _map(document, "x-gguf.metadata")
        model = metadata.get("tokenizer.ggml.model")
        pre = metadata.get("tokenizer.ggml.pre")
        if model != "gpt2" or not isinstance(pre, str) or pre not in _PATTERNS:
            raise TokenizationError(
                f"GGUF tokenizer {model!r}, pre-tokenizer {pre!r}, cannot yet "
                "be reconstructed for encoding. Use the model's tokenizer.json "
                "or explicitly select a smoke-test codebook."
            )
        try:
            pieces = _gguf_array(document, "tokenizer.ggml.tokens")
            raw_merges = _gguf_array(document, "tokenizer.ggml.merges")
            types = _gguf_array(document, "tokenizer.ggml.token_type")
        except DetokenizationError as error:
            raise TokenizationError(str(error)) from error
        if not pieces or any(not isinstance(piece, str) for piece in pieces):
            raise TokenizationError("This GGUF has no usable embedded vocabulary.")
        vocabulary = {str(piece): index for index, piece in enumerate(pieces)}
        if len(vocabulary) != len(pieces):
            raise TokenizationError("The GGUF vocabulary contains duplicate pieces.")
        if not set(_byte_encoder().values()) <= vocabulary.keys():
            raise TokenizationError("The GGUF has no complete byte-level vocabulary.")
        if not raw_merges:
            raise TokenizationError("This GGUF has no embedded BPE merge table.")
        merges: list[tuple[str, str]] = []
        for merge in raw_merges:
            if not isinstance(merge, str) or " " not in merge:
                raise TokenizationError("The GGUF BPE merge table is malformed.")
            left, right = merge.split(" ", 1)
            merges.append((left, right))
        try:
            tokenizer = Tokenizer(
                models.BPE(
                    vocab=vocabulary,
                    merges=merges,
                    ignore_merges=pre in {"llama3", "llama-bpe", "llama-v3"},
                )
            )
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Split(Regex(_PATTERNS[str(pre)]), "isolated"),
                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
                ]
            )
            special_ids = {
                int(value)
                for key, value in metadata.items()
                if key.startswith("tokenizer.")
                and key.endswith("_token_id")
                and isinstance(value, int)
                and not isinstance(value, bool)
            }
            special_ids.update(i for i, kind in enumerate(types) if kind in (3, 4))
            tokenizer.add_special_tokens(
                [
                    AddedToken(str(pieces[index]), normalized=False, special=True)
                    for index in sorted(special_ids)
                    if 0 <= index < len(pieces)
                ]
            )
        except Exception as error:
            raise TokenizationError(
                f"Cannot build the GGUF tokenizer: {error}"
            ) from error
        self._tokenizer = tokenizer
        self.vocab_size = len(pieces)
        self.fidelity = (
            f"GGUF embedded vocabulary from {Path(document.source.path).name}: "
            f"{self.vocab_size:,} tokens, {len(merges):,} merges, {pre} rules. "
            "Token IDs come from this model; no external GPT-2 vocabulary is used."
        )

    def encode(self, text: str) -> tuple[int, ...]:
        try:
            ids = tuple(
                self._tokenizer.encode(
                    _checked_text(text), add_special_tokens=False
                ).ids
            )
        except Exception as error:
            raise TokenizationError(
                f"Cannot encode text with the GGUF vocabulary: {error}"
            ) from error
        if not ids and text:
            raise TokenizationError("The GGUF tokenizer produced no token IDs.")
        return ids
