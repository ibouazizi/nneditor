"""Model-bound decoding of token IDs and bounded next-token predictions."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from nneditor.adapters.gguf.preview import preview_kv_array
from nneditor.ir.capabilities import ArtifactKind
from nneditor.ir.core import Document, JsonValue
from nneditor.tokenization import MAX_VOCABULARY_BYTES, _byte_encoder
from nneditor.tracing.contracts import ActivationRecord

MAX_DECODE_TOKENS = 4096
MAX_LOGIT_BYTES = 32 * 1024 * 1024


class DetokenizationError(ValueError):
    """The model has no usable decoder or the requested tokens are invalid."""


class Decoder(Protocol):
    @property
    def name(self) -> str: ...

    def piece(self, token_id: int) -> str: ...

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str: ...


def _checked_ids(ids: Iterable[int]) -> tuple[int, ...]:
    result: list[int] = []
    for value in ids:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DetokenizationError("Token IDs must be nonnegative whole numbers.")
        result.append(value)
        if len(result) > MAX_DECODE_TOKENS:
            raise DetokenizationError(
                f"Decode at most {MAX_DECODE_TOKENS} tokens at once."
            )
    return tuple(result)


class JsonDecoder:
    name = "Hugging Face tokenizer.json"

    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer

        if path.stat().st_size > MAX_VOCABULARY_BYTES:
            raise DetokenizationError("Tokenizer exceeds the 32 MiB vocabulary limit.")
        try:
            self._tokenizer = Tokenizer.from_str(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise DetokenizationError(f"Cannot load tokenizer.json: {error}") from error

    def piece(self, token_id: int) -> str:
        piece = self._tokenizer.id_to_token(token_id)
        if piece is None:
            raise DetokenizationError(
                f"Token ID {token_id} is outside this vocabulary."
            )
        return piece

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        values = _checked_ids(ids)
        for value in values:
            self.piece(value)
        return self._tokenizer.decode(list(values), skip_special_tokens=skip_special)


@dataclass(frozen=True, slots=True)
class VocabularyDecoder:
    vocabulary: tuple[str, ...]
    algorithm: str
    special_ids: frozenset[int] = frozenset()
    source_name: str = ""

    @property
    def name(self) -> str:
        source = "GGUF embedded vocabulary"
        if self.source_name:
            source += f": {self.source_name}"
        return f"{source} ({len(self.vocabulary):,} tokens; {self.algorithm} decoding)"

    def piece(self, token_id: int) -> str:
        if token_id < 0 or token_id >= len(self.vocabulary):
            raise DetokenizationError(
                f"Token ID {token_id} is outside this vocabulary."
            )
        return self.vocabulary[token_id]

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        values = _checked_ids(ids)
        for value in values:
            self.piece(value)
        pieces = [
            self.piece(i)
            for i in values
            if not skip_special or i not in self.special_ids
        ]
        if self.algorithm == "bert":
            from tokenizers.decoders import WordPiece

            return str(WordPiece().decode(pieces))
        raw = bytearray()
        inverse = {char: byte for byte, char in _byte_encoder().items()}
        for piece in pieces:
            fallback = re.fullmatch(r"<0x([0-9a-fA-F]{2})>", piece)
            if fallback:
                raw.append(int(fallback[1], 16))
            elif self.algorithm == "gpt2" and all(char in inverse for char in piece):
                raw.extend(inverse[char] for char in piece)
            else:
                raw.extend(
                    (
                        piece.replace("▁", " ")
                        if self.algorithm in {"llama", "t5"}
                        else piece
                    ).encode("utf-8")
                )
        text = raw.decode("utf-8", errors="replace")
        if self.algorithm in {"llama", "t5"} and pieces and pieces[0].startswith("▁"):
            text = text[1:]
        return text


def _map(document: Document, key: str) -> dict[str, JsonValue]:
    value = dict(document.extensions).get(key)
    return value if isinstance(value, dict) else {}


def _gguf_array(document: Document, key: str) -> tuple[JsonValue, ...]:
    inline = _map(document, "x-gguf.metadata").get(key)
    if isinstance(inline, list):
        return tuple(inline)
    if not isinstance(inline, dict) or inline.get("type") != "array":
        return ()
    model = _map(document, "x-gguf.model")
    offset, length = inline.get("offset"), inline.get("length")
    if not isinstance(offset, int) or not isinstance(length, int):
        raise DetokenizationError("GGUF vocabulary has no valid recorded byte range.")
    if length > MAX_VOCABULARY_BYTES:
        raise DetokenizationError("Tokenizer exceeds the 32 MiB vocabulary limit.")
    version = model.get("version", 3)
    values, count = preview_kv_array(
        document.source.path,
        offset=offset,
        length=length,
        version=version if isinstance(version, int) else 3,
        endian=str(model.get("endian", "little")),
        limit=1_000_000,
    )
    if count > len(values):
        raise DetokenizationError("Tokenizer exceeds the one-million-token limit.")
    return values


def decoder_for_document(document: Document) -> Decoder:
    if document.artifact_kind is ArtifactKind.HUGGINGFACE_MODEL:
        path = Path(document.source.path) / "tokenizer.json"
        if path.is_file():
            return JsonDecoder(path)
        raise DetokenizationError(
            "This snapshot has no tokenizer.json. "
            "Add the model's tokenizer to decode token IDs."
        )
    if document.artifact_kind is ArtifactKind.GGUF_MODEL:
        metadata = _map(document, "x-gguf.metadata")
        name = metadata.get("tokenizer.ggml.model")
        if not isinstance(name, str) or name not in {"gpt2", "llama", "t5", "bert"}:
            raise DetokenizationError(
                f"GGUF tokenizer {name!r} is not supported for decoding."
            )
        vocabulary = _gguf_array(document, "tokenizer.ggml.tokens")
        if not vocabulary or not all(isinstance(v, str) for v in vocabulary):
            raise DetokenizationError("This GGUF has no usable token vocabulary.")
        special = {
            int(value)
            for key, value in metadata.items()
            if key.startswith("tokenizer.")
            and key.endswith("_token_id")
            and isinstance(value, int)
            and not isinstance(value, bool)
        }
        special.update(
            i
            for i, kind in enumerate(_gguf_array(document, "tokenizer.ggml.token_type"))
            if kind in (3, 5)
        )
        return VocabularyDecoder(
            tuple(str(v) for v in vocabulary),
            str(name),
            frozenset(special),
            Path(document.source.path).name,
        )
    raise DetokenizationError(
        "Text decoding needs the model's GGUF or Hugging Face tokenizer."
    )


@dataclass(frozen=True, slots=True)
class TokenCandidate:
    token_id: int
    piece: str
    probability: float


@dataclass(frozen=True, slots=True)
class DecodedActivation:
    text: str
    ids: tuple[int, ...]
    pieces: tuple[str, ...]
    decoder: str
    note: str
    candidates: tuple[TokenCandidate, ...] = ()


def text_activation_kind(document: Document, record: ActivationRecord) -> str | None:
    if document.artifact_kind not in {
        ArtifactKind.GGUF_MODEL,
        ArtifactKind.HUGGINGFACE_MODEL,
    }:
        return None
    graph = document.main_graph
    if not graph.nodes or graph.nodes[0].domain != "nneditor.llm":
        return None
    if record.value_id in graph.inputs and record.numpy_dtype in {
        "int32",
        "int64",
        "uint32",
        "uint64",
    }:
        return "tokens"
    if record.value_id in graph.outputs and len(record.shape) in (2, 3):
        producer = graph.producer(record.value_id)
        if producer and graph.node(producer[0]).domain == "nneditor.llm":
            return "logits"
    return None


def decode_activation(
    document: Document,
    record: ActivationRecord,
    read: Callable[..., bytes],
    decoder: Decoder,
    *,
    batch: int = 0,
    position: int | None = None,
    skip_special: bool = True,
) -> DecodedActivation:
    kind = text_activation_kind(document, record)
    if not record.readable or kind is None:
        raise DetokenizationError(
            "This capture has no readable token IDs or language-model logits."
        )
    dtype = np.dtype(record.numpy_dtype)
    shape = record.shape
    batches = shape[0] if len(shape) > (1 if kind == "tokens" else 2) else 1
    if not 0 <= batch < batches:
        raise DetokenizationError(f"Batch index must be below {batches}.")
    if kind == "tokens":
        length = shape[-1]
        start = position or 0
        if not 0 <= start < length:
            raise DetokenizationError(f"Token position must be below {length}.")
        offset = (batch * length + start) * dtype.itemsize
        count = min(
            length - start,
            MAX_DECODE_TOKENS,
            max(0, (record.stored_byte_length - offset) // dtype.itemsize),
        )
        if not count:
            raise DetokenizationError("These tokens are outside the captured prefix.")
        ids = tuple(
            int(i)
            for i in np.frombuffer(
                read(record.value_id, offset=offset, length=count * dtype.itemsize),
                dtype=dtype,
            )
        )
        return DecodedActivation(
            decoder.decode(ids, skip_special=skip_special),
            ids,
            tuple(decoder.piece(i) for i in ids),
            decoder.name,
            f"Batch {batch}; token positions {start}-{start + count - 1} of {length}. "
            "Decoded token IDs. Hashed-word inputs do not preserve "
            "the original prompt.",
        )
    sequence, vocabulary = shape[-2:]
    width = vocabulary * dtype.itemsize
    if width > MAX_LOGIT_BYTES or width == 0:
        raise DetokenizationError("A logit row exceeds the 32 MiB inspection limit.")
    complete = min(sequence, record.stored_byte_length // width - batch * sequence)
    pos = complete - 1 if position is None else position
    if not 0 <= pos < sequence or pos >= complete:
        raise DetokenizationError(
            "Choose a position with a complete captured logit row; "
            "a partial vocabulary cannot give a reliable prediction."
        )
    offset = (batch * sequence + pos) * width
    logits = np.frombuffer(
        read(record.value_id, offset=offset, length=width), dtype=dtype
    ).astype(np.float64)
    if not np.all(np.isfinite(logits)):
        raise DetokenizationError(
            "Logits contain non-finite values; a reliable prediction is unavailable."
        )
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    best = np.argsort(-logits, kind="stable")[:10]
    ids = (int(best[0]),)
    candidates = tuple(
        TokenCandidate(int(i), decoder.piece(int(i)), float(weights[i])) for i in best
    )
    return DecodedActivation(
        decoder.decode(ids, skip_special=skip_special),
        ids,
        tuple(decoder.piece(i) for i in ids),
        decoder.name,
        f"Batch {batch}; position {pos}. Greedy next-token prediction. "
        "A forward-pass prediction is not an autoregressively generated response.",
        candidates,
    )
