"""Detokenization uses model vocabularies and complete, bounded logit rows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models

from nneditor.adapters.gguf.to_ir import open_gguf
from nneditor.adapters.huggingface.to_ir import open_huggingface
from nneditor.detokenization import (
    DetokenizationError,
    JsonDecoder,
    VocabularyDecoder,
    decode_activation,
    decoder_for_document,
)
from nneditor.tokenization import _byte_encoder
from tests.fixtures.gguf_models import write_gguf
from tests.fixtures.huggingface_models import write_llama_snapshot
from tests.unit.test_tensor_page import record_for


def test_sentencepiece_byte_fallback_and_special_tokens() -> None:
    decoder = VocabularyDecoder(
        ("<s>", "▁Hello", "▁world", "!", "<0xF0>", "<0x9F>", "<0x98>", "<0x80>"),
        "llama",
        frozenset({0}),
    )
    assert decoder.decode(range(8)) == "Hello world!😀"
    assert "<s>" in decoder.decode([0, 1], skip_special=False)
    for ids in ((99,), (-1,), (True,)):
        with pytest.raises(DetokenizationError):
            decoder.decode(ids)


def test_byte_bpe_reassembles_unicode_across_token_boundaries() -> None:
    mapping = _byte_encoder()
    text = " café 😀"
    decoder = VocabularyDecoder(tuple(mapping[i] for i in range(256)), "gpt2")
    assert decoder.decode(text.encode("utf-8")) == text
    bert = VocabularyDecoder(
        ("[CLS]", "hello", "##s", ",", "world"), "bert", frozenset({0})
    )
    assert bert.decode(range(5)) == "hellos, world"


@pytest.mark.parametrize(
    "algorithm,pieces,expected",
    [
        ("gpt2", ("custom", "Ġcheckpoint"), "custom checkpoint"),
        ("llama", ("▁custom", "▁checkpoint"), "custom checkpoint"),
        ("t5", ("▁custom", "▁checkpoint"), "custom checkpoint"),
        ("bert", ("custom", "##checkpoint"), "customcheckpoint"),
    ],
)
def test_gguf_decoder_uses_the_open_files_vocabulary(
    tmp_path: Path, algorithm: str, pieces: tuple[str, str], expected: str
) -> None:
    path = write_gguf(
        tmp_path / "my-model.gguf",
        version=3,
        kv=(
            ("general.architecture", 8, "llama"),
            ("tokenizer.ggml.model", 8, algorithm),
            ("tokenizer.ggml.tokens", 9, (8, pieces)),
        ),
    )
    decoder = decoder_for_document(open_gguf(path))
    assert decoder.decode([0, 1]) == expected
    assert decoder.piece(0) == pieces[0]
    assert "GGUF embedded vocabulary: my-model.gguf" in decoder.name
    assert f"2 tokens; {algorithm} decoding" in decoder.name


def test_gguf_without_a_vocabulary_does_not_fall_back_to_gpt2(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "missing-vocabulary.gguf",
        version=3,
        kv=(
            ("general.architecture", 8, "llama"),
            ("tokenizer.ggml.model", 8, "gpt2"),
        ),
    )
    with pytest.raises(DetokenizationError, match="no usable token vocabulary"):
        decoder_for_document(open_gguf(path))


def test_native_json_decoder_and_missing_or_invalid_tokenizer(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    tokenizer = Tokenizer(
        models.WordPiece(
            {"[UNK]": 0, "hello": 1, "##s": 2, "world": 3}, unk_token="[UNK]"
        )
    )
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.save(str(path))
    decoder = JsonDecoder(path)
    assert decoder.decode([1, 2, 3]) == "hellos world"
    assert decoder.piece(2) == "##s"
    with pytest.raises(DetokenizationError):
        decoder.decode([99])
    path.write_text("invalid")
    with pytest.raises(DetokenizationError, match="Cannot load"):
        JsonDecoder(path)
    root = write_llama_snapshot(tmp_path / "snapshot")
    (root / "tokenizer.json").unlink()
    with pytest.raises(DetokenizationError, match=r"no tokenizer\.json"):
        decoder_for_document(open_huggingface(root))


def _model(root: Path) -> Any:
    return open_huggingface(write_llama_snapshot(root))


def test_capture_token_batches_and_complete_logit_predictions(tmp_path: Path) -> None:
    document = _model(tmp_path / "model")
    graph = document.main_graph
    decoder = VocabularyDecoder(tuple("abcdefgh"), "gpt2")
    ids = np.array([[0, 1, 2], [3, 4, 5]], np.int64)
    record = replace(record_for(ids), value_id=graph.inputs[0])
    raw = ids.tobytes()

    def read(_key: str, **kw: Any) -> bytes:
        return raw[kw["offset"] : kw["offset"] + kw["length"]]

    result = decode_activation(document, record, read, decoder, batch=1, position=1)
    assert result.text == "ef" and result.ids == (4, 5)
    logits = np.zeros((2, 3, 8), np.float32)
    logits[1, 2, 6] = 8
    record = replace(record_for(logits), value_id=graph.outputs[0])
    raw = logits.tobytes()
    result = decode_activation(document, record, read, decoder, batch=1)
    assert result.ids == (6,) and result.text == "g"
    assert result.candidates[0].probability > 0.99
    assert sum(c.probability for c in result.candidates) == pytest.approx(1)
    assert "position 2" in result.note and "not an autoregressively" in result.note


def test_partial_logit_row_is_not_a_prediction_and_nonfinite_is_explicit(
    tmp_path: Path,
) -> None:
    document = _model(tmp_path / "model")
    decoder = VocabularyDecoder(tuple("abcdefgh"), "gpt2")
    logits = np.arange(24, dtype=np.float32).reshape(1, 3, 8)
    record = replace(record_for(logits, 40), value_id=document.main_graph.outputs[0])
    calls: list[tuple[int, int]] = []

    def read(_key: str, **kw: Any) -> bytes:
        calls.append((kw["offset"], kw["length"]))
        return logits.tobytes()[kw["offset"] : kw["offset"] + kw["length"]]

    result = decode_activation(document, record, read, decoder)
    assert result.ids == (7,) and "position 0" in result.note
    assert calls == [(0, 32)]
    with pytest.raises(DetokenizationError, match="complete captured logit row"):
        decode_activation(document, record, read, decoder, position=1)
    partial = replace(record_for(logits, 16), value_id=record.value_id)
    with pytest.raises(DetokenizationError):
        decode_activation(document, partial, read, decoder)
    logits[0, 0, 0] = np.nan
    with pytest.raises(DetokenizationError, match="non-finite"):
        decode_activation(document, record, read, decoder)


def test_decode_limits_and_batch_bounds(tmp_path: Path) -> None:
    decoder = VocabularyDecoder(("a",), "gpt2")
    with pytest.raises(DetokenizationError, match="4096"):
        decoder.decode([0] * 4097)
    document = _model(tmp_path / "model")
    logits = np.ones((1, 1, 8), np.float32)
    record = replace(record_for(logits), value_id=document.main_graph.outputs[0])
    with pytest.raises(DetokenizationError, match="Batch index"):
        decode_activation(document, record, lambda *_a, **_kw: b"", decoder, batch=-1)


@pytest.mark.parametrize(
    "version,endian", [(1, "little"), (2, "little"), (3, "little"), (3, "big")]
)
def test_elided_gguf_vocabulary_is_loaded_on_demand(
    tmp_path: Path, version: int, endian: str
) -> None:
    path = write_gguf(
        tmp_path / "tokens.gguf",
        version=version,
        endian=endian,
        kv=(
            ("general.architecture", 8, "llama"),
            ("tokenizer.ggml.model", 8, "llama"),
            ("tokenizer.ggml.tokens", 9, (8, tuple(f"▁word{i}" for i in range(5000)))),
        ),
    )
    document = open_gguf(path)
    metadata = dict(document.extensions)["x-gguf.metadata"]
    assert isinstance(metadata, dict) and not isinstance(
        metadata["tokenizer.ggml.tokens"], list
    )
    decoder = decoder_for_document(document)
    assert decoder.decode([4998, 4999]) == "word4998 word4999"
