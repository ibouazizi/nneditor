"""Text codebooks and language-model token-id generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nneditor.input_generation import InputGenerationError, generate_token_ids_tensor
from nneditor.tokenization import (
    MAX_VOCABULARY_BYTES,
    TOKENIZER_CHOICES,
    BpeCodebook,
    ByteCodebook,
    Codebook,
    CodepointCodebook,
    TokenizationError,
    WordHashCodebook,
    choice_for,
    load_codebook,
)


def _byte_vocabulary() -> dict[str, int]:
    """A byte-level vocabulary covering every single byte, GPT-2 style."""
    from nneditor.tokenization import _byte_encoder

    return {char: index for index, char in enumerate(_byte_encoder().values())}


def _write_gpt2_files(
    directory: Path,
    *,
    merges: list[str],
    extra_tokens: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    vocabulary = _byte_vocabulary()
    for token in extra_tokens:
        vocabulary[token] = len(vocabulary)
    vocabulary_path = directory / "vocab.json"
    vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")
    merges_path = directory / "merges.txt"
    merges_path.write_text(
        "#version: 0.2\n" + "\n".join(merges) + "\n", encoding="utf-8"
    )
    return vocabulary_path, merges_path


class TestSelfContainedCodebooks:
    def test_every_choice_is_loadable_and_declares_its_needs(self) -> None:
        assert {choice.id for choice in TOKENIZER_CHOICES} == {
            "byte",
            "codepoint",
            "word-hash",
            "gpt2-bpe",
            "tokenizer-json",
        }
        for choice in TOKENIZER_CHOICES:
            assert choice.note.strip()
            assert choice_for(choice.id) is choice
            if not choice.needs_files:
                codebook = load_codebook(choice.id)
                assert isinstance(codebook, Codebook)
                assert codebook.fidelity.strip()

    def test_unknown_choice_is_refused(self) -> None:
        with pytest.raises(TokenizationError, match="unknown tokenizer choice"):
            choice_for("no-such-tokenizer")

    def test_byte_codebook_matches_utf8(self) -> None:
        codebook = ByteCodebook()
        assert codebook.encode("hi") == (104, 105)
        # Multi-byte characters become their individual UTF-8 bytes.
        assert codebook.encode("é") == tuple("é".encode())
        assert max(codebook.encode("héllo")) < codebook.vocab_size

    def test_codepoint_codebook_folds_into_the_vocabulary(self) -> None:
        assert CodepointCodebook().encode("hi") == (104, 105)
        folded = CodepointCodebook(vocab_size=100)
        assert folded.encode("hi") == (104 % 100, 105 % 100)
        assert "wrap" in folded.fidelity
        assert "exact" in CodepointCodebook().fidelity

    def test_word_hash_codebook_is_deterministic_and_in_range(self) -> None:
        codebook = WordHashCodebook(vocab_size=512)
        first = codebook.encode("the quick brown fox")
        assert first == codebook.encode("the quick brown fox")
        assert len(first) == 4
        # Reserved ids stay free for padding, and nothing escapes the vocabulary.
        assert all(1 <= identifier < 512 for identifier in first)
        assert "not the model's own ids" in codebook.fidelity

    def test_word_hash_rejects_an_impossible_vocabulary(self) -> None:
        with pytest.raises(TokenizationError, match="at least two"):
            WordHashCodebook(vocab_size=1)
        with pytest.raises(TokenizationError, match="inside the vocabulary"):
            WordHashCodebook(vocab_size=4, reserved_ids=4)


class TestByteLevelBpe:
    def test_merges_are_applied_in_rank_order(self, tmp_path: Path) -> None:
        # "hi" merges into one token; "h" and "i" alone would be two.
        vocabulary_path, merges_path = _write_gpt2_files(
            tmp_path, merges=["h i"], extra_tokens=("hi",)
        )
        codebook = load_codebook(
            "gpt2-bpe",
            vocabulary_path=vocabulary_path,
            merges_path=merges_path,
        )
        vocabulary = _byte_vocabulary()
        assert codebook.encode("hi") == (len(vocabulary),)

        # Without the merge the same text stays two byte tokens.
        plain_directory = tmp_path / "plain"
        plain_directory.mkdir()
        plain_vocabulary, plain_merges = _write_gpt2_files(plain_directory, merges=[])
        plain = load_codebook(
            "gpt2-bpe",
            vocabulary_path=plain_vocabulary,
            merges_path=plain_merges,
        )
        assert plain.encode("hi") == (vocabulary["h"], vocabulary["i"])

    def test_leading_space_is_part_of_the_token(self, tmp_path: Path) -> None:
        vocabulary_path, merges_path = _write_gpt2_files(tmp_path, merges=[])
        codebook = load_codebook(
            "gpt2-bpe",
            vocabulary_path=vocabulary_path,
            merges_path=merges_path,
        )
        vocabulary = _byte_vocabulary()
        # GPT-2 encodes " a" as the space-prefixed piece, so the space maps to
        # its printable stand-in rather than a raw 0x20.
        space_prefixed = codebook.encode(" a")
        assert space_prefixed[0] == vocabulary["Ġ"]
        assert space_prefixed[1] == vocabulary["a"]

    def test_inconsistent_vocabulary_and_merges_are_reported(
        self, tmp_path: Path
    ) -> None:
        # A merge produces a token the vocabulary does not contain.
        vocabulary_path, merges_path = _write_gpt2_files(tmp_path, merges=["h i"])
        codebook = load_codebook(
            "gpt2-bpe",
            vocabulary_path=vocabulary_path,
            merges_path=merges_path,
        )
        with pytest.raises(TokenizationError, match="may not belong together"):
            codebook.encode("hi")

    def test_tokenizer_json_carries_vocabulary_and_merges(self, tmp_path: Path) -> None:
        vocabulary = _byte_vocabulary()
        vocabulary["hi"] = len(vocabulary)
        payload = {
            "model": {"type": "BPE", "vocab": vocabulary, "merges": ["h i"]},
        }
        path = tmp_path / "tokenizer.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        codebook = load_codebook("tokenizer-json", vocabulary_path=path)
        assert codebook.encode("hi") == (vocabulary["hi"],)

    def test_unsupported_tokenizer_json_model_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "tokenizer.json"
        path.write_text(
            json.dumps({"model": {"type": "WordPiece", "vocab": {"a": 0}}}),
            encoding="utf-8",
        )
        with pytest.raises(TokenizationError, match="only byte-level BPE"):
            load_codebook("tokenizer-json", vocabulary_path=path)

    def test_missing_and_malformed_files_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TokenizationError, match="needs a vocabulary file"):
            load_codebook("gpt2-bpe")
        vocabulary_path, merges_path = _write_gpt2_files(tmp_path, merges=[])
        with pytest.raises(TokenizationError, match="needs a merges file"):
            load_codebook("gpt2-bpe", vocabulary_path=vocabulary_path)
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        with pytest.raises(TokenizationError, match="not valid JSON"):
            load_codebook("gpt2-bpe", vocabulary_path=broken, merges_path=merges_path)
        with pytest.raises(TokenizationError, match=r"cannot read|not a readable"):
            load_codebook(
                "gpt2-bpe",
                vocabulary_path=tmp_path / "absent.json",
                merges_path=merges_path,
            )

    def test_oversized_vocabulary_is_refused_before_parsing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("nneditor.tokenization.MAX_VOCABULARY_BYTES", 8)
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps({"a": 0, "b": 1}), encoding="utf-8")
        with pytest.raises(TokenizationError, match="above the"):
            load_codebook("tokenizer-json", vocabulary_path=path)
        assert MAX_VOCABULARY_BYTES > 8

    def test_empty_vocabulary_is_refused(self) -> None:
        with pytest.raises(TokenizationError, match="vocabulary is empty"):
            BpeCodebook({}, [])


class TestTokenIdGeneration:
    def test_fixed_length_pads_and_records_the_disclosure(self, tmp_path: Path) -> None:
        generated = generate_token_ids_tensor(
            tmp_path / "ids.npy",
            text="hi",
            codebook=ByteCodebook(),
            sequence_length=8,
            pad_id=0,
        )
        assert generated.shape == (1, 8)
        assert generated.dtype == "int64"
        array = np.load(generated.path)
        assert array[0].tolist() == [104, 105, 0, 0, 0, 0, 0, 0]
        assert "6 pad id(s)" in generated.source
        # The fidelity of the codebook travels with the artifact.
        assert "byte-level models" in generated.source

    def test_bos_and_eos_wrap_the_sequence(self, tmp_path: Path) -> None:
        generated = generate_token_ids_tensor(
            tmp_path / "ids.npy",
            text="hi",
            codebook=ByteCodebook(),
            bos_id=1,
            eos_id=2,
            layout="T",
        )
        assert generated.shape == (4,)
        assert np.load(generated.path).tolist() == [1, 104, 105, 2]

    def test_truncation_preserves_the_end_of_sequence_id(self, tmp_path: Path) -> None:
        generated = generate_token_ids_tensor(
            tmp_path / "ids.npy",
            text="hello world",
            codebook=ByteCodebook(),
            sequence_length=4,
            eos_id=2,
        )
        values = np.load(generated.path)[0].tolist()
        assert len(values) == 4
        assert values[-1] == 2
        assert "truncated" in generated.source

    def test_refusing_to_truncate_is_an_explicit_error(self, tmp_path: Path) -> None:
        with pytest.raises(InputGenerationError, match="exceed the requested"):
            generate_token_ids_tensor(
                tmp_path / "ids.npy",
                text="hello world",
                codebook=ByteCodebook(),
                sequence_length=2,
                truncate=False,
            )

    def test_ids_outside_the_vocabulary_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(InputGenerationError, match="outside the codebook"):
            generate_token_ids_tensor(
                tmp_path / "ids.npy",
                text="hi",
                codebook=ByteCodebook(),
                bos_id=999,
            )

    def test_empty_text_without_markers_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(InputGenerationError, match="produced no tokens"):
            generate_token_ids_tensor(
                tmp_path / "ids.npy",
                text="",
                codebook=ByteCodebook(),
            )

    def test_float_dtypes_are_refused_for_token_ids(self, tmp_path: Path) -> None:
        with pytest.raises(InputGenerationError):
            generate_token_ids_tensor(
                tmp_path / "ids.npy",
                text="hi",
                codebook=ByteCodebook(),
                dtype="float32",
            )

    def test_generation_is_reproducible(self, tmp_path: Path) -> None:
        first = generate_token_ids_tensor(
            tmp_path / "a.npy",
            text="the quick brown fox",
            codebook=WordHashCodebook(vocab_size=1024),
            sequence_length=16,
        )
        second = generate_token_ids_tensor(
            tmp_path / "b.npy",
            text="the quick brown fox",
            codebook=WordHashCodebook(vocab_size=1024),
            sequence_length=16,
        )
        assert np.array_equal(np.load(first.path), np.load(second.path))
