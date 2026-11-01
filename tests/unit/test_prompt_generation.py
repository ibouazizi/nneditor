"""Prompt catalogs and model-bound token-id generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nneditor.adapters.gguf import open_gguf
from nneditor.adapters.huggingface import open_huggingface
from nneditor.application.session import ApplicationService
from nneditor.prompt_inputs import (
    DEFAULT_PROMPT,
    SAMPLE_PROMPTS,
    codebook_for_document,
    generate_prompt_token_tensor,
    prompt_by_id,
    prompt_settings_for_document,
)
from nneditor.tokenization import WordHashCodebook
from nneditor.ui.input_workspace import (
    InputTarget,
    TestInputWorkspace,
    preset_for_target,
)
from tests.fixtures.gguf_tokenizers import write_tokenized_gguf
from tests.fixtures.huggingface_models import write_traceable_llama_snapshot
from tests.unit.test_shell import make_shell


def test_catalog_contains_the_default_pangram() -> None:
    assert prompt_by_id("pangram").text == DEFAULT_PROMPT
    assert {prompt.id for prompt in SAMPLE_PROMPTS} >= {
        "pangram",
        "photosynthesis",
        "primes",
        "village",
        "attention",
    }


def test_huggingface_tiny_tokenizer_falls_back_to_word_hash(tmp_path: Path) -> None:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    document = open_huggingface(root)
    codebook = codebook_for_document(document)
    assert isinstance(codebook, WordHashCodebook)
    assert codebook.vocab_size == 8
    settings = prompt_settings_for_document(document)
    assert settings.tokenizer_choice == "word-hash"
    assert settings.bos_id == 1
    assert settings.eos_id == 2


def test_gguf_uses_its_embedded_vocabulary_and_merges(tmp_path: Path) -> None:
    path = write_tokenized_gguf(tmp_path / "model.gguf")
    document = open_gguf(path)
    codebook = codebook_for_document(document)
    assert codebook.name == "gguf-embedded"
    assert codebook.vocab_size == 267
    settings = prompt_settings_for_document(document)
    assert settings.bos_id is None
    assert settings.eos_id is None


def test_generate_prompt_tensor_uses_the_catalog(tmp_path: Path) -> None:
    path = write_tokenized_gguf(tmp_path / "model.gguf")
    document = open_gguf(path)
    generated = generate_prompt_token_tensor(
        tmp_path / "tokens.npy",
        document=document,
        prompt_id="pangram",
    )
    array = np.load(generated.path, allow_pickle=False)
    assert array.dtype == np.dtype("int64")
    assert array.ndim == 2
    assert array.shape[0] == 1
    assert array.shape[1] >= 1
    assert int(array.max()) < 267
    assert "gguf-embedded" in generated.source


def test_text_token_preset_keeps_natural_length_for_symbolic_seq() -> None:
    target = InputTarget(
        name="tokens",
        element_type="int64",
        shape=("batch", "seq"),
        prefer_form="text-tokens",
    )
    preset = preset_for_target(target)
    assert preset.kind == "text-tokens"
    assert preset.sequence_length is None
    assert preset.shape_description == "(1, prompt length)"
    assert "natural token length" in " ".join(preset.assumptions)


def test_text_token_preset_generation_encodes_the_catalog_prompt(
    tmp_path: Path,
) -> None:
    path = write_tokenized_gguf(tmp_path / "model.gguf")
    document = open_gguf(path)
    target = InputTarget(
        name="tokens",
        element_type="int64",
        shape=("batch", "seq"),
        prefer_form="text-tokens",
        prompt=prompt_settings_for_document(document),
    )
    generated = TestInputWorkspace._generate_preset(
        preset_for_target(target),
        tmp_path / "tokens.npy",
    )
    array = np.load(generated.path, allow_pickle=False)
    assert array.dtype == np.dtype("int64")
    assert array.shape[0] == 1
    assert array.shape[1] >= 1
    assert int(array.max()) < 267
    assert "gguf-embedded" in generated.source


def test_gguf_session_defaults_the_generator_to_sample_prompts(tmp_path: Path) -> None:
    path = write_tokenized_gguf(tmp_path / "model.gguf")
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(path))
        generator = shell.input_generator
        assert generator.kind.value == "text-tokens"
        assert generator.token_section.visible
        assert generator.token_text.value == DEFAULT_PROMPT
        assert generator.token_codebook.value == "gguf-embedded"
        assert generator.token_sequence_length.value == ""
        assert generator.token_bos_id.value == ""
        assert generator.token_eos_id.value == ""
        catalog = {str(option.key) for option in generator.token_prompt_catalog.options}
        assert catalog >= {prompt.id for prompt in SAMPLE_PROMPTS}


def test_huggingface_session_defaults_the_generator_to_sample_prompts(
    tmp_path: Path,
) -> None:
    root = write_traceable_llama_snapshot(tmp_path / "hf")
    with ApplicationService() as service:
        shell, _page = make_shell(service)
        shell.show_session(service.open_model(root))
        generator = shell.input_generator
        assert generator.kind.value == "text-tokens"
        assert generator.token_text.value == DEFAULT_PROMPT
        assert generator.token_codebook.value == "word-hash"
