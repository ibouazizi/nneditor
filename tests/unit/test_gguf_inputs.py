"""Generated GGUF prompts must use embedded token IDs through NPZ binding."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from nneditor.adapters.gguf.to_ir import open_gguf
from nneditor.application.session import ApplicationService
from nneditor.detokenization import decoder_for_document
from nneditor.gguf_tokenization import GgufCodebook
from nneditor.input_files import publish_input_archive, read_input_array
from nneditor.input_generation import GeneratedTensor
from nneditor.prompt_inputs import prompt_settings_for_document
from nneditor.tokenization import TokenizationError
from nneditor.tracing.inputs import bind_inputs
from tests.fixtures.gguf_models import write_gguf
from tests.fixtures.gguf_tokenizers import write_tokenized_gguf
from tests.unit.test_shell import make_shell


@pytest.mark.parametrize(
    "pre", ["gpt-2", "qwen2", "qwen35", "llama3", "llama-bpe", "llama-v3"]
)
def test_embedded_bpe_merges_round_trip_model_specific_ids(
    tmp_path: Path, pre: str
) -> None:
    path = write_tokenized_gguf(tmp_path / "model.gguf", pre=pre)
    document = open_gguf(path)
    codebook = GgufCodebook(document)
    decoder = decoder_for_document(document)
    ids = codebook.encode("Hello world!")
    assert ids == (259, 264, 0)
    assert decoder.decode(ids) == "Hello world!"
    multilingual = "Hello café 👋\n你好 12345 a\u0301"
    assert decoder.decode(codebook.encode(multilingual)) == multilingual
    assert "267 tokens" in codebook.fidelity
    settings = prompt_settings_for_document(document)
    assert settings.tokenizer_choice == "gguf-embedded"
    assert settings.bos_id is None and settings.eos_id is None


def test_missing_or_unknown_gguf_tokenizer_never_silently_hashes_text(
    tmp_path: Path,
) -> None:
    unknown = write_tokenized_gguf(tmp_path / "unknown.gguf", pre="not-implemented")
    with pytest.raises(TokenizationError, match="cannot yet be reconstructed"):
        GgufCodebook(unknown)
    missing = write_gguf(tmp_path / "missing.gguf", version=3)
    with pytest.raises(TokenizationError):
        GgufCodebook(missing)


def test_generate_test_input_defaults_to_gguf_vocabulary_and_named_npz(
    tmp_path: Path,
) -> None:
    path = write_tokenized_gguf(tmp_path / "model.gguf")
    with ApplicationService(prewarm_layouts=False) as service:
        shell, _page = make_shell(service)
        session = service.open_model(path)
        shell.show_session(session)
        generator = shell.input_generator
        assert generator.token_codebook.value == "gguf-embedded"
        assert generator.token_vocabulary_path.value == str(path)
        assert generator.output_format.value == "npz"
        assert generator.generated_input_name() == "tokens.npz"
        generator.token_text.value = "Hello world!"
        generated = generator.generate_current_input(tmp_path / "prompt.npz")
        assert generated.path.suffix == ".npz"
        with np.load(generated.path, allow_pickle=False) as archive:
            assert archive.files == ["tokens"]
            np.testing.assert_array_equal(archive["tokens"], [[259, 264, 0]])
        binding = session.trace_tensor_input("tokens", generated.path)
        specification = session.default_trace_inputs(bindings={"tokens": binding})
        arrays = bind_inputs(session.document, specification)
        np.testing.assert_array_equal(arrays["tokens"], [[259, 264, 0]])
        assert arrays["tokens"].dtype == np.dtype("int64")


@pytest.mark.parametrize("name", ["tokens", "allow_pickle", "file"])
def test_npz_preserves_named_numeric_arrays(tmp_path: Path, name: str) -> None:
    original = np.arange(12, dtype=np.int64).reshape(1, 12)
    np.save(tmp_path / "input.npy", original, allow_pickle=False)
    tensor = GeneratedTensor(
        tmp_path / "input.npy", original.shape, "int64", original.nbytes, "test"
    )
    result = publish_input_archive(tensor, tmp_path / "input.npz", name)
    np.testing.assert_array_equal(read_input_array(result.path, name), original)


def test_npz_rejects_objects_ambiguous_arrays_and_forged_allocations(
    tmp_path: Path,
) -> None:
    objects = tmp_path / "objects.npz"
    np.savez(objects, tokens=np.array([{"not": "executed"}], dtype=object))
    with pytest.raises(ValueError, match="real numeric"):
        read_input_array(objects, "tokens")
    ambiguous = tmp_path / "ambiguous.npz"
    np.savez(ambiguous, a=np.zeros(1), b=np.zeros(1))
    with pytest.raises(ValueError, match="unambiguous"):
        read_input_array(ambiguous, "tokens")
    huge = tmp_path / "huge.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"shape": (2**40,), "fortran_order": False, "descr": "<f4"}
    )
    with zipfile.ZipFile(huge, "w") as archive:
        archive.writestr("tokens.npy", header.getvalue())
    with pytest.raises(ValueError, match="512 MiB"):
        read_input_array(huge, "tokens")
