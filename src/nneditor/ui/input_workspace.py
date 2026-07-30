"""Self-contained test-input generation workspace."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flet as ft

from nneditor.input_generation import (
    GeneratedTensor,
    InputGenerationError,
    generate_csv_tensor,
    generate_image_tensor,
    generate_mask_tensor,
    generate_synthetic_tensor,
    generate_time_series_tensor,
    generate_token_ids_tensor,
    parse_columns,
    parse_shape,
)
from nneditor.ir.core import Dim
from nneditor.tokenization import (
    TOKENIZER_CHOICES,
    Codebook,
    TokenizationError,
    TokenizerChoice,
    choice_for,
    load_codebook,
)
from nneditor.ui.shell_layout import ShellPalette

__all__ = ["InputTarget", "TestInputWorkspace", "build_input_workspace"]


@dataclass(frozen=True, slots=True)
class InputTarget:
    """A model input summarized without coupling the form to a session."""

    name: str
    element_type: str | None
    shape: tuple[Dim, ...] | None
    is_mask: bool = False


class TestInputWorkspace:
    """Own the generator form, safe publication, and target defaults."""

    def __init__(
        self,
        *,
        page: ft.Page,
        picker: ft.FilePicker,
        palette: ShellPalette,
        assign: Callable[[str, GeneratedTensor], None],
        on_error: Callable[[str], None],
        on_status: Callable[[str], None],
        clear_error: Callable[[], None],
        watch_text_focus: Callable[[ft.TextField], ft.TextField],
    ) -> None:
        self.page = page
        self.picker = picker
        self.palette = palette
        self._assign = assign
        self._on_error = on_error
        self._on_status = on_status
        self._clear_error = clear_error
        self._watch_text_focus = watch_text_focus
        self._targets: dict[str, InputTarget] = {}
        self._can_assign = False

        self.kind = ft.Dropdown(
            value="image",
            label="Input source",
            dense=True,
            options=[
                ft.DropdownOption(key="image", text="Image file"),
                ft.DropdownOption(key="mask", text="Mask"),
                ft.DropdownOption(key="text-tokens", text="Text tokens"),
                ft.DropdownOption(key="csv", text="CSV / tabular data"),
                ft.DropdownOption(key="time-series", text="Synthetic time series"),
                ft.DropdownOption(key="tensor", text="Generic tensor"),
            ],
            on_select=self._on_kind_changed,
        )
        self.target_input = ft.Dropdown(
            label="Assign to model input after saving",
            hint_text="Open a model to select an input",
            dense=True,
            options=[],
            disabled=True,
            on_select=self._on_target_changed,
        )

        self.image_path = self._field(
            label="Image",
            hint_text="PNG, JPEG, WebP, BMP, or TIFF",
            read_only=True,
            expand=True,
        )
        # Tensor shapes conventionally spell the spatial dimensions H, W.
        # Keep the form in that same order so copying a declared shape cannot
        # silently transpose a non-square image.
        self.image_height = self._field(label="Height (H)", value="224")
        self.image_width = self._field(label="Width (W)", value="224")
        self.image_layout = ft.Dropdown(
            value="NCHW",
            label="Layout",
            dense=True,
            options=[
                *[
                    ft.DropdownOption(key=value, text=value)
                    for value in ("NCHW", "NHWC", "CHW", "HWC", "NHW", "HW")
                ],
                ft.DropdownOption(
                    key="QWEN3VL_PATCHES",
                    text="Qwen3-VL flattened patches",
                ),
            ],
        )
        self.image_color = ft.Dropdown(
            value="rgb",
            label="Color",
            dense=True,
            options=[
                ft.DropdownOption(key="rgb", text="RGB"),
                ft.DropdownOption(key="grayscale", text="Grayscale"),
            ],
        )
        self.image_normalization = ft.Dropdown(
            value="zero-one",
            label="Normalization",
            dense=True,
            options=[
                ft.DropdownOption(key="none", text="None"),
                ft.DropdownOption(key="zero-one", text="0 to 1"),
                ft.DropdownOption(key="minus-one-one", text="-1 to 1"),
                ft.DropdownOption(key="imagenet", text="ImageNet"),
            ],
        )
        self.image_dtype = self._dtype_dropdown("float32")
        self._set_columns(
            (
                self.image_height,
                self.image_width,
                self.image_layout,
                self.image_color,
                self.image_normalization,
                self.image_dtype,
            ),
            {"xs": 12, "sm": 4},
        )
        self.image_section = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            self.image_path,
                            ft.TextButton(
                                content="Choose image",
                                icon=ft.Icons.IMAGE_SEARCH_ROUNDED,
                                on_click=self._on_choose_image,
                            ),
                        ]
                    ),
                    self._responsive_row(
                        self.image_height,
                        self.image_width,
                        self.image_layout,
                    ),
                    self._responsive_row(
                        self.image_color,
                        self.image_normalization,
                        self.image_dtype,
                    ),
                ],
                spacing=10,
            )
        )

        self.mask_shape = self._field(label="Shape", value="1,64,64")
        self.mask_fill = ft.Dropdown(
            value="ones",
            label="Fill",
            dense=True,
            options=[
                ft.DropdownOption(key="ones", text="Ones / all valid"),
                ft.DropdownOption(key="zeros", text="Zeros"),
                ft.DropdownOption(key="checkerboard", text="Checkerboard"),
                ft.DropdownOption(key="random-binary", text="Random binary"),
            ],
        )
        self.mask_dtype = self._dtype_dropdown("int64")
        self.mask_seed = self._field(label="Seed", value="0")
        self._set_columns(
            (
                self.mask_shape,
                self.mask_fill,
                self.mask_dtype,
                self.mask_seed,
            ),
            {"xs": 12, "sm": 6, "md": 3},
        )
        self.mask_section = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        "An all-ones mask is the usual all-valid attention or "
                        "pixel mask.",
                        size=10,
                        color=palette.muted,
                    ),
                    self._responsive_row(
                        self.mask_shape,
                        self.mask_fill,
                        self.mask_dtype,
                        self.mask_seed,
                    ),
                ],
                spacing=10,
            ),
            visible=False,
        )

        self.token_section = self._build_token_section(palette)

        self.csv_path = self._field(
            label="CSV / text file",
            read_only=True,
            expand=True,
        )
        self.csv_delimiter = ft.Dropdown(
            value=",",
            label="Delimiter",
            dense=True,
            options=[
                ft.DropdownOption(key=",", text="Comma"),
                ft.DropdownOption(key=r"\t", text="Tab"),
                ft.DropdownOption(key=";", text="Semicolon"),
                ft.DropdownOption(key="whitespace", text="Whitespace"),
            ],
        )
        self.csv_skip_rows = self._field(label="Skip rows", value="0")
        self.csv_columns = self._field(
            label="Columns (optional)",
            hint_text="0,2,4",
        )
        self.csv_dtype = self._dtype_dropdown("float32")
        self.csv_transpose = ft.Checkbox(label="Transpose")
        self.csv_add_batch = ft.Checkbox(label="Add batch dimension")
        self._set_columns(
            (
                self.csv_delimiter,
                self.csv_skip_rows,
                self.csv_columns,
                self.csv_dtype,
            ),
            {"xs": 12, "sm": 6, "md": 3},
        )
        self.csv_section = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            self.csv_path,
                            ft.TextButton(
                                content="Choose data",
                                icon=ft.Icons.TABLE_VIEW_ROUNDED,
                                on_click=self._on_choose_csv,
                            ),
                        ]
                    ),
                    self._responsive_row(
                        self.csv_delimiter,
                        self.csv_skip_rows,
                        self.csv_columns,
                        self.csv_dtype,
                    ),
                    ft.Row(
                        controls=[
                            self.csv_transpose,
                            self.csv_add_batch,
                        ]
                    ),
                ],
                spacing=10,
            ),
            visible=False,
        )

        self.series_samples = self._field(label="Samples", value="256")
        self.series_channels = self._field(label="Channels", value="1")
        self.series_waveform = ft.Dropdown(
            value="sine",
            label="Waveform",
            dense=True,
            options=[
                ft.DropdownOption(key=value, text=value.replace("-", " ").title())
                for value in ("sine", "cosine", "sawtooth", "random-walk")
            ],
        )
        self.series_sample_rate = self._field(label="Sample rate", value="100")
        self.series_frequency = self._field(label="Frequency", value="1")
        self.series_amplitude = self._field(label="Amplitude", value="1")
        self.series_layout = self._dropdown(
            "Layout",
            "NTC",
            ("NTC", "NCT", "TC", "CT", "NC"),
        )
        self.series_dtype = self._dtype_dropdown("float32")
        self.series_seed = self._field(label="Seed", value="0")
        self._set_columns(
            (
                self.series_samples,
                self.series_channels,
                self.series_waveform,
                self.series_sample_rate,
                self.series_frequency,
                self.series_amplitude,
                self.series_layout,
                self.series_dtype,
                self.series_seed,
            ),
            {"xs": 12, "sm": 4},
        )
        self.series_section = ft.Container(
            content=ft.Column(
                controls=[
                    self._responsive_row(
                        self.series_samples,
                        self.series_channels,
                        self.series_waveform,
                    ),
                    self._responsive_row(
                        self.series_sample_rate,
                        self.series_frequency,
                        self.series_amplitude,
                    ),
                    self._responsive_row(
                        self.series_layout,
                        self.series_dtype,
                        self.series_seed,
                    ),
                ],
                spacing=10,
            ),
            visible=False,
        )

        self.tensor_shape = self._field(label="Shape", value="1,3,224,224")
        self.tensor_distribution = ft.Dropdown(
            value="zeros",
            label="Values",
            dense=True,
            options=[
                ft.DropdownOption(key=value, text=value.title())
                for value in ("zeros", "ones", "uniform", "normal", "ramp")
            ],
        )
        self.tensor_dtype = self._dtype_dropdown("float32")
        self.tensor_seed = self._field(label="Seed", value="0")
        self._set_columns(
            (
                self.tensor_shape,
                self.tensor_distribution,
                self.tensor_dtype,
                self.tensor_seed,
            ),
            {"xs": 12, "sm": 6},
        )
        self.tensor_section = ft.Container(
            content=self._responsive_row(
                self.tensor_shape,
                self.tensor_distribution,
                self.tensor_dtype,
                self.tensor_seed,
            ),
            visible=False,
        )
        self.sections = (
            self.image_section,
            self.mask_section,
            self.token_section,
            self.csv_section,
            self.series_section,
            self.tensor_section,
        )
        self.generate_button = ft.FilledButton(
            content="Generate & save",
            icon=ft.Icons.SAVE_ALT_ROUNDED,
            on_click=self._on_generate,
        )
        self.generate_and_assign_button = ft.FilledButton(
            content="Generate, save & assign",
            icon=ft.Icons.INPUT_ROUNDED,
            on_click=self._on_generate_and_assign,
            disabled=True,
        )
        self.status = ft.Text(
            "Configure a source, then choose where to save the .npy file.",
            size=11,
            color=palette.muted,
        )
        self.summary = ft.Text(
            "No tensor generated yet.",
            size=11,
            color=palette.ink,
            selectable=True,
        )
        self.control = build_input_workspace(
            palette=palette,
            kind=self.kind,
            target_input=self.target_input,
            sections=self.sections,
            generate_button=self.generate_button,
            generate_and_assign_button=self.generate_and_assign_button,
            status=self.status,
            summary=self.summary,
        )

    def refresh_targets(
        self,
        targets: Sequence[InputTarget],
        *,
        can_assign: bool,
    ) -> None:
        """Replace model targets and preserve a still-valid selection."""
        previous = self.target_input.value
        self._targets = {target.name: target for target in targets}
        self._can_assign = can_assign
        names = tuple(self._targets)
        self.target_input.options = [
            ft.DropdownOption(
                key=target.name,
                text=(
                    f"{target.name} · {target.element_type or 'unknown'} · "
                    f"{target.shape if target.shape is not None else 'shape unknown'}"
                ),
            )
            for target in targets
        ]
        self.target_input.value = (
            previous if previous in self._targets else (names[0] if names else None)
        )
        self.target_input.disabled = not bool(names)
        self.target_input.hint_text = (
            "Select a model input" if names else "Open a model to select an input"
        )
        self._apply_target_defaults()
        self._refresh_assign_button()

    def _build_token_section(self, palette: ShellPalette) -> ft.Container:
        """Build the language-model token-id form and its codebook slots.

        Kept out of ``__init__`` because the codebook selection owns two file
        slots whose visibility follows the selected tokenizer, which is more
        wiring than the other sources need.
        """
        self.token_text = self._field(
            label="Text",
            hint_text="The prompt to encode into token ids",
            value="The quick brown fox jumps over the lazy dog.",
            multiline=True,
            min_lines=2,
            max_lines=6,
        )
        self.token_codebook = ft.Dropdown(
            value=TOKENIZER_CHOICES[0].id,
            label="Tokenizer",
            dense=True,
            options=[
                ft.DropdownOption(key=choice.id, text=choice.label)
                for choice in TOKENIZER_CHOICES
            ],
            on_select=self._on_codebook_changed,
        )
        self.token_note = ft.Text(size=10, color=palette.muted)
        self.token_sequence_length = self._field(
            label="Sequence length",
            hint_text="Blank keeps the natural length",
        )
        self.token_layout = self._dropdown("Layout", "NT", ("NT", "T"))
        # Token ids are indexes, so the generator only accepts integer dtypes.
        self.token_dtype = self._dropdown(
            "Dtype",
            "int64",
            ("int32", "int64", "uint32", "uint64"),
        )
        self.token_vocab_size = self._field(
            label="Vocabulary size",
            hint_text="Blank uses the codebook default",
        )
        self.token_pad_id = self._field(label="Pad id", value="0")
        self.token_bos_id = self._field(label="BOS id (optional)")
        self.token_eos_id = self._field(label="EOS id (optional)")
        self.token_truncate = ft.Checkbox(
            label="Truncate text longer than the sequence length",
            value=True,
        )
        self.token_vocabulary_path = self._field(
            label="Vocabulary",
            read_only=True,
            expand=True,
        )
        self.token_merges_path = self._field(
            label="Merges (merges.txt)",
            read_only=True,
            expand=True,
        )
        self.token_vocabulary_row = ft.Row(
            controls=[
                self.token_vocabulary_path,
                ft.TextButton(
                    content="Choose vocabulary",
                    icon=ft.Icons.MENU_BOOK_ROUNDED,
                    on_click=self._on_choose_vocabulary,
                ),
            ],
            visible=False,
        )
        self.token_merges_row = ft.Row(
            controls=[
                self.token_merges_path,
                ft.TextButton(
                    content="Choose merges",
                    icon=ft.Icons.CALL_MERGE_ROUNDED,
                    on_click=self._on_choose_merges,
                ),
            ],
            visible=False,
        )
        self._set_columns(
            (
                self.token_sequence_length,
                self.token_layout,
                self.token_dtype,
                self.token_vocab_size,
            ),
            {"xs": 12, "sm": 6, "md": 3},
        )
        self._set_columns(
            (
                self.token_pad_id,
                self.token_bos_id,
                self.token_eos_id,
            ),
            {"xs": 12, "sm": 4},
        )
        self._refresh_token_choice()
        return ft.Container(
            content=ft.Column(
                controls=[
                    self.token_text,
                    self.token_codebook,
                    self.token_note,
                    self.token_vocabulary_row,
                    self.token_merges_row,
                    self._responsive_row(
                        self.token_sequence_length,
                        self.token_layout,
                        self.token_dtype,
                        self.token_vocab_size,
                    ),
                    self._responsive_row(
                        self.token_pad_id,
                        self.token_bos_id,
                        self.token_eos_id,
                    ),
                    ft.Row(controls=[self.token_truncate]),
                ],
                spacing=10,
            ),
            visible=False,
        )

    def _field(self, **kwargs: Any) -> ft.TextField:
        return self._watch_text_focus(ft.TextField(dense=True, **kwargs))

    @staticmethod
    def _dropdown(
        label: str,
        default: str,
        values: Sequence[str],
    ) -> ft.Dropdown:
        return ft.Dropdown(
            value=default,
            label=label,
            dense=True,
            options=[ft.DropdownOption(key=value, text=value) for value in values],
        )

    @staticmethod
    def _dtype_dropdown(default: str) -> ft.Dropdown:
        return TestInputWorkspace._dropdown(
            "Dtype",
            default,
            (
                "bool",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "int8",
                "int16",
                "int32",
                "int64",
                "float16",
                "float32",
                "float64",
            ),
        )

    @staticmethod
    def _set_columns(
        controls: Sequence[ft.Control],
        columns: Any,
    ) -> None:
        for control in controls:
            control.col = columns

    @staticmethod
    def _responsive_row(*controls: ft.Control) -> ft.ResponsiveRow:
        return ft.ResponsiveRow(
            controls=list(controls),
            spacing=8,
            run_spacing=8,
        )

    def _on_kind_changed(self, event: ft.Event[ft.Dropdown] | None) -> None:
        selected = self.kind.value or "image"
        for key, section in zip(
            ("image", "mask", "text-tokens", "csv", "time-series", "tensor"),
            self.sections,
            strict=True,
        ):
            section.visible = key == selected
        if event is not None:
            self.page.update()

    def _on_codebook_changed(self, event: ft.Event[ft.Dropdown] | None) -> None:
        self._refresh_token_choice()
        if event is not None:
            self.page.update()

    def _token_choice(self) -> TokenizerChoice:
        """The selected codebook entry, defaulting to the first choice."""
        return choice_for(self.token_codebook.value or TOKENIZER_CHOICES[0].id)

    def _refresh_token_choice(self) -> None:
        """Show the selected codebook's note and only the files it needs."""
        choice = self._token_choice()
        self.token_note.value = choice.note
        self.token_vocabulary_row.visible = choice.needs_vocabulary
        self.token_merges_row.visible = choice.needs_merges
        self.token_vocabulary_path.label = (
            "Vocabulary (tokenizer.json)"
            if choice.id == "tokenizer-json"
            else "Vocabulary (vocab.json)"
        )
        # A file-backed codebook takes its size from the vocabulary, so the
        # field would be a lie rather than a setting.
        self.token_vocab_size.disabled = choice.needs_files
        self.token_vocab_size.hint_text = (
            "Set by the vocabulary file"
            if choice.needs_files
            else "Blank uses the codebook default"
        )

    async def _on_choose_image(
        self,
        event: ft.Event[ft.TextButton],
    ) -> None:
        files = await self.picker.pick_files(
            dialog_title="Choose an image to convert to a NumPy tensor",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=[
                "png",
                "jpg",
                "jpeg",
                "webp",
                "bmp",
                "tif",
                "tiff",
            ],
            allow_multiple=False,
            with_data=False,
        )
        if files and files[0].path is not None:
            self.image_path.value = files[0].path
            self.status.value = f"Selected {files[0].name}"
            self.page.update()
        elif files:
            self._on_error("Image conversion is available in the desktop app.")
            self.page.update()

    async def _on_choose_csv(
        self,
        event: ft.Event[ft.TextButton],
    ) -> None:
        files = await self.picker.pick_files(
            dialog_title="Choose numeric CSV, TSV, or text data",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["csv", "tsv", "txt"],
            allow_multiple=False,
            with_data=False,
        )
        if files and files[0].path is not None:
            self.csv_path.value = files[0].path
            self.status.value = f"Selected {files[0].name}"
            self.page.update()
        elif files:
            self._on_error("CSV conversion is available in the desktop app.")
            self.page.update()

    async def _on_choose_vocabulary(
        self,
        event: ft.Event[ft.TextButton],
    ) -> None:
        files = await self.picker.pick_files(
            dialog_title="Choose a vocab.json or tokenizer.json vocabulary",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["json"],
            allow_multiple=False,
            with_data=False,
        )
        if files and files[0].path is not None:
            self.token_vocabulary_path.value = files[0].path
            self.status.value = f"Selected {files[0].name}"
            self.page.update()
        elif files:
            self._on_error("Vocabulary files are read in the desktop app.")
            self.page.update()

    async def _on_choose_merges(
        self,
        event: ft.Event[ft.TextButton],
    ) -> None:
        files = await self.picker.pick_files(
            dialog_title="Choose a merges.txt merge table",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["txt"],
            allow_multiple=False,
            with_data=False,
        )
        if files and files[0].path is not None:
            self.token_merges_path.value = files[0].path
            self.status.value = f"Selected {files[0].name}"
            self.page.update()
        elif files:
            self._on_error("Merge files are read in the desktop app.")
            self.page.update()

    async def _on_generate(self, event: ft.Event[ft.Button]) -> None:
        await self.generate(assign=False)

    async def _on_generate_and_assign(self, event: ft.Event[ft.Button]) -> None:
        await self.generate(assign=True)

    async def generate(self, *, assign: bool) -> None:
        """Prompt for a destination, publish, and optionally bind the tensor."""
        if assign and self.target_input.value is None:
            self._on_error("Open a model and select a target input first.")
            self.page.update()
            return
        destination = await self.picker.save_file(
            dialog_title="Save generated test input",
            file_name=self.generated_input_name(),
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["npy"],
        )
        if destination is None:
            return
        self.generate_button.disabled = True
        self.generate_and_assign_button.disabled = True
        self.status.value = "Generating and validating tensor…"
        self.page.update()
        generated: GeneratedTensor | None = None
        try:
            generated = await asyncio.to_thread(
                self.generate_current_input,
                Path(destination),
            )
            if assign:
                input_name = self.target_input.value
                assert input_name is not None
                self._assign(input_name, generated)
        except (
            InputGenerationError,
            TokenizationError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            if generated is None:
                self._on_error(f"Could not generate test input: {error}")
                self.status.value = "Generation failed; no partial file remains."
            else:
                self._on_error(
                    f"Saved {generated.path.name}, but could not assign it: {error}"
                )
                self.status.value = f"Saved {generated.path.name}; assignment failed."
        else:
            assert generated is not None
            self._clear_error()
            self.status.value = (
                f"Saved and assigned {generated.path.name}"
                if assign
                else f"Saved {generated.path.name}"
            )
            self.summary.value = (
                f"{generated.path}\n"
                f"shape={generated.shape} · dtype={generated.dtype} · "
                f"{generated.byte_size:,} bytes\n"
                f"source={generated.source}"
            )
            self._on_status(str(self.status.value))
        finally:
            self.generate_button.disabled = False
            self._refresh_assign_button()
            self.page.update()

    def generate_current_input(self, destination: Path) -> GeneratedTensor:
        """Create the tensor described by the currently visible form."""
        kind = self.kind.value or "image"
        if kind == "image":
            return generate_image_tensor(
                self._required_text(self.image_path, "choose an image file"),
                destination,
                width=self._integer_field(self.image_width, "image width"),
                height=self._integer_field(self.image_height, "image height"),
                layout=self.image_layout.value or "NCHW",  # type: ignore[arg-type]
                color_mode=self.image_color.value or "rgb",  # type: ignore[arg-type]
                normalization=self.image_normalization.value  # type: ignore[arg-type]
                or "zero-one",
                dtype=self.image_dtype.value or "float32",
            )
        if kind == "mask":
            return generate_mask_tensor(
                destination,
                shape=parse_shape(self.mask_shape.value or ""),
                fill=self.mask_fill.value or "ones",  # type: ignore[arg-type]
                dtype=self.mask_dtype.value or "int64",
                seed=self._integer_field(self.mask_seed, "mask seed"),
            )
        if kind == "text-tokens":
            return generate_token_ids_tensor(
                destination,
                # Empty text is the generator's business: a bos/eos id alone is
                # a valid one-token sequence, and it explains the refusal when
                # neither is set.
                text=self.token_text.value or "",
                codebook=self._token_codebook(),
                sequence_length=self._optional_integer_field(
                    self.token_sequence_length,
                    "sequence length",
                ),
                layout=self.token_layout.value or "NT",  # type: ignore[arg-type]
                dtype=self.token_dtype.value or "int64",
                pad_id=self._integer_field(self.token_pad_id, "pad id"),
                bos_id=self._optional_integer_field(self.token_bos_id, "BOS id"),
                eos_id=self._optional_integer_field(self.token_eos_id, "EOS id"),
                truncate=bool(self.token_truncate.value),
            )
        if kind == "csv":
            return generate_csv_tensor(
                self._required_text(
                    self.csv_path,
                    "choose a CSV, TSV, or text file",
                ),
                destination,
                delimiter=self.csv_delimiter.value or ",",
                skip_rows=self._integer_field(
                    self.csv_skip_rows,
                    "rows to skip",
                ),
                columns=parse_columns(self.csv_columns.value or ""),
                transpose=bool(self.csv_transpose.value),
                add_batch=bool(self.csv_add_batch.value),
                dtype=self.csv_dtype.value or "float32",
            )
        if kind == "time-series":
            return generate_time_series_tensor(
                destination,
                samples=self._integer_field(
                    self.series_samples,
                    "sample count",
                ),
                channels=self._integer_field(
                    self.series_channels,
                    "channel count",
                ),
                waveform=self.series_waveform.value or "sine",  # type: ignore[arg-type]
                sample_rate=self._float_field(
                    self.series_sample_rate,
                    "sample rate",
                ),
                frequency=self._float_field(
                    self.series_frequency,
                    "frequency",
                ),
                amplitude=self._float_field(
                    self.series_amplitude,
                    "amplitude",
                ),
                layout=self.series_layout.value or "NTC",  # type: ignore[arg-type]
                dtype=self.series_dtype.value or "float32",
                seed=self._integer_field(
                    self.series_seed,
                    "time-series seed",
                ),
            )
        return generate_synthetic_tensor(
            destination,
            shape=parse_shape(self.tensor_shape.value or ""),
            distribution=self.tensor_distribution.value  # type: ignore[arg-type]
            or "zeros",
            dtype=self.tensor_dtype.value or "float32",
            seed=self._integer_field(self.tensor_seed, "tensor seed"),
        )

    def _token_codebook(self) -> Codebook:
        """Build the selected codebook, refusing before a needed file is read."""
        choice = self._token_choice()
        vocabulary_path = (
            self._required_text(
                self.token_vocabulary_path,
                f"choose a vocabulary file for {choice.label}",
            )
            if choice.needs_vocabulary
            else None
        )
        merges_path = (
            self._required_text(
                self.token_merges_path,
                f"choose a merges file for {choice.label}",
            )
            if choice.needs_merges
            else None
        )
        return load_codebook(
            choice.id,
            vocabulary_path=vocabulary_path,
            merges_path=merges_path,
            vocab_size=(
                None
                if choice.needs_files
                else self._optional_integer_field(
                    self.token_vocab_size,
                    "vocabulary size",
                )
            ),
        )

    def _on_target_changed(self, event: ft.Event[ft.Dropdown] | None) -> None:
        self._apply_target_defaults()
        self._refresh_assign_button()
        if event is not None:
            self.page.update()

    def _apply_target_defaults(self) -> None:
        target_name = self.target_input.value
        target = self._targets.get(target_name or "")
        if target is None:
            return
        supported_dtypes = {str(option.key) for option in self.tensor_dtype.options}
        if target.element_type in supported_dtypes:
            for dtype_field in (
                self.image_dtype,
                self.mask_dtype,
                self.csv_dtype,
                self.series_dtype,
                self.tensor_dtype,
            ):
                dtype_field.value = target.element_type
        # Token ids only take the integer dtypes the token generator accepts,
        # so a float target must leave the token dtype alone.
        token_dtypes = {str(option.key) for option in self.token_dtype.options}
        if target.element_type in token_dtypes:
            self.token_dtype.value = target.element_type
        if target.shape is not None and all(
            isinstance(extent, int) for extent in target.shape
        ):
            shape = tuple(extent for extent in target.shape if isinstance(extent, int))
            shape_text = ",".join(str(extent) for extent in shape)
            self.mask_shape.value = shape_text
            self.tensor_shape.value = shape_text
            if len(shape) == 4 and shape[1] in {1, 3}:
                self.kind.value = "image"
                self.image_layout.value = "NCHW"
                self.image_color.value = "rgb" if shape[1] == 3 else "grayscale"
                self.image_height.value = str(shape[2])
                self.image_width.value = str(shape[3])
            elif len(shape) == 4 and shape[3] in {1, 3}:
                self.kind.value = "image"
                self.image_layout.value = "NHWC"
                self.image_color.value = "rgb" if shape[3] == 3 else "grayscale"
                self.image_height.value = str(shape[1])
                self.image_width.value = str(shape[2])
            elif len(shape) == 3 and shape[0] in {1, 3}:
                self.kind.value = "image"
                self.image_layout.value = "CHW"
                self.image_color.value = "rgb" if shape[0] == 3 else "grayscale"
                self.image_height.value = str(shape[1])
                self.image_width.value = str(shape[2])
            elif len(shape) == 3 and shape[2] in {1, 3}:
                self.kind.value = "image"
                self.image_layout.value = "HWC"
                self.image_color.value = "rgb" if shape[2] == 3 else "grayscale"
                self.image_height.value = str(shape[0])
                self.image_width.value = str(shape[1])
            elif (
                len(shape) == 2
                and shape[1] == 1536
                and shape[0] % 4 == 0
                and "pixel" in target.name.lower()
            ):
                grid_height, grid_width = self._closest_landscape_patch_grid(shape[0])
                self.kind.value = "image"
                self.image_layout.value = "QWEN3VL_PATCHES"
                self.image_color.value = "rgb"
                self.image_normalization.value = "minus-one-one"
                self.image_height.value = str(grid_height * 16)
                self.image_width.value = str(grid_width * 16)
                self._on_kind_changed(None)
            elif len(shape) == 2 and target.element_type in token_dtypes:
                # A rank-2 integer input is a batch of token ids, not a picture.
                self.kind.value = "text-tokens"
                self.token_sequence_length.value = str(shape[1])
                self._on_kind_changed(None)
            elif len(shape) == 2:
                self.kind.value = "image"
                self.image_layout.value = "HW"
                self.image_color.value = "grayscale"
                self.image_height.value = str(shape[0])
                self.image_width.value = str(shape[1])
            if self.kind.value == "image":
                self._on_kind_changed(None)
        if target.is_mask:
            self.kind.value = "mask"
            self._on_kind_changed(None)

    @staticmethod
    def _closest_landscape_patch_grid(patch_count: int) -> tuple[int, int]:
        """Choose the least-distorted even grid for a fixed Qwen patch count."""
        candidates = [
            (height, patch_count // height)
            for height in range(2, int(patch_count**0.5) + 1, 2)
            if patch_count % height == 0 and (patch_count // height) % 2 == 0
        ]
        if not candidates:
            raise InputGenerationError(
                "Qwen3-VL patch count needs an even height and width"
            )
        return min(candidates, key=lambda item: item[1] / item[0])

    def _refresh_assign_button(self) -> None:
        self.generate_and_assign_button.disabled = (
            not self._can_assign or self.target_input.value is None
        )

    def generated_input_name(self) -> str:
        kind = self.kind.value or "tensor"
        if kind == "image" and self.image_path.value:
            return f"{Path(self.image_path.value).stem}.npy"
        if kind == "csv" and self.csv_path.value:
            return f"{Path(self.csv_path.value).stem}.npy"
        return {
            "mask": "mask.npy",
            "text-tokens": "tokens.npy",
            "time-series": "time-series.npy",
            "tensor": "tensor.npy",
        }.get(kind, "input.npy")

    @staticmethod
    def _required_text(field: ft.TextField, message: str) -> str:
        value = (field.value or "").strip()
        if not value:
            raise InputGenerationError(message)
        return value

    @staticmethod
    def _integer_field(field: ft.TextField, label: str) -> int:
        try:
            return int((field.value or "").strip(), 10)
        except ValueError as error:
            raise InputGenerationError(f"{label} must be an integer") from error

    @staticmethod
    def _optional_integer_field(field: ft.TextField, label: str) -> int | None:
        """An integer field whose blank value means "let the generator decide"."""
        if not (field.value or "").strip():
            return None
        return TestInputWorkspace._integer_field(field, label)

    @staticmethod
    def _float_field(field: ft.TextField, label: str) -> float:
        try:
            return float((field.value or "").strip())
        except ValueError as error:
            raise InputGenerationError(f"{label} must be a number") from error


def build_input_workspace(
    *,
    palette: ShellPalette,
    kind: ft.Dropdown,
    target_input: ft.Dropdown,
    sections: tuple[ft.Container, ...],
    generate_button: ft.FilledButton,
    generate_and_assign_button: ft.FilledButton,
    status: ft.Text,
    summary: ft.Text,
) -> ft.Container:
    """Build the standalone generator workspace from stateful controls."""
    configuration = ft.Container(
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Container(
                            content=ft.Icon(
                                ft.Icons.SCIENCE_ROUNDED,
                                size=24,
                                color=palette.accent,
                            ),
                            width=46,
                            height=46,
                            alignment=ft.Alignment.CENTER,
                            bgcolor=palette.accent_soft,
                            border_radius=13,
                        ),
                        ft.Column(
                            controls=[
                                ft.Text(
                                    "Generate test input",
                                    size=21,
                                    weight=ft.FontWeight.W_700,
                                    color=palette.ink,
                                ),
                                ft.Text(
                                    "Create bounded, pickle-free .npy tensors "
                                    "for tracing and model inputs.",
                                    size=11,
                                    color=palette.muted,
                                ),
                            ],
                            spacing=1,
                            expand=True,
                        ),
                    ],
                    spacing=12,
                ),
                ft.Divider(height=20, color=palette.border),
                kind,
                *sections,
                ft.Divider(height=20, color=palette.border),
                target_input,
                ft.Row(
                    controls=[generate_button, generate_and_assign_button],
                    wrap=True,
                    spacing=8,
                ),
            ],
            spacing=12,
        ),
        col={"sm": 12, "lg": 7},
        padding=24,
        bgcolor=palette.panel,
        border=ft.Border.all(1, palette.border),
        border_radius=16,
    )
    guidance = ft.Container(
        content=ft.Column(
            controls=[
                ft.Text(
                    "Output",
                    size=16,
                    weight=ft.FontWeight.W_700,
                    color=palette.ink,
                ),
                status,
                ft.Container(
                    content=summary,
                    padding=14,
                    bgcolor=palette.canvas,
                    border=ft.Border.all(1, palette.border),
                    border_radius=12,
                ),
                ft.Divider(height=20, color=palette.border),
                ft.Text(
                    "Built for common model inputs",
                    size=13,
                    weight=ft.FontWeight.W_700,
                    color=palette.ink,
                ),
                _tip(
                    palette,
                    ft.Icons.IMAGE_ROUNDED,
                    "Images",
                    "Resize, convert to RGB or grayscale, choose tensor layout, "
                    "and apply optional ImageNet normalization.",
                ),
                _tip(
                    palette,
                    ft.Icons.MASKS_ROUNDED,
                    "Masks",
                    "Generate all-valid, zero, checkerboard, or deterministic "
                    "random binary masks with an explicit dtype and shape.",
                ),
                _tip(
                    palette,
                    ft.Icons.TEXT_FIELDS_ROUNDED,
                    "Text tokens",
                    "Encode a prompt into language-model token ids from a "
                    "self-contained codebook or a HuggingFace vocabulary; the "
                    "summary records how faithful the ids are.",
                ),
                _tip(
                    palette,
                    ft.Icons.TABLE_CHART_ROUNDED,
                    "CSV and time series",
                    "Select columns, skip headers, transpose or batch tabular "
                    "data, or synthesize repeatable multi-channel signals.",
                ),
                ft.Text(
                    "Saved tensors are validated by the selected model input "
                    "before assignment. A failed assignment never removes the "
                    "generated file.",
                    size=10,
                    color=palette.muted,
                ),
            ],
            spacing=12,
        ),
        col={"sm": 12, "lg": 5},
        padding=22,
        bgcolor=palette.panel,
        border=ft.Border.all(1, palette.border),
        border_radius=16,
    )
    return ft.Container(
        content=ft.ListView(
            controls=[
                ft.ResponsiveRow(
                    controls=[configuration, guidance],
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    spacing=18,
                    run_spacing=18,
                )
            ],
            expand=True,
            padding=24,
        ),
        expand=True,
        bgcolor=palette.canvas,
    )


def _tip(
    palette: ShellPalette,
    icon: ft.IconData,
    title: str,
    description: str,
) -> ft.Row:
    return ft.Row(
        controls=[
            ft.Icon(icon, size=19, color=palette.accent),
            ft.Column(
                controls=[
                    ft.Text(
                        title,
                        size=11,
                        weight=ft.FontWeight.W_700,
                        color=palette.ink,
                    ),
                    ft.Text(description, size=10, color=palette.muted),
                ],
                spacing=1,
                expand=True,
            ),
        ],
        vertical_alignment=ft.CrossAxisAlignment.START,
        spacing=9,
    )
