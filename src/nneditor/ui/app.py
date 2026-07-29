"""The Flet application shell.

The workspace is deliberately architecture-first: model and block metadata live
in the left context panel, the graph owns the middle of the screen, and model
navigation lives in the right explorer. Editing tools use progressive
disclosure so the normal viewing path stays calm.

The shell owns *view* state only — which session is shown, the viewport, the
selection. Model state lives in the application service, and everything the
shell prints comes from `nneditor.ui.viewmodel`, so this module is wiring:
events in, service calls out, controls updated.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import math
import sys
import tempfile
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import flet as ft
import flet.canvas as cv

from nneditor.analysis.lod import DetailLevel, detail_for_scale
from nneditor.analysis.statistics import TensorStatistics
from nneditor.application.editing import SidecarPersistenceError
from nneditor.application.hierarchy import OrganizationMode
from nneditor.application.jobs import Job
from nneditor.application.navigation import Direction, MiniMap
from nneditor.application.persistence import SessionStateStore, ViewState
from nneditor.application.session import ApplicationService, ExportOutcome, ModelSession
from nneditor.application.slices import GraphSlice
from nneditor.artifact_formats import MODEL_FILE_EXTENSIONS
from nneditor.desktop import (
    FileAssociationError,
    open_default_apps_settings,
    register_file_associations,
    unregister_file_associations,
)
from nneditor.editing.validation import (
    EditRequest,
    EditTransaction,
    InsertUnaryRequest,
    ReconnectInputRequest,
    RemoveUnaryRequest,
    RenameNodeRequest,
    ReplaceOperatorRequest,
    SetAttributeRequest,
    parse_attribute_value,
)
from nneditor.input_generation import GeneratedTensor
from nneditor.ir.capabilities import ArtifactKind, Availability, Capability
from nneditor.ir.core import AttrKind, Storage
from nneditor.rendering import create_flet_renderer
from nneditor.rendering.contract import InteractiveGraphRenderer, RendererFactory
from nneditor.rendering.scene import Scene, Viewport
from nneditor.tracing.comparison import TraceComparison
from nneditor.tracing.contracts import (
    ActivationRecord,
    InputBinding,
    TraceApproval,
    TraceLimits,
    TraceRequest,
    TraceResult,
)
from nneditor.tracing.runner import recommended_trace_limits
from nneditor.tracing.visualization import ActivationVisualization
from nneditor.transformations.engine import (
    TransformationProposal,
    TransformationRequest,
)
from nneditor.transformations.schema import Granularity
from nneditor.ui import input_workspace, overview, shell_layout, tensor_tools, viewmodel
from nneditor.ui.activation_layers import build_activation_layer_viewer
from nneditor.ui.trace_graph import TraceGraphPresentation, build_trace_graph

APP_TITLE = "NNEditor"
APP_ASSETS_DIRECTORY = Path(__file__).resolve().parents[1] / "assets"
APP_ICON_PATH = APP_ASSETS_DIRECTORY / "nneditor.png"
APP_WINDOW_ICON_PATH = APP_ASSETS_DIRECTORY / "nneditor.ico"
_WEB_UPLOAD_TEMP = tempfile.TemporaryDirectory(prefix="nneditor-web-upload-")
_WEB_UPLOAD_DIRECTORY = Path(_WEB_UPLOAD_TEMP.name)


def _uses_automatic_mask(input_name: str) -> bool:
    normalized = input_name.lower().replace("-", "_")
    return normalized == "mask" or normalized.endswith("_mask")


_SURFACE_FALLBACK = (1200.0, 800.0)
_NARROW_BREAKPOINT = 1280.0
_ACCENT = "#5B5CE2"
_ACCENT_SOFT = "#EEEFFD"

# The minimap draws at 220x110; beyond this many dots each is sub-pixel and
# every extra control only inflates the per-update diff.
_MINIMAP_MAX_DOTS = 600
_INK = "#172033"
_MUTED = "#667085"
_BORDER = "#E4E7EC"
_PANEL = "#FFFFFF"
_CANVAS = "#F6F7FB"
_SUBTLE = "#F8FAFC"
_WARNING = "#B54708"
_WARNING_SOFT = "#FFFAEB"
_DANGER = "#B42318"
_DANGER_SOFT = "#FEF3F2"
_INFO = "#175CD3"
_SIDEBAR_WIDTH = 304
_MAX_EXPLORER_ROWS = 200
_SHELL_PALETTE = shell_layout.ShellPalette(
    panel=_PANEL,
    border=_BORDER,
    accent=_ACCENT,
    accent_soft=_ACCENT_SOFT,
    ink=_INK,
    muted=_MUTED,
    canvas=_CANVAS,
    sidebar_width=_SIDEBAR_WIDTH,
)


class Shell:
    """One page's worth of UI state and wiring."""

    def __init__(
        self,
        page: ft.Page,
        service: ApplicationService,
        *,
        renderer_factory: RendererFactory = create_flet_renderer,
    ) -> None:
        self.page = page
        self.service = service
        self.session: ModelSession | None = None
        self.surface_size = _SURFACE_FALLBACK
        self.view = {"scale": 1.0, "x": 0.0, "y": 0.0}
        self.current_graph: str | None = None
        self.current_root_group: str | None = None
        self.current_detail = DetailLevel.ARCHITECTURE
        self.auto_detail = False
        self.current_slice: GraphSlice | None = None
        self.logical_selection: frozenset[str] = frozenset()
        self.minimap_model: MiniMap | None = None
        self._typed_preview_consent: set[str] = set()
        self._hex_offsets: dict[str, int] = {}
        self._hex_drafts: dict[tuple[str, int], str] = {}
        self._hex_errors: dict[str, str] = {}
        # Expansion state per tensor card, so a hex-pager or statistics
        # interaction does not re-collapse every card the user opened.
        self._tensor_card_expanded: dict[str, bool] = {}
        # The ids the inspector last rendered; async completions consult this
        # so a stale job cannot clobber whatever the user looks at now.
        self._inspected_ids: frozenset[str] = frozenset()
        # True while any TextField owns the caret; global arrow-key
        # navigation must not steal keystrokes from a text input.
        self._text_input_active = False
        self.pending_edit: EditTransaction | None = None
        self.pending_transformation: TransformationProposal | None = None
        self.open_job: Job[ModelSession] | None = None
        self.export_job: Job[ExportOutcome] | None = None
        self.trace_job: Job[TraceResult] | None = None
        self._trace_preparing = False
        # The revision represented by the last successful export in this
        # shell. None is the immutable source artifact; recovered sidecar
        # revisions therefore reopen as unsaved and offer an explicit save.
        self._saved_revision_id: str | None = None
        self.active_trace_id: str | None = None
        self.active_trace_comparison: TraceComparison | None = None
        self._activation_statistics: dict[tuple[str, str], TensorStatistics] = {}
        self._activation_visualizations: dict[
            tuple[str, str], tuple[ActivationVisualization, ...]
        ] = {}
        self._activation_visualization_errors: dict[tuple[str, str], str] = {}
        self._activation_loading: set[tuple[str, str, str]] = set()
        self._trace_input_bindings: dict[str, InputBinding] = {}
        self._trace_graph_presentation: TraceGraphPresentation | None = None
        self._pending_web_upload: tuple[Path, str] | None = None
        # The node a pending edit/transformation was built from. A pending
        # transaction that outlives its selection would apply to whatever the
        # user happened to click next, so selection changes discard it.
        self.pending_target: str | None = None
        # None means "follow the window width"; True/False is an explicit
        # user choice that survives resizes.
        self.left_panel_override: bool | None = None
        self.right_panel_override: bool | None = None

        self.renderer: InteractiveGraphRenderer = renderer_factory(self._on_selected)
        renderer_control = self.renderer.control
        if not isinstance(renderer_control, ft.Control):
            raise TypeError("the Flet renderer factory returned no Flet control")
        self.renderer_control = renderer_control
        detector = renderer_control
        assert isinstance(detector, ft.GestureDetector)
        detector.on_pan_update = self._on_pan
        detector.on_scroll = self._on_scroll
        detector.on_hover = self._on_surface_hover
        detector.on_exit = self._on_surface_exit
        detector.on_double_tap = self._on_surface_double_tap
        detector.hover_interval = 48
        # Unthrottled drag events (the Flet default) arrive at pointer rate
        # and each one costs a canvas patch; 16 ms keeps them at frame rate.
        detector.drag_interval = 16
        # Gesture coalescing: handlers record the target viewport and a single
        # in-flight drain applies the latest one, so a burst of events costs
        # one render instead of queueing a render per event.
        self._viewport_dirty = False
        self._viewport_flush: Future[None] | None = None
        # The minimap's persistent viewport rectangle; dots rebuild only when
        # the scene changes, this rectangle mutates on every pan/zoom.
        self._minimap_view_rect: cv.Rect | None = None

        self.picker = ft.FilePicker(on_upload=self._on_upload_progress)
        # FilePicker is a Service, not a Control, in Flet 0.86: registering it
        # in `page.overlay` makes the client reject it with "Unknown control:
        # FilePicker" the moment the page renders.
        page.services.append(self.picker)

        self.title_text = ft.Text(
            APP_TITLE,
            size=17,
            color=_INK,
            weight=ft.FontWeight.W_700,
        )
        self.model_subtitle = ft.Text("Neural network explorer", size=11, color=_MUTED)
        self.job_text = ft.Text("", size=11, color=_MUTED)
        self.save_model_button = ft.FilledButton(
            content="Save changesâ€¦",
            icon=ft.Icons.SAVE_ROUNDED,
            tooltip="Export the current revision to a new artifact",
            on_click=self._on_save_model,
            visible=False,
        )
        self.close_model_button = ft.TextButton(
            content="Close model",
            icon=ft.Icons.CLOSE_ROUNDED,
            tooltip="Close the current model and return to the start screen",
            on_click=self._on_close_model,
            visible=False,
        )
        self.left_toggle = ft.TextButton(
            content="Hide panels",
            icon=ft.Icons.INFO_OUTLINE_ROUNDED,
            tooltip="Toggle details panel",
            on_click=self._on_toggle_left_panel,
        )
        self.right_toggle = ft.TextButton(
            content="Hide inspector",
            icon=ft.Icons.ACCOUNT_TREE_ROUNDED,
            tooltip="Toggle model explorer",
            on_click=self._on_toggle_right_panel,
        )
        self.status_text = ft.Text(
            "Open a supported model artifact to begin.", size=11, color=_MUTED
        )
        self.error_banner = ft.Container(
            content=ft.Text("", color="#FFFFFF"),
            bgcolor="#B42318",
            padding=ft.Padding.symmetric(horizontal=16, vertical=10),
            visible=False,
        )
        self.graph_list = ft.ListView(expand=True, spacing=4)
        self.hierarchy_list = ft.ListView(expand=True, spacing=4)
        self.multi_select_field = ft.Checkbox(
            label="Multi-select",
            on_change=self._on_multi_select_changed,
        )
        self.group_label_field = ft.TextField(
            label="Group label",
            dense=True,
        )
        self.group_button = ft.TextButton(
            content="Group", on_click=self._on_group_selected
        )
        self.merge_button = ft.TextButton(
            content="Merge", on_click=self._on_merge_selected
        )
        self.rename_button = ft.TextButton(
            content="Rename", on_click=self._on_rename_selected
        )
        self.split_button = ft.TextButton(
            content="Split", on_click=self._on_split_selected
        )
        self.lock_button = ft.TextButton(
            content="Lock/unlock", on_click=self._on_lock_selected
        )
        self.reject_button = ft.TextButton(
            content="Reject", on_click=self._on_reject_selected
        )
        self.reset_groups_button = ft.TextButton(
            content="Reset", on_click=self._on_reset_groups
        )
        self.breadcrumbs_row = ft.Row(
            spacing=0,
            scroll=ft.ScrollMode.AUTO,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.detail_segment = ft.SegmentedButton(
            segments=[
                ft.Segment(
                    value="auto",
                    icon=ft.Icons.AUTO_AWESOME_ROUNDED,
                    label="Auto",
                ),
                ft.Segment(
                    value=DetailLevel.ARCHITECTURE.value,
                    icon=ft.Icons.HUB_ROUNDED,
                    label="Architecture",
                ),
                ft.Segment(
                    value=DetailLevel.BLOCK.value,
                    icon=ft.Icons.GRID_VIEW_ROUNDED,
                    label="Blocks",
                ),
                ft.Segment(
                    value=DetailLevel.LAYER.value,
                    icon=ft.Icons.LAYERS_ROUNDED,
                    label="Layers",
                ),
                ft.Segment(
                    value=DetailLevel.OPERATOR.value,
                    icon=ft.Icons.DATA_OBJECT_ROUNDED,
                    label="Operators",
                ),
            ],
            selected=[DetailLevel.ARCHITECTURE.value],
            show_selected_icon=False,
            on_change=self._on_detail_segment_changed,
        )
        self.reset_view_button = ft.TextButton(
            content="Reset view",
            icon=ft.Icons.FIT_SCREEN_ROUNDED,
            tooltip="Fit the current architecture or block in the workspace",
            on_click=self._on_reset_view,
            disabled=True,
        )
        self.organize_button = ft.IconButton(
            icon=ft.Icons.CATEGORY_ROUNDED,
            tooltip=self._organize_tooltip(OrganizationMode.AUTO),
            on_click=self._on_cycle_organization,
        )
        self.minimap_canvas = cv.Canvas(width=220, height=110)
        self.minimap = ft.GestureDetector(
            content=self.minimap_canvas,
            on_tap_down=self._on_minimap_tap,
        )
        self.search_field = ft.TextField(
            hint_text="Find a node or operator",
            prefix_icon=ft.Icons.SEARCH_ROUNDED,
            dense=True,
            on_submit=self._on_search,
        )
        self.search_results = ft.ListView(expand=True, spacing=0)
        self.inspector = ft.ListView(expand=True, spacing=14)
        self.inspector_title = ft.Text(
            "Model overview", size=17, weight=ft.FontWeight.W_700, color=_INK
        )
        self.inspector_subtitle = ft.Text(
            "Select a block to inspect it", size=11, color=_MUTED
        )
        self.open_selection_button = ft.FilledButton(
            content="Open block",
            icon=ft.Icons.OPEN_IN_FULL_ROUNDED,
            on_click=self._on_open_selected,
            visible=False,
        )
        self.back_to_parent_button = ft.TextButton(
            content="Up one level",
            icon=ft.Icons.ARROW_BACK_ROUNDED,
            on_click=self._on_back_to_parent,
            visible=False,
        )
        self.hover_title = ft.Text("", size=13, weight=ft.FontWeight.W_700, color=_INK)
        self.hover_summary = ft.Text("", size=11, color=_MUTED, max_lines=3)
        self.hover_card = ft.Container(
            content=ft.Column(
                controls=[self.hover_title, self.hover_summary],
                tight=True,
                spacing=3,
            ),
            width=260,
            padding=12,
            bgcolor=_PANEL,
            border=ft.Border.all(1, _BORDER),
            border_radius=12,
            shadow=ft.BoxShadow(
                blur_radius=24,
                color="#240F172A",
                offset=ft.Offset(0, 8),
            ),
            visible=False,
            left=16,
            top=16,
        )
        self.graph_trace_actions = ft.Stack(controls=[], expand=True)
        self._hovered_id: str | None = None
        self.edit_kind = ft.Dropdown(
            value="rename",
            label="Edit command",
            dense=True,
            options=[
                ft.DropdownOption(key="rename", text="Rename node"),
                ft.DropdownOption(key="attribute", text="Set attribute"),
                ft.DropdownOption(key="operator", text="Replace operator"),
                ft.DropdownOption(key="insert", text="Insert unary"),
                ft.DropdownOption(key="remove", text="Remove unary"),
                ft.DropdownOption(key="reconnect", text="Reconnect input"),
            ],
        )
        self.edit_primary = ft.TextField(label="Name / operator / value ID", dense=True)
        self.edit_secondary = ft.TextField(label="Value / domain", dense=True)
        self.edit_attribute_kind = ft.Dropdown(
            value=AttrKind.STRING.value,
            label="Attribute type",
            dense=True,
            options=[
                ft.DropdownOption(key=kind.value, text=kind.value)
                for kind in (
                    AttrKind.INT,
                    AttrKind.FLOAT,
                    AttrKind.STRING,
                    AttrKind.INTS,
                    AttrKind.FLOATS,
                    AttrKind.STRINGS,
                )
            ],
        )
        self.edit_port = ft.TextField(label="Input port", value="0", dense=True)
        self.validate_edit_button = ft.FilledButton(
            content="Validate",
            on_click=self._on_validate_edit,
            disabled=True,
        )
        self.commit_edit_button = ft.FilledButton(
            content="Commit",
            on_click=self._on_commit_edit,
            disabled=True,
        )
        self.reject_edit_button = ft.TextButton(
            content="Reject",
            on_click=self._on_reject_edit,
            disabled=True,
        )
        self.undo_edit_button = ft.TextButton(
            content="Undo",
            on_click=self._on_undo_edit,
            disabled=True,
        )
        self.redo_edit_button = ft.TextButton(
            content="Redo",
            on_click=self._on_redo_edit,
            disabled=True,
        )
        self.export_button = ft.TextButton(
            content="Export…",
            on_click=self._on_export_clicked,
            disabled=True,
        )
        self.edit_findings = ft.ListView(height=90, spacing=1)
        self.transformation_kind = ft.Dropdown(
            value="weight-quantization",
            label="Transformation",
            dense=True,
            options=[
                ft.DropdownOption(
                    key="weight-quantization",
                    text="8-bit dequantized weight",
                ),
                ft.DropdownOption(key="graph-quantization", text="ONNX Q/DQ"),
                ft.DropdownOption(key="threshold-pruning", text="Threshold pruning"),
                ft.DropdownOption(key="mask-pruning", text="Explicit mask pruning"),
                ft.DropdownOption(key="nm-pruning", text="2:4 logical pruning"),
                ft.DropdownOption(
                    key="structured-pruning",
                    text="Terminal MatMul channels",
                ),
            ],
        )
        self.transformation_granularity = ft.Dropdown(
            value=Granularity.PER_TENSOR.value,
            label="Quantization granularity",
            dense=True,
            options=[
                ft.DropdownOption(
                    key=Granularity.PER_TENSOR.value,
                    text="Per tensor",
                ),
                ft.DropdownOption(
                    key=Granularity.PER_CHANNEL.value,
                    text="Per channel",
                ),
            ],
        )
        self.transformation_axis = ft.TextField(
            label="Channel axis",
            value="0",
            dense=True,
        )
        self.transformation_parameter = ft.TextField(
            label="Threshold / mask / kept channels",
            value="0.1",
            dense=True,
        )
        self.preview_transformation_button = ft.FilledButton(
            content="Preview",
            on_click=self._on_preview_transformation,
            disabled=True,
        )
        self.commit_transformation_button = ft.FilledButton(
            content="Apply",
            on_click=self._on_commit_transformation,
            disabled=True,
        )
        self.reject_transformation_button = ft.TextButton(
            content="Reject",
            on_click=self._on_reject_transformation,
            disabled=True,
        )
        self.transformation_findings = ft.ListView(height=125, spacing=1)
        self.trace_seed = ft.TextField(
            label="Deterministic seed",
            value="0",
            dense=True,
        )
        self.trace_shapes = ft.TextField(
            label="Symbolic shapes (input=2x3; ...)",
            dense=True,
        )
        self.trace_wall_seconds = ft.TextField(
            label="Wall limit (seconds)",
            value="30",
            dense=True,
        )
        self.trace_memory_mib = ft.TextField(
            label="Memory limit (MiB)",
            value="2048",
            dense=True,
        )
        self.trace_capture_mib = ft.TextField(
            label="Capture limit (MiB)",
            value="256",
            dense=True,
        )
        self.trace_chunk_kib = ft.TextField(
            label="Write chunk (KiB)",
            value="1024",
            dense=True,
        )
        self.trace_approval_notice = ft.Text(
            "Review the selected inputs and limits. The run button approves "
            "exactly one isolated trace.",
            size=10,
            color=_MUTED,
        )
        self.run_trace_button = ft.FilledButton(
            content="Approve & run trace",
            icon=ft.Icons.PLAY_ARROW_ROUNDED,
            on_click=self._on_run_trace,
            disabled=True,
        )
        self.cancel_trace_button = ft.TextButton(
            content="Cancel",
            on_click=self._on_cancel_trace,
            disabled=True,
        )
        self.trace_compare_with = ft.Dropdown(
            label="Compare active trace with",
            dense=True,
            options=[],
            disabled=True,
        )
        self.compare_trace_button = ft.TextButton(
            content="Compare traces",
            icon=ft.Icons.COMPARE_ARROWS_ROUNDED,
            on_click=self._on_compare_trace,
            disabled=True,
        )
        self.trace_status = ft.Text(
            "Open an artifact to see tracing availability.",
            size=10,
            color=_MUTED,
            expand=True,
        )
        self.trace_progress = ft.ProgressRing(
            width=18,
            height=18,
            stroke_width=2,
            visible=False,
        )
        self.file_types_button = ft.TextButton(
            content="File types",
            icon=ft.Icons.SETTINGS_APPLICATIONS_ROUNDED,
            tooltip="Register NNEditor in Windows Open with and Default apps",
            on_click=self._on_register_file_types,
            visible=sys.platform == "win32",
        )

        self.input_generator = input_workspace.TestInputWorkspace(
            page=page,
            picker=self.picker,
            palette=_SHELL_PALETTE,
            assign=self._assign_generated_input,
            on_error=self._show_error,
            on_status=self._set_status,
            clear_error=self._clear_error,
            watch_text_focus=self._watch_text_focus,
        )

        trace_controls = ft.Column(
            controls=[
                ft.Row(
                    controls=[self.trace_progress, self.trace_status],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                self.trace_seed,
                self.trace_shapes,
                ft.Row(
                    controls=[self.trace_wall_seconds, self.trace_memory_mib],
                    spacing=6,
                ),
                ft.Row(
                    controls=[self.trace_capture_mib, self.trace_chunk_kib],
                    spacing=6,
                ),
                self.trace_approval_notice,
                ft.Row(
                    controls=[self.run_trace_button, self.cancel_trace_button],
                    wrap=True,
                    spacing=4,
                ),
                self.trace_compare_with,
                self.compare_trace_button,
            ],
            spacing=8,
        )

        edit_controls = ft.Column(
            controls=[
                self.edit_kind,
                self.edit_primary,
                self.edit_secondary,
                ft.Row(
                    controls=[self.edit_attribute_kind, self.edit_port],
                    spacing=6,
                ),
                ft.Row(
                    controls=[
                        self.validate_edit_button,
                        self.commit_edit_button,
                        self.reject_edit_button,
                    ],
                    wrap=True,
                    spacing=4,
                ),
                ft.Row(
                    controls=[
                        self.undo_edit_button,
                        self.redo_edit_button,
                        self.export_button,
                    ],
                    wrap=True,
                    spacing=2,
                ),
                self.edit_findings,
            ],
            spacing=8,
        )
        transformation_controls = ft.Column(
            controls=[
                self.transformation_kind,
                self.transformation_granularity,
                ft.Row(
                    controls=[
                        self.transformation_axis,
                        self.transformation_parameter,
                    ],
                    spacing=6,
                ),
                ft.Row(
                    controls=[
                        self.preview_transformation_button,
                        self.commit_transformation_button,
                        self.reject_transformation_button,
                    ],
                    wrap=True,
                    spacing=4,
                ),
                self.transformation_findings,
            ],
            spacing=8,
        )
        self.left_panel = shell_layout.build_left_panel(
            palette=_SHELL_PALETTE,
            inspector_title=self.inspector_title,
            inspector_subtitle=self.inspector_subtitle,
            open_selection_button=self.open_selection_button,
            back_to_parent_button=self.back_to_parent_button,
            inspector=self.inspector,
            trace_controls=trace_controls,
            edit_controls=edit_controls,
            transformation_controls=transformation_controls,
        )
        hierarchy_tools = shell_layout.build_hierarchy_tools(
            multi_select_field=self.multi_select_field,
            group_label_field=self.group_label_field,
            group_button=self.group_button,
            merge_button=self.merge_button,
            rename_button=self.rename_button,
            split_button=self.split_button,
            lock_button=self.lock_button,
            reject_button=self.reject_button,
            reset_groups_button=self.reset_groups_button,
        )
        self.right_panel = shell_layout.build_right_panel(
            palette=_SHELL_PALETTE,
            search_field=self.search_field,
            search_results=self.search_results,
            graph_list=self.graph_list,
            hierarchy_list=self.hierarchy_list,
            hierarchy_tools=hierarchy_tools,
            minimap=self.minimap,
        )
        self.empty_state = shell_layout.build_empty_state(
            palette=_SHELL_PALETTE,
            on_open=self._on_open_clicked,
        )
        self.loading_title = ft.Text(
            "Preparing model",
            size=18,
            weight=ft.FontWeight.W_700,
            color=_INK,
        )
        self.loading_stage = ft.Text(
            "Indexing topology, detecting blocks, and laying out the architecture…",
            size=12,
            color=_MUTED,
            text_align=ft.TextAlign.CENTER,
        )
        self.cancel_open_button = ft.TextButton(
            content="Cancel",
            on_click=self._on_cancel_open,
        )
        self.loading_overlay = shell_layout.build_loading_overlay(
            palette=_SHELL_PALETTE,
            title=self.loading_title,
            stage=self.loading_stage,
            cancel_button=self.cancel_open_button,
        )
        self.surface = shell_layout.build_surface(
            palette=_SHELL_PALETTE,
            renderer_control=self.renderer_control,
            graph_actions=self.graph_trace_actions,
            empty_state=self.empty_state,
            hover_card=self.hover_card,
            loading_overlay=self.loading_overlay,
            on_size_change=self._on_surface_size,
        )
        for text_field in (
            self.group_label_field,
            self.search_field,
            self.edit_primary,
            self.edit_secondary,
            self.edit_port,
            self.transformation_axis,
            self.transformation_parameter,
            self.trace_seed,
            self.trace_shapes,
            self.trace_wall_seconds,
            self.trace_memory_mib,
            self.trace_capture_mib,
            self.trace_chunk_kib,
        ):
            self._watch_text_focus(text_field)

    # -- construction ------------------------------------------------------

    def build(self) -> ft.Control:
        top_bar = ft.Row(
            controls=[
                ft.Container(
                    content=ft.Image(
                        src=APP_ICON_PATH.read_bytes(),
                        width=38,
                        height=38,
                        fit=ft.BoxFit.COVER,
                        border_radius=11,
                    ),
                    width=38,
                    height=38,
                    alignment=ft.Alignment.CENTER,
                    border_radius=11,
                ),
                ft.Column(
                    controls=[self.title_text, self.model_subtitle],
                    spacing=0,
                    tight=True,
                ),
                ft.Container(width=8),
                ft.FilledButton(
                    content="Open model",
                    icon=ft.Icons.FOLDER_OPEN_ROUNDED,
                    on_click=self._on_open_clicked,
                ),
                self.save_model_button,
                self.close_model_button,
                ft.Container(expand=True),
                self.job_text,
                self.file_types_button,
                self.left_toggle,
                self.right_toggle,
            ],
            spacing=7,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        workspace_header = ft.Container(
            content=ft.Row(
                controls=[
                    self.breadcrumbs_row,
                    ft.Container(expand=True),
                    self.reset_view_button,
                    self.organize_button,
                    ft.Text(
                        "View",
                        size=10,
                        color=_MUTED,
                        weight=ft.FontWeight.W_700,
                    ),
                    self.detail_segment,
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding.symmetric(horizontal=14, vertical=8),
            bgcolor=_PANEL,
            border=ft.Border.only(bottom=ft.BorderSide(1, _BORDER)),
        )
        graph_workspace = ft.Column(
            controls=[workspace_header, self.surface],
            expand=True,
            spacing=0,
        )
        model_workspace = ft.Row(
            controls=[self.left_panel, graph_workspace, self.right_panel],
            expand=True,
            spacing=0,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        generator_workspace = self.input_generator.control
        self.workspace_tabs = ft.Tabs(
            content=ft.Column(
                controls=[
                    ft.TabBar(
                        tabs=[
                            ft.Tab(
                                label="Model graph",
                                icon=ft.Icons.ACCOUNT_TREE_ROUNDED,
                            ),
                            ft.Tab(
                                label="Generate test input",
                                icon=ft.Icons.SCIENCE_ROUNDED,
                            ),
                        ],
                        scrollable=False,
                        indicator_color=_ACCENT,
                        label_color=_ACCENT,
                        unselected_label_color=_MUTED,
                    ),
                    ft.TabBarView(
                        controls=[model_workspace, generator_workspace],
                        expand=True,
                    ),
                ],
                expand=True,
                spacing=0,
            ),
            length=2,
            selected_index=0,
            expand=True,
        )
        return ft.Column(
            controls=[
                ft.Container(
                    content=top_bar,
                    padding=ft.Padding.symmetric(horizontal=14, vertical=9),
                    bgcolor=_PANEL,
                    border=ft.Border.only(bottom=ft.BorderSide(1, _BORDER)),
                ),
                self.error_banner,
                self.workspace_tabs,
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Container(
                                width=7,
                                height=7,
                                bgcolor="#12B76A",
                                border_radius=4,
                            ),
                            self.status_text,
                        ],
                        spacing=7,
                    ),
                    padding=ft.Padding.symmetric(horizontal=14, vertical=7),
                    bgcolor=_PANEL,
                    border=ft.Border.only(top=ft.BorderSide(1, _BORDER)),
                ),
            ],
            expand=True,
            spacing=0,
        )

    def apply_layout_for_width(self, width: float) -> None:
        """Collapse the side panels on narrow windows, reversibly.

        Width only sets the *default*: once the user works a toggle their
        choice wins, because a resize that silently re-hid a panel they had
        just opened would be the same trap as having no toggle at all.
        """
        narrow = width < _NARROW_BREAKPOINT
        if self.left_panel_override is None:
            self.left_panel.visible = not narrow
        if self.right_panel_override is None:
            self.right_panel.visible = not narrow
        self._refresh_panel_toggles()

    def _refresh_panel_toggles(self) -> None:
        self.left_toggle.content = (
            "Hide panels" if self.left_panel.visible else "Show panels"
        )
        self.right_toggle.content = (
            "Hide inspector" if self.right_panel.visible else "Show inspector"
        )

    def _on_toggle_left_panel(self, event: ft.Event[ft.TextButton]) -> None:
        self.left_panel_override = not self.left_panel.visible
        self.left_panel.visible = self.left_panel_override
        self._refresh_panel_toggles()
        self.page.update()

    def _on_toggle_right_panel(self, event: ft.Event[ft.TextButton]) -> None:
        self.right_panel_override = not self.right_panel.visible
        self.right_panel.visible = self.right_panel_override
        self._refresh_panel_toggles()
        self.page.update()

    def _on_register_file_types(
        self,
        event: ft.Event[ft.TextButton],
    ) -> None:
        try:
            registration = register_file_associations(icon_path=APP_WINDOW_ICON_PATH)
            open_default_apps_settings()
        except (FileAssociationError, OSError, ValueError) as error:
            self._show_error(f"Could not register NNEditor file types: {error}")
        else:
            self.error_banner.visible = False
            self._set_status(
                f"Registered {len(registration.extensions)} model file types; "
                "choose NNEditor in Windows Default apps."
            )
        self.page.update()

    def _assign_generated_input(
        self,
        input_name: str,
        generated: GeneratedTensor,
    ) -> None:
        if self.session is None:
            raise ValueError("the model was closed before assignment")
        binding = self.session.trace_tensor_input(input_name, generated.path)
        self._trace_input_bindings[input_name] = binding
        self._refresh_graph_trace_actions()
        self._refresh_trace_actions()
        graph = self.session.document.main_graph
        value_id = next(
            value_id
            for value_id in graph.inputs
            if value_id not in graph.initializers
            and (graph.value(value_id).name or value_id) == input_name
        )
        if self.current_graph != self.session.document.entry_graph:
            self._show_graph(self.session.document.entry_graph)
        presentation = self._trace_graph_presentation
        glyph_id = (
            presentation.input_glyphs.get(value_id)
            if presentation is not None
            else None
        )
        if glyph_id is not None:
            self.workspace_tabs.selected_index = 0
            selected = frozenset({glyph_id})
            self.renderer.set_selection(selected)
            self._on_selected(selected)
        else:
            self._refresh_inspector(self._inspected_ids)

    def _refresh_generator_targets(self) -> None:
        session = self.session
        if session is None:
            self.input_generator.refresh_targets((), can_assign=False)
            return
        graph = session.document.main_graph
        targets = tuple(
            input_workspace.InputTarget(
                name=value.name or value.id,
                element_type=value.element_type,
                shape=value.shape,
                is_mask=_uses_automatic_mask(value.name or value.id),
            )
            for value_id in graph.inputs
            if value_id not in graph.initializers
            for value in (graph.value(value_id),)
        )
        self.input_generator.refresh_targets(
            targets,
            can_assign=not bool(getattr(self.page, "web", False)),
        )

    # -- event handlers ----------------------------------------------------

    async def _on_open_clicked(self, event: ft.Event[ft.Button]) -> None:
        if self.open_job is not None and not self.open_job.state.is_terminal:
            return
        if self._pending_web_upload is not None:
            # A second pick during an upload would overwrite the pending
            # target and hand the first upload's completion a half-written
            # file; the open guard is inactive here because no job exists yet.
            return
        files = await self.picker.pick_files(
            dialog_title="Open an ONNX, PyTorch, safetensors, or StableHLO model",
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=list(MODEL_FILE_EXTENSIONS),
            allow_multiple=False,
            with_data=False,
        )
        if not files:
            return
        selected = files[0].path
        if selected is None:
            suffix = Path(files[0].name).suffix[:24]
            upload_name = f"{uuid.uuid4().hex}{suffix}"
            target = _WEB_UPLOAD_DIRECTORY / upload_name
            self._pending_web_upload = (target, files[0].name)
            self.loading_title.value = f"Uploading {files[0].name}"
            self.loading_stage.value = "Uploading model… 0%"
            self.cancel_open_button.disabled = True
            self.loading_overlay.visible = True
            self._set_status(f"Uploading {files[0].name} without blocking the UI…")
            self.page.update()
            upload_url = self.page.get_upload_url(upload_name, expires=600)
            await self.picker.upload(
                [
                    ft.FilePickerUploadFile(
                        id=files[0].id,
                        name=files[0].name,
                        upload_url=upload_url,
                    )
                ]
            )
            return
        display_name = Path(selected).name
        job = self.service.open_model_async(selected)
        self._begin_open_job(job, display_name)

    def open_path(self, path: Path | str) -> None:
        """Open a path supplied by the command line or desktop shell."""
        if self.open_job is not None and not self.open_job.state.is_terminal:
            raise RuntimeError("another model is already opening")
        selected = Path(path).expanduser().resolve()
        job = self.service.open_model_async(selected)
        self._begin_open_job(job, selected.name)

    def _begin_open_job(self, job: Job[ModelSession], display_name: str) -> None:
        self.open_job = job
        self.loading_title.value = f"Opening {display_name}"
        self.loading_stage.value = (
            "Indexing topology, detecting blocks, and laying out the architecture…"
        )
        self.cancel_open_button.disabled = False
        self.loading_overlay.visible = True
        self._set_status(f"Preparing {display_name} in a background worker…")
        self.job_text.value = f"Job {job.id}: pending"
        self.page.update()
        self._watch_open(job)

    def _on_upload_progress(self, event: ft.FilePickerUploadEvent) -> None:
        pending = self._pending_web_upload
        if pending is None:
            return
        target, original_name = pending
        if event.file_name != original_name:
            # A superseded upload's progress must not act on the pending one.
            return
        if event.error:
            self._pending_web_upload = None
            self.loading_overlay.visible = False
            self._show_error(f"Could not upload {original_name}: {event.error}")
            self.page.update()
            return
        progress = event.progress or 0.0
        self.loading_stage.value = f"Uploading model… {progress:.0%}"
        self._set_status(f"Uploading {original_name}: {progress:.0%}")
        if progress >= 1.0:
            self._pending_web_upload = None
            job = self.service.open_uploaded_file_async(original_name, target)
            self.cancel_open_button.disabled = False
            self._begin_open_job(job, original_name)
            return
        self.page.update()

    def _watch_open(self, job: Job[ModelSession]) -> None:
        def wait() -> None:
            # run_thread futures are never retrieved, so an exception here
            # would vanish and leave the modal overlay covering the app
            # forever; surface it and always clear the overlay instead.
            try:
                job.wait()
                self.job_text.value = f"Job {job.id}: {job.state.value}"
                if job.state.value == "succeeded":
                    self.loading_stage.value = "Building the architecture workspace…"
                    self.page.update()
                    self.show_session(job.result())
                elif job.state.value == "failed":
                    self._show_error(f"Could not open the model: {job.error}")
                elif job.state.value == "cancelled":
                    self._set_status("Open cancelled")
            except Exception as error:
                self._show_error(f"Could not present the model: {error}")
            finally:
                self.loading_overlay.visible = False
                if self.open_job is job:
                    self.open_job = None
                self.page.update()

        self.page.run_thread(wait)

    def _on_cancel_open(self, event: ft.Event[ft.TextButton]) -> None:
        if self.open_job is None or self.open_job.state.is_terminal:
            return
        self.open_job.cancel()
        self.cancel_open_button.disabled = True
        self.loading_stage.value = "Cancelling after the current safe checkpoint…"
        self._set_status("Cancelling model open…")
        self.page.update()

    def _on_close_model(self, event: ft.Event[ft.TextButton]) -> None:
        """Close the active session and return the shell to its empty state."""
        session = self.session
        if session is None:
            return
        title = session.title
        if self.trace_job is not None and not self.trace_job.state.is_terminal:
            self.trace_job.cancel()
        if self.export_job is not None and not self.export_job.state.is_terminal:
            self.export_job.cancel()
        self.trace_job = None
        self._trace_preparing = False
        self.export_job = None
        self.service.close_session(session.id)
        self.session = None
        self._saved_revision_id = None
        self.current_graph = None
        self.current_root_group = None
        self.current_slice = None
        self.logical_selection = frozenset()
        self.minimap_model = None
        self._minimap_view_rect = None
        self._typed_preview_consent.clear()
        self._hex_offsets.clear()
        self._hex_drafts.clear()
        self._hex_errors.clear()
        self._tensor_card_expanded.clear()
        self._inspected_ids = frozenset()
        self.pending_edit = None
        self.pending_transformation = None
        self.pending_target = None
        self.active_trace_id = None
        self.active_trace_comparison = None
        self._activation_statistics.clear()
        self._activation_visualizations.clear()
        self._activation_visualization_errors.clear()
        self._activation_loading.clear()
        self._trace_input_bindings.clear()
        self._trace_graph_presentation = None
        self.search_field.value = ""
        self.search_results.controls = []
        self.graph_list.controls = []
        self.hierarchy_list.controls = []
        self.breadcrumbs_row.controls = []
        self.graph_trace_actions.controls = []
        self.minimap_canvas.shapes = []
        self.inspector.controls = []
        self.inspector_title.value = "Model overview"
        self.inspector_subtitle.value = "Open a model to inspect it"
        self.open_selection_button.visible = False
        self.back_to_parent_button.visible = False
        self.hover_card.visible = False
        self.reset_view_button.disabled = True
        self.title_text.value = APP_TITLE
        self.model_subtitle.value = "Neural network explorer"
        self.job_text.value = ""
        self.empty_state.visible = True
        self.error_banner.visible = False
        self.renderer.replace_scene(Scene(()), self._current_viewport())
        self._show_committed_diff()
        self._refresh_trace_actions()
        self._refresh_edit_actions()
        self._refresh_generator_targets()
        self._set_status(f"Closed {title}")
        self.page.update()

    def _on_pan(self, event: ft.DragUpdateEvent[ft.GestureDetector]) -> None:
        if self.session is None or event.local_delta is None:
            return
        self.view["x"] -= event.local_delta.x / self.view["scale"]
        self.view["y"] -= event.local_delta.y / self.view["scale"]
        self._request_viewport_apply()

    def _on_scroll(self, event: ft.ScrollEvent[ft.GestureDetector]) -> None:
        if self.session is None:
            return
        delta_y = event.scroll_delta.y if event.scroll_delta is not None else 0.0
        if delta_y == 0.0:
            return
        old_scale = self.view["scale"]
        position = event.local_position
        anchor_x = self.view["x"] + position.x / old_scale
        anchor_y = self.view["y"] + position.y / old_scale
        factor = 0.9 if delta_y > 0 else 1.1
        new_scale = max(0.02, min(4.0, old_scale * factor))
        self.view = {
            "scale": new_scale,
            "x": anchor_x - position.x / new_scale,
            "y": anchor_y - position.y / new_scale,
        }
        if self.auto_detail:
            resolved = detail_for_scale(new_scale)
            if resolved is not self.current_detail:
                self.current_detail = resolved
                self._replace_current_representation(fit=False)
                return
        self._request_viewport_apply()

    def _request_viewport_apply(self) -> None:
        """Apply the current viewport, coalescing gesture bursts.

        Handlers run on the Flet event loop, so work done here delays every
        queued gesture event. Recording the target and draining once keeps a
        fast pointer ahead of a slow canvas: intermediate viewports are
        skipped instead of queueing a full render each.
        """
        self._viewport_dirty = True
        run_task = getattr(self.page, "run_task", None)
        if run_task is None:
            # Headless shells (tests) have no event loop; apply directly.
            self._viewport_dirty = False
            self._apply_viewport()
            return
        if self._viewport_flush is not None and not self._viewport_flush.done():
            return
        self._viewport_flush = run_task(self._drain_viewport_requests)

    async def _drain_viewport_requests(self) -> None:
        while self._viewport_dirty:
            self._viewport_dirty = False
            self._apply_viewport()
            # Yield so queued gesture events can move the target viewport
            # before the next apply; they only record coordinates.
            await asyncio.sleep(0)

    def _on_reset_view(self, event: ft.Event[ft.TextButton]) -> None:
        """Refit the current context without changing navigation or selection."""
        if self.session is None or self.current_slice is None:
            return
        self._hide_hover_card()
        self._replace_scene_fitted(self.current_slice)
        self._restore_logical_selection()
        self._rebuild_minimap()
        self._set_status(f"{self.current_detail.value.title()} view reset")
        self._record_view()
        self.page.update()

    def _on_surface_hover(self, event: ft.PointerEvent[ft.GestureDetector]) -> None:
        """Show lightweight context without changing the user's selection."""
        if (
            self.session is None
            or self.current_graph is None
            or self.renderer.viewport is None
        ):
            return
        position = event.local_position
        world_x, world_y = self.renderer.viewport.to_world(position.x, position.y)
        hit = self.renderer.hit_test(world_x, world_y)
        if hit is None:
            if self.hover_card.visible:
                self._hide_hover_card()
                self.page.update()
            return
        if hit == self._hovered_id:
            return
        scene = self.renderer.scene
        assert scene is not None
        if scene.has_edge(hit):
            values = self._trace_values_for_glyph(hit)
            names = self._trace_value_names(values)
            self.hover_title.value = names[0] if len(names) == 1 else "Connection"
            self.hover_summary.value = (
                f"{len(values)} activation value(s)\n"
                "Click to inspect captured tensor data"
            )
            self._hovered_id = hit
            self.hover_card.left = min(
                max(position.x + 16, 12), max(self.surface_size[0] - 276, 12)
            )
            self.hover_card.top = min(
                max(position.y + 16, 12), max(self.surface_size[1] - 100, 12)
            )
            self.hover_card.visible = True
            self.page.update()
            return
        hierarchy = self.session.graph_hierarchy(self.current_graph)
        glyph = scene.node(hit)
        if hierarchy.has_group(hit):
            group = hierarchy.group(hit)
            self.hover_title.value = group.label
            self.hover_summary.value = (
                f"{group.kind.value.title()} block · {len(group.members)} operators "
                f"· {group.confidence:.0%} confidence\n"
                "Click for metadata · Double-click to open"
            )
        elif hit.startswith("grp:overview:") and self.current_slice is not None:
            members = self.current_slice.members_by_glyph.get(hit, frozenset())
            self.hover_title.value = glyph.label
            self.hover_summary.value = (
                f"Architecture region · {len(members)} operators\n"
                "Click for metadata · Double-click to see blocks"
            )
        else:
            self.hover_title.value = glyph.label
            self.hover_summary.value = (
                f"{glyph.display_type} operator\n"
                "Click to inspect inputs, outputs, attributes, and weights"
            )
        self._hovered_id = hit
        self.hover_card.left = min(
            max(position.x + 16, 12), max(self.surface_size[0] - 276, 12)
        )
        self.hover_card.top = min(
            max(position.y + 16, 12), max(self.surface_size[1] - 100, 12)
        )
        self.hover_card.visible = True
        self.page.update()

    def _on_surface_exit(self, event: ft.PointerEvent[ft.GestureDetector]) -> None:
        if self.hover_card.visible:
            self._hide_hover_card()
            self.page.update()

    def _hide_hover_card(self) -> None:
        self._hovered_id = None
        self.hover_card.visible = False

    def _on_surface_double_tap(self, event: ft.Event[ft.GestureDetector]) -> None:
        self._drill_into_selection()

    def _on_selected(self, ids: frozenset[str]) -> None:
        self._hide_hover_card()
        self._discard_stale_pending()
        if self.current_slice is not None:
            self.logical_selection = frozenset(
                member
                for glyph_id in ids
                for member in self.current_slice.members_by_glyph.get(
                    glyph_id, frozenset((glyph_id,))
                )
            )
        self._autoload_activation_views(ids)
        self._refresh_inspector(ids)
        self._record_view()
        self.page.update()

    def _on_search(self, event: ft.Event[ft.TextField]) -> None:
        if self.session is None:
            return
        document = self.session.document
        hits = self.session.search_hits(self.search_field.value or "")
        self.search_results.controls = [
            ft.TextButton(
                content=f"{hit.label} "
                f"[{document.graphs[hit.graph_id].name or hit.graph_id}]",
                on_click=self._jump_handler(hit.graph_id, hit.node_id),
            )
            for hit in hits[:50]
        ]
        self._set_status(f"{len(hits)} match(es)")
        self.page.update()

    def _jump_handler(
        self, graph_id: str, node_id: str
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def jump(event: ft.Event[ft.TextButton]) -> None:
            # A result captured from a previously opened model must never
            # mutate navigation state; validate before touching anything.
            if self.session is None or graph_id not in self.session.document.graphs:
                self._set_status(
                    "That result belongs to a previously opened model; search again."
                )
                self.page.update()
                return
            if graph_id != self.current_graph:
                self._show_graph(graph_id)
            self.focus_node(node_id)

        return jump

    def _on_detail_segment_changed(self, event: ft.Event[ft.SegmentedButton]) -> None:
        selected = self.detail_segment.selected
        if not selected:
            return
        value = selected[0]
        if value == "auto":
            self.auto_detail = True
            self.current_detail = detail_for_scale(self.view["scale"])
        else:
            self.auto_detail = False
            self.current_detail = DetailLevel(value)
        self._replace_current_representation()

    def _sync_detail_controls(self) -> None:
        """Reflect the detail state in the segment; Auto stays selected."""
        self.detail_segment.selected = [
            "auto" if self.auto_detail else self.current_detail.value
        ]

    def _on_open_selected(self, event: ft.Event[ft.Button]) -> None:
        self._drill_into_selection()

    def _drill_into_selection(self) -> None:
        if (
            self.session is None
            or self.current_graph is None
            or len(self.renderer.selection) != 1
        ):
            return
        (selected,) = self.renderer.selection
        scene = self.renderer.scene
        if scene is not None and (
            scene.has_edge(selected) or selected in self._trace_boundary_values()
        ):
            return
        hierarchy = self.session.graph_hierarchy(self.current_graph)
        next_level = {
            DetailLevel.ARCHITECTURE: DetailLevel.BLOCK,
            DetailLevel.BLOCK: DetailLevel.LAYER,
            DetailLevel.LAYER: DetailLevel.OPERATOR,
            DetailLevel.OPERATOR: DetailLevel.OPERATOR,
        }[self.current_detail]
        self.current_detail = next_level
        self.auto_detail = False
        self._sync_detail_controls()
        if hierarchy.has_group(selected):
            self.show_group(selected)
            self._refresh_inspector(frozenset({selected}))
        else:
            self._replace_current_representation()
        self._set_status(f"Opened {next_level.value} view")
        self.page.update()

    def _on_back_to_parent(self, event: ft.Event[ft.TextButton]) -> None:
        if (
            self.session is None
            or self.current_graph is None
            or self.current_root_group is None
        ):
            return
        hierarchy = self.session.graph_hierarchy(self.current_graph)
        group = hierarchy.group(self.current_root_group)
        if group.parent_id is None:
            self._show_graph(self.current_graph)
        else:
            self.show_group(group.parent_id)
        self._refresh_inspector(self.renderer.selection)
        self.page.update()

    def _on_surface_size(
        self, event: ft.LayoutSizeChangeEvent[ft.LayoutControl]
    ) -> None:
        if event.width > 0 and event.height > 0:
            self.surface_size = (float(event.width), float(event.height))
            if self.session is not None:
                self._apply_viewport()

    def _on_minimap_tap(self, event: ft.TapEvent[ft.GestureDetector]) -> None:
        if (
            self.minimap_model is None
            or event.local_position is None
            or self.session is None
        ):
            return
        world_x, world_y = self.minimap_model.world_at(
            event.local_position.x, event.local_position.y
        )
        viewport = self._current_viewport()
        self.view["x"] = world_x - viewport.width / 2
        self.view["y"] = world_y - viewport.height / 2
        self._apply_viewport()

    def _on_multi_select_changed(self, event: ft.Event[ft.Checkbox]) -> None:
        self.renderer.set_additive_selection(bool(self.multi_select_field.value))

    def _group_label(self) -> str:
        label = (self.group_label_field.value or "").strip()
        if not label:
            raise ValueError("enter a group label first")
        return label

    def _selected_group_ids(self) -> tuple[str, ...]:
        if self.session is None or self.current_graph is None:
            return ()
        hierarchy = self.session.graph_hierarchy(self.current_graph)
        return tuple(
            item for item in self.renderer.selection if hierarchy.has_group(item)
        )

    def _on_group_selected(self, event: ft.Event[ft.TextButton]) -> None:
        if self.session is None or self.current_graph is None:
            return
        try:
            group = self.session.hierarchy.group(
                self.current_graph,
                self._group_label(),
                self.logical_selection,
            )
        except (KeyError, ValueError) as error:
            self._show_error(str(error))
            self.page.update()
            return
        self.current_detail = DetailLevel.BLOCK
        self.auto_detail = False
        self._sync_detail_controls()
        self.logical_selection = group.members
        self._repair_missing_root_group()
        self._after_hierarchy_edit()

    def _on_merge_selected(self, event: ft.Event[ft.TextButton]) -> None:
        if self.session is None or self.current_graph is None:
            return
        merged = self._selected_group_ids()
        try:
            group = self.session.hierarchy.merge(
                self.current_graph,
                merged,
                self._group_label(),
            )
        except (KeyError, ValueError) as error:
            self._show_error(str(error))
            self.page.update()
            return
        if self.current_root_group in merged:
            # Merging consumed the drilled-in root; follow it into the
            # merged group as rename does, instead of pointing at a ghost.
            self.current_root_group = group.id
        self.logical_selection = group.members
        self._repair_missing_root_group()
        self._after_hierarchy_edit()

    def _on_rename_selected(self, event: ft.Event[ft.TextButton]) -> None:
        if self.session is None or self.current_graph is None:
            return
        selected = self._selected_group_ids()
        if len(selected) != 1:
            self._show_error("select exactly one detected group to rename")
            self.page.update()
            return
        old_id = selected[0]
        try:
            group = self.session.hierarchy.rename(
                self.current_graph, old_id, self._group_label()
            )
        except (KeyError, ValueError) as error:
            self._show_error(str(error))
            self.page.update()
            return
        if self.current_root_group == old_id:
            self.current_root_group = group.id
        self._after_hierarchy_edit()

    def _on_split_selected(self, event: ft.Event[ft.TextButton]) -> None:
        if self.session is None or self.current_graph is None:
            return
        selected = self._selected_group_ids()
        if len(selected) != 1:
            self._show_error("select exactly one detected group to split")
            self.page.update()
            return
        try:
            self.session.hierarchy.split(self.current_graph, selected[0])
        except (KeyError, ValueError) as error:
            self._show_error(str(error))
            self.page.update()
            return
        if self.current_root_group == selected[0]:
            self.current_root_group = None
        self._after_hierarchy_edit()

    def _on_lock_selected(self, event: ft.Event[ft.TextButton]) -> None:
        if self.session is None or self.current_graph is None:
            return
        selected = self._selected_group_ids()
        if len(selected) != 1:
            self._show_error("select exactly one detected group to lock")
            self.page.update()
            return
        hierarchy = self.session.graph_hierarchy(self.current_graph)
        target = hierarchy.group(selected[0])
        try:
            self.session.hierarchy.lock(
                self.current_graph, target.id, locked=not target.locked
            )
        except (KeyError, ValueError) as error:
            self._show_error(str(error))
            self.page.update()
            return
        self._after_hierarchy_edit()

    def _on_reject_selected(self, event: ft.Event[ft.TextButton]) -> None:
        if self.session is None or self.current_graph is None:
            return
        selected = self._selected_group_ids()
        if len(selected) != 1:
            self._show_error("select exactly one detected group to reject")
            self.page.update()
            return
        try:
            self.session.hierarchy.reject(self.current_graph, selected[0])
        except (KeyError, ValueError) as error:
            self._show_error(str(error))
            self.page.update()
            return
        if self.current_root_group == selected[0]:
            self.current_root_group = None
        self._after_hierarchy_edit()

    def _on_reset_groups(self, event: ft.Event[ft.TextButton]) -> None:
        if self.session is None or self.current_graph is None:
            return
        self.session.hierarchy.reset(self.current_graph)
        self.current_root_group = None
        self._after_hierarchy_edit()

    def _on_cycle_organization(self, event: ft.Event[ft.IconButton]) -> None:
        """Advance the block organization mode: Auto → Source → Repeated →
        Patterns → Structural → Auto."""
        if self.session is None or self.current_graph is None:
            return
        modes = list(OrganizationMode)
        mode = modes[(modes.index(self.session.hierarchy.mode) + 1) % len(modes)]
        self.session.hierarchy.set_mode(mode)
        # A drilled-in root reconciled under the previous mode may not exist
        # in this one; repair before the scene rebuild would raise on it.
        self._repair_missing_root_group()
        self.error_banner.visible = False
        self._replace_current_representation()
        self._sync_organize_button()
        self._set_status(f"Organized by {mode.value.title()}")
        self.page.update()

    @staticmethod
    def _organize_tooltip(mode: OrganizationMode) -> str:
        return f"Organize: {mode.value.title()} — press to cycle"

    def _sync_organize_button(self) -> None:
        """Reflect the session's persisted organization mode on the button."""
        mode = (
            self.session.hierarchy.mode
            if self.session is not None
            else OrganizationMode.AUTO
        )
        self.organize_button.tooltip = self._organize_tooltip(mode)

    def _repair_missing_root_group(self) -> None:
        """Drop a drilled-in root that a hierarchy mutation removed.

        Rebuilding a scene rooted at a group that no longer exists raises,
        wedging the view; falling back to the whole graph matches what the
        split and reject handlers already do.
        """
        if self.session is None or self.current_graph is None:
            return
        if self.current_root_group is None:
            return
        hierarchy = self.session.graph_hierarchy(self.current_graph)
        if not hierarchy.has_group(self.current_root_group):
            self.current_root_group = None

    def _after_hierarchy_edit(self) -> None:
        self.error_banner.visible = False
        self._replace_current_representation()
        self._refresh_hierarchy()
        self._set_status("Hierarchy updated")
        self.page.update()

    # -- session display ---------------------------------------------------

    def show_session(self, session: ModelSession) -> None:
        """Bind a freshly opened session to the surface and panels."""
        previous = self.session
        if self.trace_job is not None and not self.trace_job.state.is_terminal:
            self.trace_job.cancel()
        if self.export_job is not None and not self.export_job.state.is_terminal:
            self.export_job.cancel()
        self.trace_job = None
        self._trace_preparing = False
        self.export_job = None
        if previous is not None and previous is not session and not previous.closed:
            # A reopen must release the outgoing document, its caches, and
            # its open file handle (a lock on Windows) before the new
            # session takes over; otherwise every reopen leaks all three.
            self.service.close_session(previous.id)
        self.session = session
        suggested_limits = recommended_trace_limits(session.document.source.byte_size)
        self.trace_wall_seconds.value = f"{suggested_limits.wall_seconds:g}"
        self.trace_memory_mib.value = str(
            suggested_limits.memory_bytes // (1024 * 1024)
        )
        self._saved_revision_id = None
        self._typed_preview_consent.clear()
        self._hex_offsets.clear()
        self._hex_drafts.clear()
        self._hex_errors.clear()
        self._tensor_card_expanded.clear()
        self.active_trace_id = None
        self.active_trace_comparison = None
        self._activation_statistics.clear()
        self._activation_visualizations.clear()
        self._activation_visualization_errors.clear()
        self._activation_loading.clear()
        self._trace_input_bindings.clear()
        self._trace_graph_presentation = None
        self.pending_edit = None
        self.pending_transformation = None
        self.pending_target = None
        # Search results hold buttons bound to the previous model's graph
        # and node ids; keeping them would offer jumps into a closed model.
        self.search_field.value = ""
        self.search_results.controls = []
        self.current_graph = session.document.entry_graph
        self.current_root_group = None
        self.current_detail = DetailLevel.ARCHITECTURE
        self.auto_detail = False
        self._sync_detail_controls()
        # The organization mode is per-artifact sidecar state restored by the
        # hierarchy controller; the toolbar button must reflect it, not reset.
        self._sync_organize_button()
        self.reset_view_button.disabled = False
        self.logical_selection = frozenset()
        self.title_text.value = f"{APP_TITLE} — {session.title}"
        self.model_subtitle.value = "Architecture overview"
        self.empty_state.visible = False
        self.error_banner.visible = False
        self._refresh_graph_list()
        self._show_graph(self.current_graph)
        # Fresh artifacts open at architecture detail; returning users retain
        # their viewport and selection within that semantic representation.
        self._restore_view(session)
        counts = session.document.diagnostics
        self._set_status(
            f"Opened {session.title}: "
            f"{len(session.document.graphs)} graph(s), "
            f"{len(counts)} finding(s)"
        )
        self._refresh_inspector(frozenset())
        self._show_committed_diff()
        self._refresh_trace_actions()
        self._refresh_generator_targets()

    def _refresh_graph_list(self) -> None:
        assert self.session is not None
        entries = viewmodel.graph_entries(self.session.document)
        rows: list[ft.Control] = [
            ft.TextButton(
                content=("  " * entry.depth)
                + f"{entry.label}  ·  {entry.node_count} nodes",
                icon=(
                    ft.Icons.ACCOUNT_TREE_ROUNDED
                    if entry.depth == 0
                    else ft.Icons.SCHEMA_ROUNDED
                ),
                style=ft.ButtonStyle(
                    color=_ACCENT if entry.graph_id == self.current_graph else _INK,
                    bgcolor=(
                        _ACCENT_SOFT
                        if entry.graph_id == self.current_graph
                        else "#00FFFFFF"
                    ),
                    shape=ft.RoundedRectangleBorder(radius=9),
                    alignment=ft.Alignment.CENTER_LEFT,
                ),
                on_click=self._graph_handler(entry.graph_id),
            )
            for entry in entries[:_MAX_EXPLORER_ROWS]
        ]
        if len(entries) > _MAX_EXPLORER_ROWS:
            rows.append(
                ft.Text(
                    f"Showing {_MAX_EXPLORER_ROWS} of {len(entries)} graphs. "
                    "Use search to reach the rest.",
                    size=10,
                    color=_MUTED,
                )
            )
        self.graph_list.controls = rows

    def _graph_handler(
        self, graph_id: str
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def switch(event: ft.Event[ft.TextButton]) -> None:
            self._show_graph(graph_id)
            self.logical_selection = frozenset()
            self._refresh_graph_list()
            self._refresh_inspector(frozenset())
            self.page.update()

        return switch

    def _show_graph(self, graph_id: str) -> None:
        assert self.session is not None
        if graph_id not in self.session.document.graphs:
            # Never adopt a graph id the current document does not know: a
            # failure later in the rebuild would leave navigation wedged on
            # a nonexistent graph.
            self._set_status(f"Graph {graph_id!r} is not part of this model")
            return
        self.current_graph = graph_id
        self.current_root_group = None
        layout = self.session.scene(graph_id, detail_level=self.current_detail)
        self.current_slice = layout
        if self.auto_detail:
            bounds = layout.scene.bounds
            width, height = self.surface_size
            fit = min(
                1.0,
                width / max(bounds.width, 1.0),
                height / max(bounds.height, 1.0),
            )
            resolved = detail_for_scale(max(fit, 0.02))
            if resolved is not self.current_detail:
                self.current_detail = resolved
                layout = self.session.scene(graph_id, detail_level=self.current_detail)
                self.current_slice = layout
        self._replace_scene_fitted(layout)
        self._restore_logical_selection()
        self._refresh_hierarchy()
        self._refresh_breadcrumbs()
        self._rebuild_minimap()

    def show_group(self, group_id: str) -> None:
        """Navigate into a group without losing the operator selection."""
        if self.session is None or self.current_graph is None:
            return
        self.current_root_group = group_id
        layout = self.session.scene(
            self.current_graph,
            detail_level=self.current_detail,
            root_group=group_id,
        )
        self.current_slice = layout
        self._replace_scene_fitted(layout)
        self._restore_logical_selection()
        self._refresh_breadcrumbs()
        self._rebuild_minimap()

    def _replace_scene_fitted(self, layout: GraphSlice) -> None:
        """Fit a new architecture or block context with comfortable padding."""
        scene = self._display_scene(layout)
        bounds = scene.bounds
        width, height = self.surface_size
        fit = min(
            1.25,
            max(width - 96.0, 1.0) / max(bounds.width, 1.0),
            max(height - 96.0, 1.0) / max(bounds.height, 1.0),
        )
        scale = max(fit, 0.02)
        world_width = width / scale
        world_height = height / scale
        self.view = {
            "scale": scale,
            "x": bounds.min_x + bounds.width / 2.0 - world_width / 2.0,
            "y": bounds.min_y + bounds.height / 2.0 - world_height / 2.0,
        }
        self.renderer.replace_scene(scene, self._current_viewport())
        self._refresh_graph_trace_actions()

    def _display_scene(self, layout: GraphSlice) -> Scene:
        """Apply the active trace/comparison view without changing base layout."""
        if self.session is None:
            self._trace_graph_presentation = None
            return layout.scene
        scene = layout.scene
        if self.active_trace_comparison is not None:
            scene = self.session.comparison_overlay(
                layout,
                self.active_trace_comparison,
            )
        elif self.active_trace_id is not None:
            try:
                result = self.session.trace(self.active_trace_id)
            except (KeyError, ValueError):
                result = None
            if result is not None:
                scene = self.session.trace_overlay(layout, result)
        if self.current_graph != self.session.document.entry_graph:
            self._trace_graph_presentation = None
            return scene
        presentation = build_trace_graph(
            scene,
            self.session.document.main_graph,
            layout.members_by_glyph,
        )
        self._trace_graph_presentation = presentation
        return presentation.scene

    def _replace_current_representation(self, *, fit: bool = True) -> None:
        if self.session is None or self.current_graph is None:
            return
        layout = self.session.scene(
            self.current_graph,
            detail_level=self.current_detail,
            root_group=self.current_root_group,
        )
        self.current_slice = layout
        if fit:
            self._replace_scene_fitted(layout)
        else:
            self.renderer.replace_scene(
                self._display_scene(layout), self._current_viewport()
            )
            self._refresh_graph_trace_actions()
        self._restore_logical_selection()
        self._refresh_hierarchy()
        self._refresh_breadcrumbs()
        self._rebuild_minimap()
        self._sync_detail_controls()
        self.model_subtitle.value = f"{self.current_detail.value.title()} view"
        self._set_status(f"{self.current_detail.value.title()} detail")
        self.page.update()

    def _restore_logical_selection(self) -> None:
        layout = self.current_slice
        scene = self.renderer.scene
        if layout is None or scene is None:
            return
        visible_items: set[str] = set()
        for glyph_id in self.logical_selection:
            if scene.has_node(glyph_id) or scene.has_edge(glyph_id):
                visible_items.add(glyph_id)
                continue
            representative = layout.representative_for(glyph_id)
            if representative is not None and scene.has_node(representative):
                visible_items.add(representative)
        visible = frozenset(visible_items)
        if visible:
            self.renderer.set_selection(visible)

    def focus_node(self, node_id: str) -> None:
        """Center a node found via search, at a readable zoom."""
        if self.session is None or self.current_graph is None:
            return
        needs_rebuild = False
        if self.current_detail is not DetailLevel.OPERATOR:
            self.current_detail = DetailLevel.OPERATOR
            self.auto_detail = False
            needs_rebuild = True
        layout = self.session.scene(
            self.current_graph,
            detail_level=DetailLevel.OPERATOR,
            root_group=self.current_root_group,
        )
        if self.current_root_group is not None and not layout.scene.has_node(node_id):
            # The hit lies outside the drilled-in block; a silent no-op would
            # leave the user staring at an unrelated view, so widen the scope
            # to the whole graph before focusing.
            self.current_root_group = None
            needs_rebuild = True
            layout = self.session.scene(
                self.current_graph,
                detail_level=DetailLevel.OPERATOR,
                root_group=None,
            )
        if not layout.scene.has_node(node_id):
            self._set_status("That node is not part of the current graph")
            self.page.update()
            return
        if needs_rebuild:
            self._sync_detail_controls()
            self._replace_current_representation()
        width, height = self.surface_size
        viewport = self.renderer.viewport_focused_on(
            node_id, width=width, height=height, scale=max(self.view["scale"], 1.0)
        )
        self.view = {"scale": viewport.scale, "x": viewport.x, "y": viewport.y}
        self.renderer.set_viewport(viewport)
        self.renderer.set_selection(frozenset({node_id}))
        self.logical_selection = frozenset({node_id})
        self._refresh_inspector(frozenset({node_id}))
        self._set_status(f"Focused {layout.scene.node(node_id).label}")
        self.page.update()

    def _restore_view(self, session: ModelSession) -> None:
        """Return to where the user last was in this artifact, if known."""
        store = self.service.state_store
        if store is None:
            return
        view = store.state.view_for(session.document.source.content_hash)
        if view is None or view.graph_id not in session.document.graphs:
            return
        self._show_graph(view.graph_id)
        self.view = {"scale": view.scale, "x": view.x, "y": view.y}
        self.renderer.set_viewport(self._current_viewport())
        # The minimap rectangle was projected from the fitted viewport during
        # the rebuild above; re-project it for the restored one.
        self._update_minimap_viewport()
        scene = self.renderer.scene
        if scene is not None:
            surviving = frozenset(
                glyph_id
                for glyph_id in view.selection
                if scene.has_node(glyph_id) or scene.has_edge(glyph_id)
            )
            if surviving:
                self.renderer.set_selection(surviving)
                self.logical_selection = surviving
                self._refresh_inspector(surviving)

    def _record_view(self) -> None:
        store = self.service.state_store
        if store is None or self.session is None or self.current_graph is None:
            return
        store.state.record_view(
            self.session.document.source.content_hash,
            ViewState(
                graph_id=self.current_graph,
                x=self.view["x"],
                y=self.view["y"],
                scale=self.view["scale"],
                selection=tuple(sorted(self.renderer.selection)),
            ),
        )

    def _current_viewport(self) -> Viewport:
        width, height = self.surface_size
        scale = self.view["scale"]
        return Viewport(
            x=self.view["x"],
            y=self.view["y"],
            width=width / scale,
            height=height / scale,
            scale=scale,
        )

    def _apply_viewport(self) -> None:
        self.renderer.set_viewport(self._current_viewport())
        self._refresh_graph_trace_actions()
        stats = self.renderer.stats
        status = (
            f"{stats.visible_nodes} nodes, {stats.visible_edges} edges visible "
            f"({stats.build_ms:.0f} ms)"
        )
        if stats.dropped_nodes:
            # Never let a deliberately partial picture pass for the whole
            # graph; name the remedy the user actually has.
            status += (
                f" — {stats.dropped_nodes:,} more not drawn; zoom in or use a "
                "coarser detail level"
            )
        self._set_status(status)
        self._record_view()
        self._update_minimap_viewport()
        # Only the canvas (flushed by the renderer), the status line, and the
        # minimap viewport change here; a page-wide diff would re-walk every
        # mounted control — including thousands of canvas shapes — per event.
        self._update_control(self.status_text)
        self._update_control(self.graph_trace_actions)

    @staticmethod
    def _update_control(control: ft.Control) -> None:
        """Push one control's state if it is mounted; headless shells skip."""
        try:
            page = control.page
        except Exception:
            return
        if page is not None:
            control.update()

    def _refresh_inspector(self, ids: frozenset[str]) -> None:
        if self.session is None:
            return
        self._inspected_ids = ids
        document = self.session.document
        rows: list[ft.Control] = []
        self.back_to_parent_button.visible = self.current_root_group is not None
        self.open_selection_button.visible = False
        if len(ids) == 1 and self.current_graph is not None:
            self.inspector.spacing = 14
            (node_id,) = ids
            traced_rows = self._trace_glyph_inspector(node_id)
            if traced_rows is not None:
                self.inspector.controls = traced_rows
                self._refresh_edit_actions()
                return
            hierarchy = self.session.graph_hierarchy(self.current_graph)
            show_tensors = False
            if hierarchy.has_group(node_id):
                group = hierarchy.group(node_id)
                self.inspector_title.value = group.label
                self.inspector_subtitle.value = (
                    f"{group.kind.value.title()} block · {len(group.members)} operators"
                )
                self.open_selection_button.visible = group.id != self.current_root_group
                rows.extend(
                    overview.selected_block_controls(
                        pattern=group.kind.value,
                        members=len(group.members),
                        confidence=group.confidence,
                        explanation=group.explanation,
                    )
                )
                rows.extend(
                    self._group_activation_rows(group.members, owner_id=node_id)
                )
            elif node_id.startswith("grp:overview:") and self.current_slice is not None:
                members = self.current_slice.members_by_glyph[node_id]
                self.inspector_title.value = "Architecture region"
                self.inspector_subtitle.value = f"{len(members)} operators"
                self.open_selection_button.visible = True
                rows.extend(overview.selected_region_controls(len(members)))
                rows.extend(self._group_activation_rows(members, owner_id=node_id))
            else:
                details = viewmodel.node_details(document, self.current_graph, node_id)
                graph = document.graphs[self.current_graph]
                node = graph.node(node_id)
                self.inspector_title.value = node.source_name or node.op_type
                self.inspector_subtitle.value = (
                    f"{node.qualified_op_type} · Operator metadata"
                )
                rows.append(
                    overview.metadata_section(
                        details,
                        title="Operator metadata",
                        icon=ft.Icons.DATA_OBJECT_ROUNDED,
                        role="selection-item-metadata",
                    )
                )
                show_tensors = True
            if show_tensors:
                rows.extend(self._tensor_rows(node_id))
                rows.extend(self._activation_rows(node_id))
        else:
            self.inspector.spacing = 14
            if ids:
                self.inspector_title.value = f"{len(ids)} items selected"
                self.inspector_subtitle.value = "Use hierarchy tools to group them"
            else:
                self.inspector_title.value = "Model overview"
                node_count = sum(len(graph.nodes) for graph in document.graphs.values())
                self.inspector_subtitle.value = (
                    f"{viewmodel.humanize_identifier(document.artifact_kind.value)}"
                    f" · {node_count:,} operators"
                )
            rows.extend(overview.model_overview_controls(document))
        self.inspector.controls = rows
        self._refresh_edit_actions()

    def _trace_glyph_inspector(self, glyph_id: str) -> list[ft.Control] | None:
        """Inspector content for a model boundary or dataflow connection."""
        if (
            self.session is None
            or self.current_graph != self.session.document.entry_graph
        ):
            return None
        scene = self.renderer.scene
        if scene is None:
            return None
        boundary = self._trace_boundary_values()
        is_connection = scene.has_edge(glyph_id)
        if not is_connection and glyph_id not in boundary:
            return None
        value_ids = self._trace_values_for_glyph(glyph_id)
        graph = self.session.document.main_graph
        rows: list[ft.Control] = []
        if is_connection:
            names = self._trace_value_names(value_ids)
            self.inspector_title.value = (
                names[0] if len(names) == 1 else f"{len(names)} connections"
            )
            self.inspector_subtitle.value = (
                "Activation on this connection · captured tensor data"
            )
            rows.append(
                overview.section_heading(
                    "Connection values",
                    ft.Icons.CABLE_ROUNDED,
                    trailing=str(len(value_ids)),
                )
            )
        else:
            value_id = boundary[glyph_id]
            value = graph.value(value_id)
            node = scene.node(glyph_id)
            is_input = node.kind == "graph-input"
            self.inspector_title.value = value.name or value.id
            self.inspector_subtitle.value = (
                "Model input · choose the tensor used for tracing"
                if is_input
                else "Model output · captured result tensor"
            )
            if is_input:
                input_name = value.name or value.id
                binding = self._trace_input_bindings.get(input_name)
                automatic_mask = binding is None and _uses_automatic_mask(input_name)
                source = (
                    f".npy file · {Path(binding.tensor_file).name}"
                    if binding is not None and binding.tensor_file is not None
                    else "All-valid mask · generated automatically"
                    if automatic_mask
                    else "Deterministic random data"
                )
                rows.extend(
                    [
                        ft.Container(
                            content=ft.Text(
                                f"Next trace input: {source}",
                                size=11,
                                color=(
                                    "#027A48"
                                    if binding is not None or automatic_mask
                                    else _MUTED
                                ),
                            ),
                            padding=10,
                            bgcolor=(
                                "#ECFDF3"
                                if binding is not None or automatic_mask
                                else _SUBTLE
                            ),
                            border_radius=9,
                        ),
                        ft.Row(
                            controls=[
                                ft.FilledButton(
                                    data=f"trace-input-picker:{value_id}",
                                    content=(
                                        "Replace tensor"
                                        if binding is not None
                                        else "Choose .npy tensor"
                                    ),
                                    icon=ft.Icons.UPLOAD_FILE_ROUNDED,
                                    on_click=self._choose_trace_tensor_handler(
                                        value_id
                                    ),
                                ),
                                ft.TextButton(
                                    content=(
                                        "Use automatic mask"
                                        if _uses_automatic_mask(input_name)
                                        else "Use random"
                                    ),
                                    visible=binding is not None,
                                    on_click=self._reset_trace_tensor_handler(
                                        input_name,
                                        glyph_id,
                                    ),
                                ),
                            ],
                            wrap=True,
                            spacing=4,
                        ),
                    ]
                )
        for value_id in value_ids:
            value = graph.value(value_id)
            producer = graph.producer(value_id)
            rows.append(
                overview.metadata_section(
                    (
                        ("Value", value.name or value.id),
                        ("Dtype", value.element_type or "unknown"),
                        ("Shape", str(list(value.shape or ()))),
                        (
                            "Role",
                            "model input"
                            if value_id in graph.inputs
                            and value_id not in graph.initializers
                            else "model output"
                            if value_id in graph.outputs
                            else "intermediate activation",
                        ),
                        (
                            "Producer",
                            graph.node(producer[0]).source_name
                            or graph.node(producer[0]).op_type
                            if producer is not None
                            else "external input",
                        ),
                    ),
                    title="Tensor metadata",
                    icon=ft.Icons.DATA_ARRAY_ROUNDED,
                    role=f"trace-value-metadata:{value_id}",
                )
            )
        rows.extend(
            self._activation_rows_for_values(
                value_ids,
                owner_id=glyph_id,
                title=(
                    "Activation on this connection"
                    if is_connection
                    else "Captured tensor"
                ),
            )
        )
        return rows

    # -- inference tracing (Phase 9) --------------------------------------

    def _trace_boundary_values(self) -> dict[str, str]:
        presentation = self._trace_graph_presentation
        if presentation is None:
            return {}
        return {
            **{
                glyph_id: value_id
                for value_id, glyph_id in presentation.input_glyphs.items()
            },
            **{
                glyph_id: value_id
                for value_id, glyph_id in presentation.output_glyphs.items()
            },
        }

    def _trace_values_for_glyph(self, glyph_id: str) -> tuple[str, ...]:
        presentation = self._trace_graph_presentation
        if presentation is None:
            return ()
        return presentation.values_by_glyph.get(glyph_id, ())

    def _trace_value_names(self, value_ids: tuple[str, ...]) -> tuple[str, ...]:
        if self.session is None:
            return ()
        graph = self.session.document.main_graph
        return tuple(
            graph.value(value_id).name or graph.value(value_id).id
            for value_id in value_ids
        )

    def _refresh_graph_trace_actions(self) -> None:
        """Position one tensor-picker button inside every visible input glyph."""
        controls: list[ft.Control] = []
        presentation = self._trace_graph_presentation
        viewport = self.renderer.viewport
        scene = self.renderer.scene
        session = self.session
        if (
            presentation is None
            or viewport is None
            or scene is None
            or session is None
            or bool(getattr(self.page, "web", False))
        ):
            self.graph_trace_actions.controls = controls
            return
        status = session.capability(Capability.TRACING)
        if status.availability is not Availability.AVAILABLE:
            self.graph_trace_actions.controls = controls
            return
        graph = session.document.main_graph
        for value_id, glyph_id in presentation.input_glyphs.items():
            if not scene.has_node(glyph_id):
                continue
            node = scene.node(glyph_id)
            left, top = viewport.to_screen(node.x, node.y)
            width = node.width * viewport.scale
            height = node.height * viewport.scale
            if (
                left + width < 0
                or top + height < 0
                or left > self.surface_size[0]
                or top > self.surface_size[1]
            ):
                continue
            value = graph.value(value_id)
            input_name = value.name or value.id
            selected = self._trace_input_bindings.get(input_name)
            automatic_mask = selected is None and _uses_automatic_mask(input_name)
            button = ft.IconButton(
                data=f"trace-input-picker:{value_id}",
                icon=(
                    ft.Icons.CHECK_CIRCLE_ROUNDED
                    if selected is not None or automatic_mask
                    else ft.Icons.UPLOAD_FILE_ROUNDED
                ),
                icon_color=(
                    "#027A48" if selected is not None or automatic_mask else _ACCENT
                ),
                tooltip=(
                    f"Tensor: {Path(selected.tensor_file).name}"
                    if selected is not None and selected.tensor_file is not None
                    else (
                        f"{input_name} uses an all-valid mask automatically; "
                        "choose a .npy tensor to override it"
                    )
                    if automatic_mask
                    else f"Choose a .npy tensor for {input_name}"
                ),
                on_click=self._choose_trace_tensor_handler(value_id),
            )
            controls.append(
                ft.Container(
                    data=f"trace-input-action:{value_id}",
                    content=button,
                    left=left + max(width - 38.0, 2.0),
                    top=top + max((height - 36.0) / 2.0, 0.0),
                    width=36,
                    height=36,
                    bgcolor="#F2FFFFFF",
                    border_radius=18,
                )
            )
        self.graph_trace_actions.controls = controls

    def _choose_trace_tensor_handler(
        self,
        value_id: str,
    ) -> Callable[..., Any]:
        async def choose(event: ft.Event[ft.IconButton | ft.Button]) -> None:
            if self.session is None:
                return
            graph = self.session.document.main_graph
            value = graph.value(value_id)
            input_name = value.name or value.id
            files = await self.picker.pick_files(
                dialog_title=f"Choose a .npy tensor for {input_name}",
                allow_multiple=False,
                with_data=False,
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["npy"],
            )
            if not files:
                return
            selected = files[0].path
            if selected is None:
                self._show_error("Tensor-file tracing is available in the desktop app.")
                self.page.update()
                return
            try:
                binding = self.session.trace_tensor_input(input_name, selected)
            except (OSError, TypeError, ValueError) as error:
                self._show_error(f"Cannot use that tensor for {input_name}: {error}")
                self.page.update()
                return
            self._trace_input_bindings[input_name] = binding
            self.error_banner.visible = False
            self._refresh_graph_trace_actions()
            self._refresh_inspector(self._inspected_ids)
            self._refresh_trace_actions()
            self._set_status(
                f"{input_name} will use {Path(selected).name} on the next trace"
            )
            self.page.update()

        return choose

    def _reset_trace_tensor_handler(
        self,
        input_name: str,
        owner_id: str,
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def reset(event: ft.Event[ft.TextButton]) -> None:
            self._trace_input_bindings.pop(input_name, None)
            self._refresh_graph_trace_actions()
            self._refresh_inspector(frozenset({owner_id}))
            self._refresh_trace_actions()
            self._set_status(
                f"{input_name} will use deterministic random data on the next trace"
            )
            self.page.update()

        return reset

    def _refresh_trace_actions(self) -> None:
        session = self.session
        running = self.trace_job is not None and not self.trace_job.state.is_terminal
        busy = self._trace_preparing or running
        web = bool(getattr(self.page, "web", False))
        available = False
        reason = "Open an artifact to see tracing availability."
        if session is not None:
            status = session.capability(Capability.TRACING)
            self.trace_approval_notice.value = (
                f"Selecting Approve & run authorizes one isolated trace of "
                f"{session.title} using the graph inputs and four limits shown."
            )
            available = status.availability is Availability.AVAILABLE and not web
            reason = (
                "Web tracing awaits the Phase 8 isolated worker service."
                if web and status.availability is Availability.AVAILABLE
                else status.reason
            )
        else:
            self.trace_approval_notice.value = (
                "Review the selected inputs and limits. The run button approves "
                "exactly one isolated trace."
            )
        self.trace_progress.visible = busy
        self.run_trace_button.disabled = not available or busy
        self.cancel_trace_button.disabled = not running
        active_result = (
            session.trace(self.active_trace_id)
            if session is not None and self.active_trace_id is not None
            else None
        )
        if self.active_trace_comparison is not None:
            self.trace_status.value = (
                f"Compared {len(self.active_trace_comparison.nodes)} node(s); "
                "overlay shows maximum absolute error"
            )
        elif self.active_trace_id is None:
            self.trace_status.value = (
                f"{reason} Choose .npy tensors with the buttons on model inputs, "
                "or keep automatic all-valid masks and deterministic random data."
            )
        elif session is not None and active_result is not None:
            state = "Partial" if active_result.partial else "Complete"
            revision = active_result.key.revision_id or "base"
            stale = (
                " • not current revision"
                if active_result.key.revision_id != session.editing.current_revision_id
                else ""
            )
            self.trace_status.value = (
                f"{state} trace • revision {revision[:18]}{stale} • "
                f"{len(active_result.captured_value_ids)} readable value(s) • "
                f"{active_result.runtime}"
            )
        traces = session.traces() if session is not None else ()
        alternatives = [
            result
            for result in traces
            if result.id != self.active_trace_id
            and active_result is not None
            and result.key.input_specification_hash
            == active_result.key.input_specification_hash
        ]
        self.trace_compare_with.options = [
            ft.DropdownOption(
                key=result.id,
                text=(
                    "base"
                    if result.key.revision_id is None
                    else result.key.revision_id[:18]
                )
                + (" • partial" if result.partial else " • complete"),
            )
            for result in alternatives
        ]
        valid_ids = {result.id for result in alternatives}
        if self.trace_compare_with.value not in valid_ids:
            self.trace_compare_with.value = (
                alternatives[-1].id if alternatives else None
            )
        self.trace_compare_with.disabled = not alternatives or busy
        self.compare_trace_button.disabled = (
            self.active_trace_id is None or not alternatives or busy
        )

    def _on_run_trace(self, event: ft.Event[ft.Button]) -> None:
        session = self.session
        if session is None or self._trace_preparing:
            return
        if self.trace_job is not None and not self.trace_job.state.is_terminal:
            return
        if getattr(self.page, "web", False):
            self._show_error(
                "Web tracing is unavailable until the Phase 8 isolated worker "
                "service is deployed."
            )
            self.page.update()
            return
        seed_text = self.trace_seed.value or ""
        shapes_text = self.trace_shapes.value or ""
        wall_text = self.trace_wall_seconds.value or ""
        memory_text = self.trace_memory_mib.value or ""
        capture_text = self.trace_capture_mib.value or ""
        chunk_text = self.trace_chunk_kib.value or ""
        bindings = dict(self._trace_input_bindings)
        self._trace_preparing = True
        self.error_banner.visible = False
        self.active_trace_comparison = None
        self._refresh_trace_actions()
        self.trace_status.value = f"Preparing approved trace for {session.title}…"
        self._set_status("Preparing inputs and isolated trace worker…")
        self.page.update()

        def prepare() -> None:
            try:
                seed = int(seed_text)
                shapes = viewmodel.parse_shape_overrides(shapes_text)
                specification = session.default_trace_inputs(
                    seed=seed,
                    shapes=shapes,
                    bindings=bindings,
                )
                limits = TraceLimits(
                    wall_seconds=float(wall_text),
                    memory_bytes=int(memory_text) * 1024 * 1024,
                    capture_bytes=int(capture_text) * 1024 * 1024,
                    chunk_bytes=int(chunk_text) * 1024,
                )
                approval = TraceApproval.approve(
                    session.title,
                    session.document.source.content_hash,
                    specification,
                    limits,
                )
                request = TraceRequest(specification, limits, approval)
                job = session.trace_async(request)
            except (OSError, TypeError, ValueError) as error:
                if self.session is session:
                    self._trace_preparing = False
                    self._refresh_trace_actions()
                    self._show_error(f"Trace configuration is invalid: {error}")
                    self.page.update()
                return
            if self.session is not session:
                if not job.state.is_terminal:
                    job.cancel()
                return
            self._trace_preparing = False
            self.trace_job = job
            self._refresh_trace_actions()
            self.trace_status.value = (
                f"Loading trace for {session.title} • inputs "
                f"{specification.hash[:19]} • {limits.wall_seconds:g}s / "
                f"{limits.memory_bytes // (1024 * 1024)} MiB / "
                f"{limits.capture_bytes // (1024 * 1024)} MiB capture"
            )
            self._set_status("Running approved inference trace in an isolated worker…")
            self.page.update()
            self._watch_trace(job)

        self.page.run_thread(prepare)

    def _watch_trace(self, job: Job[TraceResult]) -> None:
        def wait() -> None:
            try:
                job.wait()
                if job.state.value == "succeeded":
                    result = job.result()
                    self.active_trace_id = result.id
                    self.active_trace_comparison = None
                    self.error_banner.visible = False
                    state = "partial" if result.partial else "complete"
                    self._set_status(
                        f"Trace {state}: {len(result.captured_value_ids)} "
                        "readable activation(s)"
                    )
                    if self.current_slice is not None:
                        self.renderer.replace_scene(
                            self._display_scene(self.current_slice),
                            self._current_viewport(),
                        )
                        self._refresh_graph_trace_actions()
                        self._rebuild_minimap()
                    self._autoload_activation_views(self._inspected_ids)
                    self._refresh_inspector(self._inspected_ids)
                elif job.state.value == "cancelled":
                    self._set_status("Trace cancelled; no partial trace was saved")
                else:
                    self._show_error(f"Trace failed with no saved state: {job.error}")
            except Exception as error:
                self._show_error(f"Could not present the trace: {error}")
            finally:
                if self.trace_job is job:
                    self.trace_job = None
                self._refresh_trace_actions()
                self.page.update()

        self.page.run_thread(wait)

    def _on_cancel_trace(self, event: ft.Event[ft.TextButton]) -> None:
        if self.trace_job is None or self.trace_job.state.is_terminal:
            return
        self.trace_job.cancel()
        self.cancel_trace_button.disabled = True
        self.trace_status.value = "Cancelling and discarding staged captures…"
        self._set_status("Cancelling trace at the next worker checkpoint…")
        self.page.update()

    def _on_compare_trace(self, event: ft.Event[ft.TextButton]) -> None:
        if (
            self.session is None
            or self.active_trace_id is None
            or self.trace_compare_with.value is None
        ):
            return
        try:
            job = self.session.compare_traces_async(
                self.trace_compare_with.value,
                self.active_trace_id,
            )
        except ValueError as error:
            self._show_error(f"Cannot compare traces: {error}")
            self.page.update()
            return
        self.trace_status.value = "Loading per-node trace comparison…"
        self.compare_trace_button.disabled = True

        def wait() -> None:
            job.wait()
            if job.state.value == "succeeded":
                self.active_trace_comparison = job.result()
                if self.current_slice is not None:
                    self.renderer.replace_scene(
                        self._display_scene(self.current_slice),
                        self._current_viewport(),
                    )
                    self._refresh_graph_trace_actions()
                self.trace_status.value = (
                    f"Compared {len(self.active_trace_comparison.nodes)} node(s); "
                    "overlay shows maximum absolute error"
                )
                self._set_status("Trace comparison overlay active")
            elif job.state.value == "cancelled":
                self._set_status("Trace comparison cancelled")
            else:
                self._show_error(f"Trace comparison failed: {job.error}")
            self._refresh_trace_actions()
            self.page.update()

        self.page.run_thread(wait)

    # -- transactional edit UI (Phase 4) ---------------------------------

    def _discard_stale_pending(self) -> None:
        """Drop a prepared edit once the selection leaves the node it targets.

        A validated transaction names its target when it is built, so keeping
        it armed after the user clicks elsewhere would apply it to a node they
        are no longer looking at.
        """
        if self.pending_edit is None and self.pending_transformation is None:
            return
        if self._selected_operator_id() == self.pending_target:
            return
        if self.session is not None:
            if self.pending_edit is not None:
                self.session.reject_edit(self.pending_edit)
            if self.pending_transformation is not None:
                self.session.reject_transformation(self.pending_transformation)
        self.pending_edit = None
        self.pending_transformation = None
        self.pending_target = None
        self.edit_findings.controls = [
            ft.Text(
                "The prepared change was discarded because the selection "
                "moved to another node. Validate again to re-arm it.",
                size=10,
            )
        ]
        self._set_status("Pending change discarded: selection changed")
        self._refresh_edit_actions()

    def _reselect_after_rebuild(self, node_id: str | None) -> None:
        """Re-select an operator after the scene was rebuilt.

        Rebuilding may land on a coarser detail level whose scene holds group
        glyphs rather than operators, so selecting the operator id directly
        would raise. Prefer the glyph that now represents it, and simply
        clear the selection when nothing does.
        """
        if node_id is None:
            return
        scene = self.renderer.scene
        if scene is None:
            return
        target = node_id
        if not scene.has_node(target):
            representative = (
                self.current_slice.representative_for(node_id)
                if self.current_slice is not None
                else None
            )
            if representative is None or not scene.has_node(representative):
                self.renderer.set_selection(frozenset())
                return
            target = representative
        self.renderer.set_selection(frozenset({target}))

    def _selected_operator_id(self) -> str | None:
        if (
            self.session is None
            or self.current_graph is None
            or len(self.renderer.selection) != 1
        ):
            return None
        (selected,) = self.renderer.selection
        graph = self.session.document.graphs[self.current_graph]
        if any(node.id == selected for node in graph.nodes):
            return selected
        if self.current_slice is None:
            return None
        members = self.current_slice.members_by_glyph.get(selected, frozenset())
        return next(iter(members)) if len(members) == 1 else None

    def _selected_weight_id(self) -> str:
        assert self.session is not None and self.current_graph is not None
        node_id = self._selected_operator_id()
        if node_id is None:
            raise ValueError("select exactly one operator before transforming")
        tensor_ids = viewmodel.node_tensor_ids(
            self.session.document,
            self.current_graph,
            node_id,
        )
        if not tensor_ids:
            raise ValueError("the selected operator has no initializer input")
        return tensor_ids[0]

    def _transformation_request(self) -> TransformationRequest:
        assert self.session is not None and self.current_graph is not None
        kind = self.transformation_kind.value or "weight-quantization"
        node_id = self._selected_operator_id()
        if node_id is None:
            raise ValueError("select exactly one operator before transforming")
        parameter = (self.transformation_parameter.value or "").strip()
        tensor_id = None if kind == "structured-pruning" else self._selected_weight_id()
        return viewmodel.transformation_request(
            kind=kind,
            graph_id=self.current_graph,
            node_id=node_id,
            tensor_id=tensor_id,
            granularity_value=(
                self.transformation_granularity.value or Granularity.PER_TENSOR.value
            ),
            axis_value=self.transformation_axis.value or "0",
            parameter=parameter,
        )

    def _on_preview_transformation(self, event: ft.Event[ft.Button]) -> None:
        assert self.session is not None
        try:
            proposal = self.session.prepare_transformation(
                self._transformation_request()
            )
        except Exception as error:
            self.pending_transformation = None
            self.transformation_findings.controls = [
                ft.Text(f"[error] {error}", size=10, color="#C0392B")
            ]
            self._set_status("Transformation preview failed")
        else:
            self.pending_transformation = proposal
            self.pending_target = self._selected_operator_id()
            controls: list[ft.Control] = [
                ft.Text("Validation", weight=ft.FontWeight.BOLD, size=10),
                *[
                    ft.Text(
                        f"[{finding.level.value}] {finding.code}: {finding.message}",
                        size=10,
                        color=(
                            "#C0392B"
                            if finding.level.value == "error"
                            else "#8A5A00"
                            if finding.level.value == "warning"
                            else None
                        ),
                    )
                    for finding in proposal.findings
                ],
            ]
            preview = proposal.preview
            if preview is not None:
                manifest = proposal.manifest
                assert manifest is not None
                controls.extend(
                    [
                        ft.Text(
                            "Capability report",
                            weight=ft.FontWeight.BOLD,
                            size=10,
                        ),
                        ft.Text(
                            f"Target: {manifest.target_runtime.value}; "
                            f"representation: "
                            f"{manifest.operator_representation.value}",
                            size=10,
                        ),
                        ft.Text(
                            "Mathematical conversion: "
                            + ("yes" if preview.mathematical_conversion else "no"),
                            size=10,
                        ),
                        ft.Text(
                            "Executable support: "
                            + (
                                "claimed (verified at export by checker and reopen)"
                                if preview.executable_graph
                                else "no"
                            ),
                            size=10,
                        ),
                        ft.Text(
                            "Storage reduction: "
                            + ("yes" if preview.storage_reduction else "no"),
                            size=10,
                        ),
                        ft.Text(
                            "Expected acceleration: "
                            + ("yes" if preview.expected_acceleration else "no"),
                            size=10,
                        ),
                        ft.Text(preview.acceleration_reason, size=10),
                        ft.Text(
                            f"Bytes: {preview.source_bytes:,} -> "
                            f"{preview.result_bytes:,}; max error "
                            f"{preview.max_abs_error:.6g}; sparsity "
                            f"{preview.before_sparsity:.1%} -> "
                            f"{preview.after_sparsity:.1%}",
                            size=10,
                        ),
                        ft.Text(preview.error_basis, size=10),
                    ]
                )
            self.transformation_findings.controls = controls
            self._set_status(
                "Transformation is ready to apply"
                if proposal.ok
                else "Transformation was rejected by validation"
            )
        self._refresh_edit_actions()
        self.page.update()

    def _on_commit_transformation(self, event: ft.Event[ft.Button]) -> None:
        if self.session is None or self.pending_transformation is None:
            return
        proposal = self.pending_transformation
        durability_warning: str | None = None
        try:
            self.session.commit_transformation(proposal)
        except SidecarPersistenceError as error:
            # The transformation IS applied and caches already follow it;
            # only the sidecar save failed. Warn about durability instead of
            # presenting an applied revision as a failed one.
            durability_warning = f"Transformation applied; durability warning: {error}"
        except Exception as error:
            self._show_error(f"Could not apply transformation: {error}")
            return
        self.pending_transformation = None
        selected = self._selected_operator_id()
        self._show_graph(self.current_graph or self.session.document.entry_graph)
        self._reselect_after_rebuild(selected)
        # Re-render the inspector so hex pages and tensor summaries reflect
        # the committed bytes rather than the pre-transformation state.
        self._refresh_inspector(self.renderer.selection)
        self._set_status(durability_warning or "Transformation committed atomically")
        self._show_committed_diff()
        self._refresh_edit_actions()
        self.page.update()

    def _on_reject_transformation(self, event: ft.Event[ft.TextButton]) -> None:
        if self.session is not None and self.pending_transformation is not None:
            self.session.reject_transformation(self.pending_transformation)
        self.pending_transformation = None
        self.transformation_findings.controls = []
        self._set_status("Prepared transformation rejected; the model was not changed")
        self._refresh_edit_actions()
        self.page.update()

    def _edit_request(self) -> EditRequest:
        assert self.session is not None and self.current_graph is not None
        node_id = self._selected_operator_id()
        if node_id is None:
            raise ValueError("select exactly one operator before editing")
        kind = self.edit_kind.value or "rename"
        primary = (self.edit_primary.value or "").strip()
        secondary = self.edit_secondary.value or ""
        port = int(self.edit_port.value or "0")
        if kind == "rename":
            return RenameNodeRequest(self.current_graph, node_id, primary)
        if kind == "attribute":
            attribute_kind = AttrKind(
                self.edit_attribute_kind.value or AttrKind.STRING.value
            )
            return SetAttributeRequest(
                self.current_graph,
                node_id,
                primary,
                attribute_kind,
                parse_attribute_value(attribute_kind, secondary),
            )
        if kind == "operator":
            return ReplaceOperatorRequest(
                self.current_graph,
                node_id,
                primary,
                secondary.strip(),
            )
        if kind == "insert":
            return InsertUnaryRequest(
                self.current_graph,
                node_id,
                port,
                primary,
                secondary.strip(),
            )
        if kind == "remove":
            return RemoveUnaryRequest(self.current_graph, node_id)
        return ReconnectInputRequest(
            self.current_graph,
            node_id,
            port,
            primary,
        )

    def _on_validate_edit(self, event: ft.Event[ft.Button]) -> None:
        assert self.session is not None
        try:
            request = self._edit_request()
            transaction = self.session.prepare_edit(request)
        except Exception as error:
            self.pending_edit = None
            self.edit_findings.controls = [
                ft.Text(f"[error] {error}", size=10, color="#C0392B")
            ]
            self._set_status("Edit validation failed")
        else:
            self.pending_edit = transaction
            self.pending_target = self._selected_operator_id()
            self.edit_findings.controls = [
                ft.Text("Preview", weight=ft.FontWeight.BOLD, size=10),
                *[ft.Text(summary, size=10) for summary in transaction.summaries],
                ft.Text("Capability impact", weight=ft.FontWeight.BOLD, size=10),
                *[
                    ft.Text(change, size=10)
                    for change in transaction.capability_changes
                ],
                ft.Text("Validation", weight=ft.FontWeight.BOLD, size=10),
                *[
                    ft.Text(
                        f"[{finding.level.value}] {finding.code}: {finding.message}",
                        size=10,
                        color=(
                            "#C0392B"
                            if finding.level.value == "error"
                            else "#8A5A00"
                            if finding.level.value == "warning"
                            else None
                        ),
                    )
                    for finding in transaction.findings
                ],
            ]
            self._set_status(
                "Edit is ready to commit"
                if transaction.ok
                else "Edit was rejected by validation"
            )
        self._refresh_edit_actions()
        self.page.update()

    def _on_commit_edit(self, event: ft.Event[ft.Button]) -> None:
        if self.session is None or self.pending_edit is None:
            return
        transaction = self.pending_edit
        durability_warning: str | None = None
        try:
            self.session.commit_edit(transaction)
        except SidecarPersistenceError as error:
            # The edit IS applied and caches already follow it; only the
            # sidecar save failed. Warn about durability instead of
            # presenting an applied revision as a failed one.
            durability_warning = f"Edit committed; durability warning: {error}"
        except Exception as error:
            self._show_error(f"Could not commit edit: {error}")
            return
        self.pending_edit = None
        selected = self._selected_operator_id()
        self._show_graph(self.current_graph or self.session.document.entry_graph)
        self._reselect_after_rebuild(selected)
        # The inspector still shows the pre-commit document (a renamed node
        # keeps its old name, hex pages their old bytes) until re-rendered.
        self._refresh_inspector(self.renderer.selection)
        self._set_status(durability_warning or "Edit committed atomically")
        self._show_committed_diff()
        self._refresh_edit_actions()
        self.page.update()

    def _on_reject_edit(self, event: ft.Event[ft.TextButton]) -> None:
        if self.session is not None and self.pending_edit is not None:
            self.session.reject_edit(self.pending_edit)
        self.pending_edit = None
        self.edit_findings.controls = []
        self._set_status("Prepared edit rejected; the model was not changed")
        self._refresh_edit_actions()
        self.page.update()

    def _on_undo_edit(self, event: ft.Event[ft.TextButton]) -> None:
        if self.session is None:
            return
        durability_warning: str | None = None
        try:
            self.session.undo_edit()
        except SidecarPersistenceError as error:
            # The undo IS applied; only the sidecar save failed.
            durability_warning = f"Edit undone; durability warning: {error}"
        except Exception as error:
            self._show_error(f"Could not undo: {error}")
            return
        self.pending_edit = None
        self.pending_transformation = None
        self._show_graph(self.current_graph or self.session.document.entry_graph)
        self._refresh_inspector(self.renderer.selection)
        self._set_status(durability_warning or "Edit undone")
        self._show_committed_diff()
        self._refresh_edit_actions()
        self.page.update()

    def _on_redo_edit(self, event: ft.Event[ft.TextButton]) -> None:
        if self.session is None:
            return
        durability_warning: str | None = None
        try:
            self.session.redo_edit()
        except SidecarPersistenceError as error:
            # The redo IS applied; only the sidecar save failed.
            durability_warning = f"Edit redone; durability warning: {error}"
        except Exception as error:
            self._show_error(f"Could not redo: {error}")
            return
        self.pending_edit = None
        self.pending_transformation = None
        self._show_graph(self.current_graph or self.session.document.entry_graph)
        self._refresh_inspector(self.renderer.selection)
        self._set_status(durability_warning or "Edit redone")
        self._show_committed_diff()
        self._refresh_edit_actions()
        self.page.update()

    def _show_committed_diff(self) -> None:
        if self.session is None:
            self.edit_findings.controls = []
            return
        preview = self.session.editing.preview()
        controls: list[ft.Control] = []
        if preview.commands:
            controls.append(
                ft.Text("Committed diff", weight=ft.FontWeight.BOLD, size=10)
            )
            controls.extend(ft.Text(command, size=10) for command in preview.commands)
        for tensor in preview.tensors:
            elements = (
                "unknown"
                if tensor.elements_changed is None
                else str(tensor.elements_changed)
            )
            controls.append(
                ft.Text(
                    f"{tensor.tensor_id}: {tensor.span_count} span(s), "
                    f"{tensor.bytes_changed} byte(s), {elements} element(s)",
                    size=10,
                )
            )
        for transformation in preview.transformations:
            result = transformation.preview
            controls.extend(
                [
                    ft.Text(
                        f"{transformation.manifest.kind.value}: "
                        f"{transformation.target_id}",
                        weight=ft.FontWeight.BOLD,
                        size=10,
                    ),
                    ft.Text(
                        "Mathematical conversion: "
                        + ("yes" if result.mathematical_conversion else "no")
                        + "; executable support: "
                        + ("yes" if result.executable_graph else "no")
                        + "; storage reduction: "
                        + ("yes" if result.storage_reduction else "no")
                        + "; expected acceleration: "
                        + ("yes" if result.expected_acceleration else "no"),
                        size=10,
                    ),
                ]
            )
        self.edit_findings.controls = controls

    async def _on_export_clicked(self, event: ft.Event[ft.TextButton]) -> None:
        await self._export_current_session()

    async def _on_save_model(self, event: ft.Event[ft.Button]) -> None:
        await self._export_current_session()

    async def _export_current_session(self) -> None:
        session = self.session
        if session is None:
            return
        if self.export_job is not None and not self.export_job.state.is_terminal:
            return
        plan = session.export_plan()
        if not plan.available:
            self._set_status(f"Export unavailable: {plan.reason}")
            self.page.update()
            return
        destination = await self.picker.save_file(
            dialog_title=plan.dialog_title,
            file_name=plan.file_name,
            allowed_extensions=list(plan.allowed_extensions),
        )
        if destination is None:
            if getattr(self.page, "web", False):
                self._set_status(
                    "This web client cannot publish a multi-file export "
                    "directly; use the desktop shell."
                )
            else:
                # On desktop a None destination simply means the user
                # dismissed the dialog; that is not an error.
                self._set_status("Export cancelled")
            self.page.update()
            return
        # The model may have been closed while the native save dialog was
        # open. Never start a job against a session that is no longer active.
        if self.session is not session or session.closed:
            return
        revision_id = session.editing.current_revision_id
        self._set_status(f"Exporting to {destination}…")
        job = session.export_artifact_async(destination)
        self.export_job = job
        self._refresh_edit_actions()

        def wait() -> None:
            job.wait()
            active = self.session is session and not session.closed
            if job.state.value == "succeeded":
                outcome = job.result()
                if active:
                    detail = (
                        f"; report: {outcome.report_path.name}"
                        if outcome.report_path is not None
                        else ""
                    )
                    current_revision = session.editing.current_revision_id
                    newer = ""
                    if current_revision == revision_id:
                        self._saved_revision_id = revision_id
                    else:
                        newer = "; newer changes remain unsaved"
                    self._set_status(
                        f"Exported {len(outcome.written_files)} artifact file(s) "
                        f"({outcome.fidelity.value}){detail}{newer}"
                    )
            elif job.state.value == "cancelled" and active:
                self._set_status("Export cancelled")
            elif active:
                self._show_error(f"Export failed: {job.error}")
            if self.export_job is job:
                self.export_job = None
            if active:
                self._refresh_edit_actions()
                self.page.update()

        self.page.run_thread(wait)
        self.page.update()

    def _refresh_edit_actions(self) -> None:
        session = self.session
        exporting = (
            self.export_job is not None and not self.export_job.state.is_terminal
        )
        selected = self._selected_operator_id() is not None
        self.validate_edit_button.disabled = (
            session is None or not selected or exporting
        )
        self.commit_edit_button.disabled = not (
            self.pending_edit is not None and self.pending_edit.ok and not exporting
        )
        self.reject_edit_button.disabled = self.pending_edit is None
        self.preview_transformation_button.disabled = (
            session is None or not selected or exporting
        )
        self.commit_transformation_button.disabled = not (
            self.pending_transformation is not None
            and self.pending_transformation.ok
            and not exporting
        )
        self.reject_transformation_button.disabled = self.pending_transformation is None
        self.undo_edit_button.disabled = (
            session is None or not session.editing.can_undo or exporting
        )
        self.redo_edit_button.disabled = (
            session is None or not session.editing.can_redo or exporting
        )
        plan = session.export_plan() if session is not None else None
        self.export_button.disabled = plan is None or not plan.available or exporting
        self.export_button.content = (
            "Export…"
            if plan is None
            else "Export unavailable"
            if not plan.available
            else (
                "Export ONNX…"
                if session is not None
                and session.document.artifact_kind is ArtifactKind.ONNX_MODEL
                else "Export weights…"
            )
        )
        self.export_button.tooltip = None if plan is None else plan.reason
        current_revision = (
            session.editing.current_revision_id if session is not None else None
        )
        has_unsaved_revision = (
            session is not None and current_revision != self._saved_revision_id
        )
        self.save_model_button.visible = has_unsaved_revision
        self.save_model_button.disabled = (
            not has_unsaved_revision or plan is None or not plan.available or exporting
        )
        self.save_model_button.tooltip = (
            "Export in progress"
            if exporting
            else plan.reason
            if plan is not None and not plan.available
            else "Export the current revision to a new artifact"
        )
        self.close_model_button.visible = session is not None

    # -- tensor inspector (task P3.4) --------------------------------------

    def _tensor_rows(self, node_id: str) -> list[ft.Control]:
        """Inspector rows for every weight tensor feeding the selected node."""
        assert self.session is not None and self.current_graph is not None
        try:
            tensor_ids = viewmodel.node_tensor_ids(
                self.session.document, self.current_graph, node_id
            )
        except KeyError:
            return []
        if not tensor_ids:
            return []
        rows: list[ft.Control] = [
            ft.Divider(height=8, color=_BORDER),
            overview.section_heading(
                "Weights & tensors",
                ft.Icons.MEMORY_ROUNDED,
                trailing=str(len(tensor_ids)),
            ),
        ]
        rows.extend(
            self._tensor_card(
                tensor_id,
                node_id,
                # The first card starts open; after that the user's own
                # expand/collapse choices survive every inspector rebuild.
                expanded=self._tensor_card_expanded.get(tensor_id, position == 0),
            )
            for position, tensor_id in enumerate(tensor_ids)
        )
        return rows

    def _activation_rows(self, node_id: str) -> list[ft.Control]:
        """Lazy captured input/output cards beside the node's weights."""
        if self.session is None or self.active_trace_id is None:
            return []
        try:
            records = self.session.node_activations(self.active_trace_id, node_id)
        except (KeyError, ValueError):
            records = ()
        return self._activation_record_rows(
            records,
            owner_id=node_id,
            title="Captured activations",
        )

    def _activation_records_for_selection(
        self,
        ids: frozenset[str],
    ) -> tuple[ActivationRecord, ...]:
        """Readable trace records represented by one selected graph glyph."""
        if (
            self.session is None
            or self.active_trace_id is None
            or len(ids) != 1
            or self.current_graph != self.session.document.entry_graph
        ):
            return ()
        (glyph_id,) = ids
        value_ids = self._trace_values_for_glyph(glyph_id)
        if value_ids:
            result = self.session.trace(self.active_trace_id)
            by_value = {record.value_id: record for record in result.records}
            return tuple(
                record
                for value_id in value_ids
                if (record := by_value.get(value_id)) is not None
            )
        try:
            return self.session.node_activations(self.active_trace_id, glyph_id)
        except (KeyError, ValueError):
            pass
        if self.current_slice is None:
            return ()
        members = self.current_slice.members_by_glyph.get(glyph_id, frozenset())
        value_ids = self._activation_value_ids_for_members(members)
        if not value_ids:
            return ()
        result = self.session.trace(self.active_trace_id)
        by_value = {record.value_id: record for record in result.records}
        return tuple(
            record
            for value_id in value_ids
            if (record := by_value.get(value_id)) is not None
        )

    def _autoload_activation_views(self, ids: frozenset[str]) -> None:
        """Start bounded activation views as soon as a traced glyph is selected."""
        if self.active_trace_id is None or len(ids) != 1:
            return
        (owner_id,) = ids
        for record in self._activation_records_for_selection(ids):
            if record.readable:
                self._request_activation_views(
                    self.active_trace_id,
                    record.value_id,
                    owner_id,
                )

    def _group_activation_rows(
        self,
        members: frozenset[str],
        *,
        owner_id: str,
    ) -> list[ft.Control]:
        """Captured values crossing a selected block/region boundary."""
        if (
            self.session is None
            or self.active_trace_id is None
            or self.current_graph != self.session.document.entry_graph
        ):
            return []
        value_ids = self._activation_value_ids_for_members(members)
        return self._activation_rows_for_values(
            value_ids,
            owner_id=owner_id,
            title="Block inputs & outputs",
        )

    def _activation_value_ids_for_members(
        self,
        members: frozenset[str],
    ) -> tuple[str, ...]:
        """Values crossing the boundary of an aggregated graph glyph."""
        if (
            self.session is None
            or not members
            or self.current_graph != self.session.document.entry_graph
        ):
            return ()
        graph = self.session.document.main_graph
        value_ids: set[str] = set()
        for node_id in members:
            node = graph.node(node_id)
            for value_id in node.inputs:
                producer = graph.producer(value_id)
                if value_id not in graph.initializers and (
                    producer is None or producer[0] not in members
                ):
                    value_ids.add(value_id)
            for value_id in node.outputs:
                if value_id in graph.outputs or any(
                    consumer not in members
                    for consumer, _port in graph.consumers(value_id)
                ):
                    value_ids.add(value_id)
        return tuple(sorted(value_ids))

    def _activation_rows_for_values(
        self,
        value_ids: tuple[str, ...],
        *,
        owner_id: str,
        title: str,
    ) -> list[ft.Control]:
        """Activation cards for selected connection or boundary values."""
        if self.session is None:
            return []
        if self.active_trace_id is None:
            return [
                ft.Divider(height=8, color=_BORDER),
                overview.section_heading(
                    title,
                    ft.Icons.PLAY_CIRCLE_OUTLINE_ROUNDED,
                    trailing="not run",
                ),
                ft.Container(
                    content=ft.Text(
                        "Run a trace once, then click any node, connection, input, "
                        "or output to inspect its activation here.",
                        size=10,
                        color=_INFO,
                    ),
                    padding=10,
                    bgcolor="#EFF8FF",
                    border_radius=9,
                ),
            ]
        result = self.session.trace(self.active_trace_id)
        records_by_value = {record.value_id: record for record in result.records}
        records = tuple(
            record
            for value_id in dict.fromkeys(value_ids)
            if (record := records_by_value.get(value_id)) is not None
        )
        return self._activation_record_rows(
            records,
            owner_id=owner_id,
            title=title,
            expand_all=True,
        )

    def _activation_record_rows(
        self,
        records: tuple[ActivationRecord, ...],
        *,
        owner_id: str,
        title: str,
        expand_all: bool = False,
    ) -> list[ft.Control]:
        assert self.active_trace_id is not None
        rows: list[ft.Control] = [
            ft.Divider(height=8, color=_BORDER),
            overview.section_heading(
                title,
                ft.Icons.PLAY_CIRCLE_OUTLINE_ROUNDED,
                trailing=str(len(records)),
            ),
        ]
        if not records:
            rows.append(
                ft.Text(
                    "No activation was captured for this node. The trace may "
                    "have used a narrowed selection or stopped earlier.",
                    size=10,
                    color=_MUTED,
                )
            )
            return rows
        rows.extend(
            self._activation_card(
                self.active_trace_id,
                record,
                owner_id,
                expanded=expand_all or record.role == "node-output",
            )
            for record in records
        )
        return rows

    def _activation_card(
        self,
        trace_id: str,
        record: ActivationRecord,
        node_id: str,
        *,
        expanded: bool,
    ) -> ft.Control:
        assert self.session is not None
        lines = [
            ("Value", record.value_name),
            ("Role", record.role),
            ("State", record.state.value),
            ("Dtype", record.element_type),
            ("Shape", str(list(record.shape))),
            (
                "Captured bytes",
                f"{record.stored_byte_length:,} / {record.full_byte_length:,}",
            ),
        ]
        if record.reason:
            lines.append(("Disclosure", record.reason))
        controls: list[ft.Control] = [
            overview.metadata_section(
                tuple(lines),
                title="Activation metadata",
                icon=ft.Icons.DATA_ARRAY_ROUNDED,
                role=f"activation-metadata:{record.value_id}",
            )
        ]
        key = (trace_id, record.value_id)
        if record.readable:
            try:
                view = self.session.activations(trace_id)
                raw = view.read(
                    record.value_id,
                    length=min(record.stored_byte_length, 64),
                )
                preview = viewmodel.preview_values_text(record.element_type, raw)
                controls.append(
                    ft.Text(
                        f"Preview: {preview}",
                        size=10,
                        font_family="monospace",
                        color=_INK,
                    )
                )
            except Exception as error:
                controls.append(
                    ft.Text(f"Preview error: {error}", size=10, color=_DANGER)
                )
            stats = self._activation_statistics.get(key)
            if stats is not None:
                controls.append(
                    overview.metadata_section(
                        viewmodel.statistics_lines(stats),
                        title="Statistics",
                        icon=ft.Icons.QUERY_STATS_ROUNDED,
                        role=f"activation-statistics:{record.value_id}",
                    )
                )
                for label, count, fraction in viewmodel.histogram_rows(stats):
                    bar = "█" * max(1, round(fraction * 20)) if count else ""
                    controls.append(
                        ft.Text(
                            f"{label:>24} {bar} {count}",
                            size=10,
                            font_family="monospace",
                            color=_INK,
                        )
                    )
            elif (trace_id, record.value_id, "statistics") in self._activation_loading:
                controls.append(ft.ProgressRing(width=18, height=18))
                controls.append(ft.Text("Loading activation statistics…", size=10))
            else:
                controls.append(
                    ft.TextButton(
                        content="Compute statistics",
                        icon=ft.Icons.QUERY_STATS_ROUNDED,
                        on_click=self._activation_stats_handler(
                            trace_id, record.value_id, node_id
                        ),
                    )
                )
            visualizations = self._activation_visualizations.get(key)
            if visualizations is not None:
                if visualizations:
                    controls.append(
                        ft.FilledButton(
                            content="Open large view",
                            icon=ft.Icons.OPEN_IN_FULL_ROUNDED,
                            on_click=self._open_activation_views_handler(
                                trace_id,
                                record,
                            ),
                        )
                    )
                for plot in visualizations:
                    controls.append(self._activation_plot_control(plot))
                    controls.append(
                        overview.metadata_section(
                            (
                                ("Kind", plot.kind.value),
                                (
                                    "Bins"
                                    if plot.kind.value == "histogram"
                                    else "Displayed shape",
                                    (
                                        str(len(plot.values))
                                        if plot.kind.value == "histogram"
                                        else str(list(plot.shape))
                                    ),
                                ),
                                ("Colormap", plot.colormap),
                                ("Normalization", plot.normalization),
                                ("Downsampling", plot.downsampling),
                                (
                                    "Capture",
                                    "partial" if plot.partial else "complete",
                                ),
                            ),
                            title=plot.title,
                            icon=ft.Icons.INSERT_CHART_OUTLINED_ROUNDED,
                            role=(
                                f"activation-view:{record.value_id}:{plot.kind.value}"
                            ),
                        )
                    )
                if not visualizations:
                    controls.append(
                        ft.Text(
                            "This activation has no finite values to visualize.",
                            size=10,
                            color=_MUTED,
                        )
                    )
            elif (
                trace_id,
                record.value_id,
                "visualization",
            ) in self._activation_loading:
                controls.append(ft.Text("Loading activation views…", size=10))
            elif view_error := self._activation_visualization_errors.get(key):
                controls.extend(
                    [
                        ft.Text(
                            f"Activation views could not be built: {view_error}",
                            size=10,
                            color=_DANGER,
                        ),
                        ft.TextButton(
                            content="Retry activation views",
                            icon=ft.Icons.REFRESH_ROUNDED,
                            on_click=self._activation_views_handler(
                                trace_id,
                                record.value_id,
                                node_id,
                            ),
                        ),
                    ]
                )
            else:
                controls.append(
                    ft.TextButton(
                        content="Build activation views now",
                        icon=ft.Icons.INSERT_CHART_OUTLINED_ROUNDED,
                        on_click=self._activation_views_handler(
                            trace_id, record.value_id, node_id
                        ),
                    )
                )
        else:
            controls.append(
                ft.Container(
                    content=ft.Text(
                        record.reason
                        or f"This capture is {record.state.value} and cannot be read.",
                        size=10,
                        color=_WARNING,
                    ),
                    padding=8,
                    bgcolor=_WARNING_SOFT,
                    border_radius=8,
                )
            )
        return ft.ExpansionTile(
            data=f"activation-card:{record.value_id}",
            title=ft.Text(record.value_name, size=12, weight=ft.FontWeight.W_700),
            subtitle=ft.Text(
                f"{record.role} • {record.state.value}",
                size=10,
                color=_MUTED,
            ),
            controls=[ft.Column(controls=controls, spacing=8, tight=True)],
            controls_padding=ft.Padding.only(left=8, right=8, bottom=12),
            expanded=expanded,
            maintain_state=True,
            dense=True,
            bgcolor=_SUBTLE,
            collapsed_bgcolor=_SUBTLE,
        )

    @staticmethod
    def _activation_plot_control(
        view: ActivationVisualization,
        *,
        width: float = 180.0,
        height: float = 82.0,
    ) -> ft.Control:
        """Render a bounded adapter view; sampling/color choices came from core."""
        if view.layer_pngs:
            return build_activation_layer_viewer(
                view,
                width=width,
                height=max(height, 190.0),
                panel_color=_PANEL,
                border_color=_BORDER,
                accent_color=_ACCENT,
                muted_color=_MUTED,
            )
        if view.raster_png and view.raster_size is not None:
            raster_width, raster_height = view.raster_size
            scale = min(
                width / max(raster_width, 1),
                height / max(raster_height, 1),
            )
            rendered_width = max(1.0, raster_width * scale)
            rendered_height = max(1.0, raster_height * scale)
            return ft.Container(
                data=f"activation-plot:{view.kind.value}",
                content=ft.Image(
                    src=view.raster_png,
                    width=rendered_width,
                    height=rendered_height,
                    fit=ft.BoxFit.FILL,
                    filter_quality=ft.FilterQuality.NONE,
                ),
                width=rendered_width,
                height=rendered_height,
                alignment=ft.Alignment.CENTER,
                padding=8,
                bgcolor=_PANEL,
                border=ft.Border.all(1, _BORDER),
                border_radius=8,
            )
        shapes: list[cv.Shape] = []
        if view.kind.value == "feature-map-grid" and len(view.shape) == 3:
            maps, rows, columns = view.shape
            grid_columns = max(1, math.ceil(math.sqrt(maps)))
            grid_rows = max(1, math.ceil(maps / grid_columns))
            large = width >= 500
            gap = 6.0 if large else 2.0
            cell = min(
                (width - gap * (grid_columns - 1)) / max(grid_columns * columns, 1),
                (height - gap * (grid_rows - 1)) / max(grid_rows * rows, 1),
            )
            if not large:
                cell = min(cell, 3.0)
            cell = max(cell, 0.5)
            map_width = columns * cell
            map_height = rows * cell
            count = maps * rows * columns
            for index, color in enumerate(view.colors[:count]):
                map_index, within = divmod(index, rows * columns)
                row, column = divmod(within, columns)
                map_column = map_index % grid_columns
                map_row = map_index // grid_columns
                shapes.append(
                    cv.Rect(
                        x=map_column * (map_width + gap) + column * cell,
                        y=map_row * (map_height + gap) + row * cell,
                        width=max(0.5, cell - 0.35),
                        height=max(0.5, cell - 0.35),
                        paint=ft.Paint(color=color, style=ft.PaintingStyle.FILL),
                    )
                )
            width = grid_columns * map_width + (grid_columns - 1) * gap
            height = grid_rows * map_height + (grid_rows - 1) * gap
        elif view.kind.value in {"heatmap", "attention-map"} and len(view.shape) >= 2:
            rows, columns = view.shape[-2:]
            cell = min(
                width / max(columns, 1),
                height / max(rows, 1),
            )
            if width < 500:
                cell = min(cell, 7.0)
            count = rows * columns
            for index, color in enumerate(view.colors[:count]):
                shapes.append(
                    cv.Rect(
                        x=(index % columns) * cell,
                        y=(index // columns) * cell,
                        width=max(1.0, cell - 0.5),
                        height=max(1.0, cell - 0.5),
                        paint=ft.Paint(color=color, style=ft.PaintingStyle.FILL),
                    )
                )
            width = columns * cell
            height = rows * cell
        elif view.kind.value == "histogram":
            maximum = max(view.values, default=0.0)
            bar_width = width / max(len(view.values), 1)
            for index, value in enumerate(view.values):
                bar_height = 0.0 if maximum == 0 else value / maximum * height
                shapes.append(
                    cv.Rect(
                        x=index * bar_width,
                        y=height - bar_height,
                        width=max(1.0, bar_width - 1.0),
                        height=bar_height,
                        paint=ft.Paint(
                            color=view.colors[index],
                            style=ft.PaintingStyle.FILL,
                        ),
                    )
                )
        else:
            finite = [value for value in view.values if math.isfinite(value)]
            minimum = min(finite, default=0.0)
            maximum = max(finite, default=0.0)
            span = maximum - minimum

            def point(index: int, value: float) -> tuple[float, float]:
                x = index / max(len(view.values) - 1, 1) * width
                normalized = 0.5 if span == 0 else (value - minimum) / span
                return x, height - normalized * height

            previous: tuple[float, float] | None = None
            for index, value in enumerate(view.values):
                if not math.isfinite(value):
                    previous = None
                    continue
                current = point(index, value)
                if previous is not None:
                    shapes.append(
                        cv.Line(
                            x1=previous[0],
                            y1=previous[1],
                            x2=current[0],
                            y2=current[1],
                            paint=ft.Paint(
                                color=view.colors[index],
                                stroke_width=2,
                                style=ft.PaintingStyle.STROKE,
                            ),
                        )
                    )
                previous = current
        return ft.Container(
            data=f"activation-plot:{view.kind.value}",
            content=cv.Canvas(shapes=shapes, width=width, height=height),
            alignment=ft.Alignment.CENTER,
            padding=8,
            bgcolor=_PANEL,
            border=ft.Border.all(1, _BORDER),
            border_radius=8,
        )

    def _activation_stats_handler(
        self,
        trace_id: str,
        value_id: str,
        node_id: str,
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def start(event: ft.Event[ft.TextButton]) -> None:
            if self.session is None:
                return
            loading = (trace_id, value_id, "statistics")
            self._activation_loading.add(loading)
            job = self.session.compute_activation_statistics_async(trace_id, value_id)
            self._refresh_inspector(frozenset({node_id}))
            self.page.update()

            def wait() -> None:
                job.wait()
                self._activation_loading.discard(loading)
                if job.state.value == "succeeded":
                    self._activation_statistics[(trace_id, value_id)] = job.result()
                elif job.state.value == "failed":
                    self._show_error(f"Activation statistics failed: {job.error}")
                if self._inspected_ids == frozenset({node_id}):
                    self._refresh_inspector(self._inspected_ids)
                self.page.update()

            self.page.run_thread(wait)

        return start

    def _request_activation_views(
        self,
        trace_id: str,
        value_id: str,
        owner_id: str,
        *,
        force: bool = False,
    ) -> bool:
        """Queue one bounded visualization job unless it is already resolved."""
        session = self.session
        if session is None:
            return False
        key = (trace_id, value_id)
        loading = (trace_id, value_id, "visualization")
        if (
            key in self._activation_visualizations
            or loading in self._activation_loading
        ):
            return False
        if key in self._activation_visualization_errors and not force:
            return False
        self._activation_visualization_errors.pop(key, None)
        self._activation_loading.add(loading)
        try:
            job = session.activation_visualizations_async(
                trace_id,
                value_id,
                attention=self._node_is_attention(owner_id),
            )
        except Exception as error:
            self._activation_loading.discard(loading)
            self._activation_visualization_errors[key] = str(error)
            return False

        def wait() -> None:
            job.wait()
            self._activation_loading.discard(loading)
            active = self.session is session and self.active_trace_id == trace_id
            if active and job.state.value == "succeeded":
                self._activation_visualizations[key] = job.result()
                self._activation_visualization_errors.pop(key, None)
            elif active and job.state.value == "failed":
                self._activation_visualization_errors[key] = str(job.error)
                self._set_status(f"Activation view failed: {job.error}")
            if active and self._inspected_ids == frozenset({owner_id}):
                self._refresh_inspector(self._inspected_ids)
            if active:
                self.page.update()

        self.page.run_thread(wait)
        return True

    def _activation_views_handler(
        self,
        trace_id: str,
        value_id: str,
        node_id: str,
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def start(event: ft.Event[ft.TextButton]) -> None:
            if self._request_activation_views(
                trace_id,
                value_id,
                node_id,
                force=True,
            ):
                self._refresh_inspector(frozenset({node_id}))
                self.page.update()

        return start

    def _open_activation_views_handler(
        self,
        trace_id: str,
        record: ActivationRecord,
    ) -> Callable[[ft.Event[ft.Button]], None]:
        def open_overlay(event: ft.Event[ft.Button]) -> None:
            views = self._activation_visualizations.get((trace_id, record.value_id))
            if not views:
                self._set_status("Activation views are not ready yet")
                self.page.update()
                return
            rows: list[ft.Control] = [
                overview.metadata_section(
                    (
                        ("Value", record.value_name),
                        ("Role", record.role),
                        ("Dtype", record.element_type),
                        ("Shape", str(list(record.shape))),
                        (
                            "Captured bytes",
                            f"{record.stored_byte_length:,} / "
                            f"{record.full_byte_length:,}",
                        ),
                    ),
                    title="Captured activation",
                    icon=ft.Icons.DATA_ARRAY_ROUNDED,
                    role=f"activation-overlay-metadata:{record.value_id}",
                )
            ]
            for plot in views:
                rows.extend(
                    [
                        overview.section_heading(
                            plot.title,
                            ft.Icons.INSERT_CHART_OUTLINED_ROUNDED,
                            trailing=plot.kind.value,
                        ),
                        self._activation_plot_control(
                            plot,
                            width=780.0,
                            height=420.0,
                        ),
                        overview.metadata_section(
                            (
                                ("Source shape", str(list(plot.source_shape))),
                                (
                                    "Bins"
                                    if plot.kind.value == "histogram"
                                    else "Displayed shape",
                                    (
                                        str(len(plot.values))
                                        if plot.kind.value == "histogram"
                                        else str(list(plot.shape))
                                    ),
                                ),
                                ("Colormap", plot.colormap),
                                ("Normalization", plot.normalization),
                                ("Downsampling", plot.downsampling),
                                (
                                    "Capture",
                                    "partial" if plot.partial else "complete",
                                ),
                            ),
                            title="View details",
                            icon=ft.Icons.INFO_OUTLINE_ROUNDED,
                            role=(
                                f"activation-overlay-view:{record.value_id}:"
                                f"{plot.kind.value}"
                            ),
                        ),
                    ]
                )
            dialog = ft.AlertDialog(
                data=f"activation-overlay:{record.value_id}",
                modal=True,
                title=ft.Text(
                    f"Activation · {record.value_name}",
                    weight=ft.FontWeight.W_700,
                ),
                content=ft.Container(
                    width=860,
                    height=620,
                    content=ft.ListView(
                        controls=rows,
                        spacing=14,
                        expand=True,
                    ),
                ),
                actions=[
                    ft.TextButton(
                        content="Close",
                        icon=ft.Icons.CLOSE_ROUNDED,
                        on_click=self._close_activation_overlay,
                    )
                ],
                scrollable=False,
            )
            self.page.show_dialog(dialog)

        return open_overlay

    def _close_activation_overlay(self, event: ft.Event[ft.TextButton]) -> None:
        self.page.pop_dialog()

    def _node_is_attention(self, node_id: str) -> bool:
        if self.session is None or self.current_graph is None:
            return False
        hierarchy = self.session.graph_hierarchy(self.current_graph)
        return any(
            node_id in group.members
            and "attention"
            in " ".join(
                (
                    group.label,
                    group.explanation,
                    *(evidence.code for evidence in group.evidence),
                )
            ).lower()
            for group in hierarchy.groups
        )

    def _tensor_card(
        self, tensor_id: str, node_id: str, *, expanded: bool
    ) -> ft.Control:
        assert self.session is not None
        session = self.session
        tensor = session.document.tensors[tensor_id]
        materialization = (
            "generated"
            if tensor.storage is Storage.GENERATED
            else session.store.materialization(tensor_id).value
        )
        byte_length = session.editing.byte_length(tensor_id)
        tensor_controls: list[ft.Control] = [
            overview.metadata_section(
                viewmodel.tensor_lines(
                    session.document,
                    tensor_id,
                    materialization=materialization,
                    byte_length=byte_length,
                ),
                title="Tensor metadata",
                icon=ft.Icons.DATA_OBJECT_ROUNDED,
                role=f"tensor-metadata:{tensor_id}",
            )
        ]
        requires_consent = (
            tensor.storage is not Storage.GENERATED
            and materialization == "full-parse"
            and tensor_id not in self._typed_preview_consent
        )
        if requires_consent:
            tensor_controls.append(
                ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.Text(
                                "Inspection requires materialization",
                                size=11,
                                weight=ft.FontWeight.W_700,
                                color=_WARNING,
                            ),
                            ft.Text(
                                "This tensor is stored in typed protobuf fields. "
                                "Loading values will parse the full tensor once.",
                                size=10,
                                color=_MUTED,
                            ),
                            self._preview_row(tensor_id),
                        ],
                        spacing=6,
                        tight=True,
                    ),
                    padding=10,
                    bgcolor=_WARNING_SOFT,
                    border=ft.Border.all(1, "#FEDF89"),
                    border_radius=10,
                )
            )
        else:
            tensor_controls.extend(
                [
                    self._tensor_visualization(tensor_id),
                    self._hex_editor(tensor_id, node_id),
                ]
            )
        tensor_controls.extend(self._tensor_statistics_controls(tensor_id, node_id))

        short_name = tensor_id.rsplit("#", 1)[-1]
        shape = " x ".join(str(dimension) for dimension in tensor.dims) or "scalar"
        return ft.ExpansionTile(
            data=f"tensor-card:{tensor_id}",
            title=ft.Text(
                short_name,
                size=12,
                weight=ft.FontWeight.W_700,
                color=_INK,
                max_lines=1,
                overflow=ft.TextOverflow.ELLIPSIS,
                tooltip=tensor_id,
            ),
            subtitle=ft.Text(
                f"{tensor.element_type} · {shape}",
                size=10,
                color=_MUTED,
            ),
            leading=ft.Icon(ft.Icons.DATA_ARRAY_ROUNDED, color=_ACCENT),
            controls=[ft.Column(controls=tensor_controls, spacing=12, tight=True)],
            controls_padding=ft.Padding.only(left=8, right=8, top=4, bottom=12),
            tile_padding=ft.Padding.symmetric(horizontal=8, vertical=2),
            expanded=expanded,
            on_change=self._tensor_expansion_handler(tensor_id),
            maintain_state=True,
            dense=True,
            shape=ft.RoundedRectangleBorder(radius=12),
            collapsed_shape=ft.RoundedRectangleBorder(radius=12),
            bgcolor=_SUBTLE,
            collapsed_bgcolor=_SUBTLE,
        )

    def _tensor_expansion_handler(
        self, tensor_id: str
    ) -> Callable[[ft.Event[ft.ExpansionTile]], None]:
        def remember(event: ft.Event[ft.ExpansionTile]) -> None:
            data = getattr(event, "data", None)
            self._tensor_card_expanded[tensor_id] = (
                data if isinstance(data, bool) else str(data).lower() == "true"
            )

        return remember

    def _tensor_statistics_controls(
        self, tensor_id: str, node_id: str
    ) -> list[ft.Control]:
        assert self.session is not None
        edited = tensor_id in self.session.editing.edited_tensor_ids()
        stats = None if edited else self.session.statistics(tensor_id)
        if stats is None:
            if edited:
                return [
                    ft.Text(
                        "Statistics were cleared because this tensor has byte "
                        "revisions.",
                        size=10,
                        color=_MUTED,
                        italic=True,
                    )
                ]
            if self.session.store.readable(tensor_id):
                return [
                    ft.TextButton(
                        content="Compute statistics",
                        icon=ft.Icons.QUERY_STATS_ROUNDED,
                        data=f"tensor-compute-statistics:{tensor_id}",
                        on_click=self._stats_handler(tensor_id, node_id),
                    )
                ]
            return []

        controls: list[ft.Control] = [
            overview.metadata_section(
                viewmodel.statistics_lines(stats),
                title="Statistics",
                icon=ft.Icons.QUERY_STATS_ROUNDED,
                role=f"tensor-statistics:{tensor_id}",
            )
        ]
        for label, count, fraction in viewmodel.histogram_rows(stats):
            bar = "█" * max(1, round(fraction * 20)) if count else ""
            controls.append(
                ft.Text(
                    f"{label:>24} {bar} {count}",
                    size=10,
                    font_family="monospace",
                    color=_INK,
                )
            )
        return [
            ft.Column(
                controls=controls,
                spacing=5,
                tight=True,
                data=f"tensor-statistics-block:{tensor_id}",
            )
        ]

    def _tensor_visualization(self, tensor_id: str) -> ft.Control:
        """A bounded value heatmap; at most 64 elements are ever read."""
        assert self.session is not None
        tensor = self.session.document.tensors[tensor_id]
        from nneditor.analysis.statistics import element_width

        width = element_width(tensor.element_type)
        if width is None:
            return ft.Container(
                data=f"tensor-heatmap:{tensor_id}",
                content=ft.Text(
                    f"{tensor.element_type} values cannot be decoded; "
                    "the raw bytes remain available in the hex editor.",
                    size=10,
                    color=_MUTED,
                ),
                padding=10,
                bgcolor=_SUBTLE,
                border=ft.Border.all(1, _BORDER),
                border_radius=10,
            )
        total = self.session.editing.byte_length(tensor_id) or 0
        try:
            raw = self.session.editing.read(
                tensor_id,
                offset=0,
                length=min(total, tensor_tools.HEATMAP_ELEMENTS * width),
            )
        except Exception as error:
            return ft.Text(
                f"Value visualization unavailable: {error}",
                size=10,
                color=_DANGER,
            )
        values = tensor_tools.sample_values(tensor.element_type, raw)
        if not values:
            return ft.Text("This tensor has no values.", size=10, color=_MUTED)

        maximum_absolute = max(
            (abs(value) for value in values if math.isfinite(value)),
            default=0.0,
        )
        cell = 13.0
        gap = 2.0
        shapes: list[cv.Shape] = []
        for index, value in enumerate(values):
            column = index % tensor_tools.HEATMAP_COLUMNS
            row = index // tensor_tools.HEATMAP_COLUMNS
            shapes.append(
                cv.Rect(
                    x=column * (cell + gap),
                    y=row * (cell + gap),
                    width=cell,
                    height=cell,
                    border_radius=3,
                    paint=ft.Paint(
                        color=tensor_tools.heat_color(value, maximum_absolute),
                        style=ft.PaintingStyle.FILL,
                    ),
                )
            )
        row_count = math.ceil(len(values) / tensor_tools.HEATMAP_COLUMNS)
        canvas_width = tensor_tools.HEATMAP_COLUMNS * (cell + gap) - gap
        canvas_height = row_count * (cell + gap) - gap
        finite = tuple(value for value in values if math.isfinite(value))
        value_range = (
            f"{min(finite):.5g} to {max(finite):.5g}"
            if finite
            else "non-finite values only"
        )
        return ft.Column(
            data=f"tensor-heatmap:{tensor_id}",
            controls=[
                overview.section_heading(
                    "Value map",
                    ft.Icons.GRID_ON_ROUNDED,
                    trailing=f"FIRST {len(values)}",
                ),
                ft.Container(
                    content=cv.Canvas(
                        shapes=shapes,
                        width=canvas_width,
                        height=canvas_height,
                    ),
                    alignment=ft.Alignment.CENTER,
                    padding=10,
                    bgcolor=_PANEL,
                    border=ft.Border.all(1, _BORDER),
                    border_radius=10,
                ),
                ft.Row(
                    controls=[
                        ft.Container(width=10, height=10, bgcolor="#2E90FA"),
                        ft.Text("negative", size=9, color=_MUTED),
                        ft.Container(width=10, height=10, bgcolor="#F2F4F7"),
                        ft.Text("zero", size=9, color=_MUTED),
                        ft.Container(width=10, height=10, bgcolor="#7F56D9"),
                        ft.Text("positive", size=9, color=_MUTED),
                    ],
                    spacing=4,
                    wrap=True,
                ),
                ft.Text(
                    f"Sample range: {value_range} · "
                    "Preview: "
                    f"{viewmodel.preview_values_text(tensor.element_type, raw)}",
                    size=10,
                    color=_MUTED,
                    selectable=True,
                ),
            ],
            spacing=7,
            tight=True,
        )

    def _hex_editor(self, tensor_id: str, node_id: str) -> ft.Control:
        """Build a 128-byte page editor backed by reversible tensor revisions."""
        assert self.session is not None
        total = self.session.editing.byte_length(tensor_id) or 0
        last_offset = (
            max(total - 1, 0) // tensor_tools.HEX_PAGE_BYTES
        ) * tensor_tools.HEX_PAGE_BYTES
        offset = min(max(self._hex_offsets.get(tensor_id, 0), 0), last_offset)
        self._hex_offsets[tensor_id] = offset
        length = min(tensor_tools.HEX_PAGE_BYTES, max(total - offset, 0))
        try:
            raw = self.session.editing.read(tensor_id, offset=offset, length=length)
        except Exception as error:
            return ft.Container(
                data=f"tensor-hex-editor:{tensor_id}",
                content=ft.Text(
                    f"Raw bytes unavailable: {error}", size=10, color=_DANGER
                ),
                padding=10,
                bgcolor=_DANGER_SOFT,
                border_radius=10,
            )

        draft_key = (tensor_id, offset)
        draft = self._hex_drafts.get(draft_key, tensor_tools.format_hex_bytes(raw))
        validation_error = self._hex_errors.get(tensor_id)
        page_end = offset + len(raw)
        return ft.Column(
            data=f"tensor-hex-editor:{tensor_id}",
            controls=[
                overview.section_heading(
                    "Hex editor",
                    ft.Icons.CODE_ROUNDED,
                    trailing=f"{len(raw)} BYTE PAGE",
                ),
                ft.Text(
                    "Edits create a reversible revision; the source model is "
                    "never overwritten.",
                    size=10,
                    color=_MUTED,
                ),
                ft.Row(
                    controls=[
                        ft.IconButton(
                            icon=ft.Icons.CHEVRON_LEFT_ROUNDED,
                            tooltip="Previous 128 bytes",
                            disabled=offset == 0,
                            data=f"tensor-hex-previous:{tensor_id}",
                            on_click=self._hex_page_handler(
                                tensor_id,
                                node_id,
                                -tensor_tools.HEX_PAGE_BYTES,
                            ),
                        ),
                        self._watch_text_focus(
                            ft.TextField(
                                value=f"0x{offset:X}",
                                label="Offset",
                                dense=True,
                                expand=True,
                                data=f"tensor-hex-offset:{tensor_id}",
                                on_submit=self._hex_offset_handler(tensor_id, node_id),
                            )
                        ),
                        ft.IconButton(
                            icon=ft.Icons.CHEVRON_RIGHT_ROUNDED,
                            tooltip="Next 128 bytes",
                            disabled=page_end >= total,
                            data=f"tensor-hex-next:{tensor_id}",
                            on_click=self._hex_page_handler(
                                tensor_id,
                                node_id,
                                tensor_tools.HEX_PAGE_BYTES,
                            ),
                        ),
                    ],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                ft.Text(
                    f"Bytes {offset:,}-{max(offset, page_end - 1):,} of {total:,}",
                    size=9,
                    color=_MUTED,
                ),
                self._watch_text_focus(
                    ft.TextField(
                        value=draft,
                        multiline=True,
                        min_lines=2,
                        max_lines=8,
                        dense=True,
                        text_style=ft.TextStyle(
                            font_family="monospace",
                            size=10,
                            color=_INK,
                        ),
                        label="Hex bytes",
                        helper=(
                            "Whitespace-separated byte pairs, for example: 00 FF 2A"
                        ),
                        error=validation_error,
                        autocorrect=False,
                        enable_suggestions=False,
                        smart_dashes_type=False,
                        smart_quotes_type=False,
                        data=f"tensor-hex-input:{tensor_id}",
                        on_change=self._hex_draft_handler(tensor_id, offset),
                    )
                ),
                ft.Container(
                    content=ft.Text(
                        tensor_tools.ascii_preview(raw) or "(empty)",
                        size=10,
                        font_family="monospace",
                        color=_MUTED,
                        selectable=True,
                    ),
                    padding=8,
                    bgcolor=_SUBTLE,
                    border=ft.Border.all(1, _BORDER),
                    border_radius=8,
                    tooltip="ASCII preview of the current persisted byte page",
                ),
                ft.Row(
                    controls=[
                        ft.FilledButton(
                            content="Apply bytes",
                            icon=ft.Icons.SAVE_ROUNDED,
                            disabled=not raw,
                            data=f"tensor-hex-apply:{tensor_id}",
                            on_click=self._hex_apply_handler(
                                tensor_id, node_id, offset, len(raw)
                            ),
                        ),
                        ft.TextButton(
                            content="Revert input",
                            data=f"tensor-hex-revert:{tensor_id}",
                            on_click=self._hex_revert_handler(
                                tensor_id, node_id, offset
                            ),
                        ),
                    ],
                    wrap=True,
                    spacing=5,
                ),
            ],
            spacing=7,
            tight=True,
        )

    def _hex_draft_handler(
        self, tensor_id: str, offset: int
    ) -> Callable[[ft.Event[ft.TextField]], None]:
        def remember(event: ft.Event[ft.TextField]) -> None:
            self._hex_drafts[(tensor_id, offset)] = event.control.value
            self._hex_errors.pop(tensor_id, None)

        return remember

    def _hex_page_handler(
        self, tensor_id: str, node_id: str, delta: int
    ) -> Callable[[ft.Event[ft.IconButton]], None]:
        def navigate(event: ft.Event[ft.IconButton]) -> None:
            assert self.session is not None
            total = self.session.editing.byte_length(tensor_id) or 0
            maximum = (
                max(total - 1, 0) // tensor_tools.HEX_PAGE_BYTES
            ) * tensor_tools.HEX_PAGE_BYTES
            current = self._hex_offsets.get(tensor_id, 0)
            self._hex_offsets[tensor_id] = min(max(current + delta, 0), maximum)
            self._hex_errors.pop(tensor_id, None)
            self._refresh_inspector(frozenset({node_id}))
            self.page.update()

        return navigate

    def _hex_offset_handler(
        self, tensor_id: str, node_id: str
    ) -> Callable[[ft.Event[ft.TextField]], None]:
        def jump(event: ft.Event[ft.TextField]) -> None:
            assert self.session is not None
            total = self.session.editing.byte_length(tensor_id) or 0
            try:
                offset = tensor_tools.parse_offset(
                    event.control.value, total_bytes=total
                )
            except ValueError as error:
                self._hex_errors[tensor_id] = str(error)
            else:
                self._hex_offsets[tensor_id] = offset
                self._hex_errors.pop(tensor_id, None)
            self._refresh_inspector(frozenset({node_id}))
            self.page.update()

        return jump

    def _hex_apply_handler(
        self, tensor_id: str, node_id: str, offset: int, page_length: int
    ) -> Callable[[ft.Event[ft.Button]], None]:
        def apply(event: ft.Event[ft.Button]) -> None:
            assert self.session is not None
            key = (tensor_id, offset)
            try:
                current = self.session.editing.read(
                    tensor_id, offset=offset, length=page_length
                )
                draft = self._hex_drafts.get(
                    key, tensor_tools.format_hex_bytes(current)
                )
                replacement = tensor_tools.parse_hex_bytes(
                    draft, max_bytes=tensor_tools.HEX_PAGE_BYTES
                )
                if len(replacement) > page_length:
                    raise ValueError(
                        f"this page has only {page_length} editable byte(s)"
                    )
                self.session.editing.replace_bytes(tensor_id, offset, replacement)
            except Exception as error:
                self._hex_errors[tensor_id] = str(error)
                self._refresh_inspector(frozenset({node_id}))
                self._set_status(f"Hex edit rejected: {error}")
                self.page.update()
                return
            self._hex_drafts.pop(key, None)
            self._hex_errors.pop(tensor_id, None)
            self._refresh_inspector(frozenset({node_id}))
            self._show_committed_diff()
            self._refresh_edit_actions()
            self._set_status(
                f"Applied {len(replacement)} byte(s) to {tensor_id} at 0x{offset:X}"
            )
            self.page.update()

        return apply

    def _hex_revert_handler(
        self, tensor_id: str, node_id: str, offset: int
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def revert(event: ft.Event[ft.TextButton]) -> None:
            self._hex_drafts.pop((tensor_id, offset), None)
            self._hex_errors.pop(tensor_id, None)
            self._refresh_inspector(frozenset({node_id}))
            self._set_status("Hex input restored to the current tensor revision")
            self.page.update()

        return revert

    def _preview_row(self, tensor_id: str) -> ft.Control:
        """The first few decoded values, or the reason they are unavailable."""
        assert self.session is not None
        session = self.session
        tensor = session.document.tensors[tensor_id]
        if (
            tensor.storage is not Storage.GENERATED
            and session.store.materialization(tensor_id).value == "full-parse"
            and tensor_id not in self._typed_preview_consent
        ):
            return ft.TextButton(
                content="Load preview (materializes the full typed tensor)",
                on_click=self._typed_preview_handler(tensor_id),
            )
        try:
            raw = session.editing.read(
                tensor_id, offset=0, length=self._preview_bytes(tensor_id)
            )
        except Exception as error:
            return ft.Text(f"Preview unavailable: {error}", size=11, italic=True)
        return ft.Text(
            f"Preview: {viewmodel.preview_values_text(tensor.element_type, raw)}",
            size=11,
            selectable=True,
        )

    def _typed_preview_handler(
        self, tensor_id: str
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def load(event: ft.Event[ft.TextButton]) -> None:
            self._typed_preview_consent.add(tensor_id)
            self._refresh_inspector(self.renderer.selection)
            self.page.update()

        return load

    def _preview_bytes(self, tensor_id: str) -> int:
        assert self.session is not None
        from nneditor.analysis.statistics import element_width

        tensor = self.session.document.tensors[tensor_id]
        width = element_width(tensor.element_type) or 1
        total = self.session.editing.byte_length(tensor_id) or 0
        return min(total, 9 * width)

    def _stats_handler(
        self, tensor_id: str, node_id: str
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def compute(event: ft.Event[ft.TextButton]) -> None:
            assert self.session is not None
            session = self.session
            self._set_status(f"Computing statistics for {tensor_id}…")
            job = session.compute_statistics_async(tensor_id)

            def wait() -> None:
                # run_thread futures are never retrieved, so anything that
                # raises here must be surfaced in the status bar itself.
                try:
                    job.wait()
                    if job.state.value == "succeeded":
                        self._set_status(f"Statistics ready for {tensor_id}")
                    elif job.state.value == "cancelled":
                        self._set_status("Statistics cancelled")
                    else:
                        self._set_status(f"Statistics failed: {job.error}")
                    # Only re-render when the model was not reopened and the
                    # inspector still shows the node whose card started the
                    # job; otherwise the completion would clobber whatever
                    # the user is looking at now.
                    if self.session is session and self._inspected_ids == frozenset(
                        {node_id}
                    ):
                        self._refresh_inspector(frozenset({node_id}))
                except Exception as error:
                    self._set_status(f"Statistics failed: {error}")
                finally:
                    self.page.update()

            self.page.run_thread(wait)
            self.page.update()

        return compute

    def _refresh_hierarchy(self) -> None:
        if self.session is None or self.current_graph is None:
            return
        hierarchy = self.session.graph_hierarchy(self.current_graph)
        rows: list[ft.Control] = []

        def add(group_id: str, depth: int) -> None:
            if len(rows) >= _MAX_EXPLORER_ROWS:
                return
            group = hierarchy.group(group_id)
            rows.append(
                ft.TextButton(
                    content=("  " * depth)
                    + f"{group.label}  ·  {len(group.members)} ops",
                    icon=ft.Icons.GRID_VIEW_ROUNDED,
                    tooltip=group.explanation,
                    style=ft.ButtonStyle(
                        color=(
                            _ACCENT if group.id == self.current_root_group else _INK
                        ),
                        bgcolor=(
                            _ACCENT_SOFT
                            if group.id == self.current_root_group
                            else "#00FFFFFF"
                        ),
                        shape=ft.RoundedRectangleBorder(radius=9),
                        alignment=ft.Alignment.CENTER_LEFT,
                    ),
                    on_click=self._group_handler(group.id),
                )
            )
            for child in hierarchy.children(group.id):
                add(child.id, depth + 1)

        for root in hierarchy.roots:
            add(root.id, 0)
            if len(rows) >= _MAX_EXPLORER_ROWS:
                break
        if not rows:
            rows.append(ft.Text("No groups detected", size=11))
        elif len(hierarchy.groups) > _MAX_EXPLORER_ROWS:
            rows.append(
                ft.Text(
                    f"Showing {_MAX_EXPLORER_ROWS} of "
                    f"{len(hierarchy.groups):,} blocks. Use search or the graph "
                    "to inspect the rest.",
                    size=10,
                    color=_MUTED,
                )
            )
        self.hierarchy_list.controls = rows

    def _group_handler(
        self, group_id: str
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def navigate(event: ft.Event[ft.TextButton]) -> None:
            self.show_group(group_id)
            self._refresh_inspector(frozenset({group_id}))
            self.page.update()

        return navigate

    def _refresh_breadcrumbs(self) -> None:
        if self.session is None or self.current_graph is None:
            return
        crumbs = self.session.breadcrumbs(self.current_graph, self.current_root_group)
        controls: list[ft.Control] = []
        for position, crumb in enumerate(crumbs):
            if position:
                controls.append(
                    ft.Icon(
                        ft.Icons.CHEVRON_RIGHT_ROUNDED,
                        size=15,
                        color=_MUTED,
                    )
                )
            if crumb.kind == "graph":
                controls.append(
                    ft.TextButton(
                        content=crumb.label,
                        icon=ft.Icons.HOME_ROUNDED,
                        style=ft.ButtonStyle(color=_INK),
                        on_click=self._graph_handler(crumb.id),
                    )
                )
            else:
                controls.append(
                    ft.TextButton(
                        content=crumb.label,
                        style=ft.ButtonStyle(color=_INK),
                        on_click=self._group_handler(crumb.id),
                    )
                )
        self.breadcrumbs_row.controls = controls
        self.back_to_parent_button.visible = self.current_root_group is not None

    def _rebuild_minimap(self) -> None:
        """Rebuild the minimap dots; runs only when the scene itself changes.

        Rebuilding per viewport change was the single largest interaction
        cost: one fresh control per scene node, re-diffed on every pan event.
        The dots depend only on the scene, so they persist and the viewport
        rectangle alone mutates as the user pans and zooms.
        """
        if self.session is None or self.renderer.scene is None:
            return
        model = self.session.minimap(
            self.renderer.scene,
            width=220.0,
            height=110.0,
            viewport=self.renderer.viewport,
        )
        self.minimap_model = model
        fill = ft.Paint(color="#98A2B3", style=ft.PaintingStyle.FILL)
        # At 220x110 the dots are sub-pixel long before this cap; sampling
        # keeps the density picture while bounding the control count.
        stride = max(1, math.ceil(len(model.nodes) / _MINIMAP_MAX_DOTS))
        shapes: list[cv.Shape] = [
            cv.Rect(
                x=node.x,
                y=node.y,
                width=node.width,
                height=node.height,
                paint=fill,
            )
            for node in model.nodes[::stride]
        ]
        # Canvas shapes have no visibility flag; a zero-size rectangle parked
        # off-canvas draws nothing until the first viewport projection.
        self._minimap_view_rect = cv.Rect(
            x=-10.0,
            y=-10.0,
            width=0.0,
            height=0.0,
            paint=ft.Paint(
                color=_ACCENT,
                style=ft.PaintingStyle.STROKE,
                stroke_width=1.5,
            ),
        )
        shapes.append(self._minimap_view_rect)
        self.minimap_canvas.shapes = shapes
        self._sync_minimap_view_rect()

    def _update_minimap_viewport(self) -> None:
        """Move the persistent viewport rectangle and push only the minimap."""
        if self._sync_minimap_view_rect():
            self._update_control(self.minimap_canvas)

    def _sync_minimap_view_rect(self) -> bool:
        rect = self._minimap_view_rect
        viewport = self.renderer.viewport
        if rect is None or self.minimap_model is None or viewport is None:
            return False
        bounds = self.minimap_model.project_viewport(viewport)
        rect.x = bounds.min_x
        rect.y = bounds.min_y
        rect.width = bounds.width
        rect.height = bounds.height
        return True

    def on_keyboard(self, event: ft.KeyboardEvent) -> None:
        """Arrow-key navigation over the visible semantic representation."""
        if self._text_input_active:
            # A focused TextField owns the caret; navigating now would tear
            # the field down mid-keystroke and discard the pending edit.
            return
        if (
            self.session is None
            or self.renderer.scene is None
            or len(self.renderer.selection) != 1
        ):
            return
        directions = {
            "Arrow Left": Direction.LEFT,
            "Arrow Right": Direction.RIGHT,
            "Arrow Up": Direction.UP,
            "Arrow Down": Direction.DOWN,
        }
        direction = directions.get(event.key)
        if direction is None:
            return
        (current,) = self.renderer.selection
        if self.renderer.scene.has_edge(current):
            return
        target = self.session.directional(self.renderer.scene, current, direction)
        if target is not None:
            selected = frozenset((target,))
            self.renderer.set_selection(selected)
            self._on_selected(selected)

    # -- helpers -----------------------------------------------------------

    def _watch_text_focus(self, field: ft.TextField) -> ft.TextField:
        """Track a text field's focus so keyboard navigation yields to it."""
        field.on_focus = self._on_text_focus
        field.on_blur = self._on_text_blur
        return field

    def _on_text_focus(self, event: ft.Event[ft.TextField]) -> None:
        self._text_input_active = True

    def _on_text_blur(self, event: ft.Event[ft.TextField]) -> None:
        self._text_input_active = False

    def _set_status(self, text: str) -> None:
        self.status_text.value = text

    def _clear_error(self) -> None:
        self.error_banner.visible = False

    def _show_error(self, text: str) -> None:
        banner = self.error_banner.content
        assert isinstance(banner, ft.Text)
        banner.value = text
        self.error_banner.visible = True
        self._set_status("Error")


def main(page: ft.Page, *, launch_path: Path | None = None) -> None:
    """Flet entry point."""
    page.title = APP_TITLE
    page.window.icon = str(APP_WINDOW_ICON_PATH)
    page.padding = 0
    page.bgcolor = _CANVAS
    page.theme_mode = ft.ThemeMode.LIGHT
    page.theme = ft.Theme(
        color_scheme_seed=_ACCENT,
        use_material3=True,
    )
    service = ApplicationService(job_listener=None, state_store=SessionStateStore())

    shell = Shell(page, service)

    def on_resize(event: ft.PageResizeEvent) -> None:
        if page.width:
            shell.apply_layout_for_width(float(page.width))
            page.update()

    def on_disconnect(event: ft.Event[ft.Page]) -> None:
        service.close()

    page.on_resize = on_resize
    page.on_keyboard_event = shell.on_keyboard
    page.on_disconnect = on_disconnect
    page.add(shell.build())
    if page.width:
        shell.apply_layout_for_width(float(page.width))
    if launch_path is not None:
        shell.open_path(launch_path)
    page.update()


def run(launch_path: Path | None = None) -> None:
    """Launch NNEditor using Flet's current runtime."""
    target = (
        functools.partial(main, launch_path=launch_path)
        if launch_path is not None
        else main
    )
    ft.run(
        target,
        assets_dir=str(APP_ASSETS_DIRECTORY),
        upload_dir=str(_WEB_UPLOAD_DIRECTORY),
    )


def cli(argv: Sequence[str] | None = None) -> None:
    """Parse desktop launch and file-association commands."""
    parser = argparse.ArgumentParser(
        prog="nneditor",
        description="Open and inspect neural-network model artifacts.",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--register-file-types",
        action="store_true",
        help="register NNEditor in Windows Open with and Default apps",
    )
    actions.add_argument(
        "--unregister-file-types",
        action="store_true",
        help="remove NNEditor's per-user Windows file-type registration",
    )
    actions.add_argument(
        "--choose-default-app",
        action="store_true",
        help="open Windows Settings so the user can choose NNEditor as a default",
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        help="model artifact to open (.onnx, .pt, .pth, .bin, and others)",
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.register_file_types:
            registration = register_file_associations(
                icon_path=APP_WINDOW_ICON_PATH,
            )
            print(
                f"Registered NNEditor for {len(registration.extensions)} "
                "model file types."
            )
            return
        if arguments.unregister_file_types:
            unregister_file_associations()
            print("Removed NNEditor's per-user file-type registration.")
            return
        if arguments.choose_default_app:
            open_default_apps_settings()
            return
    except FileAssociationError as error:
        parser.error(str(error))
    run(arguments.model)
