"""The Flet application shell.

The workspace is deliberately architecture-first: the left panel shows
information only — model metadata, the current selection, and captured
activations — in collapsible sections, the graph owns the middle of the
screen, and model navigation lives in the right explorer. Operations (trace,
edit, optimize) live in the workspace toolbar and open one at a time in a
drawer docked over the graph surface, so the normal viewing path stays calm.

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
import re
import sys
import tempfile
import uuid
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import flet as ft
import flet.canvas as cv

from nneditor.analysis.hierarchy import Group, Hierarchy
from nneditor.analysis.lod import DetailLevel, detail_for_scale
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
from nneditor.ir.capabilities import ArtifactKind
from nneditor.ir.core import AttrKind
from nneditor.rendering import create_flet_renderer
from nneditor.rendering.contract import InteractiveGraphRenderer, RendererFactory
from nneditor.rendering.scene import NodeGlyph, Scene, Viewport
from nneditor.tracing.contracts import TraceDevice
from nneditor.tracing.runner import recommended_trace_limits
from nneditor.transformations.engine import (
    TransformationProposal,
    TransformationRequest,
)
from nneditor.transformations.schema import Granularity
from nneditor.ui import input_workspace, overview, shell_layout, viewmodel
from nneditor.ui.activation_inspector import ActivationInspector
from nneditor.ui.trace_graph import build_trace_graph
from nneditor.ui.trace_panel import TracePanel, uses_automatic_mask
from nneditor.ui.watch_panel import WatchPanel

APP_TITLE = "NNEditor"
APP_ASSETS_DIRECTORY = Path(__file__).resolve().parents[1] / "assets"
APP_ICON_PATH = APP_ASSETS_DIRECTORY / "nneditor.png"
APP_WINDOW_ICON_PATH = APP_ASSETS_DIRECTORY / "nneditor.ico"
_WEB_UPLOAD_TEMP = tempfile.TemporaryDirectory(prefix="nneditor-web-upload-")
_WEB_UPLOAD_DIRECTORY = Path(_WEB_UPLOAD_TEMP.name)

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
_SIDEBAR_WIDTH = 304
_MAX_EXPLORER_ROWS = 200
# Hierarchies nested deeper than this are almost certainly corrupt sidecar
# data; the Blocks walk stops there instead of recursing without bound.
_MAX_HIERARCHY_DEPTH = 32
_EXPLORER_INDENT = 14.0
"""Pixels of indentation per hierarchy depth in the Blocks pane."""
SHELL_PALETTE = shell_layout.ShellPalette(
    panel=_PANEL,
    border=_BORDER,
    accent=_ACCENT,
    accent_soft=_ACCENT_SOFT,
    ink=_INK,
    muted=_MUTED,
    canvas=_CANVAS,
    sidebar_width=_SIDEBAR_WIDTH,
)

_DETAIL_SEQUENCE: Final = (
    DetailLevel.ARCHITECTURE,
    DetailLevel.BLOCK,
    DetailLevel.LAYER,
    DetailLevel.OPERATOR,
)

_OVERLAY_MODES: Final = ("off", "nonfinite", "magnitude")
"""The anomaly-overlay cycle: Off -> Non-finite -> Magnitude -> Off."""

_OVERLAY_LABELS: Final = {
    "off": "Overlay: Off",
    "nonfinite": "Overlay: Non-finite",
    "magnitude": "Overlay: Magnitude",
}

_OPERATIONS: Final[dict[str, tuple[str, str, ft.IconData, str]]] = {
    "trace": (
        "Trace",
        "Trace activations",
        ft.Icons.PLAY_CIRCLE_OUTLINE_ROUNDED,
        "Run an approved inference trace and capture activations",
    ),
    "edit": (
        "Edit",
        "Edit model",
        ft.Icons.EDIT_ROUNDED,
        "Validate and commit transactional graph edits",
    ),
    "optimize": (
        "Optimize",
        "Optimize weights",
        ft.Icons.TUNE_ROUNDED,
        "Preview and apply quantization and pruning transformations",
    ),
}
"""The workspace operations: key -> (button label, drawer title, icon,
tooltip).  Each toggles the operations drawer over the graph surface, which
hosts that operation's retained controls Column."""


@runtime_checkable
class _SupportsGlyphTint(Protocol):
    """Renderers accepting a per-glyph fill override (the anomaly overlay).

    Structural on purpose: the seam stays out of the mandatory renderer
    contract, so a renderer without it simply shows no overlay instead of
    failing to plug in.
    """

    def set_tint(self, tint: Callable[[str], str | None] | None) -> None: ...


_DRILL_COVERAGE: Final = 0.45
"""Surface share a hovered group glyph must reach before zooming in advances
the representation regardless of absolute scale.

``detail_for_scale`` thresholds assume layouts whose glyphs are readable near
scale 1.0. A huge model's architecture fit sits at the minimum scale, an
order of magnitude below the first transition threshold, so a region glyph
can fill the screen while the absolute thresholds are still ~19 wheel steps
away. What the user means by "zoom into that block" is glyph screen size, so
the drill-through criterion is relative to it."""


def next_detail_level(detail: DetailLevel) -> DetailLevel | None:
    """The next-deeper semantic representation, or None at operator detail."""
    position = _DETAIL_SEQUENCE.index(detail)
    if position + 1 == len(_DETAIL_SEQUENCE):
        return None
    return _DETAIL_SEQUENCE[position + 1]


def glyph_screen_coverage(
    glyph_width: float,
    glyph_height: float,
    scale: float,
    surface_width: float,
    surface_height: float,
) -> float:
    """The dominant share of the surface a glyph occupies at a zoom level."""
    if surface_width <= 0.0 or surface_height <= 0.0:
        return 0.0
    return max(
        glyph_width * scale / surface_width,
        glyph_height * scale / surface_height,
    )


def nearest_glyph_center(
    nodes: Sequence[NodeGlyph],
    x: float,
    y: float,
) -> tuple[float, float]:
    """The center of the glyph closest to a world position."""
    nearest = min(
        nodes,
        key=lambda node: (
            (node.x + node.width / 2.0 - x) ** 2 + (node.y + node.height / 2.0 - y) ** 2
        ),
    )
    return (nearest.x + nearest.width / 2.0, nearest.y + nearest.height / 2.0)


def _natural_key(text: str) -> tuple[tuple[int, int, str], ...]:
    """A case-insensitive, numeric-aware ordering key: "2" before "10".

    Each chunk is a uniformly typed triple so tuples always compare cleanly:
    digit runs order numerically before text at the same position.
    """
    return tuple(
        (0, int(chunk), "") if chunk.isdigit() else (1, 0, chunk)
        for chunk in re.split(r"(\d+)", text.casefold())
        if chunk
    )


def resolve_initial_palette(brightness: object) -> shell_layout.ShellPalette:
    """Follow the platform brightness when Flet exposes it; default to light."""
    value = getattr(brightness, "value", brightness)
    if str(value).lower() == "dark":
        return shell_layout.DARK_SHELL_PALETTE
    return SHELL_PALETTE


class Shell:
    """One page's worth of UI state and wiring."""

    # Chrome containers composed by _compose_chrome and build; declared here
    # because both consult the previous instance (visibility, tab index)
    # before reassigning it on a theme rebuild.
    left_panel: ft.Container
    right_panel: ft.Container
    empty_state: ft.Container
    loading_overlay: ft.Container
    surface: ft.Container
    workspace_tabs: ft.Tabs
    operations_drawer: ft.Container
    info_model_tile: ft.ExpansionTile
    info_selection_tile: ft.ExpansionTile
    info_activations_tile: ft.ExpansionTile

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
        # Blocks-pane expansion state (group id -> expanded), per session.
        # Unlisted groups default to expanded roots and collapsed subtrees.
        self._hierarchy_expanded: dict[str, bool] = {}
        # Group ids the Blocks pane currently highlights for the renderer
        # selection; compared before rebuilding so reselecting the same
        # glyph never pays for a full tree rebuild.
        self._hierarchy_highlight: frozenset[str] = frozenset()
        self.current_detail = DetailLevel.ARCHITECTURE
        self.auto_detail = False
        self.current_slice: GraphSlice | None = None
        self.logical_selection: frozenset[str] = frozenset()
        self.minimap_model: MiniMap | None = None
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
        # The revision represented by the last successful export in this
        # shell. None is the immutable source artifact; recovered sidecar
        # revisions therefore reopen as unsaved and offer an explicit save.
        self._saved_revision_id: str | None = None
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
        # One blank-canvas recovery may run per applied viewport; the flag
        # keeps a recovery's own rebuilds from recursing into more recovery.
        self._blank_recovery_active = False
        # The minimap's persistent viewport rectangle; dots rebuild only when
        # the scene changes, this rectangle mutates on every pan/zoom.
        self._minimap_view_rect: cv.Rect | None = None

        self._init_theme_and_services(page)

        self.title_text = ft.Text(
            APP_TITLE,
            size=17,
            color=self.palette.ink,
            weight=ft.FontWeight.W_700,
        )
        self.model_subtitle = ft.Text(
            "Neural network explorer", size=11, color=self.palette.muted
        )
        self.job_text = ft.Text("", size=11, color=self.palette.muted)
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
            "Open a supported model artifact to begin.",
            size=11,
            color=self.palette.muted,
        )
        self.device_icon = ft.Icon(
            ft.Icons.MEMORY_ROUNDED,
            size=15,
            color=self.palette.muted,
        )
        self.device_text = ft.Text(
            "Device: idle",
            size=10,
            color=self.palette.muted,
            weight=ft.FontWeight.W_600,
        )
        self.device_indicator = ft.Container(
            content=ft.Row(
                controls=[self.device_icon, self.device_text],
                spacing=5,
                tight=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding.symmetric(horizontal=9, vertical=6),
            bgcolor=self.palette.canvas,
            border=ft.Border.all(1, self.palette.border),
            border_radius=14,
            tooltip="No inference trace is running.",
        )
        self.error_banner = ft.Container(
            content=ft.Text("", color="#FFFFFF"),
            bgcolor=self.palette.danger,
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
            "Model overview",
            size=17,
            weight=ft.FontWeight.W_700,
            color=self.palette.ink,
        )
        self.inspector_subtitle = ft.Text(
            "Select a block to inspect it", size=11, color=self.palette.muted
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
        self.hover_title = ft.Text(
            "", size=13, weight=ft.FontWeight.W_700, color=self.palette.ink
        )
        self.hover_summary = ft.Text("", size=11, color=self.palette.muted, max_lines=3)
        self.hover_card = ft.Container(
            content=ft.Column(
                controls=[self.hover_title, self.hover_summary],
                tight=True,
                spacing=3,
            ),
            width=260,
            padding=12,
            bgcolor=self.palette.panel,
            border=ft.Border.all(1, self.palette.border),
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
            palette=self.palette,
            assign=self._assign_generated_input,
            on_error=self._show_error,
            on_status=self._set_status,
            clear_error=self._clear_error,
            watch_text_focus=self._watch_text_focus,
        )
        self._init_watch_and_overlay(page)
        self.activations = ActivationInspector(
            page=page,
            picker=self.picker,
            palette=self.palette,
            session=lambda: self.session,
            current_graph=lambda: self.current_graph,
            current_slice=lambda: self.current_slice,
            active_trace_id=lambda: self.trace.active_trace_id,
            trace_values_for_glyph=lambda glyph_id: self.trace.values_for_glyph(
                glyph_id
            ),
            inspected_ids=lambda: self._inspected_ids,
            selection=lambda: self.renderer.selection,
            refresh_inspector=self._refresh_inspector,
            refresh_edit_actions=self._refresh_edit_actions,
            show_committed_diff=self._show_committed_diff,
            on_error=self._show_error,
            on_status=self._set_status,
            watch_text_focus=self._watch_text_focus,
            pin_value=self.watch.pin,
            on_statistics_ready=self._on_statistics_landed,
        )
        self.trace = TracePanel(
            page=page,
            picker=self.picker,
            palette=self.palette,
            renderer=self.renderer,
            session=lambda: self.session,
            current_graph=lambda: self.current_graph,
            current_slice=lambda: self.current_slice,
            surface_size=lambda: self.surface_size,
            inspected_ids=lambda: self._inspected_ids,
            selected_node_ids=lambda: self.renderer.selection,
            activation_rows=self.activations.activation_rows_for_values,
            set_heading=self._set_inspector_heading,
            refresh_inspector=self._refresh_inspector,
            autoload_activation_views=self.activations.autoload_views,
            redraw_scene=self._redraw_scene,
            rebuild_minimap=self._rebuild_minimap,
            on_error=self._show_error,
            on_status=self._set_status,
            on_device=self._set_trace_device,
            clear_error=self._clear_error,
            watch_text_focus=self._watch_text_focus,
        )

        self._edit_controls = ft.Column(
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
        self._transformation_controls = ft.Column(
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
        self.loading_title = ft.Text(
            "Preparing model",
            size=18,
            weight=ft.FontWeight.W_700,
            color=self.palette.ink,
        )
        self.loading_stage = ft.Text(
            "Indexing topology, detecting blocks, and laying out the architecture…",
            size=12,
            color=self.palette.muted,
            text_align=ft.TextAlign.CENTER,
        )
        self.cancel_open_button = ft.TextButton(
            content="Cancel",
            on_click=self._on_cancel_open,
        )
        self._compose_chrome()
        for text_field in (
            self.group_label_field,
            self.search_field,
            self.edit_primary,
            self.edit_secondary,
            self.edit_port,
            self.transformation_axis,
            self.transformation_parameter,
        ):
            self._watch_text_focus(text_field)

    # -- construction ------------------------------------------------------

    def _init_theme_and_services(self, page: ft.Page) -> None:
        """Pick the starting palette and register the page-level services."""
        self.palette = resolve_initial_palette(
            getattr(page, "platform_brightness", None)
        )
        dark = self.palette is shell_layout.DARK_SHELL_PALETTE
        page.theme_mode = ft.ThemeMode.DARK if dark else ft.ThemeMode.LIGHT
        page.bgcolor = self.palette.canvas
        self.theme_toggle = ft.IconButton(
            icon=(ft.Icons.LIGHT_MODE_ROUNDED if dark else ft.Icons.DARK_MODE_ROUNDED),
            tooltip="Switch between light and dark mode",
            on_click=self._on_toggle_theme,
        )
        self._root: ft.Control | None = None
        self.picker = ft.FilePicker(on_upload=self._on_upload_progress)
        self.clipboard = ft.Clipboard()
        # FilePicker and Clipboard are Services, not Controls, in Flet 0.86:
        # registering them in `page.overlay` makes the client reject them with
        # "Unknown control" the moment the page renders.
        page.services.append(self.picker)
        page.services.append(self.clipboard)

    def _init_watch_and_overlay(self, page: ft.Page) -> None:
        """Construct the watch strip, anomaly-overlay, operations-drawer,
        and info-section state.

        The watch panel's accessors are deliberately lazy lambdas: the
        activation inspector and trace panel they reach are constructed
        after this call, and are only dereferenced at event time.
        """
        # The user's explicit expand/collapse choices for the left panel's
        # information sections this app session; a key absent here follows
        # the state-driven default (see _info_section_expanded).
        self._info_section_state: dict[str, bool] = {}
        # Which operation's controls the drawer hosts; None means closed.
        self._active_operation: str | None = None
        # The Model section's retained content column; _refresh_inspector
        # re-renders the overview rows into it.
        self.model_info = ft.Column(spacing=14, tight=True, data="model-info-content")
        self.trace_status_line = ft.Text(
            "No activation trace yet.",
            size=10,
            color=self.palette.muted,
            data="trace-status-line",
        )
        self.drawer_title = ft.Text(
            "",
            size=13,
            weight=ft.FontWeight.W_700,
            color=self.palette.ink,
        )
        self.drawer_close_button = ft.IconButton(
            icon=ft.Icons.CLOSE_ROUNDED,
            icon_size=16,
            tooltip="Close the operations drawer (Esc)",
            data="operations-drawer-close",
            on_click=self._on_close_operations_drawer,
        )
        # The retained container the drawer wraps; its content is swapped to
        # the active operation's controls Column so those Columns stay the
        # exact objects the rest of the shell mutates.
        self.drawer_body = ft.Container(data="operations-drawer-body")
        self.operation_buttons: dict[str, ft.TextButton] = {
            operation: ft.TextButton(
                content=label,
                icon=icon,
                tooltip=tooltip,
                data=f"operation:{operation}",
                style=ft.ButtonStyle(color=self.palette.ink),
                on_click=self._operation_click_handler(operation),
            )
            for operation, (label, _title, icon, tooltip) in _OPERATIONS.items()
        }
        self.overlay_mode: str = "off"
        # Per-operator anomaly severity: node id -> (rank, tint color). A
        # semantic glyph tints by the worst rank among its members.
        self._overlay_node_severity: dict[str, tuple[int, str]] = {}
        self.overlay_button = ft.TextButton(
            content=_OVERLAY_LABELS["off"],
            icon=ft.Icons.GRADIENT_ROUNDED,
            tooltip=(
                "Tint glyphs by numeric anomalies in the active trace's "
                "computed statistics — press to cycle Off, Non-finite, "
                "Magnitude"
            ),
            style=ft.ButtonStyle(color=self.palette.ink),
            on_click=self._on_cycle_overlay,
        )
        self.overlay_legend = ft.Text(
            "",
            size=10,
            color=self.palette.muted,
            visible=False,
        )
        self.watch = WatchPanel(
            page=page,
            palette=self.palette,
            session=lambda: self.session,
            active_trace_id=lambda: self.trace.active_trace_id,
            statistics=lambda: self.activations.activation_statistics,
            statistics_loading=lambda: self.activations.activation_loading,
            on_statistics_ready=self._on_statistics_landed,
            on_status=self._set_status,
            on_error=self._show_error,
        )

    # -- anomaly overlay ---------------------------------------------------

    def _on_cycle_overlay(self, event: ft.Event[ft.TextButton] | None = None) -> None:
        """Advance the anomaly overlay: Off -> Non-finite -> Magnitude."""
        position = _OVERLAY_MODES.index(self.overlay_mode)
        self.overlay_mode = _OVERLAY_MODES[(position + 1) % len(_OVERLAY_MODES)]
        self._refresh_overlay_tint()
        if self.overlay_mode == "off":
            self._set_status("Anomaly overlay off")
        else:
            self._set_status(
                f"{_OVERLAY_LABELS[self.overlay_mode]} — only values with "
                "computed statistics participate; compute more from the "
                "inspector or the watch list"
            )
        self.page.update()

    def _on_statistics_landed(self) -> None:
        """A statistics job landed: refresh the watch strip and overlay."""
        self.watch.refresh()
        if self.overlay_mode != "off":
            self._refresh_overlay_tint()

    def _refresh_overlay_tint(self) -> None:
        """Recompute glyph tints from already-computed statistics.

        The overlay never schedules statistics jobs: it renders whatever has
        landed so far and the legend reports the resulting coverage. The
        renderer keeps the tint callable across scene rebuilds, so the
        overlay survives pan, zoom, and detail changes without re-arming.
        """
        self._overlay_node_severity = self._overlay_severities()
        if isinstance(self.renderer, _SupportsGlyphTint):
            active = self.overlay_mode != "off" and bool(self._overlay_node_severity)
            self.renderer.set_tint(self._overlay_tint if active else None)
        self._sync_overlay_controls()

    def _overlay_severities(self) -> dict[str, tuple[int, str]]:
        """Per-operator severity from the active trace's computed statistics."""
        trace_id = self.trace.active_trace_id
        if self.overlay_mode == "off" or self.session is None or trace_id is None:
            return {}
        try:
            result = self.session.trace(trace_id)
        except (KeyError, ValueError):
            return {}
        store = self.activations.activation_statistics
        magnitudes: dict[str, float] = {}
        flagged: dict[str, tuple[int, str]] = {}
        for record in result.records:
            if record.node_id is None:
                continue
            stats = store.get((trace_id, record.value_id))
            if stats is None:
                continue
            if self.overlay_mode == "nonfinite":
                if stats.nan_count + stats.inf_count > 0:
                    flagged[record.node_id] = (1, self.palette.danger)
            else:
                magnitude = max(
                    abs(stats.minimum or 0.0),
                    abs(stats.maximum or 0.0),
                )
                magnitudes[record.node_id] = max(
                    magnitudes.get(record.node_id, 0.0), magnitude
                )
        if self.overlay_mode == "nonfinite":
            return flagged
        if not magnitudes:
            return {}
        # Three tercile steps over the per-node max-|value| distribution,
        # routed through the palette's accent/warning scale.
        ordered = sorted(magnitudes.values())
        steps = (
            self.palette.accent_soft,
            self.palette.warning_border,
            self.palette.warning,
        )
        low = ordered[max(0, math.ceil(len(ordered) / 3) - 1)]
        high = ordered[max(0, math.ceil(2 * len(ordered) / 3) - 1)]
        severities: dict[str, tuple[int, str]] = {}
        for node_id, magnitude in magnitudes.items():
            step = 0 if magnitude <= low else 1 if magnitude <= high else 2
            severities[node_id] = (step, steps[step])
        return severities

    def _overlay_tint(self, glyph_id: str) -> str | None:
        """The override fill for one glyph: its worst member's severity."""
        severity = self._overlay_node_severity
        if not severity:
            return None
        members: frozenset[str] = frozenset((glyph_id,))
        if self.current_slice is not None:
            members = self.current_slice.members_by_glyph.get(glyph_id, members)
        worst: tuple[int, str] | None = None
        for member in members:
            entry = severity.get(member)
            if entry is not None and (worst is None or entry[0] > worst[0]):
                worst = entry
        return None if worst is None else worst[1]

    def _sync_overlay_controls(self) -> None:
        """Reflect the overlay mode and coverage on the toolbar controls."""
        self.overlay_button.content = _OVERLAY_LABELS[self.overlay_mode]
        self.overlay_button.style = ft.ButtonStyle(
            color=(
                self.palette.accent if self.overlay_mode != "off" else self.palette.ink
            )
        )
        if self.overlay_mode == "off":
            self.overlay_legend.value = ""
            self.overlay_legend.visible = False
            return
        scene = self.renderer.scene
        tinted = 0
        total = 0
        if scene is not None:
            total = scene.node_count
            tinted = sum(
                1 for node in scene.nodes if self._overlay_tint(node.id) is not None
            )
        self.overlay_legend.value = f"tinted {tinted:,} of {total:,}"
        self.overlay_legend.visible = True

    def _reset_overlay(self) -> None:
        """Drop the overlay with its per-trace severity state."""
        self.overlay_mode = "off"
        self._overlay_node_severity = {}
        if isinstance(self.renderer, _SupportsGlyphTint):
            self.renderer.set_tint(None)
        self._sync_overlay_controls()

    # -- operations drawer -------------------------------------------------

    def _operation_click_handler(
        self, operation: str
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        """A toolbar handler toggling the drawer onto one operation."""

        def toggle(event: ft.Event[ft.TextButton]) -> None:
            self._active_operation = (
                None if self._active_operation == operation else operation
            )
            self._sync_operations_drawer()
            self.page.update()

        return toggle

    def _on_close_operations_drawer(self, event: ft.Event[ft.IconButton]) -> None:
        if self._close_operations_drawer():
            self.page.update()

    def _close_operations_drawer(self) -> bool:
        """Close the operations drawer; report whether it was open."""
        if self._active_operation is None:
            return False
        self._active_operation = None
        self._sync_operations_drawer()
        return True

    def _operation_controls(self) -> dict[str, ft.Column]:
        """The retained controls Column each operation's drawer hosts."""
        return {
            "trace": self.trace.control,
            "edit": self._edit_controls,
            "optimize": self._transformation_controls,
        }

    def _sync_operations_drawer(self) -> None:
        """Reflect the active operation on the drawer and toolbar buttons."""
        operation = self._active_operation
        for key, button in self.operation_buttons.items():
            active = key == operation
            button.style = ft.ButtonStyle(
                color=self.palette.accent if active else self.palette.ink,
                bgcolor=self.palette.accent_soft if active else None,
            )
        if operation is None:
            self.operations_drawer.visible = False
            self.drawer_body.content = None
            return
        self.drawer_title.value = _OPERATIONS[operation][1]
        self.drawer_body.content = self._operation_controls()[operation]
        self.operations_drawer.visible = True

    # -- left-panel information sections -----------------------------------

    def _info_section_expanded(self, key: str) -> bool:
        """The stored user choice for a section, else its state default.

        Defaults: the Model section opens while nothing is selected, the
        Selection section is always open, and the Activations section opens
        once a trace exists.
        """
        stored = self._info_section_state.get(key)
        if stored is not None:
            return stored
        if key == "info:model":
            return not self._inspected_ids
        if key == "info:activations":
            return self.trace.active_trace_id is not None
        return True

    def _section_toggle_handler(
        self, key: str
    ) -> Callable[[ft.Event[ft.ExpansionTile]], None]:
        def remember(event: ft.Event[ft.ExpansionTile]) -> None:
            data = getattr(event, "data", None)
            self._info_section_state[key] = (
                data if isinstance(data, bool) else str(data).lower() == "true"
            )

        return remember

    def _sync_info_sections(self) -> None:
        """Re-derive each tile's expanded state; explicit choices win."""
        if not hasattr(self, "info_model_tile"):
            return
        for key, tile in (
            ("info:model", self.info_model_tile),
            ("info:selection", self.info_selection_tile),
            ("info:activations", self.info_activations_tile),
        ):
            tile.expanded = self._info_section_expanded(key)

    def _build_info_sections(
        self,
    ) -> tuple[ft.ExpansionTile, ft.ExpansionTile, ft.ExpansionTile]:
        """(Re)build the left panel's section tiles around retained content."""
        self.info_model_tile = shell_layout.build_info_section(
            palette=self.palette,
            key="info:model",
            title="Model",
            icon=ft.Icons.INSERT_DRIVE_FILE_ROUNDED,
            content=[self.model_info],
            expanded=self._info_section_expanded("info:model"),
            on_change=self._section_toggle_handler("info:model"),
        )
        self.info_selection_tile = shell_layout.build_info_section(
            palette=self.palette,
            key="info:selection",
            title="Selection",
            icon=ft.Icons.INFO_OUTLINE_ROUNDED,
            content=shell_layout.build_selection_section_content(
                palette=self.palette,
                inspector_title=self.inspector_title,
                inspector_subtitle=self.inspector_subtitle,
                open_selection_button=self.open_selection_button,
                back_to_parent_button=self.back_to_parent_button,
                inspector=self.inspector,
            ),
            expanded=self._info_section_expanded("info:selection"),
            on_change=self._section_toggle_handler("info:selection"),
        )
        self.info_activations_tile = shell_layout.build_info_section(
            palette=self.palette,
            key="info:activations",
            title="Activations",
            icon=ft.Icons.PUSH_PIN_ROUNDED,
            content=[self.trace_status_line, self.watch.control],
            expanded=self._info_section_expanded("info:activations"),
            on_change=self._section_toggle_handler("info:activations"),
        )
        return (
            self.info_model_tile,
            self.info_selection_tile,
            self.info_activations_tile,
        )

    def _refresh_trace_status_line(self) -> None:
        """The Activations section's read-only summary of the active trace."""
        if not hasattr(self, "trace"):
            return
        trace_id = self.trace.active_trace_id
        if self.session is None or trace_id is None:
            self.trace_status_line.value = "No activation trace yet."
            self.trace_status_line.color = self.palette.muted
            return
        try:
            result = self.session.trace(trace_id)
        except (KeyError, ValueError):
            self.trace_status_line.value = f"Trace {trace_id[:12]}"
            self.trace_status_line.color = self.palette.muted
            return
        self.trace_status_line.value = (
            f"Trace {trace_id[:12]} · "
            f"{result.execution_device.value.upper()} · {result.runtime}"
        )
        self.trace_status_line.color = self.palette.ink

    def _compose_chrome(self) -> None:
        """(Re)compose the palette-bearing containers around retained controls.

        Runs at construction and again on a theme toggle. Visibility state
        carries over so a rebuild never reopens a panel the user closed or
        drops the loading overlay mid-open.
        """
        left_visible = self.left_panel.visible if hasattr(self, "left_panel") else True
        right_visible = (
            self.right_panel.visible if hasattr(self, "right_panel") else True
        )
        empty_visible = (
            self.empty_state.visible if hasattr(self, "empty_state") else True
        )
        loading_visible = (
            self.loading_overlay.visible if hasattr(self, "loading_overlay") else False
        )
        model_section, selection_section, activations_section = (
            self._build_info_sections()
        )
        self.left_panel = shell_layout.build_left_panel(
            palette=self.palette,
            model_section=model_section,
            selection_section=selection_section,
            activations_section=activations_section,
        )
        self.left_panel.visible = left_visible
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
            palette=self.palette,
            search_field=self.search_field,
            search_results=self.search_results,
            graph_list=self.graph_list,
            hierarchy_list=self.hierarchy_list,
            hierarchy_tools=hierarchy_tools,
            minimap=self.minimap,
        )
        self.right_panel.visible = right_visible
        self.operations_drawer = shell_layout.build_operations_drawer(
            palette=self.palette,
            title=self.drawer_title,
            close_button=self.drawer_close_button,
            body=self.drawer_body,
        )
        # A theme rebuild re-hosts the open operation in the fresh drawer.
        self._sync_operations_drawer()
        self.empty_state = shell_layout.build_empty_state(
            palette=self.palette,
            on_open=self._on_open_clicked,
        )
        self.empty_state.visible = empty_visible
        self.loading_overlay = shell_layout.build_loading_overlay(
            palette=self.palette,
            title=self.loading_title,
            stage=self.loading_stage,
            cancel_button=self.cancel_open_button,
        )
        self.loading_overlay.visible = loading_visible
        self.surface = shell_layout.build_surface(
            palette=self.palette,
            renderer_control=self.renderer_control,
            graph_actions=self.trace.graph_actions,
            empty_state=self.empty_state,
            hover_card=self.hover_card,
            operations_drawer=self.operations_drawer,
            loading_overlay=self.loading_overlay,
            on_size_change=self._on_surface_size,
        )

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
                self.device_indicator,
                self.job_text,
                self.file_types_button,
                self.theme_toggle,
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
                    *self.operation_buttons.values(),
                    ft.Container(width=1, height=22, bgcolor=self.palette.border),
                    self.reset_view_button,
                    self.organize_button,
                    self.overlay_button,
                    self.overlay_legend,
                    ft.Text(
                        "View",
                        size=10,
                        color=self.palette.muted,
                        weight=ft.FontWeight.W_700,
                    ),
                    self.detail_segment,
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding.symmetric(horizontal=14, vertical=8),
            bgcolor=self.palette.panel,
            border=ft.Border.only(bottom=ft.BorderSide(1, self.palette.border)),
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
        selected_tab = (
            self.workspace_tabs.selected_index if hasattr(self, "workspace_tabs") else 0
        )
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
                        indicator_color=self.palette.accent,
                        label_color=self.palette.accent,
                        unselected_label_color=self.palette.muted,
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
            selected_index=selected_tab,
            expand=True,
        )
        root = ft.Column(
            controls=[
                ft.Container(
                    content=top_bar,
                    padding=ft.Padding.symmetric(horizontal=14, vertical=9),
                    bgcolor=self.palette.panel,
                    border=ft.Border.only(bottom=ft.BorderSide(1, self.palette.border)),
                ),
                self.error_banner,
                self.workspace_tabs,
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Container(
                                width=7,
                                height=7,
                                bgcolor=self.palette.success,
                                border_radius=4,
                            ),
                            self.status_text,
                        ],
                        spacing=7,
                    ),
                    padding=ft.Padding.symmetric(horizontal=14, vertical=7),
                    bgcolor=self.palette.panel,
                    border=ft.Border.only(top=ft.BorderSide(1, self.palette.border)),
                ),
            ],
            expand=True,
            spacing=0,
        )
        self._root = root
        return root

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

    def _on_toggle_theme(self, event: ft.Event[ft.IconButton] | None = None) -> None:
        self._set_theme(dark=self.palette is not shell_layout.DARK_SHELL_PALETTE)

    def _set_theme(self, *, dark: bool) -> None:
        """Switch the palette and rebuild the chrome around the live state.

        The canvas, panels, status bar, and inspector chrome all recolor in
        place; the trace and test-input panels keep their constructed colors
        until the next model open, which the status line discloses.
        """
        self.palette = shell_layout.DARK_SHELL_PALETTE if dark else SHELL_PALETTE
        self.page.theme_mode = ft.ThemeMode.DARK if dark else ft.ThemeMode.LIGHT
        self.page.bgcolor = self.palette.canvas
        self.theme_toggle.icon = (
            ft.Icons.LIGHT_MODE_ROUNDED if dark else ft.Icons.DARK_MODE_ROUNDED
        )
        # These panels re-render from `self.palette` on their next refresh.
        self.activations.palette = self.palette
        self.trace.palette = self.palette
        self.input_generator.palette = self.palette
        self.watch.palette = self.palette
        self._apply_palette_to_controls()
        old_root = self._root
        self._compose_chrome()
        new_root = self.build()
        controls = getattr(self.page, "controls", None)
        if controls is not None and old_root in controls:
            controls[controls.index(old_root)] = new_root
        # The watch strip bakes colors at render time even when empty, so it
        # re-renders regardless of whether a model is open.
        self.watch.refresh()
        if self.session is not None:
            # Dynamic rows bake colors at render time; re-render them all so
            # the open model recolors live rather than on the next click.
            self._refresh_graph_list()
            self._refresh_hierarchy()
            self._refresh_breadcrumbs()
            self._refresh_inspector(self._inspected_ids)
            self._rebuild_minimap()
            if self.overlay_mode != "off":
                # Severity colors were baked from the previous palette.
                self._refresh_overlay_tint()
        self._set_status(
            f"Switched to {'dark' if dark else 'light'} mode; trace and "
            "test-input panels adopt it fully when a model is next opened"
        )
        self.page.update()

    def _apply_palette_to_controls(self) -> None:
        """Recolor the retained stateful controls the chrome rebuild reuses."""
        palette = self.palette
        self.title_text.color = palette.ink
        self.model_subtitle.color = palette.muted
        self.job_text.color = palette.muted
        self.status_text.color = palette.muted
        self.inspector_title.color = palette.ink
        self.inspector_subtitle.color = palette.muted
        self.loading_title.color = palette.ink
        self.loading_stage.color = palette.muted
        self.hover_title.color = palette.ink
        self.hover_summary.color = palette.muted
        self.hover_card.bgcolor = palette.panel
        self.hover_card.border = ft.Border.all(1, palette.border)
        self.error_banner.bgcolor = palette.danger
        self.device_indicator.bgcolor = palette.canvas
        self.overlay_legend.color = palette.muted
        self.overlay_button.style = ft.ButtonStyle(
            color=palette.accent if self.overlay_mode != "off" else palette.ink
        )
        self.drawer_title.color = palette.ink
        self._refresh_trace_status_line()

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
        self.trace.bind_tensor(input_name, binding)
        self.trace.refresh_graph_actions()
        self.trace.refresh_actions()
        graph = self.session.document.main_graph
        value_id = next(
            value_id
            for value_id in graph.inputs
            if value_id not in graph.initializers
            and (graph.value(value_id).name or value_id) == input_name
        )
        if self.current_graph != self.session.document.entry_graph:
            self._show_graph(self.session.document.entry_graph)
        presentation = self.trace.presentation
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
                is_mask=uses_automatic_mask(value.name or value.id),
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

    async def _on_open_clicked(self, event: ft.Event[ft.Button] | None = None) -> None:
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
        self.trace.abandon_job()
        if self.export_job is not None and not self.export_job.state.is_terminal:
            self.export_job.cancel()
        self.export_job = None
        self.service.close_session(session.id)
        self.session = None
        self._saved_revision_id = None
        self.current_graph = None
        self.current_root_group = None
        self._hierarchy_expanded.clear()
        self._hierarchy_highlight = frozenset()
        self.current_slice = None
        self.logical_selection = frozenset()
        self.minimap_model = None
        self._minimap_view_rect = None
        self.activations.reset()
        self._inspected_ids = frozenset()
        self.pending_edit = None
        self.pending_transformation = None
        self.pending_target = None
        self.trace.reset()
        # Pins and overlay severities are per-session state.
        self.watch.reset()
        self._reset_overlay()
        self.search_field.value = ""
        self.search_results.controls = []
        self.graph_list.controls = []
        self.hierarchy_list.controls = []
        self.breadcrumbs_row.controls = []
        self.trace.graph_actions.controls = []
        self.minimap_canvas.shapes = []
        self.inspector.controls = []
        self.model_info.controls = []
        self._sync_info_sections()
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
        self.trace.refresh_actions()
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
        position = event.local_position
        self._zoom_at(position.x, position.y, 0.9 if delta_y > 0 else 1.1)

    def _zoom_at(self, screen_x: float, screen_y: float, factor: float) -> None:
        """Zoom about a screen anchor, advancing auto-detail when due."""
        old_scale = self.view["scale"]
        anchor_x = self.view["x"] + screen_x / old_scale
        anchor_y = self.view["y"] + screen_y / old_scale
        new_scale = max(0.02, min(4.0, old_scale * factor))
        self.view = {
            "scale": new_scale,
            "x": anchor_x - screen_x / new_scale,
            "y": anchor_y - screen_y / new_scale,
        }
        if self.auto_detail:
            resolved = detail_for_scale(new_scale)
            if resolved is not self.current_detail:
                self.current_detail = resolved
                self._replace_current_representation(fit=False)
                return
            if factor > 1.0 and self._drill_through_at(anchor_x, anchor_y, new_scale):
                return
        self._request_viewport_apply()

    def _drill_through_at(
        self,
        world_x: float,
        world_y: float,
        scale: float,
    ) -> bool:
        """Advance the representation when zooming into a dominant group glyph.

        The absolute ``detail_for_scale`` thresholds never fire on huge models
        whose architecture fit sits at the minimum scale, so a group glyph
        under the cursor that already covers most of the surface is the
        relative signal that the user is zooming *into* it. The transition
        itself reuses the existing auto-detail mechanics (rebuild in place,
        cursor anchor preserved).
        """
        if self.current_detail not in (DetailLevel.ARCHITECTURE, DetailLevel.BLOCK):
            return False
        scene = self.renderer.scene
        if scene is None:
            return False
        hit = self.renderer.hit_test(world_x, world_y)
        if hit is None or scene.has_edge(hit):
            return False
        glyph = scene.node(hit)
        if glyph.kind != "group":
            return False
        width, height = self.surface_size
        coverage = glyph_screen_coverage(
            glyph.width, glyph.height, scale, width, height
        )
        if coverage < _DRILL_COVERAGE:
            return False
        advanced = next_detail_level(self.current_detail)
        if advanced is None:
            return False
        self.current_detail = advanced
        self._replace_current_representation(fit=False)
        return True

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

    def _on_reset_view(self, event: ft.Event[ft.TextButton] | None = None) -> None:
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
            values = self.trace.values_for_glyph(hit)
            names = self.trace.value_names(values)
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
        self.activations.autoload_views(ids)
        self._refresh_inspector(ids)
        self._sync_hierarchy_selection()
        self.trace.refresh_capture_scope()
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
            scene.has_edge(selected) or selected in self.trace.boundary_values()
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
        self.trace.abandon_job()
        if self.export_job is not None and not self.export_job.state.is_terminal:
            self.export_job.cancel()
        self.export_job = None
        if previous is not None and previous is not session and not previous.closed:
            # A reopen must release the outgoing document, its caches, and
            # its open file handle (a lock on Windows) before the new
            # session takes over; otherwise every reopen leaks all three.
            self.service.close_session(previous.id)
        self.session = session
        suggested_limits = recommended_trace_limits(session.document.source.byte_size)
        self.trace.wall_seconds.value = f"{suggested_limits.wall_seconds:g}"
        self.trace.memory_mib.value = str(
            suggested_limits.memory_bytes // (1024 * 1024)
        )
        self._saved_revision_id = None
        self.activations.reset()
        self.trace.reset()
        # Pins and overlay severities belong to the outgoing session.
        self.watch.reset()
        self._reset_overlay()
        self.pending_edit = None
        self.pending_transformation = None
        self.pending_target = None
        # Search results hold buttons bound to the previous model's graph
        # and node ids; keeping them would offer jumps into a closed model.
        self.search_field.value = ""
        self.search_results.controls = []
        self.current_graph = session.document.entry_graph
        self.current_root_group = None
        # Expansion state describes the outgoing model's groups.
        self._hierarchy_expanded.clear()
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
        self.trace.refresh_actions()
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
                    color=(
                        self.palette.accent
                        if entry.graph_id == self.current_graph
                        else self.palette.ink
                    ),
                    bgcolor=(
                        self.palette.accent_soft
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
                    color=self.palette.muted,
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
        self._sync_hierarchy_selection()
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
        self.trace.refresh_graph_actions()
        if self.overlay_mode != "off":
            # The tint survives the rebuild inside the renderer; only the
            # legend's glyph counts depend on the replaced scene.
            self._sync_overlay_controls()

    def _redraw_scene(self) -> None:
        """Re-render the current context in place, keeping the viewport."""
        if self.current_slice is None:
            return
        self.renderer.replace_scene(
            self._display_scene(self.current_slice),
            self._current_viewport(),
        )
        # A redraw follows trace and comparison changes; the watched rows
        # and overlay tints derive from exactly that state.
        self.watch.refresh()
        if self.overlay_mode != "off":
            self._refresh_overlay_tint()

    def _display_scene(self, layout: GraphSlice) -> Scene:
        """Apply the active trace/comparison view without changing base layout."""
        if self.session is None:
            self.trace.presentation = None
            return layout.scene
        scene = layout.scene
        if self.trace.active_comparison is not None:
            scene = self.session.comparison_overlay(
                layout,
                self.trace.active_comparison,
            )
        elif self.trace.active_trace_id is not None:
            try:
                result = self.session.trace(self.trace.active_trace_id)
            except (KeyError, ValueError):
                result = None
            if result is not None:
                scene = self.session.trace_overlay(layout, result)
        if self.current_graph != self.session.document.entry_graph:
            self.trace.presentation = None
            return scene
        presentation = build_trace_graph(
            scene,
            self.session.document.main_graph,
            layout.members_by_glyph,
        )
        self.trace.presentation = presentation
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
            self.trace.refresh_graph_actions()
            if self.overlay_mode != "off":
                self._sync_overlay_controls()
            # A kept viewport can land outside the replacement layout's
            # populated area; recover here because this path never goes
            # through _apply_viewport.
            self._recover_blank_canvas()
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
        self.trace.refresh_graph_actions()
        self._recover_blank_canvas()
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
        self._update_control(self.trace.graph_actions)

    def _culled_scene_is_blank(self) -> bool:
        scene = self.renderer.scene
        if scene is None or scene.node_count == 0:
            return False
        stats = self.renderer.stats
        return stats.visible_nodes == 0 and stats.shape_count == 0

    def _recover_blank_canvas(self) -> None:
        """Never leave the user staring at nothing while the graph has nodes.

        A cursor-anchored zoom deep into inter-glyph space, or a semantic
        rebuild whose layout populates a different region of world space, can
        leave the culled viewport empty. One recovery runs per applied
        viewport: under auto-detail a deeper representation is tried first
        (anchored in place, the existing transition mechanics), and if the
        view is still empty the viewport is clamped onto the nearest glyph so
        at least one stays visible. The flag keeps the recovery's own
        rebuilds from recursing into further recovery — never a loop.
        """
        if self._blank_recovery_active or not self._culled_scene_is_blank():
            return
        self._blank_recovery_active = True
        try:
            if self.auto_detail:
                advanced = next_detail_level(self.current_detail)
                if advanced is not None:
                    self.current_detail = advanced
                    self._replace_current_representation(fit=False)
            if self._culled_scene_is_blank():
                scene = self.renderer.scene
                assert scene is not None
                viewport = self._current_viewport()
                center_x, center_y = nearest_glyph_center(
                    scene.nodes,
                    viewport.x + viewport.width / 2.0,
                    viewport.y + viewport.height / 2.0,
                )
                self.view["x"] = center_x - viewport.width / 2.0
                self.view["y"] = center_y - viewport.height / 2.0
                self.renderer.set_viewport(self._current_viewport())
        finally:
            self._blank_recovery_active = False

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
        # The Model section always mirrors the open document; the inspector
        # list below shows only what the selection is.
        self.model_info.controls = overview.model_overview_controls(
            document, palette=self.palette
        )
        self._sync_info_sections()
        rows: list[ft.Control] = []
        self.back_to_parent_button.visible = self.current_root_group is not None
        self.open_selection_button.visible = False
        if len(ids) == 1 and self.current_graph is not None:
            self.inspector.spacing = 14
            (node_id,) = ids
            traced_rows = self.trace.glyph_inspector(node_id)
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
                        palette=self.palette,
                        pattern=group.kind.value,
                        members=len(group.members),
                        confidence=group.confidence,
                        explanation=group.explanation,
                    )
                )
                rows.extend(
                    self.activations.group_activation_rows(
                        group.members, owner_id=node_id
                    )
                )
            elif node_id.startswith("grp:overview:") and self.current_slice is not None:
                members = self.current_slice.members_by_glyph[node_id]
                self.inspector_title.value = "Architecture region"
                self.inspector_subtitle.value = f"{len(members)} operators"
                self.open_selection_button.visible = True
                rows.extend(
                    overview.selected_region_controls(
                        len(members), palette=self.palette
                    )
                )
                rows.extend(
                    self.activations.group_activation_rows(members, owner_id=node_id)
                )
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
                        palette=self.palette,
                        title="Operator metadata",
                        icon=ft.Icons.DATA_OBJECT_ROUNDED,
                        role="selection-item-metadata",
                    )
                )
                rows.append(
                    ft.TextButton(
                        content="Copy as JSON",
                        icon=ft.Icons.CONTENT_COPY_ROUNDED,
                        data=f"copy-node-json:{node_id}",
                        tooltip="Copy this node's metadata to the clipboard",
                        on_click=self._copy_node_json_handler(node_id),
                    )
                )
                show_tensors = True
            if show_tensors:
                rows.extend(self.activations.tensor_rows(node_id))
                rows.extend(self.activations.activation_rows(node_id))
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
                rows.append(
                    ft.Text(
                        "Nothing is selected. Click a block or operator to "
                        "inspect it here; model-level details live in the "
                        "Model section above.",
                        size=10,
                        color=self.palette.muted,
                    )
                )
        self.inspector.controls = rows
        self._refresh_edit_actions()

    def _copy_node_json_handler(
        self, node_id: str
    ) -> Callable[[ft.Event[ft.TextButton]], None]:
        def copy(event: ft.Event[ft.TextButton]) -> None:
            if self.session is None or self.current_graph is None:
                return
            details = viewmodel.node_details(
                self.session.document, self.current_graph, node_id
            )
            self._start_async(self.clipboard.set, viewmodel.node_details_json(details))
            self._set_status("Node metadata copied to the clipboard as JSON")
            self.page.update()

        return copy

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
                ft.Text(f"[error] {error}", size=10, color=self.palette.danger)
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
                            self.palette.danger
                            if finding.level.value == "error"
                            else self.palette.warning
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
                ft.Text(f"[error] {error}", size=10, color=self.palette.danger)
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
                            self.palette.danger
                            if finding.level.value == "error"
                            else self.palette.warning
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

    def _on_undo_edit(self, event: ft.Event[ft.TextButton] | None = None) -> None:
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

    def _on_redo_edit(self, event: ft.Event[ft.TextButton] | None = None) -> None:
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

    def _hierarchy_selection_ids(self, hierarchy: Hierarchy) -> frozenset[str]:
        """The deepest group per branch the renderer selection maps into.

        A selected group glyph names its group directly; a selected node
        highlights its most specific containing group. Ancestors of another
        highlighted group are pruned so exactly one row lights up per branch.
        """
        highlighted: set[str] = set()
        for item in self.renderer.selection:
            if hierarchy.has_group(item):
                highlighted.add(item)
                continue
            containers = hierarchy.groups_for_node(item)
            if containers:
                highlighted.add(containers[0].id)
        ancestors: set[str] = set()
        for group_id in highlighted:
            ancestors.update(item.id for item in hierarchy.breadcrumbs(group_id)[:-1])
        return frozenset(highlighted - ancestors)

    def _sync_hierarchy_selection(self) -> None:
        """Rebuild the Blocks pane only when its highlight actually changed."""
        if self.session is None or self.current_graph is None:
            return
        hierarchy = self.session.graph_hierarchy(self.current_graph)
        if self._hierarchy_selection_ids(hierarchy) != self._hierarchy_highlight:
            self._refresh_hierarchy()

    def _refresh_hierarchy(self) -> None:
        if self.session is None or self.current_graph is None:
            return
        hierarchy = self.session.graph_hierarchy(self.current_graph)
        highlight = self._hierarchy_selection_ids(hierarchy)
        if highlight != self._hierarchy_highlight:
            # Ancestors open only when the highlight itself moves, so the
            # highlighted row is visible yet a user can still collapse the
            # branch above it afterwards.
            for group_id in highlight:
                for ancestor in hierarchy.breadcrumbs(group_id)[:-1]:
                    self._hierarchy_expanded[ancestor.id] = True
            self._hierarchy_highlight = highlight
        graph = self.session.document.graphs.get(self.current_graph)
        node_order: dict[str, int] = (
            {node.id: index for index, node in enumerate(graph.nodes)}
            if graph is not None
            else {}
        )
        unknown = len(node_order)

        def sort_key(group: Group) -> tuple[int, tuple[tuple[int, int, str], ...]]:
            # Graph position first — the group's earliest member in the
            # serialized node order — then a numeric-aware label so equally
            # placed blocks still read "2" before "10".
            position = min(
                (node_order.get(member, unknown) for member in group.members),
                default=unknown,
            )
            return (position, _natural_key(group.label))

        rows: list[ft.Control] = []
        visible = 0

        def add(group_id: str, depth: int) -> None:
            nonlocal visible
            group = hierarchy.group(group_id)
            children = (
                sorted(hierarchy.children(group.id), key=sort_key)
                if depth + 1 < _MAX_HIERARCHY_DEPTH
                else []
            )
            visible += 1
            if visible <= _MAX_EXPLORER_ROWS:
                rows.append(self._hierarchy_row(group, depth, bool(children)))
            if children and self._hierarchy_expanded.get(group.id, depth == 0):
                for child in children:
                    add(child.id, depth + 1)

        for root in sorted(hierarchy.roots, key=sort_key):
            add(root.id, 0)
        if not rows:
            rows.append(ft.Text("No groups detected", size=11))
        elif visible > _MAX_EXPLORER_ROWS:
            rows.append(
                ft.Text(
                    f"{visible - _MAX_EXPLORER_ROWS:,} more… Use search or "
                    "the graph to inspect the rest.",
                    size=10,
                    color=self.palette.muted,
                )
            )
        self.hierarchy_list.controls = rows

    def _hierarchy_row(
        self, group: Group, depth: int, has_children: bool
    ) -> ft.Control:
        """One Blocks-pane row: an optional chevron plus the select button."""
        selected = group.id in self._hierarchy_highlight
        active = group.id == self.current_root_group
        label = ft.TextButton(
            content=f"{group.label}  ·  {len(group.members)} ops",
            icon=ft.Icons.GRID_VIEW_ROUNDED,
            tooltip=group.explanation,
            data=f"hierarchy-row:{group.id}",
            style=ft.ButtonStyle(
                color=(self.palette.accent if selected or active else self.palette.ink),
                bgcolor=(
                    self.palette.accent_soft if active and not selected else "#00FFFFFF"
                ),
                shape=ft.RoundedRectangleBorder(radius=9),
                alignment=ft.Alignment.CENTER_LEFT,
            ),
            on_click=self._group_handler(group.id),
        )
        row: ft.Control = label
        if has_children or depth:
            controls: list[ft.Control] = []
            if depth:
                controls.append(ft.Container(width=depth * _EXPLORER_INDENT))
            if has_children:
                expanded = self._hierarchy_expanded.get(group.id, depth == 0)
                controls.append(
                    ft.IconButton(
                        icon=(
                            ft.Icons.EXPAND_MORE_ROUNDED
                            if expanded
                            else ft.Icons.CHEVRON_RIGHT_ROUNDED
                        ),
                        icon_size=16,
                        icon_color=self.palette.muted,
                        tooltip="Collapse" if expanded else "Expand",
                        data=f"hierarchy-toggle:{group.id}",
                        on_click=self._hierarchy_toggle_handler(group.id, depth),
                    )
                )
            controls.append(label)
            row = ft.Row(
                controls=controls,
                spacing=0,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        if not selected:
            return row
        # The selection wash covers the whole row, indent and chevron
        # included, which keeps it distinct from the drilled-in root's
        # button-only chip; when a row is both, the selection look wins.
        return ft.Container(
            content=row,
            bgcolor=self.palette.accent_soft,
            border_radius=9,
            data=f"hierarchy-selected:{group.id}",
        )

    def _hierarchy_toggle_handler(
        self, group_id: str, depth: int
    ) -> Callable[[ft.Event[ft.IconButton]], None]:
        def toggle(event: ft.Event[ft.IconButton]) -> None:
            expanded = self._hierarchy_expanded.get(group_id, depth == 0)
            self._hierarchy_expanded[group_id] = not expanded
            self._refresh_hierarchy()
            self.page.update()

        return toggle

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
                        color=self.palette.muted,
                    )
                )
            if crumb.kind == "graph":
                controls.append(
                    ft.TextButton(
                        content=crumb.label,
                        icon=ft.Icons.HOME_ROUNDED,
                        style=ft.ButtonStyle(color=self.palette.ink),
                        on_click=self._graph_handler(crumb.id),
                    )
                )
            else:
                controls.append(
                    ft.TextButton(
                        content=crumb.label,
                        style=ft.ButtonStyle(color=self.palette.ink),
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
                color=self.palette.accent,
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
        """Global shortcuts plus arrow-key navigation over the visible scene."""
        if self._text_input_active:
            # A focused TextField owns the caret; acting on its keystrokes
            # would tear the field down mid-edit and discard the pending text.
            return
        if self._handle_command_shortcut(event):
            return
        if self._handle_view_shortcut(event):
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

    def _handle_command_shortcut(self, event: ft.KeyboardEvent) -> bool:
        """Ctrl-modified application commands; True when one was consumed."""
        if not event.ctrl or event.alt or event.meta:
            return False
        key = (event.key or "").upper()
        if key == "O":
            self._start_async(self._on_open_clicked, None)
            return True
        if key == "S":
            if self.save_model_button.visible and not self.save_model_button.disabled:
                self._start_async(self._export_current_session)
            return True
        if key == "Z":
            if not self.undo_edit_button.disabled:
                self._on_undo_edit(None)
            return True
        if key == "Y":
            if not self.redo_edit_button.disabled:
                self._on_redo_edit(None)
            return True
        if key == "F":
            self._start_async(self.search_field.focus)
            return True
        return False

    def _handle_view_shortcut(self, event: ft.KeyboardEvent) -> bool:
        """Unmodified viewport keys; True when one was consumed."""
        if event.ctrl or event.alt or event.meta:
            return False
        key = event.key or ""
        if key == "Escape":
            # Dismissal order: the large activation overlay first, then the
            # operations drawer, and only then the selection itself.
            if self.activations.close_overlay():
                self.page.update()
            elif self._close_operations_drawer():
                self.page.update()
            else:
                self.renderer.set_selection(frozenset())
                self._on_selected(frozenset())
            return True
        if self.session is None:
            return False
        if key in {"+", "=", "Numpad Add"}:
            self._zoom_about_center(1.1)
            return True
        if key in {"-", "Numpad Subtract"}:
            self._zoom_about_center(0.9)
            return True
        if key in {"0", "Numpad 0"}:
            self._on_reset_view(None)
            return True
        return False

    def _zoom_about_center(self, factor: float) -> None:
        width, height = self.surface_size
        self._zoom_at(width / 2.0, height / 2.0, factor)

    def _start_async(
        self,
        action: Callable[..., Coroutine[Any, Any, object]],
        *args: object,
    ) -> None:
        """Start an async control action from a synchronous handler.

        Headless shells (tests) have no event loop, so the action runs
        inline; a real page schedules it on its own loop.
        """
        run_task = getattr(self.page, "run_task", None)
        if run_task is not None:
            run_task(action, *args)
        else:
            asyncio.run(action(*args))

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

    def _set_inspector_heading(self, title: str, subtitle: str) -> None:
        self.inspector_title.value = title
        self.inspector_subtitle.value = subtitle

    def _set_status(self, text: str) -> None:
        self.status_text.value = text

    def _set_trace_device(
        self,
        device: TraceDevice | None,
        provider: str,
    ) -> None:
        if device is None:
            unavailable = provider == "Unavailable"
            self.device_text.value = (
                "Device: unavailable" if unavailable else "Device: idle"
            )
            color = self.palette.danger if unavailable else self.palette.muted
            self.device_indicator.tooltip = (
                "The requested inference device was unavailable."
                if unavailable
                else "No inference trace is running."
            )
        elif provider == "Unavailable":
            self.device_text.value = f"{device.value.upper()} / unavailable"
            color = self.palette.danger
            self.device_indicator.tooltip = (
                f"No installed execution provider could run this trace on "
                f"{device.value.upper()}."
            )
        elif provider == "Selecting provider":
            self.device_text.value = f"{device.value.upper()} / selecting"
            color = self.palette.accent
            self.device_indicator.tooltip = (
                f"Selecting an installed provider for the requested "
                f"{device.value.upper()} trace."
            )
        else:
            labels = (
                ("Tensorrt", "TensorRT / CUDA"),
                ("CUDA", "CUDA"),
                ("OpenVINO", "OpenVINO"),
                ("QNN", "QNN HTP"),
                ("VitisAI", "Vitis AI"),
                ("Dml", "DirectML"),
                ("MIGraphX", "MIGraphX"),
                ("ROCM", "ROCm"),
                ("CPUExecution", "ONNX Runtime"),
                ("ReferenceEvaluator", "Reference"),
            )
            provider_label = next(
                (label for marker, label in labels if marker in provider),
                provider,
            )
            self.device_text.value = f"{device.value.upper()} / {provider_label}"
            color = {
                TraceDevice.CPU: self.palette.success,
                TraceDevice.GPU: self.palette.info,
                TraceDevice.NPU: "#6941C6",
            }.get(device, self.palette.accent)
            self.device_indicator.tooltip = (
                f"Last completed trace: {device.value.upper()} via {provider}. "
                "Captured outputs are copied to host memory."
            )
        self.device_icon.color = color
        self.device_text.color = color
        self.device_indicator.border = ft.Border.all(1, color)
        # Trace state changed with the device report: mirror it on the
        # Activations info section and its default expansion.
        self._refresh_trace_status_line()
        self._sync_info_sections()

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
    page.theme = ft.Theme(
        color_scheme_seed=_ACCENT,
        use_material3=True,
    )
    service = ApplicationService(job_listener=None, state_store=SessionStateStore())

    # The shell picks the initial palette from the platform brightness and
    # sets page.theme_mode and page.bgcolor itself.
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
