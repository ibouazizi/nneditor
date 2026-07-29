"""Block organization modes: controller semantics, persistence, and toolbar.

The organization mode filters the already-detected candidate pool by kind
before reconciliation. AUTO keeps today's priority-reconciled behavior; each
named mode keeps one detector family plus control-flow containment. Manual
corrections are user intent and apply in every mode. The chosen mode is
per-artifact sidecar state restored on reopen, and the toolbar button cycles
through the modes in declaration order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import flet as ft
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from nneditor.analysis.hierarchy import CandidateKind
from nneditor.application.hierarchy import (
    HierarchySidecarStore,
    OrganizationMode,
)
from nneditor.application.session import ApplicationService, ModelSession
from nneditor.ui.app import Shell

_CYCLE = (
    OrganizationMode.AUTO,
    OrganizationMode.SOURCE,
    OrganizationMode.REPEATED,
    OrganizationMode.PATTERNS,
    OrganizationMode.STRUCTURAL,
)


class StubPage:
    """The minimal page surface the shell touches."""

    def __init__(self) -> None:
        self.overlay: list[ft.Control] = []
        self.services: list[object] = []
        self.updates = 0
        self.width = 1400

    def update(self) -> None:
        self.updates += 1

    def run_thread(self, target: Any) -> None:
        target()


@pytest.fixture
def service() -> Any:
    with ApplicationService() as instance:
        yield instance


def make_shell(service: ApplicationService) -> tuple[Shell, StubPage]:
    page = StubPage()
    shell = Shell(cast(ft.Page, page), service)
    shell.build()
    return shell, page


def build_scoped_attention_model(path: Path) -> None:
    """Exporter scopes *and* an attention motif in one four-node model.

    Scope prefixes give the source detector ``blockA`` = {gemm_q, softmax}
    and ``blockB`` = {gemm_o, relu}; the Gemm → Softmax → Gemm spine gives
    the pattern detector one ``Attention`` motif spanning both scopes — so
    SOURCE and PATTERNS modes must reconcile visibly different hierarchies.
    """
    weights = np.eye(4, dtype=np.float32)
    w1 = helper.make_tensor(
        "w1", TensorProto.FLOAT, [4, 4], weights.tobytes(), raw=True
    )
    w2 = helper.make_tensor(
        "w2", TensorProto.FLOAT, [4, 4], weights.tobytes(), raw=True
    )
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Gemm", ["input", "w1"], ["t1"], name="blockA/gemm_q"),
            helper.make_node("Softmax", ["t1"], ["t2"], name="blockA/softmax"),
            helper.make_node("Gemm", ["t2", "w2"], ["t3"], name="blockB/gemm_o"),
            helper.make_node("Relu", ["t3"], ["output"], name="blockB/relu"),
        ],
        name="scoped_attention",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 4])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 4])],
        initializer=[w1, w2],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="nneditor"
    )
    onnx.save_model(model, path)


def open_scoped_session(tmp_path: Path, service: ApplicationService) -> ModelSession:
    path = tmp_path / "model.onnx"
    build_scoped_attention_model(path)
    return service.open_model(path)


def node_ids_by_name(session: ModelSession) -> dict[str, str]:
    return {
        node.source_name: node.id
        for node in session.document.main_graph.nodes
        if node.source_name
    }


def test_toolbar_cycles_through_all_modes_and_wraps(
    tmp_path: Path, service: ApplicationService
) -> None:
    shell, _page = make_shell(service)
    session = open_scoped_session(tmp_path, service)
    shell.show_session(session)
    assert session.hierarchy.mode is OrganizationMode.AUTO
    assert shell.organize_button.tooltip == "Organize: Auto — press to cycle"

    visited: list[OrganizationMode] = []
    for _ in range(len(_CYCLE)):
        shell._on_cycle_organization(cast(Any, None))
        visited.append(session.hierarchy.mode)

    assert tuple(visited) == (*_CYCLE[1:], OrganizationMode.AUTO)
    assert shell.organize_button.tooltip == "Organize: Auto — press to cycle"


def test_cycling_updates_status_tooltip_and_blocks_panel(
    tmp_path: Path, service: ApplicationService
) -> None:
    shell, page = make_shell(service)
    session = open_scoped_session(tmp_path, service)
    shell.show_session(session)

    shell._on_cycle_organization(cast(Any, None))

    assert session.hierarchy.mode is OrganizationMode.SOURCE
    assert shell.status_text.value == "Organized by Source"
    assert shell.organize_button.tooltip == "Organize: Source — press to cycle"
    labels = " ".join(
        str(control.content)
        for control in shell.hierarchy_list.controls
        if isinstance(control, ft.TextButton)
    )
    assert "blockA" in labels and "blockB" in labels
    assert page.updates > 0


def test_source_and_patterns_modes_reconcile_different_groups(
    tmp_path: Path, service: ApplicationService
) -> None:
    session = open_scoped_session(tmp_path, service)
    ids = node_ids_by_name(session)
    auto_revision = session.graph_hierarchy().revision

    session.hierarchy.set_mode(OrganizationMode.SOURCE)
    source = session.graph_hierarchy()
    assert {group.label for group in source.groups} == {"blockA", "blockB"}
    assert all(group.kind is CandidateKind.SOURCE for group in source.groups)

    session.hierarchy.set_mode(OrganizationMode.PATTERNS)
    patterns = session.graph_hierarchy()
    assert [group.label for group in patterns.groups] == ["Attention"]
    attention = patterns.groups[0]
    assert attention.kind is CandidateKind.PATTERN
    assert attention.members == {
        ids["blockA/gemm_q"],
        ids["blockA/softmax"],
        ids["blockB/gemm_o"],
    }

    # Different groups mean different content-derived revisions, which is
    # what keeps the semantic-layout slice caches from serving stale scenes.
    assert source.revision != patterns.revision
    assert auto_revision not in (source.revision, patterns.revision)

    session.hierarchy.set_mode(OrganizationMode.AUTO)
    assert session.graph_hierarchy().revision == auto_revision


def test_manual_groups_survive_mode_switches(
    tmp_path: Path, service: ApplicationService
) -> None:
    session = open_scoped_session(tmp_path, service)
    ids = node_ids_by_name(session)
    members = frozenset({ids["blockB/gemm_o"], ids["blockB/relu"]})
    manual = session.hierarchy.group(session.document.entry_graph, "Tail", members)

    for mode in (
        OrganizationMode.PATTERNS,
        OrganizationMode.STRUCTURAL,
        OrganizationMode.SOURCE,
        OrganizationMode.AUTO,
    ):
        session.hierarchy.set_mode(mode)
        hierarchy = session.graph_hierarchy()
        assert hierarchy.has_group(manual.id), mode
        survivor = hierarchy.group(manual.id)
        assert not survivor.automatic
        assert survivor.label == "Tail"
        assert survivor.members == members


def test_mode_persists_across_close_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    build_scoped_attention_model(path)
    state = tmp_path / "state"

    with ApplicationService(hierarchy_store=HierarchySidecarStore(state)) as first:
        session = first.open_model(path)
        session.hierarchy.set_mode(OrganizationMode.PATTERNS)
        first.close_session(session.id)

    with ApplicationService(hierarchy_store=HierarchySidecarStore(state)) as second:
        reopened = second.open_model(path)
        assert reopened.hierarchy.mode is OrganizationMode.PATTERNS
        assert [group.label for group in reopened.graph_hierarchy().groups] == [
            "Attention"
        ]
        shell, _page = make_shell(second)
        shell.show_session(reopened)
        assert shell.organize_button.tooltip == "Organize: Patterns — press to cycle"


def test_drilled_in_root_is_repaired_when_the_mode_drops_it(
    tmp_path: Path, service: ApplicationService
) -> None:
    shell, _page = make_shell(service)
    session = open_scoped_session(tmp_path, service)
    shell.show_session(session)
    attention = next(
        group
        for group in session.graph_hierarchy().groups
        if group.kind is CandidateKind.PATTERN
    )
    shell.show_group(attention.id)
    assert shell.current_root_group == attention.id

    # AUTO → SOURCE: the attention group is not reconciled from source
    # scopes, so the drilled-in root must fall back to the whole graph
    # instead of wedging the rebuild on a nonexistent group.
    shell._on_cycle_organization(cast(Any, None))

    assert session.hierarchy.mode is OrganizationMode.SOURCE
    assert shell.current_root_group is None
    assert shell.renderer.scene is not None
    assert not session.graph_hierarchy().has_group(attention.id)


def test_version1_sidecar_without_mode_loads_as_auto(tmp_path: Path) -> None:
    """Older sidecars stay readable; their corrections and AUTO default load."""
    state = tmp_path / "state"
    state.mkdir()
    content_hash = "sha256:" + "a" * 64
    payload = {
        "version": 1,
        "artifacts": {
            content_hash: {
                "manual_groups": [
                    {
                        "id": "user:kept",
                        "graph_id": "g:main",
                        "label": "Kept",
                        "members": ["n0", "n1"],
                        "locked": False,
                    }
                ],
                "rejected": ["auto:gone"],
                "renames": {},
                "locked": [],
            }
        },
    }
    (state / "hierarchy-view.json").write_text(json.dumps(payload), encoding="utf-8")

    store = HierarchySidecarStore(state)
    overrides = store.for_artifact(content_hash)
    assert overrides.mode is OrganizationMode.AUTO
    assert overrides.manual_groups["user:kept"].label == "Kept"
    assert overrides.rejected == {"auto:gone"}

    # Saving upgrades the document in place without losing anything.
    overrides.mode = OrganizationMode.REPEATED
    store.save()
    raw = json.loads((state / "hierarchy-view.json").read_text(encoding="utf-8"))
    assert raw["version"] == 2
    reloaded = HierarchySidecarStore(state).for_artifact(content_hash)
    assert reloaded.mode is OrganizationMode.REPEATED
    assert reloaded.manual_groups["user:kept"].members == frozenset({"n0", "n1"})
