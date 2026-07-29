"""Tests for stable identifier derivation (P0.3)."""

from __future__ import annotations

import pytest

from nneditor.ir.identity import (
    ROOT_GRAPH_ID,
    NodeIdStability,
    attribute_subgraph_id,
    auto_group_id,
    decode_segment,
    encode_segment,
    initializer_id,
    missing_value_id,
    node_id,
    user_group_id,
    value_id,
)


@pytest.mark.parametrize(
    "raw",
    [
        "plain",
        "with/slash",
        "with#hash",
        "with:colon",
        "with%percent",
        "encoder.layer.4/attn:q#0",
        "unicode-é",
    ],
)
def test_segment_encoding_round_trips(raw: str) -> None:
    assert decode_segment(encode_segment(raw)) == raw


@pytest.mark.parametrize("char", ["/", "#", ":"])
def test_encoding_removes_structural_characters(char: str) -> None:
    """Structural characters cannot survive; ``%`` remains as the escape."""
    assert char not in encode_segment(f"a{char}b")


def test_a_literal_percent_is_escaped_so_decoding_is_unambiguous() -> None:
    assert encode_segment("a%2Fb") == "a%252Fb"
    assert decode_segment("a%252Fb") == "a%2Fb"


def test_control_characters_are_encoded() -> None:
    assert encode_segment("a\nb") == "a%0Ab"
    assert decode_segment("a%0Ab") == "a\nb"


def test_non_printable_characters_above_the_byte_range_are_encoded() -> None:
    """U+2028 renders as a line break in some UIs, so it must not survive."""
    hostile = "a\u2028b"
    assert encode_segment(hostile) == "a%u2028b"
    assert decode_segment("a%u2028b") == hostile


def test_a_hostile_name_cannot_forge_another_identifier() -> None:
    """Two different names must never collapse to the same identifier."""
    honest = value_id(ROOT_GRAPH_ID, "weight")
    forged = value_id(ROOT_GRAPH_ID, "g:main#weight")
    assert honest != forged


def test_node_id_prefers_the_source_name() -> None:
    identifier, stability = node_id(
        ROOT_GRAPH_ID, name="scale", first_output="scaled", ordinal=3
    )
    assert identifier == "n:g:main#name:scale"
    assert stability is NodeIdStability.NAMED


def test_node_id_falls_back_to_the_first_output() -> None:
    identifier, stability = node_id(ROOT_GRAPH_ID, first_output="scaled", ordinal=3)
    assert identifier == "n:g:main#out:scaled"
    assert stability is NodeIdStability.OUTPUT_DERIVED


def test_node_id_falls_back_to_position_and_says_so() -> None:
    identifier, stability = node_id(ROOT_GRAPH_ID, ordinal=3)
    assert identifier == "n:g:main#idx:3"
    assert stability is NodeIdStability.POSITIONAL


def test_named_node_id_is_stable_when_neighbours_move() -> None:
    first, _ = node_id(ROOT_GRAPH_ID, name="scale", ordinal=0)
    later, _ = node_id(ROOT_GRAPH_ID, name="scale", ordinal=17)
    assert first == later


def test_subgraph_id_encodes_the_attachment_path() -> None:
    parent, _ = node_id(ROOT_GRAPH_ID, name="branch", ordinal=0)
    then_branch = attribute_subgraph_id(ROOT_GRAPH_ID, parent, "then_branch")
    else_branch = attribute_subgraph_id(ROOT_GRAPH_ID, parent, "else_branch")
    assert then_branch != else_branch
    assert then_branch.startswith(f"{ROOT_GRAPH_ID}/")


def test_repeated_subgraphs_are_distinguished_by_index() -> None:
    parent, _ = node_id(ROOT_GRAPH_ID, name="loop", ordinal=0)
    first = attribute_subgraph_id(ROOT_GRAPH_ID, parent, "bodies", 0)
    second = attribute_subgraph_id(ROOT_GRAPH_ID, parent, "bodies", 1)
    assert first.endswith("[0]")
    assert second.endswith("[1]")


def test_value_and_tensor_identifiers_are_distinct_namespaces() -> None:
    assert value_id(ROOT_GRAPH_ID, "weight") != initializer_id(ROOT_GRAPH_ID, "weight")


def test_missing_value_id_is_scoped_to_its_port() -> None:
    owner, _ = node_id(ROOT_GRAPH_ID, name="conv", ordinal=0)
    assert missing_value_id(owner, 2) != missing_value_id(owner, 3)


def test_auto_group_id_ignores_member_order() -> None:
    members = ["n:g:main#name:a", "n:g:main#name:b", "n:g:main#name:c"]
    forward = auto_group_id("attention", 1, members)
    backward = auto_group_id("attention", 1, reversed(members))
    assert forward == backward


def test_auto_group_id_changes_with_membership_and_version() -> None:
    members = ["n:g:main#name:a", "n:g:main#name:b"]
    base = auto_group_id("attention", 1, members)
    assert base != auto_group_id("attention", 2, members)
    assert base != auto_group_id("attention", 1, [*members, "n:g:main#name:c"])
    assert base != auto_group_id("residual", 1, members)


def test_user_group_ids_are_scoped_to_a_graph() -> None:
    assert user_group_id(ROOT_GRAPH_ID, "Encoder") != user_group_id(
        "g:main/sub", "Encoder"
    )
