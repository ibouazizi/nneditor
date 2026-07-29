"""Deterministic ONNX fixtures.

The reference ``onnx`` package builds these files. It is a test-only dependency:
using it here proves the hand-written lazy indexer agrees with the canonical
parser, while keeping that parser out of the product, where it would defeat the
point by materializing every tensor.

Fixtures stay small except where a test needs a payload large enough for a
bytes-read budget to mean something.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.external_data_helper import set_external_data

__all__ = [
    "LARGE_TENSOR_ELEMENTS",
    "build_attribute_variety_model",
    "build_control_flow_model",
    "build_custom_domain_model",
    "build_embedded_model",
    "build_external_model",
    "build_function_model",
    "build_masked_image_model",
    "build_matmul_model",
    "build_optional_input_model",
    "build_symbolic_shape_model",
    "build_tensor_only_model",
    "external_tensor_model",
]

LARGE_TENSOR_ELEMENTS = 512 * 1024
"""Two mebibytes of float32 weights: large enough to dominate the file."""

_OPSET = 18


def _weights(elements: int, *, seed: int = 7) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.standard_normal(elements, dtype=np.float32)


def _external_tensor(
    name: str,
    dims: list[int],
    *,
    location: str,
    offset: int,
    length: int | None,
) -> TensorProto:
    """Build a tensor whose data lives outside the model file.

    ``set_external_data`` requires a ``raw_data`` field to be present, so the
    field is set and then cleared once the external metadata is attached.
    """
    tensor = TensorProto()
    tensor.name = name
    tensor.data_type = TensorProto.FLOAT
    tensor.dims.extend(dims)
    tensor.raw_data = b""
    if length is None:
        set_external_data(tensor, location=location, offset=offset)
    else:
        set_external_data(tensor, location=location, offset=offset, length=length)
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL
    return tensor


def build_embedded_model(
    path: Path, *, elements: int = LARGE_TENSOR_ELEMENTS
) -> np.ndarray:
    """A two-node model whose weights are embedded ``raw_data``."""
    values = _weights(elements)
    weight = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=[elements],
        vals=values.tobytes(),
        raw=True,
    )
    bias = helper.make_tensor(
        name="bias", data_type=TensorProto.FLOAT, dims=[1], vals=[0.5], raw=False
    )
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Mul", ["input", "weight"], ["scaled"], name="scale"),
            helper.make_node("Add", ["scaled", "bias"], ["output"], name="shift"),
        ],
        name="embedded",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [elements])],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [elements])
        ],
        initializer=[weight, bias],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", _OPSET)], producer_name="nneditor"
    )
    model.producer_version = "test"
    onnx.save_model(model, path)
    return values


def build_external_model(
    directory: Path,
    *,
    elements: int = LARGE_TENSOR_ELEMENTS,
    location: str = "weights.bin",
) -> tuple[Path, np.ndarray]:
    """A model whose weights live in a sibling file, addressed by range."""
    values = _weights(elements, seed=11)
    payload = values.tobytes()
    (directory / location).write_bytes(payload)

    weight = _external_tensor(
        "weight", [elements], location=location, offset=0, length=len(payload)
    )

    graph = helper.make_graph(
        nodes=[helper.make_node("Mul", ["input", "weight"], ["output"], name="scale")],
        name="external",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [elements])],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [elements])
        ],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    path = directory / "model.onnx"
    onnx.save_model(model, path)
    return path, values


def build_matmul_model(
    path: Path,
    *,
    rows: int = 3,
    columns: int = 4,
    batch: int = 2,
    terminal: bool = True,
    opset: int = _OPSET,
) -> np.ndarray:
    """A statically shaped MatMul with a rank-2 initializer."""
    values = np.arange(rows * columns, dtype=np.float32).reshape(rows, columns)
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [rows, columns],
        values.tobytes(),
        raw=True,
    )
    matmul_output = "output" if terminal else "projected"
    nodes = [
        helper.make_node(
            "MatMul",
            ["input", "weight"],
            [matmul_output],
            name="projection",
        )
    ]
    value_info: list[onnx.ValueInfoProto] = []
    if not terminal:
        nodes.append(
            helper.make_node("Identity", [matmul_output], ["output"], name="tail")
        )
        value_info.append(
            helper.make_tensor_value_info(
                matmul_output,
                TensorProto.FLOAT,
                [batch, columns],
            )
        )
    graph = helper.make_graph(
        nodes,
        "matmul",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, rows])],
        [
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT,
                [batch, columns],
            )
        ],
        initializer=[weight],
        value_info=value_info,
    )
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)]),
        path,
    )
    return values


def build_shared_weight_matmul_model(
    path: Path, *, rows: int = 3, columns: int = 4, batch: int = 2
) -> np.ndarray:
    """Two terminal MatMuls sharing one weight initializer.

    Every guard of the structured-pruning pattern passes for either MatMul
    except consumer exclusivity — this fixture exists to prove that resizing
    a shared weight is rejected instead of silently corrupting the sibling.
    """
    values = np.arange(rows * columns, dtype=np.float32).reshape(rows, columns)
    weight = helper.make_tensor(
        "weight", TensorProto.FLOAT, [rows, columns], values.tobytes(), raw=True
    )
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weight"], ["output"], name="first"),
            helper.make_node(
                "MatMul", ["input", "weight"], ["second_output"], name="second"
            ),
        ],
        "shared_weight",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, rows])],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [batch, columns]
            ),
            helper.make_tensor_value_info(
                "second_output", TensorProto.FLOAT, [batch, columns]
            ),
        ],
        initializer=[weight],
    )
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)]),
        path,
    )
    return values


def external_tensor_model(
    directory: Path,
    *,
    location: str,
    offset: int,
    length: int | None,
    dims: list[int],
    payload: bytes | None = None,
    payload_name: str = "weights.bin",
) -> Path:
    """A one-tensor model with fully controlled external-data metadata.

    Used for the malformed and hostile cases: traversal, bad offsets, missing
    files, and length disagreements.
    """
    if payload is not None:
        (directory / payload_name).write_bytes(payload)
    weight = _external_tensor(
        "weight", dims, location=location, offset=offset, length=length
    )

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Identity", ["weight"], ["output"], name="passthrough")
        ],
        name="controlled",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, dims)],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    path = directory / "model.onnx"
    onnx.save_model(model, path)
    return path


def build_symbolic_shape_model(path: Path) -> None:
    """A model with a named dynamic batch dimension and an unknown extent."""
    graph = helper.make_graph(
        nodes=[helper.make_node("Relu", ["input"], ["output"], name="activation")],
        name="symbolic",
        inputs=[
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["batch", 3, "height", 224]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, ["batch", 3, "height", 224]
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    onnx.save_model(model, path)


def build_masked_image_model(path: Path) -> None:
    """Image input plus a required integer mask sharing a symbolic batch."""
    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", ["pixel_values"], ["output"])],
        name="masked-image",
        inputs=[
            helper.make_tensor_value_info(
                "pixel_values",
                TensorProto.FLOAT,
                ["batch_size", 3, 224, 224],
            ),
            helper.make_tensor_value_info(
                "pixel_mask",
                TensorProto.INT64,
                ["batch_size", 64, 64],
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT,
                ["batch_size", 3, 224, 224],
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    onnx.save_model(model, path)


def build_custom_domain_model(path: Path) -> None:
    """A model using an operator from a non-standard domain."""
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "FusedGelu",
                ["input"],
                ["output"],
                name="fused",
                domain="com.example.ops",
                approximation="tanh",
            )
        ],
        name="custom",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", _OPSET),
            helper.make_opsetid("com.example.ops", 1),
        ],
    )
    onnx.save_model(model, path)


def build_control_flow_model(path: Path) -> None:
    """An ``If`` node carrying two subgraphs, each with its own constant."""
    then_constant = helper.make_tensor(
        "then_value", TensorProto.FLOAT, [1], struct.pack("<f", 1.0), raw=True
    )
    else_constant = helper.make_tensor(
        "else_value", TensorProto.FLOAT, [1], struct.pack("<f", -1.0), raw=True
    )
    then_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Constant",
                [],
                ["branch_output"],
                name="then_const",
                value=then_constant,
            )
        ],
        name="then_body",
        inputs=[],
        outputs=[
            helper.make_tensor_value_info("branch_output", TensorProto.FLOAT, [1])
        ],
    )
    else_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Constant",
                [],
                ["branch_output"],
                name="else_const",
                value=else_constant,
            )
        ],
        name="else_body",
        inputs=[],
        outputs=[
            helper.make_tensor_value_info("branch_output", TensorProto.FLOAT, [1])
        ],
    )
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "If",
                ["condition"],
                ["output"],
                name="branch",
                then_branch=then_graph,
                else_branch=else_graph,
            )
        ],
        name="control_flow",
        inputs=[helper.make_tensor_value_info("condition", TensorProto.BOOL, [])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    onnx.save_model(model, path)


def build_function_model(path: Path) -> None:
    """A model carrying a local function definition."""
    function = helper.make_function(
        domain="com.example.fn",
        fname="Scale",
        inputs=["x"],
        outputs=["y"],
        nodes=[helper.make_node("Mul", ["x", "x"], ["y"], name="square")],
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Scale", ["input"], ["output"], name="call", domain="com.example.fn"
            )
        ],
        name="with_function",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", _OPSET),
            helper.make_opsetid("com.example.fn", 1),
        ],
        functions=[function],
    )
    onnx.save_model(model, path)


def build_attribute_variety_model(path: Path) -> None:
    """One custom-domain node carrying every scalar and list attribute kind."""
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Probe",
                ["input"],
                ["output"],
                name="probe",
                domain="com.example.attrs",
                axis=-2,
                alpha=1.5,
                mode="reflect",
                pads=[0, 1, 2],
                scales=[0.5, 2.0],
                labels=["a", "b"],
            )
        ],
        name="attributes",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", _OPSET),
            helper.make_opsetid("com.example.attrs", 1),
        ],
    )
    onnx.save_model(model, path)


def build_optional_input_model(path: Path) -> None:
    """A ``Clip`` whose optional ``min`` input is omitted by position."""
    limit = helper.make_tensor(
        "limit", TensorProto.FLOAT, [], struct.pack("<f", 6.0), raw=True
    )
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Clip", ["input", "", "limit"], ["output"], name="clip")
        ],
        name="optional_input",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [4])],
        initializer=[limit],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    onnx.save_model(model, path)


def build_tensor_only_model(
    path: Path, *, dims: list[int], raw_bytes: bytes, data_type: int
) -> None:
    """A one-initializer model used to exercise payload validation."""
    weight = TensorProto()
    weight.name = "weight"
    weight.data_type = data_type
    weight.dims.extend(dims)
    weight.raw_data = raw_bytes
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Identity", ["weight"], ["output"], name="passthrough")
        ],
        name="tensor_only",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, dims)],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", _OPSET)])
    onnx.save_model(model, path)
