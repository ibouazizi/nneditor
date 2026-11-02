"""Metadata-only graph stages bound to explicit runtime module hooks."""

from __future__ import annotations

from collections.abc import Mapping

from nneditor.ir.core import Attribute, AttrKind, Graph, Node, Value
from nneditor.ir.identity import ROOT_GRAPH_ID, node_id, value_id


class StageBuilder:
    def __init__(self, weights: Mapping[str, str], title: str) -> None:
        self.weights = weights
        self.title = title
        self.values: dict[str, Value] = {}
        self.nodes: list[Node] = []
        self.inputs: list[str] = []

    def value(
        self,
        name: str,
        shape: tuple[int | str, ...] | None = None,
        dtype: str = "float32",
    ) -> str:
        key = value_id(ROOT_GRAPH_ID, name)
        self.values[key] = Value(key, name=name, element_type=dtype, shape=shape)
        return key

    def input(
        self, name: str, shape: tuple[int | str, ...], dtype: str = "float32"
    ) -> str:
        key = self.value(name, shape, dtype)
        self.inputs.append(key)
        return key

    def stage(
        self,
        name: str,
        label: str,
        operation: str,
        feeds: tuple[str, ...],
        outputs: tuple[str, ...],
        module: str,
        *,
        domain: str,
        weights_from: str | None = None,
        capture_input: str | None = None,
    ) -> None:
        identifier, stability = node_id(
            ROOT_GRAPH_ID, name=name, ordinal=len(self.nodes)
        )
        prefix = module if weights_from is None else weights_from
        if capture_input is not None and weights_from is None:
            prefix = "output"
        weights: tuple[Attribute, ...] = ()
        if prefix != "output":
            weights = tuple(
                Attribute(name, AttrKind.TENSOR, identifier)
                for name, identifier in self.weights.items()
                if name == prefix or name.startswith(prefix + ".")
            )
        self.nodes.append(
            Node(
                identifier,
                stability,
                operation,
                feeds,
                outputs,
                domain=domain,
                source_name=label,
                attributes=(
                    Attribute("module", AttrKind.STRING, module),
                    *(
                        (Attribute("capture_input", AttrKind.STRING, capture_input),)
                        if capture_input is not None
                        else ()
                    ),
                    *weights,
                ),
            )
        )

    def blocks(
        self,
        prefix: str,
        count: int,
        operation: str,
        module: str,
        previous: str,
        *,
        domain: str,
        also: tuple[str, ...] = (),
    ) -> str:
        for index in range(count):
            output = self.value(f"{prefix}-{index:03d}")
            self.stage(
                f"{prefix}-{index:03d}",
                f"{operation} {index}",
                operation,
                (previous, *also),
                (output,),
                module.format(index=index),
                domain=domain,
            )
            previous = output
        return previous

    def step(
        self,
        name: str,
        operation: str,
        feeds: tuple[str, ...],
        module: str,
        *,
        domain: str,
        capture_input: str | None = None,
        weights_from: str | None = None,
    ) -> str:
        output = self.value(name)
        self.stage(
            name,
            name.replace("-", " ").capitalize(),
            operation,
            feeds,
            (output,),
            module,
            domain=domain,
            capture_input=capture_input,
            weights_from=weights_from,
        )
        return output

    def finish(
        self,
        hidden: str,
        head: tuple[str, str],
        vocab: int,
        domain: str,
        *,
        sequence_dimension: str = "seq",
    ) -> str:
        normalized = self.value("final-norm")
        self.stage(
            "final-norm",
            "Final normalization",
            "LayerNorm",
            (hidden,),
            (normalized,),
            head[0],
            domain=domain,
        )
        logits = self.value("logits", ("batch", sequence_dimension, vocab))
        self.stage(
            "logits",
            "Logits",
            "Output",
            (normalized,),
            (logits,),
            "output",
            domain=domain,
            weights_from=head[1],
        )
        return logits

    def graph(self, output: str) -> Graph:
        return Graph(
            id=ROOT_GRAPH_ID,
            name=self.title,
            nodes=self.nodes,
            values=self.values.values(),
            inputs=self.inputs,
            outputs=(output,),
            initializers=tuple(dict.fromkeys(self.weights.values())),
        )
