"""A working LLM tracer extension. Replace the forward pass for a new runtime.

Set your plugin's id, name, version, and supported architectures in the JSON
manifest. Install dependencies in NNEditor's Python environment. This file
and sibling helper modules are imported only by the approved trace worker.
"""

from nneditor.tracing.sdk import TraceContext, TraceOutcome


def trace(context: TraceContext) -> TraceOutcome:
    # Working baseline: captures inputs and every inferred decoder layer.
    return context.trace_numpy_decoder()

    # For a different runtime, replace the line above with your forward pass:
    #   context.capture_inputs()
    #   tokens = context.inputs["tokens"]
    #   ... read weights through context.tensors, run layers ...
    #   context.capture("embed", hidden)
    #   context.capture("block-000-attn", hidden)
    #   context.capture("lm-head", logits)
    # Capture names must match context.document.main_graph.values.
    # Emit arrays immediately: NNEditor writes only requested captures and
    # applies the approved byte budget. Do not keep every layer in a dict.
