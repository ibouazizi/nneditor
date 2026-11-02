"""BitNet tracing for a Hugging Face snapshot or a float GGUF."""

from nneditor.tracing.sdk import TraceContext, TraceOutcome


def trace(context: TraceContext) -> TraceOutcome:
    if context.document.artifact_kind.value == "gguf_model":
        from nneditor.tracing.gguf_bitnet import trace_gguf_bitnet

        return trace_gguf_bitnet(context)
    from nneditor.tracing.modality_runtime import trace_modality

    return trace_modality(context)
