"""GGUF vision-tower and Whisper tracing."""

from nneditor.tracing.gguf_media import trace_gguf_media
from nneditor.tracing.sdk import TraceContext, TraceOutcome


def trace(context: TraceContext) -> TraceOutcome:
    return trace_gguf_media(context)
