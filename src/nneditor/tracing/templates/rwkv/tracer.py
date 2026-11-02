"""RWKV recurrent tracing on CPU."""

from nneditor.tracing.modality_runtime import trace_modality
from nneditor.tracing.sdk import TraceContext, TraceOutcome


def trace(context: TraceContext) -> TraceOutcome:
    return trace_modality(context)
