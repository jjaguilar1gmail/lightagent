"""
app.tracing — stable tracing substrate for lightagent.

Public API (never changes even as nodes/edges/tools evolve):

    from app.tracing import emit, new_run, traced_node, traced_router, traced_tool

Sinks:
    ConsoleSink  — pretty narrative (default) or raw JSON debug output
    JSONLSink    — traces/<run_id>.jsonl  (one JSON line per event)
    SQLiteSink   — traces/traces.db       (queryable cross-run store)
    SSESink      — asyncio queue for FastAPI /trace/stream SSE endpoint
"""

from .events import EventType, TraceEvent
from .emitter import TraceEmitter, emit, new_run
from .decorator import traced_node, traced_router, traced_tool
from .callbacks import TracingCallbackHandler
from .sinks.console import ConsoleSink
from .sinks.jsonl import JSONLSink
from .sinks.sqlite import SQLiteSink
from .sinks.sse import SSESink, make_sse_router

__all__ = [
    # core types
    "TraceEvent",
    "EventType",
    "TraceEmitter",
    # emit API
    "emit",
    "new_run",
    # decorators
    "traced_node",
    "traced_router",
    "traced_tool",
    # callback bridge
    "TracingCallbackHandler",
    # sinks
    "ConsoleSink",
    "JSONLSink",
    "SQLiteSink",
    "SSESink",
    "make_sse_router",
]
