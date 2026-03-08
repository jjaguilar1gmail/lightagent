from __future__ import annotations

import itertools
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .context import _emitter_var, get_emitter
from .events import EventType, TraceEvent


class TraceEmitter:
    """
    Fan-out emitter for a single run. Thread-safe: seq uses itertools.count
    (CPython int increment is GIL-protected). Sinks must be cheap; errors are
    swallowed so the agent is never affected by a broken sink.
    """

    def __init__(self, run_id: Optional[str] = None, sinks: Optional[List] = None):
        self.run_id: str = run_id or uuid.uuid4().hex[:12]
        self._sinks: List = list(sinks or [])
        self._counter = itertools.count(1)
        self._run_start: float = time.monotonic()

    def add_sink(self, sink) -> None:
        self._sinks.append(sink)

    def emit(
        self,
        type: EventType,
        node: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> TraceEvent:
        event = TraceEvent(
            type=type,
            run_id=self.run_id,
            seq=next(self._counter),
            ts=TraceEvent.now_ts(),
            node=node,
            payload=payload or {},
        )
        for sink in self._sinks:
            try:
                sink.handle(event)
            except Exception:
                pass  # never crash the agent because a sink failed
        return event

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._run_start) * 1000)


# ---------------------------------------------------------------------------
# Module-level helpers — the stable public API that nodes/tools call
# ---------------------------------------------------------------------------

def emit(
    type: EventType,
    node: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit an event on the active run's emitter. No-op if no run is active."""
    emitter = get_emitter()
    if emitter is not None:
        emitter.emit(type, node=node, payload=payload)


@contextmanager
def new_run(sinks=None, run_id: Optional[str] = None):
    """
    Context manager that starts a traced run.

    Usage::

        with new_run([ConsoleSink(), JSONLSink()]) as run:
            result = graph.invoke(state)
    """
    emitter = TraceEmitter(run_id=run_id, sinks=sinks or [])
    token = _emitter_var.set(emitter)
    emitter.emit("run_start", payload={"run_id": emitter.run_id})
    try:
        yield emitter
    except Exception as exc:
        emitter.emit(
            "run_end",
            payload={
                "run_id": emitter.run_id,
                "status": "error",
                "error": str(exc),
                "elapsed_ms": emitter.elapsed_ms(),
            },
        )
        raise
    else:
        emitter.emit(
            "run_end",
            payload={
                "run_id": emitter.run_id,
                "status": "ok",
                "elapsed_ms": emitter.elapsed_ms(),
            },
        )
    finally:
        _emitter_var.reset(token)
