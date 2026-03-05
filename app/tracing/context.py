from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .emitter import TraceEmitter

# One ContextVar per process. Inherits into threads automatically (Python 3.7+).
_emitter_var: contextvars.ContextVar[Optional["TraceEmitter"]] = contextvars.ContextVar(
    "lightagent_trace_emitter", default=None
)


def get_emitter() -> Optional["TraceEmitter"]:
    return _emitter_var.get()
