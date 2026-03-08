from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

# Every event type that can ever be emitted. Stable forever.
EventType = Literal[
    "run_start",
    "run_end",
    "node_start",
    "node_end",
    "node_error",
    "llm_call_start",
    "llm_call_end",
    "tool_call_start",
    "tool_call_end",
    "tool_error",
    "routing_decision",
]


@dataclass
class TraceEvent:
    type: EventType
    run_id: str
    seq: int                              # monotonic per run
    ts: str                               # ISO-8601 UTC
    node: Optional[str]                   # node/tool name, None for run-level events
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "run_id": self.run_id,
            "seq": self.seq,
            "ts": self.ts,
            "node": self.node,
            "payload": self.payload,
        }

    @staticmethod
    def now_ts() -> str:
        return datetime.now(timezone.utc).isoformat()
