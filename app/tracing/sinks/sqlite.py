from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Union

from ..events import TraceEvent

_DDL = """
CREATE TABLE IF NOT EXISTS trace_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  TEXT    NOT NULL,
    seq     INTEGER NOT NULL,
    ts      TEXT    NOT NULL,
    type    TEXT    NOT NULL,
    node    TEXT,
    payload TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_id  ON trace_events (run_id);
CREATE INDEX IF NOT EXISTS idx_type    ON trace_events (type);
CREATE INDEX IF NOT EXISTS idx_run_seq ON trace_events (run_id, seq);
"""

# Useful queries to paste into a SQLite browser or Python session:
#
#   -- All events for a run in order
#   SELECT seq, ts, type, node, payload FROM trace_events
#   WHERE run_id = '<id>' ORDER BY seq;
#
#   -- All tool calls across runs
#   SELECT run_id, ts, node, payload FROM trace_events
#   WHERE type IN ('tool_call_start','tool_call_end') ORDER BY ts;
#
#   -- Runs that hit an error
#   SELECT DISTINCT run_id FROM trace_events WHERE type = 'node_error';


class SQLiteSink:
    """
    Writes every event into ``traces/traces.db`` (configurable).

    check_same_thread=False is safe here because we serialise through the GIL
    and only ever write one event at a time.
    """

    def __init__(self, path: Union[str, Path] = "traces/traces.db"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_DDL)
        self._conn.commit()

    def handle(self, event: TraceEvent) -> None:
        self._conn.execute(
            """
            INSERT INTO trace_events (run_id, seq, ts, type, node, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.run_id,
                event.seq,
                event.ts,
                event.type,
                event.node,
                json.dumps(event.payload, default=str),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
