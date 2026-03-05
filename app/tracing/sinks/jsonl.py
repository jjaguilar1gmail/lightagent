from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Optional, Union

from ..events import TraceEvent


class JSONLSink:
    """
    Appends one JSON line per event to ``{directory}/{run_id}.jsonl``.

    The file is opened lazily on the first event so the run_id is known.
    Flush after every write so a crash mid-run still yields a useful trace.
    """

    def __init__(self, directory: Union[str, Path] = "traces"):
        self._dir = Path(directory)
        self._file: Optional[IO[str]] = None
        self._path: Optional[Path] = None

    def handle(self, event: TraceEvent) -> None:
        if self._file is None:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._path = self._dir / f"{event.run_id}.jsonl"
            self._file = open(self._path, "a", encoding="utf-8")

        self._file.write(json.dumps(event.to_dict(), default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
