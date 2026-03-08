from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Optional

from ..events import TraceEvent


class SSESink:
    """
    Puts serialised events into an ``asyncio.Queue`` so a FastAPI SSE endpoint
    can stream them to a browser or other HTTP client in real-time.

    Works from synchronous LangGraph nodes via ``call_soon_threadsafe``.

    Usage with FastAPI::

        sink = SSESink()
        router = make_sse_router(sink)

        app = FastAPI()
        app.include_router(router)

        # In your run handler:
        with new_run([ConsoleSink(), sink]) as run:
            result = graph.invoke(state)
    """

    def __init__(self, maxsize: int = 512):
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)

    def handle(self, event: TraceEvent) -> None:
        data = json.dumps(event.to_dict(), default=str)
        try:
            # Try to get the running event loop (works in async context).
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self._queue.put_nowait, data)
        except RuntimeError:
            # No running loop (pure sync context); drop silently.
            # For a fully sync CLI run SSESink is effectively a no-op.
            pass

    async def stream(self, run_id: Optional[str] = None) -> AsyncIterator[str]:
        """Yield SSE-formatted lines, optionally filtered to one run_id."""
        while True:
            data = await self._queue.get()
            if run_id is not None:
                try:
                    obj = json.loads(data)
                    if obj.get("run_id") != run_id:
                        continue
                except Exception:
                    pass
            yield f"data: {data}\n\n"


def make_sse_router(sink: SSESink):
    """
    Return a FastAPI ``APIRouter`` with a single endpoint:

        GET /trace/stream?run_id=<optional>

    Streams Server-Sent Events.  Add it to your FastAPI app::

        app.include_router(make_sse_router(sink))
    """
    try:
        from fastapi import APIRouter  # type: ignore
        from fastapi.responses import StreamingResponse  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "fastapi is required for SSE support. "
            "It is already in requirements.txt — make sure it is installed."
        ) from exc

    router = APIRouter()

    @router.get("/trace/stream")
    async def stream_trace(run_id: Optional[str] = None):
        return StreamingResponse(
            sink.stream(run_id=run_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return router
