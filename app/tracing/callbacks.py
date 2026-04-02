from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .emitter import emit


class TracingCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback that bridges LLM lifecycle events into the trace stream.
    Attach it via RunnableConfig — the @traced_node decorator does this automatically.

    Args:
        node:      Name of the owning node (used as the event's node field).
        max_chars: Maximum characters to store for prompt/response content.
                   None (default) stores the full content.
    """

    # Inherit raise_error=False (default) so LLM errors propagate normally.
    raise_error = False

    def __init__(self, node: Optional[str] = None, max_chars: Optional[int] = None):
        super().__init__()
        self.node = node
        self.max_chars = max_chars
        self._starts: Dict[str, float] = {}  # run_id → monotonic start time

    def _clip(self, text: str) -> str:
        if self.max_chars is None or len(text) <= self.max_chars:
            return text
        return text[: self.max_chars] + f"…[+{len(text) - self.max_chars} chars]"

    def _clip_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._clip(value)
        if isinstance(value, list):
            return [self._clip_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._clip_value(item) for key, item in value.items()}
        return value

    def _serialize_tool_calls(self, generation: Any) -> List[Dict[str, Any]]:
        message = getattr(generation, "message", None)
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        tool_calls: List[Dict[str, Any]] = []
        for call in raw_tool_calls:
            if not isinstance(call, dict):
                continue
            tool_calls.append(
                {
                    "name": call.get("name"),
                    "args": self._clip_value(call.get("args") or {}),
                    "id": call.get("id"),
                    "type": call.get("type"),
                }
            )
        return tool_calls

    # ------------------------------------------------------------------
    # LLM events
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id,
        **kwargs: Any,
    ) -> None:
        self._starts[str(run_id)] = time.monotonic()
        model = (
            serialized.get("kwargs", {}).get("model_name")
            or serialized.get("kwargs", {}).get("model")
            or serialized.get("name", "?")
        )
        emit(
            "llm_call_start",
            node=self.node,
            payload={
                "model": model,
                "prompt": self._clip(prompts[0] if prompts else ""),
            },
        )

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs: Any) -> None:
        start = self._starts.pop(str(run_id), None)
        elapsed = int((time.monotonic() - start) * 1000) if start is not None else None

        preview = ""
        tool_calls: List[Dict[str, Any]] = []
        if response.generations:
            gen0 = response.generations[0]
            if gen0:
                preview = getattr(gen0[0], "text", "") or ""
                tool_calls = self._serialize_tool_calls(gen0[0])

        emit(
            "llm_call_end",
            node=self.node,
            payload={
                "elapsed_ms": elapsed,
                "response": self._clip(preview),
                "response_text": self._clip(preview),
                "tool_calls": tool_calls,
            },
        )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id,
        **kwargs: Any,
    ) -> None:
        start = self._starts.pop(str(run_id), None)
        elapsed = int((time.monotonic() - start) * 1000) if start is not None else None
        # Emit llm_call_end with error flag; node_error is emitted by the
        # @traced_node decorator so we don't duplicate it here.
        emit(
            "llm_call_end",
            node=self.node,
            payload={"elapsed_ms": elapsed, "error": str(error)},
        )
