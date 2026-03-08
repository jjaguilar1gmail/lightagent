from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, Optional

from langchain_core.callbacks.manager import CallbackManager

from .emitter import emit
from .callbacks import TracingCallbackHandler


def _state_snapshot(state: Any, result: Any = None) -> dict[str, Any]:
    base = dict(state or {}) if isinstance(state, dict) else {}
    if isinstance(result, dict):
        base.update(result)
    plan = base.get("plan", [])
    return {
        "step": base.get("step"),
        "done": base.get("done"),
        "final_answer": (base.get("final_answer") or "")[:160],
        "termination_reason": (base.get("termination_reason") or "")[:120],
        "parse_errors": base.get("parse_errors"),
        "plan": plan,
        "plan_len": len(plan),
        "user_goal": (base.get("user_goal") or "")[:120],
    }


def traced_node(fn: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator for LangGraph node functions.

    Automatically emits node_start / node_end / node_error and injects a
    TracingCallbackHandler into the RunnableConfig so LLM calls inside the
    node emit llm_call_start / llm_call_end without any per-node boilerplate.

    Works on both ``fn(state, config)`` and ``fn(state)`` signatures.
    """
    if fn is None:
        # Called as @traced_node(name="custom")
        return functools.partial(traced_node, name=name)

    raw_name = name or fn.__name__
    # Strip _node suffix so decorator names match graph-registered names
    # (planner_node -> planner, agent_node -> agent, etc.)
    node_name = raw_name.removesuffix("_node")
    _has_config = "config" in inspect.signature(fn).parameters

    @functools.wraps(fn)
    def wrapper(state, config=None):
        # --- inject tracing callback into config ---
        if _has_config:
            if config is None:
                config = {}
            handler = TracingCallbackHandler(node=node_name)
            callbacks = config.get("callbacks")
            if isinstance(callbacks, CallbackManager):
                # LangGraph already built a CallbackManager for this node;
                # add our handler directly — it owns this instance.
                callbacks.add_handler(handler, inherit=True)
            else:
                existing = list(callbacks or [])
                existing.append(handler)
                config = {**config, "callbacks": existing}

        # --- build concise snapshots of relevant state fields ---
        start_payload = _state_snapshot(state)

        emit("node_start", node=node_name, payload=start_payload)
        start = time.monotonic()
        try:
            result = fn(state, config) if _has_config else fn(state)
            elapsed = int((time.monotonic() - start) * 1000)
            end_payload = _state_snapshot(state, result=result)
            emit("node_end", node=node_name, payload={**end_payload, "elapsed_ms": elapsed})
            return result
        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            error_payload = _state_snapshot(state)
            emit(
                "node_error",
                node=node_name,
                payload={**error_payload, "elapsed_ms": elapsed, "error": str(exc)},
            )
            raise

    return wrapper


def traced_router(fn: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator for LangGraph routing functions.
    Emits a routing_decision event with the chosen destination.
    """
    if fn is None:
        return functools.partial(traced_router, name=name)

    router_name = name or fn.__name__

    @functools.wraps(fn)
    def wrapper(state):
        destination = fn(state)
        emit(
            "routing_decision",
            node=router_name,
            payload={"destination": destination},
        )
        return destination

    return wrapper


class _TracedToolWrapper:
    """
    Thin wrapper around a LangChain StructuredTool that intercepts ``invoke``
    to emit tracing events without touching the underlying Pydantic model.

    All other attribute accesses (including ``get_input_schema``, ``name``,
    ``description``, etc.) are delegated to the original tool so that
    ``llm.bind_tools(TOOLS)`` and schema generation continue to work normally.
    """

    def __init__(self, tool_obj: Any) -> None:
        # Store without going through our own __setattr__ / __getattr__
        object.__setattr__(self, "_tool", tool_obj)
        object.__setattr__(self, "name", tool_obj.name)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        tool_obj = object.__getattribute__(self, "_tool")
        args = input if isinstance(input, dict) else {"input": str(input)}
        emit(
            "tool_call_start",
            node=tool_obj.name,
            payload={"tool": tool_obj.name, "args": args},
        )
        start = time.monotonic()
        try:
            result = (
                tool_obj.invoke(input, config, **kwargs)
                if config is not None
                else tool_obj.invoke(input, **kwargs)
            )
            elapsed = int((time.monotonic() - start) * 1000)
            emit(
                "tool_call_end",
                node=tool_obj.name,
                payload={
                    "tool": tool_obj.name,
                    "result": str(result)[:500],
                    "elapsed_ms": elapsed,
                },
            )
            return result
        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            emit(
                "tool_error",
                node=tool_obj.name,
                payload={
                    "tool": tool_obj.name,
                    "error": str(exc),
                    "elapsed_ms": elapsed,
                },
            )
            raise

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (schema, description, metadata…) to the
        # original so bind_tools() and tool schema generation work normally.
        return getattr(object.__getattribute__(self, "_tool"), name)

    def __repr__(self) -> str:
        return f"TracedTool({object.__getattribute__(self, '_tool')!r})"


def traced_tool(tool_obj: Any) -> _TracedToolWrapper:
    """
    Wrap a LangChain @tool instance so every invoke emits
    tool_call_start / tool_call_end / tool_error.

    Returns a lightweight wrapper; the original tool is never mutated.
    Use inline: ``TOOLS = [traced_tool(calc), traced_tool(now_utc)]``
    """
    return _TracedToolWrapper(tool_obj)
