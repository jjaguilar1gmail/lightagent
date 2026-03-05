from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from dotenv import load_dotenv

from .tools import calc, echo_json, now_utc
from .tracing import traced_node, traced_router, traced_tool
import json
import os

load_dotenv()

# ----------------------------
# 1) State definition
# ----------------------------

class AgentState(TypedDict, total=False):
    # Automatically append new messages
    messages: Annotated[List[AnyMessage], add_messages]

    # A place to put structured stuff your app needs
    user_goal: str
    plan: List[str]
    scratch: Dict[str, Any]

    # Control / safety
    step: int
    max_steps: int
    done: bool
    # Non-empty only when we have a user-safe terminal answer.
    final_answer: str
    # Useful for traces/debugging why a run ended.
    termination_reason: str
    # Bounded parser-failure counter for agent/reflect strict-JSON paths.
    parse_errors: int


# ----------------------------
# 2) LLM + Tools
# ----------------------------

# TODO: Load your OpenRouter API key securely.
# For example, from a .env file or environment variables.
# import os
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Originals used for bind_tools (schema generation) — must stay as StructuredTool instances.
TOOLS = [calc, now_utc, echo_json]
# Wrapped versions used at execution time — emit tracing events around every call.
TRACED_TOOLS = [traced_tool(t) for t in TOOLS]

SYSTEM_PROMPT = """You are a helpful agent in a LangGraph demo.
You MUST:
- Be concise.
- Use tools when helpful.
- If you need math or current time, call a tool.
- If no tool is needed, respond as JSON: {"done": true|false, "final_answer": string}.
"""

model_name = "meta-llama/llama-3.3-70b-instruct"#"openai/gpt-oss-120b:free"  # Or any other model on OpenRouter
llm = ChatOpenAI(
    model=model_name,  # Or any other model on OpenRouter
    temperature=0.2,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
).bind_tools(TOOLS)


# ----------------------------
# 3) Nodes
# ----------------------------

def _ensure_defaults(state: AgentState) -> AgentState:
    # Keep state initialization centralized so every node sees the same defaults.
    state.setdefault("step", 0)
    state.setdefault("max_steps", 8)
    state.setdefault("done", False)
    state.setdefault("final_answer", "")
    state.setdefault("termination_reason", "")
    state.setdefault("parse_errors", 0)
    state.setdefault("scratch", {})
    state.setdefault("plan", [])
    return state


def _parse_json_object(content: Any) -> Optional[Dict[str, Any]]:
    # Best-effort parser for model output:
    # 1) parse plain JSON object
    # 2) parse fenced JSON block
    # 3) parse first object-shaped substring
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None

    text = content.strip()
    if not text:
        return None

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    # Try full parse first.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Fallback: first balanced-ish JSON object substring.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _retry_agent_json(msgs: List[AnyMessage], config: RunnableConfig) -> Optional[Dict[str, Any]]:
    # Single repair attempt that reuses full conversation context and asks for
    # strict terminal JSON. This avoids brittle repairs based on empty strings.
    repair_prompt = list(msgs)
    repair_prompt.append(
        SystemMessage(
            content=(
                "Previous output was invalid or empty. "
                "Return ONLY valid JSON with keys: done (boolean), final_answer (string). "
                "No markdown and no tool calls."
            )
        )
    )
    repair_model = ChatOpenAI(
        model=model_name,
        temperature=0.0,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    repaired = repair_model.invoke(repair_prompt, config=config)
    return _parse_json_object(repaired.content)


def _has_user_facing_answer(state: AgentState) -> bool:
    # Guard against ending with only tool calls / empty content.
    # A "real" answer is an AI message with non-empty text and no pending tool calls.
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage):
            content = str(getattr(msg, "content", "") or "").strip()
            has_tool_calls = bool(getattr(msg, "tool_calls", None))
            if content and not has_tool_calls:
                return True
    return False


@traced_node
def planner_node(state: AgentState, config: RunnableConfig) -> AgentState:
    state = _ensure_defaults(state)

    # Only plan once (or re-plan if you want)
    if state["plan"]:
        return state

    goal = state.get("user_goal", "").strip()
    if not goal:
        # Derive goal from last user message if not provided
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                goal = msg.content
                break
        state["user_goal"] = goal

    prompt = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Create a short step-by-step plan (3-6 bullets) to solve the user's goal.\n"
                f"Goal: {goal}\n"
                "Respond as a JSON object with key 'plan' = list of strings."
            )
        ),
    ]
    resp = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    ).invoke(prompt, config=config)

    # Minimal robust parse (don’t overengineer starter)

    try:
        obj = json.loads(resp.content)
        plan = obj.get("plan", [])
        if isinstance(plan, list):
            state["plan"] = [str(x) for x in plan][:8]
    except Exception:
        state["plan"] = ["Think step-by-step", "Use tools as needed", "Answer concisely"]
    return state


@traced_node
def agent_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    state = _ensure_defaults(state)
    # Step is persisted through returned state updates (not in-place mutation).
    current_step = int(state.get("step", 0)) + 1

    # Deterministic hard stop to prevent unbounded loops.
    if current_step >= state["max_steps"]:
        return {
            "step": current_step,
            "done": True,
            "termination_reason": "max_steps",
            "messages": [AIMessage(content="I’m stopping because the step budget was reached.")],
        }

    msgs: List[AnyMessage] = []
    if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
        msgs.append(SystemMessage(content=SYSTEM_PROMPT))

    # Add a lightweight “plan context” message so the agent acts consistently
    if state.get("plan"):
        msgs.append(SystemMessage(content=f"Current plan:\n- " + "\n- ".join(state["plan"])))

    msgs.append(
        SystemMessage(
            content=(
                "If you need a tool, call it. "
                "If no tool is needed, respond ONLY as JSON with keys done (boolean) and final_answer (string)."
            )
        )
    )

    msgs.extend(state["messages"])
    # show response in agent node for debugging
    resp = llm.invoke(msgs, config=config)
    # Tool calls are executed first; terminal JSON is handled only on non-tool turns.
    if getattr(resp, "tool_calls", None):
        return {"messages": [resp], "done": False, "step": current_step}

    parsed = _parse_json_object(resp.content)
    # Retry once with strict instruction only when model emitted non-empty invalid text.
    if parsed is None and str(resp.content or "").strip():
        parsed = _retry_agent_json(msgs, config)

    if parsed is None:
        # Bounded parse-error budget: stop cleanly with a fallback answer.
        parse_errors = int(state.get("parse_errors", 0)) + 1
        updates: Dict[str, Any] = {
            "messages": [resp],
            "done": False,
            "step": current_step,
            "parse_errors": parse_errors,
        }
        if parse_errors >= 2:
            updates["done"] = True
            updates["termination_reason"] = "agent_parse_error_budget"
            fallback = "I’m unable to format a reliable final response right now. Please try rephrasing your request."
            updates["final_answer"] = fallback
            updates["messages"] = [AIMessage(content=fallback)]
        return updates

    done = bool(parsed.get("done", False))
    final_answer = str(parsed.get("final_answer", "")).strip()

    # Valid terminal packet: convert to a normal AI message for downstream UX.
    if done and final_answer:
        return {
            "messages": [AIMessage(content=final_answer)],
            "done": True,
            "step": current_step,
            "final_answer": final_answer,
            "termination_reason": "completed",
            "parse_errors": 0,
        }

    if done and not final_answer:
        # "done" without answer is treated as malformed terminal output.
        parse_errors = int(state.get("parse_errors", 0)) + 1
        updates = {
            "messages": [resp],
            "done": False,
            "step": current_step,
            "parse_errors": parse_errors,
        }
        if parse_errors >= 2:
            updates["done"] = True
            updates["termination_reason"] = "agent_done_without_answer"
            fallback = "I reached a completion state but do not have a valid final answer to return."
            updates["final_answer"] = fallback
            updates["messages"] = [AIMessage(content=fallback)]
        return updates

    return {"messages": [resp], "step": current_step, "done": False}


@traced_node(name="tools")
def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute any pending tool calls from the last AI message and return ToolMessages.
    """
    from langchain_core.messages import AIMessage

    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return {"messages": []}

    tool_messages: List[ToolMessage] = []
    tool_map = {t.name: t for t in TRACED_TOOLS}

    for call in getattr(last, "tool_calls", []) or []:
        name = call.get("name")
        args = call.get("args") or {}
        tool = tool_map.get(name)
        if tool is None:
            tool_messages.append(
                ToolMessage(
                    content=f"ERROR: unknown tool '{name}'",
                    tool_call_id=call.get("id", "unknown"),
                )
            )
            continue

        try:
            result = tool.invoke(args)
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call.get("id", "unknown"),
                )
            )
        except Exception as e:
            tool_messages.append(
                ToolMessage(
                    content=f"ERROR: {type(e).__name__}: {e}",
                    tool_call_id=call.get("id", "unknown"),
                )
            )

    return {"messages": tool_messages}

@traced_node
def reflect_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Optional: a cheap “are we done?” pass that sets done flag.
    You can delete this node later if you want.
    """
    state = _ensure_defaults(state)

    # Fast-path: once final answer exists, reflection should never reopen the loop.
    if state.get("done") or state.get("final_answer"):
        state["done"] = True
        if not state.get("termination_reason"):
            state["termination_reason"] = "completed"
        return state

    # Hard stop if too many steps
    if state["step"] >= state["max_steps"]:
        state["done"] = True
        state["termination_reason"] = "max_steps"
        return state

    # Ask a strict binary checker whether we can finish now.
    check_prompt = [
        SystemMessage(
            content=(
                "Return ONLY JSON with a single key: done (boolean). "
                "No markdown, no prose, no tool calls."
            )
        ),
    ]
    # Use the last few messages as context
    tail = state["messages"][-8:]
    check_prompt.extend(tail)

    checker = ChatOpenAI(
        model=model_name,
        temperature=0.0,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    resp = checker.invoke(check_prompt, config=config)
    obj = _parse_json_object(resp.content)
    if obj is None:
        # One bounded retry for invalid checker output.
        retry_prompt = list(check_prompt)
        retry_prompt.append(SystemMessage(content="Previous output was invalid. Output strict JSON only."))
        retry = checker.invoke(retry_prompt, config=config)
        obj = _parse_json_object(retry.content)

    if obj is not None and "done" in obj:
        proposed_done = bool(obj.get("done", False))
        # Never allow reflect to terminate unless there's an actual user-facing answer.
        if proposed_done and not (state.get("final_answer") or _has_user_facing_answer(state)):
            state["done"] = False
        else:
            state["done"] = proposed_done
    else:
        state["parse_errors"] = int(state.get("parse_errors", 0)) + 1
        if state["parse_errors"] >= 2:
            state["done"] = True
            state["termination_reason"] = "reflect_parse_error_budget"
        else:
            state["done"] = False

    return state


# ----------------------------
# 4) Routing
# ----------------------------

@traced_router
def route_after_agent(state: AgentState) -> Literal["tools", "reflect", "end"]:
    from langchain_core.messages import AIMessage

    # Termination state always wins over tool routing.
    if state.get("done") or state.get("final_answer"):
        return "end"

    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "reflect"


@traced_router
def route_after_reflect(state: AgentState) -> Literal["agent", "end"]:
    # Reflect can only end when terminal state has been materialized.
    return "end" if state.get("done") or state.get("final_answer") else "agent"


# ----------------------------
# 5) Build graph
# ----------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("reflect", reflect_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "agent")

    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "reflect": "reflect", "end": END})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("reflect", route_after_reflect, {"agent": "agent", "end": END})

    return g.compile()