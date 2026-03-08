from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from pydantic import BaseModel

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

from .tools import calc, final_answer, now_utc
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


# ----------------------------
# 2) LLM + Tools
# ----------------------------

# TODO: Load your OpenRouter API key securely.
# For example, from a .env file or environment variables.
# import os
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Tools visible to the LLM for schema generation.
# final_answer is included so the model knows to call it for terminal responses.
TOOLS = [calc, now_utc, final_answer]
# Tools actually executed at runtime.
# final_answer is intentionally excluded — agent_node intercepts it before tool_node.
TRACED_TOOLS = [traced_tool(t) for t in [calc, now_utc]]

SYSTEM_PROMPT = """You are a helpful agent in a LangGraph demo.
You MUST:
- Be concise.
- Use tools when helpful.
- If you need math or current time, call a tool.
- When you have a complete answer ready, call the final_answer tool.
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
    state.setdefault("scratch", {})
    state.setdefault("plan", [])
    return state


class _ReflectDecision(BaseModel):
    # Structured output schema for the reflect node's binary done-check.
    done: bool


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

    # Minimal robust parse (don't overengineer starter)
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
        fallback = "I reached the step budget and could not complete the task."
        return {
            "step": current_step,
            "done": True,
            "termination_reason": "max_steps",
            "final_answer": fallback,
            "messages": [AIMessage(content=fallback)],
        }

    msgs: List[AnyMessage] = []
    if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
        msgs.append(SystemMessage(content=SYSTEM_PROMPT))

    # Add a lightweight "plan context" message so the agent acts consistently.
    if state.get("plan"):
        msgs.append(SystemMessage(content="Current plan:\n- " + "\n- ".join(state["plan"])))

    msgs.extend(state["messages"])
    resp = llm.invoke(msgs, config=config)

    tool_calls = getattr(resp, "tool_calls", None) or []

    # Intercept final_answer tool call: extract answer, mark terminal, never execute.
    for call in tool_calls:
        if call.get("name") == "final_answer":
            answer = str((call.get("args") or {}).get("answer", "")).strip()
            if answer:
                return {
                    "messages": [resp],
                    "done": True,
                    "step": current_step,
                    "final_answer": answer,
                    "termination_reason": "completed",
                }

    # Repeated-tool-call guard: if every proposed call duplicates a prior call
    # (same tool name + identical args), we're stuck in a loop — terminate early.
    if tool_calls:
        seen = {
            (c.get("name"), json.dumps(c.get("args") or {}, sort_keys=True))
            for msg in state["messages"]
            if isinstance(msg, AIMessage)
            for c in (getattr(msg, "tool_calls", None) or [])
        }
        novel = [
            c for c in tool_calls
            if (c.get("name"), json.dumps(c.get("args") or {}, sort_keys=True)) not in seen
        ]
        if not novel:
            fallback = "I appear to be stuck repeating the same tool calls. Stopping to avoid a loop."
            return {
                "messages": [resp],
                "done": True,
                "step": current_step,
                "termination_reason": "repeated_tool_call",
                "final_answer": fallback,
            }

    # Any other tool calls go to tool_node for execution.
    if tool_calls:
        return {"messages": [resp], "done": False, "step": current_step}

    # No tool calls and no final_answer — send to reflect to decide next.
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
    Binary "are we done?" check. done=True signals the agent to call final_answer next turn.
    """
    state = _ensure_defaults(state)

    # Fast-path: terminal state already set by agent_node.
    if state.get("done") or state.get("final_answer"):
        state["done"] = True
        if not state.get("termination_reason"):
            state["termination_reason"] = "completed"
        return state

    # Hard stop if too many steps.
    if state["step"] >= state["max_steps"]:
        state["done"] = True
        state["termination_reason"] = "max_steps"
        return state

    # Structured-output binary check: reliable, no text parsing needed.
    check_prompt = [
        SystemMessage(
            content=(
                "Does the conversation below contain enough information to give a complete final answer? "
                "Respond with done=true or done=false."
            )
        ),
    ]
    # Use the last few messages as context.
    check_prompt.extend(state["messages"][-8:])

    checker = ChatOpenAI(
        model=model_name,
        temperature=0.0,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    ).with_structured_output(_ReflectDecision)

    try:
        result = checker.invoke(check_prompt, config=config)
        state["done"] = result.done
    except Exception:
        # If structured output fails, keep going rather than crashing.
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