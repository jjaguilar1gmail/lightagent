from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .tools import calc, echo_json, now_utc


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


# ----------------------------
# 2) LLM + Tools
# ----------------------------

# TODO: Load your OpenRouter API key securely.
# For example, from a .env file or environment variables.
# import os
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"  # <-- PASTE YOUR KEY HERE

TOOLS = [calc, now_utc, echo_json]

SYSTEM_PROMPT = """You are a helpful agent in a LangGraph demo.
You MUST:
- Be concise.
- Use tools when helpful.
- If you need math or current time, call a tool.
- If you have enough info, set done=true in your final response.
"""

llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",  # Or any other model on OpenRouter
    temperature=0.2,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
).bind_tools(TOOLS)


# ----------------------------
# 3) Nodes
# ----------------------------

def _ensure_defaults(state: AgentState) -> AgentState:
    state.setdefault("step", 0)
    state.setdefault("max_steps", 8)
    state.setdefault("done", False)
    state.setdefault("scratch", {})
    state.setdefault("plan", [])
    return state


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
        model="openai/gpt-oss-120b:free",
        temperature=0.2,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    ).invoke(prompt, config=config)

    # Minimal robust parse (don’t overengineer starter)
    import json
    try:
        obj = json.loads(resp.content)
        plan = obj.get("plan", [])
        if isinstance(plan, list):
            state["plan"] = [str(x) for x in plan][:8]
    except Exception:
        state["plan"] = ["Think step-by-step", "Use tools as needed", "Answer concisely"]

    return state


def agent_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    state = _ensure_defaults(state)
    state["step"] += 1

    msgs: List[AnyMessage] = []
    if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
        msgs.append(SystemMessage(content=SYSTEM_PROMPT))

    # Add a lightweight “plan context” message so the agent acts consistently
    if state.get("plan"):
        msgs.append(SystemMessage(content=f"Current plan:\n- " + "\n- ".join(state["plan"])))

    msgs.extend(state["messages"])

    resp = llm.invoke(msgs, config=config)

    # If the model called tools, we route to tool execution next.
    return {"messages": [resp]}


def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute any pending tool calls from the last AI message and return ToolMessages.
    """
    from langchain_core.messages import AIMessage

    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return {"messages": []}

    tool_messages: List[ToolMessage] = []
    tool_map = {t.name: t for t in TOOLS}

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


def reflect_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Optional: a cheap “are we done?” pass that sets done flag.
    You can delete this node later if you want.
    """
    state = _ensure_defaults(state)

    # Hard stop if too many steps
    if state["step"] >= state["max_steps"]:
        state["done"] = True
        return state

    # Ask a small model: do we have enough to answer?
    check_prompt = [
        SystemMessage(content="Return JSON: {\"done\": true|false}. done=true if a final answer can be given now."),
    ]
    # Use the last few messages as context
    tail = state["messages"][-8:]
    check_prompt.extend(tail)

    checker = ChatOpenAI(
        model="openai/gpt-oss-120b:free",
        temperature=0.0,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    resp = checker.invoke(check_prompt, config=config)

    import json
    try:
        obj = json.loads(resp.content)
        state["done"] = bool(obj.get("done", False))
    except Exception:
        # Default to not done unless we’re near budget
        state["done"] = state["step"] >= state["max_steps"] - 1

    return state


# ----------------------------
# 4) Routing
# ----------------------------

def route_after_agent(state: AgentState) -> Literal["tools", "reflect"]:
    from langchain_core.messages import AIMessage

    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "reflect"


def route_after_reflect(state: AgentState) -> Literal["agent", "end"]:
    return "end" if state.get("done") else "agent"


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

    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "reflect": "reflect"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("reflect", route_after_reflect, {"agent": "agent", "end": END})

    return g.compile()