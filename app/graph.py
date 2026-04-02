from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .agent_policy import PolicyContext, evaluate_completion, tool_outcome_guidance
from .prompts import BASE_SYSTEM_PROMPT
from .tools import ALL_TOOLS, FINAL_ANSWER_TOOL_NAME
from .tracing import traced_node, traced_router, traced_tool
import os

load_dotenv()


class AgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    user_goal: str
    scratch: Dict[str, Any]
    step: int
    max_steps: int
    done: bool
    final_answer: str
    termination_reason: str


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TOOLS = list(ALL_TOOLS)
TRACED_TOOL_MAP = {
    tool.name: traced_tool(tool)
    for tool in TOOLS
    if tool.name != FINAL_ANSWER_TOOL_NAME
}

model_name = "meta-llama/llama-3.3-70b-instruct"
llm = ChatOpenAI(
    model=model_name,
    temperature=0.2,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
).bind_tools(TOOLS)


def _ensure_defaults(state: AgentState) -> AgentState:
    state.setdefault("step", 0)
    state.setdefault("max_steps", 6)
    state.setdefault("done", False)
    state.setdefault("final_answer", "")
    state.setdefault("termination_reason", "")
    state.setdefault("scratch", {})
    return state


def _user_goal(state: AgentState) -> str:
    goal = str(state.get("user_goal", "")).strip()
    if goal:
        return goal

    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            goal = str(msg.content).strip()
            if goal:
                state["user_goal"] = goal
                return goal

    return ""


@traced_node
def agent_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    state = _ensure_defaults(state)
    current_step = int(state.get("step", 0)) + 1

    if current_step > int(state.get("max_steps", 0)):
        fallback = "I reached the step budget before producing a reliable final answer."
        return {
            "step": current_step,
            "done": True,
            "termination_reason": "max_steps",
            "final_answer": fallback,
            "messages": [AIMessage(content=fallback)],
        }

    msgs: List[AnyMessage] = []
    if not state.get("messages") or not isinstance(state["messages"][0], SystemMessage):
        msgs.append(SystemMessage(content=BASE_SYSTEM_PROMPT))

    goal = _user_goal(state)
    if goal:
        msgs.append(
            SystemMessage(
                content=(
                    "Answer the user's question directly. Use tools only when they help materially. "
                    "If you already have a complete answer, call final_answer instead of continuing to deliberate."
                )
            )
        )

    policy_context = PolicyContext(messages=state.get("messages", []))
    tool_guidance = tool_outcome_guidance(policy_context)
    if tool_guidance:
        msgs.append(SystemMessage(content=tool_guidance))

    msgs.extend(state.get("messages", []))
    resp = llm.invoke(msgs, config=config)
    tool_calls = getattr(resp, "tool_calls", None) or []

    for call in tool_calls:
        if call.get("name") != FINAL_ANSWER_TOOL_NAME:
            continue

        answer = str((call.get("args") or {}).get("answer", "")).strip()
        if not answer:
            return {
                "messages": [
                    resp,
                    ToolMessage(
                        content="ERROR: final_answer requires a non-empty answer.",
                        tool_call_id=call.get("id", "unknown"),
                    ),
                ],
                "done": False,
                "step": current_step,
            }

        decision = evaluate_completion(policy_context, answer)
        if not decision.allowed:
            return {
                "messages": [
                    resp,
                    ToolMessage(
                        content=f"ERROR: {decision.feedback}",
                        tool_call_id=call.get("id", "unknown"),
                    ),
                ],
                "done": False,
                "step": current_step,
            }

        return {
            "messages": [resp],
            "done": True,
            "step": current_step,
            "final_answer": answer,
            "termination_reason": "completed",
        }

    if tool_calls:
        return {"messages": [resp], "done": False, "step": current_step}

    answer = str(resp.content).strip()
    if answer:
        decision = evaluate_completion(policy_context, answer)
        if not decision.allowed:
            tool_call_id = "completion_policy"
            if decision.feedback and "no supporting tool result" in decision.feedback:
                tool_call_id = "unsupported_termination"
            elif decision.feedback and "Do not give up before attempting an action" in decision.feedback:
                tool_call_id = "must_act_before_giving_up"
            return {
                "messages": [
                    resp,
                    ToolMessage(
                        content=f"ERROR: {decision.feedback}",
                        tool_call_id=tool_call_id,
                    ),
                ],
                "done": False,
                "step": current_step,
            }

        return {
            "messages": [resp],
            "done": True,
            "step": current_step,
            "final_answer": answer,
            "termination_reason": "completed",
        }

    fallback = "I could not produce a useful answer from the available tools and context."
    return {
        "messages": [resp, AIMessage(content=fallback)],
        "done": True,
        "step": current_step,
        "final_answer": fallback,
        "termination_reason": "empty_response",
    }


@traced_node(name="tools")
def tool_node(state: AgentState) -> Dict[str, Any]:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return {"messages": []}

    tool_messages: List[ToolMessage] = []
    for call in getattr(last, "tool_calls", []) or []:
        name = call.get("name")
        if name == FINAL_ANSWER_TOOL_NAME:
            continue

        tool = TRACED_TOOL_MAP.get(name)
        if tool is None:
            tool_messages.append(
                ToolMessage(
                    content=f"ERROR: unknown tool '{name}'",
                    tool_call_id=call.get("id", "unknown"),
                )
            )
            continue

        try:
            result = tool.invoke(call.get("args") or {})
        except Exception as exc:
            result = f"ERROR: {type(exc).__name__}: {exc}"

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call.get("id", "unknown"),
            )
        )

    return {"messages": tool_messages}


@traced_router
def route_after_agent(state: AgentState) -> Literal["tools", "agent", "end"]:
    if state.get("done") or state.get("final_answer"):
        return "end"

    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        return "agent"
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "agent": "agent", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()