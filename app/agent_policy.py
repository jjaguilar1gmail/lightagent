from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage


@dataclass(frozen=True)
class PolicyContext:
    messages: Sequence[AnyMessage]


@dataclass(frozen=True)
class TurnFacts:
    has_messages: bool
    has_tool_observation: bool
    has_tool_error: bool
    has_real_tool_attempt: bool
    has_final_answer_attempt: bool
    last_message_is_tool_error: bool
    last_message_is_tool_success: bool


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    feedback: str | None = None


_UNSUPPORTED_SUPPORT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bI (?:calculated|compute[ds]?|estimated|converted|verified|checked|looked up)\b",
        r"\bwe (?:calculated|computed|estimated|converted|verified|checked)\b",
        r"\blet[’']?s (?:calculate|compute|estimate|convert|verify|check|do)\b",
        r"\b(?:calculating|computed?|estimating|estimated|converting|converted|verifying|verified|checking|checked)\b",
        r"\b(?:calculation|estimate|estimation|conversion|verification|lookup)\b",
        r"\b(?:dividing|multiplying|adding|subtracting)\b",
    ]
]

_EVADE_OR_GIVE_UP_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bunknown\b",
        r"\bI (?:don'?t know|do not know|can'?t determine|cannot determine|can'?t tell|cannot tell)\b",
        r"\b(?:not enough|insufficient) information\b",
        r"\bdepends on several factors\b",
        r"\bunable to determine\b",
        r"\bcannot be determined\b",
    ]
]


def _message_tool_calls(message: AnyMessage) -> list[dict[str, Any]]:
    tool_calls = getattr(message, "tool_calls", None) or []
    return [call for call in tool_calls if isinstance(call, dict)]


def _is_real_tool_call(call: dict[str, Any]) -> bool:
    return str(call.get("name") or "").strip() not in {"", "final_answer"}


def _has_tool_observation(messages: Sequence[AnyMessage]) -> bool:
    return any(
        isinstance(message, ToolMessage) and not str(message.content).startswith("ERROR:")
        for message in messages
    )


def _has_tool_error(messages: Sequence[AnyMessage]) -> bool:
    return any(
        isinstance(message, ToolMessage) and str(message.content).startswith("ERROR:")
        for message in messages
    )


def _has_real_tool_attempt(messages: Sequence[AnyMessage]) -> bool:
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        if any(_is_real_tool_call(call) for call in _message_tool_calls(message)):
            return True
    return False


def _has_final_answer_attempt(messages: Sequence[AnyMessage]) -> bool:
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        if any(str(call.get("name") or "").strip() == "final_answer" for call in _message_tool_calls(message)):
            return True
    return False


def derive_turn_facts(context: PolicyContext) -> TurnFacts:
    messages = context.messages
    last = messages[-1] if messages else None
    return TurnFacts(
        has_messages=bool(messages),
        has_tool_observation=_has_tool_observation(messages),
        has_tool_error=_has_tool_error(messages),
        has_real_tool_attempt=_has_real_tool_attempt(messages),
        has_final_answer_attempt=_has_final_answer_attempt(messages),
        last_message_is_tool_error=isinstance(last, ToolMessage) and str(last.content).startswith("ERROR:"),
        last_message_is_tool_success=isinstance(last, ToolMessage) and not str(last.content).startswith("ERROR:"),
    )


def _unsupported_completion_feedback(facts: TurnFacts, answer: str) -> str | None:
    if facts.has_tool_observation:
        return None

    text = str(answer or "").strip()
    if not text:
        return None

    for pattern in _UNSUPPORTED_SUPPORT_PATTERNS:
        if pattern.search(text):
            return (
                "Your answer describes a calculation, estimate, lookup, verification, or other support-dependent step, "
                "but there is no supporting tool result in the conversation. Either call an appropriate tool first or "
                "revise the answer so it only states what is already supported."
            )

    return None


def _must_act_before_giving_up_feedback(facts: TurnFacts, answer: str) -> str | None:
    if facts.has_real_tool_attempt:
        return None

    text = str(answer or "").strip()
    if not text:
        return None

    for pattern in _EVADE_OR_GIVE_UP_PATTERNS:
        if pattern.search(text):
            return (
                "Do not give up before attempting an action. If an available tool could materially reduce uncertainty, call it now. "
                "Only conclude that the answer is unknown or underdetermined after you have made a reasonable tool attempt or can explain why no available tool can help."
            )

    return None


def evaluate_completion(context: PolicyContext, answer: str) -> PolicyDecision:
    facts = derive_turn_facts(context)

    unsupported_feedback = _unsupported_completion_feedback(facts, answer)
    if unsupported_feedback:
        return PolicyDecision(allowed=False, feedback=unsupported_feedback)

    must_act_feedback = _must_act_before_giving_up_feedback(facts, answer)
    if must_act_feedback:
        return PolicyDecision(allowed=False, feedback=must_act_feedback)

    return PolicyDecision(allowed=True)


def tool_outcome_guidance(context: PolicyContext) -> str | None:
    facts = derive_turn_facts(context)
    if not facts.has_messages:
        return None

    if facts.last_message_is_tool_error:
        return (
            "The last tool call failed. Do not ignore the failure. Either fix the tool input, choose a different tool, "
            "or revise your answer so it stays within what is already supported."
        )

    if facts.last_message_is_tool_success:
        return (
            "A tool result is now available in the conversation. Use that observation directly if it is sufficient, or call another tool only if more support is still needed. "
            "Do not restate an intention to act when you already have an observation to use."
        )

    return None