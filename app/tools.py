from __future__ import annotations

import math
from datetime import datetime, timezone

from langchain_core.tools import tool


FINAL_ANSWER_TOOL_NAME = "final_answer"


def _normalize_calc_expression(expression: str) -> str:
    normalized = expression.strip()
    # LLMs often emit ^ for exponentiation even though Python uses **.
    normalized = normalized.replace("^", "**")
    return normalized


@tool
def calc(expression: str) -> str:
    """
    Evaluate a simple math expression safely-ish.
    Examples: "2+2", "sin(0.5)", "sqrt(9)".
    """
    allowed = {
        "pi": math.pi,
        "e": math.e,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "abs": abs,
        "pow": pow,
    }
    try:
        normalized = _normalize_calc_expression(expression)
        # NOTE: still not perfectly safe for hostile input; good enough for a starter.
        value = eval(normalized, {"__builtins__": {}}, allowed)  # noqa: S307
        return str(value)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def now_utc() -> str:
    """Return the current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


@tool
def final_answer(answer: str) -> str:
    """
    Signal that you have a complete answer for the user.
    Call this — instead of replying in plain text — when you are ready to respond.
    The 'answer' argument should be a clear, complete response to the user's question.
    """
    # Never executed: agent_node intercepts this call before tool_node runs.
    return answer


ALL_TOOLS = (calc, now_utc, final_answer)
