from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .tracing import emit

load_dotenv()

Outcome = Literal[
    "answered_with_sufficient_support",
    "not_answered_but_existing_tool_should_have_been_used",
    "not_answered_because_missing_tool_capability",
    "not_answered_due_to_ambiguous_or_unanswerable_request",
]

SupportLevel = Literal["sufficient", "insufficient", "mixed"]
BestNextSupport = Literal[
    "none",
    "existing_tools",
    "user_clarification",
    "missing_tool",
    "no_reasonable_support_path",
]


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    call_id: str
    result: str | None = None
    succeeded: bool | None = None


@dataclass(frozen=True)
class EvaluatorInput:
    user_goal: str
    final_answer: str
    termination_reason: str
    available_tools: tuple[str, ...]
    messages: Sequence[AnyMessage]
    tool_calls: tuple[ToolCallRecord, ...]


@dataclass(frozen=True)
class RecommendedTool:
    name: str
    purpose: str
    inputs: tuple[str, ...]
    outputs: str


@dataclass(frozen=True)
class EvaluatorResult:
    answered: bool
    support_level: SupportLevel
    outcome: Outcome
    best_next_support: BestNextSupport
    reason: str
    retry_with_existing_tools: bool
    missing_capability: bool
    suggested_clarification: str | None = None
    helpful_tool_idea: RecommendedTool | None = None
    tool_gap_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.helpful_tool_idea is not None:
            payload["helpful_tool_idea"]["inputs"] = list(self.helpful_tool_idea.inputs)
        return payload


class _RecommendedToolModel(BaseModel):
    name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)
    outputs: str = Field(min_length=1)


class _EvaluatorOutputModel(BaseModel):
    answered: bool
    support_level: SupportLevel
    outcome: Outcome
    best_next_support: BestNextSupport
    reason: str = Field(min_length=1)
    retry_with_existing_tools: bool
    missing_capability: bool
    suggested_clarification: str | None = None
    helpful_tool_idea: _RecommendedToolModel | None = None
    tool_gap_summary: str | None = None


def _extract_user_goal(messages: Sequence[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            text = str(message.content).strip()
            if text:
                return text
    return ""


def _extract_final_answer(state: dict[str, Any]) -> str:
    answer = str(state.get("final_answer", "")).strip()
    if answer:
        return answer

    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            text = str(message.content).strip()
            if text:
                return text

    return ""


def _collect_tool_calls(messages: Sequence[AnyMessage]) -> tuple[ToolCallRecord, ...]:
    records: list[ToolCallRecord] = []
    pending: dict[str, ToolCallRecord] = {}
    record_index_by_call_id: dict[str, int] = {}

    for message in messages:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", None) or []:
                name = str(call.get("name") or "").strip()
                if not name or name == "final_answer":
                    continue
                call_id = str(call.get("id") or f"{name}_{len(records) + 1}")
                record = ToolCallRecord(
                    name=name,
                    args=dict(call.get("args") or {}),
                    call_id=call_id,
                )
                pending[call_id] = record
                record_index_by_call_id[call_id] = len(records)
                records.append(record)
                continue

        if not isinstance(message, ToolMessage):
            continue

        call_id = str(message.tool_call_id or "")
        if not call_id or call_id not in pending:
            continue

        prior = pending[call_id]
        pending[call_id] = ToolCallRecord(
            name=prior.name,
            args=prior.args,
            call_id=prior.call_id,
            result=str(message.content),
            succeeded=not str(message.content).startswith("ERROR:"),
        )
        records[record_index_by_call_id[call_id]] = pending[call_id]

    return tuple(records)


def build_evaluator_input(final_state: dict[str, Any], available_tools: Sequence[str]) -> EvaluatorInput:
    messages = tuple(final_state.get("messages", []))
    return EvaluatorInput(
        user_goal=str(final_state.get("user_goal", "")).strip() or _extract_user_goal(messages),
        final_answer=_extract_final_answer(final_state),
        termination_reason=str(final_state.get("termination_reason", "")).strip(),
        available_tools=tuple(available_tools),
        messages=messages,
        tool_calls=_collect_tool_calls(messages),
    )


def _input_summary(run_input: EvaluatorInput) -> dict[str, Any]:
    return {
        "user_goal": run_input.user_goal,
        "final_answer": run_input.final_answer,
        "termination_reason": run_input.termination_reason,
        "available_tools": list(run_input.available_tools),
        "tool_calls": [
            {
                "name": record.name,
                "args": record.args,
                "call_id": record.call_id,
                "result": record.result,
                "succeeded": record.succeeded,
            }
            for record in run_input.tool_calls
        ],
    }


def _build_prompt_messages(run_input: EvaluatorInput) -> list[AnyMessage]:
    summary = json.dumps(_input_summary(run_input), indent=2)
    return [
        SystemMessage(
            content=(
                "You are the post-run evaluator for a lightweight agent. "
                "Judge whether the user was answered well enough, whether the answer was supported, "
                "and whether any failure came from poor use of existing tools, a missing tool capability, "
                "a need for user clarification, or a genuinely ambiguous or unanswerable request. "
                "Choose the single best next support path before classifying something as ambiguous. "
                "Treat estimation or Fermi-style questions as answerable when reasonable assumptions plus existing tools could produce a useful approximate answer. "
                "Be conservative about proposing new tools, but feel free to suggest a realistic helpful tool idea even when existing tools are still the best immediate next step."
            )
        ),
        HumanMessage(
            content=(
                "Return a structured evaluation for this completed run.\n\n"
                "Rules:\n"
                "- If the answer is materially responsive and adequately supported, choose answered_with_sufficient_support.\n"
                "- If the answer is weak but an existing tool could plausibly have closed the gap, choose not_answered_but_existing_tool_should_have_been_used.\n"
                "- If no existing tool could plausibly close the gap, choose not_answered_because_missing_tool_capability.\n"
                "- If the system mostly needs a follow-up question from the user, keep the outcome aligned with the main failure mode but set best_next_support to user_clarification.\n"
                "- Only choose not_answered_due_to_ambiguous_or_unanswerable_request when there is no reasonable support path via existing tools, clarification, or a realistic missing tool.\n"
                "- Set best_next_support to one of: none, existing_tools, user_clarification, missing_tool, no_reasonable_support_path.\n"
                "- For Fermi or rough-estimate questions, prefer existing_tools or user_clarification over ambiguous.\n"
                "- suggested_clarification should be a concrete follow-up question only when clarification would materially help.\n"
                "- helpful_tool_idea may be provided whenever a realistic external capability would have materially improved the result, even if existing_tools is still the best immediate next step.\n\n"
                f"Run summary:\n{summary}"
            )
        ),
    ]


def _default_structured_model() -> Any:
    api_key = os.getenv("OPENROUTER_API_KEY")
    return ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct",
        temperature=0,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    ).with_structured_output(_EvaluatorOutputModel)


def _coerce_result(raw: Any) -> EvaluatorResult:
    parsed = raw if isinstance(raw, _EvaluatorOutputModel) else _EvaluatorOutputModel.model_validate(raw)
    helpful_tool_idea = None
    if parsed.helpful_tool_idea is not None:
        helpful_tool_idea = RecommendedTool(
            name=parsed.helpful_tool_idea.name,
            purpose=parsed.helpful_tool_idea.purpose,
            inputs=tuple(parsed.helpful_tool_idea.inputs),
            outputs=parsed.helpful_tool_idea.outputs,
        )

    return EvaluatorResult(
        answered=parsed.answered,
        support_level=parsed.support_level,
        outcome=parsed.outcome,
        best_next_support=parsed.best_next_support,
        reason=parsed.reason,
        retry_with_existing_tools=parsed.retry_with_existing_tools,
        missing_capability=parsed.missing_capability,
        suggested_clarification=parsed.suggested_clarification,
        helpful_tool_idea=helpful_tool_idea,
        tool_gap_summary=parsed.tool_gap_summary,
    )


def evaluate_run(run_input: EvaluatorInput, model: Any = None) -> EvaluatorResult:
    start = time.monotonic()
    emit(
        "node_start",
        node="evaluator",
        payload={
            "user_goal": run_input.user_goal[:120],
            "final_answer": run_input.final_answer[:160],
            "termination_reason": run_input.termination_reason[:120],
        },
    )
    try:
        runnable = model or _default_structured_model()
        raw = runnable.invoke(_build_prompt_messages(run_input))
        result = _coerce_result(raw)
        elapsed = int((time.monotonic() - start) * 1000)
        emit("node_end", node="evaluator", payload={**result.to_dict(), "elapsed_ms": elapsed})
        return result
    except Exception as exc:
        elapsed = int((time.monotonic() - start) * 1000)
        emit(
            "node_error",
            node="evaluator",
            payload={
                "user_goal": run_input.user_goal[:120],
                "termination_reason": run_input.termination_reason[:120],
                "elapsed_ms": elapsed,
                "error": str(exc),
            },
        )
        raise