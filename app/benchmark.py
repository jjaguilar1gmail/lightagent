from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .evaluator import BestNextSupport, EvaluatorResult, Outcome, ToolFamily, build_evaluator_input, evaluate_run
from .structured_output import schema_instruction, validate_structured_output
from .graph import build_graph
from .tools import ALL_TOOLS

load_dotenv()

AnswerQuality = Literal["excellent", "good", "partial", "poor", "failed"]
FailureMode = Literal[
    "none",
    "premature_refusal",
    "missed_existing_tool_use",
    "should_have_asked_for_clarification",
    "missing_capability_exposed",
    "unsupported_confident_claim",
    "nonresponsive",
    "other",
]


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    question: str
    context: str
    tags: tuple[str, ...]
    expected_best_next_support: BestNextSupport
    expected_outcome: Outcome
    expected_helpful_tool_family: ToolFamily | None
    current_tools_sufficient: bool
    answer_expectation: str


def load_split_ids(split_path: str | Path, split: str) -> set[str]:
    data = json.loads(Path(split_path).read_text(encoding="utf-8"))
    split_ids = data.get(split)
    if not isinstance(split_ids, list):
        raise ValueError(f"Unknown split '{split}' in {split_path}")
    return {str(item) for item in split_ids}


@dataclass(frozen=True)
class AnswerJudgeResult:
    answered_well: bool
    score: int
    quality: AnswerQuality
    main_failure_mode: FailureMode
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkRecord:
    question_id: str
    question: str
    tags: tuple[str, ...]
    expected_best_next_support: BestNextSupport
    expected_outcome: Outcome
    expected_helpful_tool_family: ToolFamily | None
    final_answer: str
    termination_reason: str
    graph_error: str | None
    evaluator_error: str | None
    evaluator_result: EvaluatorResult | None
    answer_judge: AnswerJudgeResult
    elapsed_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "tags": list(self.tags),
            "expected_best_next_support": self.expected_best_next_support,
            "expected_outcome": self.expected_outcome,
            "expected_helpful_tool_family": self.expected_helpful_tool_family,
            "final_answer": self.final_answer,
            "termination_reason": self.termination_reason,
            "graph_error": self.graph_error,
            "evaluator_error": self.evaluator_error,
            "evaluator_result": self.evaluator_result.to_dict() if self.evaluator_result is not None else None,
            "answer_judge": self.answer_judge.to_dict(),
            "elapsed_ms": self.elapsed_ms,
        }


class _AnswerJudgeOutputModel(BaseModel):
    answered_well: bool
    score: int = Field(ge=0, le=4)
    quality: AnswerQuality
    main_failure_mode: FailureMode
    reason: str = Field(min_length=1)


def load_benchmark_questions(dataset_path: str | Path) -> list[BenchmarkQuestion]:
    raw_items = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    questions: list[BenchmarkQuestion] = []
    for item in raw_items:
        questions.append(
            BenchmarkQuestion(
                id=str(item["id"]),
                question=str(item["question"]),
                context=str(item.get("context", "")),
                tags=tuple(item.get("tags", [])),
                expected_best_next_support=item["expected_best_next_support"],
                expected_outcome=item["expected_outcome"],
                expected_helpful_tool_family=item.get("expected_helpful_tool_family"),
                current_tools_sufficient=bool(item["current_tools_sufficient"]),
                answer_expectation=str(item["answer_expectation"]),
            )
        )
    return questions


def _build_user_prompt(question: BenchmarkQuestion) -> str:
    if not question.context:
        return question.question
    return f"Context:\n{question.context}\n\nQuestion:\n{question.question}"


def _extract_final_answer(final_state: dict[str, Any]) -> str:
    answer = str(final_state.get("final_answer", "")).strip()
    if answer:
        return answer
    for message in reversed(final_state.get("messages", [])):
        content = str(getattr(message, "content", "")).strip()
        if getattr(message, "type", "") == "ai" and content:
            return content
    return ""


def _default_answer_judge_model() -> Any:
    return ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct",
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


def _looks_like_unresolved_tool_artifact(final_answer: str) -> bool:
    text = str(final_answer or "").strip()
    if not text:
        return False

    lowered = text.lower()
    return (
        lowered.startswith("{'function_name':")
        or lowered.startswith('{"function_name":')
        or lowered.startswith("{'name':")
        or lowered.startswith('{"name":')
        or ("tool_call" in lowered and "args" in lowered)
    )


def judge_answer(
    question: BenchmarkQuestion,
    final_answer: str,
    termination_reason: str,
    graph_error: str | None = None,
    model: Any = None,
) -> AnswerJudgeResult:
    if graph_error:
        return AnswerJudgeResult(
            answered_well=False,
            score=0,
            quality="failed",
            main_failure_mode="other",
            reason=f"The graph errored before producing a usable answer: {graph_error}",
        )

    if _looks_like_unresolved_tool_artifact(final_answer):
        return AnswerJudgeResult(
            answered_well=False,
            score=0,
            quality="failed",
            main_failure_mode="nonresponsive",
            reason="The final answer leaked an internal tool-call artifact instead of a user-facing answer.",
        )

    prompt_payload = {
        "question": question.question,
        "context": question.context,
        "final_answer": final_answer,
        "termination_reason": termination_reason,
        "tags": list(question.tags),
        "current_tools_sufficient": question.current_tools_sufficient,
        "expected_best_next_support": question.expected_best_next_support,
        "answer_expectation": question.answer_expectation,
    }
    runnable = model or _default_answer_judge_model()
    raw = runnable.invoke(
        [
            SystemMessage(
                content=(
                    "You are grading benchmark answers from a lightweight agent. "
                    "Judge usefulness and problem-solving quality, not style. "
                    "If the benchmark says current tools were sufficient, a refusal or limitation-only answer is usually poor. "
                    "For rough-estimate questions, explicit assumptions plus a ballpark answer can count as good. "
                    f"{schema_instruction(_AnswerJudgeOutputModel)}"
                )
            ),
            HumanMessage(content=json.dumps(prompt_payload, indent=2)),
        ]
    )
    parsed = validate_structured_output(raw, _AnswerJudgeOutputModel)
    return AnswerJudgeResult(
        answered_well=parsed.answered_well,
        score=parsed.score,
        quality=parsed.quality,
        main_failure_mode=parsed.main_failure_mode,
        reason=parsed.reason,
    )


def run_benchmark(
    dataset_path: str | Path,
    output_dir: str | Path = "benchmark_results",
    *,
    limit: int | None = None,
    split: str | None = None,
    split_path: str | Path | None = None,
    evaluator_model: Any = None,
    answer_judge_model: Any = None,
) -> dict[str, Any]:
    questions = load_benchmark_questions(dataset_path)
    if split is not None:
        active_split_path = Path(split_path or Path(dataset_path).with_name("splits.json"))
        selected_ids = load_split_ids(active_split_path, split)
        questions = [question for question in questions if question.id in selected_ids]
    if limit is not None:
        questions = questions[:limit]

    graph = build_graph()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records: list[BenchmarkRecord] = []

    for item in questions:
        started = time.monotonic()
        final_state: dict[str, Any] = {"messages": [], "final_answer": "", "termination_reason": "error"}
        graph_error = None
        evaluator_result = None
        evaluator_error = None
        try:
            final_state = graph.invoke(
                {
                    "messages": [HumanMessage(content=_build_user_prompt(item))],
                    "max_steps": 8,
                }
            )
        except Exception as exc:
            graph_error = f"{type(exc).__name__}: {exc}"

        final_answer = _extract_final_answer(final_state)
        termination_reason = str(final_state.get("termination_reason", "")).strip()

        if graph_error is None:
            try:
                evaluator_input = build_evaluator_input(final_state, available_tools=[tool.name for tool in ALL_TOOLS])
                evaluator_result = evaluate_run(evaluator_input, model=evaluator_model)
            except Exception as exc:
                evaluator_error = f"{type(exc).__name__}: {exc}"

        answer_judge = judge_answer(
            item,
            final_answer=final_answer,
            termination_reason=termination_reason,
            graph_error=graph_error,
            model=answer_judge_model,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        records.append(
            BenchmarkRecord(
                question_id=item.id,
                question=item.question,
                tags=item.tags,
                expected_best_next_support=item.expected_best_next_support,
                expected_outcome=item.expected_outcome,
                expected_helpful_tool_family=item.expected_helpful_tool_family,
                final_answer=final_answer,
                termination_reason=termination_reason,
                graph_error=graph_error,
                evaluator_error=evaluator_error,
                evaluator_result=evaluator_result,
                answer_judge=answer_judge,
                elapsed_ms=elapsed_ms,
            )
        )

    summary = summarize_benchmark(records)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "dataset_path": str(dataset_path),
        "split": split,
        "split_path": str(split_path) if split_path is not None else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "records": [record.to_dict() for record in records],
    }
    run_file = output_path / f"benchmark_{stamp}.json"
    run_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["output_file"] = str(run_file)
    return result


def summarize_benchmark(records: Sequence[BenchmarkRecord]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "total_questions": 0,
            "answer_success_rate": 0.0,
            "evaluator_best_next_support_accuracy": 0.0,
            "evaluator_outcome_accuracy": 0.0,
            "helpful_tool_family_match_rate": 0.0,
            "graph_failure_modes": {},
            "support_confusion": {},
            "by_tag": {},
            "by_expected_support": {},
            "common_helpful_tool_families": {},
            "weak_points": [],
        }

    answer_successes = 0
    support_matches = 0
    outcome_matches = 0
    helpful_family_matches = 0
    helpful_family_total = 0
    graph_failure_modes: Counter[str] = Counter()
    support_confusion: Counter[str] = Counter()
    common_helpful_tool_families: Counter[str] = Counter()
    by_tag: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    by_support: dict[str, list[BenchmarkRecord]] = defaultdict(list)

    for record in records:
        if record.answer_judge.answered_well:
            answer_successes += 1
        graph_failure_modes[record.answer_judge.main_failure_mode] += 1
        by_support[record.expected_best_next_support].append(record)
        for tag in record.tags:
            by_tag[tag].append(record)

        predicted_support = record.evaluator_result.best_next_support if record.evaluator_result is not None else "error"
        support_confusion[f"{record.expected_best_next_support} -> {predicted_support}"] += 1
        if record.evaluator_result is not None and record.evaluator_result.best_next_support == record.expected_best_next_support:
            support_matches += 1
        if record.evaluator_result is not None and record.evaluator_result.outcome == record.expected_outcome:
            outcome_matches += 1

        predicted_family = None
        if record.evaluator_result is not None and record.evaluator_result.helpful_tool_idea is not None:
            predicted_family = record.evaluator_result.helpful_tool_idea.family
            common_helpful_tool_families[predicted_family] += 1
        if record.expected_helpful_tool_family is not None:
            helpful_family_total += 1
            if predicted_family == record.expected_helpful_tool_family:
                helpful_family_matches += 1

    def _slice_stats(items: Sequence[BenchmarkRecord]) -> dict[str, Any]:
        if not items:
            return {"count": 0, "answer_success_rate": 0.0, "support_accuracy": 0.0}
        answers = sum(1 for item in items if item.answer_judge.answered_well)
        supports = sum(
            1
            for item in items
            if item.evaluator_result is not None and item.evaluator_result.best_next_support == item.expected_best_next_support
        )
        return {
            "count": len(items),
            "answer_success_rate": round(answers / len(items), 3),
            "support_accuracy": round(supports / len(items), 3),
        }

    tag_stats = {tag: _slice_stats(items) for tag, items in sorted(by_tag.items())}
    support_stats = {support: _slice_stats(items) for support, items in sorted(by_support.items())}

    weak_points = _derive_weak_points(records, support_stats, tag_stats, support_confusion, graph_failure_modes)

    return {
        "total_questions": total,
        "answer_success_rate": round(answer_successes / total, 3),
        "evaluator_best_next_support_accuracy": round(support_matches / total, 3),
        "evaluator_outcome_accuracy": round(outcome_matches / total, 3),
        "helpful_tool_family_match_rate": round(helpful_family_matches / helpful_family_total, 3)
        if helpful_family_total
        else None,
        "graph_failure_modes": dict(graph_failure_modes.most_common()),
        "support_confusion": dict(support_confusion.most_common()),
        "by_tag": tag_stats,
        "by_expected_support": support_stats,
        "common_helpful_tool_families": dict(common_helpful_tool_families.most_common()),
        "weak_points": weak_points,
    }


def _derive_weak_points(
    records: Sequence[BenchmarkRecord],
    support_stats: dict[str, dict[str, Any]],
    tag_stats: dict[str, dict[str, Any]],
    support_confusion: Counter[str],
    graph_failure_modes: Counter[str],
) -> list[str]:
    weak_points: list[str] = []

    existing_tool_stats = support_stats.get("existing_tools")
    if existing_tool_stats and existing_tool_stats["answer_success_rate"] < 0.6:
        weak_points.append(
            "The ReAct graph underperforms on questions that should be solvable with the current tool set, which usually means it is refusing too early or not turning assumptions into calculations."
        )

    clarification_stats = support_stats.get("user_clarification")
    if clarification_stats and clarification_stats["support_accuracy"] < 0.6:
        weak_points.append(
            "The evaluator is weak at recognizing when the next best move is a follow-up question instead of a hard limitation or a missing tool diagnosis."
        )

    missing_tool_stats = support_stats.get("missing_tool")
    if missing_tool_stats and missing_tool_stats["support_accuracy"] < 0.6:
        weak_points.append(
            "The evaluator is inconsistent at identifying genuine capability gaps, so missing-tool recommendations are not yet reliable enough to automate blindly."
        )

    if graph_failure_modes.get("premature_refusal", 0) >= max(2, len(records) // 8):
        weak_points.append(
            "Premature refusal is a frequent graph failure mode, which suggests the agent is defaulting to limitation language instead of attempting assumptions, math, or decomposition."
        )

    if graph_failure_modes.get("missed_existing_tool_use", 0) >= max(2, len(records) // 8):
        weak_points.append(
            "Missed existing-tool use shows up often enough that tool selection and completion policy likely need tighter prompting or a stronger action-before-refusal rule."
        )

    confusing_pattern = next(
        (
            pattern
            for pattern, count in support_confusion.most_common()
            if count >= 2 and "missing_tool -> no_reasonable_support_path" in pattern
        ),
        None,
    )
    if confusing_pattern:
        weak_points.append(
            "The evaluator sometimes treats real capability gaps as dead ends, which hides obvious next tools instead of surfacing them."
        )

    low_tags = [
        tag
        for tag, stats in tag_stats.items()
        if stats["count"] >= 2 and stats["answer_success_rate"] < 0.5
    ]
    if low_tags:
        weak_points.append(
            "Low answer quality clusters in these benchmark tags: " + ", ".join(sorted(low_tags[:6])) + "."
        )

    if not weak_points:
        weak_points.append(
            "No single dominant weakness stands out yet; the next step is to expand the benchmark and inspect individual mismatch cases."
        )

    return weak_points


def _print_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print(f"questions: {summary['total_questions']}")
    print(f"answer success rate: {summary['answer_success_rate']}")
    print(f"evaluator best-next-support accuracy: {summary['evaluator_best_next_support_accuracy']}")
    print(f"evaluator outcome accuracy: {summary['evaluator_outcome_accuracy']}")
    family_rate = summary["helpful_tool_family_match_rate"]
    if family_rate is not None:
        print(f"helpful tool family match rate: {family_rate}")
    print("weak points:")
    for item in summary["weak_points"]:
        print(f"- {item}")
    print(f"output: {result['output_file']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightagent benchmark suite.")
    parser.add_argument("--dataset", default="benchmarks/questions.json", help="Path to benchmark dataset JSON file.")
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory for benchmark result files.")
    parser.add_argument("--split", default=None, help="Optional split name, such as train or holdout.")
    parser.add_argument("--split-path", default=None, help="Optional path to split manifest JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of questions to run.")
    args = parser.parse_args()

    result = run_benchmark(
        args.dataset,
        output_dir=args.output_dir,
        limit=args.limit,
        split=args.split,
        split_path=args.split_path,
    )
    _print_summary(result)


if __name__ == "__main__":
    main()