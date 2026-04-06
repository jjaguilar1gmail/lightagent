from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage

from app.benchmark import (
    AnswerJudgeResult,
    BenchmarkQuestion,
    BenchmarkRecord,
    judge_answer,
    load_benchmark_questions,
    load_split_ids,
    summarize_benchmark,
)
from app.evaluator import EvaluatorResult, RecommendedTool


class BenchmarkTests(unittest.TestCase):
    def test_load_benchmark_questions_has_diverse_seed_set(self) -> None:
        questions = load_benchmark_questions("d:/repos/lightagent/benchmarks/questions.json")

        self.assertGreaterEqual(len(questions), 80)
        self.assertEqual(len({question.id for question in questions}), len(questions))
        self.assertIn("missing_tool", {question.expected_best_next_support for question in questions})
        self.assertIn("existing_tools", {question.expected_best_next_support for question in questions})
        self.assertIn("user_clarification", {question.expected_best_next_support for question in questions})

    def test_load_split_ids_reads_train_and_holdout(self) -> None:
        train_ids = load_split_ids("d:/repos/lightagent/benchmarks/splits.json", "train")
        holdout_ids = load_split_ids("d:/repos/lightagent/benchmarks/splits.json", "holdout")

        self.assertGreaterEqual(len(train_ids), 60)
        self.assertGreaterEqual(len(holdout_ids), 20)
        self.assertTrue(train_ids.isdisjoint(holdout_ids))

    def test_summarize_benchmark_surfaces_support_and_failure_weaknesses(self) -> None:
        records = [
            BenchmarkRecord(
                question_id="estimate_1",
                question="How many marbles fit inside a milk jug?",
                tags=("estimation", "fermi"),
                expected_best_next_support="existing_tools",
                expected_outcome="not_answered_but_existing_tool_should_have_been_used",
                expected_helpful_tool_family="web_retrieval",
                final_answer="I cannot know.",
                termination_reason="completed",
                graph_error=None,
                evaluator_error=None,
                evaluator_result=EvaluatorResult(
                    answered=False,
                    support_level="insufficient",
                    outcome="not_answered_due_to_ambiguous_or_unanswerable_request",
                    best_next_support="no_reasonable_support_path",
                    reason="The prompt is ambiguous.",
                    retry_with_existing_tools=False,
                    missing_capability=False,
                    suggested_clarification=None,
                    helpful_tool_idea=RecommendedTool(
                        name="web_search",
                        family="web_retrieval",
                        purpose="Look up typical milk jug volumes",
                        inputs=("query",),
                        outputs="search results",
                    ),
                    tool_gap_summary="Optional retrieval could help.",
                ),
                answer_judge=AnswerJudgeResult(
                    answered_well=False,
                    score=1,
                    quality="poor",
                    main_failure_mode="premature_refusal",
                    reason="The graph should have estimated instead of refusing.",
                ),
                elapsed_ms=100,
            ),
            BenchmarkRecord(
                question_id="clarify_1",
                question="Which laptop should I buy?",
                tags=("clarification", "shopping"),
                expected_best_next_support="user_clarification",
                expected_outcome="not_answered_due_to_ambiguous_or_unanswerable_request",
                expected_helpful_tool_family="shopping_lookup",
                final_answer="I need more information.",
                termination_reason="completed",
                graph_error=None,
                evaluator_error=None,
                evaluator_result=EvaluatorResult(
                    answered=False,
                    support_level="insufficient",
                    outcome="not_answered_due_to_ambiguous_or_unanswerable_request",
                    best_next_support="missing_tool",
                    reason="A shopping tool is needed.",
                    retry_with_existing_tools=False,
                    missing_capability=True,
                    suggested_clarification=None,
                    helpful_tool_idea=RecommendedTool(
                        name="product_search",
                        family="shopping_lookup",
                        purpose="Retrieve current laptop options",
                        inputs=("query",),
                        outputs="product list",
                    ),
                    tool_gap_summary="Missing shopping retrieval.",
                ),
                answer_judge=AnswerJudgeResult(
                    answered_well=False,
                    score=2,
                    quality="partial",
                    main_failure_mode="should_have_asked_for_clarification",
                    reason="The agent should have asked about budget and use case.",
                ),
                elapsed_ms=100,
            ),
            BenchmarkRecord(
                question_id="missing_1",
                question="What is the current weather in Seattle?",
                tags=("missing_tool", "weather"),
                expected_best_next_support="missing_tool",
                expected_outcome="not_answered_because_missing_tool_capability",
                expected_helpful_tool_family="api_lookup",
                final_answer="I do not have live weather access.",
                termination_reason="completed",
                graph_error=None,
                evaluator_error=None,
                evaluator_result=EvaluatorResult(
                    answered=False,
                    support_level="insufficient",
                    outcome="not_answered_due_to_ambiguous_or_unanswerable_request",
                    best_next_support="no_reasonable_support_path",
                    reason="The model cannot know.",
                    retry_with_existing_tools=False,
                    missing_capability=False,
                    suggested_clarification=None,
                    helpful_tool_idea=None,
                    tool_gap_summary=None,
                ),
                answer_judge=AnswerJudgeResult(
                    answered_well=True,
                    score=3,
                    quality="good",
                    main_failure_mode="none",
                    reason="It did not hallucinate live weather.",
                ),
                elapsed_ms=100,
            ),
        ]

        summary = summarize_benchmark(records)

        self.assertEqual(summary["total_questions"], 3)
        self.assertIn("premature_refusal", summary["graph_failure_modes"])
        self.assertIn("existing_tools -> no_reasonable_support_path", summary["support_confusion"])
        self.assertGreaterEqual(len(summary["weak_points"]), 2)

    def test_judge_answer_rejects_unresolved_tool_artifact(self) -> None:
        question = BenchmarkQuestion(
            id="time_001",
            question="What time is it in UTC right now?",
            context="",
            tags=("time", "current_info"),
            expected_best_next_support="none",
            expected_outcome="answered_with_sufficient_support",
            expected_helpful_tool_family="time",
            current_tools_sufficient=True,
            answer_expectation="A good answer should use the current UTC time.",
        )

        result = judge_answer(
            question,
            final_answer="{'function_name': 'now_utc', 'args': []}",
            termination_reason="completed",
            model=None,
        )

        self.assertFalse(result.answered_well)
        self.assertEqual(result.main_failure_mode, "nonresponsive")
        self.assertEqual(result.quality, "failed")

    def test_judge_answer_parses_json_with_trailing_text(self) -> None:
        question = BenchmarkQuestion(
            id="knowledge_005",
            question="What is recursion in programming, in one short paragraph?",
            context="",
            tags=("knowledge", "technology", "explanation"),
            expected_best_next_support="none",
            expected_outcome="answered_with_sufficient_support",
            expected_helpful_tool_family=None,
            current_tools_sufficient=True,
            answer_expectation="A good answer should explain recursion briefly.",
        )
        stub_model = type(
            "StubJudgeModel",
            (),
            {
                "invoke": lambda self, messages: AIMessage(
                    content=(
                        '{"answered_well": true, "score": 1, "quality": "good", '
                        '"main_failure_mode": "none", "reason": "The answer is correct."}\n\nExtra note.'
                    )
                )
            },
        )()

        result = judge_answer(
            question,
            final_answer="Recursion is when a function calls itself with a base case.",
            termination_reason="completed",
            model=stub_model,
        )

        self.assertTrue(result.answered_well)
        self.assertEqual(result.quality, "good")
