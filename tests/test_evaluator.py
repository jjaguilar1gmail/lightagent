from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.evaluator import build_evaluator_input, evaluate_run


class _StubEvaluatorModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.response


class EvaluatorTests(unittest.TestCase):
    def test_build_evaluator_input_collects_tool_history(self) -> None:
        final_state = {
            "messages": [
                HumanMessage(content="What is 6 times 7?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calc",
                            "args": {"expression": "6 * 7"},
                            "id": "calc_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(content="42", tool_call_id="calc_1"),
                AIMessage(content="42"),
            ],
            "final_answer": "42",
            "termination_reason": "completed",
        }

        evaluator_input = build_evaluator_input(final_state, available_tools=["calc", "now_utc", "final_answer"])

        self.assertEqual(evaluator_input.user_goal, "What is 6 times 7?")
        self.assertEqual(evaluator_input.final_answer, "42")
        self.assertEqual(evaluator_input.termination_reason, "completed")
        self.assertEqual(evaluator_input.available_tools, ("calc", "now_utc", "final_answer"))
        self.assertEqual(len(evaluator_input.tool_calls), 1)
        self.assertEqual(evaluator_input.tool_calls[0].name, "calc")
        self.assertEqual(evaluator_input.tool_calls[0].result, "42")
        self.assertTrue(evaluator_input.tool_calls[0].succeeded)

    def test_evaluate_run_coerces_answered_result(self) -> None:
        final_state = {
            "messages": [HumanMessage(content="What is 6 times 7?"), AIMessage(content="42")],
            "final_answer": "42",
            "termination_reason": "completed",
        }
        evaluator_input = build_evaluator_input(final_state, available_tools=["calc", "now_utc", "final_answer"])
        stub_model = _StubEvaluatorModel(
            {
                "answered": True,
                "support_level": "sufficient",
                "outcome": "answered_with_sufficient_support",
                "best_next_support": "none",
                "reason": "The answer directly resolves the user question.",
                "retry_with_existing_tools": False,
                "missing_capability": False,
                "suggested_clarification": None,
                "helpful_tool_idea": None,
                "tool_gap_summary": None,
            }
        )

        result = evaluate_run(evaluator_input, model=stub_model)

        self.assertTrue(result.answered)
        self.assertEqual(result.outcome, "answered_with_sufficient_support")
        self.assertEqual(result.support_level, "sufficient")
        self.assertEqual(result.best_next_support, "none")
        self.assertEqual(len(stub_model.calls), 1)

    def test_evaluate_run_coerces_recommended_tool(self) -> None:
        final_state = {
            "messages": [HumanMessage(content="What is the weather in Seattle right now?"), AIMessage(content="I cannot tell.")],
            "final_answer": "I cannot tell.",
            "termination_reason": "completed",
        }
        evaluator_input = build_evaluator_input(final_state, available_tools=["calc", "now_utc", "final_answer"])
        stub_model = _StubEvaluatorModel(
            {
                "answered": False,
                "support_level": "insufficient",
                "outcome": "not_answered_because_missing_tool_capability",
                "best_next_support": "missing_tool",
                "reason": "A current weather lookup requires external retrieval.",
                "retry_with_existing_tools": False,
                "missing_capability": True,
                "suggested_clarification": None,
                "helpful_tool_idea": {
                    "name": "weather_lookup",
                    "purpose": "Fetch current weather conditions by location",
                    "inputs": ["location"],
                    "outputs": "current weather conditions with timestamp",
                },
                "tool_gap_summary": "No external weather retrieval tool exists.",
            }
        )

        result = evaluate_run(evaluator_input, model=stub_model)

        self.assertFalse(result.answered)
        self.assertTrue(result.missing_capability)
        self.assertEqual(result.best_next_support, "missing_tool")
        self.assertIsNotNone(result.helpful_tool_idea)
        self.assertEqual(result.helpful_tool_idea.name, "weather_lookup")
        self.assertEqual(result.helpful_tool_idea.inputs, ("location",))

    def test_evaluate_run_allows_helpful_tool_idea_even_when_existing_tools_are_best_next_step(self) -> None:
        final_state = {
            "messages": [
                HumanMessage(content="How many marbles fit inside a milk jug?"),
                AIMessage(content="I cannot provide an exact answer without more details."),
            ],
            "final_answer": "I cannot provide an exact answer without more details.",
            "termination_reason": "completed",
        }
        evaluator_input = build_evaluator_input(final_state, available_tools=["calc", "now_utc", "final_answer"])
        stub_model = _StubEvaluatorModel(
            {
                "answered": False,
                "support_level": "insufficient",
                "outcome": "not_answered_but_existing_tool_should_have_been_used",
                "best_next_support": "existing_tools",
                "reason": "This is a rough estimation question and the agent could have answered with explicit assumptions plus calc.",
                "retry_with_existing_tools": True,
                "missing_capability": False,
                "suggested_clarification": "What jug size and marble diameter should I assume?",
                "helpful_tool_idea": {
                    "name": "web_search",
                    "purpose": "Look up typical milk jug volumes and marble diameters",
                    "inputs": ["query"],
                    "outputs": "ranked factual snippets with sources",
                },
                "tool_gap_summary": "External lookup could improve realism, but it is not required for a rough estimate.",
            }
        )

        result = evaluate_run(evaluator_input, model=stub_model)

        self.assertEqual(result.outcome, "not_answered_but_existing_tool_should_have_been_used")
        self.assertEqual(result.best_next_support, "existing_tools")
        self.assertEqual(result.suggested_clarification, "What jug size and marble diameter should I assume?")
        self.assertIsNotNone(result.helpful_tool_idea)
        self.assertEqual(result.helpful_tool_idea.name, "web_search")


if __name__ == "__main__":
    unittest.main()