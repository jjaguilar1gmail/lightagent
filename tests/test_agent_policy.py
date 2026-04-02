from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent_policy import PolicyContext, derive_turn_facts, evaluate_completion, tool_outcome_guidance


class AgentPolicyTests(unittest.TestCase):
    def test_derive_turn_facts_ignores_final_answer_as_real_tool_attempt(self) -> None:
        context = PolicyContext(
            messages=[
                HumanMessage(content="How many golf balls fit in a school bus?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "final_answer",
                            "args": {"answer": "unknown"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        )

        facts = derive_turn_facts(context)

        self.assertTrue(facts.has_final_answer_attempt)
        self.assertFalse(facts.has_real_tool_attempt)

    def test_derive_turn_facts_marks_nonterminal_tool_call_as_real_attempt(self) -> None:
        context = PolicyContext(
            messages=[
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
            ]
        )

        facts = derive_turn_facts(context)

        self.assertTrue(facts.has_real_tool_attempt)
        self.assertFalse(facts.has_final_answer_attempt and not facts.has_real_tool_attempt)

    def test_tool_outcome_guidance_for_failed_tool(self) -> None:
        context = PolicyContext(
            messages=[ToolMessage(content="ERROR: NameError: x is not defined", tool_call_id="calc_1")]
        )

        guidance = tool_outcome_guidance(context)

        self.assertIsNotNone(guidance)
        self.assertIn("last tool call failed", guidance)

    def test_tool_outcome_guidance_for_successful_tool(self) -> None:
        context = PolicyContext(messages=[ToolMessage(content="42", tool_call_id="calc_1")])

        guidance = tool_outcome_guidance(context)

        self.assertIsNotNone(guidance)
        self.assertIn("tool result is now available", guidance)

    def test_evaluate_completion_rejects_unsupported_plain_text_termination(self) -> None:
        context = PolicyContext(messages=[HumanMessage(content="How many golf balls fit in a school bus?")])

        decision = evaluate_completion(
            context,
            "Let's calculate the bus volume and estimate the total number of golf balls that fit.",
        )

        self.assertFalse(decision.allowed)
        self.assertIsNotNone(decision.feedback)
        self.assertIn("no supporting tool result", decision.feedback)

    def test_evaluate_completion_rejects_curly_apostrophe_termination(self) -> None:
        context = PolicyContext(messages=[HumanMessage(content="How many golf balls fit in a school bus?")])

        decision = evaluate_completion(
            context,
            "Let’s do a rough calculation and estimate the total number that fit.",
        )

        self.assertFalse(decision.allowed)
        self.assertIsNotNone(decision.feedback)
        self.assertIn("no supporting tool result", decision.feedback)

    def test_evaluate_completion_requires_action_before_giving_up(self) -> None:
        context = PolicyContext(messages=[HumanMessage(content="How many golf balls fit in a school bus?")])

        decision = evaluate_completion(context, "The answer is unknown.")

        self.assertFalse(decision.allowed)
        self.assertIsNotNone(decision.feedback)
        self.assertIn("Do not give up before attempting an action", decision.feedback)

    def test_evaluate_completion_requires_real_action_before_giving_up(self) -> None:
        context = PolicyContext(
            messages=[
                HumanMessage(content="How many golf balls fit in a school bus?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "final_answer",
                            "args": {"answer": "unknown"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(content="ERROR: try again", tool_call_id="call_1"),
            ]
        )

        decision = evaluate_completion(context, "The answer is unknown.")

        self.assertFalse(decision.allowed)
        self.assertIsNotNone(decision.feedback)
        self.assertIn("Do not give up before attempting an action", decision.feedback)

    def test_evaluate_completion_allows_supported_plain_text_after_tool(self) -> None:
        context = PolicyContext(
            messages=[
                HumanMessage(content="What is 6 times 7?"),
                ToolMessage(content="42", tool_call_id="calc_1"),
            ]
        )

        decision = evaluate_completion(context, "I calculated the result: 42.")

        self.assertTrue(decision.allowed)
        self.assertIsNone(decision.feedback)


if __name__ == "__main__":
    unittest.main()