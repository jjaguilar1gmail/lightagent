from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import graph


class GraphWorkflowTests(unittest.TestCase):
    def test_agent_node_completes_with_plain_text_response(self) -> None:
        state = {
            "messages": [HumanMessage(content="How many golf balls fit in a school bus?")],
            "user_goal": "How many golf balls fit in a school bus?",
            "step": 0,
            "max_steps": 8,
        }

        stub_llm = type("StubLLM", (), {"invoke": lambda self, msgs, config=None: AIMessage(content="Final answer text")})()
        with patch("app.graph.llm", new=stub_llm):
            result = graph.agent_node(state, config={})

        self.assertTrue(result["done"])
        self.assertEqual(result["termination_reason"], "completed")
        self.assertEqual(result["final_answer"], "Final answer text")

    def test_agent_node_intercepts_final_answer_tool(self) -> None:
        state = {
            "messages": [HumanMessage(content="test")],
            "user_goal": "test",
            "step": 0,
            "max_steps": 5,
        }
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "final_answer",
                    "args": {"answer": "42"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )

        stub_llm = type("StubLLM", (), {"invoke": lambda self, msgs, config=None: response})()
        with patch("app.graph.llm", new=stub_llm):
            result = graph.agent_node(state, config={})

        self.assertTrue(result["done"])
        self.assertEqual(result["final_answer"], "42")
        self.assertEqual(result["termination_reason"], "completed")

    def test_agent_node_allows_unknown_after_tool_attempt(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="How many golf balls fit in a school bus?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calc",
                            "args": {"expression": "45 * 10 * 9"},
                            "id": "calc_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(content="ERROR: NameError: x is not defined", tool_call_id="calc_1"),
            ],
            "user_goal": "How many golf balls fit in a school bus?",
            "step": 1,
            "max_steps": 5,
        }

        stub_llm = type(
            "StubLLM",
            (),
            {"invoke": lambda self, msgs, config=None: AIMessage(content="The answer is unknown.")},
        )()
        with patch("app.graph.llm", new=stub_llm):
            result = graph.agent_node(state, config={})

        self.assertTrue(result["done"])
        self.assertEqual(result["final_answer"], "The answer is unknown.")

    def test_tool_node_executes_calc_calls(self) -> None:
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calc",
                            "args": {"expression": "6 * 7"},
                            "id": "calc_1",
                            "type": "tool_call",
                        },
                    ],
                )
            ],
        }

        result = graph.tool_node(state)

        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][-1].content, "42")

    def test_agent_node_enforces_step_budget(self) -> None:
        state = {
            "messages": [HumanMessage(content="test")],
            "user_goal": "test",
            "step": 5,
            "max_steps": 5,
        }

        result = graph.agent_node(state, config={})

        self.assertTrue(result["done"])
        self.assertEqual(result["termination_reason"], "max_steps")


if __name__ == "__main__":
    unittest.main()