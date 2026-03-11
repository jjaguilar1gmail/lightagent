from __future__ import annotations

import unittest
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.tracing.callbacks import TracingCallbackHandler


class TracingCallbackTests(unittest.TestCase):
    def test_on_llm_end_emits_tool_calls_for_chat_generation(self) -> None:
        handler = TracingCallbackHandler(node="agent")
        run_id = uuid4()
        handler._starts[str(run_id)] = 0.0
        response = LLMResult(
            generations=[
                [
                    ChatGeneration(
                        text="",
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "final_answer",
                                    "args": {"answer": "42"},
                                    "id": "call_1",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                    )
                ]
            ]
        )

        with patch("app.tracing.callbacks.emit") as emit_mock:
            handler.on_llm_end(response, run_id=run_id)

        emit_mock.assert_called_once()
        payload = emit_mock.call_args.kwargs["payload"]
        self.assertEqual(payload["response"], "")
        self.assertEqual(payload["response_text"], "")
        self.assertEqual(len(payload["tool_calls"]), 1)
        self.assertEqual(payload["tool_calls"][0]["name"], "final_answer")
        self.assertEqual(payload["tool_calls"][0]["args"]["answer"], "42")


if __name__ == "__main__":
    unittest.main()