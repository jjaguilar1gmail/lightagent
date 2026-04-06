from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from .evaluator import build_evaluator_input, evaluate_run
from .graph import build_graph
from .tools import ALL_TOOLS
from .tracing import new_run, ConsoleSink, JSONLSink, SQLiteSink


def main():
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("Missing OPENROUTER_API_KEY (set it in env or .env).")

    debug = "--debug" in sys.argv
    llm_verbose = "--llm-verbose" in sys.argv
    pretty = not debug

    graph = build_graph()

    print("LangGraph Starter CLI. Type your prompt and press Enter.")
    if debug:
        print("[debug mode: raw JSON event stream]")
    elif llm_verbose:
        print("[llm-verbose mode: full LLM responses shown]")
    print()
    user = input("> ").strip()

    init_state = {
        "messages": [HumanMessage(content=user)],
        "max_steps": 8,
    }

    jsonl = JSONLSink(directory="traces")
    sinks = [
        ConsoleSink(pretty=pretty, llm_verbose=llm_verbose),
        jsonl,
        SQLiteSink(path="traces/traces.db"),
    ]

    evaluation = None
    with new_run(sinks=sinks) as run:
        final = graph.invoke(init_state)
        evaluator_input = build_evaluator_input(final, available_tools=[tool.name for tool in ALL_TOOLS])
        try:
            evaluation = evaluate_run(evaluator_input)
        except Exception:
            evaluation = None

    # Prefer the structured state field; fall back to scanning messages.
    final_answer_text = final.get("final_answer", "")
    if final_answer_text:
        print("\n---\n")
        print(final_answer_text)
    else:
        msgs = final["messages"]
        for m in reversed(msgs):
            if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
                print("\n---\n")
                print(m.content)
                break

    if evaluation is not None:
        print("\n[evaluator]")
        print(f"outcome: {evaluation.outcome}")
        print(f"support: {evaluation.support_level}")
        print(f"best next support: {evaluation.best_next_support}")
        print(evaluation.reason)
        if evaluation.suggested_clarification:
            print(f"suggested clarification: {evaluation.suggested_clarification}")
        if evaluation.helpful_tool_idea is not None:
            print(
                "helpful tool idea: "
                f"{evaluation.helpful_tool_idea.name}"
                f" [{evaluation.helpful_tool_idea.family}]"
                f" ({evaluation.helpful_tool_idea.purpose})"
            )
        if evaluation.tool_gap_summary:
            print(f"tool gap: {evaluation.tool_gap_summary}")

    print(f"\n[trace written to traces/{run.run_id}.jsonl and traces/traces.db]")


if __name__ == "__main__":
    main()