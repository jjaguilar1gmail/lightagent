from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from .graph import build_graph
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

    with new_run(sinks=sinks) as run:
        final = graph.invoke(init_state)

    # Print final assistant message
    msgs = final["messages"]
    for m in reversed(msgs):
        if getattr(m, "type", "") == "ai":
            print("\n---\n")
            print(m.content)
            break

    print(f"\n[trace written to traces/{run.run_id}.jsonl and traces/traces.db]")


if __name__ == "__main__":
    main()