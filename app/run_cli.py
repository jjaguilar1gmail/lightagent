from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from .graph import build_graph


def main():
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("Missing OPENROUTER_API_KEY (set it in env or .env).")

    graph = build_graph()

    print("LangGraph Starter CLI. Type your prompt and press Enter.\n")
    user = input("> ").strip()

    init_state = {
        "messages": [HumanMessage(content=user)],
        "max_steps": 8,
    }

    final = graph.invoke(init_state)

    # Print final assistant message(s)
    # The last message is usually the final AI message, but tools might be at the end—so search backwards.
    msgs = final["messages"]
    for m in reversed(msgs):
        if getattr(m, "type", "") == "ai":
            print("\n---\n")
            print(m.content)
            break


if __name__ == "__main__":
    main()