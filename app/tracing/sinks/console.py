from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from ..events import TraceEvent

# ---------------------------------------------------------------------------
# Narrative lookup — maps (event_type, node_name) → human-readable string.
# node_name=None means "match any node".
# ---------------------------------------------------------------------------

def _fmt_args(args: Dict[str, Any]) -> str:
    if not args:
        return ""
    parts = [f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:4]]
    return ", ".join(parts)


def _narrative(event: TraceEvent) -> Optional[str]:
    """Return a Rich-markup string for pretty mode, or None to suppress."""
    t = event.type
    n = event.node
    p = event.payload

    # ── run-level ──────────────────────────────────────────────────────────
    if t == "run_start":
        return f"[bold cyan]▶  Run started[/bold cyan]  [dim]{p.get('run_id', '')}[/dim]"

    if t == "run_end":
        elapsed = p.get("elapsed_ms")
        ms = f" [dim]({elapsed}ms)[/dim]" if elapsed else ""
        if p.get("status") == "error":
            err = p.get("error", "")
            return f"[bold red]✗  Run failed{ms}  {err}[/bold red]"
        return f"[bold green]✓  Run complete{ms}[/bold green]"

    # ── node_start: the "what is happening right now" line ─────────────────
    if t == "node_start":
        if n == "planner":
            goal = (p.get("user_goal") or "")[:80]
            suffix = f"  [dim]{goal}[/dim]" if goal else ""
            return f"[yellow]◆  Planning next steps…[/yellow]{suffix}"

        if n == "agent":
            step = p.get("step") or 0
            plan: list = p.get("plan") or []
            # step is pre-increment in the payload (incremented at start of fn)
            step_str = ""
            if plan:
                idx = min(step, len(plan) - 1)
                step_str = f"  [dim]step {step + 1}/{len(plan)}: {plan[idx][:70]}[/dim]"
            elif step is not None:
                step_str = f"  [dim]step {step + 1}[/dim]"
            return f"[yellow]◆  Thinking…[/yellow]{step_str}"

        if n == "reflect":
            return "[yellow]◆  Checking whether we're done…[/yellow]"

        if n == "tools":
            return "[yellow]◆  Executing tools…[/yellow]"

        return f"[yellow]◆  {n}[/yellow]"

    # ── node_end: brief completion annotation ──────────────────────────────
    if t == "node_end":
        elapsed = p.get("elapsed_ms")
        ms = f" {elapsed}ms" if elapsed else ""
        if n == "planner":
            pl = p.get("plan_len", 0)
            return f"  [dim]└ plan ready: {pl} step(s){ms}[/dim]"
        if n == "reflect":
            done = p.get("done")
            return f"  [dim]└ done={done}{ms}[/dim]"
        if n == "agent":
            return f"  [dim]└ agent responded{ms}[/dim]"
        return None  # silent for other node_end by default

    # ── errors ─────────────────────────────────────────────────────────────
    if t == "node_error":
        return f"  [bold red]✗  Error in {n}: {p.get('error', '')}[/bold red]"

    # ── tool calls ─────────────────────────────────────────────────────────
    if t == "tool_call_start":
        tool = p.get("tool") or n or "?"
        args_str = _fmt_args(p.get("args", {}))
        return f"  [cyan]⚙  {tool}({args_str})[/cyan]"

    if t == "tool_call_end":
        result = str(p.get("result", ""))[:100]
        elapsed = p.get("elapsed_ms")
        ms = f" [dim]{elapsed}ms[/dim]" if elapsed else ""
        return f"  [dim]   → {result}{ms}[/dim]"

    if t == "tool_error":
        return f"  [bold red]   ✗ {p.get('error', '')}[/bold red]"

    # ── routing ────────────────────────────────────────────────────────────
    if t == "routing_decision":
        dest = p.get("destination", "?")
        return f"  [dim]→ {dest}[/dim]"

    # ── LLM calls: silent in pretty mode, visible in debug ─────────────────
    if t == "llm_call_start":
        return None

    if t == "llm_call_end":
        err = p.get("error")
        if err:
            return f"  [bold red]   LLM ✗ {str(err)[:120]}[/bold red]"
        elapsed = p.get("elapsed_ms")
        if elapsed:
            return f"  [dim]   LLM ← {elapsed}ms[/dim]"
        return None

    return None


_RICH_MARKUP = re.compile(r"\[/?[^\]]+\]")


class ConsoleSink:
    """
    Pretty (default) or debug console sink.

    ``pretty=True``  — narrative summaries with colour via Rich (or plain fallback).
    ``pretty=False`` — raw JSON dump of every event; use for ``--debug`` mode.
    """

    def __init__(self, pretty: bool = True):
        self.pretty = pretty
        self._console = None
        if pretty:
            try:
                from rich.console import Console  # type: ignore
                self._console = Console(highlight=False)
            except ImportError:
                pass  # graceful degradation to plain print

    def handle(self, event: TraceEvent) -> None:
        if not self.pretty:
            print(json.dumps(event.to_dict(), default=str))
            return

        text = _narrative(event)
        if text is None:
            return

        if self._console:
            self._console.print(text)
        else:
            # Strip Rich markup and fall back to plain print
            plain = _RICH_MARKUP.sub("", text)
            print(plain)
