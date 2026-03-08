# Proposed Fixes for Branch: tracing

## Summary
These updates address the functional issues identified in the `3395f7b1faed` trace and the current graph intent/behavior mismatch.

## Proposed Fixes

1. **Make completion explicit in graph state**
   - **Status: Completed**
   - `done: bool`, `final_answer: str`, and `termination_reason: str` are first-class fields in `AgentState`.
   - All termination paths write these explicitly; no node relies on free-text markers in message content.

2. **Enforce structured outputs for decision nodes**
   - **Status: Completed**
   - `agent_node` uses the `final_answer` pseudo-tool -- the model signals completion via a tool call, not free text. Intercepted before `tool_node` runs.
   - `reflect_node` uses `.with_structured_output(_ReflectDecision)` (Pydantic schema) for a reliable binary done-check.
   - All manual JSON-parsing helpers removed (`_parse_json_object`, `_retry_agent_json`, `_has_user_facing_answer`, `parse_errors` budget).

3. **Fix routing precedence to prevent loops**
   - **Status: Completed**
   - `route_after_agent` checks `done`/`final_answer` before tool-call routing -- a terminal state can never re-enter the tool loop.
   - `route_after_reflect` mirrors the same guard.

4. **Unify completion ownership**
   - **Status: Completed**
   - `agent_node` is the sole owner of termination: it intercepts `final_answer` tool calls and writes `done=True` + `final_answer` + `termination_reason="completed"`.
   - `reflect_node` is advisory only -- it sets `done=True` as a signal for the next agent turn to call `final_answer`; it does not skip to END on its own.
   - `final_answer` is in `TOOLS` (schema visible to LLM) but excluded from `TRACED_TOOLS` (never executed by `tool_node`).

5. **Add no-progress and repetition safeguards**
   - **Status: Completed**
   - After the `final_answer` intercept in `agent_node`, all proposed tool calls are compared against every prior `AIMessage.tool_calls` in the message history.
   - If every call is a duplicate (same tool name + identical JSON-serialised args), the agent terminates immediately with `termination_reason="repeated_tool_call"` and a fallback answer.
   - The `max_steps` hard ceiling remains as a secondary backstop.

6. **Correct trace payload accuracy**
   - **Status: Completed**
   - `node_end` in `tracing/decorator.py` now logs a post-execution state snapshot (`_state_snapshot(state, result=result)`), not the pre-node snapshot.
   - `final_answer`, `termination_reason`, `done`, `step`, `plan`, `user_goal`, and `elapsed_ms` are all captured.

7. **Align tools to user-query intent**
   - **Status: Dropped -- out of scope**
   - The original concern (Sun-Moon distance query) was specific to one demo trace, not a framework-level issue.
   - Adding domain-specific tools (e.g. ephemeris) belongs to a separate feature branch, not this tracing/termination work.

8. **Clarify `echo_json` tool scope**
   - **Status: Completed**
   - Removed `echo_json` from `TOOLS`, `TRACED_TOOLS`, and `tools.py` entirely -- it was demo scaffolding with no real usage.
   - Removed the system prompt guard that was only needed because the tool was visible to the model.

9. **Improve termination observability**
   - **Status: Completed**
   - Every exit path sets `termination_reason`: `"completed"` (via `final_answer` intercept) and `"max_steps"` (via step-budget check in both `agent_node` and `reflect_node`).
   - `node_end` trace events carry these fields in the post-execution snapshot.
   - `run_end` inherits them from the final emitted state.

## Status Summary
| # | Item | Status |
|---|------|--------|
| 1 | Explicit completion state | Completed |
| 2 | Structured outputs | Completed |
| 3 | Routing precedence | Completed |
| 4 | Completion ownership | Completed |
| 5 | No-progress safeguards | Completed |
| 6 | Trace payload accuracy | Completed |
| 7 | Ephemeris tooling | Dropped |
| 8 | `echo_json` scope | Completed |
| 9 | Termination observability | Completed |
