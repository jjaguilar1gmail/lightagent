# Proposed Fixes for Branch: tracing

## Summary
These updates address the functional issues identified in the `3395f7b1faed` trace and the current graph intent/behavior mismatch.

## Proposed Fixes

1. **Make completion explicit in graph state**
   - **Status: Completed**
   - Parse structured agent output into a real `done: bool` state field (optionally `final_answer`) inside `agent_node`.
   - Do not rely on free-text markers like `done=true` in message content.

2. **Enforce structured outputs for decision nodes**
   - **Status: Completed**
   - `agent_node` now uses the `final_answer` tool-call mechanism (no text parsing required).
   - `reflect_node` uses `with_structured_output(_ReflectDecision)` for a reliable binary done-check.
   - All JSON-parsing helpers (`_parse_json_object`, `_retry_agent_json`, `_has_user_facing_answer`) removed.

3. **Fix routing precedence to prevent loops**
   - **Status: Completed**
   - In `route_after_agent`, check terminal condition (`state.done` or `final_answer`) before tool-call routing.
   - Prevent accidental tool-call formatting from re-entering the tool loop after a complete answer is available.

4. **Unify completion ownership**
   - **Status: Completed**
   - Chose Option A variant: `agent_node` intercepts the `final_answer` tool call and sets `done=True` + `final_answer` + `termination_reason`.
   - `reflect_node` is a lightweight binary check only (no termination authority on its own).
   - `final_answer` is in `TOOLS` (schema) but excluded from `TRACED_TOOLS` (execution); termination is handled entirely in `agent_node`.

5. **Add no-progress and repetition safeguards**
   - **Status: Not started**
   - Keep `max_steps` but add:
     - repeated-tool-call detection (same tool+args repeated N times),
     - repeated-answer/no-progress detection.
   - End gracefully with a termination reason when progress stalls.

6. **Correct trace payload accuracy**
   - **Status: Completed**
   - Update tracing decorator so `node_end` logs post-execution state/result snapshot, not pre-node snapshot.
   - Ensure trace visibility for fields such as `done`, `step`, and any termination metadata.

7. **Align tools to user-query intent**
   - **Status: Not started**
   - Current tools (`calc`, `now_utc`, `echo_json`) cannot compute a truly time-specific Sun–Moon distance.
   - Add an astronomy/ephemeris-backed tool and require timestamped tool evidence for “today, specifically” queries.

8. **Constrain tool policy to avoid recursive formatting calls**
   - **Status: Partial**
   - Restrict `echo_json` to output formatting only.
   - Prevent `echo_json` from being treated as a reasoning step that can recursively trigger additional tool usage.

9. **Improve termination observability**
   - **Status: Completed**
   - `termination_reason` field is set on every exit path: `completed`, `max_steps`.
   - `node_end` trace events now include `termination_reason` and `final_answer` in the post-execution state snapshot.
   - `run_end` inherits these from the final state (tracked by `TraceEmitter._state`).

## Suggested Rollout Order
1. Graph termination contract (`done` state + routing precedence + ownership).
2. Structured output enforcement and parse/error handling.
3. Trace accuracy update (`node_end` post-state snapshot).
4. Loop safeguards (`no_progress` + repeated calls).
5. Tooling alignment (ephemeris integration) for intent-correct factual answers.
