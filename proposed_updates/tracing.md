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
   - Require strict JSON schema (or equivalent structured output) for both `agent_node` and `reflect_node`.
   - Treat empty/non-JSON outputs as controlled parse errors with bounded retry behavior.

3. **Fix routing precedence to prevent loops**
   - **Status: Completed**
   - In `route_after_agent`, check terminal condition (`state.done` or `final_answer`) before tool-call routing.
   - Prevent accidental tool-call formatting from re-entering the tool loop after a complete answer is available.

4. **Unify completion ownership**
   - **Status: Not started**
   - Choose one owner for termination logic:
     - Option A: `agent` sets completion and `reflect` is removed.
     - Option B: `reflect` sets completion and `agent` prompt no longer says to set `done=true`.
   - Remove mixed responsibility between agent prompt and reflect node.

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
   - **Status: Partial**
   - Emit explicit termination metadata in trace events (for example: `completed`, `max_steps`, `no_progress`, `parse_error_budget`).
   - Make `run_end` include the final termination reason for fast diagnostics.

## Suggested Rollout Order
1. Graph termination contract (`done` state + routing precedence + ownership).
2. Structured output enforcement and parse/error handling.
3. Trace accuracy update (`node_end` post-state snapshot).
4. Loop safeguards (`no_progress` + repeated calls).
5. Tooling alignment (ephemeris integration) for intent-correct factual answers.
