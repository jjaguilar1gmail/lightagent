# Proposed Update: Post-Run Evaluator

## Summary

This document proposes a separate post-run evaluator that executes after the main ReAct graph finishes.

The evaluator does not replace the current graph. It judges whether the final answer was sufficiently responsive and supported, whether the agent failed to use an existing tool it already had, or whether the run exposed a missing tool capability that should be added later.

The intended first version is advisory only. It should diagnose the run and emit a structured result, but it should not yet re-enter the graph or override the final answer automatically.

## Why Add It

The current graph already handles:

1. Producing an answer.
2. Calling tools when the model decides they are useful.
3. Blocking a few weak completion modes through `evaluate_completion`.

What it does not currently do is make a strong retrospective judgment about:

1. Whether the user's question was actually answered.
2. Whether the answer was sufficiently supported by the information gathered.
3. Whether an existing tool should have been used but was not.
4. Whether a missing tool capability was the real blocker.

That diagnostic layer is a separate concern from the execution loop, so it should be implemented as a separate module rather than folded into the current completion policy.

## Design Goals

1. Keep the main graph focused on solving the task.
2. Add a clear post-run judgment stage without changing the existing ReAct loop semantics.
3. Distinguish missing capability from poor use of existing capability.
4. Produce structured output that is easy to trace, test, and extend.
5. Start advisory-first so the feature improves observability before it affects control flow.

## Non-Goals

1. Do not turn the evaluator into a second autonomous tool-using agent in the first version.
2. Do not automatically loop the run back into the graph in the first version.
3. Do not require perfect semantic truth-checking of the final answer.
4. Do not add filesystem auto-discovery of tools.

## Proposed Separation

The repo should have two distinct responsibilities:

1. `graph.py`
   - Solve the task with the current ReAct-style agent loop.

2. `evaluator.py`
   - Judge the completed run and emit a structured evaluation result.

3. `run_cli.py`
   - Orchestrate `run graph -> run evaluator -> print final answer + evaluation summary`.

This is a good separation because the graph is generative and action-oriented, while the evaluator is retrospective and diagnostic.

## Evaluator Inputs

The evaluator should receive the smallest complete snapshot needed to judge the run:

1. Original user goal.
2. Final answer text.
3. Full message history.
4. Available tool names.
5. Actual tool calls made during the run.
6. Tool outcomes, including successes and errors.
7. Termination reason from graph state.

The evaluator should not need direct access to tracing sinks or external storage in the first version.

## Proposed Output Shape

The evaluator should return a typed result, not free text.

Example shape:

```json
{
  "answered": true,
  "support_level": "sufficient",
  "outcome": "answered_with_sufficient_support",
  "reason": "The final answer is responsive to the user question and is consistent with the evidence gathered during the run.",
  "retry_with_existing_tools": false,
  "missing_capability": false,
  "recommended_tool": null,
  "tool_gap_summary": null
}
```

When the evaluator judges that a missing tool capability is the blocker:

```json
{
  "answered": false,
  "support_level": "insufficient",
  "outcome": "not_answered_because_missing_tool_capability",
  "reason": "The question requires external factual lookup, but no retrieval tool is available.",
  "retry_with_existing_tools": false,
  "missing_capability": true,
  "recommended_tool": {
    "name": "web_search",
    "purpose": "Retrieve current external factual information",
    "inputs": ["query"],
    "outputs": "text results with source metadata"
  },
  "tool_gap_summary": "No external lookup capability exists in the current tool set."
}
```

## Core Outcome Categories

The evaluator should classify runs into one of these buckets:

1. `answered_with_sufficient_support`
   - The user was answered well enough for the current system.

2. `not_answered_but_existing_tool_should_have_been_used`
   - The system likely had enough capability already, but the graph or model failed to use it correctly.

3. `not_answered_because_missing_tool_capability`
   - The available tool set could not reasonably supply the missing information.

4. `not_answered_due_to_ambiguous_or_unanswerable_request`
   - The user request was genuinely under-specified, internally ambiguous, or impossible to answer within the system boundary.

These categories matter because they point to different fixes.

## Decision Criteria

The evaluator should consider the following questions in order:

1. Was the final answer materially responsive to the user's actual question?
2. Was the answer supported by the information gathered in the run?
3. If support was weak, did the system already have a tool that could likely have closed the gap?
4. If no existing tool could plausibly close the gap, does the failure imply a missing capability?
5. If neither tools nor missing capability are the issue, is the request ambiguous or unanswerable as posed?

This ordering reduces the chance that the evaluator blames missing tools for what was really a poor answer or a missed existing tool.

## First-Version Architecture

The evaluator should be implemented as a plain module, not a second graph.

Recommended shape:

1. A small `EvaluatorInput` typed structure.
2. A small `EvaluatorResult` typed structure.
3. A function such as `evaluate_run(input: EvaluatorInput) -> EvaluatorResult`.
4. A separate evaluator prompt, likely with structured output.

This keeps the separation clean without introducing extra routing complexity.

## Why Not Make It a Separate Graph Yet

A second graph is only justified if the evaluator itself needs:

1. Multiple internal decision steps.
2. Retries or backtracking.
3. Its own tool use.
4. Branching control flow that materially affects execution.

The initial evaluator does not need any of that. It only needs to judge a completed run and return a structured result.

Promoting it to a graph can come later if the system starts routing based on evaluator feedback.

## Integration Plan

### Phase 1: Advisory Evaluator

1. Add `app/evaluator.py`.
2. Build evaluator input from final graph state.
3. Run the evaluator after the main graph completes.
4. Print or trace the result in a compact form.
5. Do not alter graph control flow.

### Phase 2: Trace and Test Hardening

1. Record evaluator results in traces.
2. Add unit tests for evaluator classification cases.
3. Add fixtures for representative runs.
4. Ensure evaluator output remains stable and structured.

### Phase 3: Optional Control-Flow Integration

Only after the advisory path is useful in practice:

1. Route back into the graph when the evaluator says an existing tool should have been used.
2. Terminate with a structured tool-gap artifact when the evaluator says a capability is missing.
3. Preserve the original answer while exposing evaluator disagreement explicitly.

## Recommended Prompting Boundary

The evaluator prompt should not ask only:

1. Was the question answered?
2. Do we need a new tool?

It should ask:

1. Was the answer materially responsive?
2. Was it supported by gathered evidence?
3. If not, was the failure due to poor use of existing tools, missing capability, or genuine ambiguity?

That framing is more reliable and less likely to over-diagnose tool gaps.

## Risks

1. The evaluator may over-attribute failures to missing tools.
   - Mitigation: explicitly require a check for existing-tool sufficiency before suggesting new tools.

2. The evaluator may disagree with the graph in noisy ways.
   - Mitigation: keep the first version advisory and make disagreements visible in traces.

3. The evaluator may produce unstable free-form suggestions.
   - Mitigation: force structured outputs and keep `recommended_tool` small and typed.

4. The evaluator may become a hidden second policy layer.
   - Mitigation: keep graph completion policy and post-run diagnosis separate in code and naming.

## Testing Plan

The evaluator should have explicit tests for at least these cases:

1. The answer is adequate and supported by a successful tool result.
2. The answer is inadequate, but an existing tool clearly could have been used.
3. The answer is inadequate because the required capability does not exist in the current tool set.
4. The user request is too ambiguous or inherently underdetermined.
5. The graph terminated on `max_steps`, but the evaluator still distinguishes between tool misuse and missing capability.

## Open Questions

1. Should the evaluator use the same model as the main graph, or a cheaper/smaller one?
2. Should the evaluator inspect the full message history or a compacted summary of tool actions and outputs?
3. Should evaluator recommendations become user-visible by default in the CLI, or only in debug mode?
4. At what confidence threshold should evaluator feedback be allowed to re-enter graph control flow?

## Recommended First Implementation

The first implementation should do only this:

1. Add a separate evaluator module.
2. Run it after the graph completes.
3. Return a structured diagnosis with the four outcome categories above.
4. Include optional `recommended_tool` only when the evaluator judges that a missing capability is the blocker.
5. Surface the result in traces and optionally in CLI output.

That is enough to improve observability and tool-gap discovery without overengineering the current system.