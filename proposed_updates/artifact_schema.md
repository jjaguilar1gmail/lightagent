# Proposed Update: Minimal Generic Artifact Schema

## Summary

This document defines the smallest generic step-artifact contract for the agent.

The goal is to make step progression deterministic without baking problem-specific knowledge into the graph. The graph should validate that an artifact is the right *kind of thing* for the current step, not that the artifact is factually correct for a specific domain.

## Design Goals

1. Keep the schema generic across tasks.
2. Make invalid control-flow states hard to represent.
3. Avoid domain-specific validators such as `bus_volume must be between X and Y`.
4. Preserve enough provenance to debug how a step result was produced.
5. Keep the interface small enough that the model can reliably use it.

## Minimal Artifact Schema

Each plan step should declare:

```json
{
  "text": "Determine the volume of a standard school bus in cubic feet",
  "artifact_name": "bus_volume",
  "artifact_description": "The volume of the school bus in cubic feet",
  "artifact_kind": "numeric_quantity"
}
```

Each recorded artifact should use the same `artifact_name` and conform to this generic shape:

```json
{
  "artifact_name": "bus_volume",
  "artifact_kind": "numeric_quantity",
  "value": "4061.25",
  "units": "cubic_feet",
  "provenance_kind": "tool_output",
  "evidence": "calc returned 4061.25 from 45 * 9.5 * 9.5"
}
```

## Field Definitions

- `artifact_name`
  - Stable identifier for the expected output of the current step.
  - Must exactly match the current plan step.

- `artifact_kind`
  - Broad category of artifact content.
  - This is for generic validation only, not domain truth.

- `value`
  - The artifact payload itself.
  - Can remain a string at the tool boundary if needed for simplicity.

- `units`
  - Optional in principle, but required for quantity-like artifacts.
  - Should be omitted or set to `null` for non-quantity artifacts.

- `provenance_kind`
  - How this artifact was produced.
  - Recommended values:
    - `tool_output`
    - `explicit_assumption`
    - `derived_from_prior_artifacts`
    - `conversation_fact`

- `evidence`
  - Short explanation of the support for the artifact.
  - Should reference the latest tool result, stated assumption, or prior artifact.

## Minimal Generic Artifact Kinds

The initial set should stay small:

1. `numeric_quantity`
   - A scalar with meaningful units.
   - Examples: volume, distance, cost, count estimate.

2. `fact_set`
   - Small structured factual bundle.
   - Examples: dimensions, lookup results, named attributes.

3. `assumption_bundle`
   - Explicit assumptions that later steps depend on.
   - Examples: packing factor, occupancy assumption, chosen approximation.

4. `derived_estimate`
   - Result of combining earlier artifacts.
   - Examples: estimated total count, weighted score.

5. `final_response`
   - User-facing final answer.

## What The Graph Should Validate

The graph should enforce only control-safe, generic invariants:

1. `artifact_name` must match the current step's expected artifact.
2. `artifact_kind` must match the current step's expected artifact kind.
3. Required fields must be present.
4. `units` must be present for `numeric_quantity`.
5. `provenance_kind` must be present.
6. `evidence` must be non-empty.
7. A step cannot record a later-step artifact.
8. `final_answer` cannot be emitted until all required artifacts exist.
9. `tool_output` provenance should correspond to an actual successful tool result from the current step.
10. `derived_from_prior_artifacts` should correspond to at least one previously recorded artifact.

## What The Graph Should Not Validate

The graph should not hardcode domain truth such as:

1. Acceptable numeric ranges for a particular concept.
2. Problem-specific formulas.
3. Whether a bus is realistically a certain size.
4. Whether a golf ball volume is physically correct.

Those checks are either:

1. Tool-level responsibilities.
2. Model reasoning responsibilities.
3. Future domain-specific extensions, if intentionally added.

## Why This Boundary Matters

This keeps the architecture generic.

The graph is responsible for answering:

- Did the agent produce the expected type of intermediate result for this step?
- Is that result grounded enough to hand off to later steps?

The graph is not responsible for answering:

- Is this bus volume realistic?
- Is this the best formula for a sphere?

That split avoids turning the core workflow into a collection of hidden task-specific heuristics.

## Anticipated Risks

1. The schema may become too rigid.
   - Mitigation: keep artifact kinds broad and avoid domain enums.

2. The model may produce well-formed but wrong artifacts.
   - Mitigation: require provenance and keep tool outputs visible in traces.

3. The model may still try to record downstream conclusions too early.
   - Mitigation: reject mismatched `artifact_name` and `artifact_kind` for the current step.

4. Prompt/schema drift may create inconsistent behavior.
   - Mitigation: make planner output, step prompt, record tool, and trace display all read from the same plan-step structure.

5. Context may bloat over long runs.
   - Mitigation: pass compact artifact summaries rather than full conversational history when possible.

## Recommended First Implementation

The first implementation should add only these requirements:

1. Plan steps carry `text`, `artifact_name`, `artifact_description`, and `artifact_kind`.
2. `record_step_result` accepts:
   - `artifact_name`
   - `artifact_kind`
   - `value`
   - `units`
   - `provenance_kind`
   - `evidence`
3. The graph validates name/kind/required fields before advancing.
4. Traces display recorded artifacts compactly.

That is enough to make step progression meaningfully safer without prematurely overengineering the system.

## Next Generic Grounding Pass

Without adding domain-specific validators, the graph can still strengthen provenance by:

1. Tracking tool outputs generated during the current step.
2. Rejecting `tool_output` provenance when no successful current-step tool output exists.
3. Rejecting non-`tool_output` provenance when the current step already has a successful tool output.
4. Rejecting `derived_from_prior_artifacts` when no prior artifacts exist.
5. Storing a compact provenance detail string alongside each recorded artifact for debugging and trace review.

## Next Generic Calculation Boundary

For calculation-heavy tasks, the next generic boundary is to separate:

1. Free-form local math.
2. Calculations that claim to derive from previously recorded artifacts.

The graph can enforce this without domain-specific truth checks by:

1. Keeping `calc` available for local standalone math.
2. Adding a `grounded_calc(expression, bindings)` tool for calculations that depend on prior numeric artifacts.
3. Requiring `grounded_calc` to bind prior artifact names explicitly.
4. Rejecting a derived-step `calc` call when prior numeric artifacts already exist and the model should be computing from them.
5. Validating that `grounded_calc` bindings match prior numeric artifact names and values.

This does not make the math semantically correct, but it does make the dependency chain explicit and inspectable.