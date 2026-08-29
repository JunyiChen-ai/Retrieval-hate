# C04 Refined Proposal V4 — Final Normative Overlay

**Status:** `FROZEN / PENDING FRESH INDEPENDENT REVIEW`  
**Bases:** `C04_REFINED_PROPOSAL_V2.md`,
`C04_REFINED_PROPOSAL_V3.md`  
**User contract:** unchanged  
**Execution authority:** none

Only the clauses below supersede V3.

## State-artifact falsifier

The V3 `(content_f,state_f)` producer remains, but fallback controls have the
following scientific interpretation:

- `STATE_ONLY`: replace every slot content by literal `NO_CONTENT_f` and retain
  its true four-state token. This is the reliability-mask artifact model.
- `STATE_BLIND`: retain every slot content but remove the state token completely.
- `FALLBACK_COLLAPSE`: retain content and replace **all four** state values by
  the one literal `<fallback=unavailable>`; no stable/single/conflict/missing
  distinction survives.
- `SHUFFLE_FALLBACK_MASK` and `NOISE_FALLBACK_MASK` retain their V3 producer and
  split-fitting algorithms, but are equivalence/sensitivity controls rather than
  mechanisms expected to lower performance.

C04 is supported only if state-only cannot explain FULL and state-blind retains
the FULL gain. Collapse/shuffle/noise must be practically equivalent to FULL;
a large change in either direction indicates reliability-mask dependence and
kills the mechanism claim. Exact gates and applicability are in the V4 plan.
If a sealed dataset/slot has exactly one observed state, there is no state
distinction to corrupt: collapse/shuffle/noise are exact no-ops whose serialized
targets must match FULL byte-for-byte.

## Prompt failure taxonomy

A malformed/schema-invalid **teacher response** is one invalid prompt form. It
may produce `single_valid` when the other form is valid or `missing` when neither
is valid. It does not individually HALT or KILL.

Only the already frozen completed-cache rates decide teacher semantic
reliability:

- per-slot `stable+single_valid >=85%`;
- per-slot `missing <=10%`;
- per-slot `conflict <=20%`;
- joint all-four usable coverage `>=60%`;
- no canonical non-fallback value frequency `>90%`.

Crossing any threshold after a valid complete cache is
`KILL_C04_TEACHER_SEMANTIC_RELIABILITY`, with no rewrite/retry fork. Parser
implementation failure on frozen conformance fixtures, hash/schema mismatch,
crash or incomplete rows remains infrastructure HALT. V3's undefined
“confidence/agreement sensitivity/calibration threshold” language is deleted.
Confidence affects only the frozen state construction; A/B disagreement affects
only `conflict`. V2's explicitly numeric OOF Brier/ECE small gate remains a
separate downstream scientific gate.
