# C04 Experiment Plan V4 — Final Normative Overlay

**Status:** `FROZEN / PENDING FRESH INDEPENDENT REVIEW`  
**Bases:** `C04_EXPERIMENT_PLAN_V2.md`, `C04_EXPERIMENT_PLAN_V3.md`  
**Execution authority:** none

Only the clauses below supersede V3.

## Fallback anti-artifact gate

In both small and full seed-0 nested OOF, separately on both datasets:

1. `FULL - STATE_ONLY >=+0.020` accuracy and macro-F1, with paired-bootstrap
   95% lower bounds `>0`.
2. For each metric with positive `Delta_FULL = FULL-BASE`,
   `Delta_STATE_BLIND / Delta_FULL >=0.80`, and the paired-bootstrap 95% CI of
   `STATE_BLIND-FULL` must lie wholly inside `[-0.010,+0.010]`.
3. For FALLBACK_COLLAPSE and every applicable mask SHUFFLE/NOISE comparison, the
   paired-bootstrap 95% CI of `CONTROL-FULL` for accuracy and macro-F1 must lie
   wholly inside `[-0.010,+0.010]`. A difference outside that equivalence margin
   in either direction is `KILL_C04_FALLBACK_MASK_ARTIFACT`.

These three fallback controls are removed from V3's “FULL must exceed by
`+.020`” list. Tuple/slot/role/matched-feature-noise and all REMOVE arms retain
that V3 `+.020` plus CI-lower-`>0` requirement.

Applicability is frozen per dataset/slot over the sealed evaluation bank:

- if all rows have the same state, shuffle/noise/collapse must be byte-identical
  to FULL and are recorded `NA_DEGENERATE_EXACT`; this satisfies only the
  sensitivity clause, while STATE_ONLY/STATE_BLIND remain mandatory;
- if at least two states occur but any nonempty state has fewer than 10 rows in
  the 200-ID tranche or 20 rows in the full bank, shuffle/noise are
  `NA_LOW_SUPPORT` with counts reported; collapse remains applicable, and
  STATE_ONLY/STATE_BLIND remain mandatory;
- otherwise all three sensitivity controls are applicable.

No `NA` waives STATE_ONLY, STATE_BLIND, BASE, structural or full-bank gates.

## Equal tuning/search budget

V3's complete eight-point grid, independent per-arm selection, five-outer
aggregation, checkpoint budget and tie rule additionally bind
`STANCE_ONLY`, `HARM_ONLY` and `CAPACITY_ONLY_NATIVE`.

- STANCE_ONLY/HARM_ONLY use `lambda_slot` for their retained named slot,
  `lambda_joint` for its 257-D retained composition target, and `beta` for branch
  scale; all eight tuples are semantically active.
- CAPACITY_ONLY_NATIVE has no teacher losses: both lambdas multiply exact zero by
  construction. It still executes all eight registered tuples with identical
  initialization/data order/steps; the four duplicates at each beta must be
  byte-identical or HALT. The normal V3 tie rule then selects lower lambdas.

Thus every mandatory retained control receives eight trials, the same optimizer
steps/checkpoint rule and the same selection opportunity; none borrows FULL's
winner or receives a reduced search.

## Unified semantic-reliability decision

Malformed/missing teacher forms contribute only to the five frozen numeric rate
gates in the V4 proposal. No single record kills a tranche. Undefined
confidence/agreement sensitivity KILL criteria are absent. The explicit
downstream Brier/ECE thresholds from V2 remain unchanged.
