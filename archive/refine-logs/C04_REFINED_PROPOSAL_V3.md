# C04 Refined Proposal V3 — Normative Repair Overlay

**Status:** `FROZEN / PENDING FRESH INDEPENDENT REVIEW`  
**Base:** `C04_REFINED_PROPOSAL_V2.md`  
**User contract:** unchanged  
**Execution authority:** none

This file changes only the clauses below. Every unmentioned V2 clause remains
binding.

## R1 — fallback representation and controls

The sealed producer stores, per slot `f`, `(content_f, state_f)`. `state_f` is
exactly `stable|single_valid|conflict|missing`. Stable/single-valid content
follows V2. Conflict/missing content is literal `NO_CONTENT_f`. The model render
is `content_f || "<fallback=" || state_f || ">"`; numeric confidence remains
audit-only. Thus fallback state is explicit and has isolated controls:

- `FALLBACK_COLLAPSE`: leave content fixed; map `conflict` and `missing` to
  `fallback`, leaving `stable` and `single_valid` distinct. No distribution fit.
- `SHUFFLE_FALLBACK_MASK`: leave content fixed; within each dataset, outer fold
  partition and slot, sort IDs by
  `sha256("C04-FB-SHUFFLE-v3"||dataset||outer||partition||slot||video_id)` and
  rotate only `state_f` by one. Outer-train and outer-held are deranged
  separately; no label or cross-partition donor enters.
- `NOISE_FALLBACK_MASK`: leave content fixed; estimate the four-state empirical
  categorical probabilities separately per slot from outer train only. For
  outer-train and outer-held IDs, draw a state by inverse-CDF from
  `uint64_be(sha256("C04-FB-NOISE-v3"||dataset||outer||partition||slot||video_id)[0:8])/2^64`.
  The held partition never fits a probability.

Student controls corrupt only outer-train supervision. DIRECT held targets use
the registered held corruption, while native-only dev never reads a teacher
field. Expected direction is `FULL - corrupted > 0`; small and full numeric
gates are frozen in the V3 plan. An outer partition of size `<2` is an
infrastructure HALT.

## R2 — complete lower-order control

For every `1<=|A|<=3`, form the V2 257-D `q_A`; concatenate the 14 blocks in
lexicographic order
`S,P,T,H,SP,ST,SH,PT,PH,TH,SPT,SPH,STH,PTH` to
`x_LE3 in R^3598`.

Use a dense block-covered matrix `J in {-1/16,+1/16}^{256 x 3598}`:

```text
J[k,j] = (1/16) * (1 if
  LSB(sha256("C04-LE3-DENSEJL-v3" || uint16_be(k) || uint16_be(j))) == 0
  else -1)
q_LE3 = concat(safe(J x_LE3), [1 if ||J x_LE3||2 <= 1e-12 else 0])
```

Every coordinate of every one-/two-/three-way block contributes to all 256
outputs; this is not coordinate selection. `ADDITIVE` uses the analogous
separately tagged dense Rademacher map from `[u_S;u_P;u_T;u_H]`.
Canonical row-major float32 payloads and exact SHA256 values are mandatory
pre-execution code-review fields. Missing/mismatched payload hashes HALT. Both
controls remain 257-D, fixed, label-blind and are followed by the same
capacity-matched retained branch and tuning budget as FULL.

## R3 — zero-frame case

If the video reports `N=0` decodable frames, all eight inputs are the fixed RGB
black frame and `frame_decode_failed=true`; there is no modulo/index operation,
retry or neighbor substitution. If `N>0`, V2 indices apply and each failed
requested decode is independently black-filled. An all-black/empty-transcript
record remains in the cache and normally becomes missing/conflict; it contributes
to the semantic-reliability gate.

## R4 — failure taxonomy

Infrastructure failures alone HALT: wrong/missing hashes or IDs, access violation,
producer crash, parser/schema implementation failing frozen conformance fixtures,
NaN, fold corruption, namespace collision or resource-ledger failure.

A completed valid producer whose model outputs malformed JSON, missing slots,
low usable/stable coverage, high conflict, degenerate values or failed
confidence/agreement sensitivity is a **scientific**
`KILL_C04_TEACHER_SEMANTIC_RELIABILITY`. It is not repairable by rewriting a
prompt, changing thresholds/model/frames/transcript cap, retrying outputs or
selecting rows. Such a change requires an explicitly authorized new candidate
contract, not a C04 V3 fork.

