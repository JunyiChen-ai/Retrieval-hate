# C04 Problem Anchor — Source–Proposition–Stance–Harm Tensor

**Status:** `FROZEN FOR INDEPENDENT DESIGN REVIEW`  
**Date:** 2026-07-29 (Pacific/Auckland)  
**Scope:** design and read-only audit only

## Immutable problem

Improve the strongest same-protocol RGCL/RA-HMD hateful-video detector by at
least `+0.030` absolute accuracy and `+0.030` absolute macro-F1 on at least two
datasets, with paired seeds `0/1/2`, 3/3 positive deltas and corrected confidence
bounds above zero. Final inference remains one native full-video representation
per video followed by the ordinary train-memory top-20 kNN rule.

## Bottleneck

The current representation frequently encodes a harmful surface proposition but
does not bind it to:

1. who introduced the proposition;
2. what proposition is being presented;
3. whether the current presenter endorses, rejects, reports or performs it; and
4. what protected-target harm act the proposition realizes.

This is a role-binding problem, not a request for a scalar hate score. It is
motivated by stable errors, but it must improve aggregate held-out performance
rather than route only a post-hoc error cluster.

## Evidence that fixes the scope

- HateMM has five 3/3-seed stable slur-bearing false positives involving
  quotation, archive footage, lyrics or documentary framing. Their label depends
  on presenter stance, but their break-free ceiling is only `+0.0233` accuracy.
- MHC-EN has five lexical-surface false positives and two counter-speech /
  meta-commentary false positives. Its archive `mechanism` field agrees with the
  wrong model on the lexical cases, so that field cannot be treated as a fix.
- MHC-ZH has five topic-versus-stance false positives, but the enrichment is not
  significant (`p=0.5022`). ZH therefore cannot be declared a stance win from
  the forensic taxonomy alone.
- P4 showed that independently predicting archive fields is decodable and
  label-correlated yet does not improve final performance. C04 must demonstrate
  value from the *joint role interaction*, beyond all factor marginals and
  lower-order interactions.

## Hard supervision and data constraints

- `parent_video_binary_label` is the only gold supervision.
- Source, proposition, stance, target, harm act, rationale, segment and span are
  never gold. Any such teacher output is noisy privileged information.
- The factor prompt and schema are label-blind and frozen before cache creation.
- Any new teacher cache is train-only, local-open-weight and sealed before train
  labels enter.
- No dev/test teacher output, API, OCR, cross-dataset mixing, per-item router,
  sample selector, scalar verdict fusion or test-time MLLM is permitted.
- Test remains untouched until one final frozen lineage.

## One dominant contribution and at most two claims

**Dominant contribution:** a nonseparable four-role tensor is used as a
train-only privileged *dense representation target*. A native tensor student
internalizes this target into the single full-video memory embedding; teacher
outputs and factor files are absent at inference.

- **C1 — role-binding information:** the joint source–proposition–stance–harm
  interaction provides conditional information that the native representation,
  individual factors and all lower-order factor combinations do not.
- **C2 — internalized memory geometry:** a native tensor student can retain that
  information under teacher-free ordinary kNN inference, and the gain disappears
  under REMOVE, SHUFFLE and NOISE controls.

No claim is made that quotation detection, stance detection, target
disentanglement, tensor products, distillation or kNN retrieval is individually
new.

## Success and stop conditions

C04 survives only through the serial gates in `C04_EXPERIMENT_PLAN.md`. In
particular, the current registry requires an existing-bank Stage-0 before any
new teacher or GPU spend. Failure of that gate kills C04 under the current
contract. Bypassing it requires an explicit user-approved registry amendment;
the design team may not silently reinterpret the gate.

