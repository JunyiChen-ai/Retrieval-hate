# Review Summary: LB-SCGP Global

## Status

- Final reviewer result: READY.
- Final score: 9.1 overall.
- Drift: NONE.
- Experiment-plan readiness: READY FOR EXPERIMENT-PLAN.
- Experimental status: unvalidated. No experiment, implementation, or SLURM job has been run for this global pivot.

## Problem Anchor

hateful video detection adapting RGCL/RA-HMD to video; MLLM meaningful+novel; final MHC-EN/MHC-ZH seeds0/1/2 vs strongest same-protocol non-MLLM, acc and macro-F1 each ≥+0.030, all paired seed deltas positive, hierarchical paired bootstrap lower>0, Holm; only parent-video binary gold, no segment/timestamp/span/localization/stance/target/mechanism/rationale gold; train-only label-blind MLLM cache; test ordinary full-video train-memory top20 kNN, no teacher/head/rerank; SLURM; no sample weighting/key selection/pair-triplet/SupCon/segment route; REMOVE/SHUFFLE/NOISE/direct attribution. Local rank-cell v7 formally retired, no v8.

Absolutely do not assume any fragment/segment has gold annotation. The only gold is parent_video_binary_label. Any segment/timestamp/span/localization/stance/target/mechanism/rationale output is not gold and may not be treated as supervision, pseudo-groups, selection, or evaluation gold. Preserve this literally and operationally.

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict | Drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 8.5 | 6.5 | 7.0 | 7.5 | 6.0 | 8.0 | 6.5 | 7.1 | REVISE | NONE |
| 2 | 9.5 | 8.2 | 8.1 | 8.3 | 7.2 | 8.7 | 7.7 | 8.3 | REVISE | NONE |
| 3 | 9.8 | 9.1 | 9.0 | 9.0 | 8.6 | 9.2 | 8.9 | 9.1 | READY | NONE |

## Round 1 Resolution

Reviewer verdict: REVISE, overall 7.1.

Main issues:

- Encoder realizability was not closed because `rank(G*)` could exceed `d`.
- Cross-basis `vech` and separate replica SVD coordinate comparisons were not dimensionally valid.
- Robust intervals from row trust were likely vacuous.
- Solver replay traces were not enough; a normal-cone/KKT certificate was needed.
- Slacked structural equalities/bands were too close to hard-constraint language.
- Stance-like schema names risked gold-boundary confusion.

Author resolution in Round 1:

- Accepted the rank blocker and added a fail-closed `rank_eps(G*)<=d` gate rather than a nonconvex rank constraint or truncation overclaim.
- Accepted the moment-interface blocker and replaced cross-basis comparisons with one common consensus basis `Q`; `vech` is used only for square symmetric `M_Q(G)`.
- Accepted robust-interval critique and added a prospective coordinate trust radius `rho_coord` plus fail-open geometry/no safety claim when coverage is low.
- Accepted solver critique and specified an H-metric normal-cone/KKT certificate, though Round 2 later required making it the only acceptance path.
- Accepted soft-vs-hard critique and reduced structural matching to a single regularized preference.
- Accepted schema critique and consistently documented certificate atoms as noisy structural observables, never gold.

Remaining after Round 1:

- KKT serialization still needed tightening.
- Rank-tail reporting and compute feasibility were underspecified.
- Robust safety needed to be disabled by default unless coverage passed.

## Round 2 Resolution

Reviewer verdict: REVISE, overall 8.3.

Main issues:

- Solver acceptance had to require a serialized H-metric normal-cone/KKT certificate as the only optimality acceptance path.
- Rank-tail audit needed explicit non-gameable reporting around `eps_rank`.
- Rank-gate failure had to be terminal, with no rescue route.
- Full-bank PSD projection and KKT verification needed concrete resource-envelope analysis.
- Robust safety needed to be subordinate and off by default until coverage passed.
- Scalar difficulty/direct-feature discussion needed to be framed as attribution control, not an impossibility theorem.

Author resolution in Round 2:

- Accepted KKT blocker and made serialized normal-cone/KKT payload the only solver acceptance path; finite VI probes and traces are diagnostics only.
- Accepted rank-tail blocker and required reporting `lambda_d`, `lambda_{d+1}`, `rank_eps`, positive eigenmass, omitted positive tail, tail ratio, negative mass, `lambda_min`, and exact reconstruction residual.
- Accepted terminal rank-failure rule: `ENCODER_RANK_GATE_FAIL` has no prompt/schema/tolerance/teacher/epoch/scale/truncation/adapter/nonconvex rescue.
- Accepted compute feasibility issue and added dense storage, eigensolver, operator, checkpoint, CPU/GPU, memory cap, and STOP conditions for realistic MHC train sizes.
- Accepted robust safety simplification and made robust constraints disabled by default unless a prospective G0 coverage gate passes.
- Accepted scalar/direct wording issue and made direct/scalar controls decisive attribution tests.

Remaining after Round 2:

- Minor implementation hygiene only: PSD normal sign convention, manifest-derived `N`/`d`, robust coverage replay, and exact KKT payload carry-over into experiment planning.

## Round 3 Resolution

Reviewer verdict: READY, overall 9.1.

Reviewer conclusion:

- Anchor and no-fragment-gold boundary preserved.
- Solver, rank-tail, compute, robust default-off, scalar/direct controls, common `Q`, valid `vech`, coordinate trust, exact factor/Procrustes, and uniform fit were resolved.
- Drift warning: NONE.
- Method has one focused contribution and stays within complexity caps.
- Remaining items are implementation-spec hygiene, not conceptual blockers.

Remaining minor items for experiment-plan handoff:

- Spell out the PSD normal sign convention explicitly in the certificate schema.
- Verify `N` and `d` from fold manifests during preflight rather than relying on prose estimates.
- Keep robust constraints disabled unless the coverage report passes and is replayed.
- Carry the exact KKT payload schema into the experiment-plan handoff so solver traces cannot replace it.

## Final Method Evolution

The method evolved from a broad global-pivot proposal into a certifiable encoder-realizable global geometry interface:

```text
train-only label-blind structural certificates
  -> common-basis structural moment M_Q(G)
  -> one closed strongly convex PSD/unit-diag projection
  -> KKT-only solver certificate
  -> rank-tail gate and exact factor/Procrustes Z*
  -> uniform encoder fit
  -> ordinary full-video train-memory top20 kNN
```

Reusable evidence:

- supervision isolation and no-segment-gold discipline;
- exact train-memory top20 kNN endpoint definitions;
- PSD/unit-diagonal/projector/hash/replay discipline;
- prior negative evidence for direct RA-HMD/video MLLM routes;
- local v7 retirement and pivot authority.

Retired evidence/claims:

- local signed-gap/rank-cell stationarity;
- SLSQP near-miss as success;
- `NO_WITNESS` as infeasibility;
- local v8;
- robust intervals as a central safety claim when coverage is low.

## Final Validation Shape

The final validation remains three blocks:

1. Conceptual G0 plus real-fold and teacher-cache gates.
2. Mechanism attribution against REMOVE, SHUFFLE, NOISE, DIRECT-MOMENT, DIRECT-CERT-FEATURE, and SCALAR-PROPENSITY.
3. Final paired performance gate on MHC-EN and MHC-ZH, seeds 0/1/2, strongest same-protocol non-MLLM comparator, ordinary top20 kNN, accuracy and macro-F1 each ≥+0.030, all paired seed deltas positive, hierarchical paired bootstrap lower bound >0, and Holm correction.

This is readiness for experiment planning, not evidence of experimental success.
