# LB-SCGP G0 v7 Result-to-Claim Pivot Review

Reviewer thread/session: fresh independent RESULT-TO-CLAIM reviewer in
`/data/jehc223/RGCL`, main Codex session, no subagents.

Boundary: read-only shell inspection plus this one report write. I did not edit
implementation, configs, freezes, artifacts, trackers, plans, proposals, or any
other report. I did not submit, control, monitor, cancel, release, requeue, or
rerun any SLURM job. I did not run Python, imports, model code, numerical
compute, GPU code, training, replay, or solver code.

Evidence read:

- `refine-logs/lb_scgp/G0_V5_RESULT_TO_CLAIM_REVIEW.md`
- `refine-logs/lb_scgp/G0_V6_ACTUAL_RESULT_TO_CLAIM_REVIEW.md`
- `refine-logs/lb_scgp/G0_V7_FINAL_REPAIR_HANDOFF.md`
- `refine-logs/lb_scgp/v7/G0_V7_PREREGISTERED_DESIGN.json`
- `refine-logs/lb_scgp/v7/results/v7_actual_certificate_12896.json`
- `refine-logs/lb_scgp/v7/results/v7_independent_replay_12898.json`
- v7 runtime source needed to validate signed-gap, Phase-I, Phase-II, replay,
  pivot, supervision, and hash gates
- `refine-logs/lb_scgp/FINAL_PROPOSAL.md`
- `refine-logs/lb_scgp/EXPERIMENT_PLAN.md`

## claim_supported

no

## route

pivot

## confidence

high

## intended_claim

The intended v7 claim was that the final authorized local rank-cell repair could
certify the actual G0 oriented fixture under the preregistered signed-cell
contract: strict signed gaps with `tau=1e-7`, fixed `eta=1e-12`, RHS
`tau+eta=1.00001e-7`, replayed Phase-I compatibility for every canonical
compatible cell, and then Phase-II original-objective local stationarity with
independent replay. Passing that gate would have supported continuing the
LB-SCGP local rank-cell stationarity route toward G0. It would still not have
been a performance, MLLM, teacher, validation, test, or segment claim.

## what_results_support

The v7 evidence supports a clean fail-closed pivot trigger.

- The preregistered v7 design fixed `eta=1e-12`, `tau=1e-7`, strict
  `G[q,a]-G[q,b] >= tau+eta`, `topk=20`, Dykstra violation `1e-6`, relative
  change `1e-7`, max independent orientations `8`, and max pivots `32`.
- Static/certificate/replay source hashes match the v7 handoff, and replay
  status is `REPLAY_OK_PIVOT_TRIGGERED`.
- Canonical-cell enumeration was complete for the actual oriented fixture:
  one boundary descriptor `p00: p15 vs p18`, compatible assignments `[-1]` and
  `[1]`, and 528 additive signed-gap rows per cell.
- Both cells kept the target canonical top20 in the selected witness and had
  small original residuals and positive PSD margins. Cell 0 selected
  `max_589_original_residual=1.5452176205243973e-09`,
  `psd_min_eigenvalue=0.004263333136541338`, and
  `top20_equal_cell=true`. Cell 1 selected
  `max_589_original_residual=1.7364167603783898e-09`,
  `psd_min_eigenvalue=0.004697389340520555`, and
  `top20_equal_cell=true`.
- The replay validates the decisive failure: cell 0 signed-gap
  `max_residual=6.736095449260811e-15`, `min_margin=-6.736095449260811e-15`,
  `pass=false`; cell 1 signed-gap
  `max_residual=1.7186069668545097e-14`,
  `min_margin=-1.7186069668545097e-14`, `pass=false`.
- The replay and producer both label the Phase-I cell statuses
  `NO_COMPATIBILITY_WITNESS_NO_FARKAS`, set `incompatibility_claim=false`, and
  state that nonconvergence/no witness is not infeasibility.
- Phase II was empty because no Phase-I strict compatibility witness existed.
  Therefore no original-objective stationarity, VI, PSD/SOC/linear dual, or
  complementarity certificate exists.
- The supervision boundary remains intact:
  `only_gold_supervision=parent_video_binary_label`,
  `segment_gold_exists=false`, `segment_gold_used=false`, and all MLLM, OCR,
  teacher cache, held, validation, and test counters are zero.

Together with v5/v6, the evidence still supports reuse of the broader
engineering discipline: hash-bound artifacts, independent replay, exact kNN
semantics, PSD/unit-diagonal machinery, no-clobber reporting, negative-control
thinking, and strict supervision isolation.

## what_results_dont_support

The results do not support the intended v7 claim. Under the preregistered strict
gate, signed-gap misses of `6.7e-15` and `1.7e-14` are failures, not numerical
successes. v7 did not produce any accepted Phase-I strict signed-cell
compatibility witness and did not reach Phase II.

The results do not support G0 PASS, freeze, real-fold continuation, G1, teacher,
MLLM, OCR, validation, test, accuracy, macro-F1, or any final hateful-video
detection claim. The results also do not prove incompatibility of the target
cells, because no replayed Farkas/conic certificate was produced.

The final authorized solver repair has failed. The project should not continue
with a v8 or further local rank-cell solver pursuit.

## root_causes

1. The main gate was too brittle for the scientific role it was asked to play.
   The local signed rank-cell stationarity route made an exact strict
   cell-boundary certificate the deciding G0 gate. In v7, the selected
   witnesses satisfy the original residual map and preserve target top20, but
   miss the preregistered strict gap by about `1e-14`. Since the design made
   signed-gap pass exact and tolerance-free beyond `tau+eta`, these are hard
   failures.

2. The route became solver/certificate-chasing around a local rank cell rather
   than a robust mechanism test of label-blind structural geometry. v5 exposed
   top20/full-ranking and oriented-boundary issues. v6 exposed strict-cell
   encoding ambiguity and lack of original-objective local stationarity. v7
   repaired the cell semantics but still failed the strict signed-gap
   compatibility gate.

3. The failure is not an infeasibility result. The replay explicitly preserves
   `NO_COMPATIBILITY_WITNESS_NO_FARKAS` as neither compatibility nor
   incompatibility. That distinction matters: the route is retired because this
   was the final authorized repair and the main gate is no longer a productive
   mechanism gate, not because the target cell was mathematically proven empty.

## meaning_of_strict_signed_gap_failure

The v7 signed-gap rule is `G[q,a]-G[q,b] >= tau+eta`, with `tau=1e-7` and
`eta=1e-12`. The v7 source computes residuals as `max(0, rhs-lhs)` and sets
`pass=true` only when `max_residual <= 0.0`, `min_margin >= 0.0`, and all 528
rows are present. There is no replay tolerance that allows negative margins.

Therefore:

- cell 0 `min_margin=-6.736095449260811e-15` is FAIL;
- cell 1 `min_margin=-1.7186069668545097e-14` is FAIL;
- the fact that these values are tiny does not permit retrospective tolerance,
  eta/tau relaxation, expected-status relabeling, or PASS/freeze;
- `top20_equal_cell=true` is not enough, because the preregistered gate was
  strict signed-cell compatibility, not merely canonical top20 equality.

## invalid_or_non_evidence

- `NO_COMPATIBILITY_WITNESS_NO_FARKAS` is not an incompatibility proof and not
  an infeasibility certificate.
- Small original residuals, positive PSD margins, top20 equality, SLSQP
  candidates, and selected witnesses are not strict signed-cell compatibility
  when `signed_gap_summary.pass=false`.
- Phase-I feasibility or near-feasibility is not Phase-II original-objective
  local stationarity.
- Empty Phase II cannot support stationarity, VI, dual feasibility, or
  complementarity.
- v5/v6 partial machinery evidence is not v7 PASS and not G0 GO.
- No current evidence involves real folds, G1 OOF, final accuracy, macro-F1,
  MLLM behavior, OCR behavior, teacher behavior, validation, test, or segment
  modeling.
- Segment gold cannot be inferred. The only gold supervision is the
  parent-video binary label.

## missing_evidence

- No strict signed-cell compatibility witness for either v7 canonical cell.
- No replayed Farkas/conic certificate proving signed-cell incompatibility.
- No Phase-II original-objective certificate with stationarity, VI, PSD/SOC/
  linear dual feasibility, and complementarity.
- No G0 PASS/freeze/decision artifact.
- No real-fold microbenchmark or G1 ordinary-kNN OOF evidence after the v7
  failure.
- No performance evidence on MHC-EN or MHC-ZH.
- No train-only label-blind MLLM structural-certificate evidence in the current
  v7 outcome; all current MLLM/OCR/teacher/cache counters are zero.

## revised_claim

The supported claim is:

`v5-v7 provide replayed, hash-bound evidence that parts of the LB-SCGP
projector, PSD/unit-diagonal, exact-vote, rank-cell diagnostic, isolation, and
fail-closed machinery are executable under parent-video-label-only supervision.
The final authorized v7 strict signed-cell repair did not produce a
compatibility witness, did not produce a Farkas incompatibility certificate, did
not enter Phase II, and independently replayed as pivot-triggered. The local
rank-cell stationarity route is not supported and is retired as the main G0
gate.`

## reusable_old_evidence

Reusable evidence and design assets:

- parent-video-label-only supervision contract and zero segment-gold boundary;
- no held/validation/test access discipline and manifest counters;
- immutable hashes, source-hash replay, payload hashes, no-clobber artifacts,
  and independent verifier/replay pattern;
- PSD/unit-diagonal/full-bank Gram representation;
- exact ordinary top20 full-video kNN endpoint definition;
- projector/KKT/PSD/diag/factor/rollback and exact-vote synthetic evidence
  from v5 where it passed;
- negative controls for over-budget orientation/pivot, unresolved ties,
  duplicate IDs, no-segment, and fail-closed statuses;
- direct-control, shuffle/noise-control, bootstrap/statistical, and moving
  comparator logic from the proposal and experiment plan;
- v7's strict replay discipline as a warning that any future stability claim
  must be machine-replayable and fail-closed.

## invalidated_or_retired_evidence

Retired as claim support:

- local rank-cell stationarity as the main G0 gate;
- strict signed-cell compatibility as the route that must be solved by more
  local solver tuning;
- the idea that signed-gap misses around `1e-14` can be rounded into PASS;
- SLSQP residuals, selected candidates, top20 equality, or Phase-I
  near-feasibility as success evidence;
- `NO_COMPATIBILITY_WITNESS_NO_FARKAS` as incompatibility or infeasibility;
- any v8/further local rank-cell solver pursuit;
- any G0 PASS/freeze/realfold/G1/performance claim from the v7 outcome.

## next_experiments_needed

Do not implement the pivot in this review. The next research route should be a
mechanism-preserving pivot with the fixed skeleton:

`train-only, label-blind structural certificates -> one global full-bank Gram
target -> uniform encoder fit -> ordinary test kNN`.

### Pivot Method Specification

Goal preserved: use a meaningful, novel, removable MLLM component to improve
ordinary full-video train-memory kNN accuracy and macro-F1, ultimately by the
existing final criterion on MHC-EN and MHC-ZH. The MLLM remains train-only and
label-blind. It sees whole-video inputs only and receives no label, prediction,
margin, error, neighbor/key/ID, split, loss, or gradient. It outputs only fixed
structural certificate atoms/confidences. Those certificates materially define
the global feasible and regularization geometry; they are not sample weights,
selectors, rerankers, keys, teachers, or losses.

Let a sealed train fold have normalized bank `Z0 in R^{N x d}`,
`G0 = Z0 Z0^T`, train IDs, and parent-video labels only for train endpoint
constraints and final kNN evaluation. Let the closed label-blind certificate
cache compile deterministic structural operators:

- `A_eq vec(G) = b_eq` for aggregate structural reflection equalities;
- `A_band vec(G)` grouped by certificate family/state/confidence for global
  structural bands;
- `A_reg vec(G) - b_reg` for the same structural moment residuals used by the
  direct control.

The compiler may use train parent labels only after cache closure to define
allowed train-only vote/margin baselines. It may not use segment labels, held,
validation, test, teacher feedback, endpoint correctness, or neighbor keys.

Optimization variables:

```text
G in Sym_N
u_eq in R^{m_eq}
u_band in R^{m_band}
```

Single global target problem:

```text
minimize_G,u
    0.5 * ||G - G0||_F^2
  + 0.5 * lambda_eq * ||u_eq||_2^2
  + 0.5 * lambda_band * ||u_band||_2^2
  + 0.5 * lambda_reg * ||A_reg vec(G) - b_reg||_2^2

subject to
    G = G^T
    diag(G) = 1
    G is PSD
    -1 + delta <= G_ij <= 1 - delta                     for i != j
    ||row_i(G - G0)||_2 <= rho_row                       for all train i
    ||mean_{i in class c} row_i(G - G0)||_2 <= rho_class  for c in {0,1}
    A_eq vec(G) - b_eq = u_eq
    lower_band <= A_band vec(G) + u_band <= upper_band
    train parent-label global/class vote-margin baselines do not decrease
    robust-edge inequalities E_rob vec(G) >= gamma_rob    only for robust edges
```

`A_eq`, `A_band`, `A_reg`, `b_eq`, `b_reg`, `lower_band`, and `upper_band` are
deterministic functions of closed train-only label-blind certificate records
and their aggregate structural states. They define global geometry across the
full bank, including entries outside old top-k neighborhoods. The target is a
single full-bank PSD correlation matrix, not separate local cells.

### Robust and Ambiguous Top20 Edges

Interval-certified top20 stability is used only to decide which rank edges may
be safely constrained or claimed stable. It is not the main gate.

For each train query `q` and candidate `j`, compute a train-only interval
`I_qj = [lo_qj, hi_qj]` from frozen train bank scores, deterministic numerical
rounding bounds, target trust radii, canonical-ID tie tolerance, and any
predeclared structural interval uncertainty. This computation uses no segment
gold, no held/validation/test content, no endpoint correctness, and no
test-time MLLM.

An edge `a before b` is robust iff:

```text
lo_qa - hi_qb >= tau + eta_edge
```

with preregistered `eta_edge > 0`. A query has robust final top20 membership
only if every required 20th-vs-outsider and internal certified edge has that
interval separation. Robust edges may enter `E_rob`. Ambiguous edges add no
geometric constraint: the optimizer is fail-open for geometry. Ambiguous edges
also cannot support an exact-vote safety/stability claim: the claim system is
fail-closed and must report their coverage separately.

### Explicitly Forbidden Degenerations

The pivot must not become sample weighting, confidence weighting, reranking,
neighbor/key selection, memory replacement, pair loss, triplet loss, SupCon,
segment-gold/segment modeling, teacher selection, score fusion, direct
semantic rule loss as the main method, test-time teacher, or test-time MLLM.

The encoder stage fits the certified global target uniformly: every train
video row has the same target-fit sampling rule and the same objective form.
No certificate record may select which sample, neighbor, pair, or triplet gets
optimized.

### Minimal Executable New G0 Gate

New G0 should make no performance claim. It should certify only that the global
structure-certified target is executable, replayable, isolated, nondegenerate,
and distinguishable from forbidden degenerations.

Machine-replayable criteria:

- static design hash, source hashes, dependency hashes, and immutable
  train-only inputs match;
- certificate schema is label-blind, closed before train labels enter the
  compiler, and contains no verdict, score, selected key, rationale, segment,
  timestamp, split, prediction, correctness, loss, or gradient field;
- `A_eq`, `A_band`, `A_reg`, robust-edge intervals, and all target constraints
  are independently rebuilt from closed train-only records;
- the global PSD target replay verifies objective, KKT/VI or certified convex
  optimality residuals, PSD, unit diagonal, box, trust, structural residuals,
  robust-edge residuals, hashes, and finite values;
- ambiguous edges are absent from geometry constraints and from safety claims;
- factorization/reconstruction and uniform encoder-fit dry block pass
  deterministic no-collapse and rollback checks;
- no held/validation/test content or labels, teacher cache, OCR, test-time
  MLLM, or segment artifact is opened;
- no performance metric is emitted by G0.

Fixed expected statuses:

```text
FULL_GLOBAL_TARGET_SYNTH        -> GLOBAL_TARGET_CERTIFIED
FULL_GLOBAL_TARGET_REAL_TRAIN   -> GLOBAL_TARGET_CERTIFIED
REMOVE_NO_CERT                  -> GLOBAL_TARGET_CERTIFIED_NULL
SHUFFLE_CERT_IDENTITY           -> CONTROL_TARGET_CERTIFIED
NOISE_CERT_ATOMS                -> CONTROL_TARGET_CERTIFIED
DIRECT_SAME_MOMENT              -> DIRECT_CONTROL_BUILT_NOT_MAIN_METHOD
AMBIGUOUS_EDGE_FIXTURE          -> GEOMETRY_FAIL_OPEN_CLAIM_FAIL_CLOSED
ROBUST_EDGE_FIXTURE             -> ROBUST_EDGE_CERTIFIED
SEGMENT_GOLD_INJECTION          -> REJECTED_SUPERVISION_VIOLATION
HELD_VAL_TEST_ACCESS_ATTEMPT    -> REJECTED_ISOLATION_VIOLATION
SAMPLE_WEIGHT_SELECTOR_ATTEMPT  -> REJECTED_DEGENERATION
RERANK_KEY_SELECTOR_ATTEMPT     -> REJECTED_DEGENERATION
PAIR_TRIPLET_SUPCON_ATTEMPT     -> REJECTED_DEGENERATION
```

G0 acceptance requires every expected status to replay exactly, all payload
hashes to match, and all isolation counters to satisfy the supervision
contract. Any missing status, relaxed threshold, unbounded ambiguity claim, or
forbidden degeneration is STOP.

### Theoretical and Novelty Claim

Defensible theoretical claim after the new G0 only:

`A closed train-only, label-blind structural certificate cache can define a
single replayable PSD/unit-diagonal full-bank proximal geometry whose target is
independent of local rank-cell stationarity and whose encoder-fit interface is
uniform and removable.`

This remains novel relative to LB-SCGP because the contribution becomes global
structure-certified proximal geometry, not local rank-cell certification. The
scientific mechanism is still structural certificate -> global Gram target ->
uniform encoder fit -> ordinary kNN. The method must later prove that this
interface beats direct use of the identical structural moments and ordinary
comparators; G0 alone cannot claim accuracy or macro-F1.

### Required Ablations

- `REMOVE`: run the same training and kNN endpoint with no certificate-derived
  structural geometry. Falsifies the possibility that gains come from ordinary
  RGCL continuation, target-fit scheduling, or implementation churn.
- `SHUFFLE`: permute sealed certificate records or structural states among
  train IDs while preserving coverage, confidence, missingness, and label
  marginals. Falsifies content/identity-specific structural semantics.
- `NOISE`: corrupt certificate atoms/confidences at preregistered rates while
  preserving schema and closure. Falsifies arbitrary regularization and should
  monotonically remove any FULL gain.
- `direct`: directly optimize the same structural moment residuals
  `A_reg vec(G_theta)-b_reg` with coefficient fixed before outcomes, without
  the proximal global Gram target. Falsifies the claim that the global
  proximal geometry interface adds value beyond direct semantic moment use.
- `ISOLATION`: replay the compiler and endpoint with held/validation/test
  paths physically unavailable and with leakage sentinels that must remain
  unopened. Falsifies cross-split/test-time/certificate leakage rather than a
  performance mechanism.

### Performance Stage Restoration After G0

After and only after a verified pivot G0 GO plus fresh authorization, restore
the existing staged performance plan without inventing segment supervision:

- datasets: MHC-EN (`MHC`) and MHC-ZH (`MHC_zh`);
- folds: fixed five outer folds from `artifacts/ssr/v1/folds/{MHC,MHC_zh}.json`;
- endpoint: ordinary top20 arithmetic-rank, similarity-signed full-video
  train-memory kNN, with no teacher, certificate, compiler, target, rerank,
  fusion, native head, validation, or test-time MLLM;
- seeds: final seeds `0/1/2`;
- baselines and controls: REMOVE, moving LABEL-ONLY or strongest same-protocol
  comparator, SHUFFLE, NOISE, frozen strongest direct control, and any required
  isolation gate;
- primary final claim conditions: on both datasets, accuracy and macro-F1 each
  exceed the moving strongest same-protocol comparator by at least `+0.030`,
  all paired seed deltas are positive, hierarchical paired-bootstrap lower
  bounds exceed zero, Holm FWER `0.05` survives for the four dataset-metric
  tests, FULL significantly beats REMOVE and SHUFFLE, NOISE removes gain, and
  FULL beats the direct control.

### Strict Pivot Acceptance and Stop Conditions

Accept the pivot only if:

- G0 certifies the single global PSD/unit-diagonal target with independent
  replay and exact expected statuses;
- certificates are train-only, label-blind, closed before labels enter the
  compiler, and materially define `A_eq`, `A_band`, `A_reg`, and robust-edge
  geometry;
- ambiguous rank edges are fail-open for geometry and fail-closed for claims;
- uniform encoder fit and rollback are replayable;
- supervision and isolation counters pass exactly;
- no performance claim is made before the restored performance stages.

Stop immediately if any of the following is needed: eta/tau relaxation,
retrospective tolerance, local rank-cell v8 tuning, segment gold, test-time
MLLM, teacher/key selection, sample weighting, reranking, pair/triplet/SupCon
main loss, direct semantic loss as the main method, endpoint peeking, missing
negative controls, or G0 performance claims.

There is no return to v8 solver tuning.

## supervision_contract

Current v7 evidence:

```text
only_gold_supervision = parent_video_binary_label
segment_gold_exists = false
segment_gold_used = false
mllm_call_count = 0
ocr_call_count = 0
teacher_cache_read_count = 0
teacher_cache_write_count = 0
outer_held_label_read_count = 0
outer_held_content_read_count = 0
val_content_read_count = 0
test_content_read_count = 0
val_test_teacher_artifact_count = 0
```

Future pivot contract:

- The only gold supervision remains parent-video binary label.
- No segment, timestamp, span, localization, stance, target, rationale, or
  mechanism gold exists or may be assumed.
- MLLM structural certificates, when introduced, are train-only,
  label-blind privileged pseudo-signals, not annotations or labels.
- Validation/test never load MLLM records, teacher records, compiler artifacts,
  target banks, or certificate-derived geometry.

## nonclaims

- No G0 PASS or freeze.
- No v7 local stationarity certificate.
- No incompatibility or infeasibility proof.
- No real-fold, G1, validation, test, teacher, MLLM, OCR, segment, accuracy, or
  macro-F1 claim.
- No segment gold or segment endpoint.
- No authorization for v8 or further local rank-cell solver pursuit.
- No authorization in this report to implement, submit, monitor, or rerun any
  job.
