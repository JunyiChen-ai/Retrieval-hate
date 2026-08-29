# LB-SCGP G0 v5 Result-to-Claim Review

Reviewer: sole fresh independent local reviewer. No subagent, SLURM action,
implementation edit, config edit, threshold edit, protocol edit, artifact edit,
tracker edit, TARGET edit, Python run, model run, data run, GPU run, training
run, experiment run, or rerun of job 12833 was used. This report is the only
authorized write.

## claim_supported

partial

## confidence

high

## route

supplement

## verdict_summary

The v5 synthetic evidence partially supports executability of the frozen
projector, exact-vote, rank-cell negative-control, factor, Farkas, rollback,
hashing, and no-segment-gold machinery. It does not support the full intended
claim that the finite rank-cell certificate can correctly decide
`LOCAL_STATIONARY_CERTIFIED` versus `BOUNDED_SEARCH_FEASIBLE` under the frozen
synthetic ledger, and it is not safe to advance to the real-fold gate.

The current branch remains STOP at synthetic. No result is updated to PASS.
This is not accuracy evidence, not macro-F1 evidence, and not evidence of
final hateful-video detection performance.

## evidence_table

| item | value |
|---|---|
| synthetic job | `12833`, `lbscgp_g0_cpu`, FAILED `2:0`, elapsed `00:00:44` |
| job log | `slurm/logs/lbscgp_g0_cpu_12833.out` |
| job log SHA256 | `a8a249101ebf8ebe3ab56d5b152b8df35a8f593c2271e26e111b04364726ce49` |
| terminal log result | `{"expected_statuses_ok":false,"run_id":"LBSCGP-G0-SYNTH-v5","status":"FAIL","thresholds_ok":true}` |
| manifest SHA256 | `07dc7d5d17194cd7a2b5d42d539adb9e8248e78b4dc629bbcdaf9d4f64719242` |
| manifest status | `FAIL`, `thresholds_ok=true`, `expected_statuses_ok=false` |
| manifest gates | `dykstra_gate=false`, `rank_gate=true`, `farkas_gate=true`, `factor_gate=true`, `rollback_gate=true` |
| manifest payload SHA256 | `751b5ede4cdd6f05032768b4c9295b56ba62fbe370be11436f7ca3f7dbec3fc5` |
| dirty diff SHA256 | `1c8284781fb57e90714b390fdbef362e978b70789632b19df3d8161dfe8827b7` |
| implementation SHA256 | `939acffbafbd9204fc654972cd73f174393c8466c61f4af045b1c20948a6b687` |
| `lb_scgp_g0.py` SHA256 | `1ac305b9236a69881759ae77f3fa9445bad206eb46add8c0e23505a83b1e223e` |
| `lb_scgp_common.py` SHA256 | `f1c95add65d59c6bc692682b4a91daf8f5912a9deb6a19998b412b2e404282bb` |
| independent verifier SHA256 | `f0f49f41de4efee9abf2267b27b75be440f0020583baef565955c3d0c2988b2d` |
| config file SHA256 | `a51981045073e8f5b69da272654d2102ef3f2f5c8739b765d0b161c1f8c75346` |
| config canonical SHA256 | `4a45fb6c66884b6b8aa4571961dff3ef7751c2b9f97e2df1584521cfe1eb3dba` |
| freeze SHA256 | `254e45afe9c0355892824c0c26bc73b4b0854cb20c67c3703982762fad010931` |
| freeze lock SHA256 | `54dc06d236d5fc1f3ac96400f1a81faeb1d2c0c8e5af075065d1964260de98a9`, content `3734131` |
| synthetic lock SHA256s | all eight synthetic `.publish.lock` files hash to `f97fa43cf254437bb42bd682a9e8bbeb3f588b703ea5496f203d91657fd8415b`, content `115811` |
| `cases.jsonl` SHA256 | `8d619f3579886fc9e6a515c5c6c1215a09f4bc628a89ba5e418c18887d27b698` |
| `dykstra.jsonl` SHA256 | `c28090bbd26da0d6ba89ca67340355c8cebc6e24e20899954282dfeed02a92f6` |
| `rank_cells.jsonl` SHA256 | `aad6d4d573b18043059476049279ba5919084b6b220d56cafaf6924efb215a98` |
| `projectors.jsonl` SHA256 | `ad5e26c63125c8d5db78ed254a8d1b6112735eaa72a39bc0287d8eb45fc79091` |
| `exact_vote.jsonl` SHA256 | `bc0c1f4b5823f153c49a9cfb1efb7de531a7a7550890c553810f468b8d00690b` |
| `farkas.jsonl` SHA256 | `b7abd457d44ffcc07b701590d9302e293019842e298ece4aaa70611823c30e90` |
| `factor.jsonl` SHA256 | `75df4129030cd528bee20f0de2c750898a4eeaf90fd49af5698b2b64cc0af6dc` |
| source plan SHA256 | `EXPERIMENT_PLAN.md`: `9eb1f8c69d0a0e1b7c967b658b9a6d11af38b7653348bb38e2b7c2c6b25c2bc7` |
| source proposal SHA256 | `FINAL_PROPOSAL.md`: `94d7b6e9305e8c6095e0a9f20351bb4cafff042f7aa048f2a934ac6d1a3a0a0c` |
| numerical execution record SHA256 | `07a02d976c0630554d82d62bea84e68972d54e41c928cef5a6388ecfe11c14e2` |

## exact_case_metrics

Frozen thresholds from `configs/lb_scgp/lb_scgp_v5.json` are
`topk=20`, `dykstra_set_violation_tolerance=1e-6`,
`dykstra_relative_change_tolerance=1e-7`, `max_dykstra_cycles=500`,
`tie_tolerance=1e-7`, `max_independent_orientations=8`, and
`max_pivots=32`.

| case | expected | emitted status | cycles/max | max_set_violation | relative_change | top20 rank_cell_stable | search_reason | orientations | adjacent checked/total | pivots |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|
| `feasible_interior` | `LOCAL_STATIONARY_CERTIFIED` | `BOUNDED_SEARCH_FEASIBLE` | `1/5` | `3.1086244689504383e-15` | `6.704483993371942e-16` | `true` | `unresolved_cell` | `0` | `1/1` | `0` |
| `feasible_boundary` | `LOCAL_STATIONARY_CERTIFIED` | `BOUNDED_SEARCH_FEASIBLE` | `84/500` | `9.771865633061666e-7` | `7.032126641120063e-9` | `false` | `unresolved_cell` | `0` | `1/1` | `0` |
| `deliberately_capped` | `BOUNDED_SEARCH_FEASIBLE` | `BOUNDED_SEARCH_FEASIBLE` | `1/1` | `0.0014713825104551823` | `0.0011778818356851844` | `false` | `base_cell_not_converged` | `0` | `1/1` | `0` |
| `feasible_oriented_boundary` | `LOCAL_STATIONARY_CERTIFIED` | `BOUNDED_SEARCH_FEASIBLE` | `500/500` | `6.752627108152793e-6` | `5.1725509722744595e-8` | `false` | `base_cell_not_converged` | `1` | `1/1` | `0` |
| `infeasible_constraints` | `BOUNDED_SEARCH_FEASIBLE` | `BOUNDED_SEARCH_FEASIBLE` | `10/10` | `1.0` | `0.0` | `false` | null | null | null | null |

Additional artifact checks:

- `feasible_interior` and `feasible_boundary` both store `fixture.n=24`,
  `fixture.selected_cell_rankings` length `20`, and adjacent-cell ranking
  length `20`.
- `feasible_oriented_boundary` stores no selected local cell ledger because
  the base cell never converged; it stops at `500/500` cycles.
- `rank_cells.jsonl` has `exact_top20=PASS`, `simultaneous_ties=PASS`,
  `orientation_over_budget=REMOVE` with `9 > 8`, `pivot_over_budget=REMOVE`
  with `33 > 32`, `unresolved_tie_map=REMOVE`, and
  `incomplete_adjacent_enumeration=REMOVE`.
- `exact_vote.jsonl` has one `exact_top20=PASS` ledger with 24 queries and
  all top-20 neighbor lists length 20.
- `projectors.jsonl` has seven `PASS` rows; the largest visible dense
  reference/projector metric remains below the frozen projector tolerance,
  including semantic-positive dense-reference error
  `7.385747512579136e-10`.
- `farkas.jsonl` has `known_out_of_cone=PASS` and `known_in_cone=PASS`.
- `factor.jsonl` has `repeated_and_null=PASS` and `negative_reject=PASS`.

## what_results_support

The run supports the narrow claim that the v5 synthetic stage is executable
far enough to produce the expected formal files, locks, manifest, hashes, and
fail-closed status under SLURM. The job created all expected synthetic JSONL
artifacts and a manifest, and the manifest status is consistently `FAIL`
rather than a silent or partial unlock.

The primitive numerical components have meaningful positive evidence:

- isolated projectors pass the registered feasibility, KKT, dense-reference,
  idempotence, finite-difference, variational, and adjoint checks recorded in
  `projectors.jsonl`;
- exact top-20 voting emits a 24-row ledger with top-20 neighbor lists and
  `PASS`;
- rank-cell negative controls remove over-budget orientation, over-budget
  pivot, unresolved tie-map, and incomplete adjacent enumeration cases;
- Farkas and factor synthetic controls pass;
- rollback synthetic control passes;
- all JSONL data are finite under the manifest's `overflow_nan_inf_count=0`.

The run also supports the supervision and access boundary for this synthetic
stage. The manifest and freeze record
`only_gold_supervision=parent_video_binary_label`,
`segment_gold_exists=false`, `segment_gold_used=false`,
`mllm_call_count=0`, `ocr_call_count=0`,
`teacher_cache_read_count=0`, `teacher_cache_write_count=0`,
`outer_held_label_read_count=0`, `outer_held_content_read_count=0`,
`val_content_read_count=0`, `test_content_read_count=0`, and
`val_test_teacher_artifact_count=0`.

The run supports a fail-closed decision discipline. Since the manifest has
`status=FAIL` and `expected_statuses_ok=false`, no realfold, replay, decision,
G1, teacher, MLLM, OCR, held, validation, test, accuracy, or macro-F1 stage is
unlocked by this evidence.

## what_results_dont_support

The run does not support advancing to the real-fold gate. The frozen expected
status ledger requires `feasible_interior`, `feasible_boundary`, and
`feasible_oriented_boundary` to certify `LOCAL_STATIONARY_CERTIFIED`, but all
three emit `BOUNDED_SEARCH_FEASIBLE`.

The run does not support the claim that the current finite rank-cell
controller correctly decides local certification versus bounded fallback:

- In `feasible_interior`, scalar Dykstra convergence is excellent
  (`3.1086244689504383e-15 <= 1e-6` and
  `6.704483993371942e-16 <= 1e-7`) and emitted top-20
  `rank_cell_stable=true`. However, `scripts/analysis/lb_scgp_g0.py` creates
  fixture rankings with `stable_rankings(..., topk=20)` in `_product_fixture`
  and `_refresh_fixture_rank_fields`, while the zero-orientation branch of
  `_rank_search_controller` compares
  `stable_rankings(gram, fixture["ids"], n-1, ...)` against those 20-entry
  fixture rankings. For this 24-item fixture, `n-1=23`. That is an
  implementation/certification comparison defect around the final operator,
  not a scalar numerical failure and not evidence that the fixture expectation
  is wrong.
- In `feasible_boundary`, the same length-inconsistent controller path is
  present, and scalar convergence also meets the frozen thresholds
  (`9.771865633061666e-7 <= 1e-6` and
  `7.032126641120063e-9 <= 1e-7`). But the emitted top-20
  `rank_cell_stable=false`, so this case is not explained solely by the
  `n-1` versus top-20 comparison defect. It is a mixed certification issue:
  controller comparison is defective, and the final top-20 cell stability
  artifact also says the projected case is not stable.
- In `feasible_oriented_boundary`, the failure is not the zero-orientation
  length mismatch. The case has one independent orientation, reaches
  `500/500` cycles, and has
  `max_set_violation=6.752627108152793e-6`, which is greater than the frozen
  `1e-6` threshold, even though relative change
  `5.1725509722744595e-8` is within `1e-7`. This is a genuine frozen-budget
  numerical/certification failure for the current solver and fixture.

The run does not support any performance claim. No real-fold microbenchmark,
GPU replay, G0 decision, G1 OOF result, accuracy, macro-F1, statistical test,
or final endpoint was produced.

## failure_classification

The failure is a mixture:

- `feasible_interior`: implementation/certification defect. The final
  top-20 operator is stable and scalar convergence passes, but the controller
  compares a 23-entry ranking to a 20-entry fixture ranking.
- `feasible_boundary`: mixed implementation/certification defect plus
  unresolved top-20 stability. It is exposed to the same 23-vs-20 comparison
  defect, but emitted top-20 `rank_cell_stable=false` means a corrected
  comparison alone is not proven sufficient.
- `feasible_oriented_boundary`: method-numerical or solver-certificate
  failure under the frozen v5 budget and tolerance. The local certificate is
  not achieved at 500 cycles because the max independent set violation remains
  above threshold.

This is not a simple test expectation defect. The expected statuses should not
be relabeled merely to make v5 pass. If a later route claims an expectation is
wrong, it must prove that with an independent high-precision or analytic
certificate before any fixture expectation is changed.

## missing_evidence

- No corrected final-top20 rank controller evidence.
- No v6 synthetic rerun showing that the three expected LOCAL cases certify
  under the frozen status ledger.
- No independent high-precision certificate deciding whether the oriented
  boundary fixture is truly locally certifiable under the registered
  constraints.
- No proof that the oriented fixture should not expect LOCAL.
- No proof that a different solver or splitting certifies the oriented
  boundary within the frozen cycle budget and thresholds.
- No sealed real-fold microbenchmark, no GPU fit replay, no independent G0
  decision, and no ten-fold cost upper bound.
- No G1 OOF accuracy or macro-F1 evidence.
- No evidence involving teacher, MLLM, OCR, held, validation, or test stages;
  those counts are correctly zero and those stages remain locked.

## suggested_claim_revision

Replace the intended claim with:

`Under v5 synthetic-only evidence, LB-SCGP's isolated PSD/unit-diagonal,
projector, exact top-20 vote, rank-cell negative-control, Farkas, factor, and
rollback machinery is executable and fail-closed with parent-video-label-only
supervision. The v5 finite rank-cell/Dykstra certificate does not yet correctly
certify all preregistered LOCAL synthetic cases, and v5 is not safe to advance
to real-fold, replay, decision, G1, teacher, MLLM, OCR, validation, test, or
performance stages.`

Do not claim that v5 proves finite rank-cell certification. Do not claim G0 GO.
Do not claim accuracy or macro-F1 evidence.

## next_experiments_needed

No experiment is authorized by this report. The next work must be a fresh v6
repair and review authorization. After that authorization, the minimum
supplemental work should be:

1. Define and test a canonical rank-controller object strictly around the
   final top-20 operator.
2. Add synthetic cases that make top-20 rankings stable while the 21-through-N
   order changes, so an `n-1` comparison cannot pass as a top-20 certificate.
3. Independently decide the oriented-boundary fixture with a high-precision or
   analytic certificate before changing any expected status.
4. If the oriented fixture is truly LOCAL, implement a solver or splitting that
   certifies it within the frozen budget and `1e-6`/`1e-7` thresholds.
5. If the oriented fixture is not truly LOCAL, revise the fixture only through
   a documented proof and independent review, not as a pass mechanism.
6. Rerun only the synthetic gate after a new freeze/audit authorization, and
   only advance to realfold after synthetic passes with independent review.

## minimum_defensible_v6_route

The rank controller must be defined strictly around the final top-20 operator:

- `topk=20` is the only vote operator. It defines margin coefficients,
  emitted rank-cell stability, exact-vote ledgers, selected-cell hashes, and
  local certification.
- Full `n-1` rankings may be used only as an internal enumeration aid to find
  outsiders and boundary orientations. They must not be directly compared
  against a top-20 selected cell.
- The certificate should store both objects explicitly, for example
  `final_top20_rankings` and `full_outsider_order_for_enumeration`, with
  separate hashes and verifier checks.
- A top-20 certificate must include the 19 internal top-20 inequalities,
  20th-vs-all-outsider inequalities, self exclusion, canonical-ID tie
  semantics, and all compatible final-top20 adjacent orientations.
- A negative control must permute only ranks 21 through `n-1` while preserving
  top 20; this must not invalidate a LOCAL certificate.

For `feasible_oriented_boundary`, v6 should not use max-cycle inflation,
threshold relaxation, expected-status relabeling, or fixture weakening. The
defensible order is:

1. Produce an independent high-precision or analytic certificate for the frozen
   oriented fixture. The certificate must answer whether a
   `LOCAL_STATIONARY_CERTIFIED` top-20 cell exists under the registered PSD,
   unit-diagonal, box, trust, margin, slack, semantic, and rank-halfspace
   constraints.
2. If LOCAL exists, replace or supplement the current cyclic Dykstra backend
   with a solver/splitting that reaches the same certificate within the frozen
   budget. Acceptable candidates include an active-set convex projection/QP
   certificate, a primal-dual splitting with explicit residual certificates,
   or a Dykstra ordering/blocking change that preserves exact Euclidean
   projections and persistent corrections. The emitted artifact must still be
   replayable by an independent verifier.
3. If high precision proves the fixture should not be LOCAL, record that proof
   and revise the fixture expectation only through a new reviewed protocol
   change. That is not a v5 pass, and it must not be done merely because the
   current solver failed.

This route preserves the LB-SCGP novel mechanism if the repair remains a
label-blind structural-reflection moment compiled into an exact-vote-safe
full-bank proximal Gram target followed by uniform encoder fit and ordinary
kNN. A solver replacement is acceptable if it is only the numerical backend
for the same projection and certificate. A method pivot becomes necessary only
if independently certified adversarial top-20 cases repeatedly cannot be
solved or certified under realistic G0 budgets, or if the repair changes the
method into a direct semantic moment objective, reranker, teacher-key selector,
sample weighting method, generic pair/triplet/SupCon method, or segment route.

Mandatory new adversarial synthetic cases and negative controls:

- top-20 stable but 21-through-N order shuffled, to catch full-ranking
  comparison defects;
- zero-orientation scalar convergence with top-20 stability true;
- zero-orientation scalar convergence with top-20 stability false;
- one-boundary-orientation case with an independently known LOCAL solution;
- one-boundary-orientation case with an independently known BOUNDED or
  nonlocal result;
- near-threshold cases with max violation just below and just above `1e-6`;
- relative-change-without-feasibility case, matching the v5 oriented failure
  mode;
- canonical-ID tie cases at, below, and above `1e-7`;
- duplicate-ID and unresolved-tie-map negative controls;
- over-budget orientation and pivot negative controls;
- PSD/unit-diagonal/box/trust intersection stress cases with deterministic
  hashes;
- no-segment and zero-counter manifest negative controls.

Independent acceptance criteria:

- synthetic manifest `status=PASS`, `thresholds_ok=true`,
  `expected_statuses_ok=true`, and all gates true;
- every expected LOCAL Dykstra case has max independent set violation
  `<=1e-6`, relative iterate change `<=1e-7`, final top-20
  `rank_cell_stable=true`, `search_reason=all_adjacent_checked`, and complete
  adjacent-cell enumeration;
- expected BOUNDED cases are bounded for preregistered reasons only, such as
  deliberate cap, infeasible control, or explicit registered over-budget
  controls;
- independent verifier recomputes all projector transitions, rank-cell
  stability, adjacent cells, objectives, exact-vote ledgers, locks, hashes,
  and no-segment/zero-counter fields without importing the producer logic;
- no realfold submission occurs until synthetic has passed and a fresh
  result-to-claim review authorizes the real-fold gate.

## forbidden_pass_mechanisms

The following are explicitly forbidden as pass mechanisms:

- increasing `max_dykstra_cycles`;
- relaxing `dykstra_set_violation_tolerance=1e-6`;
- relaxing `dykstra_relative_change_tolerance=1e-7`;
- changing expected statuses merely to make the suite pass;
- weakening fixtures, removing adversarial boundary cases, or deleting
  negative controls;
- changing `topk=20` semantics;
- using segment-level gold or inherited subclip labels as segment supervision;
- enabling teacher, MLLM, OCR, held, validation, or test access.

## final_boundary

Final v5 result-to-claim verdict: partial support only, route supplement,
current branch STOP at synthetic.

Exact next authorization boundary: a fresh v6 repair/review authorization is
required before any implementation change, config change, threshold/protocol
change, synthetic rerun, realfold, replay, decision, G1, teacher, MLLM, OCR,
held, validation, test, or performance work. This report authorizes no SLURM
job and no PASS update.
