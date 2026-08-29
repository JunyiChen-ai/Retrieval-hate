# G0 v6 Scientific Repair Handoff

Thread/session: fresh GPT-5.5 xhigh v6 scientific-repair implementer in
`/data/jehc223/RGCL`; no subagents.

Scope: prospective v6-only nonformal development sanity. Existing v1-v5
artifacts, freezes, formal evidence, method/config/freeze/formal artifacts are
immutable. Job `12846` is read-only and must not be cancelled, modified,
signalled, requeued, released, held, suspended, or otherwise interfered with.

## Phase-II Scheme

Variables: Gram matrix `G in S_+^n` and original nonnegative slack vector `xi`.

Convexity/locality: the scientific projection is convex only when kept in Gram
space for a fixed local top20 cell. Any factor-space result is local-only and
cannot imply Gram convex optimality.

Objective: original frozen scientific projection form,
`0.5 * ||G - G0||_F^2` plus original quadratic slack terms.

Constraints: unit diagonal, PSD, off-diagonal box, top20 rank halfspaces,
19 internal top20 adjacent inequalities per row, 20th-vs-all-outsider
inequalities per row, self-exclusion, canonical-ID ties, all compatible
adjacent top20 orientations, and nonnegative/budgeted slacks.

Diagnostics: original-objective stationarity, active-set KKT residual,
variational inequality residual, PSD minimum eigenvalue, complementarity,
rank-cell completeness, independent replay, and no-segment/zero-counter checks.

Fail-closed semantics: any failed replay, residual, KKT/VI/PSD/complementarity,
hash, top20 completeness, or no-segment check is `BOUNDED/REMOVE`, never G0
PASS and never infeasibility proof.

## Design Files

- `refine-logs/lb_scgp/v6/G0_V6_SCIENTIFIC_REPAIR_DESIGN_TEST_MACHINE.json`
- `refine-logs/lb_scgp/v6/G0_V6_PROSPECTIVE_TRACKER_FINDING.md`
- `refine-logs/lb_scgp/v6/runtime/validate_scientific_repair_v6.py`
- `refine-logs/lb_scgp/v6/runtime/validate_scientific_repair_v6.sbatch`
- `refine-logs/lb_scgp/v6/runtime/scientific_repair_sanity.py`
- `refine-logs/lb_scgp/v6/runtime/scientific_repair_sanity.sbatch`
- `refine-logs/lb_scgp/v6/runtime/scientific_repair_replay.py`

## Execution Plan

1. Submit fresh validator:
   `sbatch refine-logs/lb_scgp/v6/runtime/validate_scientific_repair_v6.sbatch`
2. Wait for natural terminal state and inspect machine JSON.
3. Only if validator naturally completes and machine gate is `OK`, submit:
   `sbatch refine-logs/lb_scgp/v6/runtime/scientific_repair_sanity.sbatch`
4. Wait for natural terminal state and inspect producer plus replay JSON.

No `--time` is used. Added jobs request at most `4 CPU / 24G`; with job
`12846` using `4 CPU / 24G`, account totals remain within `16 CPU / 128G / 2 GPU`.

## Execution Record

Validator job `12867`: `COMPLETED`, exit `0:0`, elapsed `00:00:01`,
allocation `2 CPU / 4G`, MaxRSS `7468K`. Machine result
`refine-logs/lb_scgp/v6/results/scientific_repair_validation_12867.json`
reported `status=OK`, payload
`cd7f44303b63ee83211cec7f8b7ceadd81d84a2bc6e559f3a969f428b2e5df20`.

First prospective nonformal sanity job `12868`: `FAILED`, exit `2:0`,
elapsed `00:00:02`, allocation `4 CPU / 24G`, MaxRSS `5348K`. Producer
wrote `refine-logs/lb_scgp/v6/results/scientific_repair_sanity_12868.json`
with `status=BOUNDED_REMOVE`, payload
`0377e501248b4931e258979b0f7cd8b895be5ecbbc6ba0f7254df51d9eb36355`.
This was a real failed development run, not infeasibility proof. The failing
case was `known_local_one_boundary_orientation`: expected
`LOCAL_STATIONARY_CERTIFIED`, actual `BOUNDED_SEARCH_FEASIBLE`, because the
fixture accidentally inherited 21 circle-distance adjacent tie descriptors and
therefore exceeded the intended local-orientation sanity.

Repair after `12868`: changed only
`refine-logs/lb_scgp/v6/runtime/scientific_repair_sanity.py` so the local
boundary-orientation fixture starts from the tie-broken PSD stress Gram and
then introduces a single 20th-vs-outsider tie. No expected status was relaxed,
no threshold was changed, and no fixture was weakened to remove a negative
control.

Validator rerun `12869`: `COMPLETED`, exit `0:0`, elapsed `00:00:01`,
allocation `2 CPU / 4G`, MaxRSS `3320K`. Machine result
`refine-logs/lb_scgp/v6/results/scientific_repair_validation_12869.json`
reported `status=OK`, payload
`e4765dfa58c7094dc7912c5a3c6bd07d50161b98e5fb4e0ee1c0c951b321c5d6`.

Prospective nonformal sanity rerun `12870`: `COMPLETED`, exit `0:0`, elapsed
`00:00:01`, allocation `4 CPU / 24G`, MaxRSS `3364K`. Producer result
`refine-logs/lb_scgp/v6/results/scientific_repair_sanity_12870.json` reported
`status=NONFORMAL_SANITY_OK`, payload
`06ed0f04ea28b0db663757f2f8914a56dd7bb19a98e27d09653e1aeb70a63980`.
Independent replay
`refine-logs/lb_scgp/v6/results/scientific_repair_replay_12870.json` reported
`status=REPLAY_OK`, payload
`79fc58398808163d3f825e3383d5379abb9310450540e55efe803c6a77052677`.

Job `12846` remained read-only throughout. Last inspected state during this
handoff update: `RUNNING`, job name `lbscgp_v6_cert_slsqp`, `4 CPU / 24G`.

## Nonformal Sanity Results

All required adversarial cases in the prospective ledger had expected equal
actual status in job `12870`:

- `top20_stable_outsider_shuffle`: `PASS` / `PASS`; final top20 hashes stable
  while outsider-order hashes changed, demonstrating that direct `n-1` versus
  top20 comparison would fail.
- `zero_orientation_scalar_converged_top20_stable_true`:
  `LOCAL_STATIONARY_CERTIFIED` / `LOCAL_STATIONARY_CERTIFIED`;
  max violation `3e-15`, relative change `7e-16`.
- `zero_orientation_scalar_converged_top20_stable_false`:
  `BOUNDED_SEARCH_FEASIBLE` / `BOUNDED_SEARCH_FEASIBLE`; scalar convergence
  alone did not certify.
- `known_local_one_boundary_orientation`:
  `LOCAL_STATIONARY_CERTIFIED` / `LOCAL_STATIONARY_CERTIFIED`;
  2 descriptors, 4 compatible adjacent cells, 4 checked.
- `known_bounded_one_boundary_orientation`:
  `BOUNDED_SEARCH_FEASIBLE` / `BOUNDED_SEARCH_FEASIBLE`.
- `near_threshold_below_1e-6`:
  `LOCAL_STATIONARY_CERTIFIED` / `LOCAL_STATIONARY_CERTIFIED`;
  max violation `9e-7`, relative change `5e-8`.
- `near_threshold_above_1e-6`:
  `BOUNDED_SEARCH_FEASIBLE` / `BOUNDED_SEARCH_FEASIBLE`;
  max violation `1.1e-6`.
- `relative_change_without_feasibility`:
  `BOUNDED_SEARCH_FEASIBLE` / `BOUNDED_SEARCH_FEASIBLE`;
  max violation `6.7e-6`, relative change `5e-8`.
- canonical tie below, at, and above `1e-7`: all `PASS` / `PASS`.
- duplicate ID and unresolved tie-map negatives: `REMOVE` / `REMOVE`.
- orientation over-budget: `REMOVE` / `REMOVE` with `9 > 8`.
- pivot over-budget: `REMOVE` / `REMOVE` with `33 > 32`.
- self-exclusion, PSD/unit-diag/box/trust stress, and no-segment zero-counter
  manifest: all `PASS` / `PASS`.

Phase-II original-objective local certificate in job `12870`:

- status `LOCAL_CERTIFIED_NONFORMAL`;
- original objective `1.2450050000002838e-9`;
- Gram displacement Frobenius `4.9900000000005685e-5`;
- slack L2 `0.0`, slack term `0.0`;
- rank halfspaces `528` total, comprising `456` internal top20 adjacent and
  `72` 20th-vs-outsider constraints;
- min linear residual `-2.734857628447322e-19`;
- PSD minimum eigenvalue `0.9939019511389804`;
- diagonal, symmetry, and box residuals `0.0`;
- active KKT constraints `3`;
- stationarity infinity norm `2.4999987336912256e-11`;
- VI minimum `0.0`;
- complementarity infinity norm `1.3646939565951976e-23`;
- PSD inactive with PSD complementarity `0.0`;
- independent replay recomputed payload, original objective, residuals,
  top20/outsider hashes, KKT/VI/PSD/complementarity, and no-segment checks.

## Audit Questions

- The current Phase-II sanity is a small local Gram-space development sanity,
  not a formal replacement for the frozen v5 synthetic gate.
- The SLSQP backend is used only for a tiny convex Gram-space sanity with
  replayed KKT/VI diagnostics. A future formal route still needs a reviewed
  active-set, primal-dual, or block-Dykstra backend for the frozen fixture.
- Job `12866` establishes only Phase-I Slater/primal feasibility warm-start
  evidence. It does not establish scientific projection optimality.
- A failed solver or failed sanity remains nonconvergence/bounded-remove
  evidence only, never an infeasibility proof.

## Non-Claims

No G0 PASS, no freeze, no formal synthetic, no real-fold, no replay/decision
gate, no G1, no teacher, no MLLM, no OCR, no held/validation/test read, and no
performance experiment is claimed or authorized by this v6 sanity.

Only gold supervision remains `parent_video_binary_label`;
`segment_gold_exists=false`; `segment_gold_used=false`.
