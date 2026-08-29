# LB-SCGP G0 v6 Actual Result-to-Claim Review

Reviewer thread/session: fresh independent RESULT-TO-CLAIM reviewer in
`/data/jehc223/RGCL`, no subagents.

Boundary: read-only shell inspection plus this one report write. I did not edit
implementation, config, frozen artifacts, trackers, or other reports. I did not
submit, control, monitor, cancel, or rerun any SLURM job. I did not run Python,
imports, model code, GPU code, training, replay, or numerical compute.

This report does not claim G0 PASS, does not freeze, and does not update any
PASS status.

## claim_supported

partial

## route

supplement

## confidence

high

## intended_claim

The full intended scientific claim is that LB-SCGP can take the frozen
parent-video-label-only G0 setup, compile the MLLM/structure-certificate idea
into an exact-vote-safe full-bank Gram target, and certify the preregistered
local rank-cell projection with replayable Gram-space feasibility, KKT/VI/PSD
evidence under the frozen v5 thresholds, before any real-fold, G1, teacher,
MLLM, OCR, validation, test, accuracy, or macro-F1 claim is allowed.

For this second v6 supplement, the concrete adjudication target is the frozen
`feasible_oriented_boundary` actual fixture: whether the v6 evidence now proves
that the target top20 cell is compatible and locally stationary for the
original scientific projection objective.

## what_results_support

- The v5 result-to-claim review remains valid: v5 supported the projector,
  exact-vote, rank-cell negative-control, factor, Farkas, rollback, hashing,
  lock, and fail-closed machinery, but only partially supported the full claim.
  The v5 synthetic run failed expected statuses and stayed stopped at synthetic.
- The v6 analytic feasibility witness supports Phase-I primal feasibility for
  one actual compatible cell. Job `12866` produced a rank-9 numerical witness
  for cell 0 with NumPy max residual `1.7763568394002505e-14`, mpmath selected
  max residual `1.1391406924925865e-14`, matched target top20 hash
  `77c3a833ef562a32e05930cbc145675e174a378d88b34290fdf643b2757af5c3`, and
  replay label `accepted_feasible_replayed`. This is feasibility evidence only,
  not original-objective optimality.
- The v6 nonformal scientific sanity result supports the design direction that
  a Gram-space convex QP with separated final-top20 and full-outsider-order
  objects can pass small controlled cases. Job `12870` replayed
  `LOCAL_CERTIFIED_NONFORMAL` with original objective `1.2450050000002838e-9`,
  stationarity infinity norm `2.4999987336912256e-11`, VI minimum `0.0`, PSD
  minimum eigenvalue `0.9939019511389804`, and explicit
  `never_compared_n_minus_1_to_top20=true`. This is not formal frozen-fixture
  evidence.
- The repaired actual fixture oracle binds to the frozen v5 config and dykstra
  evidence: `constraint_set_count=589`, `set_order_matches_frozen=true`,
  topk `20`, violation tolerance `1e-6`, relative tolerance `1e-7`, tie
  tolerance `1e-7`, and source hashes matching v5 config, v5 dykstra JSONL,
  and v5 freeze.
- The actual fixture has a clear orientation geometry: one independent boundary
  descriptor `["p00","p15","p18"]`, two compatible signed cells, and 528 rank
  halfspaces per cell, consisting of 456 internal top20 adjacent constraints and
  72 20th-vs-outsider constraints.
- In repaired actual fixture job `12883`, cell 0 Phase-I has a replayed
  full-rank Slater witness: accepted eig margin `9.999721969433045e-7`,
  max 589 residual `3.552713678800501e-15`, realized top20 equal target, and
  independent replay `full_rank_replay_ok=true`.
- The supervision and access boundary is preserved throughout the accepted
  evidence: only gold supervision is `parent_video_binary_label`;
  `segment_gold_exists=false`; `segment_gold_used=false`; MLLM, OCR, teacher
  cache, outer-held, validation, and test counters are all zero.

## what_results_dont_support

- The v6 evidence does not support the full G0 scientific claim. The actual
  fixture replay status is `REPLAY_OK_BOUNDED_REMOVE`, not
  `REPLAY_OK_LOCAL_STATIONARY_CERTIFIED`.
- Cell 0 Phase-II does not certify original-objective local stationarity. Its
  actual fixture Phase-II result is `BOUNDED_REMOVE`, objective
  `1.630798945779586e-05`, max 589 residual `5.8836793827797916e-09`, PSD
  minimum eigenvalue `-5.8836793827797916e-09`, realized top20 equal target
  `false`, stationarity infinity norm `0.0022782803493536386`, complementarity
  infinity norm `8.21147387671932e-08`, and VI residual bound
  `0.002278362464092406`.
- Cell 1 does not provide an accepted target-cell witness. Multiple candidates
  have positive eig margins and tiny residuals, including anchor attempts with
  max residuals down to `5.838672220806777e-17` and eig margins near
  `1e-4`, but every such attempt realizes a different top20 hash than the
  target hash `f88c7ec0e505f1b5b3ea8464836065fd386df4452acee6d8f823f377dbffb9d1`.
  This is `NO_WITNESS`/nonconvergence for the target cell, not infeasibility.
- The current rank halfspace system does not yet prove strict canonical
  top20 orientation semantics. In cell 1, and in cell 0 Phase-II, the stored
  589 residuals can be tiny while replayed canonical top20 differs from the
  target. Therefore satisfying the present adjacent and boundary halfspaces is
  not enough to certify the realized strict/canonical cell.
- No evidence here supports real-fold execution, G1 OOF performance, accuracy,
  macro-F1, held/validation/test behavior, teacher behavior, MLLM behavior, OCR
  behavior, or segment-level modeling.

## root_causes

1. Cell0 adjudication: unresolved ambiguity, leaning toward solver/active-set
   plus strict-cell-definition deficiency, not yet scientific formulation
   incompatibility.

   Cell0 Phase-I proves that the target top20 cell has a replayed full-rank
   feasible point. That rules out simple primal incompatibility of the target
   cell. Phase-II then fails original-objective KKT/VI and fails realized
   target top20 despite small scalar residuals. This does not by itself prove
   the scientific formulation is incompatible, because the backend is SLSQP
   with a difficult active set, a small PSD violation, huge active multipliers,
   and stationarity residual `2.278e-3`.

   Evidence that would separate the cases:

   - Solver/active-set deficiency would be shown by a solver-independent
     active-set convex QP/SOC/PSD-dual certificate for the same frozen actual
     fixture and same original objective, with replayed primal residuals,
     target canonical top20, KKT stationarity, complementarity, PSD dual
     feasibility, and VI all within the frozen gates.
   - Scientific formulation incompatibility would be shown by a replayed conic
     Farkas/dual infeasibility or optimality certificate proving that no point
     in the explicitly signed target cell can satisfy the frozen PSD, unit
     diagonal, box, trust, semantic, slack, margin, and rank-cell constraints,
     or that the original-objective optimum necessarily lies outside the target
     canonical cell.
   - Current evidence provides neither certificate, so the correct scientific
     label is unresolved ambiguity, not PASS and not incompatibility proof.

2. Cell1 adjudication: the strongest reading is rank-cell encoding ambiguity,
   not target-cell incompatibility and not mere harmless numerics.

   Cell1 candidates often satisfy all 589 residuals with positive eig margin
   while realizing non-target top20 hashes. If residual satisfaction and target
   top20 realization disagree repeatedly, the encoded halfspaces are not yet a
   sufficient representation of strict canonical top20/tie semantics. This
   does not prove the target cell is incompatible, because no conic
   infeasibility certificate was produced. It is also more than a single
   numerical mismatch because the pattern recurs across starts and anchor
   margins.

   The needed implementable certificate is one of:

   - Compatibility certificate: a replayed Gram/slack witness for the explicit
     signed cell with min eig margin above the registered full-rank guard, max
     frozen residual <= `1e-8` for Phase-I, realized
     `canonical_top20(G, tau)` exactly equal to the target, and every signed
     top20/boundary gap satisfying the registered signed-gap inequalities in
     100 dps replay.
   - Incompatibility certificate: a replayed Farkas/conic-dual certificate over
     the same explicit signed cell. It must provide nonnegative linear
     multipliers, SOC dual variables in the Lorentz cones for trust constraints,
     a PSD dual matrix, equality multipliers, stationarity residual near zero,
     dual cone feasibility, and a positive contradiction margin. A search
     failure or slow solver is not such a certificate.

3. The novel mechanism is not yet harmed by one more repair, but repeated
   generic solver tuning would become engineering solver-chasing. The next
   repair must be a mechanism-preserving repair to cell semantics and convex
   certification, not a budget increase or a looser numerical search.

## missing_evidence

- No actual-fixture original-objective certificate with target top20, KKT/VI,
  PSD dual feasibility, and independent replay.
- No explicit signed tie-gap halfspace system that is proven equivalent to the
  replayed canonical top20 operator for the frozen fixture.
- No canonical compatible-cell definition that rejects boundary tie groups
  crossing the top20 cutoff and records complete enumeration under the frozen
  orientation and pivot budgets.
- No conic Farkas certificate for Cell1 target-cell incompatibility.
- No solver-independent active-set convex QP/SOC/PSD-dual certificate for Cell0
  Phase-II.
- No formal v6 synthetic PASS, no G0 freeze, no real-fold run, no G1 OOF, and
  no performance evidence.

## claim_revision

Replace the full claim with the following narrower supported claim:

`Under existing v5/v6 evidence, LB-SCGP's parent-video-label-only projector,
exact-vote, rank-cell negative-control, feasibility-witness, replay, hash, and
fail-closed machinery is executable and preserves the no-segment/no-teacher
supervision boundary. The v6 actual fixture establishes a replayed full-rank
Phase-I feasible target cell for cell 0 and identifies a concrete strict
top20/tie-cell certification gap. It does not yet establish original-objective
local stationarity, does not support G0 PASS or freeze, and does not support
any real-fold, G1, or performance claim.`

Do not claim that v6 proves finite rank-cell certification. Do not claim that
Cell1 is infeasible. Do not claim that Phase-I feasibility or factor
stationarity is Gram-space optimality.

## next_experiments_needed

I authorize at most one prospective v7 repair, with no implementation or job
authorized by this report itself.

The v7 repair is authorized only if it is exactly scoped to:

1. Explicit signed tie-gap halfspaces.
2. A canonical compatible-cell definition.
3. A solver-independent active-set convex QP/SOC/PSD-dual certificate and
   replay for the original Gram-space scientific projection objective.

### Exact v7 math

Let `tau = 1e-7` be the frozen tie tolerance and let `eta > 0` be a fixed
strict numerical guard chosen in the v7 design. `eta` may tighten the cell but
must not relax any frozen threshold. Variables are `G in S^n` and
`xi in R^n`.

For a signed compatible cell `s`, solve the convex Gram-space problem:

```text
minimize    0.5 * ||G - G0||_F^2 + 0.5 * ||xi||_2^2
subject to  diag(G) = 1
            G is PSD
            -1 <= G_ij <= offdiag_upper for i != j
            semantic * vec(G) = 0
            ||row_i(G - G0)||_2 <= rho_row        for all i
            ||mean_{i in class c} row_i(G - G0)||_2 <= rho_class_c
            xi_i >= 0
            sum_{i in class c} xi_i <= budget_c
            vote_margin_i(G) + xi_i >= ell_i       for all i
            class_mean_margin_c(G) >= baseline_c
            global_mean_margin(G) >= baseline_global
            centroid_margin(G) >= centroid_baseline
            A_s vec(G) >= b_s
```

The signed rank-cell block `A_s vec(G) >= b_s` must be built from the final
top20 operator, not from direct comparison of full `n-1` rankings to top20
rankings.

Canonical compatible-cell definition:

- Compute frozen base scores from `G0` and `canonical_top20(G0, tau)`.
- Build descriptors only for pairs whose tie or near-tie can change final
  top20 membership or position under the canonical operator.
- Enumerate independent sign assignments under the frozen
  `max_independent_orientations=8` and `max_pivots=32` budgets.
- Reject duplicate IDs, cycles, unresolved tie maps, incomplete adjacent
  enumeration, and any boundary tie group that crosses the top20 cutoff unless
  a strict signed gap selects membership.
- Store separately:
  `final_top20_rankings`, `full_outsider_order_for_enumeration`,
  `signed_gap_edges`, `cell_sha256`, and `top20_sha256`.

Signed gap constraints:

- For every query `q`, target top20 order `t_1,...,t_20`, and every internal
  ordered edge required by the signed cell, enforce
  `G[q,t_r] - G[q,t_{r+1}] >= tau + eta`.
- For every query `q`, every outsider `o`, and the target 20th neighbor
  `t_20`, enforce `G[q,t_20] - G[q,o] >= tau + eta`.
- If v7 keeps any canonical tie group wholly inside top20, it must encode that
  group with explicit two-sided tie inequalities and prove by replay that the
  deterministic ID order yields the same vote coefficients. Otherwise, ties
  inside certified cells must be converted to strict signed gaps.

The KKT/dual replay must verify a primal-dual certificate for the convex
problem. In particular, with linear multipliers `lambda >= 0`, equality
multipliers `nu`, SOC duals in their Lorentz cones, and PSD dual matrix
`S >= 0`, replay must check stationarity of the original objective, primal
feasibility, dual feasibility, complementarity for all linear/SOC/PSD
constraints, and the VI residual implied by these fields.

### v7 validation and acceptance gates

Validator gates:

- v5 config, v5 dykstra, and v5 freeze hashes match the bound inputs.
- Frozen thresholds remain unchanged: topk `20`, violation `1e-6`, relative
  `1e-7`, tie `1e-7`, max orientations `8`, max pivots `32`.
- Constraint set order remains the frozen 589-set order unless the validator
  emits a separate additive v7 map for signed-gap diagnostics without changing
  the frozen residual map.
- All source files compile; no segment gold, MLLM, OCR, teacher, held,
  validation, or test access is possible.
- Canonical-cell enumeration is complete or fail-closed under the frozen
  budgets.

Acceptance gates:

- Every accepted Phase-I cell has replayed primal residual <= `1e-8`, positive
  full-rank PSD margin, signed-gap residual pass, and realized
  `canonical_top20(G, tau)` equal to the stored final top20.
- A LOCAL Phase-II claim requires original-objective replay, max frozen set
  residual <= `1e-6`, signed-gap residual pass, realized canonical top20 equal
  target, stationarity/dual residual sufficient to imply VI <= `1e-8`, PSD dual
  feasibility, SOC dual feasibility, linear dual nonnegativity,
  complementarity <= `1e-6`, and independent replay without importing producer
  solver logic.
- A BOUNDED/REMOVE result is acceptable only for preregistered reasons:
  incompatible signed cell with conic/Farkas certificate, over-budget
  orientation/pivot, duplicate/unresolved ID/tie map, or replayed failure of a
  required certificate. Nonconvergence alone is not infeasibility.
- The report after v7 may only request a fresh result-to-claim review. It may
  not directly mark G0 PASS or freeze.

### Resource-neutral prospective sequence

This review does not submit anything. If v7 is later implemented by a worker,
the sequence must be:

1. Static validator only, CPU-only, small memory, no `--time`.
2. Actual-fixture signed-cell certificate job only if the validator passes,
   CPU-only and within the existing 4 CPU / 24G scale.
3. Independent replay, CPU-only, no solver import.
4. Fresh result-to-claim review before any PASS, freeze, real-fold, or G1 step.

### Strict v7 stop conditions

- Stop if any threshold, expected status, fixture, or supervision contract must
  be relaxed.
- Stop if signed-cell enumeration exceeds frozen orientation or pivot budgets.
- Stop if Cell0 cannot obtain either a replayed original-objective local
  certificate or a replayed conic/Farkas certificate explaining why it cannot.
- Stop if Cell1 again shows tiny residuals but non-target top20 without an
  explicit compatibility or Farkas certificate.
- Stop after this one v7 attempt. A v8 solver-only iteration would be
  engineering solver-chasing and should pivot instead.

## invalid_or_non_evidence

- Job `12840` is invalid because it was improperly cancelled. It contributes no
  numerical evidence.
- Job `12846` is slow/running/nonterminal evidence only. No infeasibility,
  nonconvergence, or scientific claim may be inferred from its slowness or
  checkpoint behavior, and this review did not interact with it.
- Job `12868` was a failed prospective development sanity caused by an
  accidental fixture issue. It is not infeasibility evidence and not a claim
  result.
- Job `12875` is a pre-repair actual fixture `NO_WITNESS` result. It is useful
  only as a fail-closed diagnostic and not as infeasibility proof.
- SLSQP success, SLSQP failure, feasibility-penalty stationarity, or
  factor-space stationarity is not Gram-space original-objective optimality.
- No threshold relaxation, budget escalation, changed expected statuses,
  fixture weakening, segment gold, teacher, MLLM, OCR, held, validation, or test
  evidence is valid for this claim.

## supervision_contract

The supervision contract remains intact and is part of the claim boundary:

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

No segment gold is assumed, created, or authorized.

## pivot_boundary

If the one authorized v7 repair fails to produce the explicit signed-cell
certificate or a conic/Farkas explanation under frozen budgets, continuing the
same path would become solver-chasing and would harm the novel mechanism.

The pivot should preserve the MLLM structure-certificate / Gram-target core but
remove fragile local rank-cell stationarity as the primary gate. A robust
alternative is a single global PSD Gram-target projection with replayed
structure-certificate constraints, interval-certified top-k stability only
where the final top20 operator is robust, uniform encoder fitting to the
full-bank target, and ordinary kNN evaluation. It must not become sample
reweighting, reranking, teacher-key selection, pair/triplet/SupCon relabeling,
or segment-level modeling/labels.
