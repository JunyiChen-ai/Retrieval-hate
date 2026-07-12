# Refinement Report: LB-SCGP Global

## Phase 5 Summary

- Problem: hateful video detection adapting RGCL/RA-HMD to video with a meaningful, novel, train-only label-blind MLLM role and ordinary full-video train-memory top20 kNN inference.
- Approach: LB-SCGP Global-R2, a certifiable encoder-realizable global proximal certificate geometry method.
- Date finalized: 2026-07-12.
- Rounds completed: 3 review rounds after Phase 0/1.
- Stop condition: met.
- Final reviewer score/verdict: 9.1, READY.
- Drift: NONE.
- Author thread ID: `019f5349-1642-7351-b1a2-7964fb7a4b78`.
- Reviewer thread ID: `019f5351-d691-7501-958d-92432aa97d8a`.
- Exact author model: GPT-5.5 xhigh.
- Exact reviewer model recorded by workflow: GPT-5.5 xhigh.
- Experiment-plan readiness: READY FOR EXPERIMENT-PLAN.
- Experimental success: not claimed. The final global method remains unvalidated and no experiments have been run for this pivot.

## Verbatim Problem Anchor

hateful video detection adapting RGCL/RA-HMD to video; MLLM meaningful+novel; final MHC-EN/MHC-ZH seeds0/1/2 vs strongest same-protocol non-MLLM, acc and macro-F1 each ≥+0.030, all paired seed deltas positive, hierarchical paired bootstrap lower>0, Holm; only parent-video binary gold, no segment/timestamp/span/localization/stance/target/mechanism/rationale gold; train-only label-blind MLLM cache; test ordinary full-video train-memory top20 kNN, no teacher/head/rerank; SLURM; no sample weighting/key selection/pair-triplet/SupCon/segment route; REMOVE/SHUFFLE/NOISE/direct attribution. Local rank-cell v7 formally retired, no v8.

Absolutely do not assume any fragment/segment has gold annotation. The only gold is parent_video_binary_label. Any segment/timestamp/span/localization/stance/target/mechanism/rationale output is not gold and may not be treated as supervision, pseudo-groups, selection, or evaluation gold. Preserve this literally and operationally.

## Output Files

- `phase-0-anchor.md`: immutable anchor and Phase 0 framing.
- `round-0-initial-proposal.md`: initial Phase 1 proposal.
- `round-1-review.md`: Round 1 reviewer response, score 7.1, REVISE.
- `round-1-refinement.md`: full Round 1 author revision.
- `round-2-review.md`: Round 2 reviewer response, score 8.3, REVISE.
- `round-2-refinement.md`: full Round 2 author revision, source for final proposal.
- `round-3-review.md`: Round 3 reviewer response, score 9.1, READY.
- `FINAL_PROPOSAL.md`: clean final proposal extracted from Round 2 revised proposal.
- `REVIEW_SUMMARY.md`: round-by-round resolution log.
- `REFINEMENT_REPORT.md`: this Phase 5 report.
- `score-history.md`: complete score table.
- `REFINE_STATE.json`: finalized workflow state.
- `FILE_HASHES.sha256`: final hash manifest.

## Complete Score Evolution

| Round | PF | MS | CQ | FL | Feas | VF | VR | Overall | Verdict | Drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 8.5 | 6.5 | 7.0 | 7.5 | 6.0 | 8.0 | 6.5 | 7.1 | REVISE | NONE |
| 2 | 9.5 | 8.2 | 8.1 | 8.3 | 7.2 | 8.7 | 7.7 | 8.3 | REVISE | NONE |
| 3 | 9.8 | 9.1 | 9.0 | 9.0 | 8.6 | 9.2 | 8.9 | 9.1 | READY | NONE |

## Round Records

### Round 1

- Verdict: REVISE.
- Overall: 7.1.
- Main blockers: rank realizability, invalid cross-basis `vech`, separate replica-basis comparisons, vacuous robust intervals, insufficient solver certificate, hard/soft structural ambiguity, schema gold-boundary optics.
- Resolution: accepted mathematical blockers; Round 1 introduced fail-closed rank gate, common `Q`, valid `vech`, coordinate trust, H-metric certificate framing, single structural penalty, and observable naming.
- Remaining after revision: KKT serialization, rank-tail reporting, robust default-off, compute feasibility.

### Round 2

- Verdict: REVISE.
- Overall: 8.3.
- Main blockers: KKT certificate had to be the only solver acceptance path; rank-tail reporting needed to be non-gameable; rank failure had to be terminal; compute envelope had to be concrete; robust safety had to be default-off; scalar/direct controls needed attribution wording.
- Resolution: accepted all method-critical items; Round 2 made KKT-only acceptance explicit, added full rank-tail audit, made rank failure terminal, added resource estimates and STOP conditions, disabled robust safety by default, and reframed scalar/direct controls.
- Remaining after revision: minor implementation hygiene only.

### Round 3

- Verdict: READY.
- Overall: 9.1.
- Final reviewer conclusion: method-readiness bar met, no drift, no critical issue, one focused dominant contribution, no bloat.
- Remaining minor implementation hygiene: PSD normal sign convention, manifest-derived `N`/`d`, robust coverage replay, exact KKT payload carry-over.

## Final Thesis Snapshot

A sealed train-only label-blind MLLM structural cache can define a certifiable, encoder-realizable full-bank PSD/unit-diagonal proximal target that the existing video encoder fits uniformly, and the mechanism is supported only if it beats direct certificate-feature and scalar propensity controls under ordinary kNN.

The method uses the MLLM as a structural-observable sensor only. It does not use MLLM outputs as gold, pseudo-groups, selected keys/pairs, sample weights, rationale supervision, or test-time decisions.

## Core Formulas and Interfaces

Common-basis structural moment:

```text
Q = orth_cap(H_N Phi, r_max=8) in R^{N x r}
M_Q(G) = Q^T (G - I_N) Q / N in S^r
m = r(r+1)/2
a_struct(G) = vech(M_Q(G)) in R^m
b_struct = vech(Q^T (K_C - I_N) Q / N) in R^m
r_struct = A_struct vec(G) - b_struct
```

Global projection:

```text
minimize_X 0.5 ||G-G0||_F^2 + 0.5 lambda_struct ||r_struct||_2^2

subject to:
G = G^T
diag(G)=1
G PSD
-1 + delta <= G_ij <= 1 - delta
|G_ij - G0_ij| <= rho_coord
row/class SOC trust constraints
r_struct = A_struct vec(G) - b_struct
optional robust constraints only if G0 coverage passes
```

KKT-only certificate:

```text
0 = H(X*-X0) + sum_j v_j
```

Acceptance requires primal feasibility, dual/cone feasibility, complementarity, stationarity residual, optional conic duality gap if materialized, payload hashes, and independent replay. Finite VI probes and solver traces are diagnostics only.

Coordinate trust and robust default-off:

```text
G_pos = positive G0 gaps over required canonical top20 edges
g_ref = median(G_pos) if nonempty
rho_coord = min(0.02, max(1e-4, 0.10 g_ref))
I_qj = [G0_qj-rho_coord-eps_num, G0_qj+rho_coord+eps_num]
```

Robust constraints are disabled unless the prospective G0 coverage gate passes. Low coverage gives no robust safety claim.

Rank-tail gate:

```text
eps_rank = max(1e-8, 1e-7 max(lambda_1, 1.0))
rank_eps = count(lambda_i > eps_rank)
```

Acceptance requires `rank_eps<=d`, numerical omitted positive tail beyond `d`, numerical negative mass, valid `lambda_min`, and exact Gram reconstruction residual. Rank failure is terminal `ENCODER_RANK_GATE_FAIL`; no truncation or rescue route is allowed.

Factor/Procrustes:

```text
Y in R^{N x d}
Y^T Z0 = L Sigma M^T
R* = L M^T in O(d)
Z* = Y R* in R^{N x d}
```

Uniform fit:

```text
L_fit(theta) = (1/N) sum_i ||normalize(f_theta(x_i)) - z_i*||_2^2
```

Every train video has the same coefficient and schedule.

## Final Evaluation Design

The validation is capped at three blocks:

1. Conceptual G0, real-fold gate, and teacher-cache gate.
2. Mechanism attribution and controls: FULL, REMOVE, SHUFFLE, NOISE, DIRECT-MOMENT, DIRECT-CERT-FEATURE, SCALAR-PROPENSITY.
3. Final paired performance gate: MHC-EN and MHC-ZH, seeds 0/1/2, strongest same-protocol non-MLLM comparator, accuracy and macro-F1 each ≥+0.030, all paired seed deltas positive, hierarchical paired bootstrap lower bound >0, Holm correction, and ordinary full-video train-memory top20 kNN only.

## Method Evolution

- Phase 0/1: formulated global pivot from train-only label-blind structural certificates to one PSD/unit-diagonal global Gram target and ordinary kNN.
- Round 1: repaired dimensional and realizability blockers with fail-closed rank gate, common `Q`, coordinate trust, and a single structural penalty.
- Round 2: closed acceptance and feasibility with KKT-only certificate, rank-tail audit, terminal rank failure, compute envelope, and robust default-off.
- Round 3: reviewer marked READY; no conceptual rewrite required.

## Pushback and Drift Log

No reviewer suggestion required drift from the anchor. The author accepted the mathematical blockers because they strengthened the same global pivot. Additions that would have caused drift were rejected or kept out:

- no local v8 or local rank-cell stationarity;
- no rank truncation or nonconvex rank rescue;
- no teacher/head/reranker/test-time MLLM;
- no sample weighting, key selection, pair/triplet/SupCon route;
- no segment route or fragment gold;
- no scalar-difficulty impossibility theorem.

## Reusable vs Retired Evidence

Reusable inherited evidence:

- no-segment and parent-video-only gold discipline;
- exact ordinary top20 kNN endpoint definitions;
- PSD/unit-diagonal/projector/hash/replay/isolation machinery;
- negative evidence against direct MLLM verdict/rationale and sparse relation routes;
- local v7 retirement authority.

Retired evidence and claims:

- local signed-gap/rank-cell stationarity;
- SLSQP near-miss as success;
- `NO_WITNESS` as infeasibility;
- local v8;
- robust top20 intervals as a success claim when coverage is low.

## Honest Remaining Weaknesses and Minor Implementation Hygiene

- The PSD normal sign convention must be written explicitly in the machine certificate schema.
- `N` and `d` must be read from fold manifests in preflight; prose `N≈550-580` is only an estimate.
- Robust coverage must be replayed before any robust constraints are enabled.
- The KKT payload schema must be carried into experiment planning so implementers cannot replace it with solver traces.
- The global pivot remains unvalidated; no real-fold target, cache, solver certificate, rank-tail pass, uniform fit, or final metric result exists yet.

## Raw Reviewer Responses

<details>
<summary>Round 1 Raw Reviewer Response</summary>

```markdown
# Round 1 Raw Review

## Overall Assessment

This is a faithful and unusually disciplined pivot away from the retired local rank-cell route. The proposal preserves the parent-video-only gold boundary, keeps the MLLM train-only and label-blind, avoids test-time teacher/head/rerank paths, caps the paper around one dominant global-geometry mechanism, and correctly refuses to treat inherited v7 evidence as validation of the new global target.

The main weakness is not contribution sprawl. The main weakness is that several mathematical interfaces are still not executable as written. In particular, the target Gram can have rank larger than the encoder dimension, the replica moment operators are dimensionally and coordinate-wise underspecified, the robust interval construction is likely valid but vacuous under the stated row trust scale, and the Dykstra certificate needs an explicit normal-cone/KKT certificate rather than only replayed residuals. These are method-level issues, not experiment-list issues.

Verdict: REVISE. The direction is promising, anchored, and appropriately modern, but it is not READY until the Gram target is made encoder-realizable and the operator/solver certificates are mathematically closed.

## Scores

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 8.5 | The proposal preserves the immutable endpoint, parent-video-only gold, train-only label-blind MLLM cache, ordinary test kNN path, and retirement of local v7/v8. Minor risk remains because the schema contains stance-like atoms and the projection includes robust pairwise rank edges, but these are currently framed as certificates/safety constraints rather than gold or MLLM-selected pairs. |
| Method Specificity | 6.5 | The pipeline is concrete, but important formulas are not yet executable: cross-replica moment dimensions are not well defined, `vech` is misused for non-symmetric cross-basis moments, rank(G*) may exceed encoder dimension d, and the optimality certificate is underspecified. |
| Contribution Quality | 7.0 | One dominant contribution is visible: certificate-to-global-proximal-Gram geometry. It is more focused than prior local rank-cell work. The claim is weakened by the current realizability gap and by the possibility that the MLLM contribution reduces to a deterministic certificate-kernel regularizer unless the direct/scalar controls are decisive. |
| Frontier Leverage | 7.5 | The frozen MLLM certificate role is appropriate and does not need trendier components. The proposal uses the MLLM as a train-only structural sensor rather than as a classifier or rationale teacher. The remaining issue is proving the role is not replaceable by coarse scalar difficulty or direct certificate features. |
| Feasibility | 6.0 | Full N x N PSD projection, Dykstra over PSD/SOC/halfspace/affine sets, rank-d factorization, and uniform encoder fit are plausible only after sharper dimensional gates and solver certificates are specified. Robust top20 coverage may be near zero with the current interval scale. |
| Validation Focus | 8.0 | The three blocks are appropriately claim-driven: G0/replay/isolation, attribution controls, final paired performance. The plan avoids benchmark bloat and includes the required REMOVE/SHUFFLE/NOISE/direct/scalar controls. |
| Venue Readiness | 6.5 | The paper shape is promising, but a top-venue method cannot leave the rank-realizability and operator definitions unresolved. Once fixed, the framing could become a clean method-first contribution. |

OVERALL SCORE = 0.15*8.5 + 0.25*6.5 + 0.25*7.0 + 0.15*7.5 + 0.10*6.0 + 0.05*8.0 + 0.05*6.5 = 7.1.

## Required Fixes for Dimensions Below 7

### Method Specificity - 6.5

Specific weakness: the central mathematical interface is not fully specified. `M_{U,V}(G)=vech(U^T(G-I)V/N)` is only valid as `vech` when the result is square and symmetric. For `U=Q^(r)`, `V=Q^(s)`, ranks may differ and coordinates from separately computed SVD bases are not aligned. Also, the factorization step assumes `rank(G*) <= d` but the convex PSD projection does not enforce that.

Concrete method-level fix: define all replica stability moments in one common coordinate system, preferably the consensus basis `Q`, or use subspace-invariant projection matrices and `vec`, not `vech`, for cross-basis objects. Add an encoder-realizability contract: either require `rank_eps(G*) <= d` as a hard certification gate, or explicitly replace `G*` by a declared rank-d realizable target `G_fit=Z*Z*^T` and make all claims about `G_fit`, not the unconstrained `G*`. If rank truncation is used, it must be part of the method with residual thresholds and no exact-reconstruction claim.

Priority: CRITICAL.

### Feasibility - 6.0

Specific weakness: the proposed robust intervals are likely too conservative to yield non-vacuous top20 safety constraints. With `rho_row = 0.05 sqrt(N-1)`, the coordinate-wise interval `G0_qj +/- rho_row` can be extremely wide for realistic train-bank sizes, often wider than the entire correlation range. In that case, robust edges and vote preservation constraints may vanish. The solver side is also heavy: full-bank PSD projection plus Dykstra corrections over many sets may be slow or numerically delicate without a dual certificate.

Concrete method-level fix: add a coordinate trust bound such as `|G_ij-G0_ij| <= rho_coord` or compute per-edge min/max bounds by solving the projection-derived bound problem before declaring robust edges. Report robust-query coverage as a G0 gate and treat low coverage as "no safety claim", not as failure of the geometry method. For the solver, define an explicit H-metric KKT/normal-cone certificate with dual residuals for affine, box, SOC, PSD, and halfspace constraints, or use an independently replayable conic-QP certificate for smaller fixtures.

Priority: CRITICAL.

### Venue Readiness - 6.5

Specific weakness: the story is focused, but the current manuscript would be attacked on exact realizability and on whether the MLLM is doing anything beyond generating a hand-engineered structural kernel. The novelty claim depends on the target being a valid encoder target and on controls showing that direct certificate features/scalar difficulty do not match it.

Concrete method-level fix: narrow the claim to "closed train-only certificate cache defines a replayable, encoder-realizable global target" and make the rank-d target interface exact. Move robust vote safety to a subordinate diagnostic unless coverage is meaningful. Keep only two claims: executable global geometry and final ordinary-kNN performance/attribution.

Priority: IMPORTANT.

## Formula and Interface Audit

1. Cross-basis moment definition has a mathematical error. `vech` is not valid for `U^T(G-I)V` when `U` and `V` are different bases or have different ranks. Use `vec` for rectangular cross moments, or avoid cross moments by evaluating all replicas in a shared basis.

2. Separately computed SVD bases are not coordinate-aligned. Even with deterministic signs, rotations within close or repeated singular subspaces can change coordinates. Comparing "corresponding" coordinates across `Q^(r)` and `Q^(s)` is not invariant. Fix by anchoring all replica moments in consensus `Q`, using projection matrices `QQ^T`, or adding a deterministic Procrustes alignment between replica bases before moment comparison.

3. The factorization step is dimensionally incomplete. `G*` can have rank up to N, while the encoder target must live in `R^d`. Zero-padding handles `rank(G*) < d`, but there is no valid step for `rank(G*) > d`. This is the most important mathematical blocker.

4. The statement "verify ||Z*Z*^T-G*|| <= 1e-6" is impossible unless the target rank is at most d. If the method keeps the convex projection, add a fail-closed `rank_eps(G*) <= d` gate. If that gate is too strict, define a rank-d approximation target and stop claiming exact reconstruction of `G*`.

5. The projection is convex as a soft-constraint product-space problem, but the wording should distinguish hard constraints from penalized slacks. `A_reg`, `A_eq`, and `A_band` with free slacks are regularized structural preferences, not exact certificate satisfaction unless slack budgets or hard residual caps are added.

6. `K_C = normalize_corr(C C^T + alpha I_missing)` is underspecified. Define `I_missing` exactly, define what happens when rows are all unresolved, and state whether the missingness indicator columns already make `alpha I_missing` redundant.

7. Robust interval construction from row L2 trust is formally conservative but likely vacuous. The proposal should either add coordinate-level trust or explicitly compute tighter per-edge intervals. Ambiguous edges are correctly fail-open for constraints and fail-closed for claims.

8. Dykstra convergence and replay traces are not by themselves an optimality certificate. The independent verifier should certify primal feasibility plus an H-metric normal-cone residual or equivalent KKT/dual certificate. The variational inequality residual must be precisely defined.

9. The product-space norm with lambda weights is acceptable, but every projector must be implemented in the H-metric, not silently in Euclidean coordinates. The solver spec should state this.

10. Vote-margin constraints use parent labels after cache closure. This is allowed by the anchor, but the proposal must keep them as train-only preservation constraints, not as a new label-driven pairwise metric-learning objective.

## Gold-Boundary Audit

The gold boundary is mostly preserved. The only gold label is `parent_video_binary_label`, and the proposal states that labels first enter only after cache sealing, inside the deterministic compiler and final kNN evaluation. Validation/test do not load MLLM records, target banks, teacher outputs, heads, or rerankers.

The main boundary risk is semantic naming. Fields such as `speaker_source_endorsement`, `quotation_condemnation_reportage_exception`, and `satire_reclaimed_or_counter_speech_exception` are stance/mechanism-like certificate atoms. They can remain only if the proposal keeps repeating and enforcing that they are noisy label-blind structural certificates, not gold stance/mechanism/rationale annotations, not pseudo-groups, not evaluation targets, and not selection keys.

The certificate aggregate operators avoid MLLM-selected pairs, sample weights, and pseudo-groups as written. However, robust rank edges are selected pairwise constraints from G0. They are not MLLM semantic pair supervision, but they should remain safety/trust constraints, not the central contribution or an implicit pairwise training route.

## Complexity-Cap Audit

The cap is mostly respected.

- New components: two, if the cache/compiler and convex target/uniform fit are treated as the two components.
- Claims: two, with the G0 executable-geometry claim and final performance/attribution claim.
- Core experiment blocks: three.
- No new head, teacher, reranker, router, segment model, sample weighting, key selection, pair/triplet/SupCon loss, or local v8 stationarity path is introduced.

Complexity risk remains in the solver/constraint system rather than in model architecture. The method may feel overbuilt if `A_reg`, `A_eq`, `A_band`, robust edge constraints, vote constraints, row trust, class trust, factorization gates, and multiple controls are all presented as equally central. The paper should make one thing central: encoder-realizable global proximal certificate geometry.

## Evidence-Status Audit

The proposal correctly states that inherited evidence supports only supervision isolation, endpoint definitions, replay/hash discipline, PSD/unit-diagonal machinery, and the retirement of local rank-cell stationarity. It does not falsely claim that v7 validates the new global pivot.

The new global target, restricted schema, real-fold G0, teacher-cache gate, uniform fit, and final MHC-EN/MHC-ZH performance remain unvalidated. That is acceptable for a proposal, but the text should keep calling them planned gates rather than evidence.

Local rank-cell v7 is retired and no v8/local stationarity path has crept back in. The proposal uses robust top20 intervals only as safety diagnostics/constraints, not as a revival of local stationarity.

## Explicit Audit Checklist

- Restricted label-blind MLLM cache: mostly correct. Parent labels first enter after cache sealing. Add a hard audit that schema atom names cannot be used as stance/mechanism gold.
- Certificate aggregate operators: mostly avoid pseudo-groups, weighting, selected MLLM pairs, and semantic pair supervision. The basis/operator definitions need dimensional repair.
- One closed convex projection: conceptually yes, but the soft-slack formulation and rank-realizability gap must be clarified. The projection is convex; the encoder-realizable target is not guaranteed.
- Dykstra/proximal solver: plausible, not yet sufficient. Needs explicit H-metric KKT/normal-cone certificate.
- Robust intervals: conservative and fail-open/fail-closed logic is correct, but the stated row radius likely makes them vacuous.
- Dimensional validity of G*: not valid until rank(G*) <= d or a rank-d target interface is specified.
- MLLM replacement risk: acknowledged by controls. The claim should remain conditional on beating direct certificate-feature and scalar-propensity controls.
- Caps: <=2 new components, <=2 claims, <=3 core experiment blocks are satisfied.
- Gates: new G0, real-fold gate, teacher-cache gate, and final performance gate are present.
- Final protocol: REMOVE/SHUFFLE/NOISE/direct attribution, strongest same-protocol comparator, MHC-EN/ZH seeds 0/1/2, both metrics +0.030, all deltas positive, hierarchical bootstrap lower >0, Holm are present.
- Novelty boundaries: credible relative to pseudo-groups/reweighting, semantic pair supervision, privileged distillation, and metric learning, subject to fixing the rank/operator issues.
- Inherited evidence: not overclaimed.
- Local v7/v8: v7 retired, no v8 path detected.

## Simplification Opportunities

1. Use one common consensus basis for all structural and replica-stability moments. This deletes cross-basis coordinate matching and removes a real mathematical failure mode.

2. Make robust rank/vote constraints diagnostic or gate-only in the first G0 unless coordinate-bound coverage is non-vacuous. This reduces selected-pair optics and keeps the paper centered on global certificate geometry.

3. Merge `A_eq` and `A_band` into a single cache-stability regularizer or gate if both are not needed. The method should not look like a list of constraint families.

## Modernization Opportunities

NONE. The frozen MLLM-as-certificate-sensor role is already the right modern primitive under the anchor. Adding a reasoning teacher, RL, a learned graph module, or a test-time agent would create drift.

## Drift Warning

NONE.

The proposal currently solves the anchored problem. The only drift risks are future revisions that treat stance/mechanism-like certificate atoms as gold, use robust rank edges as selected-pair metric learning, or revive local rank-cell stationarity under a new name.

## Remaining Action Items Ranked by Priority

1. CRITICAL: Fix encoder realizability. Specify what happens when `rank(G*) > d`, and ensure the final target fitted by the encoder is exactly defined in `R^d`.

2. CRITICAL: Repair the structural moment interface. Replace invalid cross-basis `vech` with invariant/common-basis operators and define replica ranks/alignment.

3. CRITICAL: Make robust intervals non-vacuous or explicitly diagnostic. Add coordinate trust or exact per-edge bound computation, and report robust-query coverage.

4. IMPORTANT: Add an explicit solver optimality certificate: H-metric normal-cone/KKT residuals, not just Dykstra replay traces.

5. IMPORTANT: Clarify soft versus hard structural constraints. Slacked equality/band systems should be described as regularized preferences unless bounded by certified residual caps.

6. IMPORTANT: Tighten the MLLM novelty claim around the direct/scalar controls. State that if DIRECT-CERT-FEATURE or SCALAR-PROPENSITY matches FULL, the MLLM-global-geometry claim fails.

7. MINOR: Rename or annotate stance-like schema fields as noisy structural observables to reduce gold-boundary confusion.

## Verdict

REVISE.

The proposal is anchored, focused, and has the right high-level contribution shape. It is not READY because the core convex-Gram-to-encoder interface is not yet mathematically closed, and because the solver/operator certificates need enough precision that another engineer can implement and independently verify them without interpreting intent.
```

</details>

<details>
<summary>Round 2 Raw Reviewer Response</summary>

```markdown
# Round 2 Raw Review

## Overall Assessment

The Round 1 refinement is a substantial improvement. The immutable Problem Anchor is preserved verbatim, and the literal no-fragment-gold sentence is also preserved verbatim. The revised proposal still treats `parent_video_binary_label` as the only gold supervision. It does not introduce segment, timestamp, span, localization, stance, target, mechanism, or rationale gold; it keeps the MLLM cache train-only and label-blind; and validation/test remain ordinary full-video train-memory top20 kNN with no teacher, head, reranker, or certificate artifact.

The main Round 1 mathematical blockers are mostly resolved. The proposal now uses one common consensus basis `Q`, uses `vech` only on the symmetric `Q^T(G-I)Q/N` object, adds a fail-closed `rank_eps(G*) <= d` gate before factor/Procrustes, adds a prospective coordinate trust bound for robust intervals, demotes robust coverage to safety diagnostics, removes separate `A_eq/A_band` slacked families, and sharpens direct/scalar control failure conditions.

The proposal is not READY under the strict bar. It is still a brittle method plan: a generic convex PSD projection may fail the rank gate, the full-bank PSD/KKT machinery is heavy, and the solver certificate needs one more tightening so finite-probe VI checks cannot substitute for a true normal-cone/dual certificate. These are no longer fatal conceptual errors, but they keep the plan below top-venue-ready completeness.

Verdict: REVISE. No drift. No critical issue remains, but overall is below 9.0 and feasibility/venue readiness are not yet at READY level.

## Scores

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 9.5 | Anchor and no-fragment-gold sentence are preserved. The final endpoint, train-only label-blind MLLM cache, parent-video-only gold, no local v8, no segment route, and ordinary test kNN constraints are intact. |
| Method Specificity | 8.2 | The central interfaces are now mostly executable: common `Q`, valid `vech`, explicit shapes, coordinate trust, rank gate, Procrustes convention, and H-metric certificate. Remaining underspecification is in the exact normal-cone certificate serialization and rank-tail/tolerance reporting. |
| Contribution Quality | 8.1 | The dominant contribution is sharper: encoder-realizable global proximal certificate geometry. Constraint sprawl is reduced. The novelty still depends on controls proving the MLLM-global interface beats direct feature/moment and scalar-propensity routes. |
| Frontier Leverage | 8.3 | The MLLM role is appropriate: a frozen structural-observable sensor, not a teacher or test-time reasoner. No trendier component is needed. |
| Feasibility | 7.2 | The plan is implementable in principle, but full `N x N` PSD projection, dual/KKT verification, and the post-solve rank gate are heavy and may often fail on real folds. This is a feasibility risk, not a mathematical contradiction. |
| Validation Focus | 8.7 | Three blocks remain focused and sufficient: G0/replay/isolation, attribution controls, final paired performance. The required gates and controls are present without benchmark bloat. |
| Venue Readiness | 7.7 | The method now has a credible paper shape, but it needs one more pass on certificate exactness, computational scaling, and rank-gate brittleness before it would feel submission-ready. |

OVERALL SCORE = 0.15*9.5 + 0.25*8.2 + 0.25*8.1 + 0.15*8.3 + 0.10*7.2 + 0.05*8.7 + 0.05*7.7 = 8.3.

No dimension is below 7, so there are no mandatory below-7 repair blocks.

## Round 1 Blocker Resolution Table

| Round 1 blocker | Status | Assessment |
|---|---|---|
| `rank_eps(G*) <= d` gate for exact factor/Procrustes/encoder target | Resolved with feasibility risk | The revision correctly avoids a nonconvex rank constraint and avoids truncation overclaim. If the gate passes, `Y in R^{Nxd}`, `R in O(d)`, and `Z*Z*^T=G*` are exact up to tolerance. If it fails, the run stops. Add tail-eigenvalue reporting so the tolerance boundary is auditable. |
| Common consensus basis and valid `vech` | Resolved | All structural moments now use one `Q in R^{Nxr}`. `M_Q(G)` is symmetric `r x r`, so `vech` is valid. Replica stability enters via `b_r` in the same `Q` coordinates, not unaligned replica SVD bases. |
| Prospective `rho_coord` and robust intervals | Resolved | `rho_coord` is derived from train `G0` rank gaps only, with no label/outcome/held tuning. Intervals are valid under coordinate trust. Low robust coverage correctly removes safety claims without killing global geometry. |
| Single closed strongly convex projection and soft structural preferences | Mostly resolved | The product-space objective over `(G,r_struct)` is closed, convex, and strongly convex for `lambda_struct > 0`. Structural matching is correctly described as a penalized preference, not hard semantic equality. |
| H-metric projectors and KKT/normal-cone/VI certificate | Partial | The required normal-cone/KKT ingredients are now specified. The remaining issue is that finite or probe-based VI checks are not a full certificate. The normal-cone decomposition should be mandatory; probes should be diagnostic only. |
| Simplification of central constraint sprawl | Resolved | `A_eq/A_band` are removed, robust rank/vote is subordinate, and the central mechanism is one structural penalty plus hard feasibility gates. |
| MLLM novelty and direct/scalar failure conditions | Resolved with wording risk | The failure conditions are sharp. The text should avoid overclaiming that scalar difficulty mathematically cannot reproduce any effect; the decisive argument is empirical attribution against direct/scalar controls. |
| <=2 components, <=2 claims, <=3 blocks | Resolved | The cap is maintained. |
| Evidence status and no inherited validation overclaim | Resolved | The global pivot is explicitly unvalidated. Inherited evidence is limited to isolation/replay/PSD/kNN discipline and local-v7 retirement. |
| No local rank cells/v8/SLSQP/NO_WITNESS or forbidden routes | Resolved | No local stationarity path, pair/triplet/SupCon route, sample weighting, key selection, segment route, or test teacher/head/rerank appears. |

## Formula and Interface Audit

1. Anchor preservation: verified. The immutable anchor and no-fragment-gold sentence are present in the revised proposal.

2. Rank gate: mathematically acceptable. The projection remains convex because `rank(G) <= d` is not imposed as a constraint. The post-solve gate `rank_eps(G*) <= d` is fail-closed. If it passes, `Y` is `N x d`; `Y^T Z0` is `d x d`; the SVD produces `L,M in R^{dxd}`; `R*=LM^T in O(d)`; and `Z*=YR* in R^{Nxd}`. Exact Gram reconstruction is then valid subject to the explicit reconstruction check.

3. Rank tolerance: mostly acceptable, but add a tail report. Eigenvalues below `eps_rank` are omitted from `Y`; the reconstruction check catches non-negligible tail mass. The review should require reporting `sum_{lambda_i <= eps_rank, lambda_i>0} lambda_i` and the Frobenius reconstruction residual before any GO status.

4. Structural moments: resolved. `M_Q(G)=Q^T(G-I)Q/N` is symmetric because `G` and `I` are symmetric and `Q` is common. `a_struct`, `b_struct`, and `r_struct` have dimension `m=r(r+1)/2`. `A_struct vec(G)` is well defined.

5. Replica stability: resolved. Replica kernels `K_r` are projected into the same `Q` basis, yielding comparable `b_r`. This is a cache-stability gate, not an extra slacked structural constraint.

6. Projection objective: valid. With `X=(G,r_struct)`, `X0=(G0,0)`, and H metric `diag(I, lambda_struct I)`, the objective is a product-space projection onto a closed convex set with an affine graph `r_struct=A_struct vec(G)-b_struct`. This is one global convex problem.

7. Strong convexity: valid if `lambda_struct > 0`, which the proposal fixes. The objective is strongly convex in `(G,r_struct)`; uniqueness follows over the closed convex feasible set.

8. Robust intervals: valid under coordinate trust. The interval `G0_qj +/- rho_coord +/- eps_num` is a sound coordinate-wise bound because `|G_ij-G0_ij| <= rho_coord` is a hard constraint. Robust query coverage is subordinate and may be zero without invalidating the geometry method.

9. Vote constraints: allowed but must remain subordinate. They use parent labels only after cache closure and are linear in fixed robust-query neighborhoods. They must not become a pairwise training loss or central novelty.

10. Solver certificate: improved but not fully closed. The normal-cone decomposition, dual cone membership, complementarity, and stationarity residual are the right certificate. The proposal should remove any implication that finite feasible probes can replace a true VI/KKT certificate.

## Gold-Boundary Audit

The gold boundary is preserved. The only gold is `parent_video_binary_label`. The MLLM schema is train-only and label-blind. Labels first enter after cache closure, only for train-only robust vote diagnostics/constraints, final kNN metrics, and stratified reports/controls.

The schema renaming helps. `source_alignment_observable`, `counter_context_observable`, and `context_shift_observable` are still semantically close to stance/context, but the proposal repeatedly states they are noisy structural observables, never stance, target, mechanism, rationale, localization, pseudo-group, selected key/pair, weight, or evaluation gold. That is acceptable.

No fragment/segment gold is assumed. No validation/test certificate access is allowed.

## Complexity-Cap Audit

The complexity cap is maintained:

- New components: two, cache/compiler and global target/factor/uniform fit.
- New trainable modules: zero.
- Claims: two, G0 executable geometry and final performance/attribution.
- Core experiment blocks: three.
- Robust rank/vote constraints are subordinate safety diagnostics, not an added contribution.

The central method is simpler than Round 1. The remaining complexity is implementation complexity in the solver/certificate, not contribution sprawl.

## Evidence-Status Audit

The revised proposal correctly states that the global pivot is unvalidated. It does not treat v6/v7/local evidence as proof of the revised global target. It inherits only isolation/replay/PSD/kNN discipline and the retirement of local rank-cell stationarity.

No experiment, real-fold G0, teacher cache, uniform fit, validation/test result, accuracy, or macro-F1 claim is made. This is the correct evidence posture.

## Simplification Opportunities

1. Make the normal-cone/KKT certificate the only acceptance certificate and move finite VI probes to diagnostics. This simplifies the solver story and avoids a weak proof path.

2. Keep robust rank/vote constraints disabled by default until the G0 coverage gate passes with meaningful coverage. Report coverage first; add constraints only when the method can defend that they are not selected-pair metric learning.

3. Report rank-gate failure as a terminal null outcome with no rescue path. Do not add rank truncation, learned low-rank adapters, or nonconvex rank optimization unless the whole method is reframed.

## Modernization Opportunities

NONE. The frozen MLLM structural-observable role is already the correct frontier primitive for the anchor. Adding a reasoning teacher, RL, learned graph, adapter, test-time MLLM, or head would create drift or contribution sprawl.

## Drift Warning

NONE.

The revised proposal still solves the anchored problem. The only drift risks to keep guarding are: treating structural observables as stance/mechanism gold, letting robust rank edges become a pairwise metric-learning route, or adding a teacher/head/reranker after a rank-gate failure.

## Remaining Action Items Ranked by Priority

1. IMPORTANT: Tighten the solver certificate. Require a serialized H-metric normal-cone/KKT certificate for acceptance; finite VI probes should be diagnostic only.

2. IMPORTANT: Add explicit rank-tail reporting around `eps_rank`, including omitted positive eigenmass and reconstruction residual before `Z*` is accepted.

3. IMPORTANT: Add a computational scaling note for full `N x N` PSD projection, eigendecomposition, and KKT verification under the project resource limits.

4. IMPORTANT: Make clear that rank-gate failure is a certified null/fail-closed outcome, not a prompt/schema/tolerance rescue trigger.

5. MINOR: Reword the scalar-difficulty discussion. Keep it as an attribution hypothesis and control condition rather than a broad impossibility proof.

6. MINOR: Keep the schema field names and docs consistently in "observable" language to avoid stance/mechanism-gold optics.

## Verdict

REVISE.

The revision resolves the prior critical mathematical blockers well enough to continue. It is still below READY because the strict bar requires overall >=9.0 and a method that is not only exact in principle but also convincingly feasible, certifiable, and presentation-clean. The next revision should focus on certificate exactness, rank-gate auditability, and compute feasibility without adding new components.
```

</details>

<details>
<summary>Round 3 Raw Reviewer Response</summary>

```markdown
# Round 3 Raw Review

## Overall Assessment

The Round 2 refinement should be considered READY under the strict research-refine bar. The immutable anchor is preserved verbatim, including the literal no-fragment-gold sentence. The only gold remains `parent_video_binary_label`. The proposal does not introduce fragment, segment, timestamp, span, localization, stance, target, mechanism, or rationale gold. Validation/test inference remains ordinary full-video train-memory top20 kNN with no MLLM cache, teacher, head, reranker, key selector, or certificate artifact.

The Round 2 blockers are genuinely resolved. The serialized H-metric normal-cone/KKT payload is now the only solver acceptance route, with finite VI probes and solver traces explicitly diagnostic only. The rank-tail audit is non-gameable enough for proposal readiness: it reports positive tail mass beyond `d`, tail ratio, negative mass, minimum eigenvalue, and reconstruction residual; rank failure is terminal with no truncation or rescue. Compute scaling is concrete for the stated MHC scale without fabricated runtime. Robust rank/vote constraints are disabled by default until a prospective coverage gate passes. Scalar/direct controls are now attribution hypotheses with claim-fail logic, not impossibility-theorem overclaim.

The method now has one focused contribution: certifiable encoder-realizable global proximal certificate geometry for ordinary kNN. The remaining items are implementation-spec hygiene, not conceptual blockers.

## Scores

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 9.8 | The anchor, no-fragment-gold sentence, parent-video-only gold boundary, final MHC-EN/ZH paired success condition, no local v8, no segment route, and ordinary test kNN endpoint are all preserved. |
| Method Specificity | 9.1 | The proposal gives executable interfaces: common `Q`, valid `vech`, closed product-space projection, coordinate trust, KKT payload, rank-tail gate, exact factor/Procrustes, uniform fit, rollback, and resource preflight. Minor implementation detail remains around explicitly fixing the PSD normal sign convention in the certificate schema. |
| Contribution Quality | 9.0 | The contribution is focused and parsimonious: one global, certifiable, encoder-realizable geometry interface. Supporting audit machinery strengthens the main contribution rather than becoming a second paper. |
| Frontier Leverage | 9.0 | The MLLM role is modern and appropriate: a train-only structural-observable sensor, not a teacher, rationale generator, test-time agent, or trend-driven module. |
| Feasibility | 8.6 | At `N≈550-580`, dense Gram storage, eigendecomposition, sparse robust constraints, and KKT payload verification are plausible under 16 CPU / 128 GB / 2 GPU, with explicit STOP conditions. Feasibility is not proven by runtime, but the proposal does not fabricate runtime and has a concrete envelope. |
| Validation Focus | 9.2 | The three blocks are minimal and claim-driven: G0/real-fold/teacher-cache gates, attribution controls, final paired performance. Required REMOVE/SHUFFLE/NOISE/direct/scalar controls and final statistical tests are present. |
| Venue Readiness | 8.9 | If executed cleanly, this is now a credible top-venue method proposal. It is slightly below a polished submission only because implementation evidence and real G0 results are still pending, which is acceptable at this refine stage. |

OVERALL SCORE = 0.15*9.8 + 0.25*9.1 + 0.25*9.0 + 0.15*9.0 + 0.10*8.6 + 0.05*9.2 + 0.05*8.9 = 9.1.

No dimension is below 7, so there are no mandatory below-7 repair blocks.

## Round 2 Resolution Table

| Round 2 issue | Status | Assessment |
|---|---|---|
| KKT certificate must be the only solver acceptance route | Resolved | The proposal now requires a serialized H-metric normal-cone/KKT payload. VI probes, random probes, active-face probes, and traces are diagnostic only. |
| Certificate payload executable | Resolved | The payload includes primal values, H metric, affine/box/coordinate/SOC/PSD/halfspace normals, stationarity residual, dual feasibility, complementarity, optional duality gap, and hashes. Independent replay is required. |
| Rank-tail audit around `eps_rank` | Resolved | The proposal reports `rank_eps`, `lambda_d`, `lambda_{d+1}`, positive tail mass beyond `d`, tail ratio, negative mass, `lambda_min`, and reconstruction residual. |
| Rank failure terminal with no rescue | Resolved | `ENCODER_RANK_GATE_FAIL` is terminal. No truncation, prompt/schema/tolerance/teacher/epoch/scale/adapter/nonconvex rescue is allowed. |
| Compute scaling/resource envelope | Resolved | The proposal gives `O(N^2)` storage, `O(N^3)` eigensolver scale, MHC-size estimates, CPU/GPU path, checkpoints, and STOP conditions without fabricated measured runtime. |
| Robust safety disabled by default | Resolved | Robust constraints are off by default and only enabled after a prospective coverage gate. Low coverage gives no safety claim and no selected-pair route. |
| Scalar/direct control wording | Resolved | Scalar/direct baselines are attribution hypotheses. If they match FULL, the mechanism claim fails. No broad impossibility proof is claimed. |
| Common `Q`, valid `vech`, closed projection, coordinate trust, exact `Z*` | Resolved | The Round 1 mathematical repairs are preserved. |
| <=2 components/claims and <=3 blocks | Resolved | The complexity cap is maintained. |
| Evidence posture and forbidden routes | Resolved | The global pivot remains unvalidated; no local v8/cells/SLSQP/NO_WITNESS, sample weighting, key selection, pair/triplet/SupCon, segment route, or test teacher/head/rerank appears. |

## Formula and Interface Audit

1. The anchor is preserved exactly, including the no-fragment-gold sentence.

2. The structural moment interface is dimensionally valid. `Q in R^{Nxr}`, `r<=8`; `M_Q(G)=Q^T(G-I_N)Q/N in S^r`; `vech` is used only on this symmetric object; `m=r(r+1)/2`.

3. The projection is one closed strongly convex product-space problem over `X=(G,r_struct)` with objective `0.5||G-G0||_F^2 + 0.5 lambda_struct||r_struct||^2`, `lambda_struct>0`, PSD/unit diagonal/box/coordinate/SOC/class trust constraints, and structural affine graph `r_struct=A_struct vec(G)-b_struct`.

4. The solver acceptance route is now adequate for proposal readiness. The normal-cone equation `0 = H(X*-X0)+sum_j v_j` is the certificate. The only minor implementation note is to make the PSD normal contribution sign explicit in the machine schema, because the text says "sign consistent with G >= 0" rather than spelling the sign in the serialized field name.

5. The rank-tail audit closes the previous rank loophole. If `rank_eps<=d` and tail mass beyond `d` is numerical, then `Y in R^{Nxd}` exactly reconstructs `G*` within tolerance; `Y^T Z0` is `d x d`; `R*=LM^T in O(d)`; `Z*=YR*`; and `Z*Z*^T=G*` up to the required residual.

6. Coordinate trust gives valid robust intervals. Robust coverage is prospective and subordinate. Low coverage disables robust constraints and safety claims without affecting global geometry.

7. Uniform encoder fit is clean: every train video has the same coefficient and schedule; certificates do not drive sampling, weights, keys, pairs, triplets, SupCon, or reranking.

## Gold-Boundary Audit

The gold boundary is clean. Only `parent_video_binary_label` is gold. Certificate fields are noisy structural observables and are never gold stance, target, mechanism, rationale, localization, timestamp, span, or segment annotations. Parent labels enter only after cache closure, for optional robust vote diagnostics/constraints, final kNN metrics, stratified reports, and controls.

No validation/test path loads certificates, target banks, compiler artifacts, teacher outputs, heads, rerankers, or schema features.

## Complexity-Cap Audit

The proposal stays within the cap:

- two new components: certificate cache/compiler and global target/factor/uniform fit;
- zero new trainable modules;
- two claims: executable global geometry and final performance/attribution;
- three experiment blocks;
- robust safety is optional/subordinate, not a new contribution.

There is no obvious contribution bloat.

## Evidence-Status Audit

The evidence posture is correct. The proposal explicitly says this refinement ran no implementation, experiment, web search, or SLURM job and validates no result. Inherited evidence is limited to isolation/replay/PSD/kNN discipline and local-v7 retirement. It is not used as validation of the new schema, projection, KKT certificate, rank-tail gate, resource feasibility, uniform fit, or final performance.

## Simplification Opportunities

NONE required for proposal readiness. For implementation, keep robust safety constraints off unless the prospective coverage gate passes, as already specified.

## Modernization Opportunities

NONE. The frozen MLLM structural-observable role is the right modern primitive under the anchor. Adding a teacher, RL loop, graph module, adapter, head, or test-time MLLM would be drift.

## Drift Warning

NONE.

## Remaining Action Items Ranked by Priority

1. MINOR: In the implementation handoff, spell out the PSD normal sign convention explicitly in the certificate schema.

2. MINOR: Verify `N` and `d` from fold manifests during preflight rather than relying on approximate MHC sizes in prose.

3. MINOR: Keep robust constraints disabled unless the coverage report passes and is replayed.

4. MINOR: Carry the exact KKT payload schema into the experiment-plan handoff so implementers do not replace it with solver traces.

## Verdict

READY.

The revised proposal reaches the method-readiness bar: overall >=9.0, no drift, no critical issue, one focused dominant contribution, and no obvious bloat. The next step should be experiment planning rather than another conceptual rewrite.
```

</details>
