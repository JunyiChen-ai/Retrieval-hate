# Round 2 Review: LB-SCGP Global-R1

## Parsed Header

| Field | Value |
|---|---:|
| Problem Fidelity | 9.5 |
| Method Specificity | 8.2 |
| Contribution Quality | 8.1 |
| Frontier Leverage | 8.3 |
| Feasibility | 7.2 |
| Validation Focus | 8.7 |
| Venue Readiness | 7.7 |
| Overall | 8.3 |
| Verdict | REVISE |

Drift Warning: NONE

<details open>
<summary>Full raw review</summary>

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

</details>
