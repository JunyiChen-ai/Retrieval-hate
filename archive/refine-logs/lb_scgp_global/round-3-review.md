# Round 3 Review: LB-SCGP Global-R2

## Parsed Header

| Field | Value |
|---|---:|
| Problem Fidelity | 9.8 |
| Method Specificity | 9.1 |
| Contribution Quality | 9.0 |
| Frontier Leverage | 9.0 |
| Feasibility | 8.6 |
| Validation Focus | 9.2 |
| Venue Readiness | 8.9 |
| Overall | 9.1 |
| Verdict | READY |

Drift Warning: NONE

<details open>
<summary>Full raw review</summary>

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

</details>
