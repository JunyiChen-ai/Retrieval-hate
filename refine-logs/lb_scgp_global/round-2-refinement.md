# Round 2 Refinement

## Problem Anchor

hateful video detection adapting RGCL/RA-HMD to video; MLLM meaningful+novel; final MHC-EN/MHC-ZH seeds0/1/2 vs strongest same-protocol non-MLLM, acc and macro-F1 each ≥+0.030, all paired seed deltas positive, hierarchical paired bootstrap lower>0, Holm; only parent-video binary gold, no segment/timestamp/span/localization/stance/target/mechanism/rationale gold; train-only label-blind MLLM cache; test ordinary full-video train-memory top20 kNN, no teacher/head/rerank; SLURM; no sample weighting/key selection/pair-triplet/SupCon/segment route; REMOVE/SHUFFLE/NOISE/direct attribution. Local rank-cell v7 formally retired, no v8.

Absolutely do not assume any fragment/segment has gold annotation. The only gold is parent_video_binary_label. Any segment/timestamp/span/localization/stance/target/mechanism/rationale output is not gold and may not be treated as supervision, pseudo-groups, selection, or evaluation gold. Preserve this literally and operationally.

## Anchor Check

- Original bottleneck: use a train-only, label-blind MLLM signal to change ordinary full-video kNN geometry without becoming a teacher, head, reranker, segment route, selector, weighting method, pair/triplet/SupCon loss, or local rank-cell solver.
- Round 2 status: the reviewer found no drift and no below-7 dimensions. The requested changes are exactness and feasibility tightenings, not changes to the scientific route.
- Preserved route: restricted MLLM structural-observable cache, one common-basis structural penalty, one closed strongly convex global projection, rank-gated exact factor/Procrustes target, uniform encoder fit, ordinary test kNN.
- Rejected drift: no learned low-rank adapter, truncation rescue, nonconvex rank optimization, teacher/head/reranker, schema/prompt rescue after rank failure, local v8, sample weighting, key selection, pair/triplet/SupCon, or segment route.

## Simplicity Check

- Dominant contribution remains one: encoder-realizable global proximal certificate geometry.
- New components remain two: certificate cache/compiler; global target/factor/uniform fit.
- Claims remain two: executable global geometry; final performance/attribution if all gates pass.
- Experiment blocks remain three: G0/real-fold/teacher-cache; attribution controls; final paired performance.
- Round 2 simplification: finite VI probes are demoted to diagnostics only; robust rank/vote constraints are disabled by default; rank failure is terminal. This removes weak proof and rescue paths rather than adding machinery.

## Changes Made

### 1. Solver acceptance path

- Reviewer said: a serialized H-metric normal-cone/KKT certificate must be the only optimality acceptance path. Finite VI probes cannot certify.
- Action: accepted. The revised proposal defines the exact certificate payload: per-set normal/dual elements, affine/box/coordinate/SOC/PSD/halfspace dual variables, cone membership, primal residuals, stationarity residual `H(X*-X0)+sum_j v_j`, complementarity, and objective/duality gap when a conic dual is materialized. Finite VI probes are diagnostics only.
- Reasoning/evidence: the v6/v7 evidence showed why replayed near-witnesses and solver traces cannot substitute for a proof-grade acceptance certificate.
- Impact: `GLOBAL_TARGET_CERTIFIED` is impossible without the serialized KKT/normal-cone payload and independent verification.

### 2. Rank-tail audit

- Reviewer said: the rank gate is acceptable but needs tail reporting around `eps_rank`.
- Action: accepted. The revised proposal reports `lambda_d`, `lambda_{d+1}`, `rank_eps`, total positive eigenmass, positive eigenmass beyond `d`, tail ratio, numerical negative mass, minimum eigenvalue, and exact Gram reconstruction residual.
- Reasoning/evidence: `rank_eps<=d` alone could hide many small positive eigenvalues; the tail audit prevents tolerance games.
- Impact: acceptance requires both `rank_eps(G*)<=d` and positive tail beyond `d` within numerical tolerance. No truncation is allowed.

### 3. Rank failure terminality

- Reviewer said: rank-gate failure must be certified null/fail-closed, not a prompt/schema/tolerance rescue trigger.
- Action: accepted. The revised proposal makes `ENCODER_RANK_GATE_FAIL` terminal for that G0 target.
- Reasoning/evidence: any rescue by prompt/schema/tolerance/teacher/epoch/scale/truncation/adapter/nonconvex optimization would add components or drift.
- Impact: failure emits terminal certified null/fail-closed status; no `Z*`, no fit, no performance claim.

### 4. Compute feasibility

- Reviewer said: add concrete complexity and memory estimates under 16 CPU/128 GB/2 GPU.
- Action: accepted. The revised proposal gives dense `O(N^2)` storage and `O(N^3)` eigensolver estimates for realistic MHC train sizes around `N≈550-580`, operator storage estimates, expected CPU/GPU path, checkpointing, and STOP conditions if preflight resource measurements exceed caps.
- Reasoning/evidence: full-bank PSD projection is plausible at MHC scale but must be bounded by the project resource policy.
- Impact: future G0 must run a SLURM preflight and stop if measured peak RSS, CPU/GPU use, or projected runtime envelope exceeds cap. No measured runtime is fabricated.

### 5. Robust safety story

- Reviewer said: robust rank/vote constraints should be disabled by default until prospective G0 coverage passes.
- Action: accepted. Robust coverage is first a diagnostic. Robust constraints are off by default and may be enabled only if the G0 coverage gate passes. If coverage is low, geometry continues fail-open but there is no safety claim and no robust rank/vote constraints.
- Reasoning/evidence: this avoids selected-pair optics and keeps the central method global.
- Impact: robust safety is subordinate and optional, not part of the main mechanism.

### 6. Scalar difficulty wording

- Reviewer said: avoid broad impossibility-theorem language.
- Action: accepted. The revised proposal frames scalar/direct baselines as attribution hypotheses and controls. If they match FULL, the mechanism claim fails.
- Reasoning/evidence: empirical attribution is the right standard; no need for an overbroad mathematical impossibility claim.
- Impact: novelty language is cleaner and more defensible.

### 7. Observable naming

- Reviewer said: keep schema field names consistently in "observable" language.
- Action: accepted. The revised schema preserves the `*_observable` naming and states repeatedly that atoms are noisy structural observables, never stance/target/mechanism/rationale/localization gold.
- Reasoning/evidence: this preserves the immutable supervision boundary.
- Impact: no gold optics or pseudo-group interpretation.

### 8. Preserve mathematical repairs

- Reviewer said: preserve common `Q`, valid `vech`, coordinate trust, closed strongly convex projection, exact rank-gated factor/Procrustes/uniform fit.
- Action: accepted. All Round 1 mathematical repairs remain intact.
- Reasoning/evidence: Round 2 marked these issues resolved.
- Impact: no regression in method specificity.

### 9. Preserve caps and final protocol

- Reviewer said: keep two components, two claims, three blocks, immutable metrics/statistics, and only parent-video gold.
- Action: accepted. The revised proposal preserves the caps and final success condition verbatim.
- Reasoning/evidence: the review found no drift and no contribution sprawl.
- Impact: no module pileup.

### 10. Evidence status

- Reviewer said: explicitly remain unvalidated.
- Action: accepted. The revised proposal states that no experiment has yet been run for this pivot and no global-pivot result is validated.
- Reasoning/evidence: inherited evidence supports only discipline, not the new method.
- Impact: no G0, teacher-cache, fit, validation, test, accuracy, or macro-F1 claim is made.

## Revised Proposal

# Research Proposal: LB-SCGP Global-R2 — Certifiable Encoder-Realizable Label-Blind Structural Geometry for Ordinary kNN

## Problem Anchor

hateful video detection adapting RGCL/RA-HMD to video; MLLM meaningful+novel; final MHC-EN/MHC-ZH seeds0/1/2 vs strongest same-protocol non-MLLM, acc and macro-F1 each ≥+0.030, all paired seed deltas positive, hierarchical paired bootstrap lower>0, Holm; only parent-video binary gold, no segment/timestamp/span/localization/stance/target/mechanism/rationale gold; train-only label-blind MLLM cache; test ordinary full-video train-memory top20 kNN, no teacher/head/rerank; SLURM; no sample weighting/key selection/pair-triplet/SupCon/segment route; REMOVE/SHUFFLE/NOISE/direct attribution. Local rank-cell v7 formally retired, no v8.

Absolutely do not assume any fragment/segment has gold annotation. The only gold is parent_video_binary_label. Any segment/timestamp/span/localization/stance/target/mechanism/rationale output is not gold and may not be treated as supervision, pseudo-groups, selection, or evaluation gold. Preserve this literally and operationally.

### Bottom-line problem

Build a meaningful, novel MLLM integration for adapting RGCL/RA-HMD to hateful-video detection while preserving final test inference as ordinary full-video train-memory top20 kNN. The target is final paired performance on MHC-EN and MHC-ZH, not a teacher, head, rationale, local solver, or diagnostic win.

### Must-solve bottleneck

Prior MLLM routes became scalar features, rationale/verdict teachers, key/pair/sample selectors, sample weighting, sparse relation losses, segment-adjacent routes, or local rank-cell certificate chasing. The bottleneck is to make train-only label-blind whole-video structural observations define a full-bank encoder-realizable geometry without treating any certificate atom as gold or selected supervision.

### Non-goals

No segment, timestamp, span, localization, stance, target, mechanism, or rationale supervision/evaluation claim. No test-time MLLM, teacher, head, reranker, router, score fusion, key selection, sample weighting, pair/triplet/SupCon addition, segment route, local v8, SLSQP near-miss success, or `NO_WITNESS` infeasibility inference.

### Constraints

Only `parent_video_binary_label` is gold. MLLM outputs are noisy train-only label-blind structural observables. The compiler may read train parent labels only after cache closure. Validation/test never load certificates, target banks, compiler artifacts, teacher caches, heads, or rerankers. All future computation must use SLURM in `HateVideo`, no `--time`, within 16 CPU / 128 GB / 2 GPU.

### Success condition

Final success requires MHC-EN and MHC-ZH, seeds 0/1/2, strongest same-protocol non-MLLM comparator, accuracy and macro-F1 each at least +0.030, every paired seed delta positive, hierarchical paired bootstrap lower bound >0, and Holm correction. FULL must also beat REMOVE, SHUFFLE, NOISE, and the strongest direct/scalar attribution control under ordinary kNN.

## Technical Gap

RGCL/RA-HMD give a retrieval-memory endpoint, but their geometry is driven by labels and embedding neighborhoods. Hateful-video errors often depend on full-video cross-modal structure: reference signals, harmful surface cues, source/context alignment, and modality binding. Earlier MLLM attempts turned these semantics into local pairs, selected relations, verdict features, or rank cells. That was either too sparse, too selector-like, or too brittle.

The missing interface is a certifiable global one:

```text
sealed label-blind structural cache
  -> one common-basis full-bank structural moment
  -> one closed strongly convex PSD/unit-diag projection
  -> serialized H-metric normal-cone/KKT certificate
  -> rank-tail audited encoder-realizable G*
  -> exact factor/Procrustes Z*
  -> uniform encoder fit
  -> ordinary kNN
```

The global pivot remains unvalidated. Existing evidence supports isolation, no-segment discipline, exact kNN definitions, PSD/unit-diagonal/replay/hash discipline, and retirement of local rank-cell stationarity. It does not prove this pivot works.

## Route Comparison

### Elegant minimal route

Use the MLLM only as a train-only label-blind structural-observable generator. Compile the sealed cache into one common-basis structural penalty in a global convex target. Accept only with a serialized KKT/normal-cone certificate and rank-tail audit. Fit the existing encoder uniformly and test with ordinary kNN.

### Frontier-native route

Use a reasoning teacher, DPO/RL video MLLM, learned graph, adapter, router, head, or test-time agent. This may be more expressive but violates the endpoint constraints or creates contribution sprawl.

### Choice

Choose the elegant minimal route. It is the only route that keeps the MLLM meaningful while preserving removability, ordinary kNN inference, and a single dominant contribution.

## Method Thesis

### One-sentence thesis

A sealed train-only label-blind MLLM structural cache can define a certifiable, encoder-realizable full-bank PSD/unit-diagonal proximal target that the existing video encoder fits uniformly, and the mechanism is supported only if it beats direct certificate-feature and scalar propensity controls under ordinary kNN.

### Smallest adequate intervention

The final endpoint is retrieval geometry, so the method changes only the train-bank target geometry. It adds no inference module, head, selector, pair/triplet loss, or local rank-cell route.

### Foundation-model role

The MLLM is a structural sensor. It does not generate gold labels, logits, rationales, targets, mechanisms, keys, pairs, or test-time decisions.

## Contribution Focus

### Dominant contribution

Certifiable encoder-realizable global proximal certificate geometry for ordinary kNN.

### Supporting contribution

Fail-closed audit discipline: normal-cone/KKT solver certificate, rank-tail audit, compute envelope gate, and optional robust safety diagnostics.

### Explicit non-contributions

No new MLLM reasoning method, no segment annotation, no relation adapter, no pair/triplet/SupCon mining, no new kNN rule, no auxiliary head, no local rank-cell solver, and no test-time MLLM.

## Complexity Budget

- New trainable modules: zero.
- New components: two.
  1. Restricted label-blind MLLM cache/compiler.
  2. Global convex target plus rank-gated factor/Procrustes uniform fit.
- Claims: two.
  1. G0 executable/certifiable/isolated/encoder-realizable global geometry.
  2. Final ordinary-kNN performance and attribution.
- Experiment blocks: three.
  1. G0 plus real-fold and teacher-cache gates.
  2. Attribution controls.
  3. Final paired performance.

## System Overview

```text
train videos only
  -> full-video evidence packs, no labels/splits/margins/IDs/neighbors/losses
  -> restricted deterministic MLLM structural-observable cache
  -> cache closure and Merkle root
  -> compiler opens train parent_video_binary_label only after closure
  -> Phi, K_C, common Q, b_struct
  -> one convex projection over X=(G,r_struct)
  -> serialized H-metric normal-cone/KKT certificate only
  -> rank-tail audit: rank_eps<=d, tail mass numerical
  -> Y in R^{Nxd}, R in O(d), Z*=YR
  -> uniform encoder fit
  -> validation/test ordinary full-video train-memory top20 kNN
```

## Core Mechanism

### Dimensions and notation

For a sealed train fold:

- `N`: train video count. Realistic current MHC clean train sizes are approximately `N≈550` for MHC-EN and `N≈579` for MHC-ZH.
- `d`: encoder embedding dimension, read from `Z0.shape[1]`.
- `Z0 in R^{N x d}`: paired REMOVE/comparator train bank, row-normalized.
- `G0=Z0 Z0^T in S^N`: baseline unit-diagonal Gram.
- `G in S^N`: target Gram variable.
- `X=(G,r_struct)`: product-space projection variable.
- `y_i in {0,1}`: parent video binary label, the only gold.
- `topk=20`: ordinary kNN endpoint.

### Restricted MLLM certificate schema

The MLLM receives only deterministic full-video evidence packs:

- uniform full-video frames;
- title if available;
- ASR/OCR text if available, deterministically truncated;
- no label, prediction, correctness, margin, loss, gradient, split, seed, neighbor, key, memory ID, or dataset statistic.

Strict JSON schema `scgp_global_cert_v2`:

```json
{
  "schema_version": "scgp_global_cert_v2",
  "visual_reference_observable": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "text_audio_reference_observable": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "harmful_surface_observable": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "dehumanizing_or_threat_surface_observable": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "cross_modal_binding_observable": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "source_alignment_observable": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "counter_context_observable": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "context_shift_observable": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "modality_binding_observable": {
    "state": "visual_text|visual_audio|text_audio|multi_modal|single_modal|unresolved",
    "confidence": 0
  },
  "parse_flags": []
}
```

All fields are noisy structural observables. They are not stance, target, mechanism, rationale, localization, or segment gold. Extra keys, free text, target names, proposition text, mechanism text, timestamps, spans, localization, verdicts, or rationales are parse failures.

`confidence` is `0..4` and is used only for parse/agreement diagnostics and cache-stability gates. It is not a weight or selector.

Cache protocol:

- four deterministic calls per train video;
- fixed decoding/model/processor/prompt/input/schema hashes;
- parse failures become canonical all-unresolved records;
- consensus maps observables to `+1`, `-1`, or `0`;
- all records remain in the full bank;
- cache closure writes JSONL, consensus, hashes, ID allowlist, and Merkle root before labels enter.

### Parent label entry point

After cache closure, train parent labels may enter only for:

1. optional robust vote diagnostics/constraints if coverage passes;
2. final ordinary kNN metric computation;
3. stratified reports and controls.

Labels never enter MLLM prompts, certificate acceptance, schema repair, sample selection, pair selection, key choice, or weight assignment.

### Certificate encoding and common basis

Let `R=4` be replica count and `p` encoded feature count.

Encode replica and consensus records as:

```text
Phi^(r) in R^{N x p}
Phi      in R^{N x p}
```

Encoding uses frozen one-hot/contrast columns plus explicit unresolved/missing columns so every row has nonzero norm.

Row-normalize:

```text
V_i      = Phi_i / ||Phi_i||_2
V_i^(r)  = Phi_i^(r) / ||Phi_i^(r)||_2
K_C      = V V^T in S^N
K_r      = V^(r) (V^(r))^T in S^N
```

`diag(K_C)=diag(K_r)=1`.

Center:

`H_N = I_N - 11^T/N`.

Common basis:

```text
B0 = H_N Phi
Q = orth_cap(B0, r_max=8) in R^{N x r}
```

`orth_cap` uses float64 thin SVD, threshold `sigma > max(1e-8, 1e-7 sigma_max)`, rank cap `r<=8`, and canonical-ID sign orientation. If `r=0`, emit certified null. No separate replica SVD bases are compared.

### Structural operator

```text
M_Q(G) = Q^T (G - I_N) Q / N in S^r
m = r(r+1)/2
a_struct(G) = vech(M_Q(G)) in R^m
b_struct = vech(Q^T (K_C - I_N) Q / N) in R^m
A_struct vec(G) = a_struct(G)
r_struct = A_struct vec(G) - b_struct
```

`vech` is valid because `M_Q(G)` is square and symmetric.

Replica stability gate:

```text
b_r = vech(Q^T (K_r - I_N) Q / N)
sigma_cache = max_r ||b_r - b_struct||_inf
```

Pass if:

`sigma_cache <= max(0.05, 0.25 ||b_struct||_inf)`.

Failure is `CACHE_STABILITY_FAIL` or certified null. There is no post-outcome prompt/schema rescue.

The structural term is a regularized preference:

`0.5 lambda_struct ||r_struct||_2^2`.

It is not a hard semantic equality and not gold supervision.

### Single closed strongly convex projection

Variables:

```text
X=(G,r_struct)
G in S^N
r_struct in R^m
X0=(G0,0)
```

Objective:

```text
minimize_X 0.5 ||G-G0||_F^2 + 0.5 lambda_struct ||r_struct||_2^2
```

with `lambda_struct>0`, frozen before outcomes.

Hard constraints:

```text
G = G^T
diag(G) = 1
G is PSD
-1 + delta <= G_ij <= 1 - delta                         for i != j
|G_ij - G0_ij| <= rho_coord                              for i != j
||G[i,-i] - G0[i,-i]||_2 <= rho_row                      for all i
||mean_{i:y_i=c}(G[i,:]-G0[i,:])||_2 <= rho_class_c       for c in {0,1}
r_struct = A_struct vec(G) - b_struct
optional robust safety constraints only if G0 coverage passes
```

This is one global closed convex problem. There is no local cell enumeration, no hard rank constraint, no selected semantic pairs, and no local stationarity route.

Frozen defaults:

- `delta=1e-4`.
- `lambda_struct=1` unless fixed by pre-outcome synthetic scaling.
- `rho_row=0.05 sqrt(N-1)`.
- `rho_class_c=0.02 sqrt(N)`.
- `tau=1e-7`.
- `eta_edge=1e-8`.

### Robust safety: disabled by default

Robust rank/vote constraints are disabled by default. G0 first reports coverage.

Coordinate trust:

```text
G_pos = positive G0 gaps over required canonical top20 edges
g_ref = median(G_pos) if nonempty
rho_coord = min(0.02, max(1e-4, 0.10 g_ref))
```

If `G_pos` is empty, `rho_coord=1e-4` and robust coverage is zero.

Intervals:

`I_qj=[G0_qj-rho_coord-eps_num, G0_qj+rho_coord+eps_num]`.

Robust edge:

`G0_qa-G0_qb >= 2 rho_coord + 2 eps_num + tau + eta_edge`.

Robust query: all required top20 internal and 20th-vs-outsider edges robust.

Coverage gate:

- at least 10 robust queries per parent class;
- at least 10% robust query coverage overall;
- coverage report replayed by the verifier.

If the gate fails, robust constraints remain disabled and no robust safety/vote claim is made. The global structural geometry still proceeds fail-open. If the gate passes, a preregistered safety-enabled G0 may include robust halfspaces and robust global/class vote preservation. These are G0-derived safety constraints, not MLLM-selected pairs and not the main method.

### Solver acceptance: serialized H-metric normal-cone/KKT certificate only

Let `C` be the feasible set from all hard constraints. Acceptance requires a serialized certificate proving:

`0 = H(X*-X0) + sum_j v_j`

up to tolerance, where each `v_j` is a normal element for one constraint family at `X*`.

Finite VI probes, random feasible probes, active-face probes, and solver traces are diagnostics only. They cannot certify optimality.

#### Certificate payload

The producer must serialize:

1. `primal`:
   - `G*`, `r_struct*`, objective value, all residual summaries.

2. `metric`:
   - H blocks: identity on `G`, `lambda_struct I_m` on `r_struct`.

3. `affine_normals`:
   - symmetry/diagonal affine duals if represented separately;
   - structural graph dual `nu_struct in R^m` for `r_struct - A_struct vec(G) + b_struct = 0`;
   - normal contribution to `G`: `-A_struct^T nu_struct`;
   - normal contribution to `r_struct`: `nu_struct`.

4. `box_coordinate_normals`:
   - off-diagonal box lower/upper multipliers;
   - coordinate trust lower/upper multipliers;
   - all multipliers nonnegative;
   - complementarity products.

5. `soc_normals`:
   - row-trust Lorentz duals for every active row ball;
   - class-mean trust Lorentz duals;
   - cone membership residuals and complementarity.

6. `psd_normal`:
   - PSD dual matrix `S_psd in S^N`;
   - `S_psd` dual feasible with eigen minimum `>= -1e-7`;
   - PSD normal contribution sign consistent with `G >= 0`;
   - complementarity `|tr(S_psd G*)|`.

7. `halfspace_normals`:
   - robust rank/vote halfspace multipliers if such constraints are enabled;
   - nonnegative multipliers and complementarity.

8. `stationarity`:
   - serialized `v_j` for each family;
   - residual vector

     `res = H(X*-X0) + sum_j v_j`;

   - normalized residual

     `||res||_2 / (1 + ||H(X*-X0)||_2)`.

9. `dual_feasibility`:
   - nonnegative multipliers `>= -1e-10`;
   - SOC cone residual `<=1e-7`;
   - PSD dual eigen minimum `>= -1e-7`;
   - affine duals unrestricted.

10. `complementarity`:
    - linear/box/coordinate/halfspace complementarity `<=1e-6`;
    - SOC complementarity `<=1e-6`;
    - PSD complementarity `<=1e-6`.

11. `duality_gap`:
    - if a conic dual objective is materialized, relative gap `<=1e-6`;
    - if not materialized, no duality-gap pass is claimed. Stationarity plus cone-valid normal decomposition remains the acceptance path.

12. `hashes`:
    - source/config/input/operator/constraint/certificate payload hashes.

#### Acceptance tolerances

Required:

- max primal residual `<=1e-6`;
- stationarity normalized residual `<=1e-6`;
- dual cone residuals as above;
- complementarity max `<=1e-6`;
- finite values and hash match;
- independent verifier rebuilds all normals and residuals without producer imports.

Failure of any item is STOP. Solver traces and VI probes may explain failure but cannot override it.

### Rank-tail audit and encoder realizability

After KKT-certified projection, eigendecompose:

`G* = U diag(lambda) U^T`, `lambda_1 >= ... >= lambda_N`.

Freeze:

```text
eps_rank = max(1e-8, 1e-7 max(lambda_1, 1.0))
rank_eps = count(lambda_i > eps_rank)
```

Report:

- `lambda_d` if `d<=N`, else `lambda_N`;
- `lambda_{d+1}` if `d<N`, else `0`;
- `rank_eps`;
- `positive_eigenmass = sum_i max(lambda_i,0)`;
- `omitted_positive_eigenmass_beyond_d = sum_{i>d} max(lambda_i,0)`;
- `tail_ratio = omitted_positive_eigenmass_beyond_d / max(positive_eigenmass,1e-12)`;
- `negative_eigenmass = sum_i max(-lambda_i,0)`;
- `lambda_min`;
- exact Gram reconstruction residual after forming `Y`.

Acceptance requires:

```text
rank_eps <= d
omitted_positive_eigenmass_beyond_d <= max(1e-6, 1e-8 N)
tail_ratio <= 1e-8
negative_eigenmass <= max(1e-6, 1e-8 N)
lambda_min >= -1e-7
reconstruction_residual <= 1e-6
```

No truncation is permitted. If any positive tail beyond `d` is non-numerical, the gate fails.

Rank failure is terminal:

```text
ENCODER_RANK_GATE_FAIL
```

No prompt/schema/tolerance/teacher/epoch/scale/truncation/adapter/nonconvex rescue is allowed.

If the rank audit passes, form `Y in R^{N x d}`:

- first `rank_eps` columns are `U_r diag(sqrt(lambda_1..lambda_r))`;
- remaining columns are zero;
- no omitted non-numerical positive eigenmass.

### Procrustes and `Z*`

Given `Y in R^{N x d}` and `Z0 in R^{N x d}`:

```text
Y^T Z0 = L Sigma M^T
R* = L M^T in O(d)
Z* = Y R* in R^{N x d}
```

Verifier checks:

`||Z*(Z*)^T-G*||_F / max(1,||G*||_F) <= 1e-6`.

`Z*` is accepted only after the rank-tail audit passes.

### Uniform encoder fit

`L_fit(theta) = (1/N) sum_i ||normalize(f_theta(x_i)) - z_i*||_2^2`.

Every train video has the same coefficient and schedule. Certificates never control sampling, weights, pairs, triplets, keys, or reranking.

Rollback saves model, optimizer, scheduler, scaler, RNG, sampler, and epoch cursor. Fit failure restores and replays REMOVE; hash must match direct REMOVE.

## Compute Feasibility and Resource Envelope

No runtime has been measured for this pivot. The following is a prospective complexity envelope, not an experimental result.

Realistic current MHC clean train sizes are approximately:

- MHC-EN: `N≈550`;
- MHC-ZH: `N≈579`.

For `N=600`:

- dense `G` float64 storage: `N^2 * 8 ≈ 2.9 MB`;
- ten dense `N x N` work arrays: about `29 MB`;
- one dense PSD dual plus primal/correction arrays: still well below `1 GB`;
- `A_struct` should not be materialized as dense `m x N^2` if avoidable, but even with `r<=8`, `m<=36`, dense storage is about `36*360000*8≈104 MB`;
- robust halfspaces, when enabled, are sparse and scalar-supported; they must not be stored as dense vectors;
- PSD eigendecomposition is `O(N^3)`, about `2.16e8` dense-flop scale for `N=600`;
- dense Gram and operator operations are `O(N^2 r)` or `O(N r^2)` for structural moments.

Expected hardware path:

- CPU-only for compiler, convex projection, eigendecomposition, KKT verification, rank audit, and replay;
- at most 16 CPU threads;
- memory budget target below 64 GB peak, hard STOP if projected or measured peak exceeds 96 GB before safety margin under 128 GB;
- GPU only for future uniform encoder dry-fit/training, at most 2 GPUs.

Checkpointing:

- atomic checkpoint after cache closure, operator compilation, every fixed solver cycle interval, KKT certificate serialization, rank audit, factor/Procrustes, and fit rollback state;
- checkpoints store arrays or hashes according to size, with exact payload hashes;
- no held/validation/test artifacts opened by G0.

STOP conditions:

- preflight same-`N` synthetic memory estimate or measured SLURM peak RSS exceeds 96 GB;
- CPU thread need exceeds 16;
- GPU requirement exceeds 2;
- dense halfspace materialization would exceed memory cap;
- KKT certificate payload cannot be independently verified within the resource envelope;
- measured microbenchmark suggests final planned stages would exceed the project compute cap.

All actual future compute must be submitted through SLURM, `conda activate HateVideo`, with no `--time`. Job `PENDING (JobHeldUser)` must be left for automatic release.

## Modern Primitive Usage

The MLLM is a frozen train-only structural-observable sensor. It is not a classifier, judge, rationale teacher, segment annotator, or test-time module. The modern contribution is compiling label-blind MLLM observations into a certifiable global geometry target.

## Integration

The method attaches after the strongest same-protocol non-MLLM train bank `Z0` exists. It reuses the existing encoder/projection path and ordinary kNN evaluator. The only trainable parameters are existing encoder/projection parameters during uniform target fit.

## Training Plan

1. Hash paired REMOVE/comparator `Z0`, `G0`, train IDs, labels, and source.
2. Build and seal train-only MLLM cache.
3. Encode `Phi`, compute `K_C`, `Q`, `b_struct`, and cache stability.
4. Compute `rho_coord` and robust coverage diagnostics; robust constraints remain disabled unless coverage passes.
5. Run resource preflight under SLURM if implementation is authorized.
6. Solve the single convex projection.
7. Accept only with serialized H-metric normal-cone/KKT certificate.
8. Apply rank-tail audit.
9. Factor/Procrustes to `Z*`.
10. Fit encoder uniformly.
11. Run rollback/no-collapse/isolation checks.
12. Evaluate only through ordinary kNN.

Controls:

- `REMOVE`;
- `SHUFFLE`;
- `NOISE`;
- `DIRECT-MOMENT`;
- `DIRECT-CERT-FEATURE`;
- `SCALAR-PROPENSITY`.

The direct/scalar controls are attribution hypotheses. If any matches FULL under final attribution tests, the MLLM-global-geometry mechanism claim fails.

## Inference

Validation/test inference:

1. encode full video;
2. build train memory from full train-video embeddings and train labels;
3. retrieve ordinary top20 train neighbors;
4. apply the same vote rule as the comparator.

No MLLM cache, certificate file, compiler target, teacher output, auxiliary head, schema feature, key selector, or reranker is loaded.

## Why Direct and Scalar Controls Matter

The claim is not that scalar difficulty is mathematically incapable of affecting performance. The attribution hypothesis is narrower: if scalar missingness/uncertainty/difficulty or direct certificate-feature/moment use matches FULL, then the global proximal geometry interface is not supported as the mechanism.

FULL must beat `DIRECT-CERT-FEATURE`, `DIRECT-MOMENT`, and `SCALAR-PROPENSITY`; otherwise the MLLM-global-geometry claim fails even if raw metrics improve.

## Failure Modes and Diagnostics

- Cache contamination: any forbidden label/split/prediction/margin/ID/neighbor/loss/gradient/segment/timestamp/span/localization/rationale/target/stance/mechanism field appears; STOP.
- Cache instability: `sigma_cache` fails; certified null or STOP.
- Compute envelope exceeded; STOP.
- Solver lacks serialized normal-cone/KKT certificate; STOP.
- Rank-tail audit fails; terminal `ENCODER_RANK_GATE_FAIL`.
- Robust coverage low; robust constraints disabled and no safety claim.
- Certificate degenerates into sampling, weights, keys, pairs, triplets, SupCon, or reranking; reject.
- Uniform fit collapse; rollback REMOVE.
- Direct/scalar controls match FULL; mechanism unsupported.
- Final metric/statistical condition fails; no success claim.

## Novelty and Elegance

### Pseudo-groups and reweighting

Certificates define one full-bank structural moment, not groups or sample weights.

### Semantic pair supervision

The MLLM never labels pairs or chooses positives/negatives. Robust edges are G0 safety diagnostics only and disabled by default.

### Privileged distillation

The MLLM emits no logits, labels, rationales, targets, or teacher probabilities. Direct feature/moment routes are controls.

### Metric learning

The method is not pair/triplet/SupCon or relation-margin learning. It solves one global convex projection and fits the exact rank-gated target uniformly.

### Local rank-cell LB-SCGP

Local v7 is formally retired. No v8, local cells, signed-gap rescue, SLSQP near-miss success, or `NO_WITNESS` infeasibility inference exists.

## Claim-Driven Validation Sketch

### Block 1: Conceptual G0, real-fold gate, and teacher-cache gate

Claim tested: target construction is executable, certifiable, isolated, encoder-realizable, within resource envelope, and not a forbidden degeneration.

Required statuses:

```text
FULL_GLOBAL_TARGET_SYNTH              -> GLOBAL_TARGET_CERTIFIED
FULL_GLOBAL_TARGET_REAL_TRAIN         -> GLOBAL_TARGET_CERTIFIED or CERTIFIED_NULL
SOLVER_NO_KKT_CERT_FIXTURE            -> REJECTED_NO_OPTIMALITY_CERT
ENCODER_RANK_FAIL_FIXTURE             -> ENCODER_RANK_GATE_FAIL
RESOURCE_CAP_EXCEEDED_FIXTURE         -> REJECTED_RESOURCE_CAP
REMOVE_NO_CERT                        -> GLOBAL_TARGET_CERTIFIED_NULL
SHUFFLE_CERT_IDENTITY                 -> CONTROL_TARGET_CERTIFIED
NOISE_CERT_ATOMS                      -> CONTROL_TARGET_CERTIFIED
DIRECT_SAME_MOMENT                    -> DIRECT_CONTROL_BUILT_NOT_MAIN_METHOD
AMBIGUOUS_EDGE_FIXTURE                -> GEOMETRY_FAIL_OPEN_CLAIM_FAIL_CLOSED
ROBUST_EDGE_FIXTURE                   -> ROBUST_EDGE_CERTIFIED_IF_COVERAGE_PASS
SEGMENT_GOLD_INJECTION                -> REJECTED_SUPERVISION_VIOLATION
HELD_VAL_TEST_ACCESS_ATTEMPT          -> REJECTED_ISOLATION_VIOLATION
SAMPLE_WEIGHT_SELECTOR_ATTEMPT        -> REJECTED_DEGENERATION
RERANK_KEY_SELECTOR_ATTEMPT           -> REJECTED_DEGENERATION
PAIR_TRIPLET_SUPCON_ATTEMPT           -> REJECTED_DEGENERATION
```

No accuracy or macro-F1 claim is emitted by G0.

### Block 2: Mechanism attribution and controls

Arms:

FULL, REMOVE, SHUFFLE, NOISE, DIRECT-MOMENT, DIRECT-CERT-FEATURE, SCALAR-PROPENSITY.

Gate:

- FULL must beat REMOVE, SHUFFLE, NOISE, and strongest direct/scalar control by a preregistered dev/OOF margin in both metrics on both datasets.
- NOISE should remove gain monotonically.
- If direct/scalar controls match FULL, stop the mechanism claim.

### Block 3: Final paired performance gate

Freeze all code/cache/compiler/solver/rank/control/statistics before test.

Pass requires:

- MHC-EN and MHC-ZH;
- seeds 0/1/2;
- strongest same-protocol non-MLLM comparator;
- ordinary full-video train-memory top20 kNN;
- accuracy and macro-F1 each at least +0.030;
- all paired seed deltas positive;
- hierarchical paired bootstrap lower bound >0;
- Holm correction over four dataset×metric tests;
- FULL beats REMOVE, SHUFFLE, NOISE, and strongest direct/scalar attribution control;
- no test teacher/head/rerank/certificate artifact.

## Experiment Handoff Inputs

Freeze before implementation:

- schema `scgp_global_cert_v2`;
- prompt/input/model hashes;
- `Phi`, consensus, missingness columns;
- `orth_cap`;
- `A_struct`, `b_struct`, cache-stability gate;
- `rho_coord` and robust coverage gate;
- projection constants;
- exact KKT certificate payload schema;
- rank-tail audit thresholds;
- compute preflight caps;
- factor/Procrustes convention;
- uniform fit schedule and rollback;
- controls and final statistics.

Required records:

- cache closure manifest;
- operator manifest;
- robust coverage report;
- resource preflight report;
- solver primal and serialized KKT certificate;
- independent certificate verification;
- rank-tail audit;
- factor/Procrustes report;
- fit/rollback report;
- inference isolation report.

## Compute & Timeline

This refinement performs no experiments, no code implementation, and no SLURM work. It validates no result.

Prospective staged compute if authorized later:

- G0 synthetic/resource preflight: CPU-only SLURM.
- Real-fold G0: CPU dense linear algebra plus KKT verification, expected feasible at `N≈550-580` but must pass measured resource preflight.
- MLLM cache: train-only calls only.
- Uniform dry fit/training: GPU under project limits.
- Final: MHC-EN/MHC-ZH seeds 0/1/2 and controls, ordinary kNN endpoint.

STOP if any measured or projected stage exceeds 16 CPU, 128 GB, 2 GPU, or the configured project compute envelope.

## Grounding and Evidence Status

Grounding remains local: Phase 0 anchor, Round 1 refinement, Round 2 review, prior LB-SCGP v6/v7 reports, SSR-B01 negative evidence, RA-HMD/RGCL notes, MHC notes, and local MLLM usage landscape. No web search, implementation, experiment, or SLURM run was performed for this revision.

The global pivot remains unvalidated. Existing evidence supports only inherited isolation/replay/PSD/kNN discipline and local-v7 retirement. It does not prove the revised schema, `Q` interface, projection, KKT certificate, rank-tail gate, resource feasibility, uniform fit, or final MHC-EN/MHC-ZH performance.

## Self-Audit

- Immutable anchor copied verbatim: yes.
- No-fragment-gold sentence literal: yes.
- Only `parent_video_binary_label` gold: yes.
- One common `Q`, valid `vech`: yes.
- Closed strongly convex global projection: yes.
- Serialized H-metric normal-cone/KKT certificate only: yes.
- Finite VI probes diagnostic only: yes.
- Rank-tail audit and no truncation: yes.
- Rank failure terminal: yes.
- Robust safety disabled by default until coverage passes: yes.
- Direct/scalar controls decisive without impossibility overclaim: yes.
- Compute envelope under 16 CPU/128 GB/2 GPU specified: yes.
- ≤2 components, ≤2 claims, ≤3 blocks: yes.
- No local v8, segment route, sample weighting, key selection, pair/triplet/SupCon, test teacher/head/rerank: yes.
- Global pivot unvalidated: yes.
