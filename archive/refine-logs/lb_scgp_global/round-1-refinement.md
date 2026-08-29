# Round 1 Refinement

## Problem Anchor

hateful video detection adapting RGCL/RA-HMD to video; MLLM meaningful+novel; final MHC-EN/MHC-ZH seeds0/1/2 vs strongest same-protocol non-MLLM, acc and macro-F1 each ≥+0.030, all paired seed deltas positive, hierarchical paired bootstrap lower>0, Holm; only parent-video binary gold, no segment/timestamp/span/localization/stance/target/mechanism/rationale gold; train-only label-blind MLLM cache; test ordinary full-video train-memory top20 kNN, no teacher/head/rerank; SLURM; no sample weighting/key selection/pair-triplet/SupCon/segment route; REMOVE/SHUFFLE/NOISE/direct attribution. Local rank-cell v7 formally retired, no v8.

Absolutely do not assume any fragment/segment has gold annotation. The only gold is parent_video_binary_label. Any segment/timestamp/span/localization/stance/target/mechanism/rationale output is not gold and may not be treated as supervision, pseudo-groups, selection, or evaluation gold. Preserve this literally and operationally.

## Anchor Check

- Original bottleneck: a train-only, label-blind MLLM signal must change final full-video train-memory kNN geometry without becoming a verdict, teacher, head, reranker, selector, sample weight, pair/triplet objective, segment route, or local rank-cell solver.
- Revised method preservation: the method remains a two-component global pivot: restricted certificate cache/compiler, then one encoder-realizable full-bank PSD/unit-diagonal proximal target fitted uniformly by the existing encoder. Test inference remains ordinary full-video top20 kNN.
- Reviewer suggestions accepted: the mathematical blockers on rank realizability, operator dimensions, robust intervals, solver certificate, and soft-vs-hard constraints are valid and are fixed below.
- Reviewer suggestions rejected as drift: adding a learned graph, reasoning teacher, test-time MLLM, auxiliary head, relation adapter, pair/triplet/SupCon route, sample weighting, or any v8/local stationarity repair. The review did not request these; this refinement keeps them out.

## Simplicity Check

- Dominant contribution after revision: a closed train-only certificate cache defines a single encoder-realizable global proximal Gram target, not a bundle of local constraints.
- Components removed or merged: `A_eq` and `A_band` as separate slacked constraint families are removed. Their role is replaced by one cache-stability gate and one structural moment penalty in a common consensus basis.
- Components retained: restricted MLLM cache/compiler; global convex projection/factor/Procrustes/uniform fit.
- Components still excluded: local cells, local stationarity, SLSQP near-miss success, `NO_WITNESS` inference, segment route, sample weighting, key selection, pair/triplet/SupCon, test teacher/head/rerank.
- Why this remains the smallest adequate route: every Round 1 fix closes a mathematical interface in the same two-component method; none adds a new model or inference path.

## Changes Made

### 1. Encoder realizability

- Reviewer said: `G*` can have rank greater than the encoder dimension `d`; zero-padding only handles `rank(G*) < d`; exact reconstruction is invalid unless `rank(G*) <= d`.
- Action: accepted. The revised method adds a fail-closed post-projection rank gate `rank_eps(G*) <= d`. No hard rank constraint is added to the convex projection because `rank(G) <= d` is nonconvex.
- Reasoning/evidence: the reviewer is correct; a top-venue method cannot claim an exact encoder target if the target Gram is not realizable in `R^d`. A post-solve certification gate preserves convexity and avoids rank truncation overclaims.
- Impact on core method: if the rank gate fails, the run emits `ENCODER_RANK_GATE_FAIL`, no `Z*` is formed, no uniform fit is run, and no G0 GO or performance claim is allowed. If it passes, exact reconstruction is checked against `G*`.

### 2. Structural moment interface

- Reviewer said: cross-basis `vech(U^T(G-I)V)` is invalid for rectangular or non-symmetric cross moments; separate SVD bases are not coordinate-aligned.
- Action: accepted. The revised method uses one common consensus basis `Q` for all structural operators. Only the square symmetric matrix `Q^T(G-I)Q/N` uses `vech`. Replica stability is computed by projecting each replica certificate kernel into the same `Q` coordinates and is used as a cache-stability gate, not a cross-basis equality.
- Reasoning/evidence: this eliminates non-invariant SVD coordinate comparisons and all rectangular `vech` misuse while keeping the same global certificate geometry idea.
- Impact on core method: `A_eq` and `A_band` are merged away. The single structural interface is `r_struct = A_struct vec(G) - b_struct`, with `A_struct vec(G)=vech(Q^T(G-I)Q/N)`.

### 3. Robust intervals

- Reviewer said: intervals derived from `rho_row=0.05 sqrt(N-1)` are likely vacuous.
- Action: accepted. The revised projection adds a hard coordinate trust constraint `|G_ij-G0_ij| <= rho_coord` for all off-diagonal entries. Robust edges are certified from `G0` using this coordinate bound, not the row L2 bound. Robust coverage is reported as a G0 diagnostic/gate; low coverage means no safety claim.
- Reasoning/evidence: coordinate trust is convex, replayable, and gives a non-vacuous per-entry interval. It keeps robust rank/vote logic subordinate to safety diagnostics instead of making it a selected-pair training mechanism.
- Impact on core method: global structural geometry still runs even if robust coverage is low, but robust rank/vote constraints and exact-vote safety claims are omitted when coverage is insufficient.

### 4. Solver certificate

- Reviewer said: Dykstra replay traces are insufficient without a normal-cone/KKT or VI certificate in the H metric.
- Action: accepted. The revised method specifies primal feasibility, H-metric stationarity/normal-cone residual, dual cone membership, complementarity, objective/duality gap where applicable, and VI checks. The independent verifier must certify these, not only replay traces.
- Reasoning/evidence: the previous local route failed partly because near-witnesses were overinterpreted. The global route must make optimality evidence explicit and fail closed.
- Impact on core method: no G0 target can be accepted without a replayed KKT/VI certificate.

### 5. Soft versus hard structure

- Reviewer said: slacked equalities and bands are regularized preferences unless hard residual caps are added.
- Action: accepted. The revised method removes separate slacked equality/band constraints and uses a single structural residual variable with a quadratic penalty in the strongly convex projection objective. Cache stability is a hard pre-projection gate; structural matching inside the projection is explicitly a regularized preference, not exact certificate satisfaction.
- Reasoning/evidence: this simplifies the method and avoids pretending soft slacks are hard semantics.
- Impact on core method: the contribution remains global proximal geometry; the structural residual must improve and be reported, but it is not claimed as exact semantic satisfaction.

### 6. Novelty and attribution controls

- Reviewer said: direct certificate feature and scalar propensity controls must remain decisive.
- Action: accepted. The revised claims explicitly fail if `DIRECT-CERT-FEATURE`, `DIRECT-MOMENT`, or `SCALAR-PROPENSITY` matches FULL.
- Reasoning/evidence: without this, the MLLM could be reduced to a hand-engineered kernel, scalar difficulty, or privileged feature baseline.
- Impact on core method: final success requires attribution, not only raw metric gain.

### 7. Schema gold-boundary risk

- Reviewer said: stance-like field names could invite gold-boundary confusion.
- Action: accepted. The revised schema renames these as noisy structural observables, e.g. `source_alignment_observable`, `counter_context_observable`, and `context_shift_observable`. They are never stance, target, mechanism, rationale, or localization gold.
- Reasoning/evidence: the immutable anchor forbids treating such outputs as gold, pseudo-groups, selected keys/pairs, weights, or evaluation targets.
- Impact on core method: schema remains meaningful but is operationally bounded as noisy train-only certificate input.

### 8. Dimensional audit

- Reviewer said: every matrix shape and Procrustes convention must be explicit.
- Action: accepted. The revised proposal defines `Z0 in R^{N x d}`, `G0,G* in S^N`, `Q in R^{N x r}`, `A_struct in R^{m x N^2}`, `m=r(r+1)/2`, `Y in R^{N x d}`, and `R in O(d)`. `Y^T Z0` defines the Procrustes SVD.
- Reasoning/evidence: the previous text's zero-padding statement was incomplete.
- Impact on core method: exact realizability is now auditable.

### 9. Complexity cap

- Reviewer said: complexity cap is mostly respected but the constraint system could feel overbuilt.
- Action: accepted. The revised method deletes separate `A_eq/A_band`, demotes robust rank/vote to safety diagnostics, and keeps exactly two new components, two claims, and three experiment blocks.
- Reasoning/evidence: this keeps the paper centered on one mechanism.
- Impact on core method: no module pileup is introduced.

### 10. Global pivot and no forbidden routes

- Reviewer said: drift warning is none, but future revisions must not revive local cells, selected pairs, or stance/mechanism gold.
- Action: accepted and reinforced. The revised proposal explicitly preserves the global pivot and bans local v8/SLSQP/NO_WITNESS inference, segment routes, sample weighting, key selection, pair/triplet/SupCon, and test teacher/head/rerank.
- Reasoning/evidence: this is required by the immutable anchor and v7 pivot review.
- Impact on core method: robust rank edges remain G0-derived safety diagnostics, not MLLM semantic pair supervision or the main method.

### 11. Evidence status

- Reviewer said: the global pivot is unvalidated and must remain planned validation, not evidence.
- Action: accepted. The revised proposal repeatedly states that no experiment has yet validated the global pivot.
- Reasoning/evidence: inherited evidence supports only isolation/replay/PSD machinery and retirement of local rank-cell stationarity.
- Impact on core method: no performance, G0, or MLLM behavior claim is made before future gates.

## Revised Proposal

# Research Proposal: LB-SCGP Global-R1 — Encoder-Realizable Label-Blind Structural Certificates for a Single Full-Bank Proximal Gram Target

## Problem Anchor

hateful video detection adapting RGCL/RA-HMD to video; MLLM meaningful+novel; final MHC-EN/MHC-ZH seeds0/1/2 vs strongest same-protocol non-MLLM, acc and macro-F1 each ≥+0.030, all paired seed deltas positive, hierarchical paired bootstrap lower>0, Holm; only parent-video binary gold, no segment/timestamp/span/localization/stance/target/mechanism/rationale gold; train-only label-blind MLLM cache; test ordinary full-video train-memory top20 kNN, no teacher/head/rerank; SLURM; no sample weighting/key selection/pair-triplet/SupCon/segment route; REMOVE/SHUFFLE/NOISE/direct attribution. Local rank-cell v7 formally retired, no v8.

Absolutely do not assume any fragment/segment has gold annotation. The only gold is parent_video_binary_label. Any segment/timestamp/span/localization/stance/target/mechanism/rationale output is not gold and may not be treated as supervision, pseudo-groups, selection, or evaluation gold. Preserve this literally and operationally.

### Bottom-line problem

Build a meaningful, novel MLLM integration for adapting RGCL/RA-HMD to hateful-video detection while preserving the final endpoint as ordinary full-video train-memory top20 kNN. The method must improve final paired performance on MHC-EN and MHC-ZH, not merely produce a diagnostic, head gain, teacher score, local solver artifact, or rationale.

### Must-solve bottleneck

Prior MLLM routes either became verdict/rationale features, selectors over keys/pairs/samples, sample weighting, segment-adjacent routes, sparse relation losses, head/memory redistribution, or brittle local rank-cell certificate chasing. The bottleneck is to let train-only label-blind whole-video MLLM structural observations alter full-bank retrieval geometry without treating any MLLM field as gold and without selected pairs or local cells.

### Non-goals

No segment, timestamp, span, localization, stance, target, mechanism, or rationale supervision/evaluation claim. No test-time MLLM, teacher, head, reranker, router, score fusion, key selection, sample weighting, pair/triplet/SupCon addition, segment route, local rank-cell v8, SLSQP near-miss success, or `NO_WITNESS` infeasibility inference.

### Constraints

Only `parent_video_binary_label` is gold. MLLM outputs are noisy train-only label-blind structural observables, never annotations or gold. The compiler may read train parent labels only after the cache is sealed and hashed. Validation/test never load MLLM records, certificate-derived targets, compiler artifacts, teacher caches, auxiliary heads, or reranking code. All future compute must use SLURM in `HateVideo`, without `--time`.

### Success condition

Final success requires MHC-EN and MHC-ZH, seeds 0/1/2, strongest same-protocol non-MLLM comparator, accuracy and macro-F1 each at least +0.030, every paired seed delta positive, hierarchical paired bootstrap lower bound >0, and Holm correction. FULL must also beat REMOVE, SHUFFLE, NOISE, and the strongest direct/scalar attribution control under the same ordinary kNN endpoint.

## Technical Gap

RGCL/RA-HMD supply a retrieval-memory endpoint, but their geometry is driven by labels and embedding neighborhoods. Hateful-video errors often depend on full-video cross-modal structure: whether observed reference/predicate/source signals are aligned, whether evidence is direct or contextualized, and whether modalities agree. Earlier attempts converted such semantics into local pairs, selected edges, or rank cells. That made the MLLM signal sparse, selector-like, and entangled with pair/triplet metric learning or local solver fragility.

The missing interface is a global, encoder-realizable target: a sealed label-blind MLLM cache defines low-rank full-bank structural moments; a single closed convex projection produces a PSD/unit-diagonal Gram `G*`; a fail-closed rank gate ensures `G*` can be exactly realized in the encoder dimension; and the encoder fits the resulting `Z*` uniformly. Robust top20 intervals remain subordinate safety diagnostics, not the main method.

Inherited evidence supports isolation, no-segment discipline, exact kNN endpoint definitions, PSD/unit-diagonal/projector/factor/replay/hash patterns, and retirement of local rank-cell stationarity. It does not validate this global pivot. The restricted schema, consensus-basis operators, rank gate, coordinate trust, H-metric certificate, real-fold G0, uniform fit, and final performance remain unvalidated until future gates run.

## Route Comparison

### Elegant minimal route

Use a frozen train-only MLLM as a restricted structural sensor. Seal its cache. Compile noisy structural observables into one common-basis structural penalty over the full train Gram. Solve one convex PSD/unit-diagonal proximal projection from `G0` to `G*`, certify rank realizability, factor/Procrustes to `Z*`, and fit the existing encoder uniformly. Test remains ordinary full-video train-memory top20 kNN.

### Frontier-native route

Use a reasoning teacher, learned graph module, relation adapter, DPO/RL video MLLM, or test-time agent. This may be more expressive, but it tends to violate the endpoint constraints or become privileged distillation, a head, a router, schema-as-feature, or module pileup.

### Choice

Choose the elegant minimal route. The modern primitive is still meaningful: the MLLM produces train-only label-blind structural observations. The paper contribution is the encoder-realizable global proximal geometry interface, not a larger reasoning stack.

## Method Thesis

### One-sentence thesis

A sealed train-only label-blind MLLM structural cache can define a single encoder-realizable full-bank PSD/unit-diagonal proximal Gram target, fitted uniformly by the existing video encoder, such that ordinary kNN geometry improves only if the MLLM-global geometry signal beats direct certificate-feature and scalar propensity controls.

### Smallest adequate intervention

The final problem is retrieval geometry, so the intervention is one full-bank target in the same geometry. The revision removes separate slacked equality/band families and keeps a single structural penalty plus hard feasibility/safety gates. No new inference path, head, selector, pair loss, or local stationarity route is introduced.

### Foundation-model role

The MLLM is a train-only structural sensor. It does not output labels, logits, rationales, target names, mechanisms, selected keys, or pair decisions. It supplies noisy whole-video structural observables that are meaningful only after a deterministic compiler turns them into global Gram moments.

## Contribution Focus

### Dominant contribution

An encoder-realizable global structure-certified proximal geometry interface:

`closed label-blind certificate cache -> common-basis full-bank structural moment penalty -> one convex PSD/unit-diag target -> rank gate -> factor/Procrustes -> uniform encoder fit -> ordinary kNN`.

### Supporting contribution

A fail-open/fail-closed robust interval discipline: coordinate trust can certify robust top20 edges/queries for diagnostics and optional safety constraints; ambiguous edges impose no rank constraint and support no safety claim.

### Explicit non-contributions

No new MLLM reasoning method, no segment annotation, no relation adapters, no pair/triplet/SupCon mining, no new kNN rule, no auxiliary head, no local rank-cell solver, and no test-time MLLM.

## Complexity Budget

- New trainable components: zero architecture modules.
- New non-trainable components: exactly two.
  1. Restricted label-blind MLLM cache and deterministic compiler.
  2. One global convex target plus rank-certified factor/Procrustes uniform fit.
- Claims: at most two.
  1. G0 claim: the target is executable, replayable, isolated, encoder-realizable, nondegenerate, and not a forbidden degeneration.
  2. Final claim: ordinary-kNN performance and attribution pass the immutable success condition.
- Experiment blocks: exactly three.
  1. Conceptual G0 plus real-fold and teacher-cache gates.
  2. Mechanism attribution with direct/scalar controls.
  3. Final paired performance gate.

## System Overview

```text
train videos only
  -> uniform full-video evidence packs
       fixed frames + title/ASR/OCR; no label, split, margin, ID, neighbor, loss, gradient
  -> restricted deterministic MLLM certificate calls
  -> sealed train-only cache and Merkle root
  -> compiler opens train parent_video_binary_label only after cache closure
  -> common consensus basis Q and structural target b_struct
  -> one convex PSD/unit-diag projection with coordinate/row/class trust
  -> H-metric KKT/VI certificate and independent replay
  -> rank_eps(G*) <= d gate
  -> Y in R^{Nxd}, Procrustes R in O(d), target Z*
  -> uniform encoder fit
  -> validation/test ordinary full-video top20 train-memory kNN only
```

## Core Mechanism

### Dimensions and notation

For a sealed train fold:

- `N`: train-video count.
- `d`: encoder embedding dimension, read from `Z0.shape[1]`; not assumed.
- `x_i`: full train video input.
- `y_i in {0,1}`: parent video binary label, the only gold.
- `s_i=2y_i-1`.
- `Z0 in R^{N x d}`: frozen paired REMOVE/comparator train bank, row-normalized.
- `G0=Z0 Z0^T in S^N`: baseline Gram, unit diagonal.
- `G in S^N`: projection variable.
- `topk=20`: ordinary kNN endpoint.

All Gram variables are `N x N`; all encoder targets must live in `R^d`.

### Restricted certificate schema

The MLLM receives only whole-video evidence packs:

- deterministic uniform full-video frames;
- title if available;
- ASR/OCR text if available with deterministic truncation;
- no labels, predictions, correctness, margins, losses, gradients, split names, seed IDs, neighbor IDs, memory keys, or dataset statistics.

Strict JSON schema `scgp_global_cert_v2` contains noisy structural observables:

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

There is no free text, target name, proposition text, mechanism text, rationale, segment, timestamp, span, localization, or final hate verdict. The observable names are deliberately not stance/target/mechanism labels. They are noisy train-only structural observations and may not be used as gold, pseudo-groups, selected keys/pairs, weights, or evaluation targets.

`confidence` is an integer `0..4` used only for parse/agreement diagnostics and cache-stability gates. It is not a loss weight, sample weight, pair weight, or selector.

Cache protocol:

- Four deterministic calls per train video: two prompts times two evidence orders.
- Fixed decoding, model, processor, prompt, input-builder, and schema hashes.
- Parse failure or extra keys produce a canonical all-unresolved record.
- Consensus maps each ternary observable to `+1`, `-1`, or `0`.
- Missing/unresolved records remain in the full bank.
- Cache closure writes per-call JSON, consensus JSON, hashes, ID allowlist, and Merkle root.
- Labels first enter only after this closure.

### Parent label entry point

Parent labels enter after cache closure, only for:

1. train-only robust vote diagnostics/constraints when robust coverage is sufficient;
2. final ordinary kNN metric computation;
3. stratified reports and controls.

Labels never enter MLLM prompts, certificate acceptance, schema revision, sample selection, pair selection, key choice, or weight assignment.

### Certificate encoding

Let `R=4` be the replica count and `p` the final encoded column count.

For each replica `r`, encode JSON observables as `Phi^(r) in R^{N x p}`:

- ternary states use three one-hot columns or a fixed `{-1,0,+1}` contrast plus an unresolved indicator, frozen before G0;
- categorical modality binding uses one-hot columns;
- an explicit `record_missing_or_unresolved` column ensures every row has nonzero norm.

The consensus encoding is `Phi in R^{N x p}`. Row-normalize:

```text
V_i = Phi_i / ||Phi_i||_2
V_i^(r) = Phi_i^(r) / ||Phi_i^(r)||_2
K_C = V V^T in S^N
K_r = V^(r) (V^(r))^T in S^N
```

Thus `diag(K_C)=diag(K_r)=1`. No `alpha I_missing` term is needed; missingness is explicit in `Phi`.

Centering:

`H = I_N - 11^T/N`.

Common consensus basis:

```text
B0 = H Phi
Q = orth_cap(B0, r_max=8) in R^{N x r}
```

`orth_cap`:

- float64 thin SVD of `B0`;
- singular value threshold `sigma > max(1e-8, 1e-7 sigma_max)`;
- rank cap `r <= 8`;
- deterministic sign orientation by largest absolute entry, tie-broken by canonical video ID;
- if `r=0`, emit `GLOBAL_TARGET_CERTIFIED_NULL`.

All structural moments use this one `Q`. No separate replica SVD basis is compared.

### Structural operator

Define the square symmetric common-basis moment:

```text
M_Q(G) = Q^T (G - I_N) Q / N in S^r
```

Because `G` is symmetric and `Q` is common, `M_Q(G)` is square symmetric; `vech` is valid.

Let:

```text
m = r(r+1)/2
a_struct(G) = vech(M_Q(G)) in R^m
b_struct = vech(Q^T (K_C - I_N) Q / N) in R^m
A_struct vec(G) = a_struct(G)
r_struct = A_struct vec(G) - b_struct
```

Replica cache stability is a pre-projection gate:

```text
b_r = vech(Q^T (K_r - I_N) Q / N)
sigma_cache = max_r ||b_r - b_struct||_inf
```

Pass if:

`sigma_cache <= max(0.05, 0.25 ||b_struct||_inf)`.

The constants are frozen before outcomes and use only train-only cache geometry. Failure emits `CACHE_STABILITY_FAIL` or `GLOBAL_TARGET_CERTIFIED_NULL`; it is not repaired by prompt tuning after outcomes.

The projection uses only one structural preference:

`0.5 lambda_struct ||r_struct||_2^2`.

This is a regularized preference in the strongly convex objective, not a hard semantic equality. The method reports `||r_struct(G0)||`, `||r_struct(G*)||`, and improvement. It does not claim exact semantic satisfaction.

### Single closed convex global projection

Use product variables:

```text
X = (G, r_struct)
G in S^N
r_struct in R^m
X0 = (G0, 0)
```

H-metric objective:

```text
minimize_X
  0.5 ||G - G0||_F^2 + 0.5 lambda_struct ||r_struct||_2^2

subject to
  G = G^T
  diag(G) = 1
  G is PSD
  -1 + delta <= G_ij <= 1 - delta                  for i != j
  |G_ij - G0_ij| <= rho_coord                       for i != j
  ||G[i,-i] - G0[i,-i]||_2 <= rho_row               for all i
  ||mean_{i:y_i=c}(G[i,:]-G0[i,:])||_2 <= rho_class_c for c in {0,1}
  r_struct = A_struct vec(G) - b_struct
  optional robust rank halfspaces, only if robust coverage gate passes
  optional robust-query global/class vote means, only if robust coverage gate passes
```

Hard constraints are PSD, unit diagonal, off-diagonal box, coordinate trust, row trust, class trust, structural affine graph, and any included robust safety constraints. The structural match itself is a penalized objective term through `r_struct`.

This is one closed convex projection. There is no local cell enumeration, no hard rank constraint, no selected semantic pairs, and no local stationarity claim.

Default constants, frozen in G0:

- `delta = 1e-4`.
- `lambda_struct = 1` unless a pre-outcome synthetic scale calibration fixes another value.
- `rho_row = 0.05 sqrt(N-1)` retained as a global trust cap, not used for coordinate intervals.
- `rho_class_c = 0.02 sqrt(N)`.
- `tau = 1e-7`.
- `eta_edge = 1e-8` unless G0 preregisters a stricter value.

### Coordinate trust and robust intervals

Define G0 train-only rank gaps. For every query `q`, compute its canonical self-excluded G0 order. Let `D_q` be all required top20 internal adjacent edges and 20th-vs-outsider edges. For edge `(a before b)`, gap is:

`g_qab = G0_qa - G0_qb`.

Let `G_pos` be the multiset of positive `g_qab` over all train queries and required edges. If `G_pos` is empty, robust coverage is zero. Otherwise:

```text
g_ref = median(G_pos)
rho_coord = min(0.02, max(1e-4, 0.10 g_ref))
```

This uses only train-bank geometry, no labels, no outcomes, and no MLLM. The coordinate trust constraint is convex:

`|G_ij - G0_ij| <= rho_coord` for all `i != j`.

The induced interval is:

`I_qj = [G0_qj - rho_coord - eps_num, G0_qj + rho_coord + eps_num]`.

An edge `a before b` is robust if:

`G0_qa - G0_qb >= 2 rho_coord + 2 eps_num + tau + eta_edge`.

A query is robust if all required top20 internal and 20th-vs-outsider edges are robust. G0 reports:

- robust edge coverage;
- robust query coverage;
- robust query coverage by parent class after labels enter the compiler.

If robust query coverage is below the preregistered minimum, e.g. at least 10 robust queries per class and at least 10% of train queries, robust rank/vote constraints are omitted and no safety claim is made. Geometry still proceeds fail-open because structural global geometry is the main method.

When coverage passes, robust halfspaces may be added:

`G_qa - G_qb >= tau` for robust required edges.

They are G0-derived safety constraints, not MLLM-selected semantic pairs. Robust vote preservation uses only robust queries and parent labels after cache closure:

```text
m_q(G) = s_q / 210 * sum_{k=1}^{20} (21-k) s_{pi_q(k)} G[q, pi_q(k)]
```

Constrain robust global and class mean margins not to decrease from `G0`. These constraints are subordinate diagnostics/safety constraints and cannot be used as the main novelty claim.

### Solver and H-metric certificate

The producer may use deterministic product-space Dykstra/proximal splitting in scaled H coordinates or another preregistered convex solver, but acceptance requires the same certificate.

Let `C` be the intersection of all hard convex constraints in product space. The projection solution `X*` must satisfy:

`0 in H(X* - X0) + N_C(X*)`.

The verifier must certify:

1. Primal feasibility:
   - symmetry, diagonal, PSD, box, coordinate trust, row/class SOC trust, structural affine graph, and any robust halfspace/vote constraints;
   - maximum scalar residual `<=1e-6`;
   - PSD minimum eigenvalue `>= -1e-7`.

2. H-metric stationarity:
   - recover or verify dual/normal components for every active constraint family;
   - affine equality duals unrestricted;
   - box/coordinate/halfspace multipliers nonnegative;
   - SOC dual variables in their Lorentz cones;
   - PSD dual matrix `S_psd >= -1e-7 I`;
   - structural affine graph dual included for both `G` and `r_struct`;
   - normalized residual

     `res_stat = ||H(X*-X0) + n_total||_2 / (1 + ||H(X*-X0)||_2) <= 1e-6`,

     where `n_total` is the serialized normal-cone sum.

3. Complementarity:
   - nonnegative linear multiplier times residual `<=1e-6`;
   - SOC complementarity inner products `<=1e-6`;
   - PSD complementarity `|tr(S_psd G*)| <=1e-6`.

4. Dual cone membership:
   - all inequality multipliers `>= -1e-10`;
   - SOC duals satisfy cone residual `<=1e-7`;
   - PSD dual eigen minimum `>= -1e-7`.

5. Objective/duality or VI certificate:
   - if dual objective is available, relative primal-dual gap `<=1e-6`;
   - otherwise verify the variational inequality

     `<H(X*-X0), Y-X*> >= -1e-6 (1+||Y-X*||_2)`

     for deterministic feasible probes plus analytic active-face probes generated by the verifier. Random probes are diagnostic only; analytic probes are required where constraints have exact projectors.

6. Replay:
   - independent rebuild of all operators and constraints from sealed inputs;
   - exact payload hashes;
   - Dykstra/solver trace replay is supplementary, not sufficient.

If any certificate element fails, the status is fail-closed. No SLSQP near-miss, local signed-gap near-miss, or `NO_WITNESS` result is success or infeasibility.

### Encoder realizability and rank gate

After a certified convex projection, compute eigenvalues:

`G* = U diag(lambda) U^T`, with `lambda_1 >= ... >= lambda_N`.

Define:

```text
eps_rank = max(1e-8, 1e-7 lambda_1)
rank_eps(G*) = count(lambda_i > eps_rank)
```

Require:

`rank_eps(G*) <= d`.

This is a post-solve certification gate, not a convex constraint. It preserves the convex projection while preventing impossible encoder targets.

Failure:

- emit `ENCODER_RANK_GATE_FAIL`;
- do not truncate;
- do not form `Z*`;
- do not run uniform fit;
- do not claim G0 GO, safety, or performance.

Pass:

- reject if any `lambda_i < -1e-7`;
- clip only numerical negatives in `[-1e-7,0)`;
- let `r=rank_eps(G*)`;
- form `Y in R^{N x d}`:
  - first `r` columns are `U_r diag(sqrt(lambda_1..lambda_r))`;
  - remaining `d-r` columns are zero.

Then `Y Y^T` reconstructs `G*` within numerical tolerance. No rank-d approximation is used and no truncation claim is made.

### Procrustes convention and `Z*`

Given `Y in R^{N x d}` and `Z0 in R^{N x d}`, solve:

`R* = argmin_{R in O(d)} ||Y R - Z0||_F`.

Compute SVD:

`Y^T Z0 = L Sigma M^T`,

and set:

`R* = L M^T`.

Then:

`Z* = Y R* in R^{N x d}`.

Verifier checks:

- `||Z* (Z*)^T - G*||_F / max(1,||G*||_F) <= 1e-6`;
- row norm error `max_i | ||z_i*||_2 - 1 | <= 1e-6`;
- deterministic eigenspace orientation and Procrustes SVD tie rules by canonical ID and fixed LAPACK backend.

### Uniform encoder fit

The target-fit loss is:

`L_fit(theta) = (1/N) sum_i || normalize(f_theta(x_i)) - z_i* ||_2^2`.

Every train video has the same loss coefficient and schedule. No certificate row controls sample frequency, loss weight, pair/triplet construction, key selection, or reranking. If the same-protocol comparator has a base training phase, FULL starts from the paired REMOVE state and applies only the frozen uniform fit continuation.

Rollback:

- save model, optimizer, scheduler, scaler, RNG, sampler, and epoch cursor;
- if target residual, collapse, or ordinary train-kNN guards fail, restore and replay REMOVE;
- rollback hash must match direct REMOVE.

## Modern Primitive Usage

The MLLM is used only as a train-only, label-blind structural-observable generator. This is a foundation-model-era primitive, but it is constrained by deterministic schema, cache closure, and convex compilation. The MLLM is not a classifier, teacher, judge, rationale writer, segment annotator, or test-time agent.

## Integration

The method attaches after the strongest same-protocol non-MLLM train bank `Z0` is available. It reuses the existing video encoder/projection path and final kNN evaluator. The only trainable parameters are existing encoder/projection parameters during uniform target fit.

Frozen before final:

- splits, IDs, labels, and clean-subset protocol;
- certificate schema, prompt, MLLM model, input builder, and cache protocol;
- compiler constants, `rho_coord`, solver thresholds, rank gate, Procrustes convention;
- controls and statistical tests;
- final kNN endpoint.

## Training Plan

1. Produce paired REMOVE/comparator state and hash `Z0`, `G0`, train IDs, labels, and source.
2. Build train-only MLLM cache and seal before labels enter.
3. Encode certificates, compute `Q`, `K_C`, `b_struct`, cache-stability gate, coordinate trust, and optional robust coverage.
4. Solve the single convex projection and certify H-metric KKT/VI.
5. Apply `rank_eps(G*) <= d` gate.
6. Factor/Procrustes to `Z*`.
7. Fit encoder uniformly.
8. Run rollback/no-collapse/isolation checks.
9. Evaluate only through ordinary kNN.

Controls:

- `REMOVE`: no certificate-derived target.
- `SHUFFLE`: permute sealed certificate rows among train IDs while preserving atom marginals, missingness, and parent-label counts after cache closure.
- `NOISE`: corrupt certificate atoms at preregistered rates before consensus.
- `DIRECT-MOMENT`: directly optimize `||A_struct vec(G_theta)-b_struct||^2` with fixed coefficient, no proximal target.
- `DIRECT-CERT-FEATURE`: train-only cross-fit certificate-feature baseline with no test certificate access.
- `SCALAR-PROPENSITY`: matched control using only scalar missingness/uncertainty/difficulty summaries.

If any direct/scalar control matches FULL under the final attribution tests, the MLLM-global-geometry mechanism claim fails.

## Inference

Validation/test inference:

1. encode full video with the trained encoder;
2. build train memory from full train-video embeddings and train parent labels;
3. retrieve ordinary top20 train neighbors;
4. apply the same comparator vote rule.

No MLLM cache, certificate file, compiler target, teacher output, auxiliary head, schema feature, key selector, or reranker is loaded.

## Why the MLLM Signal Is Not Scalar Difficulty

Scalar difficulty gives `d in R^N` and can at most drive row-wise, diagonal-rank, or low-order scalar interactions. The proposed signal is a full-bank structural moment:

`vech(Q^T(G-I)Q/N)`.

It is multi-axis, label-blind, and aggregate over all train videos. It is not a sample weight and not selected pairs. The claim still fails if `SCALAR-PROPENSITY` or `DIRECT-CERT-FEATURE` matches FULL, because then the MLLM did not need the global proximal geometry interface.

## Failure Modes and Diagnostics

- Cache contamination: any label, split, prediction, margin, ID, neighbor, loss, gradient, segment, timestamp, span, localization, rationale, target, stance, or mechanism field in MLLM I/O is STOP.
- Cache instability: `sigma_cache` exceeds the frozen threshold; STOP or certified null.
- Structural null: `rank(Q)=0`, structural norm too small, or `G*` does not move materially; no MLLM geometry claim.
- Solver non-certification: primal/KKT/VI/dual/replay failure; STOP.
- Rank failure: `rank_eps(G*) > d`; STOP, no truncation.
- Robust coverage low: geometry continues, no robust safety/vote claim.
- Degeneration: certificates influence sample frequency, weights, keys, pairs, triplets, SupCon, or reranking; reject.
- Uniform fit collapse: rollback and replay REMOVE.
- Attribution failure: direct/scalar/shuffle/noise controls match FULL; mechanism unsupported.
- Final metric failure: any immutable final condition fails; no success claim.

## Novelty and Elegance

### Pseudo-groups and reweighting

Certificates do not define groups for robust optimization and do not assign weights. They define one common-basis full-bank structural moment. Every train video is fitted uniformly.

### Semantic pair supervision

The MLLM never labels pairs or chooses positives/negatives. Robust rank edges, when present, are G0-derived safety constraints and not semantic pair supervision.

### Privileged distillation

The MLLM does not output logits, labels, rationales, target names, or teacher probabilities. Direct feature/moment distillation exists only as controls.

### Metric learning

The method is not pair/triplet/SupCon or relation-margin learning. It solves one global convex projection and fits its encoder-realizable target uniformly.

### Local rank-cell LB-SCGP

Local v7 is formally retired. There is no v8, no local cell enumeration, no signed-gap tolerance rescue, no SLSQP near-miss success, and no `NO_WITNESS` infeasibility inference.

## Claim-Driven Validation Sketch

### Block 1: Conceptual G0, real-fold gate, and teacher-cache gate

Claim tested: global target construction is executable, replayable, isolated, encoder-realizable, nondegenerate, and not a forbidden degeneration.

Minimal experiment:

- synthetic fixtures for FULL, REMOVE, SHUFFLE, NOISE, robust/ambiguous edge behavior, rank-gate pass/fail, and forbidden access/degeneration;
- one sealed real train fold per dataset when authorized;
- independent compiler, solver, KKT/VI, rank, factor/Procrustes, dry-fit, rollback, and isolation verification.

Required statuses:

```text
FULL_GLOBAL_TARGET_SYNTH          -> GLOBAL_TARGET_CERTIFIED
FULL_GLOBAL_TARGET_REAL_TRAIN     -> GLOBAL_TARGET_CERTIFIED or CERTIFIED_NULL
ENCODER_RANK_FAIL_FIXTURE         -> ENCODER_RANK_GATE_FAIL
REMOVE_NO_CERT                    -> GLOBAL_TARGET_CERTIFIED_NULL
SHUFFLE_CERT_IDENTITY             -> CONTROL_TARGET_CERTIFIED
NOISE_CERT_ATOMS                  -> CONTROL_TARGET_CERTIFIED
DIRECT_SAME_MOMENT                -> DIRECT_CONTROL_BUILT_NOT_MAIN_METHOD
AMBIGUOUS_EDGE_FIXTURE            -> GEOMETRY_FAIL_OPEN_CLAIM_FAIL_CLOSED
ROBUST_EDGE_FIXTURE               -> ROBUST_EDGE_CERTIFIED
SEGMENT_GOLD_INJECTION            -> REJECTED_SUPERVISION_VIOLATION
HELD_VAL_TEST_ACCESS_ATTEMPT      -> REJECTED_ISOLATION_VIOLATION
SAMPLE_WEIGHT_SELECTOR_ATTEMPT    -> REJECTED_DEGENERATION
RERANK_KEY_SELECTOR_ATTEMPT       -> REJECTED_DEGENERATION
PAIR_TRIPLET_SUPCON_ATTEMPT       -> REJECTED_DEGENERATION
```

No accuracy or macro-F1 claim is emitted by G0.

### Block 2: Mechanism attribution and direct/scalar controls

Claim tested: any development gain is due to certificate identity and global proximal geometry, not continuation, random regularization, scalar difficulty, or direct certificate features.

Arms:

FULL, REMOVE, SHUFFLE, NOISE, DIRECT-MOMENT, DIRECT-CERT-FEATURE, SCALAR-PROPENSITY.

Gate:

- FULL must beat REMOVE, SHUFFLE, NOISE at zero corruption, and strongest direct/scalar control by a preregistered dev/OOF margin in both metrics on both datasets.
- NOISE should monotonically remove gain.
- If direct/scalar controls match FULL, stop the mechanism claim.

### Block 3: Final performance gate

Claim tested: immutable final success.

Setup:

- freeze all code, cache protocol, compiler constants, solver thresholds, rank gate, controls, seeds, and comparator before test;
- datasets MHC-EN and MHC-ZH;
- seeds 0/1/2;
- endpoint ordinary full-video train-memory top20 kNN;
- comparator strongest same-protocol non-MLLM, including moving REMOVE/control if stronger.

Pass requires:

- accuracy and macro-F1 each at least +0.030 on both datasets;
- all paired seed deltas positive;
- hierarchical paired bootstrap lower bound >0;
- Holm correction over four dataset×metric tests;
- FULL beats REMOVE, SHUFFLE, NOISE, and strongest direct/scalar attribution control;
- no test teacher/head/rerank/certificate artifact loaded.

## Experiment Handoff Inputs

Freeze before implementation:

- schema `scgp_global_cert_v2`;
- prompt/evidence order/input builder/model hashes;
- atom encoding `Phi`, consensus rules, missingness columns;
- `orth_cap` rank and sign rules;
- `A_struct`, `b_struct`, cache-stability gate;
- `rho_coord` construction from G0 rank gaps;
- all projection constants and H-metric certificate tolerances;
- rank gate `eps_rank`;
- factor/Procrustes convention;
- uniform fit schedule and rollback manifest;
- control construction and final statistics.

Required machine records:

- `Z0`, `G0`, IDs, labels hash;
- certificate per-call and consensus cache;
- compiler manifest;
- operator and robust coverage report;
- solver output and H-metric KKT/VI certificate;
- rank-gate report;
- factor/Procrustes report;
- fit/rollback report;
- inference isolation report.

## Compute & Timeline

This refinement runs no experiments and validates no global-pivot result.

Expected future compute if authorized:

- conceptual G0 synthetic: CPU-only SLURM, minutes to low hours;
- real-fold G0: CPU-heavy projection plus small GPU dry-fit, no performance claim;
- train-only MLLM cache: bounded to train videos only;
- attribution controls: dev/OOF arms, likely tens of GPU-hours if full encoder continuation is required;
- final: MHC-EN/MHC-ZH seeds 0/1/2 under ordinary kNN endpoint.

All computation must use SLURM, `conda activate HateVideo`, and no `--time`.

## Grounding and Evidence Status

Grounding remains local: Round 0 proposal, Round 1 review, Phase 0 anchor, prior LB-SCGP v6/v7 reports, SSR-B01 negative evidence, RA-HMD/RGCL video notes, MHC notes, and local MLLM usage landscape. No new web search, experiment, code implementation, or SLURM run was performed for this revision.

The global pivot remains unvalidated. Existing evidence supports only inherited isolation/replay/PSD/kNN discipline and the retirement of local rank-cell stationarity. It does not prove the revised schema, consensus-basis moment interface, coordinate trust, H-metric certificate, rank gate, uniform fit, or final MHC-EN/MHC-ZH performance.

## Self-Audit

- Immutable anchor copied verbatim: yes.
- No-fragment-gold sentence literal: yes.
- One global convex projection: yes.
- Hard rank constraint avoided; fail-closed rank gate added: yes.
- Cross-basis `vech` removed; common basis `Q` used: yes.
- Coordinate trust added; robust safety subordinate and coverage-gated: yes.
- H-metric KKT/VI certificate specified: yes.
- Slacked equality/band families removed; single structural penalty used: yes.
- Direct/scalar controls decisive: yes.
- Schema atoms renamed as noisy structural observables: yes.
- Matrix dimensions audited: yes.
- At most two new components, two claims, three blocks: yes.
- No local v8, segment route, sample weighting, key selection, pair/triplet/SupCon, or test teacher/head/rerank: yes.
- Global pivot unvalidated: yes.
