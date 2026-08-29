# Research Proposal: LB-SCGP Global — Label-Blind Structural Certificates for a Single Full-Bank Proximal Gram Target

## Problem Anchor

hateful video detection adapting RGCL/RA-HMD to video; MLLM meaningful+novel; final MHC-EN/MHC-ZH seeds0/1/2 vs strongest same-protocol non-MLLM, acc and macro-F1 each ≥+0.030, all paired seed deltas positive, hierarchical paired bootstrap lower>0, Holm; only parent-video binary gold, no segment/timestamp/span/localization/stance/target/mechanism/rationale gold; train-only label-blind MLLM cache; test ordinary full-video train-memory top20 kNN, no teacher/head/rerank; SLURM; no sample weighting/key selection/pair-triplet/SupCon/segment route; REMOVE/SHUFFLE/NOISE/direct attribution. Local rank-cell v7 formally retired, no v8.

Absolutely do not assume any fragment/segment has gold annotation. The only gold is parent_video_binary_label. Any segment/timestamp/span/localization/stance/target/mechanism/rationale output is not gold and may not be treated as supervision, pseudo-groups, selection, or evaluation gold. Preserve this literally and operationally.

### Bottom-line problem

The project needs a genuinely MLLM-dependent adaptation of RGCL/RA-HMD to hateful-video detection that improves the final ordinary full-video train-memory top20 kNN endpoint. The MLLM must be meaningful and novel, but the final test path must be an ordinary encoder plus train-memory kNN vote with no test MLLM, teacher, head, or reranker.

### Must-solve bottleneck

Prior routes failed because the MLLM signal either collapsed into a scalar score/rationale feature, became a selector over keys/pairs/samples, depended on segment-like assumptions, shifted accuracy between a head and memory, or chased a brittle local rank-cell certificate. The remaining bottleneck is to make train-only label-blind MLLM semantics affect the complete train-bank geometry in a way that is global, replayable, removable, and not reducible to sample weighting or semantic pair supervision.

### Non-goals

This is not a localization, timestamp, span, rationale, target, stance, mechanism, or segment-supervision method. It is not a v8 continuation of local rank-cell stationarity. It is not a pair/triplet/SupCon loss, key selector, sample weighting scheme, teacher distillation route, native-head route, reranking method, or test-time MLLM agent.

### Constraints

Only `parent_video_binary_label` is gold. MLLM outputs are train-only, label-blind structural certificates and are never gold annotations. The compiler may read train parent labels only after the cache is sealed and hashed. Validation/test never load MLLM records, certificate-derived targets, compiler artifacts, teacher caches, auxiliary heads, or reranking code. All future compute must use SLURM in `HateVideo`, without `--time`, within the project resource limits.

### Success condition

Final success requires MHC-EN and MHC-ZH, seeds 0/1/2, strongest same-protocol non-MLLM comparator, accuracy and macro-F1 each at least +0.030, every paired seed delta positive, hierarchical paired bootstrap lower bound >0, and Holm correction. FULL must also beat REMOVE, SHUFFLE, NOISE, and the strongest direct-attribution control under the same ordinary kNN endpoint.

## Anchor Check

- Original bottleneck: a train-only MLLM must change final retrieval geometry without becoming a selector, teacher, test-time module, or segment/fragment supervision path.
- Preservation in this proposal: certificates define only global full-bank structural operators; all target fitting is uniform across train videos; final inference is unchanged full-video top20 kNN.
- Drift explicitly rejected: local rank-cell v8 repair, signed-gap tolerance rescue, SLSQP near-miss success, NO_WITNESS-as-infeasible, MLLM verdicts, certificate-derived sample weights, key/pair selection, segment-level supervision, test-time teacher/head/rerank.

## Simplicity Check

- Dominant contribution: one global PSD/unit-diagonal proximal Gram target defined by a closed train-only label-blind structural certificate cache, then uniformly fitted by the encoder.
- New components: exactly two.
  1. Restricted label-blind MLLM structural certificate cache and deterministic compiler.
  2. Single global convex Gram projection plus factor/Procrustes uniform encoder fit.
- Components intentionally not added: local cells, pair/triplet/SupCon losses, relation adapters, routers, heads, learned graph networks, sample weights, segment models, teacher distillation, test-time MLLM.
- Claims capped at two: executable global certificate geometry, and final ordinary-kNN performance/attribution if later validation passes.

## Technical Gap

RGCL and RA-HMD already establish retrieval-shaped representation learning and train-memory kNN as a strong endpoint, but their training geometry is driven by labels and embedding neighborhoods. In hateful videos, the hard errors are often not just "near duplicate but opposite label"; they arise from structural cross-modal relations such as whether a harmful predicate is actually bound to a referenced group, whether a speaker/source is endorsing or quoting/condemning, or whether visual/text/audio evidence are aligned. Prior local attempts tried to turn such semantics into selected relation edges or local rank-cell constraints. That made the MLLM signal sparse, brittle, selector-like, and easy to confuse with pair/triplet metric learning.

The missing mechanism is a full-bank interface: a sealed label-blind MLLM cache should define aggregate geometry over every train video at once, not choose which neighbors matter. The resulting target must be a single convex projection in PSD correlation-matrix space, with ordinary kNN safety claims only where G0 interval analysis proves them robust. Ambiguous rank edges must fail open for constraints and fail closed for claims.

Existing inherited evidence supports parts of this route, but not the global pivot itself:

- Reusable: parent-video-label-only supervision isolation, no segment-gold boundary, exact top20 kNN endpoint, PSD/unit-diagonal/projector/factor/replay/hash discipline, no-clobber logs, negative-control thinking, and strict fail-closed statuses.
- Retired: local rank-cell stationarity, strict signed-gap near-miss as success, SLSQP near-miss as success, `NO_WITNESS` as infeasibility, local signed-gap route, and any v8 solver tuning.
- Not yet validated: the proposed global target, the new certificate schema, real-fold global G0, and final performance.

## Route Comparison

### Route A: elegant minimal route

Use a frozen train-only MLLM only to emit a restricted whole-video structural certificate. Seal the cache. Compile the certificates into low-rank global operators over the full train-bank Gram matrix. Solve one closed convex PSD/unit-diagonal proximal projection from the baseline Gram `G0` to `G*`. Factor `G*` into `Z*`, align it to the current encoder bank by Procrustes, and fit the encoder uniformly to all train targets. Test with the ordinary full-video train-memory top20 kNN vote.

Strengths:

- One mechanism from certificate to geometry to ordinary kNN.
- No selector, no local cells, no segment route, no pair/triplet/SupCon addition.
- Replayable and fail-closed with inherited PSD/projector/hash machinery.
- MLLM is meaningful because it defines global structural operators that are absent without certificates.

Risk:

- The structural operators may be too weak or too constrained to move the encoder enough for +0.030/+0.030 on both datasets.

### Route B: frontier-native route

Train or prompt a stronger VLM/MLLM as a video reasoning teacher, produce rationales/structured decisions, distill into the encoder or a head, or add a learned graph/adapter module over certificate states.

Strengths:

- More expressive, closer to current reasoning-VLM literature.
- Could fit complex cultural/implicit hate patterns directly.

Risks:

- Violates the spirit or letter of the endpoint constraints: it tends toward privileged distillation, auxiliary heads, schema-as-feature, test-time teacher dependence, router behavior, or a learned module pile-up.
- Harder to separate from HVGuard/RAMF/IARE/TANDEM-style rationale/structured-output usage.
- Easier to overfit the small MHC clean subsets and harder to prove the MLLM is not simply a scalar difficulty teacher.

### Chosen route

Choose Route A. The project needs a narrow, defensible mechanism under strict no-segment/no-test-teacher/no-selector constraints. A modern MLLM is still central, but its role is certificate generation for a global convex geometry compiler, not free-form reasoning or prediction.

## Method Thesis

### One-sentence thesis

A sealed train-only, label-blind MLLM structural certificate cache can define a single replayable full-bank PSD/unit-diagonal proximal Gram target that a video encoder fits uniformly, yielding ordinary kNN geometry that cannot be reproduced by scalar difficulty, sample weighting, selected pairs, or direct certificate-feature use.

### Why this is the smallest adequate intervention

The bottleneck is global retrieval geometry, so the intervention is exactly one global target in the same geometry. The method does not add a new inference path, learned selector, relation-specific metric, or head. It only changes the train-bank target that the existing encoder is asked to realize.

### Why this is timely in the foundation-model era

Recent MLLM/video-hate methods mostly use reasoning text, verdicts, structured outputs, or teacher-style supervision. This proposal uses the foundation model as a restricted, label-blind structural sensor whose outputs are compiled into a convex representation target. That is a stronger and cleaner role than rationale-as-feature, but still removable at test time.

## Contribution Focus

### Dominant contribution

Global structure-certified proximal geometry: a deterministic compiler from train-only label-blind MLLM certificates to one full-bank PSD/unit-diagonal Gram target, followed by uniform encoder fitting and ordinary kNN inference.

### Optional supporting contribution

Replayable failure discipline for robust/ambiguous top20 interval safety: robust edges may constrain and support safety claims; ambiguous edges impose no rank constraint and support no safety claim.

### Explicit non-contributions

The proposal does not claim new MLLM reasoning, new segment annotation, new pair/triplet mining, new SupCon loss, new kNN voting, new local rank-cell solver, or new test-time adaptation.

## Complexity Budget

- New trainable components: zero new architecture modules; the encoder is the existing trainable encoder/MLP path used by the same-protocol comparator.
- New non-trainable components: two.
  1. Label-blind MLLM certificate cache plus deterministic compiler.
  2. Global convex Gram projection plus uniform target fitting.
- Core claims: two.
  1. G0 claim: the global certificate target is executable, replayable, isolated, nondegenerate, and not a forbidden degeneration.
  2. Final claim: after staged validation, FULL improves final ordinary kNN performance and beats attribution controls under the immutable success condition.
- Core experiment blocks: three.
  1. Conceptual G0 plus real-fold and teacher-cache gates.
  2. Mechanism attribution and direct/scalar controls.
  3. Final paired performance gate.

## System Overview

```text
train videos only
  -> uniform full-video evidence pack
       fixed frames + title/ASR/OCR, no labels, no IDs, no margins, no neighbours
  -> deterministic restricted MLLM certificate calls
  -> sealed cache: schema hashes, prompt hashes, input hashes, Merkle root
  -> compiler opens train parent_video_binary_label for vote baselines only
  -> global operators A_eq, A_band, A_reg, robust intervals, trust/box/vote constraints
  -> one convex PSD/unit-diag Gram projection: G0 -> G*
  -> PSD factorization + Procrustes: G* -> Z*
  -> uniform encoder fit over all train videos
  -> test: encode full test video, retrieve train-memory top20, ordinary vote
```

No certificate, teacher output, compiler target, or auxiliary head is loaded during validation/test inference.

## Core Mechanism

### Notation

For a sealed train fold:

- `N`: number of train videos.
- `d`: encoder embedding dimension, expected `d=1024` for the existing RGCL-style projection path unless the frozen comparator uses a different documented dimension.
- `x_i`: full train video input for video `i`.
- `y_i in {0,1}`: parent video binary label; the only gold.
- `s_i = 2y_i - 1 in {-1,+1}`.
- `f_theta(x_i)`: existing encoder/projection output.
- `z_i = normalize(f_theta(x_i)) in R^d`.
- `Z0 in R^{N x d}`: frozen baseline train bank from the paired REMOVE/comparator state.
- `G0 = Z0 Z0^T`: baseline full train Gram, symmetric with unit diagonal.
- `G`: optimization variable in `S^N`, the target train Gram.
- `topk = 20`; ordinary final vote uses full-video train-memory top20.

### Restricted MLLM certificate schema

The MLLM receives one whole train video evidence pack:

- uniformly sampled full-video frames;
- title if available;
- ASR/OCR text if available, truncated deterministically by the input builder;
- no label, prediction, correctness, margin, loss, gradient, split name, seed, neighbor/key, memory ID, or dataset statistic.

The output is strict JSON with no free text fields, no target names, no proposition text, no mechanism text, no rationale, no segment, no timestamp, no span, and no localization. Any extra key is parse failure.

Schema `scgp_global_cert_v1`:

```json
{
  "schema_version": "scgp_global_cert_v1",
  "visual_group_reference": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "text_audio_group_reference": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "derogatory_or_exclusion_predicate": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "threat_or_dehumanizing_action": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "cross_modal_reference_predicate_binding": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "speaker_source_endorsement": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "quotation_condemnation_reportage_exception": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "satire_reclaimed_or_counter_speech_exception": {"state": "supported|contradicted|unresolved", "confidence": 0},
  "modality_binding_pattern": {
    "state": "visual_text|visual_audio|text_audio|multi_modal|single_modal|unresolved",
    "confidence": 0
  },
  "parse_flags": []
}
```

`confidence` is an integer `0..4`. It is used only for cache validity, agreement diagnostics, and fail-closed schema QA. It is not a sample weight, loss weight, pair weight, key score, or selector.

Deterministic cache protocol:

- `R=4` calls per train video: two semantically equivalent prompts times two evidence orders.
- Decoding fixed: deterministic decoding, fixed max tokens, fixed model and processor hashes.
- Consensus maps each atom to `+1` for supported, `-1` for contradicted, `0` for unresolved/missing.
- A record is accepted if all required keys parse, no forbidden key appears, and nonzero atom modal agreement is at least `3/4` where the atom is used. Otherwise it becomes a canonical all-unresolved record.
- Accepted and missing records both remain in the full bank. Missingness is encoded as schema state, not as exclusion.
- The cache writes per-call JSON, consensus JSON, input-builder hash, prompt hash, model hash, code hash, ID allowlist, and a Merkle root.
- The cache is sealed before any train label is opened by the compiler.

### When parent labels first enter

Parent video labels first enter after the sealed cache root exists. They enter only the deterministic compiler, and only for:

1. train parent-label vote-margin baselines and global/class vote preservation constraints;
2. final train-memory kNN evaluation on train/validation/test endpoints;
3. stratified reports and controls.

Labels do not enter MLLM prompts, MLLM cache repair, certificate acceptance, sample selection, pair selection, key choice, or certificate schema revision.

### From certificates to global structure operators

Let `C^(r) in {-1,0,1}^{N x p}` be the atom matrix from call replica `r`, and let `C in {-1,0,1}^{N x p}` be the consensus atom matrix. Include explicit missing/unresolved indicator columns so missing records remain ordinary full-bank rows.

Define the centering matrix:

`H = I_N - (1/N) 11^T`.

Build deterministic orthonormal bases:

```text
Q      = orth_cap(H C, r_max=8)
Q^(r)  = orth_cap(H C^(r), r_max=8)
```

`orth_cap` uses float64 SVD, drops singular values below `1e-8` times the largest singular value, caps rank at 8, and orients signs by canonical video ID. If the accepted rank is zero, the compiler emits `GLOBAL_TARGET_CERTIFIED_NULL` and the FULL method is not allowed to claim MLLM geometry.

Define a label-blind certificate kernel:

```text
K_C = normalize_corr(C C^T + alpha I_missing)
diag(K_C) = 1
```

where `alpha` is fixed before outcomes and only prevents all-unresolved rows from becoming undefined. It is not tuned by dataset or metric.

For any basis pair `U,V`, define the aggregate Gram moment:

`M_{U,V}(G) = vech(U^T (G - I_N) V / N)`.

The compiler emits:

- `A_reg vec(G) - b_reg`: consensus structural moment residual, with
  `A_reg vec(G) = M_{Q,Q}(G)` and `b_reg = M_{Q,Q}(K_C)`.
- `A_eq vec(G) - b_eq = u_eq`: prompt/evidence-order aggregate equality residuals, comparing corresponding low-rank moments from `Q^(r)` and `Q^(s)` across the four replicas. `b_eq=0`. The slack `u_eq` is penalized in the projection objective.
- `lower_band <= A_band vec(G) + u_band <= upper_band`: structural stability bands from the min/max of replica-derived moment coordinates plus a preregistered numerical guard. The slack `u_band` is penalized.

Important boundary:

- `C`, `Q`, and `K_C` do not define pseudo-groups.
- No certificate row selects a sample, neighbor, pair, triplet, or training weight.
- `A_reg`, `A_eq`, and `A_band` are full-bank linear operators over `vec(G)`. Every train video remains present under the same optimizer and same target-fit rule.
- Certificate atoms are not evaluated as gold stance, target, mechanism, rationale, or localization.

### Single closed convex global projection

Lift the problem into product space so the whole solver is one projection, not a local-cell search.

Variables:

```text
G in S^N
u_eq in R^{m_eq}
u_band in R^{m_band}
r_reg in R^{m_reg}
```

Scaled product norm:

```text
||X-X0||_H^2 =
    ||G-G0||_F^2
  + lambda_eq   ||u_eq||_2^2
  + lambda_band ||u_band||_2^2
  + lambda_reg  ||r_reg||_2^2
```

with `X0=(G0,0,0,0)`. Fixed default coefficients are set before outcomes, e.g. `lambda_eq=lambda_band=lambda_reg=1`, with sensitivity only in the planned controls if authorized later. No coefficient is selected on validation/test.

Projection:

```text
minimize_X 0.5 ||X-X0||_H^2

subject to
  G = G^T
  diag(G) = 1
  G is PSD
  -1 + delta <= G_ij <= 1 - delta                         for i != j
  ||G[i,-i] - G0[i,-i]||_2 <= rho_row                      for all train i
  ||mean_{i:y_i=c} (G[i,:]-G0[i,:])||_2 <= rho_class_c      for c in {0,1}
  A_reg vec(G) - b_reg = r_reg
  A_eq vec(G) - b_eq = u_eq
  lower_band <= A_band vec(G) + u_band <= upper_band
  robust rank edges E_rob vec(G) >= gamma_rob               only where interval-certified
  robust-query global vote margin mean does not decrease
  robust-query class vote margin means do not decrease
```

This is a closed convex projection over a PSD cone, affine sets, boxes, second-order trust balls, and sparse linear halfspaces. There is no local rank-cell enumeration and no stationarity claim over a selected cell.

Fixed constants:

- `delta = 1e-4` off-diagonal guard unless inherited G0 evidence requires the stricter old value.
- `rho_row = 0.05 sqrt(N-1)` as an initial inherited trust scale.
- `rho_class_c = 0.02 sqrt(N)` as an initial inherited class-mean trust scale.
- `tau = 1e-7` rank tie tolerance.
- `eta_edge > 0`, preregistered in G0, for robust interval safety.

### Robust interval construction

For train query `q` and train memory candidate `j != q`, define a feasible score interval from the frozen baseline score and row trust:

```text
I_qj = [G0_qj - rho_row - eps_num, G0_qj + rho_row + eps_num]
```

where `eps_num` accounts for deterministic float64 rounding and canonical-ID tie handling. If tighter row-specific bounds are available from the finalized operator norms, G0 may use them, but they must be computed before outcomes and replayed independently.

An edge `a before b` for query `q` is robust iff:

`lo_qa - hi_qb >= tau + eta_edge`.

A query has robust top20 membership/order only if every required internal top20 edge and every 20th-vs-outsider edge is robust. Robust edges may be added to `E_rob`. Ambiguous edges add no constraint and support no exact-vote safety claim.

Vote preservation uses only robust queries. Let `pi_q(r)` be the fixed robust top20 order from `G0`, `w_r=21-r`, and `s_i=2y_i-1`. Define:

`m_q(G)= s_q / 210 * sum_{r=1}^{20} w_r s_{pi_q(r)} G[q, pi_q(r)]`.

The projection constrains the global mean and class-conditioned means of `m_q(G)` over robust queries to be no lower than their `G0` values. Ambiguous queries remain in the target and encoder fit, but make no safety claim.

### Solver

G0 uses a deterministic float64 product-space Dykstra/proximal projection implementation, reusing inherited exact projectors where possible:

- symmetry/diagonal projector;
- PSD eigenvalue clipping projector;
- off-diagonal box projector;
- row trust SOC projector;
- class-mean trust SOC projector;
- sparse halfspace projector for robust rank and vote constraints;
- affine graph projector for `A_reg`, `A_eq`, and band slacks using low-rank normal equations;
- capped-simplex or box projectors only if a later slack-budget variant is explicitly preregistered.

Frozen cyclic set order is part of the design hash. Stop only after a full cycle when:

- independent max set violation `<=1e-6`;
- relative iterate change `<=1e-7`;
- all finite-value, symmetry, PSD, diagonal, box, trust, structural, robust-edge, and vote residual checks pass;
- max cycles not exceeded.

If the solver does not certify, the status is fail-closed. No SLSQP near-miss, local signed-gap near-miss, or `NO_WITNESS` status may be reinterpreted as success.

### Deterministic replay and hashes

Every G0 run emits:

- input manifest hash: IDs, `G0`, labels after cache closure, certificate cache root, config;
- operator hashes: `A_reg`, `b_reg`, `A_eq`, `b_eq`, `A_band`, bands, `E_rob`, `gamma_rob`;
- solver trace hash: set order, projection residuals, correction norms, cycle count;
- output hash: `G*`, slacks, residual summary;
- source/dependency hash: implementation, BLAS/LAPACK backend, NumPy/SciPy/PyTorch versions if used;
- isolation counters: no held/validation/test content or label opens; no segment artifact opens; no teacher cache read/write outside train-only cache construction.

An independent verifier must rebuild all operators from sealed records and recompute:

- objective and variational inequality residual;
- PSD eigen minimum, diagonal error, symmetry error, box residual;
- trust residuals;
- structural equality/band/reg residuals;
- robust edge residuals;
- global/class robust vote margin residuals;
- ambiguous-edge absence from constraints and claims;
- payload hashes.

### Nondegeneration

The target is rejected or downgraded to null if any condition holds:

- `rank(Q)=0` or structural operator norm below preregistered minimum;
- `||G*-G0||_F / ||G0||_F < eps_move`, meaning the MLLM signal produced no material target;
- `||G*-G0||_F / ||G0||_F > eps_trust`, meaning trust caps failed to prevent large distortion;
- effective rank, row-norm reconstruction, class robust vote means, or train-bank duplicate checks indicate collapse;
- factorization cannot reconstruct `G*` within `1e-6`;
- uniform fit cannot reduce target residual in a dry block without collapse.

These are gates, not tunable rescue knobs.

### Factorization, Procrustes, and `Z*`

After certification:

1. Eigendecompose `G* = U Lambda U^T` in float64.
2. Reject if any eigenvalue `< -1e-7`; clip only numerical negatives in `[-1e-7,0)`.
3. Form `Y = U Lambda_+^{1/2}` and zero-pad to `d` dimensions if rank `< d`.
4. Solve orthogonal Procrustes:

   `R* = argmin_{R^T R=I} ||Y R - Z0||_F`.

5. Set `Z* = Y R*`, with deterministic sign/orientation for repeated eigenspaces.
6. Verify `||Z* Z*^T - G*||_F / ||G*||_F <= 1e-6` and row-norm error `<=1e-6`.

### Uniform encoder fit

The encoder target objective is:

`L_fit(theta) = (1/N) sum_i || normalize(f_theta(x_i)) - z_i* ||_2^2`.

Every train video has the same loss form and coefficient. There is no certificate-derived sample weight, pair, triplet, selected neighbor, key, or SupCon term. If the existing same-protocol comparator has a required base classification or RGCL pretraining phase, FULL starts from the paired REMOVE state and applies this uniform target-fit continuation under a frozen schedule. The MLLM component adds only the global target.

Rollback:

- Save model, optimizer, scheduler, scaler, RNG, sampler cursor, and epoch cursor before target fitting.
- If fit residual/collapse/ordinary train kNN guards fail, restore and replay REMOVE; replay hash must match direct REMOVE.

## Modern Primitive Usage

- Primitive: frozen MLLM/VLM used as a label-blind structural certificate generator.
- Role: structural sensor for global geometry, not judge, teacher, classifier, rationale generator, annotator, selector, or test-time agent.
- Why natural: hateful-video errors often depend on cross-modal structure that an MLLM can recognize better than scalar CLIP/Qwen embeddings, but the project constraints require the signal to be train-only, auditable, and removable. A restricted certificate-to-convex-geometry compiler gives the MLLM a meaningful role without letting it become the endpoint.

## Integration

### Base pipeline

The method attaches after the paired strongest same-protocol non-MLLM train-bank state is available for a fold/seed. It reuses:

- full-video train inputs;
- existing encoder/projection path;
- existing ordinary train-memory kNN endpoint;
- existing SLURM-only execution discipline;
- inherited hash/replay/isolation patterns.

### What is frozen

- Dataset splits and clean-subset protocol.
- Parent labels.
- Train-only input builder and certificate schema after G0 design.
- MLLM model/prompt/processor hashes.
- Compiler constants and solver thresholds.
- Final comparator and seeds.
- Test inference path.

### What is trainable

Only the existing encoder/projection parameters during the uniform target-fit continuation. No new relation metric, adapter, router, head, or teacher is trained.

## Training Plan

1. Paired REMOVE/comparator state:
   - For each dataset/fold/seed, produce or load the strongest same-protocol non-MLLM train bank `Z0`.
   - Hash `Z0`, labels, IDs, and code.

2. Train-only MLLM cache:
   - Build evidence packs only for train videos.
   - Run deterministic restricted certificate calls.
   - Seal cache with Merkle root before labels enter compiler.

3. Global target:
   - Compile `A_reg`, `A_eq`, `A_band`, robust intervals, trust/box/vote constraints.
   - Solve one global convex projection `G0 -> G*`.
   - Verify by independent replay.

4. Uniform fit:
   - Factor/Procrustes `G* -> Z*`.
   - Fit encoder using uniform `L_fit`.
   - Run no-collapse, target-residual, train-kNN sanity, rollback replay.

5. Controls:
   - `REMOVE`: same schedule without certificate target.
   - `SHUFFLE`: permute sealed certificate identities/atom rows among train IDs while preserving coverage, missingness, atom marginals, and parent-label counts after cache closure; compile the same way.
   - `NOISE`: corrupt certificate atoms/confidences at preregistered rates before consensus while preserving schema and closure.
   - `DIRECT-MOMENT`: directly optimize `||A_reg vec(G_theta)-b_reg||^2` with coefficient fixed before outcomes, no proximal target.
   - `DIRECT-CERT-FEATURE`: train-only cross-fit linear/ridge certificate-feature baseline that maps certificate atoms to a privileged scalar/low-dimensional target, then distills only through the same uniform encoder interface. It cannot load certificates at validation/test. If it matches FULL, the global geometry interface is not supported.
   - `SCALAR-PROPENSITY`: collapse certificate records to scalar missingness/uncertainty/difficulty summaries and use a matched direct control. If it matches FULL, the MLLM signal is replaceable by scalar difficulty/error propensity.

The direct/scalar controls are explicitly forbidden as the main method and exist only for attribution.

## Inference

For validation/test:

1. Encode the full video with the trained encoder.
2. Build the train memory from full train-video embeddings and train parent labels.
3. Retrieve ordinary top20 nearest train videos.
4. Apply the same arithmetic-rank/similarity-signed vote used by the comparator protocol.

No MLLM cache, certificate file, compiler artifact, target bank, teacher score, auxiliary head, schema feature, key selector, reranker, or test-time prompt is loaded.

## Why the MLLM Signal Is Not Scalar Difficulty

A scalar difficulty/error-propensity signal is a vector `d in R^N`. Any scalar-only geometry influence can be expressed as row-wise or diagonal-rank modulation, such as terms depending on `d_i`, `d_j`, or `d_i d_j`. That class can change how hard examples are treated, but it cannot reproduce the multi-atom structural moment system:

`Q^T G Q`, replica-equality constraints, structural bands, and `K_C`-targeted low-rank certificate geometry.

Those operators encode multiple independent cross-video structural axes and their aggregate interactions over the full bank. They are not per-sample weights, not confidence scores, and not selected pairs. The proposal will still run scalar propensity controls. If they match FULL, the claim fails.

The direct certificate-feature baseline is also essential. If simply using the certificate vector as privileged features or distillation targets performs as well as the global proximal target, then the contribution is not the geometry interface. The final claim requires FULL to beat the strongest direct control.

## Failure Modes and Diagnostics

- Cache contamination:
  - Detect: labels, split, margins, IDs, neighbors, predictions, loss, gradients, segments, timestamps, or rationales appear in prompt/input/output logs.
  - Action: reject cache and STOP.

- Fragment-gold drift:
  - Detect: any segment/timestamp/span/localization/stance/target/mechanism/rationale field treated as supervision, pseudo-group, selection key, or evaluation gold.
  - Action: reject proposal/run; no repair by renaming fields.

- Structural null:
  - Detect: rank `Q=0`, operator norms too small, `G*=G0` within null threshold.
  - Action: report `GLOBAL_TARGET_CERTIFIED_NULL`; no MLLM geometry claim.

- Degeneration into weighting/selection:
  - Detect: certificate controls sample frequency, loss coefficient, key membership, pair/triplet construction, or reranking.
  - Action: reject as forbidden degeneration.

- Solver non-certification:
  - Detect: residuals exceed thresholds, replay mismatch, max cycles exceeded, KKT/VI fail.
  - Action: fail closed; no SLSQP near-miss or local v8 rescue.

- Ambiguous rank overclaim:
  - Detect: ambiguous edges appear in `E_rob` or exact-vote safety claims.
  - Action: reject G0.

- Uniform fit collapse:
  - Detect: embedding variance/effective rank collapse, duplicate rows, train robust vote means degrade, target residual not reduced.
  - Action: rollback and replay REMOVE.

- Attribution failure:
  - Detect: SHUFFLE/NOISE/direct/scalar controls match FULL.
  - Action: MLLM/global-proximal claim unsupported even if raw metric improves.

- Final metric failure:
  - Detect: any dataset/metric < +0.030, any paired seed delta nonpositive, bootstrap lower <=0, or Holm failure.
  - Action: no final success claim.

## Novelty and Elegance

### Relative to pseudo-groups and reweighting

The certificates do not partition data into groups for robust optimization and do not assign per-sample weights. They define low-rank full-bank linear operators over `vec(G)`. Every train video remains in the same target fit with the same coefficient.

### Relative to semantic pair supervision

The method never asks the MLLM to choose or label pairs. There are no certificate-selected positives, negatives, triplets, or SupCon batches. Any pairwise-looking quantity is an aggregate full-bank operator, not a sampled training relation.

### Relative to privileged distillation

The MLLM does not output verdicts, logits, rationales, target labels, mechanism labels, or teacher probabilities. Direct certificate-feature and direct-moment distillation are controls, not the method. The method's distinctive interface is the proximal full-bank Gram target.

### Relative to metric learning

Metric learning typically changes distances through selected pairs, triplets, class positives, hard negatives, or relation-conditioned margins. This proposal changes a PSD correlation matrix by a single global projection with trust, structural, and vote constraints, then fits that global target uniformly.

### Relative to local rank-cell LB-SCGP

The retired route tried to certify local top20 cells and stationarity under strict signed gaps. The global route removes local cell enumeration entirely. Top20 intervals are used only to identify robust edges for optional safety constraints/claims; ambiguous edges fail open for geometry and fail closed for claims.

## Claim-Driven Validation Sketch

### Block 1: Conceptual G0, real-fold gate, and teacher-cache gate

Claim tested: the global pivot is executable, replayable, isolated, nondegenerate, and not a forbidden degeneration before any performance claim.

Minimal experiment:

- Synthetic fixtures for `FULL_GLOBAL_TARGET_SYNTH`, `AMBIGUOUS_EDGE_FIXTURE`, `ROBUST_EDGE_FIXTURE`, and forbidden-degeneration injections.
- One sealed real train fold per dataset using train-only certificates or a cache stub if teacher calls are separately authorized later.
- Independent replay of compiler, projection, factorization, Procrustes, uniform dry fit, rollback, and isolation counters.

Required statuses:

```text
FULL_GLOBAL_TARGET_SYNTH        -> GLOBAL_TARGET_CERTIFIED
FULL_GLOBAL_TARGET_REAL_TRAIN   -> GLOBAL_TARGET_CERTIFIED or GLOBAL_TARGET_CERTIFIED_NULL
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

Decisive metrics:

- replay residuals;
- no forbidden access counters;
- no ambiguous-edge overclaim;
- non-null structural target when FULL is intended;
- dry-fit no-collapse and rollback hash parity.

Gate:

- Any mismatch, missing hash, segment-gold assumption, teacher-cache leak, or forbidden degeneration is STOP.
- G0 emits no accuracy or macro-F1 claim.

### Block 2: Mechanism attribution and direct/scalar controls

Claim tested: any observed gain comes from certificate identity and the global proximal geometry, not ordinary continuation, random regularization, scalar difficulty, or direct certificate-feature use.

Minimal experiment:

- Real-fold train-only development on MHC-EN and MHC-ZH, preferably OOF or seed-0 dev before final.
- Arms: FULL, REMOVE, SHUFFLE, NOISE, DIRECT-MOMENT, DIRECT-CERT-FEATURE, SCALAR-PROPENSITY.
- Same train inputs, same optimizer schedule, same target-fit schedule, same final ordinary kNN endpoint. Certificates absent at validation inference.

Decisive metrics:

- dev/OOF accuracy and macro-F1;
- target residual and realized Gram displacement;
- robust wrong-neighbor rate and robust vote margin diagnostics where interval-certified;
- monotonic degradation under NOISE;
- FULL positive over SHUFFLE and strongest direct/scalar control.

Gate:

- FULL must beat REMOVE, SHUFFLE, NOISE at zero-corruption, DIRECT-MOMENT, DIRECT-CERT-FEATURE, and SCALAR-PROPENSITY by a preregistered dev margin in both metrics on both datasets before final.
- If direct/scalar controls match FULL, the MLLM-global geometry claim fails.

### Block 3: Final performance gate

Claim tested: the immutable final success claim.

Minimal experiment:

- Freeze all code, cache protocol, compiler constants, solver thresholds, controls, seeds, and comparator choice before test.
- Datasets: MHC-EN and MHC-ZH.
- Seeds: 0/1/2.
- Comparator: strongest same-protocol non-MLLM comparator, including any moving REMOVE/LABEL-ONLY/control that is stronger under the frozen protocol.
- Endpoint: ordinary full-video train-memory top20 kNN.

Decisive metrics:

- accuracy and macro-F1;
- paired seed deltas for each dataset×metric;
- hierarchical paired bootstrap lower bound;
- Holm correction over four dataset×metric tests;
- FULL vs REMOVE/SHUFFLE/NOISE/direct attribution.

Gate:

- Accuracy and macro-F1 each at least +0.030 on MHC-EN and MHC-ZH.
- All paired seed deltas positive.
- Hierarchical paired bootstrap lower bound >0.
- Holm correction passes.
- No test teacher/head/rerank/certificate artifact loaded.

## Experiment Handoff Inputs

### Artifacts to freeze before implementation

- Certificate schema `scgp_global_cert_v1`.
- Prompt templates and evidence-order variants.
- MLLM model/processor/version hashes.
- Input builder hash and truncation rules.
- Train ID allowlist and cache Merkle root format.
- `orth_cap` basis rules and rank cap.
- Operator definitions for `A_reg`, `A_eq`, `A_band`, robust intervals, vote constraints.
- Solver constants: `delta`, `rho_row`, `rho_class`, `tau`, `eta_edge`, residual thresholds, max cycles, set order.
- Factorization/Procrustes deterministic tie rules.
- Uniform fit schedule, rollback manifest, no-collapse checks.
- Control generation: REMOVE, SHUFFLE, NOISE, DIRECT-MOMENT, DIRECT-CERT-FEATURE, SCALAR-PROPENSITY.
- Statistical test code and Holm family.

### Required machine records

- `G0` bank and hash.
- Certificate per-call JSONL and consensus JSONL.
- Cache closure manifest.
- Compiler operator manifest.
- Solver trace and output.
- Independent replay report.
- Factor/Procrustes report.
- Uniform fit and rollback report.
- Inference isolation report.

### Highest-risk assumptions

- Restricted certificates have enough stable structural rank on both MHC-EN and MHC-ZH.
- Global structural operators move the train Gram in a useful direction without local pair selection.
- Uniform fit can realize `G*` without collapse or head/memory redistribution.
- Robust interval coverage is sufficient for meaningful safety diagnostics, even though ambiguous edges make no safety claim.
- The final gain exceeds direct/scalar controls and the moving strongest comparator.

## Compute & Timeline

Phase 0/1 proposal work in this file performs no experiments.

Expected future compute if authorized:

- Conceptual G0 synthetic: CPU-only SLURM, minutes to low hours.
- Real-fold global target replay without MLLM calls: CPU-heavy projection plus one GPU dry-fit block, likely low GPU-hours.
- Train-only MLLM cache: depends on teacher choice and throughput; bounded by train videos only, with no validation/test calls.
- Mechanism attribution: several paired dev/OOF arms, likely tens of GPU-hours if full encoder continuation is required.
- Final: MHC-EN/MHC-ZH × seeds 0/1/2 × FULL/controls/comparator under SLURM.

No `--time` should be set. All jobs must run under `conda activate HateVideo`.

## Grounding Sources

No web search was used because local grounding was sufficient. Sources inspected:

- `research-wiki/EXP_p9_lmm_rgcl_video.md`: RA-HMD video adaptation findings, direct LMM/LoRA and kNN memory negative evidence.
- `research-wiki/ideas/rgcl-mllm-video-iter1.md`: archived multi-granularity/MLLM-video retrieval lessons and anti-repeat notes.
- `research-wiki/papers/mei2023_improving_hateful_meme.md`: RGCL base method summary, retrieval-guided contrastive learning and kNN inference.
- `research-wiki/papers/mei2025_robust_adaptation_large.md`: RA-HMD/LMM-RGCL base-method summary.
- `research-wiki/papers/wang2024_multihateclip_multilingual_benchmark.md`: MHC-EN/MHC-ZH benchmark context.
- `research-wiki/papers/wang2025_hateclipseg_segmentlevel_annotated.md`: segment-rich dataset context, used only to reinforce that segment gold is not available for this project.
- `research-wiki/MLLM_USAGE_LANDSCAPE.md`: local novelty map for MLLM/video-hate roles and RA-HMD usage boundaries.
- `research-wiki/HEADTOHEAD_FEASIBILITY.md`: comparator/split/clean-subset feasibility and MoRE/CRAVE context.
- `research-wiki/experiments/exp-ssr-b01.md`: SSR-MemRGCL B0/B1 negative evidence and coverage-before-semantics postmortem.
- `research-wiki/TARGET_GATE0_LITERATURE.md`: local target-driven literature map and prior failed routes.
- `refine-logs/lb_scgp/FINAL_PROPOSAL.md`: previous local rank-cell LB-SCGP proposal and inherited design discipline.
- `refine-logs/lb_scgp/G0_V6_ACTUAL_RESULT_TO_CLAIM_REVIEW.md`: v6 partial support and rank-cell ambiguity.
- `refine-logs/lb_scgp/G0_V6_NUMERICAL_CERTIFICATE_REVIEW.md`: v6 non-witness interpretation and no-infeasibility boundary.
- `refine-logs/lb_scgp/v7/G0_V7_PREREGISTERED_DESIGN.json`: v7 final local repair contract.
- `refine-logs/lb_scgp/G0_V7_RESULT_TO_CLAIM_PIVOT_REVIEW.md`: decisive pivot authority, reusable evidence, retired evidence, and global pivot skeleton.

## Existing Evidence vs Planned Validation

Existing inherited evidence supports strict supervision isolation, no segment-gold discipline, exact kNN endpoint definitions, PSD/unit-diagonal/projector/factor/replay/hash machinery, and the retirement of local rank-cell stationarity. It also documents negative evidence for direct RA-HMD-style video adaptation and sparse relation/selector routes.

The global pivot in this proposal has not yet been experimentally validated. No current repository evidence proves that the proposed restricted certificate schema, global operators, convex target, uniform fit, or final performance claim succeeds.

## Final Boundary

The method is allowed to proceed only as:

`train-only label-blind structural certificates -> single global full-bank PSD/unit-diag proximal Gram target -> uniform encoder fit -> ordinary test kNN`.

There is no local cell enumeration, no stationarity route, no v8, no segment gold, no test teacher/head/rerank, no key selection, no sample weighting, and no pair/triplet/SupCon main method.
