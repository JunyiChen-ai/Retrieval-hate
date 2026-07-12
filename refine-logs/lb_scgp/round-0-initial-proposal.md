# Research Proposal: LB-SCGP — Label-Blind Semantic-Certificate Gram Projection

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causal and removable component of hateful-video RGCL, and do not stop until one frozen method improves the final **ordinary full-video train-memory kNN** accuracy and macro-F1 by at least `+0.030` absolute each over the moving strongest same-protocol non-MLLM comparator on both MHC-EN and MHC-ZH, under paired seeds 0/1/2.
- **Must-solve bottleneck:** Prior MLLM routes supplied verdicts, scores, summaries, auxiliary fields, segment salience, sparse neighbour events, frozen swaps or pseudo-groups. Their semantics were redundant with video labels, orthogonal to vote correctness, absorbed by the fusion head, too sparse, or reducible to sample reweighting/generic gradient surgery. The successor must let label-blind whole-video semantics change the complete final retrieval geometry, including entries outside every old top-64 list, without becoming a teacher key selector, direct rule loss, sample/group weighting or generic pair/triplet metric learning.
- **Non-goals:** No segment, timestamp, span or localization endpoint; no segment weighting/loss; no MLLM verdict, score, free rationale, selected neighbour/key, memory replacement, feature/schema/summary concatenation, score fusion, reranking, veto, router/MoE, native-head claim, direct rule-energy loss, pseudo-group robust optimization, sample reweighting or generic pair/triplet/SupCon method; no test-time MLLM/certificate/target; no scale, data, epoch or ensemble rescue.
- **Constraints:** The **only gold supervision is the parent video's binary label**. No segment-level gold exists or may be assumed. Proposition, stance, quotation, condemnation, reportage and cross-modal-binding states are confidence-bearing train-only weak/privileged MLLM pseudo-signals, never annotations or gold. The teacher is strictly label-blind and receives no label, prediction, margin, error/correctness, neighbour/key/ID, loss or gradient. Teacher records close immutably before a deterministic compiler may read train video labels. Validation/test never load a certificate, target bank or compiler artifact. All future computation is SLURM-only in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no source code and launches no job.
- **Success condition:** Both datasets, seeds 0/1/2, ordinary full-video kNN accuracy and macro-F1 each exceed `max(historical strongest point, paired same-seed REMOVE mean)+0.030`; all paired deltas are positive; hierarchical paired-bootstrap lower bounds exceed zero; four dataset×metric tests survive Holm FWER `0.05`; FULL significantly beats REMOVE and a permutation of certificate identity in both metrics; corruption degrades the gain; final displacement has a verified nonnegative-cone/Farkas separation from ordinary RGCL sample and generic relation gradients. Proposal readiness, solver success, target objective, train leave-one-out accuracy, an auxiliary head or seed-0 evidence cannot close the goal.

This anchor is immutable in every round. “Certificate” never means a gold explanation: it is a constrained, fallible, label-blind teacher record.

## Technical Gap

The exact endpoint is a self-excluded cosine train-memory classifier with `K=20`, arithmetic rank weights `w_r=21-r`, and similarity-signed voting. Let `c_i=2y_i-1`. For a normalized Gram bank `G` and the stable self-excluded ranking `pi_i(G)`, the true-class margin is

`m_i(G)=c_i/210 * sum_{r=1}^{20} w_r c_{pi_i(r)} G[i,pi_i(r)]`.

P2 showed that asking an MLLM whether an individual neighbour is semantically comparable does not predict vote correctness. SSR's entire single-neighbour event universe and EDCM's frozen top-64/two-swap universe lacked dual-metric headroom. P4 showed that predicting semantic fields is label-redundant. ECM's pseudo-mode QP reduced algebraically to sample-weighted common-descent gradient surgery. CTE and SQ ended at numerical/governance gates and supply no contrary performance result.

The missing interface is therefore neither another semantic feature nor another edge loss. It is a **full-bank target operator**: compile structured label-blind clause composition and exception algebra into absolute row-profile identities on a PSD unit-diagonal Gram matrix, solve the nearest target that preserves exact vote margins under changing ranks, and fit the existing shared encoder uniformly to that stopped target. This can move every bank coordinate while keeping the final classifier unchanged.

## Method Thesis

- **One-sentence thesis:** A constrained label-blind MLLM certificate can be useful beyond video labels only when a deterministic post-cache compiler converts clause/exception composition into non-reweightable full-bank Gram row-profile constraints whose nearest exact-vote-safe target is internalized by the same encoder used by ordinary kNN.
- **Smallest adequate intervention:** one immutable teacher cache, one deterministic compiler/solver, and one uniform target-fitting loss; zero new trainable modules and zero inference additions.
- **Foundation-model role:** the MLLM is a train-only semantic certifier of compositional whole-video evidence, not a classifier, key selector, feature generator, group annotator or test-time reasoner.

## Contribution Focus

- **Dominant contribution:** the complete operator `label-blind constrained certificate -> post-cache label compiler -> pair-of-pairs Gram row profiles -> rank-cell exact-vote proximal target -> shared-encoder target fitting -> unchanged ordinary kNN`.
- **Supporting contribution:** an executable non-reweighting audit with primal cone projection and a Farkas dual separating witness. It is mechanism evidence, not a second model.
- **Explicit non-contributions:** certificates, Gram/Procrustes alignment, KD, semantic supervision and exact kNN are not claimed individually as new. Novelty rests only on their retrieval-target compiler interface and verified non-equivalence to scalar weighting/generic relation learning.

## Proposed Method

### Complexity Budget

- **Frozen/reused:** existing CLIP full-video caches, RGCL shared projection/fusion encoder, video labels, optimizer/scheduler, epochs/steps, checkpoint rule, train bank refresh, `K=20` arithmetic similarity vote, splits and seeds.
- **New trainable components:** zero. Existing encoder parameters alone are updated.
- **New train-only artifacts:** a hashed certificate cache, compiled sparse linear operators, stopped `G*`/`Z*`, and solver/audit logs.
- **Intentionally excluded:** all segment objects, teacher embeddings/keys, auxiliary heads, rules in the loss, per-record coefficients, pseudo-groups, routers and test-time artifacts.

### System Overview

```text
train whole video: uniform frames + full ASR/OCR/title
       -> frozen label-blind MLLM, 2 prompts x 2 presentation orders
       -> closed constrained certificate (no verdict/score/rationale/key)
       -> immutable cache + prompt/model/input/code/provenance hashes

cache CLOSED -- only now compiler may read parent-video binary train labels
       -> clause incidence + exception/reflection operators
       -> all eligible pair-of-pairs row-profile identities

eval-mode normalized full train bank Z0 -> G0=Z0 Z0^T
       -> rank-cell sequential proximal PSD correlation target G*
       -> eigendecomposition + orthogonal Procrustes -> Z*
       -> uniform target fit of the existing shared encoder

validation/test full video -> existing encoder -> ordinary train-memory kNN
                                         (no certificate/compiler/target)
```

### Exact Teacher Certificate

The teacher input contains one whole training video only: frozen uniform timestamp-ordered frames and the complete available ASR/OCR/title under fixed truncation. Automatic frames, ASR and OCR are input channels, not segment annotation. The payload and prompt contain none of `{label, prediction, probability, margin, error, correctness, loss, gradient, neighbour, key, memory ID, split, seed}`.

The strict JSON object has `schema_version`, eleven fixed atoms, `record_confidence`, and provenance. Each atom is exactly `{"state": supported|contradicted|unresolved, "confidence": 0|1|2|3|4}`. No additional key or string is accepted.

Fixed atoms:

1. `proposition.group_referent_present`
2. `proposition.derogatory_exclusionary_or_violent_predicate_present`
3. `proposition.target_predicate_bound`
4. `stance.speaker_asserts_or_endorses_proposition`
5. `stance.speaker_distances_from_proposition`
6. `exception.quotation_applies_to_proposition`
7. `exception.condemnation_applies_to_proposition`
8. `exception.reportage_applies_to_proposition`
9. `cross_modal.visual_speech_proposition_bound`
10. `cross_modal.visual_text_proposition_bound`
11. `cross_modal.speaker_source_stance_bound`

The atoms describe whether a specific compositional reading is supported, contradicted or unresolved. They do not ask whether the video is hateful, benign, policy-violating or correctly classified; no severity, score, target name, free-form proposition, rationale, span, timestamp or segment identifier exists.

Four independent deterministic invocations use two exact prompt paraphrases and two evidence orders. Canonical state is the modal state, ties become `unresolved`; canonical confidence is the minimum confidence among calls supporting the modal state divided by four. A record is accepted only when all four calls parse, every non-unresolved atom has modal agreement at least `3/4`, record confidence is at least `0.5`, and the fixed graph closes:

- supported `target_predicate_bound` requires supported group referent and predicate;
- supported endorsement and supported distancing cannot coexist;
- a supported exception requires a supported proposition and its fixed scope link;
- supported condemnation requires supported distancing and contradicted endorsement;
- quotation or reportage alone never implies endorsement or distancing; an unsupported stance remains unresolved;
- a supported cross-modal binding whose required endpoints are contradicted is invalid.

Parse failure, extra keys, closure failure, low confidence or disagreement rejects the **entire** record. Rejection is an exact REMOVE fallback, never repaired from labels or predictions. Infrastructure retry may repeat only the byte-identical payload. Every call stores prompt/model revision/generator/input/code hashes before any output is used; the cache closes through an ID allowlist, Merkle root and read-only manifest. After closure no teacher record can be appended, repaired or regenerated for a discovered failure.

### Label-Proxy and Contamination Firewall

The teacher is label-blind, but the video itself contains label-relevant evidence; therefore “label-blind” is not claimed to imply “label-independent.” Four safeguards are binding:

1. The schema has no verdict/severity/free-text channel and no scalar obtained by collapsing the atoms.
2. A forbidden-key/input audit and blind payload replay must show zero label/prediction/error/margin/neighbour/loss access.
3. After cache closure only, strict inner-cross-fit probes report how well the whole certificate, each atom, and its first principal component predict the video label, baseline OOF error and true-class margin. These are contamination diagnostics, never teacher-selection criteria.
4. FULL must beat capacity/strength-matched `CERT-LABEL-PROPENSITY` and `CERT-ERROR-PROPENSITY` controls that reduce the certificate to one cross-fitted scalar. It must also beat P4-style atom prediction. If scalar label/error propensity or field prediction matches FULL, the certificate is a proxy/redundant field and LB-SCGP stops.

### Deterministic Post-Cache Compiler

Map `supported/contradicted/unresolved` to `+1/-1/0` and multiply by canonical confidence. This produces `a_i in [-1,1]^11`; rejected records have no row in the semantic operator and are not deleted or weighted. Only after the certificate cache Merkle root is frozen does the compiler read `c_i=2y_i-1`.

The compiler is a pure, versioned function of `(certificate cache, train IDs, parent-video labels)`. It makes no model/teacher call and reads no prediction, margin, error, loss or validation/test item. It deterministically creates:

- `core_i`: proposition and cross-modal-binding atoms;
- `direct_i`: endorsement supported while distancing and all exceptions are contradicted;
- `exc_i^q`, `exc_i^c`, `exc_i^r`: quotation, condemnation or reportage supported with closed scope; ambiguous/multiple unresolved cases emit no semantic constraint;
- reliability weights used **inside row-profile averages only**, never as loss/sample weights.

For each certificate coordinate and each post-cache label sign, create fixed bank columns `W[:,d]` from confidence-signed atom conjunctions, normalized to unit `L1` mass. With the diagonal masked,

`R_i,d(G) = sum_{j != i} W[j,d] G[i,j]`.

`R_i(G)` is an absolute similarity row profile over every reliable bank record, not a selected-neighbour score. All eligible anchor pairs are generated exhaustively by canonical ID order:

- **equivalence:** same video label, same supported proposition/binding core, and identical direct/exception state gives `R_i(G)-R_k(G)=0` over registered columns;
- **exception reflection:** same supported proposition/binding core and video label, but one direct and one closed quotation/condemnation/reportage case gives `R_i(G)-T_e R_k(G)=0`, where fixed signed permutation `T_e` swaps the direct and named-exception profile columns and leaves unrelated columns unchanged;
- unsupported/unresolved clauses emit no equation. No video is removed, resampled or reweighted.

Each scalar equation is a pair-of-pairs identity: differences of averages of `G[i,j]` pairs must equal a second anchor's corresponding pair averages (or their fixed exception reflection). Stacking them yields sparse `A_sem vec(G)=0`. The compiler emits every equation, its exact source atoms and hashes; there is no teacher-chosen pair/key, learned compiler, free threshold, outcome-selected relation or old-top-k restriction.

To avoid a vacuous dense operator, pilot gates require support in both labels for direct and each retained exception family, nonzero held-out row-profile variance, and closure under `T_e`; a family failing support is removed by a frozen support rule before any endpoint is inspected. At least one exception family must survive on both datasets or the semantic mechanism stops.

### Rank-Cell Sequential Proximal Gram Target

At each registered bank refresh, build `Z0 in R^(N x 1024)` in `model.eval()`, normalize every row, order rows by canonical video ID, and set `G0=Z0 Z0^T`. `N=549/579 < 1024`, so any numerical PSD target has an exact rank-`<=N` factor in the existing embedding dimension; no rank surrogate is needed.

Solve the nearest correlation target

`min_G 0.5||G-G0||_F^2`

subject to:

1. `G=G^T`, `G >= 0` (PSD), `diag(G)=1`;
2. registered semantic residual contraction `||A_sem vec(G)||_2 <= kappa ||A_sem vec(G0)||_2` for FULL (`A_sem` absent in LABEL-ONLY);
3. every self-excluded exact top-20 true-class margin `m_i(G) >= ell_i`;
4. each class mean and global mean exact margin is no lower than its registered `G0` value;
5. `||G[i,-i]-G0[i,-i]||_2 <= tau_row sqrt(N-1)` and each class mean row has the analogous radius `tau_class sqrt(N)`;
6. box feasibility `-1 <= G_ij <= 1`, implied analytically by PSD/unit diagonal and checked numerically.

For the zero-teacher LABEL-ONLY compiler, `A_sem` is empty and

`ell_i=max(m_i(G0), epsilon_vote)`,

with one route-wide positive numerical `epsilon_vote` frozen before endpoint evaluation. Thus the program asks for the minimum full-bank displacement that preserves every already-correct leave-one-out vote and moves every incorrect training-memory vote strictly across the repository decision boundary. It does not use any outer-fold query label.

The top-20 ranking makes the feasible set a union of finite rank cells. The solver is an active-cell sequential proximal method:

1. remove diagonal/self before ranking; sort off-diagonal similarities descending and break exact/numerical ties by canonical video ID ascending;
2. inside the current cell, `pi_i` is fixed, hence every `m_i` constraint is linear; solve the PSD/unit-diagonal trust-region SDP to its registered primal/dual tolerances;
3. if a step reaches a rank boundary, inspect adjacent cells in lexicographic adjacent-swap order; admit the first epsilon probe that decreases target distance/semantic residual while satisfying all **recomputed** exact margin and trust constraints;
4. pivot, recompute all `N` rankings from the complete bank, and repeat; no old top-64 candidate set exists;
5. accept a proximal iterate only after the independent repository-parity evaluator recomputes ID/rank/cosine/prediction and verifies every individual/class/global envelope; otherwise halve the step, and fail closed after the registered backtracking budget.

The PSD iterate is never produced by clipping an indefinite matrix. Convex interpolation between feasible correlation matrices preserves PSD and unit diagonal during backtracking. Termination requires primal/dual/PSD residuals below frozen tolerances, no unprocessed rank boundary, stable exact rankings, and two successive merit changes below tolerance. Infeasibility or solver nonconvergence maps the complete refresh to REMOVE; fallback rate is reported. Solver constants are frozen from label-only numerical parity/resource checks, shared across EN/ZH, and never selected by dev/test accuracy.

### Factor, Procrustes and Uniform Encoder Fit

Numerically eigendecompose `G*=U Lambda U^T`, reject eigenvalues below the PSD tolerance only if within solver residual, form `Zraw=U Lambda^(1/2)` and zero-pad to 1024. Compute

`Q*=argmin_{Q^TQ=I} ||Zraw Q-Z0||_F`

by SVD and set `Z*=Zraw Q*`. Verify `||Z*Z*^T-G*||_F/||G*||_F` and row-norm errors against frozen tolerances. `Z*` is stopped gradient and is never a memory key at inference.

For a fixed number of registered fit blocks, use uniform full-video minibatches and

`L_fit = (1/N) sum_i ||normalize(f_theta(x_i))-stopgrad(Z_i*)||_2^2`.

Every video has coefficient `1/N`, including rejected/unresolved records; confidence never weights an encoder loss. The remaining fixed steps run ordinary RGCL. REMOVE spends the identical total steps, epochs, optimizer/scheduler and refresh budget on ordinary RGCL; every learned control receives the same target-fit fraction and tuning budget. No extra parameter or epoch is introduced.

After each block, rebuild the actual eval-mode bank and report target-realized displacement cosine, normalized target error, exact rank churn, individual/class/global margins, semantic residual, effective rank, per-class within-bank variance, class-centroid separation and maximum off-diagonal similarity. A block is rolled back together with optimizer state and replaced by deterministic REMOVE steps if target alignment is nonpositive, any margin guard falls, effective rank or within-class variance drops below `80%` of its pre-block value, or numerical duplicates appear. SCGP-0 must additionally meet the frozen target-realization thresholds; a good abstract `G*` with a failed encoder fit is a route failure.

### Non-Reweighting and Non-Triplet Farkas Audit

At each accepted target, project `D*=Z*-Z0` onto the product of row-sphere tangent spaces and vectorize/normalize it as `d`. Freeze two matrix-free gradient dictionaries at the same `Z0`:

- `H_ex`: one column for every ordinary RGCL per-example embedding-gradient contribution under the registered detached-bank computation;
- `H_rel`: columns for every update primitive available to the capacity-matched generic pair, triplet and SupCon controls over the same full bank and labels.

For each dictionary solve `min_{alpha>=0} ||H alpha-d||_2/||d||_2`. Both relative residuals must be at least `0.25`. From the NNLS projection residual construct and independently optimize a unit dual witness `u` satisfying

`H^T u >= -tol_dual` and `d^T u/||d|| <= -gamma`.

Require `gamma>=0.25`, primal/dual separation agreement within the registered tolerance, and a duality gap above numerical noise. This is the Farkas certificate that `d` is outside the nonnegative scalar cone, not merely a large NNLS residual report. The audit is run in SCGP-0 and teacher FULL on both datasets/folds. Learned scalar-example, pair/triplet and SupCon controls must also fail to match FULL OOF metrics/constraints; numerical separation without behavioral separation is insufficient. Failure means the method has reduced to sample/relation weighting or generic metric learning and stops.

### Inference

Validation and test perform only:

`whole video -> existing shared encoder -> normalized full-video train bank -> existing FAISS cosine top-20 arithmetic similarity vote`.

They do not load the teacher, certificate, certificate confidence, compiler, `A_sem`, `G*`, `Z*`, target-fit loss, train prediction or any segment artifact. The train bank is rebuilt from encoder outputs, never copied from `Z*`.

## Failure Modes and Diagnostics

- **Certificate becomes a label/error proxy:** FULL parity with scalar certificate-label/error propensity or P4 field prediction -> STOP.
- **Certificate pollution/leakage:** any forbidden payload/output key, nonclosed append, label/prediction/error/margin/neighbour/loss access or validation/test ID -> invalidate cache and STOP.
- **Sparse/collapsed exception algebra:** missing cross-label family support, closure failure, max-state collapse or held-out row-profile identity no better than shuffle -> STOP before full extraction.
- **Solver gaming or fixed universe:** evaluator parity, self exclusion, stable tie, PSD/unit-diagonal, rank pivot or exact envelope failure -> STOP; no tolerance relaxation.
- **Unfittable/collapsed target:** target-realized failure, effective-rank/variance collapse, native-head-only gain or rebuilt ordinary-kNN failure -> STOP.
- **Generic-loss equivalence:** either Farkas/cone audit below threshold or learned generic control parity -> STOP.
- **Missing/low confidence:** exact REMOVE record fallback; coverage/confidence/fallback reported by class and dataset.
- **No segment gold:** all whole-video atoms remain weak pseudo-signals; no segment/timestamp/span object exists in schema, compiler, loss or endpoint.

## Novelty and Elegance Argument

LEAF uses gold-grounded explanations and a generative student; TextTeacher matches semantic anchors; EmbedDistill/geometry KD matches teacher geometry; DARTVAE puts rules directly in a latent loss; formal sidecars introduce proof obligations; ECM used discrepancy modes and worst-mode optimization. LB-SCGP does none of these. Its defensible delta is the deterministic conversion of a label-blind closed compositional certificate into **absolute pair-of-pairs full-bank Gram identities**, jointly reconciled with changing exact rank-weighted vote cells and then uniformly internalized into the unchanged retrieval encoder. The Farkas audit makes the exclusion of scalar weighting/generic triplets falsifiable.

The method stays one contribution because schema, solver and fit are stages of one target operator, not parallel predictive modules. If the target operator cannot pass zero-teacher capacity/fitting and non-equivalence gates, no teacher scale or prompt rescue is allowed.

## Claim-Driven Validation Sketch

### Claim 1: the full-bank target operator is executable, fit-able and not a generic weighted metric loss

- **LB-SCGP-0:** zero teacher/new OCR calls; strict nested five-fold train OOF on MHC and MHC-ZH. For each outer fold, all labels used by target/compiler/fitting belong only to `T\F`; outer queries are encoded without their labels and classified against the rebuilt `T\F` ordinary kNN bank. Outer labels are endpoint-only.
- **Gate:** actual pooled OOF accuracy and macro-F1 each improve by at least `+0.050` over the frozen geometry on both datasets, every fold delta is positive, target realization passes, and both Farkas residual/separation audits pass. LABEL-ONLY immediately becomes a moving non-MLLM comparator if stronger. Any failure gives zero teacher calls and terminal STOP for LB-SCGP.
- **Parity:** scalar/vector PSD/factor/Procrustes parity, repository ID/rank/cosine/vote parity, self exclusion, stable ties, full-rank recomputation, deterministic repeats and measured SLURM resource estimate.

### Claim 2: compositional certificate identity adds causal value beyond labels/proxies/rules

- **LB-SCGP-1 pilot:** at most 128 unique train videos per dataset, frozen by label x strict-OOF prediction x margin-quartile stratification before calls; those strata are never included in teacher payload. Four calls/video give at most 512 calls/dataset. Freeze sample IDs/payload hashes first.
- **Governance gate:** 100% provenance, zero forbidden access, parse completeness at least 95%, accepted closed-record coverage at least 80%, state agreement at least 75%, noncollapsed proposition/direct/exception/binding support, and at least one supported exception family in each label on each dataset. Blind whole-video QC only checks schema appropriateness and contamination; it is not supervision.
- **Conditional-value gate:** on held-out pilot identities, certificate pair-of-pairs operators must reduce row-profile/reflection residual and improve target correction direction beyond LABEL-ONLY, CERT-LABEL-PROPENSITY, CERT-ERROR-PROPENSITY, P4-AUX, caption/TextTeacher anchor and within-cell CERT-SHUFFLE. The teacher target must retain both Farkas separations and, after actual strict-OOF fitting, beat every binding control by at least `+0.010` accuracy and macro-F1 on both datasets. Otherwise STOP without full-cache calls.

### Claim 3: FULL causes substantial final ordinary-kNN improvement

- **Seed-0 gate:** after pilot/full train-only cache authorization, MHC and MHC-ZH dev ordinary kNN accuracy/macro-F1 each beat REMOVE, LABEL-ONLY-TARGET, within-cell CERT-SHUFFLE, CERT-NOISE, scalar propensity, P4/TextTeacher/direct-rule and strongest generic pair/triplet control by `+0.010`; corruption `{0,.25,.50,.75,1}` must degrade gain monotonically.
- **Final gate:** paired seeds 0/1/2, both datasets, unchanged protocol, each metric at least moving baseline `+0.030`, every seed positive, hierarchical paired bootstrap lower bound above zero, Holm-corrected four tests, significant FULL-minus-REMOVE and FULL-minus-SHUFFLE. Only this gate meets the project objective.

## Experiment Handoff Inputs

- **Must-prove claims:** exact target capacity/fitting; certificate conditional value beyond scalar proxies; nonnegative-cone/Farkas separation; final ordinary-kNN `+3/+3` on two datasets/three seeds.
- **Binding controls:** REMOVE, LABEL-ONLY-TARGET, CERT-SHUFFLE, CERT-NOISE/MISSING, CERT-LABEL-PROPENSITY, CERT-ERROR-PROPENSITY, P4-AUX, TextTeacher/caption anchor, DARTVAE-style direct rule loss, generic pair/triplet/SupCon, and the strongest resulting non-MLLM arm.
- **Critical metrics:** ordinary kNN accuracy/macro-F1, paired predictions, certificate coverage/agreement/closure, row-profile residual, exact margins/rank churn, target realization/collapse, cone residual/dual separation and resource use.
- **Highest risks:** SCGP-0 may not transfer from train-target geometry to held-out queries; the certificate may collapse to a label proxy; semantic constraints may be too sparse; the rank-cell SDP may be too expensive; the target may be outside encoder capacity; the generic relation cone may span the displacement.

## Compute and Timeline Estimate

- **LB-SCGP-0:** provisional 30–80 GPU-hours total across two datasets/five folds plus CPU/solver time; an implementation audit and one SLURM numerical microbenchmark must replace this estimate before execution.
- **Pilot:** at most 256 unique videos total and 1,024 deterministic MLLM calls; zero gold annotation cost because QC/certificates are not labels.
- **Later:** seed-0 and final three-seed costs are authorized only by preceding gates; all arms match steps and use at most two GPUs.
- **Timeline:** Phase-0/implementation audit, then SCGP-0 only; teacher pilot only after dual-dataset SCGP-0 GO; no source change or job is authorized by this proposal alone.
