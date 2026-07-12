# Research Proposal: SQ-RGCL — Soft Environment Quotient Geometry for Retrieval

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM into hateful-video RGCL as a meaningful, novel, causally removable method component that produces a substantial improvement in the final ordinary full-video train-memory kNN classifier—not merely in localization, audit quality, an auxiliary/native head, or qualitative explanations.
- **Must-solve bottleneck:** Existing RGCL memory geometry confounds the binary hate decision with whole-video presentation/context nuisance. Prior MLLM routes were label-redundant, nearly orthogonal to vote correctness, too sparse in their correctable unit, or competed with the memory readout. The new route must use train-only MLLM information to reshape the exact embedding consumed by final kNN, prove dense conditional value beyond video labels and cheap controls, and avoid merely moving accuracy between heads.
- **Non-goals:** No segment/timestamp/span/localization task; no segment weighting; no stance/harm/evidence/target/mechanism field as a nuisance variable; no final-verdict teacher; no archive/schema prediction as the proposed mechanism; no concat/fusion/reranking/router/GroupDRO/test-time MLLM; no larger-model/data/epoch/ensemble story; and no claim that a cost/numerics preflight is a theoretical performance upper bound.
- **Constraints:** The only gold supervision is the parent video's binary label. No segment-level gold exists or may be assumed. Every MLLM/archive quantity is a weak or privileged train-only pseudo-signal with an explicit confidence, deterministic missing/low-confidence fallback, remove/shuffle/noise controls, and no teacher/environment artifact at validation or test inference. All eventual compute must run through SLURM in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no code and launches no job.
- **Success condition:** On at least two datasets, paired seeds 0/1/2, the final ordinary full-video kNN must improve both accuracy and macro-F1 by at least +0.030 absolute over the moving strongest same-protocol non-MLLM comparator; every paired delta must be positive, hierarchical paired bootstrap lower bounds must exceed zero, the four dataset×metric tests must survive Holm correction, and FULL must beat remove-MLLM plus a within-split permutation of the MLLM information with a significant same-direction removal cost. The method may stop only after these final gates—not after proposal readiness, preflight, or seed-0 evidence.

## Technical Gap

RGCL's final decision is a neighbor vote in a learned full-video memory space. On MHC, presentation format and social context—news/reportage, satire/skit, educational explanation, personal narrative, gaming/music, or other—can dominate visual/text similarity even when they are not the binary hate decision. A hateful news quotation and a non-hateful news report may become neighbors because they share reportage form; two videos with the same label but different presentation formats may remain unnecessarily far apart.

Prior local routes do not solve this geometry problem. P2 asks whether two archives are semantically comparable and then edits membership, but comparability was almost independent of whether a neighbor voted correctly. P4 predicts label-informative archive fields, but those fields were already recoverable from the representation and added no conditional information beyond the video label. SSR and EDCM operate on sparse or frozen candidate universes. CTE was stopped only by its frozen numerical tolerance before any capacity/performance experiment, so it is not evidence against a learnable representation route.

The missing mechanism is not another semantic feature. It is a **class-conditional quotient operation on the exact retrieval geometry**: collapse presentation variation only inside a video-label class while increasing separation between classes inside the same presentation environment. This must use the MLLM signal only as a train-time relation weight; the student must never predict, concatenate, route on, or consume that signal at inference.

## Route Comparison

### Route A — direct soft quotient geometry (selected)

Use a six-way MLLM presentation posterior and confidence to weight two full-bank relations on the current RGCL embedding: same-label/cross-environment positives and different-label/same-environment negatives. Optimize one bank-normalized contrastive ranking term on the same embedding used by final kNN.

### Route B — explicit invariant/nuisance subspaces (rejected)

Learn semantic and nuisance heads with orthogonality, HSIC, adversarial environment prediction, or projection. This adds trainable components, relies on a brittle factor-independence story, overlaps Yang/CDAL/CARE, and makes it difficult to attribute a gain to the MLLM rather than generic regularization.

**Decision:** Route A is smaller, directly tied to the final memory vote, and supports a cleaner necessity test. Route B is an explicit non-contribution and may appear only as prior art, not as an implementation branch.

## Method Thesis

- **One-sentence thesis:** A train-only MLLM can improve final hateful-video kNN by defining a soft presentation quotient of the RGCL memory: contract same-label videos across different presentation environments while repelling different-label videos within the same environment, directly on the shared full-video embedding and with no teacher artifact at inference.
- **Why this is the smallest adequate intervention:** The method adds one parameter-free full-bank loss and no environment head, router, extra representation, memory editor, or test-time path.
- **Why timely:** It uses foundation-model knowledge as privileged relational supervision rather than as another feature, explanation, or classifier, while targeting the exact retrieval endpoint where local MLLM integrations have failed.

## Contribution Focus

- **Dominant contribution:** The crossed-fiber soft quotient loss for exact RGCL memory geometry: MLLM posterior overlap defines both class-preserving nuisance contraction and nuisance-matched class separation.
- **Supporting contribution:** A zero-new-call, fail-closed conditional-enrichment preflight that tests whether the proposed MLLM relation is dense and aligned with correctable OOF retrieval pressure before any new teacher spend.
- **Explicit non-contributions:** No causal identification; no general disentanglement theorem; no new MLLM architecture; no segment supervision; no auxiliary environment classifier; no GroupDRO; no new inference module.

## Proposed Method

### Complexity Budget

- **Frozen/reused:** Existing full-video RGCL encoder/fusion, video-level binary labels, exact train-memory construction, top-20 similarity-signed rank-weighted kNN, current train split, existing v2 train archives, fixed seeds and model-selection protocol.
- **New trainable components:** None beyond the existing RGCL embedding path. There is no posterior predictor or quotient projection head.
- **New non-parametric artifact:** A train-only six-way posterior `q_i`, scalar reliability `r_i`, and provenance/parse flags.
- **Intentionally excluded:** stance, endorsement, harm act, target, mechanism, explicitness, modality evidence, rationales, timestamps, spans, segments, free-text embeddings, teacher keys, concatenation, score fusion, reranking, routers, MoE, GroupDRO, HSIC, adversarial heads, and test-time MLLM.

### Allowed presentation ontology and field governance

The fixed posterior support is:

1. `news_reportage`
2. `satire_skit`
3. `educational_explanatory`
4. `personal_narrative_discussion`
5. `gaming_music_entertainment`
6. `other_unclear`

These are **presentation/context environments**, not hate semantics. The teacher is forbidden to output or encode stance, endorsement, hateful/non-hateful verdict, harm act, target group, attack mechanism, evidence modality, explicitness, label, difficulty, prediction, margin, neighbor identity, split, seed, timestamp, span, or segment. A record contains only the ordered six probabilities, `confidence`, `parse_ok`, `source_hash`, and prompt/model provenance. Any extra key is a hard parse failure.

For the zero-new-call preflight, only `archive.neutral_summary` may be read from existing v2 train archives. The outer archive `label`, `raw_output`, `target_groups`, `mechanism`, `modality_cues`, and `explicitness` are forbidden inputs. A frozen six-prototype scorer maps the summary to a cheap `q_i^arch`; this is both a no-call feasibility artifact and a strong cheap-language control. It is not silently promoted to a teacher success result. If a later MLLM pilot is authorized, the MLLM emits the same restricted posterior schema and never sees a gold label.

### System Overview

```text
TRAIN full video ── existing RGCL encoder ──> normalized embedding z_i ──> epoch memory bank
       │                                             │
       ├─ video-level binary gold y_i ───────────────┤ class condition only
       │                                             │
       └─ train-only MLLM/archive ──> q_i, r_i ──────┤ pair weights only
                                                     │
                      same y + low posterior overlap ─┤ pull
                 different y + high posterior overlap ┤ push
                                                     ▼
                               L_base + lambda_Q L_SQ on the same z

VAL/TEST full video ── same encoder ──> z ──> ordinary train-memory top-20 kNN
                                             (no q, no teacher, no environment head)
```

### Core representation and relations

Let `z_i in S^(d-1)` be the normalized full-video embedding used by the repository kNN, `y_i in {0,1}` the only gold label, `q_i in Delta^5` the weak train-only presentation posterior, and `r_i in [0,1]` its reported reliability.

Use posterior affinity

`A_ij = sum_e sqrt(q_i[e] q_j[e]) in [0,1]`.

Two conjugate relations are defined over the full train bank:

- quotient contraction: `y_i = y_j`, weighted by `w^+_ij = r_i r_j (1 - A_ij)`;
- class transversal: `y_i != y_j`, weighted by `w^-_ij = r_i r_j A_ij`.

Thus the teacher never says whether two videos should share a label. Gold labels provide the class sign; the MLLM only says how strongly a pair crosses or shares presentation context. This avoids the P2 mistake of treating semantic comparability as vote correctness.

For each anchor, normalize positive and negative weights separately so posterior density cannot merely change total loss strength. Require a minimum effective sample size in both relations; otherwise the anchor falls back to the base loss.

### Soft Quotient Bank Loss

With `s_ij = z_i^T z_j / tau`, define separately normalized relation weights `p^+_ij` and `p^-_ik`. The sole new term is

`L_SQ(i) = -log [sum_{j:y_j=y_i} p^+_ij exp(s_ij) / (sum_{j:y_j=y_i} p^+_ij exp(s_ij) + sum_{k:y_k!=y_i} p^-_ik exp(s_ik))]`.

The epoch bank is rebuilt from the same shared encoder. Bank keys are detached within an update for bounded cost, then move at the next refresh; queries and memory therefore co-evolve across epochs. The loss acts on the exact `z` exported to final kNN—there is no projection used only during training and no native-head-only escape.

The training objective is `L = L_RGCL + lambda_Q L_SQ`, with one preregistered `lambda_Q` selected strictly inside train OOF. If `q/r` is absent, malformed, low-confidence, or fails relation-support checks, `L_SQ(i)=0`, exactly reducing that record to the non-MLLM path.

### Why this is a quotient rather than field prediction

The loss never asks `z` to reconstruct `q`. It treats environments as fibers inside each binary class and contracts only cross-environment directions within that class; simultaneously, it separates the two classes inside shared environments. There is no claim that nuisance and class are statistically independent, and no global removal of topic/style directions. This class conditioning is the safeguard against erasing genuine label evidence.

### Modern Primitive Usage

- **Primitive:** Frozen MLLM semantic posterior as learning using privileged information.
- **Exact role:** A train-only relation teacher that estimates whole-video presentation/context uncertainty.
- **Why natural:** Presentation categories require semantic recognition across visual, speech, and text context, but they are not needed at deployment. Using the posterior only to weight training geometry avoids the weak test-time classification and summary-concat paths already disproven locally.

### Training Recipe

1. Freeze split IDs, base checkpoints/protocol, archive provenance, ontology, prompt, model, confidence rule, missing fallback, and all gates before reading outcomes.
2. Run `SQ-P0`, the zero-new-call archive governance, density, and conditional-enrichment preflight, using strict five-fold train OOF ledgers and existing archives only.
3. Only if `SQ-P0` passes on both MHC and MHC-ZH, run a bounded teacher-posterior audit/pilot on train records; no validation/test artifact is generated.
4. Select the single shared `lambda_Q` and confidence floor by nested train OOF minimax across the two datasets; the final seed protocol is frozen before seed-0 validation.
5. Train FULL and all matched controls from identical initialization/data order/epochs. Rebuild the detached full bank at the same schedule in every arm.
6. Validation and test construct only full-video embeddings and use the unchanged top-20 train-memory vote. They never read `q`, archive text, environment, teacher, or a neutralized view.

### SQ-P0: zero-new-call dense conditional-enrichment and coverage preflight

This stage makes no new MLLM call and does not modify `src/`. It uses only existing strict train OOF embeddings/labels plus permitted archive summaries.

**P0.1 governance and density gates**—all must pass independently on both datasets:

- 100% input-key audit: only `neutral_summary` enters `q^arch`; every forbidden archive key access count is zero.
- At least 90% parse/coverage and at least 80% active anchors per video class.
- For each class, effective environment count `exp(H(mean_i q_i)) >= 2.5`.
- At least 80% of anchors in each class have effective sample size `ESS >= 8` for both cross-environment/same-label and same-environment/different-label relations in every outer-train bank.
- Every active environment has nontrivial mass from both binary classes; `other_unclear` cannot exceed 50% of total confident mass.
- Confidence, entropy, missingness, class/environment contingency, and all fallback rates are reported; no category is renamed after inspection.

**P0.2 conditional enrichment gate**—designed specifically against the P2 negative result:

- For each OOF query, compute the signed quotient pressure: similarity mass of same-environment/different-label neighbors minus cross-environment/same-label neighbors, using only its outer-train bank.
- Fit the preregistered inner-fold error model `baseline_error ~ baseline_margin + class + modality-energy + base-cluster + quotient_pressure`; evaluate only on the held OOF fold.
- Require quotient pressure to add at least `+0.03` OOF error-AUC beyond the controls on both datasets, with positive foldwise sign and a paired bootstrap lower bound above zero.
- Compute the first-step full-bank `L_SQ` gradient's directional alignment with the exact true-class retrieval-margin improvement direction. Require its active-anchor coverage and positive-alignment rate to exceed both within-class posterior shuffle and strength-matched random posteriors by at least 10 percentage points on both datasets.
- This is an enrichment/coverage test, not a claim that environment similarity itself predicts vote correctness and not a theoretical upper bound.

If P0.1 or P0.2 fails on either dataset, SQ stops before new teacher calls. No prompt/model/ontology/threshold sweep is allowed on the same evidence.

### Reliability, fallback, and corruption

- `r_i` is the teacher's calibrated confidence; entropy is reported separately and is not silently equated with confidence.
- Missing, extra-key, parse-failed, low-confidence, or unsupported records have `r_i=0` and contribute no quotient relation.
- Confidence calibration uses only train folds. No threshold is chosen from validation/test.
- Corruption controls mix each posterior with the dataset marginal by preregistered levels `{0.25,0.50,0.75,1.00}`; gain must degrade monotonically within uncertainty.
- Missingness stress randomly masks active records at matched levels. The deterministic endpoint remains the base RGCL path for every masked record.

### Required assignment controls

All controls use the same encoder, bank refreshes, updates, epochs, initialization, data order, pair budget, and first-step aggregate auxiliary-gradient norm where applicable.

1. **REMOVE / base RGCL:** `lambda_Q=0` and no posterior read.
2. **LABEL_ONLY:** uniform same-label positives and different-label negatives with active pair counts and aggregate gradient norm matched to FULL; tests generic supervised contrastive gain.
3. **SHUFFLE:** permute complete posterior/confidence records within dataset, video class, and confidence stratum; preserves class-conditional posterior marginals and strength but destroys video–environment assignment.
4. **STRENGTH-MATCHED RANDOM:** random soft posteriors matched in category marginal, entropy distribution, confidence, missingness, active pair count, and aggregate gradient norm.
5. **BASE-CLUSTER:** label-blind soft environments from the frozen base embedding with the same six-way capacity.
6. **CHEAP-FORMAT:** the frozen prototype posterior from existing `neutral_summary`; if this is the FULL artifact in P0, a later direct MLLM posterior must beat it before promotion.
7. **P4-FIELD-PREDICTION:** predict the exact six-way posterior with an auxiliary KL/CE head discarded at inference, with aggregate gradient norm matched. This tests whether quotient geometry—not archive-field prediction—causes the gain.
8. **POSTERIOR CORRUPTION/MASKING:** calibrated mixing and missingness controls described above.

No stance, target, mechanism, explicitness, harm, label-related archive field, GroupDRO, router, or segment weighting control is permitted to leak into FULL.

### Failure Modes and Diagnostics

- **Posterior collapse or label proxy:** Detect low effective environment count, excessive `other`, class-pure environment cells, or q-only label predictability; stop or report as dataset confounding, never reinterpret harm fields as nuisance.
- **Sparse relation support:** Detect per-class ESS and active-anchor failures; no-edge fallback, then stop at P0.
- **P2-like orthogonality:** Detect failure of conditional error-AUC and directional-enrichment gates; stop before training/teacher spend.
- **Generic supervised contrastive gain:** FULL fails to beat LABEL_ONLY or matched random; MLLM mechanism fails even if absolute metrics improve.
- **P4-like prediction redundancy:** FIELD-PREDICTION matches FULL; the quotient-specific claim fails.
- **Over-contraction of real subclasses:** Track within-class neighbor churn, class recall, and per-environment confusion; bounded `lambda_Q`, class-transversal repulsion, and corruption curve are diagnostics, not post-hoc tuning licenses.
- **Only native head improves:** Reject; only ordinary final kNN is a primary endpoint.
- **CTE misinterpretation:** Never cite CTE C0 STOP as an SQ performance ceiling; it was a frozen numerics-threshold decision.

### Novelty and Elegance Argument

Language-guided spurious-correlation mitigation, invariant subspaces, HSIC, semantic KD, and environment-specific concept directions are crowded. SQ therefore makes a narrower claim. It does not learn a nuisance representation or remove a predicted attribute. It uses a confidence-bearing MLLM posterior to define **two coupled, class-conditional relations in the exact full-bank retrieval likelihood**: cross-environment/same-label contraction and same-environment/different-label repulsion. These relations change the representation used by ordinary kNN, then disappear entirely.

The closest algorithmic alternative is environment-weighted supervised contrastive learning. SQ's defensible delta is the crossed-fiber pairing, soft uncertainty, exact final-memory endpoint, and causal control suite showing that environment assignment—not more contrastive pairs, field prediction, base clustering, or gradient strength—creates the gain. If FULL cannot beat those controls, the novelty claim is withdrawn regardless of raw accuracy.

## Claim-Driven Validation Sketch

### Claim 1 — presentation posterior defines dense, conditionally useful quotient pressure

- **Minimal experiment:** `SQ-P0` strict five-fold train OOF on MHC and MHC-ZH, zero new calls.
- **Baselines/ablations:** baseline margin/class/modality/base-cluster model; within-class posterior shuffle; matched random posterior.
- **Metrics:** coverage, per-class ESS, effective environments, incremental error-AUC, gradient directional-alignment enrichment, confidence/missingness.
- **Decisive evidence:** Every density gate passes, incremental error-AUC `>=+0.03`, and directional-enrichment positive-rate exceeds both nulls by `>=10` points on both datasets.

### Claim 2 — MLLM-conditioned quotient geometry causes substantial final kNN improvement

- **Minimal experiment:** after P0 and bounded pilot pass, paired FULL vs all critical controls on MHC and MHC-ZH, seeds 0/1/2, ordinary full-video kNN.
- **Baselines/ablations:** moving strongest non-MLLM RGCL, REMOVE, LABEL_ONLY, SHUFFLE, matched random, BASE-CLUSTER, CHEAP-FORMAT, P4-FIELD-PREDICTION, corruption/masking.
- **Metrics:** accuracy, macro-F1, every paired delta, mean±std, hierarchical paired bootstrap, Holm correction, neighbor churn and mechanism removal cost.
- **Decisive evidence:** Both metrics improve `>=+0.030` on both datasets with all three paired signs positive; corrected significance passes; FULL significantly beats REMOVE and SHUFFLE; the same posterior degrades under calibrated corruption; no test artifact exists.

## Experiment Handoff Inputs

- **Must-prove claims:** Dense conditional enrichment; quotient-specific gain beyond labels/prediction/clusters; final two-dataset +3/+3 causal improvement.
- **Must-run ablations:** REMOVE, LABEL_ONLY, within-class SHUFFLE, strength-matched RANDOM, BASE-CLUSTER, CHEAP-FORMAT, P4-FIELD-PREDICTION, posterior corruption and masking.
- **Critical datasets/metrics:** MHC-EN and MHC-ZH; final ordinary full-video top-20 kNN accuracy and macro-F1.
- **Highest-risk assumptions:** Existing summaries contain enough presentation information without harm-field leakage; each class spans multiple environments; quotient pressure is conditionally related to errors; paired contraction/repulsion does not destroy real hateful subclasses; novelty survives Yang/CDAL/CARE and standard environment-weighted SupCon comparisons.

## Compute & Timeline Estimate

- **Refinement:** documentation only; zero GPU-hours, zero jobs, zero new MLLM calls.
- **P0 estimate:** CPU/one-GPU SLURM only after a separate experiment plan and code review; approximately 2–6 GPU-hours for strict OOF gradient/influence diagnostics, with no teacher calls.
- **Bounded teacher pilot if unlocked:** at most 128 train videos per dataset initially; no validation/test calls. Exact call cap and cost must be frozen by the experiment plan.
- **Seed-0/full runs:** locked until P0 and teacher-value gates pass. Final paired seeds remain locked until both datasets pass seed-0 against all critical controls.
- **Data/annotation cost:** zero human segment annotation; no segment annotation exists or is assumed.
