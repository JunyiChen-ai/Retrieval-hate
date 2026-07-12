# Round 3 Refinement

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM into hateful-video RGCL as a meaningful, novel, causally removable method component that produces a substantial improvement in the final ordinary full-video train-memory kNN classifier—not merely in localization, audit quality, an auxiliary/native head, or qualitative explanations.
- **Must-solve bottleneck:** Existing RGCL memory geometry confounds the binary hate decision with whole-video presentation/context nuisance. Prior MLLM routes were label-redundant, nearly orthogonal to vote correctness, too sparse in their correctable unit, or competed with the memory readout. The new route must use train-only MLLM information to reshape the exact embedding consumed by final kNN, prove dense conditional value beyond video labels and cheap controls, and avoid merely moving accuracy between heads.
- **Non-goals:** No segment/timestamp/span/localization task; no segment weighting; no stance/harm/evidence/target/mechanism field as a nuisance variable; no final-verdict teacher; no archive/schema prediction as the proposed mechanism; no concat/fusion/reranking/router/GroupDRO/test-time MLLM; no larger-model/data/epoch/ensemble story; and no claim that a cost/numerics preflight is a theoretical performance upper bound.
- **Constraints:** The only gold supervision is the parent video's binary label. No segment-level gold exists or may be assumed. Every MLLM/archive quantity is a weak or privileged train-only pseudo-signal with an explicit confidence, deterministic missing/low-confidence fallback, remove/shuffle/noise controls, and no teacher/environment artifact at validation or test inference. All eventual compute must run through SLURM in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no code and launches no job.
- **Success condition:** On at least two datasets, paired seeds 0/1/2, the final ordinary full-video kNN must improve both accuracy and macro-F1 by at least +0.030 absolute over the moving strongest same-protocol non-MLLM comparator; every paired delta must be positive, hierarchical paired bootstrap lower bounds must exceed zero, the four dataset×metric tests must survive Holm correction, and FULL must beat remove-MLLM plus a within-split permutation of the MLLM information with a significant same-direction removal cost. The method may stop only after these final gates—not after proposal readiness, preflight, or seed-0 evidence.

## Anchor Check

- The anchor remains verbatim and the endpoint remains ordinary full-video kNN.
- The four changes below alter only the SQ-1 estimand and repository alignment.
- No reviewer request caused task, supervision, or inference drift.

## Simplicity Check

- **Dominant contribution:** unchanged—one posterior, one exact-top-20-exposed crossed ranking loss.
- **Deleted:** the entire rank>20 exposure branch; fixed anchor/edge minima.
- **Added modules:** none. Power, sampling, aggregation, and audit rules are protocol definitions.
- **Why still minimal:** distant positives may enter the live top 20 because the full shared bank co-moves; no far-negative prior is necessary.

## Changes Made

1. **Exact exposure:** `E=0` beyond rank 20. Inside top 20, exposure is exactly `(21-rank) * max(0,-t*s)`, matching the evaluator's arithmetic rank weights and similarity sign.
2. **Power-valid pilot:** A pre-call anchor-cluster power calculation uses the upper 95% variance bound from full-graph cheap-vs-shuffle anchor statistics. It computes class-specific required anchors for `+0.02` within-anchor AUC and `+0.10` alignment-rate effects at Holm-adjusted two-sided alpha 0.05 and 80% power. Failure to fit their endpoint closure under 128 unique videos is `STOP_INFEASIBLE` before calls.
3. **Representative closure:** Anchors are uniformly sampled within class×OOF-margin quartiles using a frozen seed before endpoints are added. No hub-overlap optimization occurs. Degree/margin/base-cluster standardized differences and known stratum sampling probabilities are reported; class-specific estimates are inverse-probability weighted and anchor-bootstrapped.
4. **Artifact freeze:** Four posteriors are averaged; reliability is their minimum confidence; failure of the frozen agreement gate sets `r=0`. All six ontology categories are active. Blind audit and invocation accounting are frozen.

## Revised Proposal

# Research Proposal: SQ-RGCL — Presentation-Crossed Exact-Vote-Exposed Ranking

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM into hateful-video RGCL as a meaningful, novel, causally removable method component that produces a substantial improvement in the final ordinary full-video train-memory kNN classifier—not merely in localization, audit quality, an auxiliary/native head, or qualitative explanations.
- **Must-solve bottleneck:** Existing RGCL memory geometry confounds the binary hate decision with whole-video presentation/context nuisance. Prior MLLM routes were label-redundant, nearly orthogonal to vote correctness, too sparse in their correctable unit, or competed with the memory readout. The new route must use train-only MLLM information to reshape the exact embedding consumed by final kNN, prove dense conditional value beyond video labels and cheap controls, and avoid merely moving accuracy between heads.
- **Non-goals:** No segment/timestamp/span/localization task; no segment weighting; no stance/harm/evidence/target/mechanism field as a nuisance variable; no final-verdict teacher; no archive/schema prediction as the proposed mechanism; no concat/fusion/reranking/router/GroupDRO/test-time MLLM; no larger-model/data/epoch/ensemble story; and no claim that a cost/numerics preflight is a theoretical performance upper bound.
- **Constraints:** The only gold supervision is the parent video's binary label. No segment-level gold exists or may be assumed. Every MLLM/archive quantity is a weak or privileged train-only pseudo-signal with an explicit confidence, deterministic missing/low-confidence fallback, remove/shuffle/noise controls, and no teacher/environment artifact at validation or test inference. All eventual compute must run through SLURM in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no code and launches no job.
- **Success condition:** On at least two datasets, paired seeds 0/1/2, the final ordinary full-video kNN must improve both accuracy and macro-F1 by at least +0.030 absolute over the moving strongest same-protocol non-MLLM comparator; every paired delta must be positive, hierarchical paired bootstrap lower bounds must exceed zero, the four dataset×metric tests must survive Holm correction, and FULL must beat remove-MLLM plus a within-split permutation of the MLLM information with a significant same-direction removal cost. The method may stop only after these final gates—not after proposal readiness, preflight, or seed-0 evidence.

## Thesis, Gap, and Contribution Boundary

Presentation similarity can pull a wrong-class memory into RGCL's final similarity-signed, rank-weighted top-20 vote. P2 established that generic semantic comparability is not vote correctness; P4 established that predicting archive fields is label-redundant. Sparse SSR/EDCM action spaces do not move the full embedding, while CTE C0 was only a numerics-policy STOP.

**Thesis:** a label-blind, train-only MLLM posterior over whole-video presentation is useful if and only if it identifies nuisance-conditioned wrong-class top-20 attraction; one crossed relation can then rank a same-label/different-presentation full-bank memory above that exposed different-label/same-presentation memory on the shared final embedding.

**One contribution:** confidence-weighted presentation-crossed triplet ranking with exact repository vote exposure. The defensible novelty is narrow: coupled positive fiber crossing and negative class transversal inside one RGCL-vote-facing ranking constraint. It is not causal deconfounding, generic quotient theory, or first metric learning.

## Method and Complexity

Use existing RGCL encoder/fusion, base losses, refreshed full train bank, split/protocol, and final top-20 vote. Add no trainable component and no test-time path. Store only train `(q_i,r_i)` and add one scalar `L_SQ`. Stance/harm/evidence/target/mechanism/explicitness, segments, field prediction/concat, key editing, reranking, score fusion, GroupDRO, router, MoE, HSIC, adversary, and explicit subspaces are excluded from FULL.

### Teacher ontology and artifact

All six categories are active a priori and none may be silently removed:

`news_reportage`; `satire_skit`; `educational_explanatory`; `personal_narrative_discussion`; `gaming_music_entertainment`; `other_unclear`.

The MLLM receives no label, prediction, margin, neighbor, stance, endorsement, harm, target, mechanism, explicitness, evidence field, rationale, timestamp, span, segment, or localization. Its strict result is six probabilities, confidence, parse status, and provenance hashes. Extra keys fail.

For prompts `a,b` and input orders `1,2`, let the four outputs be `(q_i^{a1},q_i^{a2},q_i^{b1},q_i^{b2})`. Before calls, freeze

`q_i^T = normalize((q_i^{a1}+q_i^{a2}+q_i^{b1}+q_i^{b2})/4)`

and `r_i=min` of the four reported confidences. If any parse fails, or mean pairwise JS divergence exceeds `0.10`, or modal-category agreement is below `0.80`, set `r_i=0`; there is no repair or retry except one deterministic infrastructure retry that does not change the payload.

Existing `neutral_summary` may produce the zero-new-call `q^arch` only if the original generator prompt/model/input manifest hashes prove that labels and predictions were absent. Current-reader key filtering alone is insufficient.

### Hard validity/positivity gates

- Complete label-blind generation provenance; zero forbidden semantic input/output access.
- Parse/coverage `>=0.90`.
- Every one of the six environment×class cells has effective mass `sum r_i q_i[e]>=10` and Kish ESS `>=8`; zero mass for a class is class-pure and STOP.
- Each class's effective environment count is `>=2.5`; at least 80% anchors/class have both relation ESS `>=8`.
- A fixed blind audit samples 64 unique train videos per dataset uniformly within class-blinded posterior-argmax strata. Two independent raters, blind to label/prediction/margin, view only the whole video and posterior categories and mark presentation appropriateness and hate-semantic contamination; disagreements use a third blinded adjudicator. GO requires no contamination rate above 5% and a 95% Wilson lower bound of presentation-appropriate rate at least 0.90. These marks are QC only and never training supervision.
- `q→y` AUC/accuracy/calibration are diagnostics and trigger contamination review, not universal pass/fail gates.

## One Crossed Ranking Loss

Let normalized final embeddings be `z_i`, video labels `y_i`, posteriors `q_i`, and reliabilities `r_i`. Define posterior affinity

`A_ij=sum_e sqrt(q_i[e]q_j[e])`.

For anchor `i`, full-bank positive weights are `u_ip=r_i r_p(1-A_ip)` for `y_p=y_i`; negative weights are `v_in=r_i r_n A_in` for `y_n!=y_i`. Normalize each side separately; unsupported anchors fall back exactly to base RGCL.

At each detached epoch-bank refresh, compute the repository rank `rho_i(j)` and cosine `s_ij`. With `t_ij=+1` for same label and `-1` otherwise, define exact harmful-vote exposure

`E_i(j)=(21-rho_i(j))*max(0,-t_ij s_ij)` if `rho_i(j)<=20`, otherwise `0`.

The `20..1` weights and similarity sign match the authoritative arithmetic evaluator. There is no invented tail. The loss is

`L_SQ(i)=E_{p~U_i,n~V_i}[E_i(n) softplus((s_in-s_ip+mu)/tau)]`.

Each active anchor draws 64 triplets with stateless `(seed,epoch,anchor)` RNG; controls match budget and aggregate first-step gradient norm. Negatives are exactly current harmful top-20 memories; positives come from the whole outer-train bank. Thus distant correct-class memories can move into top 20 as the shared query/key encoder and bank co-move across refreshes. This is not a fixed edge/swap universe and does not require rank>20 negative exposure.

Train `L_RGCL+lambda_Q L_SQ`. Validation/test use only full-video `z` and the unchanged ordinary kNN; they never read teacher/archive/environment artifacts.

## Staged Gates

### P0 — zero-new-call governance and conditional enrichment

Using strict train-OOF geometry and provenance-valid existing summaries only:

- pass all provenance, audit, coverage, environment×class positivity, and two-sided relation ESS gates;
- on actual top-20 edges, posterior affinity must add `>=0.03` held-out wrong-class-neighbor AUC beyond similarity/rank/class/margin/modality/base-cluster controls, positive in all folds with anchor-cluster bootstrap lower bound above zero;
- query quotient pressure must add `>=0.03` held-out baseline-error AUC;
- positive first-step alignment rate must beat within-class shuffle and strength-matched random by `>=10` points.

P0 is only a fast-fail, never an upper bound or performance claim.

### SQ-0 — learned strict-OOF actual-kNN screen

Five outer-fold models train only on outer-train video labels and `q^arch`; outer-held labels are endpoint-only. Both datasets must show ordinary OOF kNN `>=+0.050` accuracy and `>=+0.050` macro-F1 over exact base, plus `>=+0.010` on both metrics over LABEL_ONLY, within-class SHUFFLE, and matched RANDOM. Every fold sign and bootstrap lower bound must be positive; actual wrong-class top-20 signed mass must fall and corrections extend beyond SSR/EDCM old universes. A stronger non-MLLM arm raises the moving comparator. Failure stops before new calls.

### SQ-1 — representative, graph-closed, power-valid teacher pilot

Before any teacher call:

1. On the complete q-free strict-OOF top-20 graph, compute per-anchor cheap-vs-shuffle statistics: within-anchor AUC for separating wrong from correct neighbors when both exist, and binary/continuous positive alignment. For each class and endpoint, take the upper 95% confidence bound of the anchor-level variance.
2. Compute required independent anchors for a paired two-sided test with 80% power, familywise `alpha=0.05` Holm-adjusted across the two endpoints within each class, to detect `+0.02` AUC and `+0.10` alignment-rate effects. Freeze the maximum required count `n_req[c]`; edges are never treated as independent.
3. With a public frozen seed, sample `n_req[c]` anchors uniformly without replacement inside each class×OOF-margin-quartile stratum, proportional to the full stratum. Labels are used only for train-class stratification, never shown to teacher. Do not optimize endpoint overlap or degree.
4. Add every sampled anchor's top-20 endpoint to form the graph closure. If the union exceeds 128 unique videos/dataset, or any class/end point lacks its powered anchors, write `STOP_INFEASIBLE` before calls. Do not shrink, substitute hubs, or weaken power.
5. Freeze vertices/edges/ranks/strata/sampling probabilities. Report selected-versus-full standardized differences in degree, margin, and base-cluster distribution. Use inverse stratum-probability weights for the class-specific estimand and anchor-cluster bootstrap/tests.

The cap is at most 128 unique videos/dataset (`256` across both datasets). Four invocations per video mean at most 512 model invocations/dataset and `1024` total; invocation count, infrastructure retries, parse failures, and unique videos are separately reported.

Every teacher/cheap/base/shuffle/random analysis uses exactly the same vertices, edges, anchor weights, and powered class-specific estimand. `q^T` must pass hard validity and improve over each alternative by `>=+0.02` within-anchor AUC and `>=10` points alignment, with adjusted class-specific confidence lower bounds above zero. No pooled rescue. Passing permits remaining train-only extraction, never validation/test calls.

### SQ-2 — seed-0 mechanism gate

On both datasets, validation ordinary kNN accuracy and macro-F1 must each beat REMOVE, LABEL_ONLY, SHUFFLE, ENV-SUPCON, Yang-style, P4-PREDICT, BASE-CLUSTER/CHEAP-FORMAT, and RANDOM by `>=+0.010`; corruption/masking must degrade monotonically.

### SQ-3 — final target

Paired seeds 0/1/2, MHC-EN and MHC-ZH, unchanged ordinary full-video kNN. Both metrics `>=+0.030` over the moving comparator; every sign positive; hierarchical paired bootstrap lower bounds above zero; four tests Holm-corrected; FULL significantly beats REMOVE and SHUFFLE. Only SQ-3 closes the goal.

## Matched Controls

- **REMOVE:** no posterior/loss.
- **LABEL_ONLY:** same exposed triplet loss, uniform same/different-label sampling.
- **SHUFFLE:** permute whole q/r within dataset×class×confidence stratum.
- **RANDOM:** marginal/entropy/confidence/missingness/active-mass matched soft q, one strength rescale.
- **BASE-CLUSTER:** label-blind six-way soft clusters from frozen base embeddings.
- **CHEAP-FORMAT:** provenance-valid neutral-summary prototype q.
- **ENV-SUPCON:** same q/u/v as independent weighted NCA numerator/denominator, no coupled triplet or vote exposure.
- **YANG-STYLE:** linear presentation-attribute predictor plus class/presentation cross-covariance decorrelation; matched trainable capacity/update/gradient, discarded at inference.
- **P4-PREDICT:** linear KL prediction of q only, discarded, matched strength.
- **CORRUPTION/MISSING:** mix with fold marginal at `{.25,.50,.75,1}` or mask; fallback REMOVE.

All arms match backbone, initialization, data order, epochs, bank refreshes, optimizer, triplet budget where applicable, and first-step auxiliary-gradient norm.

## Failure and Novelty Boundary

Missing provenance, semantic contamination, failed positivity/ESS, P2-like orthogonality, SQ-0 `<+.05/+.05`, underpowered/oversized graph closure, teacher not beating common-edge cheap/base/nulls, or FULL not beating ENV-SUPCON/Yang/P4 are terminal route failures, not tuning invitations. Native-head-only gain is irrelevant. CTE C0 is reported only as a numerics-policy STOP.

The novelty claim remains deliberately narrow: **a train-only MLLM soft presentation posterior jointly assigns a correct-class fiber crossing and an exposed wrong-class transversal inside one exact-top-20-vote-facing full-bank ranking constraint.** Controls establish mechanism attribution; a separate final literature check establishes bibliographic novelty.

## Validation/Handoff

- Claim 1: nuisance-valid posterior conditionally predicts wrong-class vote attraction—P0 and power-valid common-edge SQ-1.
- Claim 2: crossed ranking causes substantial final ordinary-kNN gains—actual OOF SQ-0, seed-0 SQ-2, paired SQ-3.

This refinement used zero jobs, zero GPU-hours, and zero new calls. Any execution requires a separate SLURM experiment plan and code audit. Only parent-video binary labels are gold; `segment_gold_exists=false`, `segment_gold_used=false`.
