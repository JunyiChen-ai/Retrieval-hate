# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM into hateful-video RGCL as a meaningful, novel, causally removable method component that produces a substantial improvement in the final ordinary full-video train-memory kNN classifier—not merely in localization, audit quality, an auxiliary/native head, or qualitative explanations.
- **Must-solve bottleneck:** Existing RGCL memory geometry confounds the binary hate decision with whole-video presentation/context nuisance. Prior MLLM routes were label-redundant, nearly orthogonal to vote correctness, too sparse in their correctable unit, or competed with the memory readout. The new route must use train-only MLLM information to reshape the exact embedding consumed by final kNN, prove dense conditional value beyond video labels and cheap controls, and avoid merely moving accuracy between heads.
- **Non-goals:** No segment/timestamp/span/localization task; no segment weighting; no stance/harm/evidence/target/mechanism field as a nuisance variable; no final-verdict teacher; no archive/schema prediction as the proposed mechanism; no concat/fusion/reranking/router/GroupDRO/test-time MLLM; no larger-model/data/epoch/ensemble story; and no claim that a cost/numerics preflight is a theoretical performance upper bound.
- **Constraints:** The only gold supervision is the parent video's binary label. No segment-level gold exists or may be assumed. Every MLLM/archive quantity is a weak or privileged train-only pseudo-signal with an explicit confidence, deterministic missing/low-confidence fallback, remove/shuffle/noise controls, and no teacher/environment artifact at validation or test inference. All eventual compute must run through SLURM in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no code and launches no job.
- **Success condition:** On at least two datasets, paired seeds 0/1/2, the final ordinary full-video kNN must improve both accuracy and macro-F1 by at least +0.030 absolute over the moving strongest same-protocol non-MLLM comparator; every paired delta must be positive, hierarchical paired bootstrap lower bounds must exceed zero, the four dataset×metric tests must survive Holm correction, and FULL must beat remove-MLLM plus a within-split permutation of the MLLM information with a significant same-direction removal cost. The method may stop only after these final gates—not after proposal readiness, preflight, or seed-0 evidence.

## Anchor Check

- **Original bottleneck:** MLLM semantics have not produced dense, conditionally correct retrieval geometry or substantial final kNN gains.
- **Why the revision remains anchored:** It restores actual strict-OOF final-kNN gates, makes wrong-neighbor attraction the preflight target, and replaces generic NCA with one presentation-crossed ranking loss exposed to the repository's top-20 vote.
- **Reviewer suggestions rejected as drift:** None. No segment/task/endpoint expansion was requested or accepted.

## Simplicity Check

- **Dominant contribution:** One train-only presentation posterior plus one crossed-fiber vote-exposure ranking loss on the existing final embedding.
- **Components removed/merged:** P0 is a gate, not a supporting contribution; generic full-bank NCA is replaced rather than supplemented; there is no separate exact-vote loss, nuisance head, subspace, or predictor.
- **Suggestions rejected as unnecessary complexity:** No HSIC, adversary, router, GroupDRO, explicit quotient projection, or learned environment classifier is added.
- **Smallest adequate route:** The only changed training signal ranks a label-correct cross-environment memory above a label-wrong same-environment memory, weighted by soft teacher uncertainty and current vote exposure.

## Changes Made

### 1. Restored learned strict-OOF SQ-0

- **Reviewer said:** Proxy error-AUC and one-step alignment cannot replace the registered two-dataset actual-kNN capacity gate.
- **Action:** Proxy diagnostics are now P0 only. SQ-0 performs actual nested train-OOF learning and must improve ordinary top-20 kNN accuracy and macro-F1 by at least +0.050 on both datasets, plus beat matched nulls.
- **Reasoning:** Proposal readiness cannot be bought with correlational diagnostics.

### 2. Frozen nuisance-validity and leakage audit

- **Reviewer said:** Label-blind input can still be a stance/harm/label proxy.
- **Action:** Added a presentation-only prompt/schema, forbidden-input/output audit, `q→y` ceiling, class×environment overlap and class-pure-cell gates, and prompt/order stability. Class conditioning cannot rescue a failed audit.
- **Reasoning:** The MLLM must supply nuisance semantics, not a disguised verdict.

### 3. Replaced generic NCA with vote-exposed crossed ranking

- **Reviewer said:** Full-bank NCA is not the repository's signed rank-weighted top-20 vote and looks like weighted SupCon.
- **Action:** The sole proposed term now couples one same-label/cross-environment positive with one different-label/same-environment negative and weights the comparison by the negative's current repository-vote exposure. All-bank tails retain nonzero exposure, so the method is not a frozen top-64 editor. Claims now say “vote-aligned surrogate,” never “exact differentiable kNN.”
- **Reasoning:** Pair order is the minimal trainable object that maps the nuisance hypothesis to neighbor voting.

### 4. Froze actual MLLM SQ-1 and explicit P2/P4/prior-art controls

- **Reviewer said:** The direct teacher must beat cheap posteriors and the method must address P2, P4, and standard weighted SupCon/Yang-style alternatives.
- **Action:** Added a bounded actual-MLLM posterior pilot with hard coverage/leakage/wrong-neighbor enrichment gates; explicit `ENV-SUPCON`, Yang-style attribute decorrelation, and P4-posterior-prediction controls; final FULL must beat them under matched strength.
- **Reasoning:** Raw gains without these separations would not establish the MLLM mechanism or novelty.

## Revised Proposal

# Research Proposal: SQ-RGCL — Soft Quotient Crossed-Fiber Ranking for RGCL Memory

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM into hateful-video RGCL as a meaningful, novel, causally removable method component that produces a substantial improvement in the final ordinary full-video train-memory kNN classifier—not merely in localization, audit quality, an auxiliary/native head, or qualitative explanations.
- **Must-solve bottleneck:** Existing RGCL memory geometry confounds the binary hate decision with whole-video presentation/context nuisance. Prior MLLM routes were label-redundant, nearly orthogonal to vote correctness, too sparse in their correctable unit, or competed with the memory readout. The new route must use train-only MLLM information to reshape the exact embedding consumed by final kNN, prove dense conditional value beyond video labels and cheap controls, and avoid merely moving accuracy between heads.
- **Non-goals:** No segment/timestamp/span/localization task; no segment weighting; no stance/harm/evidence/target/mechanism field as a nuisance variable; no final-verdict teacher; no archive/schema prediction as the proposed mechanism; no concat/fusion/reranking/router/GroupDRO/test-time MLLM; no larger-model/data/epoch/ensemble story; and no claim that a cost/numerics preflight is a theoretical performance upper bound.
- **Constraints:** The only gold supervision is the parent video's binary label. No segment-level gold exists or may be assumed. Every MLLM/archive quantity is a weak or privileged train-only pseudo-signal with an explicit confidence, deterministic missing/low-confidence fallback, remove/shuffle/noise controls, and no teacher/environment artifact at validation or test inference. All eventual compute must run through SLURM in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no code and launches no job.
- **Success condition:** On at least two datasets, paired seeds 0/1/2, the final ordinary full-video kNN must improve both accuracy and macro-F1 by at least +0.030 absolute over the moving strongest same-protocol non-MLLM comparator; every paired delta must be positive, hierarchical paired bootstrap lower bounds must exceed zero, the four dataset×metric tests must survive Holm correction, and FULL must beat remove-MLLM plus a within-split permutation of the MLLM information with a significant same-direction removal cost. The method may stop only after these final gates—not after proposal readiness, preflight, or seed-0 evidence.

## Technical Gap

The final RGCL prediction is a similarity-signed, rank-weighted top-20 train-memory vote. Presentation format can attract a wrong-class neighbor for a reason unrelated to hate: reportage resembles reportage, a skit resembles a skit, and gaming footage resembles gaming footage. P2 proves that asking an MLLM which neighbor is “comparable” does not solve this—comparability was nearly independent of vote correctness. P4 proves that predicting label-informative archive fields also does not solve it—decodable fields were redundant with the binary objective.

SQ uses neither operation. Video labels determine whether a pair should agree or disagree; the MLLM supplies only soft presentation context. The method changes pair order in the shared memory embedding so a label-correct example from another presentation environment outranks a label-wrong example from the same presentation environment.

## Method Thesis and Contribution

- **Thesis:** Train-only MLLM presentation posteriors can correct nuisance-driven retrieval attraction when they weight a crossed-fiber ranking loss on the same full-video embedding consumed by final kNN.
- **Dominant contribution:** A confidence-weighted triplet relation coupling `(same label, different environment)` with `(different label, same environment)`, exposed to the current exact-vote neighborhood but learned over the full bank.
- **Explicit boundary:** This is not causal deconfounding, generic disentanglement, first semantic KD, or exact differentiation through kNN. It is a vote-aligned surrogate whose mechanism must beat standard environment-weighted SupCon and language-attribute decorrelation.

## Complexity Budget

- Reuse the existing RGCL encoder/fusion, train bank, base losses, final top-20 vote, splits, epochs, and selection protocol.
- Add no trainable module and no test-time path.
- Add only train records `(q_i,r_i)` and one scalar loss.
- Exclude nuisance prediction, projections, HSIC, adversaries, routers, GroupDRO, segment losses, memory editing, reranking, and score fusion.

## Presentation-Only Teacher Contract

The ordered posterior support is fixed as:

`news_reportage`, `satire_skit`, `educational_explanatory`, `personal_narrative_discussion`, `gaming_music_entertainment`, `other_unclear`.

The MLLM prompt says that it is classifying **how the whole video is presented**, never whether a statement is hateful, harmful, endorsed, quoted, explicit, targeted, evidential, or correct. The strict output contains only six probabilities summing to one, `confidence`, `parse_ok`, source hash, model/prompt hash, and input-order id. Extra semantic keys are parse failure.

Forbidden teacher inputs/outputs and nuisance fields include label, prediction, margin, neighbor, stance, endorsement, target, mechanism, harm act, explicitness, evidence modality, rationale, timestamp, span, segment, and localization. Existing archive P0 may read only `archive.neutral_summary`; it must not read the outer label/raw output or any other archive field. A direct MLLM pilot never sees a gold label.

### Nuisance-validity gates

All are preregistered and must pass separately on MHC and MHC-ZH:

- forbidden-key/input access count `=0`; parse/coverage `>=0.90`;
- every active environment has effective posterior mass `>=20` and minority-class mass share in `[0.20,0.80]`; no active class-pure cell;
- each class has effective environment count `>=2.5` and active-anchor rate `>=0.80`;
- strict-OOF logistic `q→y` AUC `<=0.70` and accuracy advantage over the class majority `<=0.10`; failure means label-proxy risk and stops the route, even if metrics look good;
- two presentation-only prompt wordings and two input orders have mean Jensen–Shannon divergence `<=0.10` on confident records and category agreement `>=0.80`;
- a fixed blind audit reports category appropriateness and any hate-semantic contamination; it is a signal audit, not new gold supervision.

## Core Mechanism

Let normalized final-memory embeddings be `z_i`, binary video labels be `y_i`, presentation posteriors be `q_i`, and reliabilities be `r_i`. Posterior affinity is the Bhattacharyya coefficient

`A_ij = sum_e sqrt(q_i[e] q_j[e])`.

For anchor `i`, define:

- positive fiber crossings `P_i={p:y_p=y_i}`, weighted `u_ip=r_i r_p(1-A_ip)`;
- negative class transversals `N_i={n:y_n!=y_i}`, weighted `v_in=r_i r_n A_in`.

Weights are normalized separately per anchor and require `ESS>=8` on both sides. Missing, malformed, low-confidence, or unsupported records have `r=0` and use only base RGCL.

At each epoch refresh, rank the detached full bank by the repository cosine score. Let `rank_i(n)` be a negative's rank. Define a frozen exposure within that refresh:

`E_i(n)=1` for `rank<=20`, otherwise `eta*exp(-(rank-20)/kappa)` with preregistered positive `eta,kappa`.

The nonzero tail preserves a full-bank learned action space: old top-64 outsiders can move into the vote. The proposed loss is

`L_SQ(i)=sum_{p in P_i,n in N_i} U_ip V_in E_i(n) softplus((z_i·z_n - z_i·z_p + mu)/tau)`.

It ranks a cross-presentation correct-class memory above a same-presentation wrong-class memory, prioritizing negatives currently exposed to the exact top-20 vote. Neighbor identities/ranks are detached within a refresh; the shared query/key encoder and whole bank move across refreshes. The objective is

`L=L_RGCL+lambda_Q L_SQ`.

There is no separate quotient head. Validation/test export `z` and execute the unchanged repository vote with no `q`, archive, teacher, or environment artifact.

## Why this is not P2, P4, or ordinary weighted SupCon

- **Not P2:** No semantic neighbor is kept/dropped. Label signs come from train gold, and `q` must conditionally enrich concrete wrong-class top-20 attraction before it can influence learning.
- **Not P4:** The model never predicts `q`; a matched q-prediction head is a required control.
- **Not ordinary environment-weighted SupCon/NCA:** SQ couples one cross-environment positive and one same-environment negative in a vote-exposed pair-order constraint. A standard independently weighted positive/negative SupCon using the same `q` is a required control.
- **Not Yang/CDAL/CARE:** No discovered-attribute decorrelation, invariant subspace, independence loss, or environment head is used. A matched Yang-style language-attribute decorrelation arm is a required novelty control.

## Staged Training and Stop Gates

### P0 — zero-new-call governance, density, and P2-style conditional enrichment

Use existing strict five-fold train-OOF embeddings and permitted archive summaries only. No new MLLM call.

1. Pass every nuisance-validity and per-class `ESS>=8` gate.
2. Within each OOF query's actual top-20, fit a pair model for `wrong_class_neighbor` using similarity, rank, query class, base margin, modality energy, and base-cluster relation. Adding `A_ij` must improve held-out pair-AUC by `>=0.03` on both datasets with positive foldwise sign and paired-bootstrap lower bound above zero.
3. Quotient pressure—the signed exposed mass of same-environment wrong-class neighbors minus cross-environment correct-class memories—must add `>=0.03` held-out query-error AUC beyond the same controls.
4. First-step directional alignment and active coverage must beat within-class q-shuffle and matched-random posterior by `>=10` percentage points.

These are fast-fail diagnostics only. Passing them does not unlock final experiments without SQ-0.

### SQ-0 — learned strict-OOF actual-kNN capacity screen

Train five outer-fold models using only each outer-train partition and `q^arch` from existing summaries. Outer-held videos receive no quotient loss, teacher artifact, or label during training; their labels are used only for the final OOF endpoint.

SQ-0 GO requires, on **both** datasets:

- ordinary top-20 OOF kNN gains `>=+0.050 accuracy` and `>=+0.050 macro-F1` over the exact frozen OOF base;
- FULL exceeds LABEL_ONLY, within-class SHUFFLE, and strength-matched RANDOM by `>=+0.010` on both metrics;
- every foldwise delta is positive and confidence intervals exclude zero;
- actual wrong-class top-20 mass decreases, correct-vote mass increases, and corrected errors are not confined to the SSR/EDCM old candidate universes.

If a non-MLLM LABEL_ONLY or base-cluster arm improves, it becomes part of the moving comparator. If SQ-0 fails either dataset, no new teacher call is permitted and SQ stops. Thresholds, ontology, prototype text, bank exposure, or loss are not tuned after failure.

### SQ-1 — bounded actual-MLLM posterior-value pilot

Only after SQ-0 GO, query at most 128 strict train videos per dataset, stratified without validation/test information. Freeze two prompt wordings, two input orders, greedy decoding, confidence, schema, missing fallback, and call cap before extraction.

Actual `q^T` must pass all nuisance-validity gates and, relative to `q^arch`, BASE-CLUSTER, within-class SHUFFLE, and matched RANDOM:

- active coverage and two-relation ESS are no worse;
- pair-level wrong-class top-20 incremental AUC and query-level quotient-pressure AUC each improve by at least `+0.02`;
- positive gradient-alignment rate improves by at least 10 points with bootstrap lower bound above zero;
- posterior confidence predicts prompt/order agreement; low confidence maps deterministically to `r=0`;
- no class fails independently; pooled success cannot rescue a failed class.

Failure stops SQ. Passing authorizes train-only posterior extraction for the remaining training records, not validation/test calls.

### SQ-2 — seed-0 mechanism gate

On both datasets, validation ordinary kNN accuracy and macro-F1 must each beat REMOVE, LABEL_ONLY, SHUFFLE, ENV-SUPCON, P4-PREDICT, BASE-CLUSTER/CHEAP-FORMAT, and matched RANDOM by at least `+0.010`; corruption response must be monotone. Only then freeze and run final seeds.

### SQ-3 — final target

MHC-EN and MHC-ZH, paired seeds 0/1/2, unchanged final ordinary full-video kNN. Binding target is `max(historical strongest, paired moving comparator mean)+0.030` for both accuracy and macro-F1, all paired signs positive, hierarchical paired bootstrap lower bounds above zero, four tests Holm-corrected, and significant FULL-vs-REMOVE/SHUFFLE removal costs. No other endpoint can close the project goal.

## Assignment and Prior-Art Controls

All arms match initialization, data order, epochs, bank refresh, pair budget, optimizer, and aggregate first-step auxiliary-gradient norm:

1. REMOVE/base RGCL.
2. LABEL_ONLY uniform same/different-label crossed ranking.
3. within-class/confidence-stratified SHUFFLE of complete q records.
4. strength-matched RANDOM q preserving marginal, entropy, confidence, missingness, and active mass.
5. BASE-CLUSTER soft environments from frozen embeddings.
6. CHEAP-FORMAT `q^arch` from permitted existing summaries.
7. ENV-SUPCON/NCA with the same q but standard independent environment-weighted positive/negative terms.
8. YANG-STYLE language-attribute decorrelation with the same allowed presentation ontology and matched capacity.
9. P4-PREDICT auxiliary q prediction, discarded at inference.
10. posterior mixing corruption `{.25,.50,.75,1.0}` and matched missingness.

FULL must beat the appropriate controls; merely beating REMOVE is insufficient.

## Failure Handling

- Forbidden semantics, q-label ceiling, class-pure environments, collapse, sparse ESS, or prompt instability: STOP; class conditioning cannot rescue.
- P2-like lack of wrong-neighbor enrichment: STOP before learned SQ-0 or calls.
- SQ-0 learned OOF fails `+.05/+.05` on either dataset: STOP before new calls.
- Actual MLLM does not beat cheap/base/shuffle in SQ-1: MLLM is unnecessary; STOP.
- ENV-SUPCON/Yang/P4 control matches FULL: novelty/mechanism claim fails even if raw metrics rise.
- Only native head improves: reject.
- CTE C0: report only as a numerics-policy STOP, never a performance upper bound.
- Any missing teacher record: exact base-RGCL fallback; no imputation from labels.

## Novelty and Elegance

The prior-art window is narrow. The claim is deliberately limited to the crossed object that existing local and literature baselines do not instantiate: a soft presentation posterior defines a **class-conditional positive fiber crossing and negative class transversal jointly in one vote-exposed full-bank ranking constraint** on the final RGCL memory embedding. The posterior disappears at inference.

This is paper-worthy only if it beats the same-posterior standard ENV-SUPCON, Yang-style decorrelation, q-prediction, base clusters, cheap format, label-only, shuffle, and matched random controls. If it does not, “soft quotient” is only a rename for weighted supervised contrastive learning and the claim is withdrawn.

## Claim-Driven Validation

### Claim 1: the posterior is nuisance-valid and conditionally actionable

- **Experiment:** P0 plus actual MLLM SQ-1 pilot, both strict train-only.
- **Metrics:** forbidden-key count, q→y ceiling, class×environment overlap, ESS, prompt/order stability, top-20 wrong-neighbor incremental AUC, quotient-pressure AUC, gradient-alignment enrichment.
- **Pass:** every hard gate on both datasets and both classes; no pooled rescue.

### Claim 2: crossed-fiber ranking causes substantial final-kNN improvement

- **Experiment:** learned OOF SQ-0, seed-0 SQ-2, then paired final SQ-3.
- **Controls:** all ten arms above, with emphasis on REMOVE, SHUFFLE, LABEL_ONLY, ENV-SUPCON/Yang, and P4-PREDICT.
- **Endpoint:** unchanged ordinary full-video train-memory top-20 kNN accuracy and macro-F1.
- **Pass:** final two-dataset, three-seed, `+0.030/+0.030`, statistical and removal gates in the immutable anchor.

## Experiment Handoff and Estimate

- Refinement changes documentation only: zero jobs, zero GPU-hours, zero new calls.
- An experiment plan must first implement/audit P0 and SQ-0 under SLURM; estimate 10–30 GPU-hours for ten strict OOF models plus controls, to be microbenchmarked before freeze.
- SQ-1 remains capped at 128 train videos per dataset until SQ-0 GO; no validation/test call is ever allowed.
- Only parent-video binary labels are gold. `segment_gold_exists=false`, `segment_gold_used=false`.
