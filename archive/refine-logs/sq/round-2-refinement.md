# Round 2 Refinement

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM into hateful-video RGCL as a meaningful, novel, causally removable method component that produces a substantial improvement in the final ordinary full-video train-memory kNN classifier—not merely in localization, audit quality, an auxiliary/native head, or qualitative explanations.
- **Must-solve bottleneck:** Existing RGCL memory geometry confounds the binary hate decision with whole-video presentation/context nuisance. Prior MLLM routes were label-redundant, nearly orthogonal to vote correctness, too sparse in their correctable unit, or competed with the memory readout. The new route must use train-only MLLM information to reshape the exact embedding consumed by final kNN, prove dense conditional value beyond video labels and cheap controls, and avoid merely moving accuracy between heads.
- **Non-goals:** No segment/timestamp/span/localization task; no segment weighting; no stance/harm/evidence/target/mechanism field as a nuisance variable; no final-verdict teacher; no archive/schema prediction as the proposed mechanism; no concat/fusion/reranking/router/GroupDRO/test-time MLLM; no larger-model/data/epoch/ensemble story; and no claim that a cost/numerics preflight is a theoretical performance upper bound.
- **Constraints:** The only gold supervision is the parent video's binary label. No segment-level gold exists or may be assumed. Every MLLM/archive quantity is a weak or privileged train-only pseudo-signal with an explicit confidence, deterministic missing/low-confidence fallback, remove/shuffle/noise controls, and no teacher/environment artifact at validation or test inference. All eventual compute must run through SLURM in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no code and launches no job.
- **Success condition:** On at least two datasets, paired seeds 0/1/2, the final ordinary full-video kNN must improve both accuracy and macro-F1 by at least +0.030 absolute over the moving strongest same-protocol non-MLLM comparator; every paired delta must be positive, hierarchical paired bootstrap lower bounds must exceed zero, the four dataset×metric tests must survive Holm correction, and FULL must beat remove-MLLM plus a within-split permutation of the MLLM information with a significant same-direction removal cost. The method may stop only after these final gates—not after proposal readiness, preflight, or seed-0 evidence.

## Anchor Check

- **Original bottleneck:** Make a train-only MLLM signal conditionally useful to the exact final memory geometry, with substantial final acc/mF1 gains.
- **Still solved:** Every method and gate remains whole-video, train-only, and ordinary-kNN facing.
- **Drifting suggestions rejected:** None. `q→y` is correctly demoted from a false causal gate rather than used to redefine nuisance.

## Simplicity Check

- **Dominant contribution:** One soft presentation posterior and one vote-exposed crossed-fiber ranking loss.
- **Removed:** Free `eta/kappa`; a universal `q→y` cutoff; ambiguous 128-video pair sampling.
- **Not added:** No head, subspace, HSIC, adversary, router, GroupDRO, segment path, or test artifact.
- **Why minimal:** Repository vote exposure supplies the only ranking prior; all other changes are fail-closed audits or controls.

## Changes Made

1. **Nuisance validity:** `q→y` is now a reported contamination diagnostic that triggers blind audit, never a universal pass/fail proxy. Hard gates are label-blind generation provenance, presentation-only blind audit, zero forbidden semantics, per-environment×class effective mass/ESS, class-pure-cell rejection, and per-anchor relation ESS.
2. **Graph-closed SQ-1:** A deterministic strict-OOF top-20 graph closure chooses anchors and every required endpoint under `<=128` unique videos/dataset. Both classes need `>=16` anchors and `>=200` observed directed edges. Every teacher/control analysis uses the exact same vertices/edges and anchor-cluster bootstrap. Two prompts × two orders freeze a maximum of 512 calls/dataset.
3. **Vote exposure:** Top-20 exposure now equals the repository rank weight times harmful signed cosine contribution. A parameter-free harmonic continuation starts from rank 20; `eta/kappa` are deleted.
4. **Archive provenance and controls:** `q^arch` is invalid unless the original neutral-summary generator prompt/model/input manifest proves labels were absent. ENV-SUPCON, Yang-style, and P4-PREDICT losses and pair-budget matching are now explicit.

## Revised Proposal

# Research Proposal: SQ-RGCL — Presentation-Crossed Vote-Exposed Ranking

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM into hateful-video RGCL as a meaningful, novel, causally removable method component that produces a substantial improvement in the final ordinary full-video train-memory kNN classifier—not merely in localization, audit quality, an auxiliary/native head, or qualitative explanations.
- **Must-solve bottleneck:** Existing RGCL memory geometry confounds the binary hate decision with whole-video presentation/context nuisance. Prior MLLM routes were label-redundant, nearly orthogonal to vote correctness, too sparse in their correctable unit, or competed with the memory readout. The new route must use train-only MLLM information to reshape the exact embedding consumed by final kNN, prove dense conditional value beyond video labels and cheap controls, and avoid merely moving accuracy between heads.
- **Non-goals:** No segment/timestamp/span/localization task; no segment weighting; no stance/harm/evidence/target/mechanism field as a nuisance variable; no final-verdict teacher; no archive/schema prediction as the proposed mechanism; no concat/fusion/reranking/router/GroupDRO/test-time MLLM; no larger-model/data/epoch/ensemble story; and no claim that a cost/numerics preflight is a theoretical performance upper bound.
- **Constraints:** The only gold supervision is the parent video's binary label. No segment-level gold exists or may be assumed. Every MLLM/archive quantity is a weak or privileged train-only pseudo-signal with an explicit confidence, deterministic missing/low-confidence fallback, remove/shuffle/noise controls, and no teacher/environment artifact at validation or test inference. All eventual compute must run through SLURM in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no code and launches no job.
- **Success condition:** On at least two datasets, paired seeds 0/1/2, the final ordinary full-video kNN must improve both accuracy and macro-F1 by at least +0.030 absolute over the moving strongest same-protocol non-MLLM comparator; every paired delta must be positive, hierarchical paired bootstrap lower bounds must exceed zero, the four dataset×metric tests must survive Holm correction, and FULL must beat remove-MLLM plus a within-split permutation of the MLLM information with a significant same-direction removal cost. The method may stop only after these final gates—not after proposal readiness, preflight, or seed-0 evidence.

## Technical Gap and Thesis

RGCL predicts with a similarity-signed, rank-weighted top-20 vote. Presentation format can attract a wrong-class memory even when format is not the hate decision. P2 showed generic archive comparability did not select vote-correct memories. P4 showed predicting decodable, label-informative archive fields was redundant. SSR/EDCM were too sparse or frozen; CTE C0 stopped only on a preregistered numerics tolerance and supplies no performance ceiling.

**Thesis:** A label-blind train-only MLLM presentation posterior is useful only if it identifies dense nuisance-driven wrong-class attraction; one crossed relation then ranks a same-label/different-presentation memory above a different-label/same-presentation memory on the exact embedding consumed by final kNN.

**Dominant contribution:** confidence-weighted presentation-crossed triplet ranking whose negative exposure is inherited from the repository vote. This is a narrow method-level claim, not general causal deconfounding or first environment invariance.

## Complexity Budget and System

- Reuse existing shared RGCL encoder/fusion, base losses, epoch bank, train split, and exact final vote.
- New trainable pieces: zero. New weak artifact: train-only `(q_i,r_i)`.
- Proposed additions: one scalar ranking loss; no nuisance predictor or quotient head.
- Excluded: harm/stance/evidence fields, segment objectives, field concat/prediction, memory editor, test teacher, reranker, score fusion, HSIC, adversary, subspace, GroupDRO, router, MoE.

```text
train whole video -> shared RGCL embedding z_i -> refreshed full train bank
video binary y_i -------------------------------> sign of crossed pair
train-only q_i,r_i -----------------------------> presentation relation weight
repository rank/signed cosine ------------------> current vote exposure
all four ---------------------------------------> one L_SQ update on z

val/test whole video -> same z -> unchanged top-20 train-memory kNN
                                  no q/archive/teacher/environment
```

## Presentation-Only Posterior Contract

Fixed categories:

1. `news_reportage`
2. `satire_skit`
3. `educational_explanatory`
4. `personal_narrative_discussion`
5. `gaming_music_entertainment`
6. `other_unclear`

The prompt asks only **how the whole video is presented**. It forbids and does not output hate/non-hate, stance, endorsement, harm, target, mechanism, explicitness, evidence modality, difficulty, prediction, margin, neighbor, split/seed, rationale, timestamp, span, segment, or localization. Strict output is six ordered probabilities, confidence, parse status, and prompt/model/input hashes. Extra semantic keys fail closed.

The original generator provenance for every existing `neutral_summary` must prove, through code/prompt/model/input-manifest hashes, that video labels and predictions were absent. Auditing only the current reader is insufficient. If provenance is missing or inconsistent, `q^arch` is invalid and zero-new-call SQ-0 cannot proceed from that artifact.

### Hard nuisance-validity gates

- Complete label-blind generation provenance and forbidden input/output access count `=0`.
- Blind whole-video presentation audit on a frozen sample: no hate/stance/harm/target/mechanism category interpretation; contamination is STOP. This is signal QC, not a gold training label.
- Parse/coverage `>=0.90`.
- For every active environment `e` and class `c`, effective posterior mass `sum_{i:y_i=c} r_i q_i[e] >=10` and Kish ESS `>=8`; an environment with zero effective mass for either class is class-pure and invalid.
- Each class has effective environment count `exp(H(weighted mean q))>=2.5`.
- At least 80% of anchors in each class have `ESS>=8` for both same-label/cross-environment positives and different-label/same-environment negatives.
- Confidence predicts two-prompt/two-order agreement; low confidence deterministically becomes `r=0`.

`q→y` OOF AUC, accuracy, calibration, and incremental label predictiveness are always reported. They trigger targeted contamination audit when unusually high but are not a universal nuisance-validity threshold: real presentation imbalance can correlate with labels, and low AUC cannot prove semantic purity.

## Core Vote-Exposed Crossed Relation

Let `z_i` be the normalized final-memory embedding, `y_i` the parent-video binary label, `q_i` the weak presentation posterior, and `r_i` confidence. Posterior affinity is

`A_ij=sum_e sqrt(q_i[e]q_j[e])`.

For anchor `i`:

- `u_ip=r_i r_p(1-A_ip)` for `y_p=y_i` (same class across presentation);
- `v_in=r_i r_n A_in` for `y_n!=y_i` (different class inside presentation).

Normalize `u` and `v` separately. Unsupported anchors fall back exactly to base RGCL.

At a detached epoch-bank refresh, let `rho_i(j)` be rank and `s_ij=z_i·z_j`. Define the rank weight

`a(r)=21-r` for `1<=r<=20`, and `a(r)=1/(r-19)` for `r>20`.

The top-20 branch is exactly the repository arithmetic rank weight. The tail is the unique parameter-free harmonic continuation from rank-20 weight one; it gives old top-64 outsiders and the rest of the bank nonzero learning exposure without an `eta/kappa` sweep.

Let `t_ij=+1` for same label and `-1` otherwise. Harmful signed-vote exposure is

`E_i(j)=a(rho_i(j))*max(0,-t_ij s_ij)`.

For a negative `n`, this is precisely its positive-similarity contribution against the true-class vote inside top 20, with the frozen continuation outside. The sole new loss is

`L_SQ(i)=E_{p~U_i,n~V_i}[ E_i(n) softplus((s_in-s_ip+mu)/tau) ]`.

It makes a correct cross-presentation memory outrank an exposed wrong-class same-presentation memory. The positive `p` is sampled from the entire outer-train bank, so the action family is not a discrete old-neighborhood swap. Query and key embeddings share the encoder and co-move across refreshes. Claims say “repository-vote-exposed ranking surrogate,” not exact differentiation through top-k.

Training uses `L_RGCL+lambda_Q L_SQ`. Each active anchor draws exactly 64 triplets with stateless RNG keyed by `(seed,epoch,anchor_id)`; every sampled id, mass, and fallback is logged. All control arms use the same 64-triplet budget, epochs, refreshes, initialization, data order, optimizer, and matched aggregate first-step auxiliary-gradient norm.

## Staged Gates

### P0 — zero-new-call density and P2-conditioned enrichment

Use only strict train-OOF embeddings, binary labels for diagnostic endpoints, and provenance-valid existing summaries.

1. Pass all provenance, presentation audit, positivity, coverage, and two-sided ESS gates.
2. On actual OOF top-20 directed edges, adding `A_ij` to a preregistered pair model containing similarity, rank, query class, base margin, modality energy, and base-cluster relation must improve held-out `wrong_class_neighbor` AUC by `>=0.03` on both datasets, positive in all folds with anchor-cluster bootstrap lower bound above zero.
3. Query quotient pressure (exposed same-environment wrong-class mass minus cross-environment correct-class mass) must add `>=0.03` held-out baseline-error AUC.
4. First-step positive gradient-alignment rate must exceed within-class posterior shuffle and strength-matched random by `>=10` points.

P0 is only a fast-fail. It is neither a theoretical upper bound nor performance evidence.

### SQ-0 — zero-new-call learned strict-OOF actual-kNN screen

Train five outer-fold models with `q^arch`, never using outer-held q/labels/loss during training. Evaluate outer-held ordinary repository kNN.

GO requires on both MHC and MHC-ZH:

- `>=+0.050` OOF accuracy and `>=+0.050` OOF macro-F1 over exact OOF base;
- `>=+0.010` both metrics over LABEL_ONLY, within-class SHUFFLE, and matched RANDOM;
- positive foldwise signs and bootstrap lower bounds above zero;
- decreased actual wrong-class top-20 signed mass, increased correct signed mass, and corrections beyond SSR/EDCM old universes.

A stronger label-only/base-cluster arm raises the moving non-MLLM comparator. Failure on either dataset stops SQ before new calls; no ontology/prototype/exposure/loss tuning follows.

### SQ-1 — graph-closed actual-MLLM pilot

Before any call, build each dataset's directed strict-OOF top-20 graph. Deterministically choose anchors by class and baseline-margin stratum while greedily maximizing endpoint overlap; include every selected anchor and all of its required top-20 endpoints. The frozen closure must satisfy:

- at most 128 unique vertices per dataset;
- at least 16 anchors per class and at least 200 observed directed anchor→neighbor edges total;
- every analyzed edge has teacher posterior at both endpoints;
- selection hashes, vertices, anchors, edges, labels used only for stratification/audit, and baseline ranks are frozen before calls.

Two prompts × two input orders yield at most `128×4=512` calls per dataset, `1024` total. Greedy decoding and no retries beyond one deterministic infrastructure retry are frozen. If a valid graph closure cannot meet the anchor/edge minima, SQ-1 is infeasible and stops without calls.

Every comparison—teacher, `q^arch`, base cluster, within-class shuffle, matched random—uses exactly this vertex/edge universe. Uncertainty is clustered/bootstrapped by anchor, never by treating edges as independent.

Teacher `q^T` must pass every hard nuisance-validity gate and beat all four alternatives by:

- at least `+0.02` pair-level wrong-neighbor AUC and `+0.02` query-pressure error-AUC;
- at least 10 points positive gradient-alignment rate with anchor-bootstrap lower bound above zero;
- no worse anchor/relation ESS; no class-specific failure; pooled success cannot rescue.

Passing permits train-only full posterior extraction, never validation/test calls.

### SQ-2 — seed-0 mechanism gate

On both datasets, validation ordinary kNN accuracy and macro-F1 must each exceed REMOVE, LABEL_ONLY, SHUFFLE, ENV-SUPCON, Yang-style, P4-PREDICT, BASE-CLUSTER/CHEAP-FORMAT, and matched RANDOM by `>=+0.010`; posterior corruption/masking must degrade gain monotonically.

### SQ-3 — final target

Paired seeds 0/1/2 on MHC-EN and MHC-ZH; unchanged final ordinary full-video kNN. Both metrics must improve `>=+0.030` over the moving strongest comparator, all paired signs positive, hierarchical paired bootstrap lower bounds above zero, four tests Holm-corrected, and FULL significantly beats REMOVE and SHUFFLE. Only SQ-3 can satisfy the project goal.

## Exact Controls

All controls share the same backbone, training schedule, 64-triplet/anchor budget where applicable, and first-step aggregate auxiliary-gradient norm.

- **REMOVE:** `lambda_Q=0`, no posterior read.
- **LABEL_ONLY:** same triplet softplus/exposure, with uniform same-label positives and different-label negatives.
- **SHUFFLE:** permute complete q/r records within dataset×class×confidence stratum.
- **RANDOM:** draw q from a fitted logistic-normal null preserving category marginal, entropy, confidence, missingness, and active mass; rescale once to matched gradient norm.
- **BASE-CLUSTER:** six-way label-blind soft k-means posterior from the frozen base embedding.
- **CHEAP-FORMAT:** provenance-valid `q^arch` from neutral-summary prototypes.
- **ENV-SUPCON:** same `q/r/u/v` but independent weighted positive numerator and negative denominator NCA, with no coupled `(p,n)` comparison and no vote exposure.
- **YANG-STYLE:** a linear six-attribute predictor trained on q plus cross-covariance decorrelation between class logits/embedding and predicted presentation attributes; head discarded; parameter/update/gradient strength matched.
- **P4-PREDICT:** a linear soft-posterior KL head only, discarded, with no decorrelation and matched strength.
- **CORRUPTION/MISSING:** mix q with its train-fold marginal at `{.25,.50,.75,1}` or mask active records; fallback is REMOVE.

## Failure Handling and Novelty Boundary

- Missing provenance, forbidden semantics, class-pure cell, low positivity/ESS, prompt instability, or P2-like orthogonality: STOP.
- SQ-0 misses actual `+.05/+.05`: STOP before new calls.
- SQ-1 teacher fails common-edge value against cheap/base/shuffle/random: MLLM unnecessary, STOP.
- ENV-SUPCON/Yang/P4 matches FULL: crossed-fiber novelty/mechanism fails even if raw metrics rise.
- Native-head-only improvement is irrelevant. No teacher artifact is available to validation/test.
- CTE C0 remains a numerics-policy STOP, never a performance ceiling.

The defensible novelty is narrow: **a train-only MLLM soft presentation assignment jointly defines a positive class fiber crossing and a negative class transversal inside one repository-vote-exposed full-bank ranking constraint.** The method is paper-worthy only if the common-posterior controls prove that coupling and vote exposure—not extra metric learning, field prediction, or generic decorrelation—cause the final kNN gain.

## Claim-Driven Validation and Handoff

1. **Nuisance validity/conditional value:** P0 and graph-closed SQ-1; provenance, blind audit, positivity/ESS, common-edge wrong-neighbor AUC, pressure, and anchor-bootstrap gradient value.
2. **Learned capacity and final causality:** actual OOF SQ-0, seed-0 SQ-2, final paired SQ-3; ordinary kNN accuracy/mF1 and exact controls above.

Refinement itself uses zero jobs, zero GPU-hours, and zero new calls. An experiment plan must independently audit code and microbenchmark strict OOF through SLURM before SQ-0. Only parent-video binary labels are gold; `segment_gold_exists=false`, `segment_gold_used=false`.
