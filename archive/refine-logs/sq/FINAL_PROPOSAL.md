# Research Proposal: SQ-RGCL — Presentation-Crossed Exact-Vote-Exposed Ranking

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM into hateful-video RGCL as a meaningful, novel, causally removable method component that produces a substantial improvement in the final ordinary full-video train-memory kNN classifier—not merely in localization, audit quality, an auxiliary/native head, or qualitative explanations.
- **Must-solve bottleneck:** Existing RGCL memory geometry confounds the binary hate decision with whole-video presentation/context nuisance. Prior MLLM routes were label-redundant, nearly orthogonal to vote correctness, too sparse in their correctable unit, or competed with the memory readout. The new route must use train-only MLLM information to reshape the exact embedding consumed by final kNN, prove dense conditional value beyond video labels and cheap controls, and avoid merely moving accuracy between heads.
- **Non-goals:** No segment/timestamp/span/localization task; no segment weighting; no stance/harm/evidence/target/mechanism field as a nuisance variable; no final-verdict teacher; no archive/schema prediction as the proposed mechanism; no concat/fusion/reranking/router/GroupDRO/test-time MLLM; no larger-model/data/epoch/ensemble story; and no claim that a cost/numerics preflight is a theoretical performance upper bound.
- **Constraints:** The only gold supervision is the parent video's binary label. No segment-level gold exists or may be assumed. Every MLLM/archive quantity is a weak or privileged train-only pseudo-signal with an explicit confidence, deterministic missing/low-confidence fallback, remove/shuffle/noise controls, and no teacher/environment artifact at validation or test inference. All eventual compute must run through SLURM in `HateVideo`, without `--time`, within 2 GPU / 16 CPU / 128 GB. This refinement changes no code and launches no job.
- **Success condition:** On at least two datasets, paired seeds 0/1/2, the final ordinary full-video kNN must improve both accuracy and macro-F1 by at least +0.030 absolute over the moving strongest same-protocol non-MLLM comparator; every paired delta must be positive, hierarchical paired bootstrap lower bounds must exceed zero, the four dataset×metric tests must survive Holm correction, and FULL must beat remove-MLLM plus a within-split permutation of the MLLM information with a significant same-direction removal cost. The method may stop only after these final gates—not after proposal readiness, preflight, or seed-0 evidence.

## Technical Gap and Thesis

RGCL predicts with a similarity-signed, rank-weighted top-20 train-memory vote. Whole-video presentation can create nuisance attraction: reportage resembles reportage, skits resemble skits, and gaming footage resembles gaming footage even across the binary hate boundary. P2 showed that generic semantic comparability is nearly orthogonal to vote correctness. P4 showed that predicting decodable archive fields is redundant with binary supervision. SSR and EDCM were limited by sparse/frozen correctable universes. CTE C0 stopped only on a frozen numerical tolerance, so it is not a performance upper bound.

**Thesis:** A label-blind train-only MLLM presentation posterior is useful only if it identifies dense nuisance-conditioned wrong-class top-20 attraction; one crossed relation can then rank a same-label/different-presentation full-bank memory above the exposed different-label/same-presentation memory on the shared final embedding.

**Dominant contribution:** Confidence-weighted presentation-crossed triplet ranking with exact repository vote exposure. The claim is deliberately narrow: an MLLM-defined positive class-fiber crossing and negative class transversal are coupled in one RGCL-vote-facing ranking constraint. This is not causal deconfounding, general quotient theory, or first metric learning.

## Complexity Budget and System

- Reuse the existing RGCL encoder/fusion, base losses, refreshed full train bank, split/selection protocol, and final top-20 vote.
- Add no trainable module and no inference path.
- Add only train records `(q_i,r_i)` and one scalar loss `L_SQ`.
- Exclude stance/harm/evidence/target/mechanism/explicitness nuisance fields, segments, field prediction/concat, key editing, reranking, score fusion, GroupDRO, router, MoE, HSIC, adversarial learning, and explicit nuisance subspaces from FULL.

```text
train whole video -> shared RGCL embedding z_i -> refreshed full train bank
video binary y_i -------------------------------> class sign
train-only MLLM q_i,r_i ------------------------> presentation relation weight
repository top-20 rank/signed cosine -----------> harmful vote exposure
all four ---------------------------------------> one L_SQ update on z

val/test whole video -> same z -> unchanged train-memory kNN
                                  no q/archive/teacher/environment
```

## Presentation-Only Teacher Contract

All six categories are active a priori:

1. `news_reportage`
2. `satire_skit`
3. `educational_explanatory`
4. `personal_narrative_discussion`
5. `gaming_music_entertainment`
6. `other_unclear`

The MLLM classifies only **how the whole video is presented**. It receives no gold label, prediction, margin, neighbor, split/seed, stance, endorsement, harm act, target, mechanism, explicitness, evidence field, rationale, timestamp, span, segment, or localization. Its strict output contains six probabilities, confidence, parse status, and provenance hashes. Extra semantic keys fail closed.

For two prompts and two input orders, the four posteriors are averaged and renormalized:

`q_i^T = normalize((q_i^{a1}+q_i^{a2}+q_i^{b1}+q_i^{b2})/4)`.

Reliability is the minimum of the four confidences. Any parse failure, mean pairwise JS divergence above `0.10`, or modal-category agreement below `0.80` sets `r_i=0`. There is no semantic repair or retry beyond one identical-payload infrastructure retry.

Existing `neutral_summary` can produce the zero-new-call `q^arch` only if the original generator prompt/model/input-manifest hashes prove labels and predictions were absent. Current reader-side key filtering alone is insufficient.

### Hard validity and positivity gates

- Complete label-blind generation provenance and zero forbidden input/output access.
- Parse/coverage at least 90%.
- Each of the six environment×class cells has effective mass `sum r_i q_i[e]>=10` and Kish ESS at least 8; a class-pure cell is STOP.
- Each class has effective environment count at least 2.5; at least 80% of anchors/class have ESS at least 8 on both crossed relations.
- A fixed blind audit samples 64 unique train videos/dataset inside label-blinded posterior-argmax strata. Two independent raters, blinded to label/prediction/margin, inspect only whole-video presentation and the posterior; a third blinded rater adjudicates disagreements. Contamination above 5% or a 95% Wilson lower bound for presentation appropriateness below 0.90 is STOP. Audit marks are QC only and never train supervision.
- OOF `q→y` AUC/accuracy/calibration are reported as contamination diagnostics, not used as a false independence gate.

## Core Mechanism

Let `z_i` be the normalized final-memory embedding, `y_i` the parent-video binary label, `q_i` the weak presentation posterior, and `r_i` its confidence. Define posterior affinity

`A_ij=sum_e sqrt(q_i[e]q_j[e])`.

For anchor `i`, define full-bank weights:

- `u_ip=r_i r_p(1-A_ip)` for `y_p=y_i`: same class across presentation;
- `v_in=r_i r_n A_in` for `y_n!=y_i`: different class inside presentation.

Normalize positive and negative weights separately. Anchors without two-sided ESS support fall back exactly to base RGCL.

At each detached epoch-bank refresh, compute repository rank `rho_i(j)` and cosine `s_ij`. Let `t_ij=+1` for the same class and `-1` otherwise. Exact harmful-vote exposure is

`E_i(j)=(21-rho_i(j))*max(0,-t_ij s_ij)` for `rho_i(j)<=20`, and `0` otherwise.

The `20..1` arithmetic weights and similarity sign match the authoritative evaluator. There is no rank>20 prior. The sole new term is

`L_SQ(i)=E_{p~U_i,n~V_i}[E_i(n) softplus((s_in-s_ip+mu)/tau)]`.

Each active anchor draws 64 triplets using a stateless RNG keyed by seed, epoch, and anchor. Negatives are current harmful top-20 memories; positives come from the complete outer-train bank. The shared query/key encoder and bank co-move at every refresh, so old-neighborhood outsiders may enter the vote without a frozen edge/swap universe.

Training uses `L_RGCL+lambda_Q L_SQ`. Validation/test use only full-video `z` and unchanged ordinary top-20 kNN. They never load `q`, archive text, teacher output, confidence, or environment.

## Staged Gates

### P0 — zero-new-call governance and conditional enrichment

Using strict train-OOF geometry and provenance-valid existing summaries only:

- pass every provenance, audit, coverage, environment×class positivity, and two-sided ESS gate;
- on actual top-20 directed edges, posterior affinity must add at least `+0.03` held-out wrong-class-neighbor AUC beyond similarity/rank/class/margin/modality/base-cluster controls, with positive fold signs and an anchor-cluster bootstrap lower bound above zero;
- quotient pressure must add at least `+0.03` held-out baseline-error AUC;
- first-step positive alignment rate must beat within-class shuffle and strength-matched random by at least 10 points.

P0 is only a fast-fail. It is neither a theoretical upper bound nor an MLLM performance claim.

### SQ-0 — zero-new-call learned strict-OOF actual-kNN screen

Train five outer-fold models using only outer-train video labels and `q^arch`. Outer-held labels are endpoint-only.

Both MHC and MHC-ZH must show:

- at least `+0.050` OOF accuracy and `+0.050` OOF macro-F1 over the exact base;
- at least `+0.010` on both metrics over LABEL_ONLY, within-class SHUFFLE, and matched RANDOM;
- positive fold signs and bootstrap lower bounds above zero;
- lower actual wrong-class top-20 signed mass and corrections beyond the SSR/EDCM old universes.

A stronger non-MLLM arm raises the moving comparator. Any dataset failure stops SQ before new teacher calls; no observed-outcome retuning of ontology, prototypes, exposure, or loss is allowed.

### SQ-1 — representative, graph-closed, power-valid actual-MLLM pilot

Before calls:

1. On the complete q-free strict-OOF top-20 graph, compute anchor-level cheap-vs-shuffle variance for within-anchor wrong/correct-neighbor AUC and positive alignment. Use the upper 95% variance bound.
2. Compute class-specific independent-anchor requirements for 80% power under Holm-adjusted two-sided familywise alpha 0.05 to detect `+0.02` AUC and `+0.10` alignment effects. Edges are never independent replicates.
3. With a frozen seed, uniformly sample the required anchors within class×OOF-margin quartiles, proportional to the full stratum; do not optimize hub degree or endpoint overlap.
4. Add every sampled anchor's top-20 endpoint. If the closure exceeds 128 unique videos/dataset or cannot meet class-specific power, write `STOP_INFEASIBLE` before calls. Do not shrink the sample or weaken power.
5. Freeze vertices, edges, ranks, strata, sampling probabilities, and hashes. Report selected-versus-full degree/margin/base-cluster balance; use inverse-probability weights and anchor-cluster inference.

The cap is at most 128 unique videos/dataset (256 total) and four invocations/video (512/dataset, 1024 total). Every teacher, archive, base-cluster, shuffle, and random analysis uses exactly the same vertices, edges, and estimand.

`q^T` must pass the hard validity gates and beat every alternative by at least `+0.02` within-anchor AUC and 10 points alignment with adjusted class-specific lower bounds above zero. No pooled rescue. Passing permits extraction for remaining training records, never validation/test calls.

### SQ-2 — seed-0 mechanism gate

On both datasets, validation ordinary kNN accuracy and macro-F1 must each beat REMOVE, LABEL_ONLY, SHUFFLE, ENV-SUPCON, Yang-style, P4-PREDICT, BASE-CLUSTER/CHEAP-FORMAT, and RANDOM by at least `+0.010`; posterior corruption/masking must degrade gains monotonically.

### SQ-3 — final target

Run paired seeds 0/1/2 on MHC-EN and MHC-ZH with unchanged ordinary full-video kNN. Both metrics must improve by at least `+0.030` over the moving comparator; every paired sign must be positive; hierarchical paired-bootstrap lower bounds must exceed zero; four dataset×metric tests must survive Holm correction; FULL must significantly beat REMOVE and SHUFFLE. Only SQ-3 can satisfy the project objective.

## Matched Controls

- **REMOVE:** no posterior or `L_SQ`.
- **LABEL_ONLY:** the same exposed triplet loss with uniform same/different-label sampling.
- **SHUFFLE:** permute complete q/r records within dataset×class×confidence stratum.
- **RANDOM:** soft posteriors matched in marginal, entropy, confidence, missingness, active mass, and aggregate gradient strength.
- **BASE-CLUSTER:** label-blind six-way soft clusters from frozen base embeddings.
- **CHEAP-FORMAT:** provenance-valid neutral-summary prototype posterior.
- **ENV-SUPCON:** the same q/u/v in independent weighted NCA numerator/denominator, with no coupled triplet or vote exposure.
- **YANG-STYLE:** linear presentation-attribute prediction plus class/presentation cross-covariance decorrelation; capacity/update/gradient matched and discarded at inference.
- **P4-PREDICT:** linear KL prediction of q only, discarded and strength matched.
- **CORRUPTION/MISSING:** mix q with its train-fold marginal at `{.25,.50,.75,1}` or mask records; fallback is REMOVE.

All arms match backbone, initialization, data order, epochs, bank refresh, optimizer, pair budget where applicable, and aggregate first-step auxiliary-gradient norm.

## Failure and Novelty Boundary

Missing provenance, semantic contamination, failed positivity/ESS, P2-like orthogonality, SQ-0 below `+.05/+.05`, underpowered/oversized graph closure, teacher failure against common-edge alternatives, or FULL failure against ENV-SUPCON/Yang/P4 are terminal route failures, not tuning invitations. Native-head-only gain is irrelevant. CTE C0 remains a numerics-policy STOP, never a performance ceiling.

The novelty claim must remain: **MLLM-defined presentation crossing × exact-vote-exposed RGCL ranking**. Controls establish causal mechanism attribution; a separate final literature check is required before a paper novelty claim.

## Validation and Handoff

- **Claim 1:** the posterior is presentation-valid and conditionally identifies wrong-class vote attraction—P0 and power-valid common-edge SQ-1.
- **Claim 2:** crossed ranking causes substantial ordinary-kNN gains—learned OOF SQ-0, seed-0 SQ-2, paired SQ-3.

Next step is an independent experiment plan and implementation audit for P0/SQ-0 only. It must freeze the power formula/worst-case variance/finite-population correction/bootstrap count, verify evaluator rank/cosine/tie parity, define audit examples/adjudication records, validate archive provenance, and microbenchmark bank/triplet/control costs. **No new teacher call is allowed before SQ-0 GO.**

Only the parent-video binary label is gold: `segment_gold_exists=false`, `segment_gold_used=false`.
