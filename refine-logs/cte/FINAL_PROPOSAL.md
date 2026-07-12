# Research Proposal: CTE-RGCL — Withholding-Informed Tangent Supervision of Full-Bank Retrieval

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL as a train-only privileged teacher, and do not stop until the unchanged ordinary full-video train-memory kNN endpoint improves by at least `+0.030` absolute in both accuracy and macro-F1 on at least two datasets and paired seeds `0/1/2`, with the full statistical and mechanism-attribution gates.
- **Must-solve bottleneck:** SSR and EDCM proved that sparse relation edges and bounded edits inside the frozen old neighbourhood cannot touch enough errors. The successor must use label-blind MLLM information to change the shared full-video representation and the whole train-memory geometry, while proving that the information is not reducible to video labels, generic modality dropout, intervention artifacts, shuffled relations, or extra optimization.
- **Non-goals:** No localization, segment classification, segment weighting, teacher-selected/replaced memory key, rationale/schema/score/summary concatenation, score fusion, test-time MLLM, reranking, veto, router/MoE, model/data/epoch/ensemble scaling, SSR or EDCM reuse/retuning, native-head-only gain, or protocol relaxation. A zero-teacher screen is a bounded empirical cost/capacity screen, never a theoretical upper bound or evidence of MLLM success.
- **Constraints:** The only gold supervision that exists is the parent video's binary label. There is no segment gold, timestamp gold, span gold, localization gold, stance gold, target gold, mechanism gold, or rationale gold. The MLLM never sees the gold label and may output only confidence-bearing weak relations `preserve`, `weaken`, `reverse`, or `unclear` between a train video's `full` condition and deterministic whole-modality `visual-neutralized` or `language-neutralized` conditions. Validation/test receive only full videos; no teacher record, neutralized view, confidence, relation, or other view artifact exists in their inference path.
- **Success condition:** Relative to `max(historical strongest non-MLLM point, paired same-seed strongest non-MLLM mean)`, FULL gains at least `+0.030` accuracy and `+0.030` macro-F1 on both MHC-EN and MHC-ZH; all three paired-seed deltas are positive; hierarchical paired-bootstrap 95% lower bounds exceed zero and the four dataset-by-metric tests survive Holm correction. FULL must also beat REMOVE, within-fold relation SHUFFLE, relation-free multiview, label-only/heuristic/random-order controls, and calibrated relation NOISE in actual final kNN, with no teacher or neutralized input at test.

## Technical Gap, Thesis and Contribution

Sparse edge/swap actions inside the old bank failed necessary headroom, while segment scores, generated fields, teacher keys and competing heads were redundant, absorbed, mismatched or endpoint-displacing. CTE instead changes shared full-video representation geometry: a frozen MLLM supplies only per-video ordinal whole-modality withholding relations; a parameter-free loss constrains the local response of the full-bank true-class retrieval margin. All full keys move when the shared encoder refreshes, so new neighbours may enter from outside the old top-64.

The relation does not logically identify the gold-class direction. CTE's sole thesis is conditional: **if** label-blind withholding relations transfer class-conditionally to the frozen supported prototype tangent, they can provide assignment-specific privileged supervision beyond video labels and generic multiview regularization. This is tested before full extraction.

- **One dominant contribution:** withholding-informed ordinal weak-relation supervision of shared full-bank retrieval geometry.
- **New trainable components:** zero.
- **Not claimed:** causal identification, first counterfactual/KD/gradient control, localization, new classifier or test-time reasoning.

## System and Complexity

```text
TRAIN: whole-video full evidence
  -> MLLM full vs visual-withheld; full vs language-withheld
  -> (preserve/weaken/reverse/unclear, confidence), no gold or other field
  -> same existing projection/align/MLP encodes full + supported local prototype paths
  -> base RGCL + ordinal cost on exact epoch-refreshed full-bank margin

VAL/TEST: full video -> same selected encoder -> ordinary full train bank -> unchanged kNN
```

Reused are cached full-video visual/language features, projections, `align` fusion, MLP, base RGCL, optimizer/epochs/checkpoint rule, FAISS/top-k/vote and splits. Forbidden are segment views/losses, zero/blank inputs, teacher keys/embeddings, second encoder, head/adapter/router, score channel and test artifact.

## Label-Blind Teacher Record

The full bundle uses deterministic uniform full-video frames and full-video ASR/OCR after stripping timestamps, segment IDs, spans and localization metadata. `visual-neutralized` and `language-neutralized` use typed `CHANNEL WITHHELD BY DESIGN` operators; no black frame, blank string, zero feature, generated replacement, selected segment or gold field.

Frozen meanings are: `preserve` = same latent moderation interpretation with comparable support; `weaken` = same interpretation with reduced support; `reverse` = dominant latent interpretation changes; `unclear` = not reliable. Strict output is only `{"relation":...,"confidence":c}`, `c∈{0,.25,.5,.75,1}`. No absolute label, rationale, score, target/stance/mechanism, timestamp, span or segment is allowed. The teacher sees no label, prediction, margin, error, neighbour, row ID, validation or test record.

Two prompts×two presentation orders are canonicalized per modality. At least 3/4 agree. `rho=agreement_fraction*median_confidence`; parse/tie/unclear/agreement<.75/rho<.5 maps to inactive. Raw calls and failures are immutable train-ID-only artifacts.

## Fixed-Anchor Supported Prototype Tangent

For each modality `m∈{V,L}`, define exactly one prototype candidate `anchor_id^m`: the spherical medoid video ID minimizing summed cosine distance in that modality at the A0-selected checkpoint. For `a∈{.05,.10,.20,.30}`,

\[
\tilde p_i^m(a)=norm((1-a)p_i^m+a p_{anchor\_id^m}^m).
\]

Support is teacher-independent per example and must pass both concatenated projected-pair and pre-MLP fused-space 5-NN radii against unperturbed train full-video points, each bounded by its 95th-percentile leave-one-out radius. Select the largest adjacent pair `(a1,a2)` with >=95% initial joint support and >=80% video-level two-modality coverage.

**Freeze before any teacher call:** for every strict A0 fold/dataset and the post-A0 full-train dataset checkpoint, hash `(anchor_id^V,anchor_id^L,a1,a2)`. Never reselect an anchor or radius. At later refreshes, re-encode those same fixed video IDs with the current shared encoder and recompute only the support mask at both fixed radii. Every control uses the identical IDs/radii/mask rule.

Log per-example direction cosine between current `z_i^{m,a1}-z_i` and the frozen A0 direction. Stop if joint support coverage falls below 90%, median direction cosine below .90, or its 10th percentile below .70. No replacement anchor/radius or dataset-specific rescue is permitted. Thus teacher transfer is always applied to the same validated tangent identity, while the shared encoder may move it within frozen drift limits.

## Exact Epoch-Refreshed Full Bank and Loss

One `f_theta` encodes query and key; no EMA/teacher encoder. At each epoch boundary and checkpoint load, eval-mode encode every full train video into detached normalized keys and re-encode the fixed anchor ID. CTE queries use eval-mode stochastic semantics with autograd. Every train video is a query once/epoch; self ID is excluded; at least one non-self same-class and one opposite-class key are required.

With `s(z,k)=z^T k`,

\[
M_i(z)=\tau LSE_{j\ne i,y_j=y_i}(s/\tau)-\tau LSE_{j,y_j\ne y_i}(s/\tau),
\quad
T_i^m(a)=\tanh\frac{M_i(z_i^{m,a})-M_i(z_i)}{a\max(MAD,sMin)+10^{-6}}.
\]

The bank is exact over every epoch-start key, not continuously current geometry. Stop if median same-ID epoch-start/end cosine <.95 or 95th-percentile angular drift >.25 rad; the only frozen fallback is common half-epoch refresh for every arm.

Freeze `Ip=[-.05,.05]`, `Iw=[-.50,-.20]`, `Ir=[-1,-.50]`. For `[l,u]`, `dist=max(l-T,0,T-u)` and `c=dist^2/4∈[0,1]`. `L_CTE=sum rho*c/(sum rho+eps)` and `L=L_base+lambda L_CTE`. This bounds cost/weight, not gradient. Use normalization epsilon `1e-6`, fail below pre-normalization norm `1e-4`, retain baseline clipping, and log CTE/base gradient ratios.

A0 selects only `tau∈{.05,.10}`, `lambda∈{.05,.10}`, `sMin∈{.05,.10}`. In paired outer fold `f`, one tuple for both datasets maximizes the minimum inner-OOF gain over four dataset×metric cells; tie: smaller lambda, larger sMin, lexicographic. Outer/dev/test/teacher results never choose it. The modal five-fold tuple is frozen from A1 onward.

## A0: Zero-Teacher Bounded Continuous Cost Screen

A0 is a learned empirical screen of this exact path/loss/bank, not a theoretical upper bound or MLLM evidence. In each outer fold, split inner train into A/B/C. For each rotation: fit each L2 logistic probe candidate `{.01,.1,1}` on A; select on B; refit the winner on A∪B; predict C. No video's target comes from a probe trained or selected with that video.

On frozen baseline pre-MLP fused representations, with `q=2y-1`, define the strict OOF target at `a1`:

\[
b_i^m=\min(0,\tanh(q_i[h(g_i^{m,a1})-h(g_i)]/\max(MAD,.05)))
\]

and target interval `[max(-1,b-.05),min(0,b+.05)]`. Cache the strict OOF target for every eligible train video before pilot ID selection.

Compare exact paired strongest non-MLLM REMOVE, assignment-free/mask-free uniform-preserve multiview, and target-histogram/gradient-norm-matched random; in A0 the label-only arm is the gradient-norm reference. Same optimizer steps, refresh and checkpoint rule. Both datasets require: initial/final support gates; outer full-video kNN acc and mF1 each >=+.050 vs REMOVE; >=28 EN and >=29 ZH baseline errors corrected with positive net correction/class; top-20 Jaccard churn >=.10 above random with paired-bootstrap lower bound >0; label-only beats multiview/random in both metrics. A passing label-only arm becomes the moving non-MLLM comparator. Failure is cost STOP, not impossibility.

## A1: At Most 128 Train Videos per Dataset

Before calls, freeze <=128 IDs/dataset by label×OOF-error×margin-tertile proportional allocation and ID-hash order. Maximum calls are `128×2 datasets×2 modalities×2 prompts×2 orders=2048`; no later extraction unless both datasets pass.

At both frozen radii compute `D=-T` with the frozen A0 encoder. Four video folds cross-fit ridge residualization of `D` on margin, energy change, path norm, OOF error and modality indicator. On held-out residuals fit reliability-weighted teacher order `preserve=0,weaken=1,reverse=2`, separately for `y=0` and `y=1`. Permute indivisible two-modality video records 10,000 times within label×margin-tertile; bootstrap videos 10,000 times.

For relation cell `r`, combine a video's modality weights as `w_vr=min(1,sum_{m:r_vm=r}rho_vm)` and define `n_eff=(sum_v w_vr)^2/sum_v w_vr^2`. Require `n_eff>=10` for every dataset×label×relation; all cell summaries are reliability-weighted. The preserve cell must have weighted `|T|<=d0`, the weighted means must satisfy `D_preserve<D_weaken<D_reverse`, and the class-specific ordinal slope 95% lower bound must exceed zero at both radii. Pooled evidence cannot rescue a class.

For the exact pilot update, clone A0 checkpoint per arm; update existing `img_proj/text_proj/retrieval-MLP` only for 20 baseline-lr AdamW steps on 3/4 folds, batch32, fixed ID order, no scheduler, baseline clipping; rebuild prototype/bank and evaluate held-out full queries. Clean must beat the cached strict-OOF label-only target, energy, assignment-free/mask-free multiview, random and feasible SHUFFLE in true-class margin change and wrong-neighbour rate with video-bootstrap lower bound >0 on both datasets.

Let raw call disagreement `e=1-mean_modal_agreement`, computed before any pilot update outcome. Freeze `eta1=clip(e,.10,.25)` and `eta2=min(.50,2eta1)`. A1 uses `eta2` as its one corruption gate; A2/final use both and require clean > eta1 > eta2. Parse completeness >=95%, active coverage >=80%, modal agreement >=.75, Fleiss kappa >=.60 and max relation share <=.85 also bind.

## Assignment Controls

All arms share encoder, fixed anchor/radii, support mask rule, bank, steps and checkpoint budget.

- **REMOVE:** strongest paired non-MLLM, including A0 label-only if stronger.
- **Multiview:** every support-valid view has uniform preserve/weight1; no teacher mask, confidence or assignment. It is assignment-free and teacher-mask-free, not fully teacher-independent: one global scalar uses only the clean training folds' aggregate first-step gradient norm to strength-match without per-video information.
- **Label-only:** cached strict-OOF A0 targets through the same loss.
- **Energy/random:** same relation space/histogram; random matches aggregate gradient norm.
- **SHUFFLE:** before calls, hash fold×label×margin-tertile cell sizes. Cells <2 merge by frozen low→mid then high→mid order. Derange the indivisible `(rV,rhoV,rL,rhoL)` with no fixed point; no post-outcome relaxation.
- **NOISE:** distribution-preserving whole-record swaps at `eta1,eta2`, fixed before update outcomes; confidence/coverage/support remain unchanged.

## A2 and Final Endpoint

After A0/A1 pass, extract and freeze train-only records once. Seed-0 dev requires both datasets' actual full-video kNN acc and mF1 each >=+.010 over every critical control; clean > eta1 > eta2. Test stays locked until pass.

Final MHC-EN/ZH pair seeds 0/1/2. Deterministic effect gate for each metric is `FULL >= max(historical strongest scalar, paired strongest non-MLLM mean)+.030`; all 12 dataset×metric×seed paired deltas versus the same-seed comparator are positive. Statistical inference is only against a same-seed comparator with paired video predictions, never a historical scalar. Hierarchical bootstrap resamples seeds and then draws one shared paired video-ID sample per dataset, applying that identical ID sample to every resampled seed so same-video dependence is preserved; acc/macro-F1 is recomputed in 10,000 replicates. For observed mean delta `dHat`, centered-null replicates are `d_b^0=d_b-dHat`; one-sided `p=(1+sum[d_b^0>=dHat])/(B+1)`. Four dataset×metric p-values use Holm FWER .05 and percentile 95% lower bounds must exceed zero. FULL-minus-REMOVE and FULL-minus-SHUFFLE use the same uncertainty. Inference is only full video, ordinary rebuilt train bank and unchanged kNN.

## Failure Rules, Novelty and Handoff

- Off-support, tangent-direction drift or class transfer failure -> STOP, never anchor/radius/prompt rescue.
- Generic multiview/label/random explanation, nonfinite gradient, excessive bank drift, or native-head-only gain -> STOP.
- Missing/low-confidence teacher record -> exact no-CTE fallback; report class-conditional coverage/confidence/fallback/noise.
- No segment/timestamp/span/localization field or gold exists anywhere; uniform frames/ASR/OCR are only whole-video input.

Closest work covers train-only semantic teachers, retrieval KD, harmful-video modality intervention and counter-reasoning. The narrow delta is label-blind whole-modality withholding relations empirically transferred to a **fixed-anchor supported tangent of the exact epoch-refreshed full-bank true-class margin**, with one shared encoder and no inference dependency.

Three claim blocks suffice: A0 action capacity; A1 <=128 class-conditional assignment value; A2/final actual +3/+3 kNN attribution. The first implementation action is a vectorized full-bank SLURM microbenchmark; only its measured runtime may replace the provisional A0 estimate of 20–40 GPU-hours. A1 <=2048 calls; A2/final only after gates. All compute uses SLURM/HateVideo. Gold annotation cost is zero; pseudo-relations are not annotations.
