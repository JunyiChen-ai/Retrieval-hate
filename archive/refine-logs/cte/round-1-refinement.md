# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL as a train-only privileged teacher, and do not stop until the unchanged ordinary full-video train-memory kNN endpoint improves by at least `+0.030` absolute in both accuracy and macro-F1 on at least two datasets and paired seeds `0/1/2`, with the full statistical and mechanism-attribution gates.
- **Must-solve bottleneck:** SSR and EDCM proved that sparse relation edges and bounded edits inside the frozen old neighbourhood cannot touch enough errors. The successor must use label-blind MLLM information to change the shared full-video representation and the whole train-memory geometry, while proving that the information is not reducible to video labels, generic modality dropout, intervention artifacts, shuffled relations, or extra optimization.
- **Non-goals:** No localization, segment classification, segment weighting, teacher-selected/replaced memory key, rationale/schema/score/summary concatenation, score fusion, test-time MLLM, reranking, veto, router/MoE, model/data/epoch/ensemble scaling, SSR or EDCM reuse/retuning, native-head-only gain, or protocol relaxation. A zero-teacher screen is a bounded empirical cost/capacity screen, never a theoretical upper bound or evidence of MLLM success.
- **Constraints:** The only gold supervision that exists is the parent video's binary label. There is no segment gold, timestamp gold, span gold, localization gold, stance gold, target gold, mechanism gold, or rationale gold. The MLLM never sees the gold label and may output only confidence-bearing weak relations `preserve`, `weaken`, `reverse`, or `unclear` between a train video's `full` condition and deterministic whole-modality `visual-neutralized` or `language-neutralized` conditions. Validation/test receive only full videos; no teacher record, neutralized view, confidence, relation, or other view artifact exists in their inference path.
- **Success condition:** Relative to `max(historical strongest non-MLLM point, paired same-seed strongest non-MLLM mean)`, FULL gains at least `+0.030` accuracy and `+0.030` macro-F1 on both MHC-EN and MHC-ZH; all three paired-seed deltas are positive; hierarchical paired-bootstrap 95% lower bounds exceed zero and the four dataset-by-metric tests survive Holm correction. FULL must also beat REMOVE, within-fold relation SHUFFLE, relation-free multiview, label-only/heuristic/random-order controls, and calibrated relation NOISE in actual final kNN, with no teacher or neutralized input at test.

## Anchor Check

- **Original bottleneck:** obtain dense, assignment-specific MLLM information that changes the shared full-video bank rather than selecting inside the old sparse neighbourhood.
- **Why the revision still addresses it:** the single mechanism remains a train-only ordinal loss on the epoch-refreshed full bank; the revision only makes its intervention transfer, orientation, bank semantics, numerical stability and controls falsifiable.
- **Reviewer suggestions rejected as drift:** absolute teacher class outputs, rationales, timestamps, spans, segment pseudo-labels/weights, teacher keys, new encoders/adapters, test-time views and reranking remain forbidden.

## Simplicity Check

- **Dominant contribution:** withholding-informed ordinal weak relations supervise the supported local response of the exact epoch-refreshed full-bank true-class margin.
- **Components removed or merged:** ambiguous prototype alternatives were reduced to one spherical medoid; conditional-information variants were reduced to one class-conditional cross-fitted ordinal-transfer statistic; all assignment controls reuse the same interval loss.
- **Reviewer suggestions rejected as unnecessary complexity:** no learned calibration, causal model, auxiliary classifier, second encoder, teacher embedding, extra benchmark, or new endpoint was added.
- **Why this remains the smallest adequate route:** zero new trainable components; one shared encoder, one full-video bank, one loss, train-only relation records, and ordinary kNN test inference.

## Changes Made

### 1. Repaired intervention semantics and support

- **Reviewer said:** complete typed withholding and a small prototype path were not the same intervention; marginal support was inadequate.
- **Action:** renamed the mechanism `withholding-informed prototype tangent`; froze a spherical medoid; added per-video joint projected-pair and fused-space support masks; required ordinal-transfer stability at two adjacent supported radii.
- **Reasoning:** this states the actual empirical assumption and gives it a hard train-only falsification gate without adding a module.
- **Impact:** unsupported views are teacher-independently inactive; no zero/blank endpoint is used.

### 2. Repaired label orientation

- **Reviewer said:** relation-only output cannot logically identify the sign of the gold-class tangent.
- **Action:** made gold orientation an explicit weak-label hypothesis and added separate `y=0` and `y=1` cross-fitted ordinal-transfer lower-bound gates at both radii. Pooled evidence cannot pass.
- **Reasoning:** the teacher schema remains relation+confidence and label-blind; empirical transfer, not hidden absolute verdicts, is the only allowed justification.
- **Impact:** a consistent but class-misoriented teacher stops CTE before full extraction.

### 3. Corrected bank and numerical claims

- **Reviewer said:** the bank is exact but epoch-stale; bounded cost did not imply bounded gradients.
- **Action:** renamed it `exact epoch-refreshed full bank`; fixed dot-product similarity, self exclusion, eval-mode keys/queries, drift logging, MAD floor, normalization epsilons and global clipping. The proposal now claims bounded cost/influence weight only.
- **Reasoning:** this matches the repository's detached-bank semantics and removes an incorrect mathematical claim.
- **Impact:** no EMA/second encoder is introduced.

### 4. Made A0/A1 and controls executable

- **Reviewer said:** probe, statistic, update, relation-free mask and Cartesian shuffle were underspecified or infeasible.
- **Action:** specified nested probe targets, shared hyperparameter selection, one video-clustered residualized ordinal statistic, exact pilot update, teacher-independent multiview mask, and pre-audited fold×label×coarse-margin derangement.
- **Reasoning:** these changes remove confounds without expanding the method.
- **Impact:** CTE-1 remains capped at 128 train videos per dataset and teacher extraction beyond it stays locked.

## Revised Proposal

# Research Proposal: CTE-RGCL — Withholding-Informed Counterfactual Tangent Evidence for Full-Bank Retrieval

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL as a train-only privileged teacher, and do not stop until the unchanged ordinary full-video train-memory kNN endpoint improves by at least `+0.030` absolute in both accuracy and macro-F1 on at least two datasets and paired seeds `0/1/2`, with the full statistical and mechanism-attribution gates.
- **Must-solve bottleneck:** SSR and EDCM proved that sparse relation edges and bounded edits inside the frozen old neighbourhood cannot touch enough errors. The successor must use label-blind MLLM information to change the shared full-video representation and the whole train-memory geometry, while proving that the information is not reducible to video labels, generic modality dropout, intervention artifacts, shuffled relations, or extra optimization.
- **Non-goals:** No localization, segment classification, segment weighting, teacher-selected/replaced memory key, rationale/schema/score/summary concatenation, score fusion, test-time MLLM, reranking, veto, router/MoE, model/data/epoch/ensemble scaling, SSR or EDCM reuse/retuning, native-head-only gain, or protocol relaxation. A zero-teacher screen is a bounded empirical cost/capacity screen, never a theoretical upper bound or evidence of MLLM success.
- **Constraints:** The only gold supervision that exists is the parent video's binary label. There is no segment gold, timestamp gold, span gold, localization gold, stance gold, target gold, mechanism gold, or rationale gold. The MLLM never sees the gold label and may output only confidence-bearing weak relations `preserve`, `weaken`, `reverse`, or `unclear` between a train video's `full` condition and deterministic whole-modality `visual-neutralized` or `language-neutralized` conditions. Validation/test receive only full videos; no teacher record, neutralized view, confidence, relation, or other view artifact exists in their inference path.
- **Success condition:** Relative to `max(historical strongest non-MLLM point, paired same-seed strongest non-MLLM mean)`, FULL gains at least `+0.030` accuracy and `+0.030` macro-F1 on both MHC-EN and MHC-ZH; all three paired-seed deltas are positive; hierarchical paired-bootstrap 95% lower bounds exceed zero and the four dataset-by-metric tests survive Holm correction. FULL must also beat REMOVE, within-fold relation SHUFFLE, relation-free multiview, label-only/heuristic/random-order controls, and calibrated relation NOISE in actual final kNN, with no teacher or neutralized input at test.

## Technical Gap and Route Choice

SSR and EDCM change sparse edges or bounded swaps in a frozen old neighbourhood. They cannot reach enough train-OOF errors, but they do not bound a learned shared representation in which all queries and full-video keys move. Prior MLLM routes also show that segment salience, free-text/schema features, teacher keys and competing heads are either redundant, absorbed, mismatched or endpoint-displacing.

The missing interface is per-video information that a binary label cannot specify and that acts directly on the final memory geometry. CTE asks a frozen label-blind MLLM only whether withholding one whole modality preserves, weakens or reverses the full bundle's latent moderation interpretation. That relation does not logically identify a gold-class direction. CTE treats its transfer to a local gold-oriented prototype tangent as a **weak empirical hypothesis**, tests it separately in both video classes, and stops if it fails.

The reserve semantic-quotient route has more prior overlap and more risk of deleting true evidence. It is not combined with CTE.

## Method Thesis and Contribution Focus

- **Thesis:** A label-blind MLLM's confidence-bearing whole-modality withholding relation can, if class-conditionally transferable, supervise a supported local response of the exact epoch-refreshed full-bank true-class retrieval margin and thereby improve unchanged full-video kNN.
- **Dominant contribution:** withholding-informed ordinal weak-relation supervision of shared full-bank retrieval geometry.
- **New trainable components:** none.
- **Explicit non-contributions:** causal identification, counterfactual generation, general semantic/retrieval KD, general gradient control, localization, a new classifier, or test-time reasoning.

## System and Complexity Budget

```text
train full-video evidence only
  -> MLLM compares full vs visual-withheld and full vs language-withheld
  -> two relation+confidence weak records; no gold/prediction/margin/ID

same existing visual/language projections + align fusion + MLP
  -> full query z_i
  -> teacher-independent supported prototype-tangent queries z_i,m^a
  -> exact epoch-refreshed bank of every full train key from same encoder
  -> base RGCL + bounded ordinal interval cost on true-class margin response

validation/test full video only -> same selected encoder -> full train bank -> unchanged kNN
```

Frozen/reused are the cached full-video features, normalized modality projections, `align` fusion, MLP, base RGCL, optimizer/epochs/checkpoint rule, FAISS metric, top-k/vote, splits and full-video endpoint. New artifacts are two train-only weak relations per video. Excluded are segment views, zero/blank inputs, teacher keys/embeddings, a second encoder, extra head, adapter, score channel, router and test artifact.

## Teacher Intervention and Allowed Record

The teacher full bundle contains deterministic uniform full-video frames and automatic full-video ASR/OCR with all timestamps, segment IDs, span fields and localization metadata stripped. `visual-neutralized` replaces the entire visual field with the typed operator `VISUAL CHANNEL WITHHELD BY DESIGN`; `language-neutralized` analogously uses `LANGUAGE CHANNEL WITHHELD BY DESIGN`. No black frame, empty string, zero vector, mean image, generated replacement, segment selection or gold field is supplied.

Prompt semantics are frozen:

- `preserve`: the neutralized whole-video condition supports the same latent moderation interpretation with comparable support;
- `weaken`: it retains that interpretation but reduces its support;
- `reverse`: the dominant latent moderation interpretation changes;
- `unclear`: none can be asserted reliably.

The strict JSON is `{"relation":"preserve|weaken|reverse|unclear","confidence":c}`, where `c` is in `{0,.25,.5,.75,1}` and refers only to the relative relation. Absolute labels, rationales, scores, target/stance/mechanism fields, timestamps, spans and segments are forbidden. The MLLM receives no video label, prediction, margin, error, neighbour, row ID, validation or test record.

Two prompts by two presentation orders are canonicalized per modality. At least three of four must agree. Reliability is modal agreement fraction times median confidence. Parse failure, tie, `unclear`, agreement below .75 or reliability below .5 maps to `unclear,rho=0`. Raw calls and failure records are immutable train-ID-only cache entries.

## Student Neutral Path and Joint Support

Let `p_i^V=norm(W_V e_i^V)` and `p_i^L=norm(W_L e_i^L)`. At every bank refresh and using only the current inner-train full-video projections, modality `m` has exactly one prototype: the **spherical medoid**, the actual inner-train point minimizing summed cosine distance. It is detached.

For `a in A={.05,.10,.20,.30}`,

\[
\tilde p_i^m(a)=\operatorname{norm}((1-a)p_i^m+a c_m),
\]

with the other modality unchanged and the same fusion/MLP. This is a **prototype tangent informed by complete teacher withholding**, not the mathematical tangent of withholding.

Support is teacher-independent and per example. In both (i) concatenated projected-pair space `[pV,pL]` and (ii) the pre-MLP fused space, the perturbed point's 5-NN distance to unperturbed inner-train full-video points must not exceed the corresponding 95th percentile leave-one-out 5-NN radius. Unsupported example/modality/radius cells are inactive in every arm. Choose the largest adjacent pair `(a1,a2)` for which at least 95% of inner-train cells pass both spaces; train at `a1`, audit transfer at both. If no adjacent pair exists or fewer than 80% of videos have two supported modality cells, stop. No zero/blank fallback is allowed.

## Exact Epoch-Refreshed Full Bank

There is one shared encoder `f_theta`; no EMA or teacher encoder. At each frozen epoch boundary and after checkpoint loading, run the model in eval mode, rebuild the spherical medoids, and encode every full inner-train video as a detached normalized key. CTE queries also run in eval-mode dropout/batch-stat semantics while retaining autograd, so bank/query differences are not stochastic-layer artifacts. Every training video is scheduled once as a CTE query per epoch. Self ID is excluded.

For `s(z,k)=z^T k` and a frozen temperature `tau`,

\[
M_i(z;B_t)=\tau\log\sum_{j\ne i,y_j=y_i}e^{s(z,k_j)/\tau}
-\tau\log\sum_{j,y_j\ne y_i}e^{s(z,k_j)/\tau}.
\]

Each bank must contain at least one non-self same-class key and one opposite-class key or the query is inactive. This is exact over every key in the **epoch-start bank**, not the continuously current encoder geometry. Log epoch-start/end parameter displacement and full-bank cosine drift; stop if median same-ID cosine falls below .95 or the 95th-percentile angular drift exceeds .25 radians. Do not repair drift with EMA or teacher keys; a preregistered every-half-epoch refresh is the only permitted fallback and must be used identically by all arms.

## Bounded Tangent Cost, Not a Bounded-Gradient Claim

For supported radius `a`, define the MAD margin scale `s_t` on inner-train full queries and `sHat=max(s_t,sMin)`:

\[
T_i^m(a)=\tanh\left(\frac{M_i(z_i^{m,a};B_t)-M_i(z_i^{full};B_t)}{a\hat s_t+10^{-6}}\right).
\]

Normalization uses epsilon `1e-6`; minimum pre-normalization norms are logged. Freeze intervals `Ip=[-d0,d0]`, `Iw=[-dr,-dw]`, `Ir=[-1,-dr]`, with `0<d0<dw<dr<1`. For interval `[l,u]`,

\[
dist(T,[l,u])=\max(l-T,0,T-u),\quad c(T,I)=dist(T,I)^2/4\in[0,1].
\]

`L_CTE=sum rho*c / (sum rho+eps)` and `L=L_base+lambda*L_CTE`. This bounds per-record **cost and influence weight**, not its gradient. Global gradient clipping remains the baseline value; log CTE/base gradient-norm ratios and fail on non-finite values or pre-normalization norm below `1e-4`.

Before any teacher call, paired nested CTE-0 selects from the finite shared grid `tau in {.05,.10}`, `(d0,dw,dr) in {(.05,.20,.50),(.10,.30,.65)}`, `lambda in {.05,.10}`, and `sMin in {.05,.10}`. For paired outer fold `f`, the same tuple for both datasets maximizes the minimum inner-OOF improvement among the four dataset-by-metric cells; ties use the smallest `lambda`, largest `sMin`, then lexicographic order. No outer prediction, dev/test result or teacher output participates. The modal tuple across the five outer folds, with the same tie rule, is frozen for CTE-1 onward.

## Explicit Weak-Label Orientation Hypothesis

Mapping `preserve/weaken/reverse` to `Ip/Iw/Ir` assumes that degrading the MLLM's latent full interpretation also degrades the gold full-video class margin. The relation record does **not** logically identify that orientation. Four-call agreement and confidence cannot prove it. CTE claims only an empirically validated weak-label transfer and requires the CTE-1 class-conditional test below. No record may be filtered using an MLLM absolute class, rationale, target, segment or gold agreement.

## Stage A0: Zero-Teacher Bounded Continuous Cost Screen

CTE-0 is a learned empirical cost/capacity screen for this exact path/loss/bank implementation. It is not a theoretical upper bound and is not MLLM evidence.

Within each outer fold, form three inner folds. On two folds, fit one L2-regularized binary logistic probe to the frozen baseline's same pre-MLP fused full representation; choose its regularization from `{.01,.1,1}` on the second fold and predict the third, rotating so every inner-train video is out-of-probe-fold. For class sign `q_i=2y_i-1`, let probe true-class margin be `q_i*h(g)`. The teacher-free modality target is

\[
b_i^m=\min\left(0,\tanh\frac{q_i[h(g_i^{m,a1})-h(g_i^{full})]}{s_probe+10^{-6}}\right)\in[-1,0],
\]

where `s_probe=max(MAD,.05)`. Its target interval is `[max(-1,b-.05),min(0,b+.05)]`. The probe and target for a video never use that video's label in probe fitting; its available video label only orients the training target.

Run the exact paired strongest non-MLLM REMOVE with identical optimizer steps, epoch-bank refresh, checkpoint rule and outer full-video kNN. Also run teacher-independent uniform-preserve multiview and target-histogram/gradient-norm-matched random targets. Both datasets must independently satisfy: joint support and >=80% tangent coverage; outer full-video kNN accuracy and macro-F1 each `>=+0.050` versus REMOVE; at least 28 EN and 29 ZH baseline-wrong videos corrected with positive net corrections per class; top-20 neighbour Jaccard churn at least .10 above random with paired-bootstrap lower bound >0; and label-only cost beats multiview/random in both metrics. Passing label-only becomes a moving non-MLLM comparator. Failure is route-cost STOP, not impossibility for other methods.

## Stage A1: At Most 128 Train Videos per Dataset

Before any call, freeze at most 128 train IDs per dataset from video-label × OOF-error × coarse OOF-margin-tertile strata. Deterministic proportional allocation uses ID-hash order; the MLLM never sees strata. The cap is absolute. At most `128*2 datasets*2 modalities*2 prompts*2 orders=2048` calls are possible. Teacher extraction beyond these IDs remains locked.

### Primary ordinal-transfer statistic

For each dataset, use four video-level folds balanced by label, OOF error and margin tertile. At both supported radii, compute frozen A0-encoder `D_i^m(a)=-T_i^m(a)`. Cross-fit an L2 ridge regression of `D` on `[OOF margin, modality-energy change, prototype-path norm, OOF error, modality indicator]`, never on teacher relation. On held-out folds obtain residual `R`. Code teacher order `o(preserve)=0,o(weaken)=1,o(reverse)=2` and fit reliability-weighted `R=beta_y(a)o+error` separately for `y=0` and `y=1`, with a modality fixed effect. The permutation unit is the whole video's two-modality record; 10,000 permutations occur within label × margin-tertile cells. The bootstrap unit is video, 10,000 replicates, 95% percentile interval.

The gate requires, separately on both datasets and both labels: effective weight >=10 for each of the three active relation levels; preserve weighted `|T|<=d0`; ordered weighted means `D_preserve < D_weaken < D_reverse`; and `beta_y(a)>0` with 95% lower bound >0 at **both adjacent radii**. Pooled association cannot rescue either class. This is the single primary conditional-transfer test; effective rank and alternative MI measures are diagnostics only.

### Exact held-out pilot update

Clone the frozen A0-selected checkpoint per arm. In each fourfold rotation, update only existing `img_proj`, `text_proj`, and retrieval MLP parameters on three folds for exactly 20 baseline-learning-rate AdamW steps, batch size 32, fixed ID-hash order, no scheduler, unchanged clipping; output-layer parameters are frozen. Use the same base RGCL minibatches plus the arm's tangent cost. Scalars for control gradient matching are computed from the update folds' first-step aggregate norm only and frozen; no held-out label chooses them. Rebuild prototypes and the full update-fold bank after step 20, then evaluate held-out full-video queries with no relation/view artifact. Primary update outcomes are change in true-class margin and top-20 wrong-neighbour rate. Clean CTE must beat label-only, energy heuristic, teacher-independent multiview, feasible relation shuffle and strength-matched random in both outcomes with video-bootstrap lower bound >0 on both datasets. One frozen distribution-preserving noise rate must reduce the effect. Directional coverage alone cannot pass.

A1 also requires parse completeness >=95%, non-unclear coverage >=80%, four-call modal agreement >=.75, Fleiss kappa >=.60, and max active-relation share <=.85. Failure of any class transfer or dataset gate stops CTE without prompt/model/path rescue.

## Controls

All controls use the same encoder, supported paths, bank refresh, optimizer-step and checkpoint budget.

- **REMOVE:** strongest exact non-MLLM comparator, including passed label-only CTE if it is stronger.
- **Relation-free multiview:** every teacher-independent joint-support-valid modality view receives uniform weight one and the preserve interval. It uses no teacher activity, confidence, missingness or assignment. One global scalar matches the clean arm's aggregate first-step CTE gradient norm using train folds only.
- **Label-only:** the exact A0 bounded target interval through the same loss.
- **Energy heuristic:** fold-local projected-energy change maps to the same relation histogram and fixed uniform reliability.
- **Random:** matches modality, support, relation/confidence histograms and aggregate train-fold gradient norm, but assignments are random.
- **SHUFFLE:** before teacher extraction, audit cells defined only by outer/pilot fold × video label × OOF-margin tertile. If a cell has size <2, merge it with the nearest margin tertile by the frozen order low→mid then high→mid; this feasibility map is hashed before calls. Derange the indivisible `(rV,rhoV,rL,rhoL)` record with no fixed point. Energy/path/difficulty are handled by the primary residualization rather than Cartesian cells. No post-outcome relaxation.
- **NOISE:** within the same feasible cells, perform distribution-preserving swaps of active whole-video records at two rates fixed from pre-outcome pilot disagreement; retain confidence, support and coverage. Clean must be best and degradation monotone.

## Stage A2, Final Endpoint and Statistics

After A0/A1 pass, freeze all train teacher records once. CTE-2 seed 0 requires actual dev full-video kNN accuracy and macro-F1 on both datasets each `>=+0.010` above REMOVE, multiview, label-only, heuristic, random and SHUFFLE; NOISE must degrade monotonically. Test remains locked until this passes.

Final MHC-EN and MHC-ZH runs pair seeds 0/1/2 across FULL and every critical arm. The binding per-metric bar is `max(historical strongest point, paired same-seed strongest non-MLLM mean)+.030`. All 12 dataset×metric×seed FULL-minus-comparator deltas must be positive. For each of four dataset×metric claims, the hierarchical paired bootstrap resamples seeds at the outer level and paired video predictions within seed, recomputing accuracy or macro-F1 from each replicate. The one-sided null is mean paired gain `<=0`; 10,000 replicates yield four p-values, Holm-adjusted at FWER .05, and 95% lower bounds must exceed zero. FULL-minus-REMOVE and FULL-minus-SHUFFLE effects are reported with the same uncertainty for both metrics/datasets. Final inference rebuilds the ordinary full train bank and uses only full dev/test videos, the unchanged FAISS metric/top-k/vote, and no teacher/view artifact.

## Failure Modes and Diagnostics

- **Path unsupported or nonmonotone:** support or two-radius class-transfer gate fails -> STOP, never zero/blank replacement.
- **Consistent wrong latent orientation:** either video class has non-positive transfer lower bound -> STOP; no absolute verdict filter.
- **Epoch-bank drift:** frozen threshold fails -> common half-epoch refresh or STOP; no EMA/teacher key.
- **Numerical amplification:** low scale/norm, non-finite gradient or extreme CTE/base ratio -> STOP under frozen floor/clip; do not claim bounded gradients.
- **Generic multiview/extra optimization explains gain:** clean fails against teacher-independent multiview/random/label-only -> MLLM claim fails.
- **Only native head changes:** final kNN gate fails -> STOP.
- **Missingness is label-correlated:** report class-conditional coverage/confidence/fallback and preserve exact missingness in record-level controls; never impute from gold.

## Novelty and Elegance

TextTeacher covers train-time semantic teachers, EmbedDistill/geometric KD cover retrieval geometry distillation, CGO covers harmful-video modality intervention/gradient control, and RAMF covers counter-reasoning fusion. CTE's narrow delta is: a label-blind, confidence-bearing whole-modality withholding relation is empirically transferred to a supported prototype tangent of the **true-class exact epoch-refreshed full-bank margin**, under one shared query/key encoder and complete teacher removal at inference. It does not claim causal counterfactual identification.

The proposal has one mechanism and zero new trainable components. Support, class-transfer and controls are necessary falsification, not parallel contributions.

## Claim-Driven Validation Sketch

### Claim 1: the supported tangent/loss action family can move enough full-video geometry

- **Experiment:** A0 nested train OOF on MHC-EN/MHC-ZH.
- **Controls:** exact REMOVE, teacher-independent multiview, matched random targets.
- **Metric/gate:** support, full-video OOF acc/mF1 +.050, corrections and neighbour churn.

### Claim 2: teacher relations transfer beyond label/cheap controls without hidden orientation

- **Experiment:** A1 <=128 train videos/dataset.
- **Controls:** label-only, energy, multiview, random, feasible SHUFFLE, NOISE.
- **Metric/gate:** class-specific two-radius ordinal-transfer lower bounds plus exact held-out margin/wrong-neighbour update.

### Claim 3: CTE causes substantial final-kNN gain

- **Experiment:** A2 seed-0 dev, then paired seeds 0/1/2 final.
- **Controls:** all critical arms through the same loss and budget.
- **Metric/gate:** seed-0 +.010 over each control, then +.030 acc/mF1 on both datasets with all seed signs and bootstrap/Holm/removability.

## Experiment Handoff and Cost

- **Must prove:** action capacity, class-conditional relation transfer, assignment-specific MLLM value, final causal gain.
- **Highest risks:** teacher withholding does not transfer to the supported path; one label class reverses orientation; label-only CTE raises the moving bar; full final +3 remains unreachable.
- **Compute:** A0 ten outer-fold runs plus finite nested controls, estimated 20–40 GPU-hours pending a SLURM microbenchmark; A1 <=2048 calls and small fixed pilot updates; A2/final only after gates. All computation must use SLURM and the `HateVideo` environment.
- **Gold annotation cost:** zero. Teacher records are weak pseudo-relations, not annotations.
