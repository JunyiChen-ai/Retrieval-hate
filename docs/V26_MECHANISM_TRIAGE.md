# V26 mechanism triage: counterfactual temporal witnesses

Status: design-only, after the formal V24/V25 negative result. No test split,
temporal ground truth, or new trial was read or run for this note.

## Decision

The single priority is **Counterfactual Temporal Witnesses (CTW)**: learn a
video-level weak-label classifier on synchronized 1 Hz multimodal tokens, learn
what each token should look like from its surrounding context using negative
training videos, and localize by the *causal deletion effect* of replacing one
observed token with that context-predicted background token.

This changes the representation and the definition of a local prediction. It is
not another choice of noisy-OR, top-k, temperature, or MIL calibration.

## What the negative result identifies

V24 reduced a 30-second interval to two Qwen margins: isolated ASR and one
center-frame-plus-ASR judgment. The mean THVL train bag has only 12.69 such
instances. Text and multimodal margins are highly redundant (pooled Pearson
0.827), and the strongest useful signal is cross-video level, not within-video
ordering. Centering destroys that level signal. V25 preserves the level through
a negative ECDF, but it still attempts to recover boundaries from the same two
coarse scalar margins. Its formal validation result falls back to epoch 0; the
post-access diagnostic within-video ROC is 0.56678 and its paired-CI gate fails.

The failure therefore does not support a sixth pooling rule. A 30-second scalar
cannot distinguish which second, modality interaction, or discourse transition
made the interval harmful. The next experiment must preserve synchronized
high-dimensional evidence before weak supervision is applied.

## Available label-blind evidence

All four in-project corpora already have aligned frozen features:

| feature | HateMM | MHC-EN | MHC-ZH | HCS | nominal grid |
|---|---:|---:|---:|---:|---|
| CLIP-B/16 visual, 512-D | 1069 | 793 | 815 | 395 | 1 Hz |
| VGGish audio, 128-D | 1069 | 793 | 815 | 395 | 1 s |
| legacy BERT CLS store, 768-D | 1069 | 793 | 815 | 394 | inventory only; V26 regenerates |
| ImageNet ViT-B/16, 768-D | 1069 | 793 | 815 | 394 | 1 Hz |
| I3D RGB, 5x1024-D | 2137 | 1585 | 1629 | 787 | denser, reducible to 1 Hz |

The first three form the frozen common interface. ViT/I3D are ablations, not
extra selectable inputs. The one missing HCS text/ViT record must be an explicit
availability mask, never silent imputation.

THVL has 314 train videos with atomic media QC and timestamped Whisper chunks,
plus the 32-video validation media/ASR cohort. Its present 30-second windows can
be replaced label-blindly by: CLIP center frame at each integer second, VGGish
one-second audio, and a newly generated pinned BERT CLS embedding of every Whisper chunk repeated over
the seconds it overlaps. Empty speech/audio and missing modalities have explicit
masks. This uses local models and existing media only; MLLM/API cost is zero.

## Frozen CTW mechanism

For second `t`, concatenate separately normalized projections of frozen visual,
audio and text features and their availability bits:

`x_t = concat(P_v LN(v_t), P_a LN(a_t), P_l LN(l_t), mask_t)`.

All projections have width 128. A shared two-layer temporal encoder uses fixed
dilations `{1, 4, 16}` seconds. Its context branch is **center-masked**: when it
constructs `c_t`, no representation from `[t-1,t+1]` may enter. A background
decoder gives `b_t=D(c_t,mask_t)`. On negative training videos it minimizes
modality-balanced Huber reconstruction of `x_t`; losses are averaged within
modality, then time, then video, so duration and dimensionality do not reweight
the corpus.

A video classifier `F(X)` is trained from that corpus's binary train video labels.
It sees the full 1 Hz sequence with fixed masked mean/attention pooling. For a
candidate second define the intervention

`X^(t) = X with x_t replaced by stopgrad(b_t)`

and the label-independent localization logit

`d_t = F(X) - F(X^(t))`.

The conceptual frame score is derived from `d_t`; the executable addendum below
fixes its clipping and mapping. At inference there is no responsibility variable,
video label, temporal GT, or dataset-specific vocabulary. The executable protocol
uses exact batched **full-T** deletion in training and inference; no position
sampling remains.

The fixed loss is

`L = BCE(F(X), y) + 1.0 * 1[y=0] mean_t Huber(d_t, 0)
                 + 0.25 * 1[y=1] softplus(-LME_tau(d_t))
                 + 0.10 * L_reconstruct_negative`,

with `tau=1`, uniform four-position sampling seed `26026+epoch`, and no temporal
smoothing. The positive term only ensures that some observed evidence is
necessary for a positive decision; it does not define the output. The output is
always the intervention effect above. Validation selects only epoch `0..8`, with
epoch 0 equal to the frozen video prior repeated at 1 Hz. No coefficient, dilation,
feature family, or sampling count is swept.

## Why the local branch can become load-bearing

V24/V25 ask a scalar local judge to recognize hate in isolation. CTW asks the
video classifier a counterfactual question: *would the same video be judged less
hateful if this synchronized moment were replaced by what its context predicts
as background?* It retains absolute semantics, cross-modal conjunctions, and
the contrast with surrounding discourse. The negative-only decoder makes the
replacement identifiable without temporal labels; the weak positive label
orients which departures matter for hate rather than generic novelty.

There are two direct load-bearing tests independent of temporal GT:

1. deleting the top-20% CTW seconds must reduce positive-video logits more than
   deleting matched random seconds;
2. inserting those observed seconds into their background-replaced sequence must
   restore the logit more than inserting shuffled seconds.

If these fail, localization cannot be rescued by an evaluation metric.

## Difference from VADCLIP and standard WS-VAD

VADCLIP and conventional WS-VAD learn per-snippet anomaly/class scores and pool
those scores to satisfy a bag label. Their local score exists before, and is
optimized through, the MIL aggregator; CLIP prompts or normal/abnormal prototypes
provide the instance semantics. CTW has no hate prompt, anomaly prototype, or
direct snippet classifier. It first learns a relational video decision and a
negative-background conditional model; localization is the counterfactual
necessity of an observed synchronized token for that decision. The scientific
claim is therefore **weak localization by conditional temporal intervention**,
not a better top-k loss.

The novelty boundary is precise: masked contextual replacement alone is not new,
nor is deletion attribution. The proposed contribution is their weakly supervised
composition in which (i) same-corpus negative videos identify the counterfactual
background, (ii) positive video labels orient a relational multimodal classifier,
and (iii) exact token replacement defines a dense localization output with
faithfulness controls. Literature novelty still requires a focused search before
paper claims are made.

## One THVL kill pilot

Run only THVL, seed 234, epochs 0--8, no test access and no hyperparameter sweep.
Feature extraction is label-blind. Training uses the 314 train video labels;
epoch selection uses validation video AP then ROC with epoch-0 tie preference.
Only after raw predictions and the epoch are frozen may the existing steward
evaluate validation temporal aggregates.

Required controls are fixed before training:

- epoch-0 constant V16 prior;
- CTW real;
- matched CTW with positive-train local sequences permuted within exact duration
  buckets while global video/label identities stay fixed;
- CTW with `b_t` replaced by the corpus-wide negative mean, testing whether
  conditional context is load-bearing.

The pilot survives only if every gate passes:

1. validation video AP improves over epoch 0 by at least 0.005 with paired
   bootstrap lower CI above zero; ROC is no worse by more than 0.005;
2. real AP exceeds permuted and negative-mean controls by at least 0.005;
3. on positive validation videos, top-20% deletion exceeds matched random
   deletion by at least 0.02 logit with video-bootstrap lower CI above zero;
4. mixed-video within macro ROC is at least `0.5868` (two absolute points above
   the known V25 diagnostic), and within AP improves over the constant prior by
   at least 0.01;
5. within-video time shuffle removes at least 80% of the ROC gain above 0.5;
6. 1 Hz coverage is 100%, scores are finite, and at least 95% of videos have
   nonzero score variance.

Failure of any gate stops CTW. Passing freezes the exact mechanism and then runs
the same code independently on HateMM, MHC-EN, MHC-ZH and HCS, each with its own
train reference, training, validation selection and test evaluation. No corpus
mixing or corpus-specific semantic rule is permitted.

## Main risks

- Hate can be diffuse; deleting one second may have small effects. The fixed
  top-20% faithfulness test detects this rather than adding a longer-window sweep.
- A negative-context decoder may model editing novelty rather than harmfulness.
  The negative-mean control and positive/permuted arm separate conditional
  background quality from generic model capacity.
- Repeated BERT CLS chunk embeddings limit sub-sentence boundaries. CTW can still
  use 1 Hz audio/visual changes, but cannot claim word-level localization.
- Exact deletion costs `T` classifier forwards per video, though these are small
  temporal networks over cached features, not MLLM calls.

---

# Executable protocol addendum

This addendum supersedes any underspecified architecture, sampled-intervention,
loss, control, or gate wording above. It leaves no model or objective choice for
the THVL pilot. It was written after the V25 validation aggregate was known and
is therefore **V25-VAL-INFORMED**. No THVL test or new temporal GT was accessed.

## 1. Exact input contract

The producer emits one record per video with the exact keys
`corpus, split, opaque_id, duration, G, G_domain, seconds, source_hashes`.
`seconds` has `T=ceil(duration)` ordered rows with exact keys
`second, visual, audio, text, availability`. `second=j`; availability is three
binary values. An available vector must be finite and have dimension
`visual=512, audio=128, text=768`; an unavailable vector is the empty list and
must have mask zero. `T_eff=sum_j 1[any availability_j]` must equal `T`; otherwise
the video fails rather than being padded into coverage.

For THVL, visual is CLIP-B/16 at `min(j+0.5,duration-1e-6)`, audio is VGGish on
`[j,min(j+1,duration))`, and text is the newly regenerated pinned BERT CLS embedding of all
Whisper chunks having positive overlap with that interval, duration-weighted
when several overlap. No overlapping speech gives unavailable text, not a zero
embedding. The final short second is retained. Model/revision, media, ASR,
per-video record and aggregate manifest hashes are mandatory.

The same schema is used for HateMM, MHC-EN, MHC-ZH and HCS. Feature normalization
is fit independently per corpus using its train split: one mean and standard
deviation per feature coordinate over available seconds, with standard deviation
floored at `1e-6`. It is frozen for val/test. Missing values do not enter moments.
The one known HCS missing-text video remains in the cohort with text mask zero at
all seconds. Dropping it, zero-imputing it as available, or using a corpus-specific
fallback is forbidden. Every corpus manifest binds train/val/test ID manifests,
feature producer and revision, normalization bytes, dimensions, grid rule,
availability counts and root hash; schemas are byte-identical apart from corpus,
IDs and hashes.

`G` is a frozen video-prior **signed logit**. THVL uses the exact arithmetic mean
of V16 `causal_continuous` Yes-minus-No margins; no sigmoid is applied. A future
corpus may use its preregistered frozen prior only after mapping a probability
once as `logit(clamp(p,1e-6,1-1e-6))`; its manifest fixes source domain and mapping
before training. All arms share the identical `G` bytes.

## 2. Independent negative background model

The background decoder `D` shares no parameter, optimizer state, projection or
gradient with the video classifier `F`. It is trained only on negative training
videos, then frozen before any `F` run.

For target second `t`, `D` receives, for each modality, only normalized features
and masks at fixed offsets

`O={-16,-8,-4,-2,+2,+4,+8,+16}`.

Out-of-range offsets are unavailable. Thus raw tokens at `t-1,t,t+1` never enter
the input graph. Each modality has its own network: available neighbor vectors
are projected to 128 dimensions, left and right masked means are concatenated
with their availability counts, then passed through `Linear(258,256)-GELU-
Linear(256,d_f)`. `D_f` predicts only modality `f`; there is no cross-modal path.

The decoder objective is Huber loss with delta 1, averaged coordinates, then
available predicted modalities, then seconds, then videos. A target modality with
mask zero has no reconstruction loss and no prediction used downstream. AdamW,
learning rate `1e-3`, weight decay `1e-4`, batch 16 videos, 20 epochs, seed 26026;
the final epoch is used without selection.

Five-fold video cross-fitting is fixed by
`fold(id)=int(SHA256(id)[:8],16) mod 5`. Fold `k` videos receive predictions only
from `D^-k`, trained on negative videos outside fold `k`. Every fold must contain
at least one negative target for every modality. A full `D` trained on all train
negatives is materialized once for val/test. The manifest binds all fold IDs,
negative IDs, states, predictions and the full state. Each train target receives
exactly one OOF background prediction.

Availability is preserved under intervention: if modality `f` is unavailable at
`t`, it is neither predicted, replaced nor scored. If available, its replacement
is `stopgrad(D_f(context_t))`. Before training, a bit-exact zero-influence test
replaces raw features at `[t-1,t,t+1]` by two different finite byte patterns and
requires `D(context_t)` to be bit-identical. Autograd must also report exactly
zero gradient from `D(context_t)` to those three input tokens. Failure is fatal.

## 3. Exact relational classifier and dense effect

Each available modality is normalized and passed through an independent
`Linear(d_f,128)-GELU`; unavailable projections are exact zero and their mask is
concatenated. A fixed `Linear(387,256)` produces tokens. `F` is two pre-norm
Transformer encoder layers, width 256, four heads, feed-forward width 512,
dropout **0**, with fixed sinusoidal positions. Masked mean pooling over the `T_eff`
tokens followed by `Linear(256,1)` gives residual `R_theta(X)`. The last linear
layer is initialized to exact zeros, so epoch 0 has

`F_theta(X,G)=G+R_theta(X)` in every arm and epoch, and at initialization
`F_theta0(X,G)=G` exactly. Training, validation and inference use deterministic
algorithms, deterministic attention kernels and fixed seeds; dropout and stochastic
augmentation are forbidden. Repeating a forward on identical bytes must give a
bit-identical logit and effect tensor.

For every available-any second, construct `X_cf_t` by replacing all and only its
available modalities with the frozen OOF/full-`D` predictions. Re-run the entire
classifier and define

`e_t = T_eff * (F_theta(X)-F_theta(X_cf_t))`.

All `T_eff` interventions are computed in every train step; there is no position
sampling. Effects are clipped once to `[-12,12]` before auxiliary losses and
frame mapping. The reported CTW local score is `sigmoid(clip(e_t,-12,12))`.
Clipping is not applied to `F` or video BCE.

Epoch 0 has `e_t=0` and hence CTW local score 0.5. This is an initialization
diagnostic only. The legal failure output is instead the frozen original-domain
constant `G` at every second, bit-for-bit; it is never called CTW and is never
passed through sigmoid.

Let `A(e)` be the single fractional top-20% log-mean-exp from the V25 definition,
with `rho=.20,tau=1`. The per-video full-T objective is

`BCEWithLogits(F(X),y)
 + 1[y=0] mean_t Huber(clip(e_t),0; delta=1)
+ .25*1[y=1] softplus(-A(clip(e)))`,

and these are all optimized terms and coefficients. Frozen-`D` reconstruction is
logged as a monitor but is not part of this objective. Losses are averaged per
video, then batch. `F` uses AdamW,
learning rate `3e-4`, weight decay `1e-4`, batch four videos, gradient norm 1,
epochs 0--8, seed 234. No scheduler, smoothing, augmentation or early stopping.
Validation selects `(AP,ROC,-epoch)` on video labels and shares the selected epoch
across arms.

## 4. Four matched arms

1. **Fallback/global-only:** no trainable `F`; outputs frozen `G` and defines the
   epoch-0 video baseline. It is the degenerate shared form `F(X,G)=G+R(X)` with
   `R(X)=0` identically.
2. **Real CTW:** the specification above.
3. **Permuted CTW:** before `F` training, whole raw feature sequences **and local
   availability masks and their already-frozen own-fold OOF background tensors
   are reassigned together among train videos in exact `T` buckets by
   sorted-ID cyclic shift `1+seed mod (n-1)`. `G`, target ID and `y` do not move.
   Recipient `r` therefore consumes `(X_donor,mask_donor,b_OOF_donor)` and never
   queries a decoder associated with recipient `r`. Singleton buckets are
   self-mapped and counted. Decoder states and the negative reference remain those
   built from real, unpermuted negatives; they are never refit to permuted labels
   or sequences. The permutation manifest records recipient, donor, donor fold,
   raw-sequence hash, mask hash, OOF-background hash, `T`, nonself flag and
   intervention coverage for every row, plus pre/post aggregate hashes. It checks
   same `T`, exact one-to-one donor coverage and at least 80% nonself videos and
   instances.
4. **Negative-mean replacement:** identical `F` architecture, initialization,
   optimizer, full-T effects, steps and seed as real. The only change is `b_t`:
   each available modality uses its frozen train-negative normalized coordinate
   mean (therefore exact zero after normalization), with the same real mask.

Before any permutation, every real train video has its complete own-fold OOF
background tensor precomputed and frozen; lazy recomputation inside an arm is
forbidden. Validation always uses the one full-negative-reference `D`, never an
OOF or recipient-specific decoder. All trained arms see the original, unmodified validation records. Checkpoint and
optimizer-step counts must match exactly. Real and negative-mean arms must have
100% intervention coverage; the permutation arm must preserve every per-video
`T`, total instance count and modality availability-count multiset. Any mismatch
fails preflight.

## 5. Independent faithfulness probe

The primary faithfulness claim does not reuse `F`. Probe `H` has the same token
projection, Transformer and pooling architecture as `F`, but no `G` residual,
no CTW effect loss, no shared weights, and no gradient path to CTW. It is trained
once on the same train video labels using video BCE only, seed 26027, AdamW
`3e-4`, batch four, eight epochs; the final epoch is frozen without validation
selection.

For each positive validation video, set `k=max(1,ceil(.20*T_eff))`. Select the
single contiguous length-`k` interval maximizing the sum of frozen CTW effects.
Replace its available tokens by full-`D` backgrounds and measure
`drop_top=H(X)-H(X_deleted)`. Generate exactly 100 matched random contiguous
length-`k` intervals per video using seed `26028+SHA256(id)[:8]`; their mean drop
is `drop_random`. The primary statistic is the video mean of
`drop_top-drop_random`, using the canonical positive-video bootstrap arrays
defined below (2,000 replicates, seed 26032).

Using `F` for the same deletion is reported only as a circular sanity check and
cannot pass the faithfulness gate. `H` receives no temporal GT, CTW score, selected
interval or validation result during training.

## 6. Validation gates and confirmatory sequence

Video AP and ROC consume the finite signed logits `F(X,G)` directly; sigmoid,
thresholding and frame pooling are forbidden for video ranking. Legal fallback
frames are the original signed `G` bytes repeated over time without sigmoid.

The **real arm alone** selects epoch by lexicographically maximizing
`(mean_seed video AP, mean_seed video ROC, -epoch)`. Pilot seed 234 applies the
same rule with its one seed. Controls are evaluated passively at that exact epoch
and never choose their own. All temporal gates use the existing encrypted steward and 2,000 video-cluster
bootstrap replicates. The V25 result is already known, so comparisons to its
within ROC are explicitly V25-VAL-INFORMED, not blind confirmation. CTW passes
the one-seed THVL kill pilot only if all conditions hold:

1. video AP minus frozen `G` is at least .005 and its paired-bootstrap lower 95%
   bound is above zero; video ROC is no worse by .005;
2. video AP exceeds both permuted and negative-mean arms by .005, with paired
   lower bounds above zero;
3. independent-`H` contiguous top-20% deletion has mean advantage at least .02
   logit over random and bootstrap lower bound above zero;
4. mixed-video within macro ROC has bootstrap lower bound above .5, point estimate
   at least .5868, and paired CTW-minus-V25 ROC is at least .02 with lower bound
   above zero; within AP exceeds the constant-`G` result by .01 with lower bound
   above zero;
5. run exactly 100 within-video effect permutations per model seed, with
   `shuffle_seed=26030000+1000*r+model_seed`, `r=0..99`. The 97.5th percentile
   shuffled ROC gain above .5 must be no more
   `0.20*(real ROC-.5)`, and the paired real-minus-shuffle bootstrap lower bound
   must be above zero;
6. Spearman correlation between duration and each of video mean/max CTW score has
   absolute value at most .20 and may exceed the corresponding V25 absolute
   correlation by at most .05; 1 Hz coverage is 100%, all scores are finite, and
   at least 95% of videos have nonzero within-video variance.

No test is opened on failure. Passing seed 234 authorizes a confirmatory rerun of
the frozen protocol at seeds 2025 and 3407. The shared validation epoch is then
selected from the three-seed mean `(AP,ROC,-epoch)`; every gate is recomputed with
matched seeds and must pass again. Only that three-seed confirmation can authorize
one frozen test evaluation or independent per-corpus extension.

### Exact uncertainty implementation

There are exactly three distinct cohort-specific canonical bootstrap-array sets,
all with `B=2000`:

1. **all-32 cohort**, RNG seed 26031: each array samples 32 indices with
   replacement from the complete validation cohort. This is the only cohort for
   which a one-class draw is invalid. Redraw from the same RNG stream until both
   video classes occur, at most 100 attempts per replicate; exceeding the limit
   fails evaluation rather than dropping a replicate.
2. **positive-video cohort**, RNG seed 26032: each array samples `N_pos` indices
   with replacement from the frozen positive-video ID list. There is no class
   redraw. These arrays are used for independent-`H` deletion faithfulness and
   every other positive-only statistic.
3. **mixed-video cohort**, RNG seed 26033: eligibility is frozen as videos whose
   1 Hz target contains both classes; each array samples `N_mixed` eligible-video
   indices with replacement. There is no class redraw. These arrays are used for
   within-video macro AP/ROC, CTW-minus-V25 and real-minus-shuffle inference.

Within each cohort, its canonical arrays are reused identically across real,
fallback, both controls, V25 and every matched model seed. Arrays are never reused
across cohorts. Seed-specific metrics are computed first and then arithmetically
averaged. Differences are taken only after applying the same cohort-specific
indices. Confidence bounds are
`numpy.quantile(values,[.025,.975],method='linear')`; every gate saying “lower
bound above” uses strict `lower > threshold`.

For shuffle quantiles, compute within-video ROC for each `r` and model seed, then
average across model seeds for that `r`; apply NumPy linear quantile at .975 to
those 100 values. For the paired shuffle CI, first compute for every eligible
video and model seed its real within ROC minus its mean within ROC over all 100
shuffles. Bootstrap these per-video paired contrasts with the same 2,000 canonical
video-index vectors, then average seeds. A video without both frame classes is
ineligible for both real and all shuffled arms and its exclusion count is fixed
before scores are opened.

## 7. Frozen producer identities for later corpora

No result-dependent language/model switch is allowed. Before any later-corpus
forward, its producer manifest must preregister this exact stack:

- visual: `openai/clip-vit-base-patch16`, Hugging Face snapshot commit and every
  model/config file SHA recorded; `CLIPImageProcessor` shortest-edge bicubic 224,
  center crop 224, released rescale/mean/std; the only readout is
  `CLIPModel.get_image_features`, the projected 512-D float32 embedding, with no
  L2 normalization. The frozen train split later supplies the coordinate-wise
  z-normalization defined above;
- audio: `torchvggish==0.2`, released AudioSet weight-file SHA recorded,
  `postprocess=False`, 16 kHz mono, 64 mel bins 125--7500 Hz, 25 ms window,
  10 ms hop, 0.96 s patch starting at each integer second with zero tail padding;
- text: **label-free regeneration is mandatory**; no legacy sentence-BERT array
  may be reused. THVL, HateMM, MHC-EN and HCS use one pinned
  `bert-base-uncased`; MHC-ZH uses one pinned `bert-base-chinese`. The exact local
  snapshot commit/config/weight/tokenizer hashes are frozen, `max_length=64`,
  released tokenizer preprocessing, and the only readout is last-hidden-state CLS
  float32 (768-D), without pooler or L2 normalization. Whisper chunk
  producer/model/revision and overlap-assignment hashes are also mandatory.

The Chinese checkpoint assignment is a predeclared corpus-language fact, not a
validation option. A producer may be replaced only by a written pre-forward
migration proving old artifacts unusable; no alternate encoder, multilingual
switch, ASR model, preprocessing or language routing may be selected after any
validation result. Existing feature stores lacking snapshot/weight provenance
must be wrapped by a hash census before use or regenerated; filename/model-name
claims alone are insufficient.
