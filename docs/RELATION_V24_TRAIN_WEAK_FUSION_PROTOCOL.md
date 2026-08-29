# V24 train-weak fusion protocol

V24 is a fixed small calibrator over three frozen inputs per video: isolated
text-window scores, isolated frame-plus-text window scores, and one global
causal text score. It never consumes segment boundaries, frame labels, temporal
GT, target categories or validation/test video labels during fitting. Training
uses the same corpus's frozen train IDs and binary video labels only.

The producer interface is an exact allowlist, not a blacklist. A labeled row
contains only `corpus, split, video_id, video_label, global_causal_score,
families, source_hashes`; an inference row omits only `video_label`. Therefore
annotations, intervals, timestamps and frame GT cannot enter under another
name. Both modality families must be present, finite, nonempty and aligned to
the same window count. The run protocol binds SHA256 values for the bag file,
exact ID manifest, producer implementation, corpus string, and canonical V23
global-values file. Every bag global is compared elementwise with that frozen
source before epoch-0 fallback is accepted.

Train evidence is produced label-blind from per-video atomic media-QC and ASR
records. It uses deterministic nonoverlapping 30-second full-timeline windows,
retains the final short window, and emits text-isolated, center-frame-plus-text
isolated margins. Empty speech is `[NO SPEECH]`; local speech keeps the first
3000 codepoints. Global evidence is not redefined on these windows: it is the
arithmetic mean of the frozen V16 `causal_continuous` ASR-chunk margins, with
the exact V16 prompt, packed token reconstruction, causal mask and continuous
positions. Each video is committed atomically and resume validates
the input-item hash, full window count and finite outputs. Only after the entire
evidence manifest is frozen may `steward_join.py` read the train weak-label
manifest and create exact-schema bags. GPU forward records never contain
labels.

The execution order is mandatory: first run and freeze exact V16 packed raw for
the target split; second prepare and freeze the V24 config/windows while binding
that V16 chain; third run V24 local text/MM forward. Qwen processor and model
are loaded with the pinned commit revision and `local_files_only=True`. Before
model loading, runtime re-hashes the current producer, V16/V23 implementations
and prompt spec, and verifies that the resolved local snapshot directory is the
pinned commit.

Every local expert is centered exactly within each video before fusion. Exact
duplicate experts are collapsed within a modality family, experts are averaged
inside family, and a learned softmax assigns one weight per family. Thus adding
copies cannot increase a family's vote. The local bag statistic is
`tau*log(mean(exp(local/tau)))`, with fixed `tau=1`; repeating an entire bag any
number of times is exactly invariant, so long videos do not win by cardinality.

The video training logit is

`(1+delta_g) * global + bias_g + gamma * LME(centered local mixture)`.

The frame output replaces LME by the centered local mixture. Parameters are two
family logits, `delta_g`, `bias_g`, and nonnegative `gamma`. Epoch 0 initializes
`delta_g=bias_g=gamma=0`, making every frame and video logit exactly equal to
the V23 global input. It is always a checkpoint candidate. One run is allowed:
seed 234, Adam learning rate `.01`, five epochs, no sweep. Validation may choose
among epochs 0..5 by video AP then video ROC, with epoch-0 tie preference; test
is evaluation only.

Three matched five-epoch arms are always trained: real local, permuted-local
negative control, and global-calibration-only. A validation selector evaluates
all epochs of all arms and writes a frozen config. Label-blind inference refuses
test at this stage. Passing video-level gates produces only
`VIDEO_VAL_PASS_PENDING_TEMPORAL`. The selector also records effective
`gamma=max(raw_gamma,0)` and family softmax weights, requiring `gamma>=.01` and
maximum family weight `<=.95`.

A separate steward must then verify the within-video AP/ROC and shuffle gates
and sign a config-hash-bound record with HMAC-SHA256. Only the finalizer can turn
that record into `FINAL_PASS`; test inference accepts no other status. Failed
video gates retain epoch-0 fallback for diagnosis but do not authorize test.
Predictions are hashed before the separate evaluator can open labels.

## Negative control

Train an identical diagnostic model after deterministically permuting whole
local bags across train video IDs while leaving global scores and labels fixed.
Permutation seed is 24024 and preserves the multiset of bag lengths by bucketing
length to the nearest power of two. It is report-only and never supplies the
deployed weights. This tests whether local families add label-aligned bag
information beyond capacity/global calibration; it does not claim that video
labels identify temporal ordering.

Permutation occurs on **training bags only**. Real, permuted-trained and
global-only arms are all evaluated on the exact same original frozen validation
bags. Permuting validation would change the evaluation distribution and is
forbidden.

## Kill criteria

Stop and retain exact V23 global fallback if any condition holds:

- epoch-0 fallback is not bit-exact, any local video mean exceeds `1e-10`, or
  whole-bag replication changes the bag logit by more than `1e-12`;
- train producer contains non-train IDs, temporal-label keys, incomplete frozen
  train coverage, non-finite scores, or cross-corpus provenance;
- selected real-local model fails validation video AP non-inferiority versus
  epoch 0 (tolerance `.002`) or ROC drops more than `.005`;
- real-local validation AP gain over a global-calibration-only matched model is
  below `.005`, or is not at least `.005` above the permuted-bag negative
  control;
- `gamma<.01`, either family weight exceeds `.95`, or seed-fixed training is not
  reproducible;
- localization is later evaluated and within-video macro AP/ROC fails to beat
  V23 global by at least `.01` AP and `.02` ROC, or fails a within-video shuffle
  control. Temporal validation labels are permitted only for this post-training
  kill decision, never for fitting.

These gates deliberately allow a negative conclusion: weak video labels may
calibrate video risk while providing no trustworthy new temporal ordering.
