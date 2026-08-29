# THVL external-confirmation frozen protocol

Status: protocol design frozen before THVL test-label access. This is an
external confirmation of the V20 dual-context judge without V8 experts, not a
new THVL-specific method.

## Task freeze

The primary task is a binary temporal task defined from the dataset's released
taxonomy: frames annotated as **Hate**, **Bias**, or **Verbal Abuse** are
positive; frames annotated as the taxonomy's explicit non-harm/background class
are negative. Every other harm category is excluded from the primary task, not
silently relabeled negative. If the official taxonomy does not contain an
unambiguous background class or permits overlapping excluded harms, this task
is blocked until an ID-level inclusion mask is derived mechanically from the
taxonomy documentation.

An optional secondary “all-harm versus background” task may be reported only if
the official taxonomy explicitly enumerates all harm leaves and background.
It must be named secondary and cannot replace the primary task after seeing
validation performance. The mapping artifact, class spellings and hierarchy
hash are frozen before any judge forward pass. Class names are never inserted
into the judge prompt.

## Inputs and minimum eligibility

Only official train/validation/test ID manifests, video/audio media and
timestamped ASR are required. No V8 expert scores, OCR, target metadata or
dataset-specific keywords are used.

- Video is required only to establish a trusted duration/timeline. The V20
  judge itself consumes ASR text.
- ASR rows require `video_id`, finite `start`, finite `end`, `end>start`, text,
  ASR model/version and source-media hash. Chunks are clipped to the verified
  media duration; clipping and dropped rows are logged.
- At least 90% of scoped validation videos and 90% of their media duration must
  be covered by valid ASR. Missing ASR is not imputed with labels.
- Split IDs must be disjoint and hashed. Test labels are stored outside the
  forward/selection environment and are not loaded until the validation gate
  passes and the frozen test bundle is produced.

## Frozen judge and score

The policy is dataset-agnostic: “Judge whether the supplied local video speech
contains hateful or severely abusive content. Use only the supplied speech and
answer Yes or No.” It contains no taxonomy label list, examples, lexicon or
corpus-specific keywords. The score is the next-token `logsumexp(Yes IDs) -
logsumexp(No IDs)` margin from the fixed local Qwen3-VL checkpoint operating in
text-only mode.

For each video the same ASR chunks are packed into two frozen arms:

1. **Global arm:** full causal attention and continuous positions. Its video
   value `G_v` is the duration-weighted mean of causal-continuous chunk margins.
2. **Local arm:** branch-isolated attention and branch-reset positions. The
   chunk margins are projected to the 1-fps timeline by overlap-weighted mean,
   then centered exactly within video:
   `L_v(t)=R_v(t)-mean_t R_v(t)`.

Reference scaling is fit on validation scores only: divide `G` by its robust
validation scale and `L` by its validation RMS, with fail-closed zero-scale
handling. The frame score is

`S_v(t) = alpha * G_v + beta * L_v(t)`.

There is no claimed identity fallback because THVL has no V8 expert backbone.
The mandatory validation candidates are global-only `(1,0)`, local-only
`(0,1)`, equal `(1,1)` and a small nonnegative grid. Global-only is the safety
baseline. Negative weights are forbidden.

## Validation-only selection

Raw global/local margins, prompt hashes and the forward manifest are frozen
before opening validation temporal labels. Validation may select only:

- `alpha,beta` from `{0, .25, .5, 1, 2}² \ {(0,0)}`;
- one frame threshold from validation score quantiles
  `{.50,.60,.70,.80,.90,.95}` for segment/F1 reporting;
- deterministic removal of predicted runs shorter than one second (off/on).

The tuple maximizes validation frame AP, breaking ties by ROC, within-video
macro AP, smaller `|alpha|+|beta|`, then lexicographic tuple. Threshold and run
filter never affect AP/ROC weight selection. After weights are frozen, choose
threshold/run-filter by validation frame F1, breaking ties by precision, then
the higher threshold and finally no run filter. No prompt, ASR window or
taxonomy mapping is tuned.

## Metrics

Primary threshold-free metrics are pooled 1-fps frame AP and ROC. Localization
evidence additionally requires within-video macro AP/ROC over videos containing
both positive and negative scoped frames, with eligible-video count, and
hateful-video-only centered pooled AP/ROC. Report video-cluster bootstrap 95%
CIs for full minus global-only (`B>=2000`) and within-video time-shuffle controls
(`B>=1000`).

Thresholded secondary metrics are frame F1 and precision/recall, segment IoU
AP at `.3/.5/.7`, and mean boundary error, using only the validation-frozen
threshold/run filter. These are omitted if the released annotation does not
define exhaustive temporal background.

## Kill gate and test discipline

Do not open THVL test labels unless all validation conditions hold:

1. split/provenance/ASR audits pass and ASR video/time coverage are each >=90%;
2. at least 20 validation videos are eligible for within-video macro metrics;
3. packed masked-reset logits match sequential references on the frozen audit
   subset (`max_abs_error<=1e-4`, Spearman `>=.999`), otherwise use sequential
   inference everywhere;
4. selected full score improves frame AP over global-only by at least `.01`,
   does not reduce frame ROC by more than `.005`, and its paired video-bootstrap
   AP lower confidence bound is nonnegative;
5. within-video macro ROC is at least `.55` and exceeds the 97.5th percentile
   of the time-shuffle control; within-video macro AP must not be below
   global-only.

Failure means stop and archive a negative external-validation result. It does
not authorize changing the task, prompt, grid or cohort. If the gate passes,
freeze config/scales/weights/threshold first, run test forward without labels,
hash it, and only then allow a separate evaluator to open test annotations once.
