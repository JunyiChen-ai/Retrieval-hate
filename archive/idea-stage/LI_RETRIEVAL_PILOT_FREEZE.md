# Late-interaction segment-level retrieval pilot — FROZEN before any candidate number was computed

Written 2026-08-09, **before** `idea-stage/li_retrieval_pilot.py` was executed on real labels.
Arm definitions, key construction, similarity, endpoints, splits and the decision rule below are
fixed at write time and are **not** edited after results appear. This is a retrieval-mechanism
pilot, not a registered verdict.

## Question

Both prior pilots point at mean pooling as the culprit, not at the segment features:

- `idea-stage/P2_FORENSIC_MEMO.md` H4: segment kNN label purity lift **+0.181** beats whole-video
  **+0.138**; class separation survives pooling but the *segment-specificity* does not.
- `idea-stage/OCR_FUSION_PILOT_RESULT.md`: mean-pooled whole-video OCR fusion buys only **+0.0094**
  and saturates at 3 windows, while the I5 gate says the OCR neighbourhood is genuinely
  **complementary** to the transcript neighbourhood (`ov@10 = 0.048` vs chance 0.017).

**Does replacing whole-video mean-pooled retrieval keys with 30 retained segment keys scored by a
MaxSim-style late interaction improve (a) neighbour label purity and (b) kNN classification
macro-F1 over the current whole-video key?** And how does any gain split between "retaining
multiple segments" and "putting OCR into the segment key"?

## What this pilot is explicitly NOT (dead ends, excluded by design)

Three routes are already killed by evidence and none of them appears here:

1. **Selecting a single/few segments by visual purity, then retrieving.** Killed: within-video
   AUROC 0.511 CI [0.488, 0.533] — frozen CLIP visual segment keys carry no within-video
   localization signal. Late interaction *retains all 30 segments*; nothing is selected, nothing is
   discarded, and no per-segment selection score is ever formed.
2. **Small-support discrete vote statistics as a selection criterion** (P2 forensic H5: `argmax`
   over a 21-level vote degenerates to a positional tie-break). Every score in this pilot is a
   continuous similarity; the only `max` is `max_j sim(q_k, m_j)` over a continuous cosine, and the
   classifier score is similarity-weighted, not a bounded count.
3. **Hard partitioning of memory by evidence type** (P3b, 0/5 NO-GO). Memory here is a single
   undifferentiated pool of segments; OCR enters as an additive term inside the similarity, never
   as a routing decision.

## Data boundary (hard)

- **HateMM-train only, 744 videos** (298 hateful / 446 not). `dev_seen` (107) is not opened;
  `test` is sealed and never referenced. The script raises `HALT_TEST_CONTACT` on any id whose
  lowercased form contains `test` or `dev_seen`, and only ever opens the three `train_*` caches
  named below.
- No cross-dataset data (HateClipSeg, MHC, ImpliHateVid untouched).

## Inputs (all frozen, all pre-existing)

| block | source | shape |
|---|---|---|
| segment visual | `data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt`, key `subclip_img_feats` | `[22320, 1024]` → `[744, 30, 1024]` |
| whole-video visual | `data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt`, key `img_feats` | `[744, 1024]` |
| whole-video transcript/title text | same file, key `text_feats` | `[744, 768]` |
| labels | same file, key `labels` | `[744]` |
| segment OCR text | `data/OCR/HateMM/ocr_windows_K30.jsonl` (PaddleOCR PP-OCRv6, K=30 midpoint windows), filter `conf >= 0.5 and len(text.strip()) >= 2`, window text = `" ".join(kept)` in file order | 30 strings/video |

The segment cache's `video_ids` order **must** equal the whole-video cache's id order
(`HALT_CACHE_ORDER` otherwise), and `ocr_windows_K30.jsonl` is SHA-256 verified against
`data/OCR/SHA256SUMS.json` at load (`HALT_OCR_CACHE_SHA` otherwise).

**OCR window embeddings.** `data/OCR/HateMM/pilot_ocr_blocks.npz` stores only the *aggregated*
744x768 arm blocks, not per-window vectors, so per-window vectors must be produced once. Recipe is
byte-identical to the one that produced that cache (`OCR_FUSION_PILOT_FREEZE.md` step 3): CLIP text
tower `openai/clip-vit-large-patch14-336`, `CLIPTokenizer(padding=True, truncation=True)` →
`CLIPTextModel(...).pooler_output`, 768-d, frozen, `eval()`/`no_grad`, identical strings encoded
once. Written to `data/OCR/HateMM/pilot_ocr_window_vecs.npz` (keyed by the same SHA) for reuse.
As a **frozen self-check**, re-aggregating these per-window vectors with the OCR-fusion freeze's
rule (drop empty windows, mean of L2-normalized, re-L2) must reproduce that file's `o3` and `o30`
to `max |Δ| < 1e-3`, else `HALT_OCR_VEC_MISMATCH`; the observed `max |Δ|` is recorded in the
result JSON either way. This guarantees the two pilots share one embedding space. The tolerance is
`1e-3` rather than exact because `ocr_fusion_pilot.json` records `device_text_encoder = "cached"`
— the device that produced the stored blocks is unknown, and CUDA-vs-CPU float32 accumulation in
the text tower differs at ~`1e-6` on L2-normalized 768-d outputs. Any *structural* error (wrong
filter, wrong window indexing, wrong pooling, wrong id order) moves these vectors by O(0.1-1), so
`1e-3` still catches every failure the check exists to catch.

## Similarity convention (one rule for every arm)

Every key is a concatenation of **independently L2-normalized blocks**, and similarity is the
**dot product of the concatenated keys**, i.e. the **sum of the per-block cosines**. A missing
block is the frozen all-zero vector and therefore contributes exactly `0`.

This is chosen over cosine-of-the-concatenated-vector deliberately and the reason is recorded here
before any result: 150/744 videos (20.2%) have no usable OCR text anywhere and 255/744 have none in
a 3-window budget, so under cosine-renormalization a segment with an empty OCR block would carry
norm 1 against a norm-`sqrt(2)` partner and be capped at similarity `0.707` — a systematic
missingness penalty that has nothing to do with retrieval quality. Under the sum-of-cosines rule a
video with no OCR scores **exactly** as it does in the visual-only arm, so the arm1→arm2 contrast
is a clean additive OCR increment with no confound. The OCR term is weighted `1.0` against the
visual term; there is no weight grid and no tuning.

For arm 0 every video has both blocks with norm 1, so the sum-of-cosines ranking is identical to
cosine on the concatenated key — arm 0 reproduces the current retrieval method's neighbour ordering
exactly.

## Arms

Let `S[v,k] ∈ R^1024` be the segment visual feature and `O[v,k] ∈ R^768` the segment OCR text
embedding (zero when the window's filtered text is empty). `l2(.)` is L2 normalization.

| arm | name | key(s) per video | video-pair similarity `S(q,m)` |
|---|---|---|---|
| **0** | **baseline (current method)** | one key `[l2(img_whole) ‖ l2(txt_whole)]` | `cos_img + cos_txt` |
| **0v** | visual-only whole video *(control, non-gating)* | one key `l2(img_whole)` | `cos_img` |
| **1** | **LI-visual** | 30 keys `l2(S[v,k])` | `mean_k max_j cos(l2(S[q,k]), l2(S[m,j]))` |
| **2** | **LI-visual+OCR** | 30 keys `[l2(S[v,k]) ‖ l2(O[v,k])]` | `mean_k max_j [ cos_vis(q_k,m_j) + cos_ocr(q_k,m_j) ]` |

`img_whole` is the cache's own 8-frame mean-pooled CLIP vector — the object the pilot is arguing
against — not a re-pool of `S`.

Arm **0v** exists because arm 1 drops the transcript block that arm 0 has, so a bare arm0→arm1
delta confounds "retaining segments" with "losing the transcript". `0v` makes the decomposition
clean and costs one extra 744x744 matrix. It is **reported only** and cannot change the verdict.

## Splits ("seeds")

The pipeline is fully deterministic given features and folds — no model training, no random
initialization, no sampling — so model seeds have structurally zero variance. The seed-consistency
requirement is therefore carried by **three 5-fold splits**:

- **split 0 (primary)** — the frozen Gate-0 seed-20260807 split at
  `artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/folds/fold_{0..4}/{train_ids,query_ids}.json`,
  the same split P1/P2/P3 and the OCR fusion pilot use. All headline purity numbers come from this
  split.
- **split 1** — `StratifiedKFold(n_splits=5, shuffle=True, random_state=20260901)` over the sorted
  744 train ids.
- **split 2** — `StratifiedKFold(n_splits=5, shuffle=True, random_state=20260902)`.

All three stay strictly inside the 744 training videos. For every split and every fold, the
**memory** is that fold's train partition and the **queries** are that fold's query partition, so
every one of the 744 videos is scored out-of-fold exactly once and a video is never its own
neighbour.

## Endpoint 1 (mechanism) — neighbour label purity @ k=10

For query video `i` in fold `f`, take the `k=10` memory videos with the largest `S(i, ·)`.
`purity_i` = fraction of those 10 whose label equals `y_i`. `chance_i` = fraction of fold-`f`
memory videos whose label equals `y_i` (the exact expectation for 10 uniformly random neighbours).

- Reported per arm: `mean_i purity_i`, `mean_i chance_i`, and `lift = mean_i (purity_i - chance_i)`.
- The gating quantity is the **paired arm-vs-arm purity difference** `Δ = mean_i (purity_i^{arm2} -
  purity_i^{arm0})`; `chance_i` is identical across arms on the same split, so `Δ` equals the
  difference of lifts exactly.
- **Bootstrap**: 2000 resamples of the 744 video indices with replacement, `numpy.random.default_rng(20260903)`,
  the *same* index draws applied to every arm (paired). 95% CI = percentiles [2.5, 97.5].
- **Tie handling**: neighbours sorted by `(-similarity, video_id)` with `video_id` compared as a
  string, so ties resolve lexicographically and never by cache position. (P2 forensic H5: index
  order tie-breaks manufacture positional artifacts.)

## Endpoint 2 (performance) — kNN classification macro-F1

Frozen classifier, `k=10`, **similarity-weighted** vote (continuous, non-saturating — P2 forensic
implication 3 forbids a bounded count as a decision statistic):

```
w_j     = max(S(i, j), 0)                      for j in the 10 nearest memory videos
score_i = sum_j w_j * y_j / sum_j w_j          (if sum_j w_j == 0: unweighted mean of y_j)
pred_i  = 1 if score_i >= theta_f else 0
```

`theta_f` is selected **entirely inside fold `f`'s memory**, never on the query partition:
run the identical leave-one-out kNN over the memory (each memory video's 10 nearest *other* memory
videos), then `theta_f, _ = select_threshold(scores_mem, y_mem)` using the project's frozen rule
(`scripts/tera_gate0/common.select_threshold`: midpoints of consecutive unique scores plus
`min-1e-6`/`max+1e-6`, rule `score >= theta`, ties → smallest `|theta-0.5|`, then smallest theta).

**Metric**: macro-F1 over the 744 out-of-fold predictions, computed once per (arm, split).
Reported per split and as the mean over the three splits.

## Decision rule (FROZEN)

Primary contrast is **arm 2 vs arm 0**. Per user ruling this pilot judges "real and non-zero", not
a large effect.

- **GO** if **either**
  - **A (performance)** — `mean over the 3 splits of (F1_arm2 - F1_arm0) >= +0.005` **and**
    `F1_arm2 - F1_arm0 > 0` on **each** of the 3 splits; **or**
  - **B (mechanism)** — on split 0, paired purity difference `Δ >= +0.020` **and** the bootstrap
    95% **lower** bound of `Δ` is `> 0`.
- **NO-GO** if `mean over the 3 splits of (F1_arm2 - F1_arm0) <= 0` **and** split-0 `Δ <= 0`
  (arm 2 fails to beat arm 0 on both endpoints).
- **AMBIGUOUS** otherwise (positive but not clearing GO).

The verdict is computed by the script from these expressions and written to
`idea-stage/li_retrieval_pilot.json`; it is transcribed unedited into the result memo.

### Attribution decomposition (reported, non-gating)

Reported on all three splits for macro-F1 and on split 0 for purity:

| step | contrast | what it isolates |
|---|---|---|
| transcript cost | `arm0v - arm0` | what dropping the transcript block costs the whole-video key |
| **multi-segment retention** | `arm1 - arm0v` | late interaction over 30 retained segments, visual-only on both sides |
| **OCR into the segment key** | `arm2 - arm1` | the additive OCR term, everything else held fixed |
| headline | `arm2 - arm0` | = sum of the three above |

## Blindness

The script is developed and smoke-tested only on (a) synthetic tensors and (b) label-permuted
HateMM-train. No real-label arm metric is printed or inspected before this document is final. The
real run is a **single submission**: all four arms x three splits x both endpoints in one process,
results written once to `idea-stage/li_retrieval_pilot.json` and `idea-stage/LI_RETRIEVAL_PILOT_RESULT.md`.
No re-run for tuning.

## Registered HALT conditions (not performance negatives)

- `HALT_TEST_CONTACT` — any id containing `test`/`dev_seen`, or any such path opened.
- `HALT_CACHE_ORDER` — segment cache `video_ids` != whole-video cache id order.
- `HALT_CACHE_SHAPE` — segment tensor not reshapable to `[744, 30, 1024]`, or block lengths disagree.
- `HALT_OCR_CACHE_SHA` — `ocr_windows_K30.jsonl` SHA-256 != `data/OCR/SHA256SUMS.json`.
- `HALT_OCR_MISSING_VIDEOS` / `HALT_OCR_WINDOW_COUNT` — a train id absent from the OCR cache, or a
  video without exactly 30 windows.
- `HALT_OCR_VEC_MISMATCH` — re-aggregated per-window vectors do not reproduce `pilot_ocr_blocks.npz`.
- `HALT_FOLD_COVERAGE` — the union of query partitions != the 744 ids, for any split.
- `HALT_INCOMPLETE_OOF` — any video without an out-of-fold prediction.
