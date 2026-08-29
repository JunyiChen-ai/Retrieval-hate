# OCR three-stream fusion pilot — FROZEN before any candidate number was computed

Written 2026-08-09, **before** `scripts/ocr_cache/ocr_fusion_pilot.py` was executed on real labels.
Everything below (arm definitions, feature construction, head, protocol, seeds, decision rule) is
fixed at write time and is **not** edited after results appear. This is a head-level pilot ladder,
not a registered verdict.

## Question

The current frozen-feature + light-head pipeline has no OCR input. The OCR cache
(`data/OCR/HateMM/`, PaddleOCR PP-OCRv6, K=30 midpoint windows) was unblocked on 2026-08-08 on the
strength of the Gate-C re-analysis (30.1% of misses = on-screen-text evidence present, speech
absent; OR 2.29). **How much train-OOF macro-F1 does adding OCR as a third stream actually buy,
and is the gain dose-dependent in the number of OCR windows read?**

## Data boundary (hard)

- **HateMM-train only, 744 videos** (298 hateful / 446 not). `dev_seen` (val, 107) is not opened
  by this pilot; `test` is sealed and never referenced.
- The OCR cache covers 851 HateMM videos (744 train + 107 val). The pilot reads
  `ocr_windows_K30.jsonl` and **filters to the 744 train ids** before any other step. Val rows are
  discarded at load time.
- No cross-dataset data (HateClipSeg is not touched).

## Folds

The frozen seed-20260807 5-fold split at
`artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/folds/fold_{0..4}/{train_ids,query_ids}.json`
— the same split PILOT_FREEZE.md (P1/P2/P3) uses. Fold membership is identical for all three arms.

Inner folds (for epoch selection only): `StratifiedKFold(n_splits=4, shuffle=True,
random_state=20260808)` over the sorted outer-train id list, i.e. the Gate-0 `INNER_FOLD_SEED`
convention.

## Feature construction

**Visual block (all arms).** `l2(img_feats)`, 1024-d, from
`data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt`
(frozen CLIP ViT-L/14-336, 8-frame mean pooling).

**Speech/title text block (all arms).** `l2(text_feats)`, 768-d, from the same cache.

**OCR block (arms 1 and 2 only), 768-d.** Constructed as follows, deterministically:

1. **Per-window text.** For video `v` and window `k ∈ {0..29}`, take the detection list in
   `data/OCR/HateMM/ocr_windows_K30.jsonl` in file order. Keep a detection iff
   `conf >= 0.5` **and** `len(text.strip()) >= 2` — the same filter as the cache-stats
   "filtered" view (`data/OCR/ocr_cache_stats.json`). The window text is `" ".join(kept texts)`,
   in file order, no de-duplication, then `.strip()`.
2. **Window set per arm.**
   - arm 1 (OCR-3): `k ∈ {5, 15, 25}` — equally spaced, the midpoints of the three equal thirds of
     the 30-window grid (`round((i + 0.5) * 30 / 3)` for `i = 0,1,2`).
   - arm 2 (OCR-30): all `k ∈ {0..29}`.
3. **Encoding.** Each non-empty window text is embedded with the **CLIP text tower**
   `openai/clip-vit-large-patch14-336`, `CLIPTokenizer(..., padding=True, truncation=True)` →
   `CLIPTextModel(...).pooler_output`, 768-d. This is the same recipe that produced the project's
   existing whole-video `text_feats` and the I5 gate's OCR key, so the OCR stream lives in the same
   embedding space as the speech stream. Encoding runs on GPU if available, else CPU; the encoder
   is frozen and in `eval()`/`no_grad` mode. Identical texts are encoded once (cache by string).
4. **Aggregation.** L2-normalize each window embedding, then take the **unweighted mean over the
   arm's non-empty windows**, then L2-normalize the mean.
5. **Missing rule (frozen).** A window with empty filtered text contributes nothing (it is dropped
   from the mean, not averaged in as a zero). A video with **no** non-empty window inside its arm's
   window set gets the **all-zero 768-d vector**. Under a linear head a zero block contributes
   exactly zero to the logit, i.e. "missing" is neutral. **No missingness indicator feature is
   added** — the arms differ only in the OCR block.

**Arm input vectors.**

| arm | name | input | dim |
|---|---|---|---|
| 0 | baseline | `[l2(img) ‖ l2(txt)]` | 1792 |
| 1 | OCR-3 | `[l2(img) ‖ l2(txt) ‖ ocr3]` | 2560 |
| 2 | OCR-30 | `[l2(img) ‖ l2(txt) ‖ ocr30]` | 2560 |

Arms 1 and 2 have **identical dimensionality**, so any arm1-vs-arm2 difference is a pure
window-budget (dose) effect, not a capacity effect.

## Head and training (identical across arms)

Mirrors the Gate-0 `A0` arm (`scripts/tera_gate0/arms.py`):

- `nn.Linear(d, 1)` → one logit. Output layer init `normal_(0, 0.01)` / `zeros_` bias.
- `BCEWithLogitsLoss()`, no `pos_weight`.
- `AdamW(lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)`.
  **No hyperparameter grid** — lr/wd are fixed at these values for every arm, seed and fold, so no
  arm can win by getting a better search.
- Batch size 64, shuffled per epoch by a generator seeded from `(seed, arm, outer fold, inner fold,
  epoch)`.
- `E_MAX = 200`, `PATIENCE = 40`, `MIN_DELTA = 1e-4`.

**Epoch and threshold selection (never touches the outer query fold).** Inside each outer fold:
the 4 inner models are advanced in lockstep one epoch at a time; after every epoch the pooled
inner-OOF macro-F1 is computed at that epoch's best threshold; the selected epoch `E*` is the
argmax under `MIN_DELTA`/`PATIENCE` early stopping. The decision threshold `theta` is the one
selected on the pooled inner-OOF scores at `E*` using the Gate-0 `select_threshold` rule
(midpoints of consecutive unique scores plus `min-1e-6` / `max+1e-6`; rule `score >= theta`;
ties → smallest `|theta - 0.5|`, then smallest `theta`). One model is then refit on the **full**
outer-train partition for exactly `E*` epochs and applied to the outer query fold with `theta`.

## Seeds

3 seeds: **20260810, 20260811, 20260812** (`MODEL_SEED_BASE` = 20260810 and successors). The seed
controls head init and batch shuffling only; folds and features are seed-independent.

## Metric

**macro-F1 over the 744 out-of-fold predictions**, computed once per (arm, seed). Reported as
mean ± sample std (ddof=1) over the 3 seeds. Paired per-seed deltas `arm_i − arm_0` are also
reported.

## Decision rule (FROZEN)

Primary comparison is **arm 2 (OCR-30) − arm 0 (baseline)**, seed-mean OOF macro-F1:

- `>= +0.015` → **GO**
- `+0.005` to `+0.015` (i.e. `>= +0.005` and `< +0.015`) → **AMBIGUOUS**
- `< +0.005` → **NO-GO**

`arm 1 − arm 0` is reported alongside as the dose point at 3/30 windows. It **does not** change the
verdict; it only characterizes the dose–response shape (how much of the OCR-30 gain is already
available from 3 windows, i.e. a 10× cheaper OCR budget).

## Blindness

The script is developed and smoke-tested only on (a) synthetic tensors and (b) label-permuted
HateMM-train, and no real-label arm metric is printed or inspected before this document is final.
The real run is a **single submission**: all three arms × three seeds in one process, results
written once to `idea-stage/OCR_FUSION_PILOT_RESULT.md` and `idea-stage/ocr_fusion_pilot.json`.
No re-run for tuning.

## Registered HALT conditions (not performance negatives)

- Any of the 744 train ids missing from the OCR window cache, or any video not having exactly 30
  windows → HALT.
- Whole-video cache id order not matching the ids used to index OCR → HALT.
- Any `test_seen` / `dev_seen` path opened → HALT.
- SHA-256 of `ocr_windows_K30.jsonl` not matching `data/OCR/SHA256SUMS.json` → HALT.
