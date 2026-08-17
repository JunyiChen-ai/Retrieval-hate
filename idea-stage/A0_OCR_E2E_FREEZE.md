# A0 ± OCR end-to-end freeze (RGCL pipeline)

**Frozen 2026-08-09, before any candidate metric was computed.** Nothing below is edited
after results exist. Results go to `idea-stage/A0_OCR_E2E_RESULT.md`.

## Question

The frozen-space pilot (`idea-stage/OCR_FUSION_PILOT_RESULT.md`) measured OCR as a third
input block to a *linear head over frozen CLIP features*: seed-paired
`arm2 - arm0 = +0.0094 ± 0.0044` macro-F1 (AMBIGUOUS under that pilot's rule).

This experiment asks a different question: when the **fusion MLP and the retrieval-guided
contrastive loss are allowed to learn how to use OCR**, does the gain amplify or shrink
relative to +0.0094?

## Pipeline (both arms)

`src/run_rac.py` on HateMM, frozen CLIP ViT-L/14-336 features
(`data/CLIP_Embedding/HateMM/{split}_openai_clip-vit-large-patch14-336_HF.pt`,
img 1024-d, txt 768-d). HateClipper-style fusion MLP is the only trainable module;
RGCL retrieval-guided contrastive loss with FAISS-mined hard negatives + BCE
(`--hybrid_loss True`); readout is the learned-space kNN retrieval vote.

Exact command line = the HateMM invocation in `scripts/slurm/enc3seed.sbatch`, with the
cluster-only bits removed and the two protocol flags below added:

```
python src/run_rac.py --batch_size 64 --lr 0.0001 --epochs 30 --topk 20 \
  --dataset HateMM --model openai_clip-vit-large-patch14-336_HF \
  --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 --fusion_mode align \
  --hard_negatives_loss True --no_hard_negatives 1 --final_eval False \
  --metric cos --loss triplet --batch_norm False --hybrid_loss True --warmup 5 \
  --majority_voting arithmetic --no_pseudo_gold_positives 1 \
  --lambda_seg 0 --seg_mode full --num_subclips 4 \
  --em_rounds 2 --consensus_topk 10 --consensus_margin 0.2 \
  --Faiss_GPU False --force False \
  --val_only_eval True --seed {SEED} [ARM-B-ONLY FLAGS]
```

Splits are the pipeline's own standard HateMM 3-split cache: train 744 / dev_seen 107 /
test_seen 215.

## Arms

| arm | description | extra flags | model class |
|---|---|---|---|
| A | current A0 baseline, unchanged | none | `classifier_hateClipper` |
| B | A + OCR third stream | `--archive_feats data/OCR/HateMM/rac_ocrmean30_{split}.pt --archive_mode stream` | `classifier_hateClipperArchive` |

Arm B reuses the pipeline's **pre-existing** third-stream mechanism (`--archive_mode
stream`, built for the MLLM structured archive): the 768-d OCR vector is concatenated at
the end of `text_feats`, split inside the model, projected by its own
`archive_proj: Linear(768, map_dim) + Dropout(0.2)`, L2-normalised, and concatenated onto
the fused `img ⊙ text` representation before the MLP. Fusion MLP input dim goes
`1024 -> 2048`. No new model code is written for this experiment. Everything else —
optimiser, loss, FAISS hard-negative mining, kNN readout, epoch/checkpoint selection — is
byte-identical between arms.

## OCR feature (frozen definition, unchanged from the pilot)

Arm 2 ("OCR-30") of `OCR_FUSION_PILOT_FREEZE.md`: 30 K-windows per video, boxes filtered
by `conf >= 0.5 and len(text) >= 2`, joined per window, each non-empty window text encoded
by the CLIP text tower (`pooler_output`, 768-d), L2-normalised per window, averaged over
non-empty windows, then L2-normalised. Videos with no usable OCR get the zero vector.

- **train (744 rows): reused verbatim** from `data/OCR/HateMM/pilot_ocr_blocks.npz['o30']`
  — not recomputed.
- **dev_seen (107 rows): encoded** by `scripts/ocr_cache/build_ocr_rac_cache.py` with the
  identical recipe; the script re-encodes 8 train videos and asserts
  `max|delta| <= 1e-3` against the reused block before writing anything.
- Per-window / segment-level OCR usage is **out of scope** (killed in frozen space); this
  experiment uses only the 30-window mean vector.

## Test-set firewall

`--val_only_eval True` (new flag in `src/run_rac.py`, default False, inert when off): the
`test_seen` split is dropped immediately after loading and replaced by a copy of
`dev_seen`. No test row reaches a dataloader, a FAISS index, a metric, or checkpoint
selection. The per-epoch `Test_Retrieval` lines therefore print **dev** numbers and are
ignored; only `Val_Retrieval` lines are read. `data/OCR/HateMM/rac_ocrmean30_test_seen.pt`
is a placeholder carrying the dev rows — HateMM test OCR was never encoded.

## Protocol

- Seeds **0, 1, 2** for both arms (the seeds `enc3seed.sbatch` uses), paired by seed.
- Model selection: the pipeline's own rule, unchanged — best epoch `>= warmup(5)` by
  `Val_Retrieval acc`, tie-broken by `Val_Retrieval roc`.
- **Reported quantity**: at the selected epoch, `Val_Retrieval macroF1` (primary) and
  `Val_Retrieval acc` (secondary), per seed, mean ± std over the 3 seeds, plus the
  seed-paired delta `B - A`.
- Single submission: all 6 runs (2 arms × 3 seeds) launched in one background process. No
  re-run, no tuning after seeing numbers. If a run crashes, the failure is reported as a
  failure; the arm is not silently re-run with different settings.
- Implementation is validated only on synthetic random-feature caches (fake ids/labels) in
  a scratch data root. No real val metric is computed before this file is committed.

## Decision rule (primary = seed-mean `B - A` on val macro-F1)

| condition | verdict |
|---|---|
| `mean(B-A) >= +0.010` **and** sign is positive on 3/3 seeds | **GO** |
| `+0.003 <= mean(B-A) < +0.010`, **or** seeds disagree in sign but mean > 0 | **AMBIGUOUS** |
| `mean(B-A) <= +0.003` | **NO-GO** |

(The `+0.003 … +0.010` band and the `>= +0.010` band are exhaustive with the NO-GO band;
a mean in `(+0.003, +0.010)` with 3/3 positive seeds is AMBIGUOUS, since GO requires both
conditions.)

## Secondary, non-gating readout (records the hypothesis under test)

Compare `mean(B-A)` here against the frozen-space `+0.0094`:

- `>= +0.0141` (1.5×) → learning space **amplifies** the OCR gain;
- `+0.0047 … +0.0141` (0.5×–1.5×) → **unchanged**, the gain is a property of the feature,
  not of where it is fused;
- `< +0.0047` (< 0.5×) → learning space **shrinks** it.

This readout does not change the GO/AMBIGUOUS/NO-GO verdict above.

## Budget

Wall-clock cap 2 h for the 6 runs. If exceeded, `--epochs` is cut and the change is
recorded verbatim in the result file.
