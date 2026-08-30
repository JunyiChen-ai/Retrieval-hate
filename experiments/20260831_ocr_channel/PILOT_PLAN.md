# Pilot O: per-second OCR channel for LOCO-ST (iteration round 2)

Date: 2026-08-31. Goal-driven: LOCO-ST leads 3/4 corpora but loses HCS to VERA
(.5450 vs .5619) and HateMM is non-significant (+.045). Diagnosis evidence for
the OCR lever: (a) HCS supervised ceiling with current features is .599 — a
richer per-second channel is the only way past VERA there; (b) the frozen
Gate-C reanalysis found on-screen-text the single significantly enriched
modality gap among misses (OR 2.29; OCR channel unlocked by user ruling
2026-08-08); (c) an OCR window cache already exists (PaddleOCR, K=30 windows
per video, HCS 395/395 complete, HateMM -2, MHC-EN/ZH ~20% missing).

## Mechanism (single change: one added input channel)

Per second t of each video: take the K=30 OCR window containing t, concatenate
its texts with conf >= .5 (cap 400 chars), embed with the SAME sentence
encoders as the existing text channel (bert-base-uncased CLS 768-d;
bert-base-chinese for ZH), empty text -> zero vector. New feature dir
`ocr_bert_1fps` alongside the frozen three. Videos missing from the OCR cache
get all-zero channels and are counted.

## Stages and frozen gates

- **O1 kill check (HCS only, cheapest)**: rebuild the supervised skyline
  (train-span-trained TemporalConv, test-evaluated) with clip+vgg+bert+ocr.
  Gate: HCS within-ROC ceiling >= .62 (vs .599 without OCR). Below -> the
  channel does not lift the ceiling; kill the round, report negative.
- **O2 (only if O1 passes)**: extract OCR embeddings for all corpora; rerun
  LOCO-ST A1 valsel + loo_zero with 4-modal features, 5 seeds, dense scores.
  Gates: (i) HCS valsel or loo_zero > .5619 (VERA) mean within-ROC and paired
  bootstrap vs VERA CI > 0; (ii) HateMM/EN/ZH must not drop more than .015
  vs the 3-modal method; (iii) attribution: zeroing the OCR channel at
  inference on HCS must remove the gain.
- Metrics/protocol: unchanged (test, within-ROC macro primary, shared
  evaluator, seeds 234/2025/3407/42/20260830).

## O1b (2026-08-31, frozen before run): HateMM ceiling check

O1 (HCS) FAILED (.5827 vs .5809 without OCR — channel does not lift the HCS
ceiling; HCS-OCR route killed). The Gate-C on-screen-text evidence came from
HateMM miss analysis, and HateMM is the corpus needing significance. O1b:
same skyline comparison on HateMM (3 seeds). Gate: 4-modal HateMM supervised
within-ROC >= .77 (3-modal reference rerun in the same script; must also
exceed that reference by >= .01) else the OCR round is closed entirely.
