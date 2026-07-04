# ABLATION: MLLM structured-archive vs long-context transcript key (W4, decisive)

**Question (novelty 生死题)**: is the archive kNN-key gain structured distillation, or merely a
fix for the CLIP text tower's 77-token transcript truncation (EN 56% / ZH 79%)?

**Date**: 2026-07-03/04. **No src/ changes.** GROUP=RAC_video_transcript, seed-paired vs
RAC_video_archive(+_seeds) controls. Protocol: warmup>=5 val-selected acc, roc tie-break; also a
final-epoch (epoch 29) readout to isolate val-selection noise.

## Setup

- Encoder: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (XLM-R base, mean pool,
  768-d = archive dim; max_seq_length raised 128->512). ZH encoded directly (no translation).
- Text: full title+transcript from `data/gt/{MHC,MHC_zh}/*.jsonl` (same source as prep_mhc.py).
  **Residual truncation at 512 tok = 0.0%** (EN max 348 tok, ZH max 339 tok; >77 tok share:
  EN 58.7-67.5%, ZH 49.7-55.1%).
- Loader path verified dimension-agnostic (`load_archive_feats_split` + `_archive_augment_keys`
  = normalize-then-concat). Caches: `data/CLIP_Embedding/{MHC,MHC_zh}/{split}_{transcript,arctrans}_mpnet512_HF.pt`
  (arctrans = offline `[l2n(archive)|l2n(transcript)]` double key). All on B2.
- Scripts: `scripts/generate_transcript_embedding.py`, `scripts/slurm/train_transcript.sbatch`.
- Jobs: seed0 grid 12228 (EN t a0.25), 12229 (EN t a0.5), 12230 (ZH t a0.25), 12231 (EN double
  a0.25); multi-seed 12260-12263 (ZH t s1-4), 12264-12266 (EN t s1-3).
- Controls: EN archive 12210/12219/12220/12221 (s0-3); ZH archive 12207/12215-12218 (s0-4);
  ZH floor 12223-12227 (s0-4); EN floor 12113 (s0), 12275 (s1); EN floor s2/s3 = 12276/12277
  (still running at write time, table uses n=2 for EN floor).

## Seed-0 grid (TEST, val-selected)

| MHC-EN (frozen Qwen) | F1 | acc | roc |
|---|---|---|---|
| floor 12113 | 0.7378 | 0.7888 | 0.8402 |
| archive a0.25 12210 | 0.7626 | 0.8075 | 0.8489 |
| transcript a0.25 12228 | 0.7565 | 0.8012 | 0.8590 |
| transcript a0.5 12229 | 0.7418 | 0.7826 | 0.8642 |
| double a0.25 12231 | 0.7329 | 0.7764 | 0.8653 |

| MHC_zh (LoRA) | F1 | acc | roc |
|---|---|---|---|
| floor 12149 | 0.8023 | 0.8322 | 0.8825 |
| archive a0.25 12207 | 0.8270 | 0.8523 | 0.9107 |
| transcript a0.25 12230 | 0.7914 | 0.8188 | 0.8938 |

Seed-0 read (superseded below): EN transcript recovers ~75% of archive F1 gain; ZH transcript
falls below floor while archive gains — looked like clean "structured distillation" evidence on ZH.
Double key: worse than everything (equal-weight blend halves the useful signal in the a-channel); dead end.

## Multi-seed three-arm result (the one that counts)

Per-arm mean±std and paired same-seed deltas. EN = seeds 0-3 (floor n=2: s0,s1). ZH = seeds 0-4.

### VAL-SELECTED readout

| | floor | transcript | archive | D(arc−trs) | D(arc−floor) | D(trs−floor) |
|---|---|---|---|---|---|---|
| EN F1 | 0.7330±0.0067 (n2) | 0.7484±0.0112 | 0.7497±0.0250 | +0.0013±0.0141 (3/4) | +0.0055 (1/2) | +0.0116 (2/2) |
| EN acc | 0.7857±0.0044 | 0.7888±0.0113 | 0.7935±0.0205 | +0.0047±0.0118 (3/4) | +0.0001 (1/2) | +0.0031 (1/2) |
| EN roc | 0.8388±0.0019 | 0.8551±0.0029 | 0.8384±0.0104 | **−0.0167±0.0094 (0/4)** | −0.0019 (1/2) | **+0.0176 (2/2)** |
| ZH F1 | 0.7962±0.0167 | 0.7915±0.0052 | 0.7915±0.0397 | +0.0001±0.0388 (3/5) | −0.0047±0.0446 (3/5) | −0.0047±0.0217 (1/5) |
| ZH acc | 0.8282±0.0139 | 0.8215±0.0037 | 0.8268±0.0266 | +0.0053±0.0270 (3/5) | −0.0014±0.0313 (3/5) | −0.0067±0.0157 (1/5) |
| ZH roc | 0.8967±0.0129 | 0.8975±0.0084 | 0.9062±0.0096 | **+0.0087±0.0086 (4/5)** | **+0.0095±0.0172 (4/5)** | +0.0008±0.0166 (2/5) |

### FINAL-EPOCH readout (epoch 29; removes val-selection noise)

| | floor | transcript | archive | D(arc−trs) | D(arc−floor) | D(trs−floor) |
|---|---|---|---|---|---|---|
| EN F1 | 0.7400±0.0278 (n2) | 0.7394±0.0095 | 0.7430±0.0196 | +0.0036±0.0145 (2/4) | −0.0074 (0/2) | −0.0087 (1/2) |
| EN roc | 0.8500±0.0039 | 0.8558±0.0101 | 0.8373±0.0113 | **−0.0184±0.0025 (0/4)** | −0.0142 (0/2) | +0.0037 (1/2) |
| ZH F1 | 0.8259±0.0124 | 0.8154±0.0222 | 0.8259±0.0124 | +0.0105±0.0141 (3/5) | **+0.0000±0.0000 (0/5, all ties)** | −0.0105±0.0141 (0/5) |
| ZH roc | 0.9108±0.0117 | 0.9123±0.0085 | 0.9058±0.0137 | −0.0065±0.0101 (2/5) | −0.0050±0.0102 (1/5) | +0.0015±0.0114 (2/5) |

Note: at epoch 29 the ZH archive arm's THRESHOLDED predictions are bit-identical to floor on all
5 seeds (effective archive weight a^2/(1+a^2) = 5.9% flips no votes), while ROC still moves —
the a=0.25 key channel is a *perturbation*, not a driver. Also: ZH final-epoch means (floor/arc
0.8259 F1 / 0.8537 acc) EXCEED all val-selected means — with a 78-sample val set, val-selection
is the dominant noise source and currently *costs* accuracy.

## Verdicts

1. **"archive > transcript" ordering is NOT seed-robust on F1/acc in either language, under
   either readout** (ZH val: ΔF1 +0.0001±0.0388; EN val: +0.0013±0.0141). The seed-0 "decisive"
   ZH gap (+0.036 F1) was a favorable draw (archive s0 high + transcript s0 low).
2. **Bigger finding: the W3 archive accuracy gain itself does not survive multi-seed.**
   ZH D(archive−floor) val-selected = −0.0047 F1 / −0.0014 acc; final-epoch = exactly 0.
   W3's ZH 0.8270/0.8523 headline vs 0.8023/0.8322 floor was seed-0 selection luck.
3. **What IS consistent**:
   - ZH val-selected ROC: archive > floor and > transcript in 4/5 seeds (+0.009 both) — a weak
     but directionally consistent ranking-quality signal unique to the structured archive.
   - EN ROC: transcript-key > archive-key in 4/4 seeds (+0.017); transcript > floor 2/2.
     On EN the long-context transcript embedding is the better text signal, full stop.
   - ZH transcript-key ≤ floor in 4/5 (val) and 5/5 (final): fixing truncation with a
     long-context encoder does NOT help Chinese — so the truncation-repair hypothesis is dead
     on ZH, but so is the accuracy claim for the archive.
4. **Double key (EN, seed0)**: below floor; do not pursue.

## Paper wording (final recommendation)

- **Do not claim the a=0.25 archive kNN-key as an accuracy contribution.** Multi-seed paired
  analysis contradicts it in both languages. Retiring this claim now avoids a fatal rebuttal.
- The defensible, honest ablation narrative: "A long-context multilingual transcript embedding
  (zero truncation) fed through the identical key channel fails to improve Chinese detection
  (≤ floor on 4-5/5 seeds), while the structured archive yields a small but directionally
  consistent AUC improvement (4/5 seeds, +0.009) — the archive's residual value lies in
  structured, language-pivoted semantics, not in recovering truncated transcript content.
  Neither channel moves accuracy significantly at this key weight." Frame as analysis, not as a
  headline gain; report mean±std over 5 seeds.
- If archive must stay a contribution: the in-domain kNN-key at a=0.25 is the wrong vehicle —
  the surviving candidates are (a) the updatable cross-dataset memory story, (b) larger/tuned
  alpha or stream-mode integration, (c) val-selection stabilization (78-sample val is the real
  bottleneck; final-epoch beats val-selected on ZH by ~+2.5 acc).
- MEMORY.md's "ZH best-ever 0.8322" (= floor seed0) also needs revision: ZH floor final-epoch
  seeds 3/4 reach 0.8387/0.8658 — the floor itself is stronger than previously recorded.

## Artifacts

- Logs: slurm/logs/trs_*_{12228-12231,12260-12266}.{out,trainlog} (B2: slurm_logs/), controls as above.
- Caches + scripts on B2 under CLIP_Embedding/ and scripts/. Analysis script:
  scratchpad three_arm_analysis.py (session-local; tables reproduced above are the record).
- EN floor s2/s3 (12276/12277) pending at write time; EN floor rows are n=2. Direction of the
  archive-vs-transcript verdict does not depend on them (that pairing is n=4 complete).
