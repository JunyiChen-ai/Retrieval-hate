# P4 — Archive-field auxiliary distillation loss

Front: P4 (campaign goal = give the MLLM a *real* method role: removing it must
measurably cost something). Backup front to P1. Same discipline: pre-register, probe-gate
before training, bit-for-bit floor reproduction, honest kill if within noise.

Idea: during head training, small auxiliary linear heads on the fused embedding predict the
MLLM archive's schema fields for each TRAIN video. `L = L_main + λ·Σ_field aux_field`. The
MLLM's structured reading shapes the embedding space at training time; the aux heads are
discarded at eval. **Removing the MLLM = removing the aux supervision = the ablation
(λ=0).**

---

## PRE-REGISTRATION (locked before training; written 2026-07-06)

### Method
- Head = the CLIP-space RGCL head, EXACT `RAC_video_CLIP` floor recipe (align fusion,
  triplet + hybrid BCE, topk=20 arithmetic vote, 30 ep, seed∈{0,1,2}, Faiss CPU).
- Aux heads: one `nn.Linear(proj_dim=1024 → dim_field)` per field, attached to the
  grad-tracked whole-video fused embedding `feats` (the SAME tensor the main loss uses; no
  second forward → no extra dropout RNG). CE for the single-label field, BCE-with-logits
  for the multi-label fields. `L = L_main + λ·Σ_field aux_field` (sum of per-field mean over
  valid samples). Heads optimised jointly (added to the AdamW param list); discarded at eval.
- **λ = 0.1, single value, NO tuning.**
- Aux supervision on the TRAIN split only. Samples with a missing/unparseable archive are
  masked out of the aux loss ONLY (never the main loss).
- Leakage: the archives are MLLM-generated from the video content alone (no gold labels), so
  using them as TRAIN targets is clean.

### Bit-for-bit guarantee (verified in code + external review)
- λ=0 ⇒ `aux_pack=None`: no aux heads built (no RNG drawn), optimizer over
  `model.parameters()` only, aux loss block skipped ⇒ byte-identical to the pre-change
  baseline. Empirical check: λ=0 seed-0 val-selected TEST must equal the known floor.
- λ>0 clean isolation: aux heads are built with the global CPU-RNG state saved/restored, so
  the DataLoader shuffle (RandomSampler, num_workers=0, global CPU generator) and dropout
  masks are identical to the floor; the ONLY source of difference is the aux gradient
  (which, as with any joint loss under grad-norm clipping, also participates in the shared
  parameter update — inherent and intended).

### Frozen field vocabularies (derived from TRAIN v2 archives, top-N by count desc + OTHER)
| field | type | MHC (EN) classes | MHC_zh classes | named-mass coverage |
|---|---|---|---|---|
| explicitness | single (CE) | none / implicit / explicit | none / implicit / explicit | 1.00 (exhaustive) |
| modality | multi (BCE) | visual / speech / on_screen_text | same | 1.00 (exhaustive) |
| mechanism | multi (BCE) | stereotyping / slur / insult / OTHER | slur / insult / stereotyping / OTHER | EN 0.956, ZH 0.920 |
| target_group | multi (BCE) | gay people / women / men / OTHER | women / men / effeminate men / OTHER | EN 0.640, ZH 0.837 |

(target_group is long-tailed; N=3+OTHER by design — the probe gate is the arbiter of whether
it is usable. Frozen JSON: `scripts/analysis/p4_out/field_vocab_<ds>.json`.)

### Probe gate (CPU, run BEFORE training — mandatory)
On the TRAIN split, frozen whole-video CLIP concat[img,text]:
- (a) DECODABILITY — 5-fold CV logistic probe per field beats the field's majority baseline
  (single-label: accuracy; multi-label: per-class ROC-AUC > 0.55).
- (b) LABEL-INFORMATIVENESS — 5-fold CV logistic regression from the field encodings to the
  hateful label beats majority (AUC > 0.55 and accuracy > majority).
- Gate CLOSED for a dataset only if BOTH (a) and (b) fail there (aux loss = noise ⇒ do not
  train, report). Otherwise OPEN.

### Conditions (one test measurement per cell)
Per gated-open dataset: floor (λ=0) vs aux (λ=0.1), seeds {0,1,2}. Report BOTH protocols
(val-selected + final-epoch), macro-F1 and accuracy, and paired per-seed deltas.

### Success criteria (pre-registered)
1. λ=0 reproduces the floor bit-for-bit (EN val-sel 0.7826 acc / 0.7113 maF1; ZH 0.8054 /
   0.7706).
2. Probe gate passes.
3. Aux beats floor with mean ΔmacroF1 > 0.01 (noise floor ≈ 1.6 videos) AND ≥2/3 seeds
   positive, on ≥1 dataset under BOTH protocols, with no >0.01 macro-F1 harm elsewhere.
   Weaker than this ⇒ within-noise, no claim, honest kill.

---

## PROBE-GATE RESULTS (run 2026-07-06, before training) — BOTH DATASETS OPEN

JSON: `scripts/analysis/p4_out/probe_gate.json`.

(a) Decodability from frozen CLIP concat[img,text] (5-fold CV):
| field / class | MHC AUC (or acc) | MHC_zh AUC (or acc) |
|---|---|---|
| explicitness (probe acc vs maj) | 0.709 vs 0.556 ✓ | 0.701 vs 0.575 ✓ |
| modality: visual / speech / on_screen_text | 0.888 / 0.926 / 0.726 | 0.820 / 0.874 / 0.774 |
| mechanism: (top3) / OTHER | 0.828 / 0.745 / 0.768 / 0.622 | 0.798 / 0.762 / 0.762 / 0.731 |
| target_group: (top3) / OTHER | 0.931 / 0.802 / 0.880 / 0.755 | 0.810 / 0.694 / 0.826 / 0.867 |

Every field is decodable (all multi-label classes AUC > 0.55; explicitness beats majority).

(b) Label-informativeness (field encodings → hateful label, 5-fold CV):
| | AUC | acc vs majority | macro-F1 | verdict |
|---|---|---|---|---|
| MHC | 0.744 | 0.749 vs 0.694 | 0.666 | INFORMATIVE |
| MHC_zh | 0.784 | 0.750 vs 0.689 | 0.676 | INFORMATIVE |

**Gate: OPEN for both MHC and MHC_zh** (a_pass ✓, b_pass ✓). The archive fields are both
linearly decodable from the frozen representation and carry video-label signal, so the aux
distillation is a plausible training signal — proceed to train both datasets.

---

## RESULTS

Training job **12360** COMPLETED (all 12 runs = 2 ds × {λ=0, λ=0.1} × seeds 0/1/2, each a
full 30-epoch run; 6m16s total — these are tiny frozen-feature heads, ~25s/run at ~1.15
epoch/s, so the short wall-clock is expected, not a crash). JSON:
`scripts/analysis/p4_out/p4_results.json`.

### Success-criteria scorecard
| # | criterion | outcome |
|---|---|---|
| 1 | λ=0 reproduces the floor bit-for-bit | **PASS (exact)** — MHC λ=0 s0 val-sel = 0.7826 acc / 0.7113 maF1; MHC_zh = 0.8054 / 0.7706; both match the known floor to 4 dp |
| 2 | probe gate passes | **PASS** — both datasets OPEN (see above) |
| 3 | aux beats floor >0.01 maF1, ≥2/3 seeds, both protocols, no >0.01 harm | **FAIL** — no win dataset; val-selected harm on both |

### Floor (λ=0) vs aux (λ=0.1), 3 seeds, paired
| dataset | protocol | floor maF1 | aux maF1 | Δ maF1 (per-seed) | seeds+ | Δ acc |
|---|---|---|---|---|---|---|
| MHC (EN) | val-selected | 0.6715 | 0.6291 | **−0.0424** [−.087, +.079, −.120] | 1/3 | −0.0269 |
| MHC (EN) | final-epoch | 0.7202 | 0.7188 | **−0.0014** [+.001, −.002, −.004] | 1/3 | +0.0041 |
| MHC_zh | val-selected | 0.7676 | 0.7551 | **−0.0124** [−.063, +.010, +.016] | 2/3 | −0.0134 |
| MHC_zh | final-epoch | 0.7720 | 0.7797 | **+0.0077** [+.013, +.020, −.010] | 2/3 | +0.0045 |

### What happened
- **Bit-for-bit and probe gate both pass**, so the negative result is trustworthy (not a
  bug and not a dead field): the aux code is a verified no-op at λ=0, and the fields it
  distils are genuinely decodable + label-informative.
- **The aux term does not reliably help.** Under the STABLE final-epoch protocol it is
  essentially flat on EN (−0.0014 maF1) and a **sub-threshold +0.0077 maF1 on ZH** (2/3
  seeds positive but below the pre-registered 0.01 bar). Under the NOISIER val-selected
  protocol it is net-negative on both (EN −0.042, ZH −0.012), driven by val-selection
  variance — e.g. MHC λ=0.1 s2 val-selects epoch 16 (maF1 0.580) though its final-epoch
  maF1 is 0.726. This matches the known regime where 78-sample dev selection is the
  dominant noise source on these ~150-sample test sets.
- **No dataset clears the pre-registered bar under BOTH protocols**, and two cells show
  >1pt harm ⇒ criterion (3) fails.
- **Mechanism (why it is neutral):** the fused embedding is already trained end-to-end with
  the hateful label (contrastive + hybrid BCE). The archive fields correlate with that label
  (probe-b AUC 0.74–0.78) but carry little signal *beyond* it, and they are the MLLM's noisy
  reading — so distilling them adds no information the label does not already provide, and
  the extra head only mildly perturbs the shared trunk. The signal is real (probe passes)
  but redundant with the supervised objective.

### Verdict (plain language)
**Archive-field auxiliary distillation does NOT earn the MLLM a method role here.** It is
within-noise-to-slightly-harmful: flat on EN, +0.8pt (sub-threshold) on ZH final-epoch,
net-negative under val-selection, and it fails the pre-registered success bar on both
datasets under both protocols. The honest kill is clean because the two guards (bit-for-bit
λ=0, probe gate) both pass — the fields are decodable and label-informative, but distilling
them into an already label-supervised embedding is redundant. No tuning of λ was attempted
(pre-registered single value) — tuning to chase the sub-threshold ZH gain would be p-hacking
against a within-noise effect.

### Jobs / artifacts / repro
- Probe gate: `scripts/analysis/p4_probe_gate.py` (CPU). Vocab: `p4_out/field_vocab_<ds>.json`.
- Code (new flag + loader + loss, no behavior change at λ=0):
  `--lambda_aux` in `src/run_rac.py`, `compute_aux_loss` in `src/model/loss.py`,
  `src/utils/p4_archive_fields.py` (shared field schema). External review: bit-for-bit
  no-op + RNG isolation confirmed.
- Training: `scripts/slurm/train_p4aux.sbatch` (job 12360, 12 runs), GROUP `RAC_video_p4aux`.
  Parser: `scripts/analysis/p4_collect.py`. Results: `p4_out/p4_results.json`.

### Jobs / artifacts / repro
- Probe gate: `scripts/analysis/p4_probe_gate.py` (CPU). Vocab: `p4_out/field_vocab_<ds>.json`.
- Code (new flag + loader + loss, no behavior change at λ=0):
  `--lambda_aux` in `src/run_rac.py`, `compute_aux_loss` in `src/model/loss.py`,
  `src/utils/p4_archive_fields.py` (shared field schema). External review: bit-for-bit
  no-op + RNG isolation confirmed.
- Training: `scripts/slurm/train_p4aux.sbatch` (12 runs = 2 ds × {0, 0.1} × 3 seeds),
  GROUP `RAC_video_p4aux`. Parser: `scripts/analysis/p4_collect.py`.
