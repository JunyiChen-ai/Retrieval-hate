# Competitor evaluation protocols: LELA (2602.09637) and TANDEM (2601.11178)

Date: 2026-08-30. Source: arXiv HTML full texts (LELA v1, 10 Feb 2026; TANDEM v2/v3 — v3 of 29 Jul 2026 is textually identical to v2 of 28 May 2026). Purpose: decide whether their numbers can sit next to our benchmark protocol (1 fps grid, wav-duration frame count, official test splits, pooled frame ROC-AUC/AP + within-hate-video macro ROC).

---

## Paper 1: LELA — "Towards Training-free Multimodal Hate Localisation with Large Language Models"

arXiv 2602.09637v1 (cs.CV, 10 Feb 2026). Authors: Yueming Sun, Long Yang (Durham), Jianbo Jiao (Birmingham), Zeyu Fu (Exeter).

### 1. Datasets and splits
- Datasets: **HateMM** and **MultiHateClip English only** (they say "MHC-English"; no MHC-zh).
- Splits: **NOT SPECIFIED anywhere in the paper.** No train/val/test split is named, no cohort sizes are given, no video counts, no hateful-video counts. It is unknowable from the paper whether they evaluate on an official test split, the full dataset, or a subset. (A training-free method could legitimately run on all videos.)
- The paper references an Appendix for a preliminary prompting study, but the arXiv HTML (v1, the only version) contains no appendix.

### 2. Frame grid / temporal resolution
- **NOT SPECIFIED.** The method is described as producing a score s_j per "frame" j (per-frame modality captions from BLIP-2 / EasyOCR / Whisper / LP-MusicCaps, plus PDVC video-snippet captions; max over the 5 per-modality LLM scores per frame), but the paper never states the sampling rate, frame count per video, or how annotated hateful spans are rasterized to frame labels.
- The only protocol anchor: "Our evaluation protocol follows the established practice introduced in [8]" — [8] = LAVAD (Zanella et al., CVPR 2024, training-free VAD). LAVAD computes frame-level AUC pooled over the concatenation of all test-video frames. So LELA is presumably pooled frame-level, but at LAVAD-style frame sampling (LAVAD captions every 16th frame and interpolates), not at a declared 1 fps grid.

### 3. Metrics and headline numbers
- Metrics: frame-level **ROC-AUC** and **AP ("PR-AUC")**, threshold-agnostic, plus thresholded Acc / Macro-F1 / F1(H) / P(H) / R(H) at a fixed LLM-score threshold tau = 0.5.
- Pooled vs per-video: never stated explicitly; following LAVAD implies **pooled over all frames of all evaluated videos**. No within-video / per-video macro metric anywhere.

Main tables, transcribed verbatim (note the trailing dashes / en-dashes in supervised rows are as printed in the paper — several numbers are typographically corrupted):

**Table 1 — HateMM** (caption: "Comparison of supervised and zero-shot methods on HateMM dataset.")

| Method | PR-AUC (%) | ROC-AUC (%) | Acc | M-F1 | F1(H) | P(H) | R(H) |
|---|---|---|---|---|---|---|---|
| HTMM [4] | - | - | 0.7481 | 0.7353 | 0.6728 | 0.6954 | 0.6546 |
| MHCL [5] | - | - | 0.7503 | 0.7407 | 0.6238 | 0.6642 | 0.6831 |
| HVGuard [38] | - | - | 0.8563 | 0.8597 | 0.8479 | 0.8228 | 0.8009 |
| Yue et al. [39] | - | - | 0.821- | - | 0.771- | 0.798- | 0.754- |
| CMFusion [40] | - | - | 0.823- | - | 0.860- | 0.817- | 0.908- |
| Wang et al. [41] | - | - | 0.82– | 0.82– | 0.80– | 0.80– | 0.79– |
| MM-HSD [42] | - | - | 0.878- | 0.874- | 0.853- | 0.849- | 0.857- |
| ZS-CLIP [43] | 0.5216 | 0.5367 | 0.5019 | 0.6455 | 0.5699 | 0.5425 | 0.5215 |
| ZS ImageBind [44] | 0.5237 | 0.5683 | 0.5317 | 0.4813 | 0.6419 | 0.5042 | 0.4981 |
| LLAVA-1.5 [27] | 0.5327 | 0.5529 | 0.5304 | 0.5149 | 0.4971 | 0.5653 | 0.4742 |
| LAVAD [8] | 0.5781 | 0.6163 | 0.5716 | 0.5862 | 0.5319 | 0.5784 | 0.6827 |
| Lin et al. [10] | 0.6239 | 0.5674 | 0.6738 | 0.6127 | 0.5573 | 0.7132 | 0.4568 |
| **LELA (ours)** | **0.7264** | **0.6756** | 0.7148 | 0.7043 | 0.6484 | 0.8152 | 0.5481 |

**Table 2 — MHC (English)** (caption: "Comparison of supervised and zero-shot methods MHC dataset.")

| Method | PR-AUC (%) | ROC-AUC (%) | Acc | M-F1 | F1(H) | P(H) | R(H) |
|---|---|---|---|---|---|---|---|
| HTMM [4] | - | - | 0.6861 | 0.7456 | 0.6817 | 0.6875 | 0.6597 |
| MHCL [5] | - | - | 0.7498 | 0.7317 | 0.6153 | 0.6750 | 0.6783 |
| HVGuard [38] | - | - | 0.8090 | 0.6646 | 0.4556 | 0.4722 | 0.5000 |
| Yue et al. [39] | - | - | 0.78– | - | 0.77– | 0.80– | 0.77– |
| Wang et al. [41] | - | - | 0.82– | 0.81– | 0.76– | 0.87– | 0.68– |
| ZS-CLIP [43] | 0.5181 | 0.5449 | 0.5021 | 0.6395 | 0.5794 | 0.5371 | 0.5289 |
| ZS ImageBind [44] | 0.5135 | 0.5753 | 0.5391 | 0.4756 | 0.6392 | 0.5074 | 0.5014 |
| LLAVA-1.5 [27] | 0.5319 | 0.5438 | 0.5342 | 0.5062 | 0.4974 | 0.5766 | 0.4694 |
| LAVAD [8] | 0.5865 | 0.6302 | 0.5833 | 0.5799 | 0.5344 | 0.5671 | 0.6926 |
| Lin et al. [10] | 0.6147 | 0.5626 | 0.6174 | 0.5849 | 0.5637 | 0.7233 | 0.4813 |
| **LELA (ours)** | **0.7227** | **0.6733** | 0.7124 | 0.6923 | 0.6568 | 0.8217 | 0.5387 |

**Internal inconsistency (load-bearing):** Table 1/2 headers put PR-AUC first (LELA HateMM: PR-AUC 0.7264, ROC-AUC 0.6756), but Table 3/4 (LLM comparison; GPT-4o Mini = the chosen LELA backbone) print the SAME numbers with columns swapped: "GPT-4o Mini: ROC_AUC 72.64, PR_AUC 67.56" (HateMM) and "ROC_AUC 72.27, PR_AUC 67.33" (MHC). The §4.2 prose says "GPT-4o Mini achieves the best results overall (e.g., 72.64% ROC_AUC on HateMM, 72.27% on MHC)". So per the running text, **LELA frame ROC-AUC = 72.64 (HateMM) / 72.27 (MHC) and frame AP = 67.56 / 67.33**, and the Table 1/2 column headers are most likely mislabeled. Either way there is a 5-point ambiguity that the paper never resolves.

Also note: the supervised rows (HTMM, MHCL, HVGuard, Yue, CMFusion, Wang, MM-HSD) carry only video-level Acc/F1 numbers copied from those papers' video-level classification protocols (which include their own train splits and, for HateMM, 5-fold CV) — i.e., the tables mix video-level supervised classification numbers with frame-level training-free localization numbers in the same columns.

Ablation anchors (HateMM, GPT-4o Mini): multi-stage prompting full = 68.28 ROC-AUC (speech only); modality composition: Speech 68.28 → +Image 68.89 → +OCR 71.47 → +Music 71.75 → +Video 72.27 (Table 7 says AUC; the final 72.27 matches the MHC ROC number in Table 4, another sloppiness signal). Threshold sensitivity (Table 5): Acc peaks 0.7148 at tau = 0.5 (tau apparently chosen on the same evaluation data — irrelevant for AUC/AP, contaminates Acc/F1).

### 4. Supervision
Fully **training-free / zero-shot**: no fine-tuning, no use of train labels. Captioners (BLIP-2, EasyOCR, Whisper, LP-MusicCaps, PDVC) + LLM scoring only at inference. Threshold 0.5 fixed (but sensitivity analysis appears to touch the eval set).

### 5. Models and inference cost
- Backbone LLM for scoring: **GPT-4o Mini** (chosen after comparing DeepSeek-R1-1.5B/7B, Qwen2.5-3B/7B, LLaMA-2-7B, Gemini-2.0-Flash / Gemini-1.5-Pro — Tables 3/4 label the Gemini row inconsistently between datasets).
- The LLM runs at inference for every frame x {summarization + multi-stage scoring} x 4 modality compositions → on the order of 5-10 API calls per scored frame. No latency/cost numbers reported.

### 6. Within-video / high-positive-fraction analysis
**None.** No per-video macro metric, no positive-fraction analysis, no per-video breakdown beyond 4 qualitative examples. Everything (apparently) pooled.

### 7. Baselines
Training-free: ZS-CLIP, ZS ImageBind, LLaVA-1.5, **LAVAD**, Lin et al. 2025 (zero-shot VAD reasoning framework, arXiv 2511.00962). Supervised (video-level, quoted): HateMM baseline, MHC baseline, HVGuard, Yue et al., CMFusion, Wang et al., MM-HSD. **No VadCLIP, no MACIL-SD, no MultiHateLoc, no TANDEM.**

### 8. Code availability
**None.** No repo URL in the paper; web search finds no LELA repository. Additional quality flag: the bibliography contains obviously fabricated placeholder entries (e.g., three different refs with arXiv IDs 2301.12345 / 2305.12345 / 2306.12345, and a duplicated HateMM citation with wrong authors), consistent with unverified LLM-assisted writing.

### LELA comparability verdict
**Cannot be placed next to our numbers as printed.** The metric family matches ours (pooled frame ROC-AUC + frame AP), which makes the temptation real, but: (a) evaluation cohort unknown (no split, no sizes — ours is the official test split); (b) frame grid unknown (ours is 1 fps over wav duration; theirs is LAVAD-style, unstated); (c) rasterization of GT spans unstated; (d) the ROC/PR column swap leaves a 5-point ambiguity in which number is which; (e) no within-video macro at all. Any of (a)-(c) alone shifts pooled frame AUC by several points (positive-frame prevalence changes with cohort and grid).

**What a rerun under our protocol requires:** full reimplementation (no code): per-frame captions at 1 fps (BLIP-2 image captions, EasyOCR, Whisper transcript alignment, LP-MusicCaps, PDVC dense video captions), then per-frame GPT-4o Mini multi-stage prompting (summarize + rationale + score) for 4 modality compositions, max-pooled. Cost driver is the LLM: roughly (test frames) x ~8 calls; HateMM test at 1 fps is on the order of 2-3 x 10^4 frames → ~2 x 10^5 GPT-4o-Mini calls per dataset. Feasible but a real API spend; the captioning stack is the engineering burden. A cheaper honest alternative: implement only the scoring recipe on our existing caption/transcript caches and label it "LELA-style prompting", not "LELA".

---

## Paper 2: TANDEM — "Temporal-Aware Neural Detection for Multimodal Hate Speech"

arXiv 2601.11178 (cs.AI); v2 28 May 2026, v3 29 Jul 2026 (v3 body text identical to v2). Authors: Girish A. Koushik, Helen Treharne, Diptesh Kanojia (Surrey). Abstract page notes ICWSM acceptance.

### 1. Datasets and splits
Table 1 of the paper (verbatim):

| Dataset | Labels | Train | Val | Test | Source | Segments / Targets |
|---|---|---|---|---|---|---|
| HateMM (Das et al. 2023) | Hate / Non-hate | 779 | 87 | 217 | BitChute (EN) | Yes / Yes |
| MultiHateClip (Wang et al. 2024) | Hateful / Offensive / Normal | 1,200 | 400 | 400 | YouTube (EN), Bilibili (ZH) | Yes / Yes |
| ImpliHateVid (Rehman et al. 2025) | Explicit / Implicit / Normal | 1,009 | 500 | 500 | YouTube (EN) | No / No |

- "For MHC, we utilize only the English data for experimentation" — yet the split row (1,200/400/400 = 2,000) counts BOTH languages; the English-only test cohort size is never stated (presumably ~200 if the 60/20/20 split is per-language, but that is our inference, not theirs).
- HateMM 779/87/217 sums to 1,083 (the full corpus), but **HateMM has no official train/val/test split** (the original paper used 5-fold CV), so this is a split of their own construction (or inherited from some prior work, uncited). Whether their 217-video test set equals our official-protocol test fold is unverifiable without their code.
- No cross-validation. IHV used for classification-only transfer evaluation.

### 2. Temporal resolution
- Videos processed in **30-second chunks**; audio as mono 16 kHz WAV per chunk; frames via scene-change keyframes (~1/scene), fallback uniform sampling at ~1 fps, max 24 frames/chunk.
- The model **emits timestamp intervals** (XML `<timestamps>` spans per chunk), not per-frame scores. GT = HateMM/MHC annotated hateful segments. Nothing is rasterized to a frame grid; evaluation is interval-vs-interval.
- Evaluation logic (Appendix A): "if a model predicts multiple segments, we compute the IoU for each against all ground-truth intervals and assign the maximum overlap score. Segments spanning the 30-second chunk boundary are treated as truncated intervals."

### 3. Metrics and headline numbers
- Classification: Acc, Macro-F1 (+ Weighted-F1 for MHC/IHV) — chunk predictions aggregated to video level.
- Temporal localization: **Avg IoU** between predicted and GT intervals, and **Acc@0.5** (fraction with IoU > 0.5). Computed **exclusively on positive (hateful) instances** — they explicitly show that including negatives inflates scores ("no timestamps" counted as perfect).
- Target ID: Avg F1 over target labels + Exact Match.
- Table 4 additionally reports **mAP** vs MultiHateLoc, but the mAP definition (IoU thresholds, averaging) is **never specified**.
- No frame-level ROC-AUC or AP anywhere. Error bars over 3 seeds {42, 108, 420}.

**Table 3 (TANDEM results), key rows verbatim** (columns: Acc, F1(M), F1(W), Avg IoU, Acc@0.5, Avg F1 targets, Exact Match; ± over 3 seeds):

HateMM (Binary):
- SFT only (100 videos, Qwen-Omni silver data): VL 0.73±0.08, 0.62±0.05, –, –, –, 0.23±0.03, 0.17±0.04; AL 0.78±0.08, 0.79±0.06, –, 0.18±0.03, 0.11±0.02, 0.71±0.11, 0.58±0.13
- SCCR+GSPO (no SFT): VL 0.73±0.08, 0.59±0.05, –, –, –, 0.29±0.04, 0.26±0.06; AL 0.71±0.07, 0.70±0.06, –, 0.32±0.06, 0.29±0.05, 0.29±0.04, 0.16±0.04
- SFT+SCCR+GSPO: VL 0.78±0.08, 0.73±0.06, –, –, –, 0.55±0.08, 0.48±0.11; AL 0.77±0.08, 0.78±0.06, –, 0.18±0.03, 0.08±0.01, 0.73±0.11, 0.57±0.13
- SFT+SCCR+GRPO: VL 0.71±0.07, 0.54±0.04, –, –, –, 0.28±0.04, 0.23±0.05; AL 0.75±0.08, 0.76±0.06, –, **0.43±0.08, 0.31±0.06**, 0.73±0.11, 0.59±0.13

MHC-en (Multiclass), best config SFT+SCCR+GSPO: VL 0.59±0.06, 0.38±0.03, 0.57±0.05, 0.13±0.02, 0.09±0.02, 0.29±0.04, 0.27±0.06; AL 0.67±0.07, 0.32±0.03, 0.54±0.04, 0.07±0.01, 0.07±0.01, 0.19±0.03, 0.18±0.04

IHV (transfer, classification only), SFT+SCCR+GRPO/GSPO (trained on HateMM or MHC): VL 0.64±0.07, 0.54±0.04, 0.59±0.05.

Note the VL model omits temporal segmentation on HateMM ("it degraded the classification and target identification performance"); on MHC the VL model does the timestamps instead of AL. So the localization numbers come from different modality branches per dataset.

**Table 4 verbatim** (vs MultiHateLoc, Sun et al. 2025):

| Dataset | Method | Acc | F1(M) | F1(W) | mAP | Avg F1 |
|---|---|---|---|---|---|---|
| HateMM | MultiHateLoc | – | – | – | 0.645 | – |
| HateMM | TANDEM (ours) | 0.78 | 0.79 | – | **0.71** | 0.77 |
| MHC-en | MultiHateLoc | – | – | – | 0.445 | – |
| MHC-en | TANDEM (ours) | 0.67 | 0.38 | 0.57 | **0.62** | 0.69 |

The MultiHateLoc numbers are quoted from Sun et al. 2025 (mAP 0.645 HateMM / 0.445 MHC), so cross-paper protocol/split alignment for Table 4 is unverified, and TANDEM's own mAP recipe is undefined in the paper.

Zero-shot baselines (Table 2, HateMM binary, Avg IoU / Acc@0.5): Gemini-2.5-Flash (A+V) 0.46/0.47; Qwen3-Omni-30B-A3B-Thinking 0.53/0.55; Qwen2.5-VL-7B (V) 0.09/0.04; Qwen2-Audio-7B 0.00/0.00. Note the ZS baselines beat trained TANDEM on Avg IoU (0.53 ZS vs 0.43 best TANDEM) — the paper's localization win over its own baselines is on target ID and classification, not IoU.

### 4. Supervision
**Span-supervised at train time.** Two stages: (i) SFT on 100 videos with silver structured outputs generated by Qwen3-Omni-30B-A3B-Thinking, filtered to keep only samples whose classification label matches ground truth; (ii) LoRA + GRPO/GSPO RL on the **full training sets**, with a composite reward containing lambda_tau * IoU(pred timestamps, **ground-truth timestamps**) and lambda_z * F1(pred targets, ground-truth targets). So ground-truth temporal spans and target labels from the train split drive the reward — this is supervised temporal grounding, not weak supervision.

### 5. Models and inference cost
- **Qwen2.5-VL-7B-Instruct** (vision-language) + **Qwen2-Audio-7B-Instruct** (audio-language), LoRA rank 8 / alpha 16, lr 5e-5, batch 2/GPU, 384-token generations, G=4 samples, trl GRPO (KL penalty 0). Cross-modal context limited to 1 chunk.
- Both 7B MLLMs run at inference on every 30 s chunk (plus a zero-shot SCCR pass by the frozen counterpart). Training compute: **2x A100 80GB, 200 steps, ~6 days for HateMM, ~72 h for MHC**. Inference cost not quantified (order: two 7B forward generations per 30 s chunk).

### 6. Within-video / positive-fraction analysis
Localization metrics are inherently **per-video (macro over hateful test videos)** — Avg IoU is a mean over positive instances. But this is interval IoU, not any frame-score metric; there is no pooled anything and no frame ROC/AP. The positives-only choice is their (correct) guard against negative-instance inflation.

### 7. Baselines
Zero-shot: Gemini-2.5-Flash, Qwen2.5-VL-7B, Qwen2-Audio-7B, Qwen3-Omni-30B-A3B-Thinking, context-augmented Qwen2.5-VL variants. Quoted supervised classification: Yue et al. 2025, Koushik et al. 2025, Rehman et al. 2025, Wang et al. 2024 (GPT-4V). Localization: **MultiHateLoc (Sun et al. 2025)** — quoted, not rerun. **No VadCLIP, no MACIL-SD, no LAVAD, no LELA** (the two papers do not cite each other).

### 8. Code availability
**Not available.** Reproducibility checklist, verbatim: "Did you include the code, data, and instructions needed to reproduce the main experimental results...? **No. It will be released upon paper's acceptance.**" As of 2026-08-30 no repository is findable.

### TANDEM comparability verdict
**Incommensurable with our benchmark as reported — different metric family, different label granularity, unverifiable splits.** They report per-video interval Avg IoU / Acc@0.5 on hateful test videos plus an undefined mAP; we report pooled 1 fps frame ROC/AP and within-hate macro ROC. Their HateMM test split (217 videos) is self-constructed (HateMM has no official split; original protocol is 5-fold CV), and the MHC-en test cohort size is not even stated. There is no continuous frame score to feed a ROC/PR curve: the model emits hard intervals, so even a rasterization of their outputs to our 1 fps grid yields a single operating point (frame precision/recall/F1), not ROC-AUC/AP.

**What a rerun under our protocol requires:** (a) full reimplementation (no code): chunking pipeline, XML prompting, SFT-data generation via Qwen3-Omni-30B, LoRA+GRPO tandem training — ~2x A100-80GB for ~6 days per dataset, i.e., outside our SLURM quota comfort zone and not Modal-compatible (raw video); (b) retraining on OUR split definition; (c) a score-ization convention for ROC/AP (e.g., token-level confidence or interval-membership as {0,1} — any choice is ours, not theirs, and must be labeled as such). Realistic alternative: place TANDEM in the related-work table qualitatively (supervised, interval-output, per-video IoU protocol) and, if a head-to-head is demanded, evaluate OUR method under THEIR metric (rasterized spans → interval IoU on positives) which is cheap, rather than the reverse.

---

## Cross-cutting conclusions for our benchmark table

1. **Neither paper's numbers can be transplanted into our results table.** LELA fails on cohort/grid underspecification plus an unresolved ROC/PR column swap; TANDEM fails on metric family and split provenance. Any table containing our pooled 1 fps frame ROC/AP next to their printed numbers would be comparing different quantities.
2. **Neither has public code** (LELA: none, ever mentioned; TANDEM: promised on acceptance). Both reruns are reimplementations from prose.
3. **Nobody else reports a within-video (per-hate-video macro) frame metric.** LELA is pooled-only; TANDEM's per-video quantities are interval IoU. Our within-hate macro ROC has no published competitor number — that is a differentiation point, not a comparability gap.
4. **Cheapest honest head-to-heads:** (i) our method evaluated under TANDEM-style interval IoU (rasterize our frame scores → threshold → intervals) — near-zero cost; (ii) a "LELA-style" GPT-4o-Mini prompting baseline over our own caption caches at our 1 fps grid on official test splits — moderate API cost, and it inherits our protocol by construction; (iii) LAVAD and Lin et al. are the shared training-free reference points both we and LELA can cite, but LELA's LAVAD numbers are on their unknown cohort, so ours must be rerun locally too.
5. **Adjacent numbers worth tracking:** MultiHateLoc (Sun et al. 2025): mAP 0.645 HateMM / 0.445 MHC (weakly supervised, top-K MIL, quoted in TANDEM); HateClipSeg (Wang et al. 2025) exists as the segment-level annotation source both communities are converging on.
