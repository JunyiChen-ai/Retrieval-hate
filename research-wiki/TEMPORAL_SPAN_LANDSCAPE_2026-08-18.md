# Hateful video temporal span detection / localization — landscape, 2026-08-18

Pure literature reconnaissance plus read-only re-measurement of annotation files already on disk.
**Zero GPU, zero training, zero test-label contact.** All external claims carry a verification tag:

- `[read-method]` — read the paper body (PDF/HTML), not just the abstract
- `[read-abstract]` — read the abstract page only
- `[title-only]` — the record exists (index hit, citation) but the content was not read

Everything marked `[measured]` was computed here, on this machine, from annotation files listed
in §1.6, by the snippets recorded in §8.

---

## 0. Executive summary

1. **The benchmark that everyone reports on is degenerate.** On HateMM the annotated hate span
   covers a **median 0.806** of the video and is a **single contiguous block in 72.8%** of hateful
   videos `[measured]`. An oracle that has a perfect *video-level* classifier and **no localization
   ability at all** — it marks every second of every hateful video as hateful — scores
   **frame-level AP 0.675** on HateMM `[measured]`. The published weakly-supervised
   state of the art on HateMM frame-level localization, MultiHateLoc, reports **mAP 0.645**
   `[read-method]`. Under our 1 fps convention the degenerate oracle is *above* the published number.
2. **MultiHateClip is worse.** Its released `Duration` column *is* a temporal span annotation
   (this is under-advertised), but the span is a single interval in **95.8% (EN) / 98.2% (ZH)** of
   annotated videos and covers a **median 0.937 (EN) / 1.000 (ZH)** of the video `[measured]`.
   The same degenerate oracle scores **frame-AP 0.786 (EN) / 0.853 (ZH)**; MultiHateLoc reports
   **0.445** on MultiHateClip.
3. **HateClipSeg is the only hateful-video benchmark where frame/segment localization is a real
   task.** Toxic coverage median **0.544**, single-block only **22.0%**, median **3** toxic blocks
   per video, degenerate-oracle frame-AP **0.530** `[measured]`. Its own paper reports segment
   localization at roughly **59.4 mAP@tIoU 0.3 falling to ~29 at 0.7**, i.e. a genuinely open task.
4. **The field is five methods wide.** Exhaustive arXiv search on `all:"hateful video"` returns 14
   papers total; exactly **five** emit a time interval or per-segment label: MultiHateLoc (weakly-
   supervised MIL), LELA (training-free LLM), TANDEM (MLLM + RL timestamps), the HateClipSeg
   ActionFormer/LSTR baselines (supervised TAL/online ports), and StreamSense (streaming). Nothing
   is proposal-based as a method contribution, nothing is DETR-style, nothing is query-conditioned,
   nothing is audio-first, and **no two published HateMM localization numbers are comparable** —
   five methods, five different metrics, no shared protocol.
5. **The best numbers are not close to usable.** HateClipSeg proposals: 52.65 F1@tIoU 0.5, 30.99
   @0.7, at precision ~40% — roughly half of what the same ActionFormer architecture does on
   THUMOS14. MLLM timestamping: best HateMM Avg IoU 0.43 for a trained model, **losing to an
   untrained Qwen3-Omni zero-shot at 0.53**. And the metric choice hides this: one WSVAD model
   (VadCLIP) reports UCF-Crime **frame-AUC 88.02 alongside mAP@tIoU 6.68** — frame-level AUC, the
   metric MultiHateLoc and LELA both use, can be high while the system cannot propose an interval.
6. **This project has already spent four routes on this axis and all four came back negative**
   (multi-granularity segment retrieval, segment-keyed purity loop, MLLM weak-supervision P11,
   TERA Gate-0). The mechanism is now visible in the data, not just in the results: **video-level
   labels already contain almost everything the segment signal would add**, because the span is the
   video. P11 measured exactly this — a plain video-label MIL head reaches wv-AUC 0.553, a 72B MLLM
   reading frames and ASR reaches 0.591, and the gap is not significant.
7. **Recommended posture: do not enter this sub-direction as a localization-performance play.**
   The highest-value contribution available here is a *measurement* correction (report coverage,
   single-block fraction, and a broadcast control; unify the protocol) and the project's standing
   method-papers-only rule closes it. The one method-shaped candidate is **localize → trim →
   re-classify with predicted boundaries**, which is empty in the literature, carries a +19 to +30
   macro-F1 *oracle* headroom, and should be attacked with a one-hour CPU kill probe first (§7.4).
   Any localization work proper would have to move to HateClipSeg or DeHate — a scope change
   requiring a user ruling (§7.5).

---

## 1. Datasets with temporal annotation

### 1.1 Inventory

Only **five** hateful/harmful-video datasets carry usable temporal annotation. Three are obtainable
and parseable today (HateMM, HateClipSeg, MultiHateClip); DeHate ships its annotations openly but
gates the media; ImpliHateVid's promised frame spans were never released.

| # | Dataset | First author / venue | Verified ID | Videos (hateful) | Temporal granularity | Task the authors define on it | Public | Verified |
|---|---|---|---|---|---|---|---|---|
| 1 | **HateMM** | Das, ICWSM 2023 | arXiv **2305.03915**; Zenodo `10.5281/zenodo.7799469` | 1,083 (431 hate), 43.26 h | continuous second-level `hate_snippet` spans (`HH:MM:SS` lists), hateful videos only | **none** — video-level binary; spans released as unused "rationales" | ungated, CC-BY-4.0, **ships raw video** (6.3 GB) | `[read-method]` + annotation file parsed twice, independently |
| 2 | **HateClipSeg** | Wang, ACM MM 2025 | arXiv **2508.01712**; DOI **10.1145/3746027.3758289** | 435 (380 with ≥1 offensive segment; 194 with ≥1 *hateful* segment), 28.78 h | **variable-length segments partitioning the whole video**, mean 8.84 s, multi-hot 6-class | **(1) trimmed classification (2) Temporal Hateful Video Localization (3) online classification** | ungated GitHub, **video IDs only** | `[read-method]` + annotation file parsed twice |
| 3 | **MultiHateClip (MHC)** | Wang, ACM MM 2024 | arXiv **2408.03468**; DOI **10.1145/3664647.3681521** | 2,000 (EN 82 H / 257 O; ZH 128 H / 196 O), ~18 h | **continuous second-level spans in the `Duration` column**, `[(start,end)]` integer seconds — not visible from the paper, confirmed in the released TSVs | video-level 3-class only; **no temporal task** | ungated GitHub, video IDs only + `video_download.py` | `[read-method]` + annotation file parsed twice |
| 4 | **DeHate** | Zhang, ACM MM 2025 | **no arXiv** (see §1.4); DOI **10.1145/3746027.3758272** | **6,689** (1,170 explicit + 950 implicit hate; 4,569 non) TikTok + BitChute | segment start/end timestamps + modality attribution + target group; official train/val/test split | dataset paper; video-level benchmarks only | **annotations ungated** (`DeHate.xlsx` on GitHub); **raw video gated** behind an application form | `[read-abstract]` + `[measured]` on the released xlsx |
| 5 | **PCLMM** | Wang, ICME 2024 | arXiv **2409.05005** | 715 Bilibili (196 PCL), Chinese | **PCL facial frame spans** — expression-scoped, not evidence-scoped | binary PCL detection; **no localization task** | availability not stated on the abstract page | `[read-abstract]` |
| — | ImpliHateVid | Rehman, ACL 2025 | arXiv **2508.06570**; DOI **10.18653/v1/2025.acl-long.842** | 2,009 (509 implicit + 500 explicit), 86.5 h | protocol says annotators marked frame spans, **but no span file is in the release and no span statistic is reported** — video-level in practice `[measured]` on local gt | video-level 3-class | **gated** (institutional email + signed agreement) | `[read-method]` |
| — | Ex-HateMM / Ex-ImpliHateVid (IARE) | Lu, SIGIR 2026 | arXiv **2606.11953** | 1,070 (419 hate) / 2,005 (1,007 hate) | **no temporal annotation** — text rationales (mean 72.7 / 60.9 tokens) + harmful-element tags | explainable detection | GitHub `DUT-lujunyu/IARE`, CC BY-NC-ND | `[read-method]` |
| — | ADIMA | Gupta, ICASSP 2022 | arXiv **2202.07991** | 11,775 audio clips, 65 h, 10 Indic languages | **whole-clip binary, audio-only, no video** | clip classification | public | `[read-abstract]` |
| — | OffVidPT | Alcântara, LREC 2020 | ACL `2020.lrec-1.531` | 400 PT YouTube videos | whole-video; ships text/statistical features only | classification | public | `[read-abstract]` |
| adj | XD-Violence | Wu, ECCV 2020 | — | 4,754 untrimmed, 217 h | frame-level GT on the **800-video test split only** | weakly-supervised violence detection | public (features + URLs) | `[title-only]` |
| adj | UCF-Crime | Sultani, CVPR 2018 | — | 1,900, 128 h | frame-level GT on the **290-video test split only** | weakly-supervised anomaly detection | public | `[title-only]` |

Two corrections to the task brief's lead list: **MultiHateLoc is a method, not a dataset** (arXiv
2512.10408, WWW 2026; trains on HateMM + MHC), and **arXiv 2606.11953 is a real ID but not a
localization paper** — it resolves to Lu et al., *Decoding Multimodal Cues: Unveiling the Implicit
Meaning Behind Hateful Videos* (SIGIR 2026), whose "fine-grained annotations" are text rationales
with no timestamps. Also **TANDEM is accepted to AAAI-ICWSM 2027**, not 2026.

### 1.2 Span-distribution statistics — the load-bearing table `[measured]`

This is the table the sub-direction is usually missing. All rows computed here from the released
annotation files, 1 fps convention, span union per video.

| Dataset (label set) | annotated videos | span coverage median | coverage mean | single-block fraction | median blocks/video | median block length |
|---|---|---|---|---|---|---|
| **HateMM** (hateful) | 430 | **0.806** | 0.693 | **72.8%** | 1 | 21.0 s |
| **MHC-EN** (any annotated) | 245 w/ duration | **0.937** | 0.785 | **95.8%** | 1 | 18.0 s |
| **MHC-EN** (Hateful only) | 62 | 0.970 | — | — | 1 | — |
| **MHC-ZH** (any annotated) | 262 w/ duration | **1.000** | 0.878 | **98.2%** | 1 | 21.0 s |
| **MHC-ZH** (Hateful only) | 84 | 1.000 | — | — | 1 | — |
| **HateClipSeg** (hateful only) | 180 | 0.418 | 0.463 | 31.1% | 3 | 8.1 s |
| **HateClipSeg** (hateful+insulting) | 312 | 0.512 | 0.500 | 24.4% | 3 | 8.1 s |
| **HateClipSeg** (any toxic) | 345 | **0.544** | 0.531 | **22.0%** | 3 | 8.1 s |

Tail behaviour: HateMM has coverage ≥ 0.90 on **33.3%** of hateful videos and ≥ 0.99 on 5.8%;
MHC-ZH has coverage ≥ 0.99 on **75.6%** of annotated videos and on **88.1%** of Hateful videos.
HateMM's span-count tail runs to 19 spans on one video; per-span length is strongly right-skewed
(mean 57.3 s vs median 21.0 s).

**Independent reproduction.** Every row of this table was computed twice, by two workers with no
shared code, one on the local `hate_spans.json` / `gold_segments.json` and one re-parsing the
upstream CSV/TSV releases from scratch. Agreement: HateMM single-span 72.8% vs 72.9%;
HateClipSeg offensive coverage mean 0.531 vs 0.530, single-block 22.0% vs 21.6%; MHC-EN single-span
95.8% vs 95.8%; MHC-ZH 98.2% vs 98.2%; MHC-ZH hateful coverage median 1.000 vs 1.000. The small
residuals are the 395/435 vs 435/435 HateClipSeg subset and the 430 vs 431 HateMM zero-length drop.

**Cross-check against what the papers do report.** HateMM's paper (Table 1) gives mean hate-video
length 2.56 ± 1.69 min and mean rationale length 1.71 ± 1.27 min, i.e. a mean coverage of ~0.67 —
consistent with the mean 0.693 measured here. HateClipSeg's paper reports mean segment length
8.84 s and "87% of videos contain at least one offensive segment", but reports **neither coverage
nor blocks-per-video**. MHC's paper reports mean span length by class (EN 33.06 s hateful / 23.84 s
offensive; ZH 24.82 / 27.20) against mean video lengths of 33.78 s (EN) and 31.78 s (ZH) — a
hateful-class coverage ratio of ≈0.98 (EN) and ≈0.78 (ZH) is derivable from the paper's own
numbers, but is never stated.
**No paper in this field reports the coverage fraction, the single-block fraction, or a degenerate
baseline.** That absence is the finding.

**Segmentation provenance matters.** HateClipSeg's segment boundaries are *automatic* (Whisper word
timestamps + NLTK Punkt sentence boundaries + ViT scene detection with a 20 s silent-interval
threshold), then human-labelled. They are semantic boundaries, not a fixed grid, and the human
judgement is the label, not the boundary. HateMM's and MHC's spans are drawn by hand.

### 1.3 What a zero-localization oracle scores `[measured]`

Predictor: *score = 1 for every second of every video the gold says is hateful/toxic, 0 elsewhere.*
It has a perfect video-level classifier and no temporal resolution whatsoever. Ties broken at random.

| Dataset | 1 fps positive base rate | zero-localization oracle frame AP | best published frame mAP |
|---|---|---|---|
| HateMM (all 1,083 videos) | 0.2869 | **0.675** | MultiHateLoc 0.645 `[read-method]` |
| MHC-EN (793 videos w/ local media) | 0.2466 | **0.786** | MultiHateLoc 0.445 (EN+ZH pooled?) `[read-method]` |
| MHC-ZH (814 videos) | 0.2539 | **0.853** | " |
| HateClipSeg (395-video subset, any-toxic) | 0.4638 | **0.530** | see §2 |

**Honest caveat, stated up front.** MultiHateLoc does not specify its frame definition, sampling
rate, split, pooling rule, or whether non-hateful videos enter the frame pool; the same gap was
already recorded in `research-wiki/EVAL_localization_hatemm.md §4`. The oracle numbers above use
*our* 1 fps convention on the *full* corpora. They are therefore an **indicative** argument that the
HateMM/MHC frame-mAP protocol is dominated by video-level separability, **not** a like-for-like
refutation of a specific published number. The argument does not depend on the exact protocol: the
span-distribution table in §1.2 is protocol-free, and it is the reason the oracle scores what it does.

Corroboration from inside this project, on a protocol we do control: on HateMM,
a plain **video-level probability broadcast** control reaches frame-level AP 0.5776 / AUC 0.7735,
while the actual segment-scoring model reaches 0.5892 / 0.7813 — a **+0.012 AP** difference
(`EVAL_localization_hatemm.md §3`). Almost all of a HateMM frame-level score is video-level
classification wearing a timeline.

### 1.4 Identifier verification `[verified against the arXiv API, title-scoped queries]`

| name | identifier | note |
|---|---|---|
| HateMM | arXiv **2305.03915v1** | exactly 1 title match |
| MultiHateClip | arXiv **2408.03468v2** | exactly 1 |
| HateClipSeg | arXiv **2508.01712v2** | exactly 1; 6 pages, submitted 2025-08-03, revised 2025-08-15 |
| MultiHateLoc | arXiv **2512.10408v3** | WWW 2026, DOI 10.1145/3774904.3793032 |
| LELA | arXiv **2602.09637v1** | 2026-02-10 |
| TANDEM | arXiv **2601.11178v3** | |
| StreamSense | arXiv **2601.22738v1** | WWW 2026 |
| Temporal label noise (Yang) | arXiv **2508.04900v1** | MUWS@MM 2025 |
| ImpliHateVid | arXiv **2508.06570v2** | video-level only |
| IARE / Ex-HateMM | arXiv **2606.11953** | SIGIR 2026; **video-level, no temporal output** — the ID in the task brief is real but the paper is not a localization paper |
| **DeHate (the dataset)** | **no arXiv ID**; DOI 10.1145/3746027.3758272 | ⚠ **name collision**: arXiv 2509.21787 "DeHate: A Stable Diffusion-based Multimodal Approach to Mitigate Hate" is a **different paper**. Do not cite it for the dataset. |

### 1.5 Leads from the task brief, resolved

- **Ex-HateMM / IARE (2606.11953)** — real paper, **no temporal annotation**; the "fine-grained"
  part is text rationales. Not a localization asset.
- **PCLMM (2409.05005)** — has "PCL facial frame spans", but they mark facial *expressions*, not
  hate evidence, and the paper defines no localization task. Public availability not stated on the
  abstract page. Low priority.
- **ImpliHateVid** — the annotation protocol mentions frame spans; **the release contains none**,
  and no span statistic is reported. Confirmed video-level on the local copy.
- **DeHate** — real and large, but see §2.6.1: 46% of its hateful videos have no usable span.
- **"MultiHateLoc" as a dataset** — it is a method. **"HateClipSeg is segment-level"** — correct.
  **"HateMM has hate span annotations"** — correct, and they are the field's most-used and
  least-examined spans.

### 1.6 Files read for §1.2–1.3 and §2.6.1

- `data/gt/HateMM/HateMM_annotation.csv` (Zenodo 7799469), `data/gt/HateMM/hate_spans.json`
- `data/gt/HateClipSeg/gold_segments.json` (395-video surviving subset; see `DATASET_hateclipseg.md §4` for the selection-bias statement)
- MultiHateClip `{English,Chinese}_data/annotation/{train,valid,test}.tsv` fetched from GitHub main
- `DeHate.xlsx` fetched from `Multimodal-Intelligence-Lab-MIL/DeHate`
- local media durations via `ffprobe` on `data/_src_Multihateclip/{English,Chinese}/video/`
- `data/gt/ImpliHateVid/*.jsonl` (no temporal fields)
- `logging/runs/gate_c_annotation/claude_c1_rows.jsonl` (for §5.1)

### 1.7 Correction to a prior internal record

`research-wiki/EVAL_localization_hatemm.md §1` (2026-07-03) reports **427 hateful videos, 671
segments, coverage median 0.459, single-segment 268/427 = 62.8%**, and concludes that HateMM
frame-level localization "is feasible and non-trivial". The released CSV contains **431 videos and
786 raw segments** `[measured]`; the current `hate_spans.json` (regenerated 2026-08-07) parses to
430 / 784 with coverage median **0.806** and single-segment **72.8%**. The July parse under-counted
by ~115 segments. **The conclusion drawn from it does not survive.** Per the project's
no-documentation-iteration rule this is recorded here and fixed on sight; it is not a blocking item
and did not gate any experiment (the July doc's own §3 numbers are unaffected — they were computed
from features, not from the span parse).

---

## 2. Methods and numbers

### 2.0 How small the field is

Exhaustive arXiv metadata search on `all:"hateful video"` returns **14 papers in total**;
`abs:"hate" AND abs:"temporal localisation"` returns **1**. Cross-checked against HuggingFace
papers search (~25 queries), DBLP at author level on the two groups that own this axis
(Zeyu Fu / Exeter, Roy Ka-Wei Lee / SUTD), and Semantic Scholar forward-citations of MultiHateLoc
and HateClipSeg. **Exactly five methods** emit a time interval or per-segment label for hate in
video. Three further datasets carry span annotations that nobody has run localization on.

### 2.1 The five methods

| # | Method | ID / venue | Framing | Supervision | Verified |
|---|---|---|---|---|---|
| 1 | **MultiHateLoc** | arXiv 2512.10408v3, WWW 2026, DOI 10.1145/3774904.3793032 | frame-score curve, **MIL ported from weakly-supervised video anomaly detection** | video-level labels only | `[read-method]` |
| 2 | **LELA** | arXiv 2602.09637v1 (2026-02) | **training-free LLM prompting**, per-frame score; extends LAVAD (training-free VAD) to hate | none | `[read-method]` |
| 3 | **TANDEM** | arXiv 2601.11178v3 (2026-01) | **MLLM emits `<timestamps>`**, RL with an IoU reward | 100-video SFT + RL on span labels | `[read-method]` |
| 4 | **HateClipSeg baselines** | arXiv 2508.01712v2, ACM MM 2025 | **direct TAL port (ActionFormer)** + **online action detection port (LSTR)** | **fully supervised on spans** | `[read-method]` |
| 5 | **StreamSense** | arXiv 2601.22738v1, WWW 2026 | **streaming per-timestamp classification** with selective VLM escalation and deferral | segment labels | `[read-method]` |

Pipelines, one line each:

1. **MultiHateLoc** — ViT-B/16 frames + VGGish audio + sentence-split Whisper→BERT text, three
   modality-specific Transformer temporal encoders, per-timestep sigmoid modality gating, cross-modal
   attention, per-frame sigmoid head; loss = modality-aware top-K MIL + 0.1·smoothness +
   0.2·frame-aligned cross-modal InfoNCE. Adaptive K = top 33% of frames.
   **Code still not released** — the official repo `Multimodal-Intelligence-Lab-MIL/MultiHateLoc`
   contains a `LICENSE` file and nothing else as of its last push, 2026-01-28 `[verified via GitHub
   API]`. The 2026-07-03 project ruling in `EVAL_localization_hatemm.md §0` ("empty repo → do not
   reproduce") therefore still holds.
2. **LELA** — decompose the video into five caption modalities (image caption / ASR / OCR / music /
   video context), multi-stage GPT-4o-mini prompting (contextualization → rationale → decision),
   composition matching, per-frame score. No training.
3. **TANDEM** — Qwen2.5-VL-7B + Qwen2-Audio-7B with LoRA, structured XML output containing a
   `<timestamps>` field; SFT on 100 curated videos, then "tandem" GRPO/GSPO RL where the two models
   condition on each other, reward = class CE + IoU(τ̂,τ*) + target/format terms. 3 seeds.
4. **HateClipSeg baselines** — frozen ViT-L (visual) + BERT-Base over [t−2s,t] + Wav2Vec-Emotion over
   [t−4s,t]; ActionFormer per modality at 4 FPS, multimodal by late fusion; LSTR with 32-s context
   and 0.25-s stride for the online task.
5. **StreamSense** — lightweight streaming encoder for most timestamps, selective routing to a
   Llama-3.2 VLM expert or **deferral** when context is insufficient; encoder trained with a
   cross-modal contrastive term and an **IoU-weighted cross-entropy** that down-weights windows
   overlapping the target segment poorly.

### 2.2 Numbers

**MultiHateLoc, frame-level mAP / AUC** (Table 1; all baselines self-reproduced on the same features):

| Model | Modality | HateMM | MultiHateClip |
|---|---|---|---|
| VAD-CLIP | V / A / T | 0.531/0.740 · 0.563/0.762 · 0.498/0.712 | 0.348/0.605 · 0.405/0.650 · 0.367/0.610 |
| Early Fusion | V+A+T | 0.565 / 0.765 | 0.410 / 0.662 |
| Late Fusion | V+A+T | 0.578 / 0.779 | 0.401 / 0.660 |
| CMFusion | V+A+T | 0.596 / 0.763 | 0.420 / 0.672 |
| **MultiHateLoc** | V+A+T | **0.645 / 0.799** | **0.445 / 0.750** |

Incremental ablation on HateMM: 0.565 → +MA-TE 0.581 → +DCM-Fusion 0.615 → +CM-Contrast 0.621 →
+MA-MIL **0.645**. No per-tIoU AP, no interval output, no seeds or CIs.

**LELA, frame-level PR-AUC / ROC-AUC** (protocol stated as "follows LAVAD"):

| Method | HateMM | MultiHateClip |
|---|---|---|
| ZS-CLIP / ImageBind / LLaVA-1.5 | 0.522/0.537 · 0.524/0.568 · 0.533/0.553 | 0.518/0.545 · 0.514/0.575 · 0.532/0.544 |
| LAVAD | 0.578 / 0.616 | 0.587 / 0.630 |
| **LELA** | **0.7264 / 0.6756** | **0.7227 / 0.6733** |

**TANDEM, Avg IoU / Acc@0.5, computed on positive instances only:**

| setting | HateMM | MHC-en |
|---|---|---|
| Qwen2.5-VL-7B zero-shot (V) | 0.09 / 0.04 | — |
| Gemini-2.5-Flash zero-shot (A+V) | 0.46 / 0.47 | 0.07 / 0.06 |
| Qwen3-Omni-30B zero-shot (A+V) | **0.53 / 0.55** | **0.13 / 0.15** |
| TANDEM SFT only | 0.18±0.03 / 0.11±0.02 | — |
| TANDEM SCCR+GSPO (no SFT) | 0.32±0.06 / 0.29±0.05 | — |
| **TANDEM SFT+SCCR+GRPO (best)** | **0.43±0.08 / 0.31±0.06** | 0.13±0.02 / 0.09±0.02 |

The trained model does not beat the untrained zero-shot baseline on Avg IoU. TANDEM's abstract
headline (0.73 F1, "30% improvement") is the **target-identification** metric, not localization.

Its Table 4 does claim a localization win over MultiHateLoc — **mAP 0.71 vs 0.645 on HateMM and
0.62 vs 0.445 on MHC-en** — and two independent readers of the paper confirm those digits are
printed there. **The comparison is not like-for-like and should not be propagated.** TANDEM
evaluates in 30-second chunks aggregated to video level, on positive instances only; MultiHateLoc
scores every frame over the whole corpus. Neither paper defines "mAP" compatibly with the other,
and a 0.71 mAP cannot be reconciled with the same system's Avg IoU of 0.43 on the same data.
Treat 0.71 / 0.62 as *reported but not comparable*. Code not released.

**HateClipSeg, F1@tIoU for the Offensive class** (proposal-level; note F1@tIoU, not mAP):

| tIoU | V | T | A | V,T,A |
|---|---|---|---|---|
| 0.30 | **59.38** | 44.49 | 40.21 | 58.98 |
| 0.50 | **52.65** | 34.60 | 25.40 | 50.92 |
| 0.70 | **30.99** | 11.89 | 18.83 | 29.42 |

Visual-only beats late-fusion multimodal at **every** threshold. Precision runs 22–46% against
recall 41–85%: the detector floods proposals. Same paper: trimmed classification 69.48 Macro-F1
(LLaMA-3.2-11B LoRA, V+T); online 62.75 Macro-F1 (LSTR, V+T+A).

**StreamSense, HateClipSeg online per-timestamp Acc / Macro-F1:**

| Model | Latency | GPU | Acc / M-F1 |
|---|---|---|---|
| OadTR / LSTR | 0.1 s | 2 GB | 63.04/62.48 · 63.21/**62.75** |
| Stream Encoder alone | 0.1 s | 2 GB | 64.01 / 63.94 |
| Qwen2.5 / LLaVA-Next / Llama-3.2 (VLM-only) | 0.7–1.1 s | 20–27 GB | 67.82/67.82 · 68.52/67.95 · 68.51/68.51 |
| **StreamSense (Llama-3.2)** | 0.3 s | 29 GB | **72.10 / 72.06** |

+9.31 M-F1 over LSTR at 10±3% VLM invocation and 17±3% deferral. The IoU-weighted CE alone is
worth ~1.2 M-F1 (β=0 → 62.71, β=1.0 → 63.94).

### 2.3 Current best per benchmark

| Benchmark | Metric | Best | Held by |
|---|---|---|---|
| HateMM | frame mAP / AUC, weakly supervised | 0.645 / 0.799 | MultiHateLoc |
| HateMM | frame PR-AUC / ROC-AUC, training-free | 0.7264 / 0.6756 | LELA |
| HateMM | Avg IoU / Acc@0.5 on positives | 0.53 / 0.55 | **Qwen3-Omni zero-shot** (beats TANDEM's trained 0.43/0.31) |
| MultiHateClip | frame mAP / AUC | 0.445 / 0.750 | MultiHateLoc |
| MultiHateClip | Avg IoU / Acc@0.5 | 0.13 / 0.15 | Qwen3-Omni zero-shot |
| HateClipSeg | F1@tIoU 0.5 / 0.7, proposals | 52.65 / 30.99 | ActionFormer, visual-only |
| HateClipSeg | online per-timestamp M-F1 | 72.06 | StreamSense |
| HateMM / MHC-en | TANDEM's own "mAP" | 0.71 / 0.62, claimed over MultiHateLoc's 0.645 / 0.445 | **reported but not comparable** — §2.2 |
| **DeHate** | — | **no localization baseline exists** | — |

**Cross-check against the degenerate oracle (§1.3).** Only one of these numbers clears a predictor
with zero temporal resolution: LELA's HateMM PR-AUC 0.7264 against the oracle's 0.675, a margin of
+0.05 on a different protocol. MultiHateLoc's 0.645 does not, and neither does anything reported on
MultiHateClip against 0.786 / 0.853. HateClipSeg's ActionFormer numbers are the only ones measured
with a metric the oracle cannot game at all.

### 2.4 There is no leaderboard

Every one of the five methods invented its own metric (frame mAP/AUC; frame PR-AUC/ROC-AUC;
Avg IoU/Acc@0.5; F1@tIoU; per-timestamp Macro-F1) and its own split. MultiHateLoc and LELA both
claim HateMM frame-level and **disagree in opposite directions**: LELA's AP (0.7264) is above
MultiHateLoc's mAP (0.645) while its ROC-AUC (0.6756) is far below MultiHateLoc's (0.799). They
cannot be on the same ground truth. LELA does not cite MultiHateLoc despite being the same lab two
months later. **No two published localization numbers on HateMM are comparable.** This is the
field's most obvious defect, and §7 discusses why it is nonetheless not an entry for this project.

### 2.5 How far from usable

- **Proposal level is roughly 2× off mature TAL.** ActionFormer gets 52.65 F1@tIoU 0.5 / 30.99 @0.7
  on HateClipSeg's *single* foreground class; the same architecture gets ~71.0 mAP@0.5 on THUMOS14
  with 20 classes, and current TAL SOTA reaches avg mAP ~71.7 (§4). F1@tIoU is also a *laxer*
  metric than mAP — no ranking integration. Precision 40.5% at recall 75.1% means ~2 wrong
  proposals per right one, which is unusable for moderation where the false-positive removal is the
  expensive error.
- **Frame level sits below where weakly-supervised video anomaly detection already is**, with more
  modalities: MultiHateLoc's HateMM frame AUC 0.799 / mAP 0.645 against XD-Violence AP ~85% and
  UCF-Crime frame AUC ~87% for visual-only WSVAD.
- **MLLM timestamping is not working.** Best HateMM Avg IoU 0.43–0.53, MHC 0.13. Moment-retrieval
  systems hit R@1 IoU=0.5 in the 60–70% range on QVHighlights/Charades-STA.
- **The prize is real but so far oracle-only.** Yang et al. (arXiv 2508.04900, MUWS@MM 2025)
  `[read-method]` show HateMM video classification going 79.30 → **98.64** Macro-F1 and MHC-EN
  64.37 → **97.31** when *ground-truth* hate boundaries are used to trim ("+19.34% and +30.45%
  macro F1"). Cross-generalization collapses: noisy→clean 76.91 / 57.63, clean→noisy 63.19 / 47.24.
  Their trimming yields 790 hate (41.36%) vs 1,120 non-hate (58.64%) segments on HateMM, and 332
  (64.84%) vs 180 (35.16%) on MHC-EN; MHC-EN trimmed hate segments average 24.50 s against 9.36 s
  for non-hate segments from the same videos. Code at
  `Multimodal-Intelligence-Lab-MIL/HatefulVideoLabelNoise` (verified populated).
  **Nobody has shown that *predicted* boundaries recover any of that gap.** The closed loop
  localize → trim → classify does not appear in any paper this sweep found.
  (Read §5.2 before getting excited: this is an oracle number, the trimmed task is an easier task
  than the original one, and on HateMM the "boundary" is ≥90% of the video a third of the time.)

### 2.6 Span-annotated but no localization method run on them

- **DeHate** (ACM MM 2025, DOI 10.1145/3746027.3758272, repo
  `Multimodal-Intelligence-Lab-MIL/DeHate`, mirror `yuchenzhang-1/DeHate`) `[read-repo, measured]` —
  6,689 videos with segment timestamps + modality carrier + target group; the released benchmark is
  video-level only (binary best 0.758 Acc / 0.708 M-F1; 3-way best 0.702 / 0.529). **Zero
  localization baselines have been run on it.** Release is a single `DeHate.xlsx` (annotations only,
  no code, no video). Measured directly from that file (§2.6.1).
#### 2.6.1 DeHate span audit `[measured]`

Parsed `DeHate.xlsx` from the official repo (6,689 rows; columns `Video ID, Hate, Explicit or
Implicit, Hate Segment, Textual/Visual/Audio Content, 6 target groups, Platform, title, desc,
Split`). Official split is released: 4,680 train / 668 val / 1,341 test. Platform 5,052 BitChute /
1,637 TikTok. Labels 4,569 NonHate / 1,170 Explicit / 950 Implicit.

The `Hate Segment` column holds a list of `(start,end)` second tuples. Of 2,111 rows with a
non-empty list:

- **976 videos (46.3% of hateful videos) have only zero-length `(0,0)` entries** — no usable span.
- **1,134 hateful videos carry at least one real span**; 1,209 real spans in total.
- **94.7% of those are a single span**; span length median **17.0 s**, mean 34.6 s, P25 8.0, P75 42.0, max 293 s.
- Availability splits sharply by platform and by explicitness: BitChute Explicit 588 real / 467
  none, BitChute Implicit 346 / 489, TikTok Explicit 105 / 10, TikTok Implicit 95 / 20. On the
  long-form platform, **59% of implicit hateful videos have no localizable span at all**.

Interpretation of `(0,0)` — most likely "no specific moment / the whole video" — is **not stated in
the release and was not verified against the paper**; treat as an open question. Either way, DeHate
delivers ~1,134 usable single-span videos, not 6,689, and video durations are not distributed, so
coverage fraction cannot be computed without fetching the media (which is gated behind an
application form; the annotations themselves are not).

The paper's coverage numbers exist only as a **log-scale figure** (`Images/HateSegment_clear.png`),
with no table. Read off the dashed mean lines — **approximate, figure-derived**: mean hate segment
length BitChute ≈ 30 s, TikTok ≈ 20–25 s; mean hate segment **ratio** BitChute ≈ 50%, TikTok
≈ 60–80%. In **both** ratio panels the rightmost bin (≈100% of the video) is the tallest bar. A
companion figure gives hate-segment **start-time** densities by platform, a distribution nobody else
publishes. Exact values require the application form.

- **FineMuSe** (arXiv 2602.15757, Computational Linguistics 2026) `[read-abstract]` — Spanish sexism
  video dataset, annotators selected temporal spans per fine-grained label across text/audio/video;
  evaluation is LLM classification only.
- **MultiHateClip** — spans released in the `Duration` column, no localization task defined (§1).

### 2.7 Adjacent-but-not-video-time

- **ViToSA** (arXiv 2506.00636, INTERSPEECH 2025) `[read-abstract + tables]` — Vietnamese speech
  toxic-span detection, fine-tuned ASR → text span tagger; best ViSoBERT Acc 0.945 / Macro-F1 0.817.
  Output is **character spans in the transcript**, not video time intervals.
- Checked and excluded as video-level only, no temporal output: HVGuard, RAMF (arXiv 2512.02743),
  MARS (2601.15115), IARE (SIGIR 2026, 2606.11953), ImpliHateVid, MM-HSD, CMFusion, SAGE, LEAF,
  HCG-MPB, SafeWatch (2412.06878 — emits timestamps inside free text but is never scored with
  tIoU/mAP), SafeLens (2605.17610), UNIVID (2606.05748).

### 2.8 Data-quality note found in passing

The HateClipSeg paper's ActionFormer citation is garbled ("Chenglin Zhang, Jianbo Wu, and Yifan Li,
2022"); the real authors are Chen-Lin Zhang, Jianxin Wu, Yin Li. Relevant only if this project
ever cites through it.

---

## 3. Task-shape coverage

| task shape | occupied? | by whom |
|---|---|---|
| Weakly-supervised MIL (video label → frame scores), ported from WSVAD | **occupied** | MultiHateLoc |
| Training-free LLM/VLM per-frame scoring, ported from LAVAD | **occupied** | LELA; this project's P6/P10 |
| MLLM that emits timestamps, RL-tuned | **occupied, and failing** | TANDEM |
| Fully-supervised TAL port with proposals + boundary regression | **occupied once, as a dataset baseline** | HateClipSeg / ActionFormer — no follow-up method paper |
| Online / streaming per-timestamp classification | **occupied** | LSTR baseline; StreamSense |
| **Proposal-based detection as a method contribution** (anchor-free heads, boundary regression, TriDet/DyFADet-class architectures tuned for hate) | **empty** | — |
| **DETR-style moment queries / set prediction** for hate segments | **empty** | — |
| **Query-conditioned grounding** — "find the moment attacking group X", or natural-language moment retrieval over a hateful video | **empty** | — |
| **Audio-first / prosody-first localization** (localize on speech energy and prosody, not frames) | **empty** — audio is the weakest single modality in every published table | — |
| **Dense video captioning framing** (emit a caption + interval per hateful event) | **empty** | — |
| **Localize → trim → re-classify closed loop** with *predicted* boundaries | **empty** | Yang et al. did the oracle version only |
| Retrieval / kNN-based localization | **thin** — MultiHateLoc explicitly has no retrieval; this project tried it and got negatives (§5.3) | — |
| **A common protocol / leaderboard** | **empty** (§2.4) | — |

---

## 4. Adjacent fields and transferability

### 4.1 The three questions, answered

**(a) Has anyone applied weakly-supervised temporal localization / MIL anomaly machinery to hateful
video? — Yes. This slot is not blank; it is the field's flagship.** MultiHateLoc is explicitly a
port of weakly-supervised video anomaly detection (WSVAD) MIL to hate: top-K MIL over frame logits,
smoothness regularization, and **VAD-CLIP as its headline baseline**. The task brief's hypothesis
that "video-level label → span in the hate domain is blank" is **false**, and this is the single
most important correction in this report for planning purposes. What *is* thin is a
*reproducible* version — MultiHateLoc's repo is still LICENSE-only (§2.1).

**(b) Has anyone applied grounding/moment-retrieval MLLMs to hate localization? — Partly.** TANDEM
is the only attempt: Qwen2.5-VL-7B + Qwen2-Audio-7B fine-tuned with an IoU-shaped RL reward to emit
`<timestamps>`. The dedicated temporal-grounding MLLM line (TimeChat, VTimeLLM, Momentor, TRACE,
Grounded-VideoLLM) has **not** been brought to hate at all. But note how TANDEM ended: the RL-tuned
model (HateMM Avg IoU 0.43) **lost to an off-the-shelf zero-shot Qwen3-Omni-30B (0.53)**. The
generic grounding ability of a large model is already better than what hate-specific tuning
produced on 100 SFT videos. That is a warning about the transfer, not an invitation.

### 4.2 Reference points from the adjacent literature

> ⚠ **Provenance warning on this table.** These anchors are `[second-hand, unverified]` — they come
> from survey summaries, not from reading the papers. The worker that produced them flagged the
> fabrication risk itself and declined to restate part of the set. **Re-pull every number here
> directly before it enters a pre-registration, a related-work table, or a paper.** The arXiv IDs
> are the reliable part; the digits are the unreliable part. They are used in this report only as
> order-of-magnitude anchors for §2.5, and no decision in §7 turns on any of them.

| family | benchmark | current level | ID |
|---|---|---|---|
| Fully-supervised TAL | THUMOS14 avg mAP@[0.3:0.7] | ActionFormer 66.8 (71.0@0.5); TriDet 69.3; TemporalMaxer 67.7; DyFADet 69.2; CausalTAD 69.75; CLTDR-GMG 74.3 with InternVideo2; **AdaTAD 76.9 avg / 80.9@0.5** | 2202.07925 / 2407.03197 / 2407.17792 / 2412.09202 / 2311.17241 |
| Fully-supervised TAL | FineAction / Multi-THUMOS | MambaTAD 29.4 / 46.6 (SOTA) | 2511.17929 |
| Weakly-supervised TAL | THUMOS14 avg mAP@[0.1:0.7] | STPN 27.0 → CoLA 40.9 → CO2-Net 44.6 → DELU 46.4 → DDG-Net 47.3 → FuSTAL 50.8 → **PseudoFormer 52.4** (43.4 avg@[0.3:0.7], 44.8@0.5) | 2103.16392 / 2107.12589 / 2307.16415 / 2504.14860 |
| Weakly-supervised anomaly / violence | UCF-Crime AUC / XD-Violence AP | Sultani-MIL 75.4 → RTFM 84.0 → MGFN 87.0 → UR-DMU 87.0 / 81.7 → VadCLIP 88.0 / 84.5 → GS-MoE 91.6 → LAS-VAD 91.05 / 89.96; LAVAD 80.3 training-free | 1801.04264 / 2101.10030 / 2211.15098 / 2302.05160 / 2308.11681 / 2404.01014 |
| Moment retrieval, specialists | QVHighlights R1@0.5 | Moment-DETR 52.9 → … → CVA 70.1 (CVPR 2026); R²-Tuning 68.03 with 2.7M trainable params | 2404.00801 |
| Moment retrieval, MLLM | Charades-STA / ActivityNet / QVHighlights R1@0.5 | TimeLens2 current open SOTA; Gemini-2.5-Pro zero-shot 61.1 / 64.2 / 75.9 on **re-annotated** splits | 2512.14698 / 2607.17423 |
| Text toxic-span | SemEval-2021 Task 5 | HITSZ-HLT 70.83 char-F1 (BERT+CRF ensemble); rationale-extraction-from-classifier only 38–60 | — |
| Rationale extraction | HateXplain | IOU-F1 **0.11–0.22** — token attribution barely agrees with human highlights | 2012.10289 |
| Speech toxic-span | ViToSA | Macro-F1 0.817 on **transcript character spans**, never mapped back to audio time | 2506.00636 |

Two facts from this table that should govern any design decision here:

- **OpenTAD's controlled study (arXiv 2502.20361)**: holding the detection head fixed, swapping the
  feature backbone moves THUMOS avg mAP **49.8 → 72.4 (+22.6)**; holding the backbone fixed, five
  years of neck/head architecture spans **67.9 → 68.4**, i.e. seed noise. **Features dominate;
  detection heads are a rounding error.** Any plan whose contribution is a new temporal head is
  buying the 0.5-point axis.
- **Frame-level AUC and mAP@tIoU are not the same measurement.** VadCLIP, one model, reports
  XD-Violence **AP 84.51 but mAP@[0.1:0.5] 24.70**, and UCF-Crime **AUC 88.02 but mAP@[0.1:0.5]
  6.68**. A model can score 88 frame-AUC and be nearly incapable of proposing a correct interval.
  MultiHateLoc and LELA both report the frame metric; only HateClipSeg reports tIoU.

### 4.3 Why the transfer is harder than it looks

1. **The label does not vary within the video** on the two benchmarks the field reports on (§1.2).
   TAL and WSVAD assume foreground is a *minority* of the timeline. HateMM's hateful region is a
   median 80.6% of the video and MHC-ZH's is 100%. Every mechanism these families rely on —
   background modelling, top-k selection, foreground/background contrast — assumes what these
   corpora violate. **The single largest gain in all of WSTAL is the explicit background class:
   24.3 → 36.6 avg mAP (+12.3) in ASM-Loc's ablation.** That gain exists because THUMOS background
   is *visually distinct*. When the background is the same person in the same room saying something
   slightly different, that loss has no gradient.
2. **Evidence is speech-carried, and the strongest localization modality reported is visual.**
   HateClipSeg F1@tIoU 0.5: visual 52.65, text 34.60, audio 25.40 — and late fusion of V+T+A is
   *worse* than visual alone at every threshold. This project measured the mirror image internally
   (CLIP-visual keys are blind to spoken hate). Nobody has reconciled the two observations.
   Independently: on XD-Violence, audio was worth ~5 AP in 2020 and is now ≲0.2 AP — the 2025-26
   SOTA there is RGB-only. **But the ordering flips with the task**: on HateClipSeg's *online*
   per-timestamp task the same paper's LSTR baseline has audio-only 60.84 beating visual-only
   57.52, the reverse of its own localization table. Prosody carries when you are classifying a
   moment; pixels carry when you are drawing its boundary. Nobody has explained this.
3. **One foreground class, defined by intent rather than appearance.** LAS-VAD's entire CVPR 2026
   contribution is bolting LLM intention reasoning onto WSVAD because visual separation is not
   recoverable — and its ablation over *which* LLM does the reasoning spans 0.08 points, meaning
   the semantic prior matters and the model identity does not.
4. **The boundaries are annotator-subjective, and the hate corpora barely measure it.** HateClipSeg
   is the only dataset that reports it, and segment-level is the worst of its four annotation
   tasks — Krippendorff α before → after its three-stage discussion protocol: offensive category
   0.840 → 0.899, video-level label 0.791 → **0.817**, target victim 0.716 → 0.721, and
   **segment-level 0.715 → 0.757**. SemEval-2021 toxic spans sit at mean pairwise Cohen's
   κ ≈ 0.61 with the organizers conceding the task is "highly subjective". **HateMM and
   MultiHateClip report no span-agreement statistic at all** — only label agreement (κ 0.625 and
   0.51–0.72). Nobody in this field has published a human-vs-human span F1, so no localization
   number here has a known ceiling. HateClipSeg's own stated reason for pre-segmenting via Whisper
   sentence boundaries and scene detection is that free-form annotator boundaries make quality
   "difficult to measure and ensure" — a designed-in admission that free boundaries do not replicate.
5. **The boundaries are annotator-generous, not evidence-minimal** (§5.2) — so even a perfect
   evidence localizer is penalised by the ground truth.
6. **The weak-to-full gap is large and widens with IoU.** THUMOS14, same protocol: TriDet (full)
   69.2 avg mAP@[0.3:0.7] vs PseudoFormer (weak) 43.4 — weak retains 63%. At IoU 0.7 it is 46.8 vs
   18.4, only 39%. That widening is the signature of a boundary-precision failure, and eight years
   of WSTAL has closed it by roughly 5 points.
7. **The corpora are small.** 431 HateMM hateful videos, ~1,134 usable DeHate spans, 380 offensive
   HateClipSeg videos. WSVAD trains on thousands.

### 4.4 What is genuinely importable

Ranked by cost, and filtered for "does not import an assumption the data violates":

1. **IoU-weighted loss reweighting at segment boundaries** — StreamSense's
   `L = −Σ IoU(W_i,S_i)^β · y_i log p(y_i|x_i)`, worth ~+1.2 Macro-F1 alone on HateClipSeg. It
   addresses a real property of the data (windows straddling a boundary carry a mixed label).
2. **An OCR channel.** LELA's modality ablation (GPT-4o-mini, ROC-AUC): speech 68.28 → +Image 68.89
   (+0.6) → **+OCR 71.47 (+2.6, the largest single jump)** → +Music 71.75 → +Video 72.27.
   An independent group found on-screen text to be the dominant modality gain in hate localization.
   **MultiHateLoc — the WWW 2026 weakly-supervised baseline — has no OCR channel at all.** This
   externally corroborates this project's 2026-08-08 OCR unblocking, which rested on the Gate-C
   finding that on-screen text is the only significantly enriched modality gap in the failures
   (30.1% of misses, OR 2.29). Two independent lines of evidence now point the same way.
3. **NumPro (arXiv 2411.10332, CVPR 2025)** — burn frame numbers into the pixels, zero training;
   Qwen2-VL-7B goes 5.4 → 36.8 R@0.5 on Charades. The cheapest transfer in the whole adjacent
   literature and it has never been pointed at hate.
4. **The concatenation trick from Speech Emotion Diarization** (arXiv **2306.12991v2**, Wang,
   Ravanelli & Yacoubi, ASRU 2023 — verified 2026-08-18; §5.1 concatenates same-speaker recordings
   into 21 h of simulated data under four transition patterns, and evaluates only on the real
   boundary-annotated ZED set) — synthesize span supervision by
   splicing clip-level-labelled data into known transition patterns. **Annotation-free**, and
   therefore compatible with this project's ban on manual annotation. Directly applicable: splice
   known-hateful and known-benign clips into synthetic videos with known boundaries.
5. **DCASE Task 4's recipe** (weak clip labels + unlabeled + small synthetic strong set, CRNN
   mean-teacher, frame posteriorgram → median filter → threshold; PSDS1 0.359 → 0.500 with BEATs
   embeddings) is structurally identical to this problem and is the mature version of it.

**Two things that look importable and are not.** (a) **Point-level supervision** is the highest-
leverage option in WSTAL — one clicked frame per instance buys ~8 mAP (HR-Pro 60.4 vs PseudoFormer
52.4 avg@[0.1:0.7]) — and it is **banned here**, because it is manual annotation. (b) Purpose-built
video temporal-grounding LLMs (TimeChat 2312.02051, VTimeLLM 2311.18445, Momentor 2402.11435,
TRACE 2410.05643, Grounded-VideoLLM 2410.03290, VTG-LLM 2405.13382, TimeSuite 2410.19702,
Time-R1 2507.18100, TimeLens 2512.14698 / 2607.17423) have **never** been applied to hate — a real
empty slot — but §4.1(b) is the warning: hate-specific RL tuning already lost to an off-the-shelf
zero-shot model on this exact task.

### 4.5 A measurement caution carried over from the grounding literature

Five papers report five different zero-shot Charades-STA numbers for the same Qwen2.5-VL-7B
(38.2 / 48.8 / 53.6 / 60.3). TimeLens re-annotated the standard grounding benchmarks and **model
rankings inverted**; frontier MLLMs that look terrible on original Charades-STA (GPT-5 at 18.3)
score 61.1 on the re-annotated version — annotation-convention fitting, not capability. Treat any
single published zero-shot MLLM temporal number as ±20 points until reproduced. The same caution
applies with more force to the hate numbers in §2.2, where no two protocols match.

---

## 5. What this project's own record says

### 5.1 TERA Gate-0 (F122, `refine-logs/TERA_GATE0_CAMPAIGN_RECORD_2026-08-07.md`)

Terminal verdict **NO-GO-C**. The binding failure: `multi_segment_complementary = 6/73 = 0.0822`
against a pre-registered 0.15 bar — only 8.2% of the baseline's false negatives need two separated
evidence units to interact. Reliability was not the problem (Cohen's kappa 0.733, PASS), and it was
not a power problem (the audit is a census of the whole false-negative population).

Two Gate-C facts bear directly on this landscape and are **descriptive, not licensed claims**:

- Union{`short_localized`, `multi_segment_complementary`, `cross_modal`} = **61/73 = 0.8356**
  (bootstrap lower bound 0.7534). Localized and cross-modal error mass is abundant.
- In the annotators' own coding of 133 audited videos: **`single_interval_sufficient` is True in
  111/133 (83.5%)**, the minimal sufficient interval set has exactly one interval in 111 videos and
  two or more in only 8, and the **`span_video_duration_ratio` has median 0.100, mean 0.300**
  `[measured, from logging/runs/gate_c_annotation/claude_c1_rows.jsonl]`.

### 5.2 The annotation gap this exposes

Put §5.1's last bullet next to §1.2: the **official HateMM span covers a median 0.806 of the video,
but the interval actually sufficient to justify the label covers a median 0.100**. The official
"hate span" is not an evidence localization — it is a coarse marking of the stretch in which
hateful content occurs, and it is roughly 8× more generous than the evidence needs to be. This is
the single most interesting empirical fact in this sub-direction and it is, as far as this sweep
found, unreported in the literature.

### 5.3 Four routes already spent on this axis, all negative

| route | record | outcome |
|---|---|---|
| Multi-granularity / segment-level temporal retrieval | `ideas/multigranularity-temporal-retrieval.md` | NEGATIVE — sign-flips by language (EN +0.015 F1 / ZH −0.066), no config beats whole-video baseline on both MHC splits; diagnosed as noisy MIL pseudo-positives |
| Segment-keyed retrieval-purity closed loop | `ideas/segment-keyed-retrieval-purity-loop.md` | KILLED by pilot; ratio 1.008 vs a 1.3× bar; the "below chance" reading was an argmax tie-break artifact (corrected 2026-08-09) |
| P11 — MLLM weak supervision for a trained segment head | `EXP_p11_weaksup_localization.md` | KILLED at the probe. MLLM teacher wv-AUC 0.5913 vs plain video-label top-k MIL 0.5526–0.5580 on identical windows; matched-operator Δ +0.0359, CI [−0.0009, +0.0730], n.s. |
| TERA Gate-0 | F122 | NO-GO-C (§5.1) |

### 5.4 Where the project's own localizers actually sit

| setting | metric | value |
|---|---|---|
| HateMM test, model-score, own 1 fps protocol | frame AP / AUC | 0.5892 / 0.7813 (video-broadcast control 0.5776 / 0.7735) |
| HateClipSeg 395, zero-training cross-dataset kNN | frame AP / AUC | 0.5447 / 0.5882 (broadcast control 0.5252 / 0.5701; random 0.4570 / 0.4885) |
| HateClipSeg, within-video mean AUC | wv-AUC | 0.5259, CI [0.5048, 0.5468], sign-test p=0.0066 — significant in 1 of 4 cells |
| HateClipSeg, MLLM window scoring (Qwen2.5-VL-72B A-fuse) | wv-AUC | **0.5755**, CI [0.5581, 0.5933] — the test pass is spent |
| HateMM calibration, same scorer | wv-AUC | 0.5913 |

Within-video AUC — the only one of these metrics that a video-level classifier cannot inflate,
because a broadcast control is 0.500 by construction — tops out at **0.576** after a 13-route
campaign that included 72B MLLMs. That is the honest state of hateful-video localization
capability inside this project.

---

## 6. Empty and occupied slots

### 6.1 Occupied — do not re-enter

| slot | held by | how firmly |
|---|---|---|
| Weakly-supervised MIL frame localization from video labels | MultiHateLoc (WWW 2026) | **firmly by publication, weakly by artifact** — the official repo is still LICENSE-only. Re-entering means competing with unreproducible numbers on a degenerate benchmark. |
| Training-free LLM per-frame scoring | LELA (2026-02) | firmly, and this project's own P6/P10 independently confirms the ceiling (~0.58 wv-AUC) |
| MLLM emitting timestamps + RL | TANDEM | occupied but **failing** — the trained model loses to a zero-shot baseline |
| Supervised TAL port as a dataset baseline | HateClipSeg / ActionFormer | occupied as a *baseline*; no follow-up method paper exists |
| Online / streaming per-timestamp classification | StreamSense (WWW 2026), LSTR | firmly, and freshly |
| Multi-granularity / segment-level *retrieval* | — published: nobody. **Internally: killed twice** (§5.3) | closed by this project's own negative results, not by the literature |

### 6.2 Empty

1. **Proposal-based detection as a method contribution.** Nobody has brought TriDet/DyFADet-class
   anchor-free heads with boundary regression to hate. HateClipSeg's ActionFormer baseline is a
   dataset-paper baseline, not a tuned method.
2. **DETR-style moment queries / set prediction** for hateful segments — untried.
3. **Query-conditioned grounding.** "Which moment attacks group X" is the natural formulation given
   that HateMM, MHC and DeHate all ship target-group labels, and nobody has tried it. This is the
   only slot where the hate domain has a structure that generic moment retrieval lacks.
4. **Audio-first / prosody-first localization.** Audio is the weakest single modality in every
   published localization table (HateClipSeg F1@0.5: A 25.40 vs V 52.65) yet hate in these corpora
   is overwhelmingly speech-carried — a contradiction nobody has investigated.
5. **Dense-captioning framing** (emit interval + explanation per hateful event) — untried.
6. **Localize → trim → re-classify with *predicted* boundaries.** The oracle version exists
   (Yang et al., +19 to +30 macro-F1); the predicted version does not exist anywhere.
7. **A shared protocol.** Five methods, five metrics, no comparable pair (§2.4).
8. **Any localization baseline on DeHate**, the second-largest span-annotated corpus.
9. **Degeneracy-aware evaluation.** No paper in this field reports coverage fraction, single-block
   fraction, or a video-level-broadcast control. §1.2–1.3 is, as far as this sweep can tell, new.
10. **An OCR channel in weakly-supervised hate localization.** MultiHateLoc uses ViT + VGGish +
    Whisper→BERT and no on-screen text; LELA's own ablation makes OCR the largest single modality
    gain (+2.6 ROC-AUC, larger than image, music or video context). Empty, and evidence-backed from
    two independent directions (§4.4.2).
11. **Purpose-built video temporal-grounding LLMs applied to hate.** TimeChat / VTimeLLM /
    Momentor / TRACE / Grounded-VideoLLM / TimeLens2 — none has been pointed at this task. TANDEM
    is general-MLLM prompting plus RL, which is a different thing.
12. **Annotation-free synthetic span supervision** (the Speech-Emotion-Diarization concatenation
    trick: splice clip-level-labelled hateful and benign material into videos with known
    boundaries). Untried in this domain and compatible with the no-manual-annotation constraint.
13. **Reporting mAP@tIoU on HateMM / MultiHateClip.** Both localization papers on those corpora use
    the frame-level VAD metric; §4.2 shows a model can hold 88 frame-AUC at 6.68 mAP@tIoU.

---

## 7. Fit with this project

### 7.1 The direct answer to "does the TERA result mean localization is degenerate on HateMM?"

**Yes, and more strongly than the TERA record alone implies.**

TERA Gate-0's `multi_segment_complementary = 6/73` says the *errors* rarely need two separated
evidence units. §1.2 says the *labels* rarely mark two separated regions: 72.8% of HateMM hateful
videos carry exactly one span, and that span covers a median 0.806 of the video. §1.3 closes the
loop: a predictor with zero temporal resolution scores frame-AP 0.675 on HateMM, above the
published state of the art. **On HateMM, "temporal localization" is video-level classification
with extra steps.** MultiHateClip is worse still (coverage median 0.937 EN / 1.000 ZH). Both are
short-form or single-topic corpora where the annotation unit *is* the video.

HateClipSeg is the exception and it is genuinely different in kind: 4-minute videos, semantic
segmentation, coverage 0.54, 3.5 toxic blocks per video, degenerate-oracle AP 0.530. If this
project ever does localization work, HateClipSeg — and DeHate's ~1,134 real-span videos — are the
only defensible arenas.

### 7.2 Why the project's four negatives were over-determined

Multi-granularity segment retrieval, the segment-keyed purity loop, P11 and TERA all failed for the
same reason, now visible in the data rather than only in the results: **on HateMM and MHC there is
almost no within-video signal to extract, because the label does not vary within the video.**
P11 measured this directly — a plain video-label top-k MIL head reaches wv-AUC 0.553–0.558, and a
72B MLLM reading frames plus ASR reaches 0.591, a difference that is not significant. That is what
"the span is the video" looks like from inside a learning curve.

This also means those four negatives **should not be read as evidence that temporal methods do not
work**. They are evidence that these two benchmarks cannot show it. The distinction matters if
anyone later proposes a temporal route on HateClipSeg or DeHate.

### 7.3 Entry options, ranked, against the project's actual constraints

Standing constraints: **method paper aimed at accuracy gains** (never benchmark / audit / metric
papers); no manual annotation; no new dataset construction; incremental but real gains acceptable.

| # | Option | Method-shaped? | Verdict |
|---|---|---|---|
| 1 | **Localize → trim → re-classify with predicted boundaries**, reported as a *video-level accuracy* gain | **yes** — the output is main-table accuracy, not a localization number | **the only live candidate.** Empty in the literature; oracle headroom +19 to +30 macro-F1 (Yang et al.); all ingredients already on disk. **Strong prior against it**, though: on HateMM the trim is a no-op a third of the time, and the project's own segment scorers top out at wv-AUC 0.59. Needs a cheap CPU pre-registered probe before anything else — see §7.4. |
| 2 | **OCR-channel localization** — add on-screen text to a weakly-supervised localizer | yes | **the strongest localization-side slot**, because it is the one place where two independent evidence lines agree (LELA's +2.6 ablation, this project's Gate-C 30.1% / OR 2.29) and the incumbent baseline has no OCR channel. Still a localization metric, so it only becomes admissible if folded into option 1 or if §7.5 is ruled on. The OCR cache already exists for HateMM train/val/test. |
| 3 | Query-conditioned grounding ("which moment attacks group X") | yes, but the metric is a localization metric | **no** under the method-paper-for-accuracy rule, unless recast as option 1 with a target-conditioned trimmer |
| 4 | Audio-first localization | yes | **no** — same reason; and §4.3.2 says the audio channel is the weakest in every published table |
| 5 | Proposal-based / DETR-style detection for hate | yes | **no** — pure localization performance; and OpenTAD (§4.2) shows the detection-head axis is worth ~0.5 mAP while the feature axis is worth ~22 |
| 6 | Fix the field's evaluation (report coverage, single-block fraction, a broadcast control, one protocol) | **no** — this is a benchmark/measurement paper | **closed by user rule.** It is the highest-value contribution available in this sub-direction and the project is not allowed to make it. It can appear only as an analysis section supporting a method claim. |
| 7 | Reproduce MultiHateLoc (no code) | no | **no** — reproduction, and the 2026-07-03 ruling already declined it; repo still empty as of 2026-01-28 |

### 7.4 If option 1 is pursued, the cheapest kill first

A pre-registered CPU probe, zero GPU, using assets that already exist:

- Segment scores: the P10-b 72B A-fuse per-window density (`train_segscoreK30_p10-p6-72b-bnb4-fuse.jsonl`)
  and the consensus-kNN memory vote — both already computed, both gold-free.
- Arena: HateMM train (the free-iteration calibration set), MHC-EN. Never the test split.
- Question: does trimming to the *predicted* top-scoring window fraction improve a video-level head
  over the untrimmed head, at any trim ratio, with a paired bootstrap CI excluding zero?
- Expected answer: **no on HateMM** (the span is the video, so trimming removes signal, and the
  Yang gain is largely a relabelling artifact of the trimmed protocol), possibly marginal on MHC-EN.
  Kill on that and the whole sub-direction is closed at ~an hour of CPU.
- The prior is bad enough that this should be framed as a kill probe, not a pilot.

### 7.5 Item requiring a user ruling

**Does training or evaluating on an already-public span-annotated dataset (HateClipSeg, DeHate)
count as "introducing a new dataset"?** The standing constraint bans new dataset construction and
manual annotation; it does not obviously ban *using* a published, already-annotated corpus, and
HateClipSeg's 395-video subset is already downloaded and its declared split
(`data/gt/HateClipSeg/p11_split.json`) is frozen and unconsumed. But adding HateClipSeg or DeHate
as a *training* arena changes the project's main table, which is a scope change.
**Flagged as pending user adjudication; nothing in this report assumes it either way.**
Note also that DeHate's raw video is gated behind an application form, so it would additionally
need a data-access decision.

---

## 8. Reproduction

The `[measured]` numbers were produced by short read-only Python snippets over the files in §1.6
(span-union coverage, contiguous-block counting, and an average-precision computation with random
tie-breaking, seed 0). No feature cache, model, checkpoint, or test-label file was opened.
No file under `data/` was modified.

---

## 9. Provenance and remaining gaps

**How the external layer was built.** Three independent sweeps: (a) datasets, (b) methods and
numbers, (c) adjacent fields. Sources used: arXiv Atom API (`export.arxiv.org/api/query`, both
`id_list` and title-scoped `search_query`), HuggingFace papers search
(`huggingface.co/api/papers/search`, ~25 queries, no rate limiting observed), DBLP author-level
queries on the two groups that own this axis, Semantic Scholar forward-citations of MultiHateLoc and
HateClipSeg, GitHub REST API tree listings and raw-file fetches, and WebFetch on
`arxiv.org/abs` and `arxiv.org/html`. OpenAlex was not relied on (daily budget risk).

**What is still `[title-only]` or unverified.**

- DeHate's exact span statistics — figure-derived only; the numeric table requires the application form.
- DeHate's `(0,0)` convention — inferred, not confirmed against the paper (ACM DL returns 403 to WebFetch).
- PCLMM public availability, and whether its "facial frame spans" are usable as hate-evidence spans.
- ImpliHateVid's claimed frame spans — the protocol mentions them, the release does not contain them.
- XD-Violence and UCF-Crime specifics are cited from secondary sources.
- TANDEM's Table-4 "mAP" figures are unverifiable from its own results and are not used anywhere here.
- MHC-EN coverage is measured on 245 of 331 span-carrying videos (local media availability);
  MHC-ZH on 262 of 327. HateClipSeg statistics are on the **395/435 surviving subset** — see
  `DATASET_hateclipseg.md §4` for the selection-bias statement that must accompany any number from it.

**A search-string warning for future novelty checks.** `abs:"multiple instance learning" AND
abs:"hate"` on the arXiv API returns **zero results** — MultiHateLoc does not put "MIL" in its
abstract. A novelty check on "MIL for hateful video" run with that phrasing would have concluded
the slot was empty when it is occupied by a WWW 2026 paper. Query on the task words
(`"temporal localisation"`, both spellings, plus `"segment"`, `"frame-level"`), not on the
mechanism words. Note also that both `localisation` and `localization` must be searched: the two
groups that own this axis use the British spelling.

**Convergent independent recommendation.** Two of the three sweeps, working separately, named the
same highest-value next action: *compute the trivial "predict the whole video as one span" baseline
on HateMM and MultiHateClip, because it may recover most of the published frame-mAP*. That
computation is §1.3 of this report, and the answer is that it does — it exceeds MultiHateLoc's
HateMM number and roughly doubles its MultiHateClip number under a 1 fps convention.

**Cost.** Zero GPU, zero training, zero cloud spend, zero test-label contact. Read-only throughout
except for this file.

---

## 10. Round-11 addendum (2026-08-18) — corrections and one structural finding

Added after a dedicated occupancy sweep over eight candidate mechanisms plus one question the
original report never asked. Verification tags as in the header.

### 10.1 The single most damaging new fact about the incumbent

**MultiHateLoc's own ablation, read from the PDF** `[read-method]` — Table 4, HateMM, adaptive
top-K expressed as a proportion of frames:

| K setting | frames selected | mAP | AUC |
|---|---|---|---|
| 1 | **all frames (100%)** | 0.612 | 0.758 |
| 2 | top 50% | 0.630 | 0.785 |
| 3 | **top 33% (their setting)** | **0.645** | **0.799** |
| 5 | top 20% | 0.620 | 0.762 |

Two readings, both damaging, and they compound §1.2:

1. The WWW 2026 hate localizer's tuned optimum **selects 33% of the timeline while HateMM's median
   gold coverage is 0.806**. The selection mechanism is in direct conflict with the label
   distribution it is fitted to.
2. **Turning selection off entirely costs 0.033 mAP** (0.612 vs 0.645). The MIL machinery — the
   paper's headline contribution — is buying 3.3 points on top of a predictor that pools every
   frame, on a benchmark where §1.3's zero-localization oracle already scores 0.675.

### 10.2 The structural finding the original report missed

> ⚠ **CORRECTION, 2026-08-18** (`idea-stage/R11_SEG_NOVELTY_CHECK.md` §2.3, cross-checked by
> gpt-5.6-sol at xhigh). **Six of the twelve families listed below do not actually break**, and the
> list must not be carried into a pre-registration or a paper in the form written here:
> *softmax-over-time pooling* (attention weights are mixture weights, not per-instant foreground
> posteriors; a_t = 1/T plus a separate sigmoid represents coverage 1.0 fine), *ActionFormer /
> TriDet focal-loss negatives* (both can emit a whole-video segment; focal loss is mis-tuned under a
> reversed class ratio, not invalid), *DETR-style no-object* (a category error — no-object applies
> to unmatched query slots, not to timestamps), *auxiliary background class / background
> suppression* (overstated — at coverage 0.8, 20% of the positive video is still background and
> wholly non-hateful videos exist), *top-K MIL pooling* (overstated — biased, not incapable), and
> *outer-inner completeness* (weakened, not invalid). The tIoU row is valid but must be reported
> split by single-span vs multi-span.
>
> **The only permitted form of the claim** is the narrowed one: *"objectives that impose low
> foreground density, or that manufacture negatives from relative within-video scores, become
> statistically inconsistent as foreground coverage approaches 1."*
>
> Two further corrections from the same check: the **"TAS has no background class, therefore it
> fits" argument in §10.3 is dropped** (under a multi-hot sigmoid, "all categories off" *is* the
> background state re-encoded — the surviving reasons are data scale and the absence of intra-video
> contrastive negatives); and **"TAS is the only surviving family" is false** — the dense action
> detection line (`1507.05738`, MS-TCT `2112.03902`, PAT `2308.05051`) is a fourth family that is
> not on the list and is a closer fit to a multi-hot toxicity timeline.
>
> R11-SEG v2 tested the narrowed claim directly (`idea-stage/R11_SEG_PILOT_RESULT.md` §3, v2) and
> **found no support for it**: a UniVTG-style score-derived intra-video negative term is null
> overall (+0.314 macro-F1, CI [−0.612, +1.268]) and null on the high-coverage stratum
> (+0.487, CI [−1.461, +2.658], n = 28). Underpowered, but this project has no positive evidence.
>
> Also missing from this report and added here: **SafeLens (AAAI-26**, Wang / Raharja / Hu / Lee,
> SUTD, pp. 41712-41714) — per-segment multimodal hate moderation fine-tuned on HateClipSeg with
> Whisper + EasyOCR + Qwen2.5-VL into a LoRA Llama3-8B, scoring segments **independently** with no
> temporal model. It is the nearest neighbour in the hate domain, it is from the HateClipSeg
> authors, and §2.1 / §6.1 should be read as if it were listed there.

The report asked which localization families are empty. It did not ask **which families are
structurally invalid at coverage 0.8-1.0**. They are, verified across TAL / WSTAL / WSVAD / TVG:

video-level MIL with top-k pooling; sparsity and attention-normalisation losses; softmax-over-time
pooling (attention mass sums to 1 across T, so it *cannot represent* "80-100% of instants are
foreground"); auxiliary background classes and background suppression (BaS-Net `1911.09963`);
**foreground/background contrastive and intra-video saliency negatives** — UniVTG `2307.16715`
treats *"other clips in the same video with saliency less than s_p"* as negatives, R²-Tuning
`2404.00801` and SDST `2507.07744` the same, and **at coverage 1.0 that entire negative set is true
foreground** `[read-method]`; outer-inner completeness scoring; instant-level detectors with
focal-loss negatives (ActionFormer, TriDet — i.e. HateClipSeg's own baselines); DETR-style
set prediction with a no-object class; relative CAS thresholding; background pseudo-label mining;
and the metrics themselves (at median coverage 0.8 a constant whole-video prediction clears
tIoU 0.3, 0.5 **and 0.7**).

**Every mechanism this project tried on the temporal axis, and every published hate-localization
method, comes from that list.** The failures were over-determined by the regime, not only by the
data scale.

### 10.3 The family that is *not* on that list

**Temporal action segmentation (TAS)** assumes 100% coverage by construction, has no background
class, and lives at this project's data scale (GTEA 28 videos, 50Salads 50, Breakfast 1712).
Three importable pieces, all 2025-2026, none applied to hate:

- **EAST** `2503.06316` (ICCV 2025 W) `[read-method]` — frozen ViT-G plus a Contract-Expand adapter
  at **4.7% of backbone parameters**; segmentation-by-detection.
- **Constraint-Aware Decoding** `2605.10149` (2026) `[read-method]` — **training-free** post-hoc
  Viterbi with transition confidences and **per-class duration bounds normalised to video length**.
  The most directly liftable coverage-prior mechanism found in the whole sweep.
- **RefDense** `2501.18509` `[read-method]` — per-frame multi-label sigmoid, **no background
  class**, therefore coverage-agnostic by construction.
- Adjacent: **Sound Event Bounding Boxes** `2406.04212` (Interspeech 2024) `[read-abstract]`
  decouples event *extent* from *confidence*, naming exactly the failure a single frame threshold
  causes at high coverage.

The composite that survives all three of this project's constraints (tiny data, intent-defined
single class, majority foreground): *frozen features + small adapter → per-instant binary or
multi-label score with plain BCE and **no intra-video contrastive negatives** → post-hoc constrained
decode with an explicit duration/coverage prior.*

**Do not lift WSVAD machinery.** UCF-Crime (1,610 train) and XD-Violence (3,954) match the data
scale and the frozen-CLIP-plus-light-head recipe is standard there, but every method is built on
normal-vs-abnormal MIL ranking or feature-magnitude contrast — a background prior in disguise,
whose entire signal comes from the abnormal video being *mostly normal*.

### 10.4 Occupancy corrections to §6.2

| §6.2 slot | corrected status |
|---|---|
| localize → trim → re-classify | **partially occupied**: Yang et al. `2508.04900` already runs a **clean→noisy** configuration — train on trimmed gold spans, test on full videos. What is unoccupied is the **duration-matched random-crop control**, which no paper in any domain has run. The control, not the crop, is the scientific content. |
| annotation-free synthetic span supervision | **occupied in video**: BSP `2011.10830` (ICCV 2021, splices trimmed clips and classifies the *boundary type*), AherNet `2008.13705` (ECCV 2020), TemPVL `2301.07463`. Only the concat-only, evaluate-on-real variant is unoccupied. ⚠ **"Background Mixup", cited in §4.4 of an earlier draft, could not be confirmed to exist** — zero arXiv hits across six formulations. Treat as unverified. |
| OCR-derived boundaries | **occupied in general video**: DuVOG `2208.11307` derives chapters from OCR'd subtitles; and the dominant free-boundary recipe is ASR, not OCR — Vid2Seq `2302.14115` (CVPR 2023) reformulates *"sentence boundaries of transcribed speech as pseudo event boundaries"*. Empty in hate. |
| query-conditioned grounding | **near-exact structural twin exists**: VadCLIP `2308.11681` (AAAI 2024) encodes anomaly class labels with frozen CLIP text and MIL-Align *"select[s] the most matched video frames for each label to represent the whole video"* `[read-method]`. Swapping the anomaly vocabulary for a protected-group vocabulary is a one-line substitution a reviewer will make out loud. |
| complement-region negatives (new slot) | **empty in hate, occupied in general TAL** — CoLA `2103.16392` hard-snippet mining, CPL (CVPR 2022) and CNM (AAAI 2022) generate negative proposals within the same video, UniVTG / R²-Tuning / SDST use lower-saliency frames of the same video as InfoNCE negatives. Yang et al. `2508.04900` *extracts* the non-hate segments of hateful videos for analysis and never trains on them. |
| modality-factored boundary vs label (new slot) | **empty, and it is the only slot in this report with no structural twin**. Decouple-SSAD `1904.07442` splits localization from classification *within one modality*; AVVP `2007.10558` predicts modality-*specific events*, not modality-specific *roles*; Centre Stage `2311.16446` explicitly fuses jointly. No paper assigns boundary prediction to one modality and label prediction to another. |
| broadcast-residual factorisation (new slot) | **the idea is occupied under other names** — RUBi `1906.10169` (NeurIPS 2019) and Learned-Mixin `1909.03683` (EMNLP 2019) are "score = bias branch + residual, train the residual". Not written for the temporal axis, but a VQA-literate reviewer names RUBi within a paragraph. |

### 10.5 Sequencing note that follows from §10.1-10.2

HateClipSeg's own published baselines (ActionFormer, LSTR) are drawn from the invalid list in
§10.2, and its paper frames its setting as *"offensive segments are sparse and embedded within
mostly normal content"* — adopting the minority-foreground assumption at 44.6% coverage. Its
online per-timestamp task is, formally, temporal action segmentation, not detection. That reframing
is free, is unclaimed, and is where §10.3's importable composite would land.

### 10.6 Verification gaps in this addendum

Semantic Scholar returned HTTP 429 on every call, so there is **no citation-graph cross-check**.
The CVF PDF for Tan et al. (WACVW 2024, video-classification-on-top-of-WSVAD) returned 403, so its
XD-Violence 78.84 → 82.10 figure is `[second-hand]` and must not be cited. CPL and CNM have no
arXiv IDs that could be located. arXiv indexes title and abstract only, so absence claims about a
*formula* — the broadcast-residual row above — are structurally weaker than the rest.
Yang et al.'s +19.34 / +30.45 and 98.64 / 97.31 remain extracted-from-HTML, not table-read, and
§7.4's warning to re-pull them before any pre-registration stands.
