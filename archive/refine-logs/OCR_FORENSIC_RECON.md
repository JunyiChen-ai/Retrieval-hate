# OCR CHANNEL — ZERO-GPU FORENSIC RECON

**Date:** 2026-07-28 NZST · **Mode:** $0 forensic recon, read-only.
**Standing status of the axis: VETOED and STILL VETOED.**
`autoresearch/goal_mllm_plus3/state/directions_tried.json:455` → `banned_constraints[0] = "OCR channel (user veto 2026-07-13)"`.

**What this document is:** an evidence package so the user can rule with numbers instead of an
impression. **It deploys nothing.**

**Compliance log (verifiable):**
- No SLURM submitted, no Modal launched, no GPU used, nothing trained. All measurement is
  CPU-only `json`/`torch.load` reads on the login node.
- **No test split was opened.** Every number below is computed on `data/gt/*/train.jsonl` only
  (one exception, explicitly labelled: `val.jsonl` id lists were enumerated in §3.5 to verify
  feature-cache *coverage* — no val label, text or metric was read).
- No repo config, prereg, model, key, training path or frozen artifact was modified. Scratch
  scripts live in the session scratchpad; this file is the only repo write.
- No raw video left the machine; no video was decoded by this recon at all (§2.2 explains why
  none was needed).

---

## 1. THE VETO'S PROVENANCE, AND WHAT HAS CHANGED SINCE

### 1.1 The veto itself

The ruling is recorded in two places, in the user's own words:

> **`autoresearch/goal_mllm_plus3/state/task_spec.md:8`**
> `- **NO OCR channel** (user veto 2026-07-13 "没啥用" — never re-propose).`

> **`research-wiki/LITERATURE_mllm_integration_2026-07-13.md:4`**
> `**用户裁定(2026-07-13,先于本文档定稿):不加 OCR 通道("没啥用")。** OCR 相关证据仅保留为 SOTA 校准,不作为候选。`

The commit that institutionalised it is `0498b9c` (2026-07-13 21:14:37 +1200). The reflection
document written the same day records what was given up:

> **`research-wiki/REFLECTION_mllm_integration_failures.md:34`**
> `文献中 HateMM 最大的已验证单通道增益是 OCR 通道(MM-HSD +2.6 M-F1),但用户 2026-07-13 裁定:不加 OCR(判定无用)——此路线关闭,仅留作 SOTA 校准证据`

So on 2026-07-13 the state of knowledge was: *one literature paper reports an OCR-channel
increment on HateMM; the user judges it not worth the channel.* No in-house measurement of OCR
on our data existed as evidence at ruling time.

### 1.2 What is genuinely new since 2026-07-13

Four things, in descending order of importance.

**(a) MM-HSD's OCR dependence was pinned down (litsweep-2 F74 / litsweep-5 F81).**
Not available on 07-13; measured against the literature on 07-25/26.

| claim | value | source |
|---|---|---|
| MM-HSD HateMM headline | **0.878 acc / 0.874 mF1** | `refine-logs/LITSWEEP5_HATEMM_EN.md:82` |
| MM-HSD **without OCR** | **0.845 mF1** | `refine-logs/LITSWEEP2_FRESH_2026.md:13`; independently `research-wiki/DRAFT_intro_related_limitations.md:146` |
| ⇒ OCR increment inside MM-HSD | **+0.029 mF1** | arithmetic on the two rows above |
| drop-any-one-modality band | mF1 **0.815 – 0.845** | `refine-logs/LITSWEEP5_HATEMM_EN.md:82`, `:180` |
| **OCR as a standalone modality** | **mF1 0.594** — the *weakest* of the four; transcript alone is 0.816 | `research-wiki/papers/cspedessarrias2025_mmhsd_multimodal_hate.md` "Key Results" |
| our own HateMM best | **0.879 acc / 0.873 mF1** (curric-LoRA, job 13241) | `refine-logs/LITSWEEP5_HATEMM_EN.md:84`, `:29-30` |

**Honesty flag on these numbers:** the "0.845 without OCR" figure is a *secondary* in-repo
record (two independent litsweep agents, no network available to me now to re-read the paper's
ablation table first-hand). The "0.815–0.845 drop-one band" and "OCR-alone 0.594" come from our
paper node, added 2026-07-01. Treat +0.029 as the recorded, not the re-verified, increment.

**The structural point the user should notice:** MM-HSD's OCR increment (+0.029 mF1) is measured
**from a base of 0.845, which is 2.8 pt BELOW our floor of 0.873.** This is the *identical shape*
to the CLAP evidence — Koushik's +2.9 mF1 from CLAP audio was measured from a base of 0.819,
also below our 0.873 (`refine-logs/LITSWEEP5_HATEMM_EN.md:80`, `:192`) — and CLAP was then
measured **dead** over our representation (§1.2(d)). "Channel adds X from a weaker base" has
already failed to predict "channel adds X over our base" once, decisively.

**(b) Error forensics (F88, 2026-07-26) independently nominated OCR as the top mechanism carrier.**
This was written *after* the veto, by an agent working the error sets, and it names OCR without
being able to use it:

> **`refine-logs/ERRPAT_HateMM_2026-07-26.md:417-419` (inside the FN1 block at `:410`)**
> `**VETOED**: OCR channel (standing user veto). This is the single most on-mechanism carrier for speech-poor hate videos, where the hateful content is frequently on-screen text/meme overlay — and it is the ablation-load-bearing channel in MM-HSD's 0.878 (F81 S2). It stays vetoed.`

The clusters OCR is nominated for, with their **oracle** ceilings (flip every member, break
nothing else):

| dataset | cluster | n | ceiling | source |
|---|---|---|---|---|
| HateMM | FN1 speech-poor visual hate | 7 | **+0.0326** acc | `refine-logs/ERRPAT_HateMM_2026-07-26.md:410` |
| MHC-EN | C4 lexical-surface FPs | 5 | **+0.0311** acc | `refine-logs/ERRPAT_MHC-EN_2026-07-26.md:386` (`LOCKED / VETOED (OCR, closed APIs)`) |
| MHC-ZH | C1a no-channel-knows | 8 | (locked, "would need a new input channel"; OCR named first) | `refine-logs/ERRPAT_MHC-ZH_2026-07-26.md:391` |
| MHC-ZH | C4 topic-vs-stance FP | 5 | not statistically significant (p=0.5022) | `refine-logs/ERRPAT_MHC-ZH_2026-07-26.md:357`, `:396` |

Note: on HateMM and MHC-EN the OCR-nominated ceilings (**+0.0326 / +0.0311**) are *already below
the +0.030 + 0.010 noise-band bar before any conversion loss*. That is not a detail; it is the
central arithmetic of this recon (§6).

**(c) The "in-box open set is EMPTY" finding (F88), and the 0-for-3 LITSWEEP6 sweep (F96/F97/F98).**
`refine-logs/ERRPAT_HateMM_2026-07-26.md:430` — `**GENUINELY OPEN in-box at $0**: none.`
`autoresearch/.../progress.json:17` — `the entire LITSWEEP6 accuracy menu measured at $0 CPU and
0-for-3 ... no GPU legal until a bar is cleared or a user ruling reopens an axis`.
The queue is empty. That changes the *opportunity cost* of the veto, not the veto's merits.

**(d) Every other new channel proposed since the veto has been measured dead.** This cuts
against OCR, and it is the honest base rate:

| finding | channel | best Δacc over `Z` | verdict | source |
|---|---|---|---|---|
| F41 / APX | eGeMAPS-88 prosody (HateMM) | **−0.0038** | KILL | `refine-logs/CLAP_GATE_RECORD.md:356` |
| F64 / LAUD | Whisper-large-v3 encoder (3 datasets) | **+0.0014 / +0.0014** (HateMM arms; global max over 6 cells +0.0041) | KILL | `refine-logs/CLAP_GATE_RECORD.md:357`; `refine-logs/LAUD_GATE_OUT.json` |
| F90 / CLAP | `laion/larger_clap_general` (HateMM, both `Z` arms) | **−0.0009 / −0.0038** | KILL | `refine-logs/CLAP_GATE_RECORD.md:358`, `:340` |
| W2-A | grounded transcript-first retrieval key | **−0.0000** (HateMM) / **−0.0038** (MHC) | KILL | `directions_tried.json` W2-A entry |
| archive-as-key | MLLM archive text as kNN key | **−0.0014 ± 0.0313** (5 seeds), **ZERO vote flips** | REFUTED | `research-wiki/ideas/archive-as-retrieval-key.md` |

**Summary of §1.** On 2026-07-13 we knew: one paper claims an OCR increment on HateMM. We now
additionally know: (i) that increment is +0.029 mF1 from a base 2.8 pt below ours; (ii) our own
error forensics nominate OCR as the top carrier for three error clusters whose *oracle* ceilings
are +0.0326 / +0.0311 / (n.s.); (iii) the in-box queue is empty; and (iv) **five** subsequent new
channels — including one keyed on the archive's own `on_screen_text` field — measured at or below
zero. The veto has become *more* consequential and, on the base rate, *not obviously wrong*.

---

## 2. DOES ON-SCREEN TEXT ACTUALLY EXIST IN OUR DATA?

### 2.1 THE HEADLINE FINDING: real OCR output is **already banked, for all three datasets, all splits**

Not the archive proxy — **actual easyocr text**, extracted in-house on 2026-07-03/04 during the
MoRE baseline reproduction, which ran *before* the veto and for a different purpose (reproducing
a published baseline's own pipeline).

| file | rows | size | mtime |
|---|---|---|---|
| `/data/jehc223/baselines/MoRE/data/HateMM/ocr.jsonl` | 1083 | 1.9 MB | 2026-07-04 04:33 |
| `/data/jehc223/baselines/MoRE/data/MultiHateClip/en/ocr.jsonl` | 1000 | 0.3 MB | 2026-07-04 17:44 |
| `/data/jehc223/baselines/MoRE/data/MultiHateClip/zh/ocr.jsonl` | 1000 | 1.3 MB | 2026-07-04 17:44 |

Extraction protocol (`/data/jehc223/baselines/MoRE/rerun/video_to_ocr_en.py:1-6`, and
`..._zh_easyocr.py:1-7`): **1 fps** frame sampling → SSIM dedup at 0.95 → easyocr →
regex clean → Levenshtein dedup → keep lines >3 chars. EN uses `easyocr.Reader(['en'])`;
ZH uses `easyocr.Reader(['ch_sim','en'])`. Note `autocorrect.Speller` is *imported but never
applied* (`video_to_ocr_en.py:87-91` only strips non-alphanumerics), so EN text is raw easyocr,
alphanumeric-only. Provenance and the paddle→easyocr substitution are documented at
`research-wiki/BASELINE_MoRE_rerun.md:83` and `:101`.

This is a **denser** census than any 8-frame sample we would build ourselves (1 fps over the
whole video vs 8 frames total), so no video decode was needed for this recon.

### 2.2 Coverage census — TRAIN SPLIT ONLY (measurement, not estimate)

| dataset | train n | OCR row present | **non-empty OCR** | median chars | median lines | max chars |
|---|---|---|---|---|---|---|
| **HateMM** | 744 | 744 (100.0%) | **540 (72.6%)** | 270 | 8 | 101,982 |
| **MHC-EN** | 549 | 549 (100.0%) | **358 (65.2%)** | 246 | 7 | 7,757 |
| **MHC-ZH** | 579 | 579 (100.0%) | **523 (90.3%)** | 498 | 18 | 19,092 |

**On-screen text is not scarce in our data. It is the majority case on all three datasets, and
near-universal on ZH** — consistent with the platform prior (Bilibili/TikTok burned-in captions
are the norm). The HateMM max of 101,982 chars is a pathological gaming-HUD/scrolling-chat
outlier; BERT's 512-token truncation (`preprocess/fea_extract/extract_text_ocr_feature.py:24`)
silently discards most of the long tail — a real quality caveat for any banked-feature gate.

### 2.3 Is the OCR text *new information* relative to what we already bank?

Our deployed text stream is `title + " . " + transcript`
(`scripts/prep_mhc.py:70-79`, `scripts/prep_video_dataset.py:126-134`). Character-4-gram
containment of the OCR string inside that banked text, non-empty rows, train only:

| dataset | median containment | mean | rows with containment <0.20 ("mostly new") |
|---|---|---|---|
| HateMM | **0.027** | 0.108 | 433 (80.2% of non-empty, **58.2% of train**) |
| MHC-EN | **0.213** | 0.299 | 172 (48.0% of non-empty, **31.3% of train**) |
| MHC-ZH | **0.005** | 0.052 | 479 (91.6% of non-empty, **82.7% of train**) |

**The OCR text is overwhelmingly not a restatement of the transcript.** This is the sharpest
contrast with the dead audio channels: F41/F64's declared failure mechanism was the **F31 hazard**
("the whisper-large-v3 *transcript* already banks spoken hate into the deployed `text_feats`",
`refine-logs/CLAP_GATE_RECORD.md:51`). That hazard **does not apply here at the surface level**.
Whether the *label-relevant* part is redundant is exactly what the gate in §3 would decide, and
surface novelty has repeatedly failed to imply conditional information in this project.

### 2.4 Does the new text carry hate surface the transcript lacks? (descriptive, unconditional)

A fixed slur/hate lexicon (~~36 EN terms, 26 ZH terms~~ — **miscounted, see erratum below**), train
split, presence-only. **This is a
prevalence statistic, NOT the G0-cond gate** — no `Z`, no classifier, no conditional read.

> **ERRATUM 2026-07-28 (this document, cosmetic — no number, percentage or conclusion changes).**
> The lexicon sizes stated above are wrong on both counts. Re-counted directly from the frozen
> `ocr_lex.py` used to produce the table: `LEX_EN` holds **31** terms (not 36; 31 unique), and
> `LEX_ZH` holds **26**. The per-dataset arms are also not "EN-only vs ZH-only" as the phrasing
> implies — the script sets `lex = LEX_EN if lang == "en" else LEX_ZH + LEX_EN`, so the **MHC-ZH arm
> was run against 26 ZH + 31 EN = 57 terms**, not 26. This is deliberate (ZH videos carry romanised
> and English slurs too) and is what was actually measured; only the prose undercounted it.
> Every cell in the table below is a *presence-only* count over whatever lexicon was in fact applied,
> so the counts, the percentages and the pos−neg gaps are unaffected, and the §2.4 conclusion
> ("new hate surface on HateMM only; the ≥2-dataset clause looks unreachable") stands verbatim.

| dataset | lexicon hit in OCR **but not** in banked text: **negatives** | **positives** | pos−neg gap |
|---|---|---|---|
| **HateMM** | 39/446 (8.7%) | **52/298 (17.4%)** | **+0.087** |
| MHC-EN | 1/381 (0.3%) | 3/168 (1.8%) | +0.015 |
| MHC-ZH | 6/399 (1.5%) | 5/180 (2.8%) | +0.013 |

**Read this carefully — it is the most decision-relevant table in §2.** OCR carries genuinely new
hate *surface* on **HateMM only**. On MHC-EN it is 3 positive videos out of 168; on MHC-ZH, 5 out
of 180. Whatever OCR is worth, **the ≥2-dataset clause of the goal bar looks unreachable through
it**, and the honest hope is a HateMM-only effect. (HateMM sample, `hate_video_1`:
`"...gorillavimonkeysinneed lynching Fuck all Digeers..."` — noisy, but the content is there.)

### 2.5 Coverage on the targeted error stratum

FN1 is defined by ≤25 transcript words (`scripts/analysis/errpat_hatemm_clusters.py:131`).
Applying the identical rule to the train split:

| dataset | stratum N (≤25 words) | % of train | with non-empty OCR | median chars |
|---|---|---|---|---|
| HateMM | 194 | 26.1% | **118 (60.8%)** | 124 |
| MHC-EN | 162 | 29.5% | 89 (54.9%) | 71 |
| MHC-ZH | 568 | 98.1% | 512 (90.1%) | 490 |

OCR **is** present for the majority of the speech-poor population that ERRPAT says is failing.
The channel reaches the diagnosed pathology. (F95/F97/F98 all reached their pathologies too, and
all still died — reach has been repeatedly shown to be necessary-not-sufficient in this project.)

### 2.6 The MLLM archive's `on_screen_text` field — a free proxy, but a contaminated one

`src/utils/generate_video_archive_HF.py:20, 89-94` defines `modality_cues.on_screen_text`. The
archive exists for **MHC-EN and MHC-ZH only** (`data/Archive/{MHC,MHC_zh}/`, v1 + v2);
**it does NOT exist for HateMM** — verified: `data/Archive/` contains exactly two subdirectories.
This confirms the task's premise.

Train-split census of the field:

| archive | train n | `on_screen_text` non-empty | median words | contains quote marks |
|---|---|---|---|---|
| MHC-EN v1 | 552 | 182 (33.0%) | 10 | 40.7% of non-empty |
| MHC-EN v2 | 549 | **136 (24.8%)** | 9 | 30.1% |
| MHC-ZH v1 | 583 | 234 (40.1%) | ~~13~~ (88 chars) | 52.1% |
| MHC-ZH v2 | 579 | **179 (30.9%)** | ~~4~~ (52 chars) | 41.9% |

> **ERRATUM 2026-07-28 (this document, cosmetic — no conclusion changes). The two MHC-ZH
> `median words` cells are the whitespace-split-on-Chinese artefact corrected campaign-wide by
> F77 / commit `298545e`** (same defect as the withdrawn "ZH transcripts median 4 words" row in
> `LITSWEEP2_INPUT_FIDELITY.md:64`). Chinese has no inter-word spaces, so `str.split()` returns
> roughly one token per punctuation-delimited run and systematically under-reports ZH volume.
> Re-measured on the same rows, the **character** medians are **MHC-ZH v2 = 52 chars** (not 4) and
> **MHC-ZH v1 = 88 chars** (not 13). The MHC-EN cells are *not* affected — English is
> whitespace-delimited, and for the record their char medians are 42 (v2) / 60 (v1).
> Nothing downstream moves: §2.6's "weak but real, strictly dominated" verdict rests on **coverage**
> (24.8% / 30.9% vs real OCR's 65.2% / 90.3%), on title regurgitation, and on meta-prose rather than
> literal strings — none of which is a length argument. Points 1–3 below and the §2.6 verdict stand
> verbatim, as does the ruling that any gate should read §2.1 rather than this field.

**Why this proxy is weak, measured three ways:**

1. **Title/transcript regurgitation.** The Qwen prompt is fed `title + transcript` as text
   (`generate_video_archive_HF.py:75-78`), so the model can and does emit the *title* as
   "on-screen text". EN v2: 36.8% of non-empty entries have 4-gram containment ≥0.80 in the
   input text. One ZH entry literally reproduces the Bilibili search markup:
   `BV1By4y1N74T → "女子扒光小三衣服并往<em class=\"keyword\">下体</em>塞辣椒粉"` — that `<em class="keyword">`
   is from the title field, not from pixels.
2. **Description, not transcription.** 40.0% (ZH v2) / 12.5% (EN v2) of entries are meta-prose
   ("The on-screen text includes phrases like…") rather than the literal string.
3. **Coverage is 2.5–3.5× worse than real OCR** (24.8%/30.9% vs 65.2%/90.3%).

Genuinely pixel-derived residue: for ZH v2, literal CJK runs (≥3 chars) that do **not** appear in
the title+transcript occur in 75/179 = 41.9% of non-empty entries = **13.0% of train**. For EN v2,
the "mostly new" bucket is 38/136 = 27.9% of non-empty = **6.9% of train**.

**Verdict on the archive proxy: it is real but strictly dominated.** The banked easyocr output
(§2.1) is denser, literal, and covers HateMM, which the archive does not. Any gate should use
§2.1, not §2.6.

### 2.7 CPU-only census with a real OCR engine — possible, but unnecessary

For the record: `easyocr` **is** installed (in the `ExMRD` env, with `cv2`, `skimage`, `av`) and
its weights are **already cached** — `~/.EasyOCR/model/{craft_mlt_25k.pth, english_g2.pth,
zh_sim_g2.pth}` (117 MB total, dated 2025-08-12 / 2026-07-03). So a CPU census would need **zero
download**. It was not run, because §2.1 already provides a denser 1-fps census of the same
videos with the same engine. `paddleocr`, `tesseract`, `doctr`, `mmocr` are absent; the
`HateVideo` env has none of the OCR stack (only `av`, `decord`).

---

## 3. THE G0-COND GATE — DESIGN ONLY, NOT RUN

### 3.1 The house template

The instrument is a conditional-information read: does the aux block add accuracy **over** the
deployed representation `Z`, with a label-oracle calibration arm and a pre-frozen numeric bar.
Cleanest templates: `refine-logs/CLAP_GATE_RECORD.md` (F90) and `refine-logs/LAUD_GATE_OUT.json`
(F64); machinery lineage `c3_fusion_probe.py` → `apx_g0cond_gate.py` → `laud_g0cond_gate.py` →
`clap_g0cond_gate.py` (`refine-logs/CLAP_GATE_RECORD.md:22-24`).

Fixed elements copied verbatim (`CLAP_GATE_RECORD.md:22-26`): `Z` standardised alone at its
Z-only inner-CV-optimal `C_Z`; aux block standardised × `s=50`, refit at `C_Z`; aux via
train-fold PCA (leak-free); decision family k ∈ {8,16} with {32,64} as context; 5×5
`RepeatedStratifiedKFold` `rs=1000+rep`; per-video-clustered bootstrap B=5000 on Δacc;
shuffled-aux control; ≥150-permutation null computed only to confirm a would-be pass. Binding
point = best of {k8, k16}. **Dev-side only (train ∪ val), single gate read.**

### 3.2 The exact OCR gate spec (`K-OCR-*`)

**Object:** OCR text as an auxiliary information block over the deployed representation.
**Datasets:** all three (HateMM primary; EN and ZH are the ≥2-dataset clause).

**`Z` arms** (identical to LAUD/CLAP, `CLAP_GATE_RECORD.md:44-45`, `LAUD_GATE_OUT.json` design):
- `deployed_7168` — per-dataset deployed encoder img⊕text:
  HateMM `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`; MHC `Qwen2.5-VL-7B-Instruct_HF`;
  MHC_zh `Qwen2.5-VL-7B-Instruct-LoRA_HF`.
- `strict_8960` — CLIP img⊕text ⊕ frozen-Qwen img⊕text (the W2-A/APX/LAUD `Z_best`).

**Aux blocks** (two, with a pre-declared primary):
- **`bert` — BINDING PRIMARY.** The already-banked BERT-CLS embedding of the OCR string:
  `fea_ocr_bert-base-uncased.pt` (HateMM, MHC-EN) / `fea_ocr_bert-base-chinese.pt` (MHC-ZH),
  768-d, `float32`. **Zero extraction cost. Coverage verified 100% of train and val ids on all
  three datasets** (§3.5). This is *exactly* the vector MoRE keys its retrieval on, so a KILL
  here is simultaneously a KILL of the published OCR-retrieval formulation.
- **`clip` — SECONDARY, cannot produce a PASS on its own.** CLIP-text-encoded OCR using the
  identical chunked 77-token mean-pooled path already used for archive text
  (`generate_video_archive_HF.py:867-905`), so the aux lives in the same text space as our own
  keys. Cost: one CPU forward over ~1,900 short strings per dataset (minutes). Declared as a
  guard against "the KILL was a BERT/truncation artifact" — a lone `clip` pass is **DISCORDANT**,
  an escalation request, not a promote (the `hidden`-block role in `CLAP_GATE_RECORD.md:40-42`).

**Pre-declared kill-switches (would be frozen and committed before any number exists):**
- **K-OCR-1 (calibration):** label-oracle `accZA < 0.99` → **MACHINERY_INVALID**.
- **K-OCR-0 (kill/promote, K9 house standard):** on `bert`, best-decision-k **Δacc < +0.040**
  OR bootstrap **CI-lower ≤ 0** OR (would-be pass only) not > all ≥150 permutation maxima →
  **KILL**. A PASS must clear **both** `Z` arms **and** ≥2 datasets (the goal bar is a
  ≥2-dataset clause; a HateMM-only pass is at most K-OCR-2).
- **K-OCR-2 (honest-partial):** +0.030 ≤ Δacc < +0.040 with CI-lower > 0 on both arms →
  documented near-miss, **not** an auto-promote.
- **K-OCR-3 (targeted stratum read, the one that matters):** restrict to train∪val items with
  **≤25 transcript words** (identical rule and word-count function as
  `errpat_hatemm_clusters.py:131`). AUC-based. **NARROW-GO** requires all of: (C1) OCR-block
  stratum AUC ≥ 0.65 with boot CI-low > 0.55; (C2) OCR stratum AUC − *transcript-BERT* stratum
  AUC ≥ +0.05 (the redundancy comparator; `fea_transcript_bert-base-*.pt` is also banked);
  (C3) conditional ΔAUC over `Z_deployed` within the stratum > 0 with boot CI-low > 0.
  **KILL-side** iff AUC ≤ 0.60 OR CI-low ≤ 0.50. Anything between = INCONCLUSIVE-NARROW =
  KILL for action purposes.
- **K-OCR-4 (blank-cell screen):** rows with empty OCR are 27.4% / 34.8% / 9.7% of train. Declare
  in advance whether they enter as zero-vectors (LAUD's convention) or are excluded; the primary
  is zero-vector (matches how every other cache in the repo handles missing modalities), with
  the non-empty-only view reported as a secondary "covered-rows" read exactly as W2-A did.

**Oracle arm (mandatory, pre-GPU):** the label-oracle calibration arm above (`accZA` must reach
~1.0, proving the machinery *can* see label information when it is present) — this is the arm
that caught and overturned a false kill once already (commit `9a0ad8c`, "shared-L2 crush bug").
Separately, the **already-measured** oracle ceilings from ERRPAT bound the upside independently:
**+0.0326 (HateMM FN1) / +0.0311 (MHC-EN C4)**.

**Declared tension, stated up front exactly as the CLAP spec did (`CLAP_GATE_RECORD.md:65-68`):**
the global +0.040 bar **exceeds the entire measured oracle ceiling of the clusters OCR targets.**
K-OCR-0 is retained anyway for graveyard comparability with F41/F64/F90 — which is precisely why
K-OCR-3 exists with its own independent bars, declared before any number rather than reached for
afterwards.

### 3.3 Split discipline (a real hazard here)

The banked `ocr.jsonl` and `fea_ocr_*.pt` files cover **all splits** (1083/1000/1000 rows). The
gate script must hard-filter to the train∪val id lists from `data/gt/*/{train,val}.jsonl` and
must never enumerate `test.jsonl`. This is stricter than the CLAP lane's declared position
(`CLAP_GATE_RECORD.md:110-114`, which argued extracting test *inputs* is not a test read) because
here the test-side features already exist and were built by a third party's pipeline.

### 3.4 Cost, if the gate were ever authorised

- CLAP gate: **681 s** on the login-node CPU for 1 dataset × 2 blocks × 2 `Z` arms + strata +
  permutations (`refine-logs/CLAP_G0COND_GATE_run.log`, `elapsed 681s`).
- LAUD gate: **530 s** for 3 datasets × 2 `Z` arms (`refine-logs/LAUD_GATE_run.log`, `elapsed 530s`).
- ⇒ **OCR gate ≈ 10–20 CPU-minutes total, on the `bert` primary: ZERO new extraction, zero GPU,
  zero download, zero SLURM.** Adding the `clip` secondary adds a few CPU-minutes of text
  encoding per dataset.

### 3.5 Verification that the primary gate is runnable today (coverage check only)

`torch.load` on the three `fea_ocr_*.pt` caches, cross-referenced against our id lists:

| dataset | keys in cache | dim | **train coverage** | **val coverage** |
|---|---|---|---|---|
| HateMM | 1083 | 768 | **744 / 744** | 107 / 107 |
| MHC-EN | 1000 | 768 | **549 / 549** | 80 / 80 |
| MHC-ZH | 1000 | 768 | **579 / 579** | 78 / 78 |

**100% coverage on both fitting splits, all three datasets. The gate has no missing-data problem
and needs nothing built.** (Only id membership was tested; no val label, text or metric was read.)

**IT WAS NOT RUN. The veto stands.** This section is a runnable spec, deliberately stopped one
command short.

---

## 4. IS "OCR AS RETRIEVAL KEY" ACTUALLY DISTINCT — AND OTHERWISE LEGAL?

### 4.1 Prior art: the formulation is not new. It is our own parent method, and a published video baseline.

**(a) RGCL itself — our method's parent — already keys retrieval on OCR text.**

> **`research-wiki/papers/mei2023_improving_hateful_meme.md:25`**
> `FROZEN CLIP ViT-L/14 encodes image + OCR text -> HateCLIPper-style fused joint embedding -> trainable projection MLP outputs classification logit + retrieval embedding.`
> **`:31`** — `text is OCR only`

In the meme domain the "text" stream **is** OCR text. Our video adaptation substituted
`title + transcript` for it (`scripts/prep_mhc.py:70-79`). So "put OCR into the retrieval key" is
not a new idea layered on RGCL — **it is RGCL's original design, and we removed it when we ported
to video.** Framing it as a novel contribution would be indefensible.

**(b) MoRE (WWW 2025) — the only published retrieval-augmented hateful-*video* method — keys its
retrieval on OCR, and we have already reproduced it.**

`/data/jehc223/baselines/MoRE/rerun/merge_feature_rerun.py:39-44`:
```
text_modal_fea = [ torch.load(fea_path / f"fea_ocr_bert-base-{mtype}.pt") ]
if "HateMM" not in dataset_dir:
    text_modal_fea.append(torch.load(fea_path / f"fea_title_bert-base-{mtype}.pt"))
```
i.e. the **"text" modality of MoRE's retrieval key is the OCR-BERT vector** — for HateMM it is
the *only* component. Retrieval is summed cosine over {audio, vision, text} with bipolar top-100
per label (`/data/jehc223/baselines/MoRE/retrieve/make_retrieval_result.py:29-35, 76-78`), which
is a video-level analogue of our kNN memory. MoRE's paper node
(`research-wiki/papers/lang2025_biting_off_more.md`) confirms the retriever is central: removing
it costs HateMM mF1 0.8235 → 0.7355.

**And we already have the head-to-head, on our clean splits** (`research-wiki/BASELINE_MoRE_rerun.md:99`,
table 3.2):

| dataset (clean test n) | MoRE as-released | MoRE bugfix | **ours** (arm) | Δ vs the **better** MoRE variant |
|---|---|---|---|---|
| HateMM (215) | 0.8140 / 0.7988 | 0.8047 / 0.7899 | **0.870 / 0.861** (frozen-Qwen) | **+5.6 acc / +6.2 F1** |
| MHC-EN (161) | 0.6894 / 0.4438 | 0.7019 / 0.5084 | **0.7888 / 0.7378** (frozen-Qwen) | +8.7 / +22.9 |
| MHC-ZH (149) | 0.7651 / 0.6882 | 0.7584 / 0.7058 | **0.8322 / 0.8023** (LoRA-SFT) | +6.7 / +9.7 |

*(acc / macro-F1. Our arm is **val-selected, warmup-consistent**, per `BASELINE_MoRE_rerun.md:136`;
all three "ours" rows trace to `research-wiki/ITERATION_LOG.md:397, 398/520, 848`.)*

> **ERRATUM 2026-07-28 (provenance audit, `PROVENANCE_AUDIT_2026-07-28.md` §7.2).** As first
> transcribed, this table showed only the *as-released* MoRE column while carrying the Δ column
> verbatim from the source — but the source computes Δ against **the better of MoRE's two variants**
> (`BASELINE_MoRE_rerun.md:131`, header `Δ(我们 − MoRE 两 variant 较优)`). Three of the six deltas
> therefore could not be reconstructed from the operands shown (EN +8.7/+22.9 and ZH F1 +9.7 read as
> +9.9/+29.4 and +11.4 against as-released). The `bugfix` column and the arm labels are restored above;
> **all six deltas then reproduce exactly**, and the 5.6–8.7 headline is unchanged.

A published OCR-keyed retrieval method for hateful video exists, we ran it on our data with our
own easyocr OCR, and it lands 5.6–8.7 acc **below** us. That is not an ablation of OCR (MoRE's
whole architecture is weaker), but it is a hard fact about the novelty claim. **Same arena, verified:**
identical clean test sets (215 / 161 / 149, *"test 完全同集"*), and MoRE trains on the official-split
labelled subset (EN 618 / ZH 633) against our clean train (EN 550 / ZH 579) — i.e. MoRE holds a
**nominal label advantage**, so the comparison is conservative in our disfavour
(`BASELINE_MoRE_rerun.md:136`). These are *not* MoRE's published full-split numbers; those are the
separate sanity table 3.1.

**(c) MM-HSD uses OCR as a CMA *query*, i.e. fusion — genuinely a different operator** than
retrieval keying (`research-wiki/papers/cspedessarrias2025_mmhsd_multimodal_hate.md`,
"Method": `Best config: O (OCR) as query, T+A+V as key/value`; and explicitly
`**No retrieval, no contrastive.**`). So MM-HSD does *not* pre-empt the retrieval framing — but
RGCL and MoRE both do.

**(d) Others using OCR as one of several modalities, none as a retrieval key:** LELA
(`research-wiki/papers/sun2026_towards_trainingfree_multimodal.md`, 5 modalities incl. OCR,
training-free LLM localization); Lu 2026 SIGIR (`research-wiki/gap_map.md:33`, frames+audio+OCR+caption).

**Novelty verdict:** *OCR-as-retrieval-key* is **NOT novel**. The auditable/editable framing
(pillars 1+4) is a presentation of an existing key space, not a new mechanism, and the project's
own D7 ruling already held that encoder-class/channel choices are not novelty
(`997ec29 ruling: D7 resolved (encoder-class ≠ novelty)`). At best OCR is a **performance /
ablation row**, never a novelty win — the same status the CLAP record assigned itself
(`refine-logs/CLAP_GATE_RECORD.md:57-59`).

### 4.2 Pre-closure audit against our own ledger

| finding | binds OCR-as-key? | reasoning |
|---|---|---|
| **archive-as-retrieval-key** (5-seed dAcc −0.0014 ± 0.0313, ZERO vote flips; `research-wiki/ideas/archive-as-retrieval-key.md`) | **PARTIALLY — and this is the most serious threat** | see below |
| **F50 / F85 fusion kills** (`directions_tried.json:355`) | **NO** | those close *trained concat/fusion operators*; a retrieval key is not a fusion operator. Note this cuts the other way too: MM-HSD's OCR gain is delivered *through* CMA fusion, which is the part we killed |
| **F66 / ISR selection-lock** (`directions_tried.json:250`) | **NO, on the letter** | F66's ban_scope is "independently re-encoded **per-segment** features feeding any NON-selecting aggregation"; a new *key channel* changes the key space, not the aggregation operator. **BUT** F66's *arithmetic* result — 91–98% of oracle headroom is formally selection-locked — is a statement about the deployed decision path that any new channel must still survive, because a new key only helps if the vote can convert it |
| **F89 / F94 / F95 / F97 / F98** | **NO** | all are *operator*-level closures on the deployed vote (whitening, k-depth, pair-verification, gating, learned aggregation). A key channel is upstream of every one of them. This is the strongest structural argument *for* OCR: it is the only remaining move that changes the **input** rather than the operator |
| **W2-A grounded key** (Δacc −0.0000 / −0.0038) | **NO, on the letter; YES, on the pattern** | W2-A re-encoded *existing* information (transcript+frames) into a new key. OCR would inject *genuinely new* text (§2.3 proves the surface is new). But W2-A is the 3rd of the "oracle-exists-unconvertible" instances the ledger names, and it is the closest procedural analogue |
| **P4 schema-distill / P8 semantic compression** | **NO** | both re-express banked information; neither adds a channel |
| **banned_constraints[1] gold annotations** | **NO** | OCR is machine-extracted from our own videos, not human annotation |
| **banned_constraints[5] MLLM-scores-as-training-signal** | **NO for easyocr; YES-adjacent for the archive route** | easyocr is not an MLLM. **If the archive `on_screen_text` field were used instead, it is Qwen-2.5-VL *generated text* — the P4/P11 family boundary. Use easyocr, not the archive field.** |
| **banned_constraints[7] external APIs** | **NO** | easyocr is local open weights, already installed, weights already cached |
| **banned_constraints[8] single-dataset train only** | **NO** | OCR is per-video, derived from that dataset's own videos |
| raw video off-machine | **NO** | OCR already extracted locally; nothing leaves |

### 4.3 Ruling on the isomorphism to `archive-as-retrieval-key`

**The threat is real and specific.** `generate_video_archive_HF.py:635` shows the archive text
that was CLIP-encoded into the retrieval key contains a line
`"On-screen text: " + (cues.get("on_screen_text") or "none")`. So **the field the task hoped was
a free proxy for on-screen text was already inside the key space that produced a 5-seed null with
zero vote flips.** That is not an adjacent negative; that is a partial direct hit.

**Why it is nevertheless not a full isomorphism — four measured differences:**

1. **Coverage.** The archive field is non-empty for 24.8% (EN) / 30.9% (ZH) of train (§2.6);
   real OCR is non-empty for 65.2% / 90.3% (§2.2). Roughly **3× more rows carry the signal**.
2. **Dilution.** `archive_to_text` concatenates **seven** fields; `on_screen_text` is one of them,
   median 4–9 words, inside a paragraph dominated by `neutral_summary`. After CLIP's 77-token
   chunking + mean pooling, the on-screen-text contribution is a small fraction of one key
   vector. Real OCR would be its own block, median 246–498 chars.
3. **Contamination.** 36.8% (EN v2) of the archive field is title/transcript regurgitation
   (§2.6), i.e. *already in the key*; the real OCR's median 4-gram containment in banked text is
   0.005–0.213 (§2.3).
4. **Coverage of HateMM.** The archive does not exist for HateMM at all — the one dataset where
   §2.4 finds real new hate surface. The archive negative therefore **carries no evidence about
   HateMM**, which is where an OCR effect would have to live.

**My ruling: NOT isomorphic, but the archive negative correctly reduces the prior on the ZH/EN
legs to near zero** — which happens to be exactly where §2.4 independently says there is almost
nothing to find (3 and 5 videos). **The two independent lines of evidence agree that the ≥2-dataset
clause is out of reach.** The live question is HateMM alone, and there the archive says nothing.

---

## 5. COSTING (hypothetical — nothing is authorised)

**Engine legality.** Local open weights only. `easyocr` (Apache-2.0) is the legal pick: installed,
weights already cached (§2.7), and it is the engine MM-HSD's EN protocol and MoRE's official EN
protocol both use (`research-wiki/BASELINE_MoRE_rerun.md:27`). PaddleOCR is **not viable on this
cluster** — measured, twice: GPU path fails cudnn dynload, CPU path SIGILLs
(`research-wiki/BASELINE_MoRE_rerun.md:83`). No hosted OCR (that would trip `banned_constraints[7]`).

**Extraction: already paid, $0 marginal.** The 1-fps easyocr pass over all splits of all three
datasets is **done** (§2.1). For reference, the measured original cost was GPU-accelerated easyocr:
~6.5 h for HateMM+MHC-EN (job 12236) + 2 h 27 m for MHC-ZH (job 12254)
(`research-wiki/BASELINE_MoRE_rerun.md:73`, `:83`) ≈ **9 GPU-h at 1 fps**. Re-extracting at our
own 8-frames/video protocol would be **≈ 0.3–1 GPU-h total** (8 frames vs 60–300 frames per video)
— but there is no reason to: the 1-fps output is strictly denser and already exists.

**Full ceremony, if OCR were unbanned AND cleared the gate:**

| stage | cost |
|---|---|
| G0-cond gate (`bert` primary, 3 datasets × 2 `Z` arms) | **10–20 CPU-min, 0 GPU-h** |
| optional `clip` secondary aux encoding | +~5 CPU-min, 0 GPU-h |
| re-extract OCR at our 8-frame protocol *(optional, not recommended)* | 0.3–1 GPU-h |
| head retrain, 3 datasets × 3 seeds, both protocols | **~0.3–0.5 GPU-h** (measured: a 2-encoder × 3-seed cell is ~0.1 GPU-h, `refine-logs/FUSIONCAT_SUBMIT_RECORD.md:138`; the CPU head path re-mints in ~1 CPU-min/seed, `refine-logs/ERRPAT_HateMM_2026-07-26.md:529`) |
| prereg + 0-context review + freeze + verdict review | 0 GPU-h, ~1 day of ceremony |
| **TOTAL to a pre-registered verdict** | **≈ 0.3–1.5 GPU-h** |

This is **cheap** — roughly one fifth of the MokA campaign's 5.573 GPU-h
(`refine-logs/MOKA_VERDICT_REVIEW.md:532`). Cost is not the reason to say no.

---

## 6. VERDICT AND RULING PACKAGE

### (a) 30-second summary

The OCR veto was issued on 2026-07-13 on a literature impression, before any in-house
measurement existed. Three things have changed. **First**, we discovered that real easyocr text
for all three datasets is **already banked** (extracted 2026-07-03/04 for the MoRE baseline
rerun), covering 100% of train ids, non-empty for 72.6% / 65.2% / 90.3% of HateMM / MHC-EN /
MHC-ZH train, and — unlike the dead audio channels — it is **not** a restatement of the transcript
(median 4-gram containment 0.027 / 0.213 / 0.005). Because BERT embeddings of that OCR are also
already banked with 100% train+val coverage, **a full G0-cond conditional-information gate is
runnable TODAY for ~10–20 CPU-minutes, zero GPU, zero download.** **Second**, our own post-veto
error forensics independently nominated OCR as the top mechanism carrier for three error clusters
— but priced their *oracle* ceilings at **+0.0326 (HateMM) / +0.0311 (MHC-EN) / n.s. (MHC-ZH)**,
i.e. **already below the +0.030 + 0.010 bar before any conversion loss**. **Third**, a descriptive
lexicon read shows OCR carries genuinely new hate surface on **HateMM only** (17.4% of positives
vs 8.7% of negatives, gap +0.087); on MHC-EN and MHC-ZH it is 3 and 5 videos respectively — so
the goal bar's ≥2-dataset clause is almost certainly unreachable through OCR. Against all of
that: the formulation is **not novel** (RGCL's own meme design keys retrieval on OCR text; MoRE
keys video retrieval on an OCR-BERT vector and we already reran it, landing 5.6–8.7 acc *below*
us), MM-HSD's +0.029 mF1 OCR increment was measured from a base 2.8 pt *below* our floor —
the exact shape of the CLAP evidence that then measured dead — and the closest own-ledger
precedent, `archive-as-retrieval-key`, put the archive's own `on_screen_text` field into the kNN
key and got **dAcc −0.0014 ± 0.0313 with ZERO vote flips over 5 seeds**. The base rate for a new
channel in this project is **0-for-5** (F41, F64, F90, W2-A, archive-as-key).

### (b) Honest probabilities

| question | my estimate | reasoning |
|---|---|---|
| **P(OCR converts to ≥ +0.030 acc on ≥ 2 datasets)** | **2–3%** | Two independent measurements say the EN/ZH legs are empty: the OCR-only lexicon signal is 3 and 5 videos (§2.4), and the ERRPAT ceilings are +0.0311 (EN, already sub-bar) and n.s. (ZH). Even a strong HateMM effect cannot satisfy a ≥2-dataset clause |
| **P(OCR converts to ≥ +0.010 acc on HateMM)** | **15–20%** | Genuine new surface (§2.3, §2.4), 60.8% coverage of the speech-poor stratum (§2.5), and the operator-level closures F89/F94/F95/F97/F98 do **not** bind an input channel. Discounted hard by: FN1 oracle ceiling +0.0326, the 0-for-5 channel base rate, the "increment from a weaker base" pattern that CLAP already falsified, and F66's selection-lock arithmetic which says even correct new information may not reach the vote |
| **P(the $0 gate returns KILL if run)** | **~75%** | It killed prosody, Whisper and CLAP; the archive-key precedent points the same way. But it would be the first channel entering with *non-redundant surface*, so this is genuinely less certain than the CLAP gate was |

**I am not selling this.** The single most likely outcome is another calibrated zero, purchased
for 15 CPU-minutes. The second most likely is a HateMM-only honest-partial that cannot satisfy
the goal bar and buys a paper ablation row, not a novelty claim.

### (c) The one question to ask the user

> **A $0, zero-GPU, zero-download conditional-information gate for OCR is runnable today on
> already-banked features (10–20 CPU-minutes, train+val only, no test contact). It is decidable
> and it is reversible — a KILL closes the last input-side axis permanently, at zero cost.
> Do you authorise running the *gate only* (not the channel, not any training), on the explicit
> understanding that (i) the veto remains in force and a PASS would still require a fresh ruling
> before any GPU, and (ii) even a full success is HateMM-only, sub-novelty, and cannot clear the
> ≥2-dataset goal bar?**

### (d) Does a $0 decidable gate exist TODAY on banked data?

**YES — unambiguously.** `fea_ocr_bert-base-{uncased,chinese}.pt` exist for all three datasets
with **verified 100% train and val id coverage** (§3.5); the `Z` arm caches are the same ones
F64/F90 used; the gate machinery is four generations of frozen, sha-pinned code; the CLAP and
LAUD gates ran in 681 s and 530 s of login-node CPU. Nothing needs to be extracted, downloaded,
installed, or submitted.

**Consequence for the ruling:** the user does **not** have to decide the OCR question on an
impression. The decision can be **deferred until after evidence** at a cost of ~15 CPU-minutes
and zero GPU. That is the actual finding of this recon — not that OCR is promising (it mostly
is not), but that **the veto no longer has to be defended or overturned in the dark.**

---

## APPENDIX — measurement provenance

All numbers in §2 were computed this session by read-only scripts in the session scratchpad
(`ost_census.py`, `ost_novelty.py`, `ost_cjk.py`, `ocr_census.py`, `ocr_lex.py`), reading:
`data/gt/{HateMM,MHC,MHC_zh}/train.jsonl`; `data/Archive/{MHC,MHC_zh}/{,v2/}train_*_archive.jsonl`;
`/data/jehc223/baselines/MoRE/data/{HateMM,MultiHateClip/en,MultiHateClip/zh}/ocr.jsonl`;
`/data/jehc223/baselines/MoRE/data/*/fea/fea_ocr_bert-base-*.pt`.
No test file was opened. No GPU, SLURM, Modal or network was used. No repo artifact was modified.
