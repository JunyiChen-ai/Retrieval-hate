# LITSWEEP-3 / Agent L2 — ZH-specific levers (MultiHateClip-ZH as the most winnable dataset)

**Round:** literature sweep round 4 of 5 · **Agent:** litsweep-3 L2 · **GPU spent:** 0 · **Date:** 2026-07-25
**Lens (unique):** MultiHateClip-ZH (Bilibili subset) — dataset-specific levers deployable ONLY through the existing
frozen/LoRA Qwen2.5-VL dual-stream → RGCL align head → rank-weighted kNN vote pipeline.
**Binding constraint box:** no gold annotations in deployed path (train labels OK); NO OCR channel (incl. no OCR-text
feature); single-dataset own-train-split only (mechanisms transfer, external data does NOT); no cross-seed ensembles;
no closed-model APIs; raw videos never leave machine; only locally-present models (Qwen2.5-VL-7B, CLIP) — downloads
user-gated; no reimplementing codeless baselines.
**House bar:** Δacc ≥ +0.030 AND ΔmF1 ≥ +0.030, 3/3 seeds, BOTH protocols (final-epoch + val-selected), on ≥1 dataset.

---

## 0. In-repo forensic corrections (verified this sweep — these reframe the whole ZH picture)

All checked directly against repo files, not inherited from ledger prose.

1. **"ZH transcripts median 4 words" is a WHITESPACE-SPLIT ARTIFACT — false as stated.** Chinese text has no
   inter-word spaces, so `text.split()` is meaningless. Measured on `data/gt/MHC_zh/{train,val,test}.jsonl`:

   | split | n | label 0/1 | gt-text CHARS median (mean/max) | whitespace-"words" median | rows w/ `<em class="keyword">` |
   |---|---|---|---|---|---|
   | train | 579 | 399/180 | **106** (134.2 / 708) | 4 | 243/579 |
   | val | 78 | 50/28 | **108.5** (131.4 / 343) | 3 | 34/78 |
   | test | 149 | 104/45 | **105** (129.4 / 361) | 4 | 63/149 |

   The deployed ZH text stream is **median ~106 Chinese characters (~50–70 words) — content-rich, not degenerate.**

2. **The deployed ZH "transcript" is the Bilibili DESCRIPTION/metadata, NOT the Whisper ASR.** The extractor
   (`src/utils/generate_VideoMLLM_embedding_HF.py:349-355`) builds the text stream as
   `TEXT_INSTRUCTION + "\nTitle: (none)" + "\nTranscript: " + gt["text"]`, and `gt["text"]` for ZH is the
   Bilibili search-result description (with literal `<em class="keyword">…</em>` highlight markup around the
   *un-obfuscated search keyword*, often the slur itself — present in 42% of train rows). The genuinely short/noisy
   Whisper ASR lives in a **separate, non-deployed** file (`data/ASR/MHC_zh/*_asrK4_whisper-large-v3.jsonl`),
   e.g. id `BV1em4y1B7bQ` ASR = `小蜜蜂嗯嗯` while its deployed gt-text is a full sentence. The `<em>` keyword
   highlight is baked into the current 0.8537 floor and inadvertently surfaces the slur — obfuscation density in
   the deployed text is therefore LOW.

3. **Title field is always "(none)"** for ZH (gt rows have only id/text/label; `item.get("title","")` → "") —
   confirms F74's audit. No title lever exists.

4. **Extraction prompts are ENGLISH** (verified `IMG_INSTRUCTION`/`TEXT_INSTRUCTION`, lines 45-52): the model reads
   Chinese title/transcript under an English task instruction. This is the one un-varied axis (see C1).

5. **ZH's binding wall is the 78-item dev val-selection, NOT representation.** LoRA-Qwen text stream already reaches
   AUC 0.925 (vs frozen 0.847, CLIP 0.802 — F45) and the encoder is Qwen2.5-VL, a **top native-Chinese model**
   (Qwen2.5-VL Technical Report, arXiv 2502.13923: leads OCRBench_v2 Chinese track by +20.6% over Gemini-1.5-Pro).
   Verified numbers (B3_PREREG_REVIEW.md §2.2): LoRA-Qwen vs CLIP paired 3-seed — final-epoch Δacc **+0.0313**
   (3/3), ΔmF1 **+0.0453** (3/3) = **PASS (marginal)**; val-sel Δacc **+0.0246** (< +0.030) = **FAIL**. ZH floor
   final-epoch 0.8537. Oracle union headroom +0.1026 is 91–98% selection-locked (F63/F66). So ZH is *one protocol
   away* and the gap is dev-selection noise (dev plateaus while test climbs — F45), not encoder quality.

**Net reframing:** the ZH ceiling is already high; any ZH-specific lever must either (a) lift the whole
representation enough that even the noisy 78-dev selection lands ≥+0.030, or (b) attack the selection variance
itself (that's the readout/SWA/gradnorm family — F62/F69/F70 all dead). Representation-side room is thin.

---

## 1. Literature sweep (verified citations only)

### A. Chinese hate / toxicity-specific representation (homophone / pinyin / euphemism)
- **ToxiCN / STATE-ToxiCN** (arXiv 2501.15451): span-level target-aware Chinese hate benchmark. Training-data
  resource only → single-dataset veto blocks direct use.
- **ToxiCloakCN** (EMNLP 2024, `2024.emnlp-main.345`, arXiv 2406.12223): homophone perturbations drop macro-F1
  2.3–6.9 pts, emoji 4.3–13.3 pts on ToxiCN-trained detectors incl. Qwen. **DECISIVE for us:** they tested
  **pinyin augmentation as a defense and it FAILED** — adding pinyin *reduced* accuracy on homophone-cloaked text
  across LLaMA/Qwen/Mistral/GPT-4o ("models recognize pinyin but struggle with pronunciation"). Homophone
  *normalization* was NOT tested (open, but the closest analog is dead).
- **MMBERT** (AAAI, arXiv 2508.00760): MoE multimodal BERT (glyph+phonetic) for cloaking-robust Chinese hate.
  Mechanism = glyph+phonetic experts; realization requires training on ToxiCN-type data → single-dataset veto;
  as an encoder it is a download-gated frozen-tower swap (D7-dead axis).
- **Homophone-aware semantic-phonetic collaboration** (Expert Syst. Appl. 2025, S0957417425033718) and
  **ChineseBERT** (glyph+pinyin pretraining, Sun et al. ACL 2021): dual textual+phonetic branches / glyph+pinyin
  embeddings help homophone robustness on Chinese text tasks. Deployable only as a text-tower swap (see C3).

**Transplant honesty:** the obfuscation problem these papers solve is largely **absent from our deployed ZH text**
(Bilibili descriptions, not adversarial comments; slur surfaced un-obfuscated in the `<em>` keyword 42% of the time)
and our encoder is already a strong native-Chinese model. Pinyin defense is empirically dead (ToxiCloakCN).

### B. Prompt-language matching for VLM-embedding extraction (our prompts are English)
- **mE5 / E5-mistral** (arXiv 2402.05672, 2401.00368) and **MMTEB** (arXiv 2502.13595): the field standard is a
  **single English task instruction across all document languages**, and instruction-tuned multilingual embedders
  with English instructions beat English-only models on multilingual retrieval. i.e. the published evidence says
  English instructions are *fine* for non-English documents.
- BUT: these are embedding-*tuned* models; Qwen2.5-VL is not embedding-tuned, and no paper I found isolates
  *instruction-language* on a frozen VLM extracting native-Chinese content. In-repo, **B1/P8c "language-match"
  was rejected at the ENCODER level** (frozen-Qwen-vs-CLIP, same English prompt — F19) and P8c tested Chinese
  *generated summaries* as a channel (worst arm 0.7168, closed). **Neither varied the extraction instruction
  language on the deployed path** → C1 is literally virgin, but thematically adjacent to the dead readout axis
  (F70 killed prompt-*structure* variants on ZH).

### C. Chinese-LLM text-stream alternatives (locally downloadable — user-gated, priced anyway)
- **Qwen3-Embedding** (arXiv 2506.05176; 0.6B/4B/8B): #1 MTEB-multilingual (70.58, Jun-2025), strong C-MTEB.
- **BGE-M3** (BAAI 2024): multilingual multi-granularity embedder, common baseline.
- **ChineseBERT / MMBERT**: glyph+pinyin, obfuscation-aware (above).
These are **frozen text-tower swaps** → the exact "换中文能力文本塔的检索族" that OPTION_KITS_terminus already flags as
"另一实验族,大概率[fail]", and the encoder-swap axis is D7-novelty-dead. They also **drop the dual-stream vision
grounding** that carried B3's Pareto conversion. Price: download-gated + extract + 3-seed head.

### D. Video-native Chinese hate datasets / methods (mechanisms transfer, data does not)
- **MultiHateClip** (ACM MM 2024, arXiv 2408.03468; 10.1145/3664647.3681521): our dataset. Reports multimodal >
  unimodal, and hateful-vs-offensive + non-Western nuance is the hard part — corroborates our dual-stream and the
  ZH difficulty we see.
- **ToxiBenchCN** ("Exploring Multimodal Challenges in Toxic Chinese Detection", arXiv 2505.24341): multimodal
  Chinese toxicity. Its "transferable" mechanisms — cross-modal fusion beats unimodal, symbols/visual-metaphor
  matter, temporal frame-before-speech — are things **we already do** (dual-stream concat; IMG_INSTRUCTION already
  asks for "symbols, gestures"; temporal operators F35/F37/F67 dead) or can't do (text-in-image needs OCR, vetoed).
  No new deployable operator.

### E. ASR-robustness / short-text (re-evaluated given §0)
Moot for the *deployed* ZH text (it's descriptions, median 106 chars, not short ASR). Augmenting with the ZH Whisper
ASR is possible but the ASR is genuinely degenerate ("小蜜蜂嗯嗯"), EXP_mm_segment_keys already found "ASR 通道对 ZH
是噪声", and F64 killed the Whisper-encoder audio axis. Prior negligible.

---

## 2. PAPER-VALUE list (verified, $0 or paper-only)

- **PV1 — "why English prompts for Chinese inputs?" is a live reviewer question.** Running C1 as a formal null (or
  positive) is clean analysis-chapter material that pre-empts it, whichever way it lands.
- **PV2 — the "ZH wall = selection noise, not representation" story now has external grounding.** ToxiCloakCN's
  pinyin-defense failure + Qwen2.5-VL native-Chinese SOTA (2502.13923) + our LoRA text-AUC 0.925 + `<em>` keyword
  surfacing = a citable explanation that ZH is representation-saturated and the residual headroom is selection-locked
  (reinforces F45/F63/F66 law-I narrative). Correcting the "median 4 words" artifact belongs in the same note.
- **PV3 — related-work grounding for the Chinese-hate multimodal challenge:** MultiHateClip (2408.03468), ToxiBenchCN
  (2505.24341), STATE-ToxiCN (2501.15451), ToxiCloakCN (2406.12223) — supports our "multimodal > unimodal" and the
  hateful-vs-offensive/cultural-nuance difficulty we observe on MHC.

## 3. User-ruling flags

- **UR1 (download + D7):** Qwen3-Embedding-4B/8B, ChineseBERT, or MMBERT-mechanism as ZH text-tower — download-gated
  AND encoder-swap (D7-novelty-dead). Priced in C3; not recommended as a novelty/perf bet.
- **UR2 (ensemble micro-ruling):** an English-prompt + Chinese-prompt text-stream ensemble (a variance-reduction on
  the val-sel gap) brushes the cross-seed-ensemble veto — F68 already flagged multi-prompt ensembling as needing a
  micro-ruling. Do not run without it.
- **UR3 (goal renegotiation):** ZH passes ONE protocol (final-epoch +0.0313/+0.0453 3/3). If the val-selection
  protocol is retired for the 78-dev noise reason (a standing user decision item), ZH is already a pass.

---

## 4. Ranked shortlist (max 3) + minimal-decisive-cell sketches + kill-switches

Honest headline: **no ZH-specific literature lever carries a prior ≥10% of clearing the +0.030/+0.030 both-protocol
bar** — consistent with the 3-agent F74 convergence that in-box ≥+3-on-2-datasets is unreachable and the ZH wall is
selection noise + representation saturation. The one item worth spending on is a *cheap virgin cell with paper value*,
not a performance bet.

### #1 — C1: Chinese-instruction re-extraction of the ZH text (+img) stream  [prior ~4–6%]
- **Transplant:** swap `IMG_INSTRUCTION`/`TEXT_INSTRUCTION` to fluent Chinese equivalents; re-extract ZH streams
  (806 videos) on frozen Qwen (and, if frozen shows any life, on the existing ZH-LoRA encoder); re-train head 3 seeds
  both protocols vs the banked English-prompt floor.
- **Cost:** ~0.3–0.6 GPU-h (extraction is 806×2 forwards + tiny head). NOT $0 (needs the forward pass) but cheap.
- **Minimal-decisive cell:** paired 3-seed, single variable = instruction language; frozen-extract img caches
  reused as clobber-guard; report Δ vs English-prompt floor at both protocols; primary decisive gate = frozen arm
  (if frozen is flat, LoRA arm auto-defunded, per the F67 spend-rule pattern).
- **Kill-switch (KS-C1):** frozen-prompt-ZH Δacc ≤ +0.015 at BOTH protocols (inside the ±0.014 ZH seed-noise band)
  → KILL, LoRA arm auto-dead; bank as the prompt-language null (PV1).
- **Strongest failure reason:** field consensus says English instructions are fine multilingually (mE5/E5-mistral);
  Qwen2.5-VL processes the Chinese body regardless of instruction language; instruction is a small span and the
  pooled readout is the assistant-header tail; adjacent to dead readout axis (F70). Most likely a clean null — which
  is still PV1.

### #2 — C2: Homophone/euphemism de-obfuscation preprocessing on the ZH text  [prior ~2–4%]
- **Transplant:** rule/dictionary de-obfuscation (homophone→standard, de-split, strip `<em>` markup) applied to
  gt-text before extraction; a lexical map is NOT gold annotation and NOT OCR; re-extract text stream + 3-seed head.
- **Cost:** preprocessing $0 (CPU) + ~0.3 GPU-h extract+head.
- **Minimal-decisive cell:** ablate two treatments (normalize-only vs normalize+strip-`<em>`) vs floor; single
  variable = text-surface transform.
- **Kill-switch (KS-C2):** best treatment Δacc ≤ +0.015 both protocols → KILL.
- **Strongest failure reason:** ToxiCloakCN VERIFIED pinyin defense fails across models incl. Qwen; deployed ZH text
  is *descriptions* with low obfuscation density and the slur already surfaced un-obfuscated in the `<em>` keyword;
  Qwen native-Chinese strength likely already robust. Prior barely above C1's floor.

### #3 — C3: Chinese text-tower swap (Qwen3-Embedding / ChineseBERT-glyph-pinyin)  [prior ~3–7%, but D7-DEAD + user-gated]
- **Transplant:** replace/augment the Qwen2.5-VL text stream with a downloaded Chinese embedder for the transcript;
  keep the Qwen image stream.
- **Cost:** user-gated download + extract + head (medium).
- **Kill-switch (KS-C3):** it is pre-killed on two independent axes before any GPU — (i) encoder-swap = D7-novelty
  excluded (cannot count as a contribution), (ii) "换中文文本塔" family flagged high-risk by OPTION_KITS_terminus;
  and it discards the dual-stream vision grounding that carried B3. Recommend **NO GPU**; hold as a user-ruling
  option only (UR1).
- **Strongest failure reason:** it must beat LoRA-Qwen text-AUC 0.925 AND cross the val-sel gap AND survive losing
  vision grounding — while adding zero novelty.

---

## 5. Bottom line (raw)

- **Shortlist + priors:** C1 Chinese-prompt re-extraction ~4–6% · C2 de-obfuscation preprocessing ~2–4% ·
  C3 Chinese text-tower swap ~3–7% (D7-dead + download-gated → no-GPU). None ≥10% vs the both-protocol bar.
- **Why so low:** the ZH wall is 78-dev val-selection noise (+0.0246 vs +0.030) plus representation saturation
  (LoRA text-AUC 0.925; Qwen2.5-VL native-Chinese SOTA), not encoder deficiency. The ledger's "median 4 words /
  degenerate ZH transcripts" premise is a whitespace artifact — the deployed text is median-106-char Bilibili
  descriptions with the slur often surfaced un-obfuscated. Pinyin-defense is empirically dead (ToxiCloakCN);
  the field uses English instructions multilingually by default.
- **Recommendation:** if any ZH GPU is authorized, spend it ONLY on **C1** and frame it as a **paper-value null**
  (answers the "why English prompts for Chinese?" reviewer question), not a performance bet. Everything else is a
  user-ruling item (downloads / ensemble micro-ruling / val-selection-protocol retirement).
- **PAPER-VALUE count: 3** (PV1 prompt-language null, PV2 ZH-representation-saturated grounding + "median-4-words"
  correction, PV3 Chinese-hate multimodal related-work grounding).
