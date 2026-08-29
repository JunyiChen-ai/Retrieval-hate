# Gap Map

_Field gaps with stable IDs._

_Field = HATEFUL / HARMFUL / TOXIC VIDEO detection (short social-media clips). Novelty judged ONLY within hateful-VIDEO. Meme / text / audio-only / general-video work is inspiration/ingredient, never field state. Built 2026-07-01 from 26 ingested deep-read records (papers/). Verdicts are honest: an "open" gap has NO hateful-video paper doing it; where a thing IS already done in video, it says so plainly._

_**Experimental updates (2026-07-02):** G4's segment/multi-granularity retrieval sub-gap is now a tested NEGATIVE (sign-flips by language, no acc≥0.85 — Phase 3 iter1/iter2); G3's cross-dataset/cross-lingual retrieval-memory opportunity is now ADDRESSED/VALIDATED vs MoRE as an updatable kNN memory (Phase 3b). See the per-axis UPDATE notes below._

_**LoRA lever vs the 0.85 ceiling (2026-07-02):** LoRA-adapted Qwen2.5-VL-7B RGCL+kNN (first LoRA-adapted RGCL runs on disk; jobs 2723309/2794237) does NOT cross the field acc≥0.85 ceiling on either MHClip split. **EN (MHC):** test acc 0.7516 / macro-F1 0.6916 (selEp26); gap to 0.85 = 0.0984 — and LoRA REGRESSES below both the frozen-CLIP (0.7826) and frozen-Qwen (0.7888) floors, i.e. it moved EN further from the bar. **ZH (MHC_zh):** test acc 0.8322 / macro-F1 0.8023 (selEp20); gap to 0.85 = 0.0178 — the closest any config has reached on ZH and the only clean apples-to-apples gain over the frozen-CLIP floor (+0.027 acc vs warmup-consistent CLIP 0.8054). Bar NOT crossed on either language; LoRA helps ZH but hurts EN (language-inconsistent, echoing the seg-mode finding). The MHClip 0.85 target stays OPEN on EN and ZH; acc≥0.85 remains met only on HateMM (frozen Qwen 0.870) and ImpliHateVid._

---

## Bottom line (the one thing that is genuinely open)

**Retrieval and contrastive each EXIST in hateful video, but their INTERSECTION does not.**
- Retrieval/kNN in hateful video: DONE by **MoRE** (WWW 2025) — but as retrieval-FOR-EXPERTS (frozen weighted-cosine retriever + attention experts + MoE router), trained with **BCE, no contrastive/InfoNCE loss**, retriever not learned.
- Contrastive in hateful video: DONE by **ImpliHateVid** (SupCon), **MultiHateLoc** (frame-level cross-modal contrastive), **IARE** (DPO preference contrast), **SCANNER** (centroid alignment) — but **none retrieve labeled exemplars (kNN pseudo-gold-positive / hard-negative mining) to drive the contrastive objective**, and none use a kNN-vote inference head.

=> **G1 (retrieval-GUIDED CONTRASTIVE embedding + kNN-vote inference, RGCL/RA-HMD-style) has NOT been done in hateful video.** This is our defensible core novelty. Everything else (cross-lingual, temporal, audio, implicit) is a *combination axis* layered on top of G1, not a standalone open gap.

---

## Datasets Table

| Dataset | Modalities | Languages | Size | Labels | Reported SOTA (metric) |
|---|---|---|---|---|---|
| **HateMM** (Das 2023, ICWSM) | frames+audio+transcript | EN | 1,083 videos (~43h) | binary hate/non-hate + frame-span rationales + target group | orig 0.790 M-F1 / 0.798 acc; **MM-HSD 0.874 M-F1** (video-level SOTA); Koushik HCC1 0.848; MoRE 0.8235; RAMF 0.851; MultiHateGNN 0.771; ImpliHateVid claims 97.58 F1 (vs weak re-runs — untrusted) |
| **MultiHateClip / MHC** (Wang 2024, ACM MM) | frames+audio+transcript+title | EN (YouTube) + **ZH (Bilibili)** | 2,000 (1,000/lang) | 3-class Hateful/Offensive/Normal + segment timestamps + target + contributing modality | EN multiclass 0.63 M-F1; **ZH multiclass 0.50** (big EN>ZH gap); binary: RAMF EN 71.7/ZH 70.9; HVGuard ZH-bin 0.822; MoRE MHClip-Y 0.752 / MHClip-B 0.7475 |
| **ImpliHateVid** (Rehman 2025, ACL) | frames+audio+transcript (+sentiment/emotion/caption) | EN | 2,009 (509 implicit/500 explicit/1,000 non) | binary + implicit/explicit split | TCL/SupCon **87.73 F1**; IARE (Ex-) 91.75 F1; TANDEM zero-shot 0.54 |
| **DeHate** (Zhang 2025, ACM MM) | frames+audio+transcript+title | EN | **6,689** (1,170 expl/950 impl/4,569 non) | explicit/implicit/non + segment timestamps + modality-contribution + 6 target groups | binary OpenAI-emb 0.708 M-F1; multiclass ~0.53 M-F1; **implicit-class F1 only 0.277** (headroom) |
| **HateClipSeg** (Wang 2025, ACM MM) | frames+audio+transcript | EN | 435 videos / 11,714 segments | Normal + 5 offensive types + target; **segment-level** | trimmed cls 69.48 M-F1; temporal-loc 59.4@tIoU0.3, ~29@0.7; online 62.75 |
| **PCLMM** (Wang 2024, ICME) | frames+**facial-expr**+audio+transcript | **ZH** | 715 Bilibili (196 PCL) | binary PCL/non-PCL (implicit microaggression) + frame spans + 6 groups | MultiPCL 84.03 acc / 81.06 M-F1 |
| **Ex-HateMM / Ex-ImpliHateVid** (Lu 2026, SIGIR) | frames+audio+OCR+caption | EN | 1,070 / 2,005 | binary + harmful-element tags + gold rationales | IARE 90.14 / 91.75 M-F1 |
| **MultimodAl Tamil Hate** (Maity 2025, TALLIP) | frames+audio+transcript | **Tamil** (code-mixed) | small (tiny test split) | 4-class offensive/sexist/racist/casteist | text 68.65 F1 (tiny-sample, high variance) |
| **ADIMA** (Gupta 2022, ICASSP) — *adjacent, audio-only* | audio | 10 Indic | 11,775 clips (65h) | binary abusive/non | ~76-86% acc per language (audio-only, NOT video) |
| **HatefulMemes / FHM / MAMI** — *inspiration only (memes)* | image+OCR text | EN | FHM 10,000 | binary/misogyny | RGCL 87.0 AUROC; RA-HMD ~91.1 AUROC (NOT video) |

_Note: HVGuard, MARS, SCANNER, MultiHateLoc, LELA, RAMF, TANDEM, MoRE, CMFusion, MultiHateGNN are METHODS reporting on the above datasets, not new datasets. HateMM + MHC(EN/ZH) + ImpliHateVid are the de-facto standard anchor trio._

---

## Per-axis whitespace analysis

### G1 — Retrieval-guided / kNN / retrieval-augmented
- **status_in_hateful_video: PARTIAL** (retrieval exists, but retrieval-GUIDED-CONTRASTIVE does not).
- **Evidence:**
  - DONE: **MoRE** (lang2025_biting_off_more, WWW 2025) — the ONLY published retrieval-augmented hateful-VIDEO method. Memory-bank video-to-video weighted-cosine retrieval, bipolar top-K hateful + top-L non-hateful neighbors feed attention experts (BHAN) + MoE router. BUT: retriever is FROZEN heuristic (not learned), and **all supervision is BCE — no contrastive/InfoNCE loss** (BHAN is "inspired by contrastive learning" but is attention). No kNN-vote inference head.
  - RELATED: **SCANNER** (li2026_shedding_facades_connecting, AAAI 2026) uses momentum K-Means centroid/prototype alignment at test time — retrieval-adjacent (prototypes as reference anchors) but source-free TTA, not training-time retrieval of labeled exemplars.
  - INSPIRATION ONLY (memes, not video): **RGCL** (mei2023) + **RA-HMD/LMM-RGCL** (mei2025) — the exact FAISS pseudo-gold-positive + hard-negative mining + kNN-vote recipe, never applied to video.
- **our_opportunity:** Port RGCL's **learned retrieval-guided contrastive embedding + kNN-vote inference** to video (8-frame CLIP + transcript/audio front-end). Genuine method-level novelty vs MoRE: (a) contrastive InfoNCE loss over FAISS-mined pseudo-gold positives / hard negatives instead of BCE-over-attention-experts; (b) a learned (not frozen-cosine) retrieval space; (c) a kNN inference head enabling training-free updates for evolving hate (which MoRE flags as needed but does not learn).

### G2 — Contrastive learning
- **status_in_hateful_video: DONE (as a technique), but NOT retrieval-driven.**
- **Evidence:**
  - **ImpliHateVid/TCL** (rehman2025_implihatevid, ACL 2025) — two-stage supervised contrastive (SupCon by class label), per-modality then cross-encoder. The canonical contrastive hateful-video method.
  - **MultiHateLoc** (sun2025_multihateloc, WWW 2026) — frame-level cross-modal contrastive alignment (same video+timestamp positives).
  - **IARE** (lu2026_decoding_multimodal_cues, SIGIR 2026) — DPO preference contrast (correct vs intentionally-wrong rationale paths), NOT standard InfoNCE.
  - **SCANNER** — centroid-alignment contrastive-style loss.
- **our_opportunity:** All existing video contrastive work chooses positives/negatives by **class label (SupCon) or same-timestamp**, NOT by **retrieval of nearest labeled exemplars**. RGCL's retrieval-mined hard negatives (opposite-label nearest neighbor = the confusable near-duplicate) + pseudo-gold positives is a distinct, unclaimed contrastive signal in video — exactly the hateful-vs-offensive / benign-confounder confusion these datasets document as hardest.

### G3 — Cross-lingual / multilingual (esp. Chinese)
- **status_in_hateful_video: PARTIAL (Chinese benchmarked & weak; no retrieval-guided cross-lingual).**
- **Evidence:**
  - Chinese covered: **MultiHateClip-Bilibili** (wang2024), **PCLMM** (wang2024_towards_patronizing, ZH implicit), **HVGuard** (jing2025, EN+ZH), **RAMF** (yang2025_reasoningaware, EN+ZH), **MARS** (yang2026_trainingfree_interpretable, HateMM+MHC-ZH), **SCANNER** (li2026, EN<->ZH cross-domain transfer). Tamil via **Maity 2025**.
  - But ZH performance is markedly worse (MHC-ZH multiclass 0.50 vs EN 0.63; RAMF ZH hate-F1 only 61.3) and cross-lingual is mostly monolingual-per-dataset; only SCANNER does explicit EN<->ZH transfer (and it's source-free TTA).
- **our_opportunity:** Cross-lingual **retrieval memory** — retrieve English labeled hateful neighbors to help score a Chinese query (and vice versa) in a language-agnostic RGCL embedding (mBERT/multilingual-CLIP/Sentence-BERT). No hateful-video paper does retrieval-augmented cross-lingual transfer; SCANNER's centroid TTA is the closest and is not retrieval-of-exemplars. Strong second axis layered on G1.
- **UPDATE (2026-07-02) — ADDRESSED / VALIDATED (as an updatable cross-dataset kNN memory).** Phase 3b (`src/eval_cross_dataset.py`, jobs 12136/12137) demonstrated the broader capability this axis rests on: because RGCL inference is a kNN vote over the labeled memory bank in the *learned fused space*, a head trained on dataset A classifies a different target T's test set by **SWAPPING the memory bank** to T's own train(+val) — **no retraining**. The learned space **transfers: above the majority baseline on 5 of the 6 informative cross cells**, lagging in-domain by a modest ~0.04–0.09 macro-F1 on working cells (Qwen tighter than CLIP; `ImpliHateVid→HateMM` Qwen ≈ in-domain-tied). This is a **capability MoRE structurally lacks** (its decision is baked into a trained MoE head that cannot be re-pointed at a new support set) => **VALIDATED novelty vs MoRE.** Caveats (honest): it is a capability demo, not a performance win (cross never beats in-domain); it **collapses when the target is MHC-EN** (falls to majority baseline); and the EN↔ZH cell specifically is **above-chance but degraded** — CLIP `MHC(EN)→MHC_zh` 0.633/0.758 and `MHC_zh→MHC(EN)` 0.645/0.739, each ~0.07–0.14 M-F1 below in-domain (cite the warmup-consistent CLIP EN↔ZH pair; the MHC_zh Qwen head is epoch-0). So the cross-lingual sub-case is real but weak; the general **updatable cross-dataset memory** is the validated contribution.

### G4 — Temporal / moment-level modeling of hate
- **status_in_hateful_video: DONE (heavily, as a task); retrieval-guided temporal is open.**
- **Evidence:**
  - Datasets: **HateClipSeg** (segment-level), **DeHate** (segment timestamps), **MHC** (segment timestamps).
  - Methods: **MultiHateLoc** (frame-level MIL localization), **LELA** (sun2026, training-free frame localization), **TANDEM** (koushik2026, timestamps via RL), **Yang temporal-label-noise** (yang2025_revealing, diagnoses +19-30% headroom from segment trimming), HateClipSeg baselines (ActionFormer/LSTR).
- **our_opportunity:** Temporal is crowded; a bare temporal claim is NOT novel. Open sub-gap: **segment-level retrieval** — retrieve similar labeled *segments* (not whole videos) as contrastive positives/negatives, exploiting Yang's finding that within-video non-hate segments are semantically-drifting hard negatives. Only compelling if fused with G1; standalone temporal is done.
- **UPDATE (2026-07-02) — NEGATIVE FINDING, DEMOTED.** We tested this sub-gap directly. Gold segment spans are **absent from our local HateMM/MHClip downloads** (Phase 1 / Phase 1 take-2: STEP 0 fails), so we fell back to **annotation-free multi-granularity** (auto sub-clip windows + within-video drifting-negative miner). The make-or-break CLIP ablation (Phase 3 iter1+iter2, jobs 12129/12131/12132–12135) shows the segment/multi-granularity term is **NOT a robust win**: the sign of the effect **flips by language** — `full (λ=0.5)` helps MHC-EN (+0.015 F1) but hurts MHC_zh (−0.066 F1); `milmax` rescues ZH (+0.017 F1) but collapses EN (−0.102 F1); `driftneg` is a near-no-op on EN and stays below baseline on ZH. **No seg_mode is ≥ baseline on both languages, and no config crosses acc≥0.85.** Diagnosed as noisy MIL positives (auto sub-clips are not the true hateful spans). => **Segment/multi-granularity retrieval is DEMOTED from headline novelty to honest analysis/ablation only.** Do not claim it as a win.

### G5 — Audio / speech / prosody as a signal
- **status_in_hateful_video: DONE (as a modality); usually the weakest, no retrieval over prosody.**
- **Evidence:** HateMM (MFCC/AudioVGG19), **CMFusion** (audio-temporal cross-attention), **MM-HSD** (wav2vec2-xlsr), **Koushik HCC1** (CLAP — biggest audio contribution to HateMM F1), **TANDEM** (Qwen2-Audio-LM), **PCLMM** (MFCC), **MultiHateLoc** (VGGish). Adjacent audio-only: **ADIMA** (prosody without ASR, Indic).
- **our_opportunity:** Audio is consistently reported as the weakest / least-separable modality (CMFusion UMAP, MoRE router audio-lowest). Retrieval could strengthen weak audio evidence via kNN over an audio-prosody memory, and prosody-based retrieval keys for implicit (no-slur) hate are untried. Supporting axis, not a standalone novelty.

### G6 — Implicit (no-slur) hate
- **status_in_hateful_video: PARTIAL (major focus, still the hardest unsolved slice).**
- **Evidence:** **ImpliHateVid** + **DeHate** + **Ex-ImpliHateVid** (datasets), **IARE** (rationale/DPO), **HVGuard** (CoT for puns/homophones), **MARS** (dual-hypothesis reasoning), **RAMF** (3-perspective reasoning), **PCLMM** (implicit ZH microaggression). DeHate quantifies the gap: implicit-class F1 = **0.277**.
- **our_opportunity:** Implicit is heavily attacked by prompting/reasoning VLMs but not by **retrieval of implicit exemplars**. RGCL's pseudo-gold-positive (retrieve a same-label implicit neighbor) directly targets the no-slur case where lexical features fail. Best framed as the *evaluation slice* where G1 wins, not a separate method.

---

## Recommendation (most open + feasible for us)

1. **PRIMARY (G1 + G2): retrieval-guided CONTRASTIVE embedding + kNN-vote inference for hateful video.** Genuinely unclaimed intersection — MoRE has retrieval-without-contrastive, ImpliHateVid/MultiHateLoc have contrastive-without-retrieval. Directly reuses our RGCL codebase (swap encoder front-end), maximally feasible, and gives a clean head-to-head vs MoRE on the exact same HateMM + MHClip-Y + MHClip-B protocol.
2. **SECONDARY (G3): updatable cross-dataset kNN memory (incl. EN<->ZH).** **VALIDATED (2026-07-02, Phase 3b):** memory-swap transfer beats majority on 5/6 cross cells, lags in-domain by ~0.04–0.09 M-F1 — a capability MoRE structurally lacks. The general updatable-cross-dataset-memory demo is the solid contribution; the pure EN<->ZH sub-case is above-chance but degraded (and MHC-EN is the one collapsing target), so lead with the cross-dataset capability, cite EN<->ZH as a weaker instance.

_Avoid claiming bare temporal (G4), bare audio (G5), or bare implicit (G6) as novelty — all are already worked in video; use them as evaluation slices / ingredients that G1+G3 improve. **Segment-level / multi-granularity retrieval (G4 sub-gap) is now a tested NEGATIVE — sign-flips by language, no acc≥0.85; keep as honest ablation only, do NOT claim as a win.**_
