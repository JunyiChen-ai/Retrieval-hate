# Research Wiki Query Pack

_Auto-generated. Do not edit._

## Project Direction
**Problem**

Hateful/harmful video detection: binary-classify short videos as harmful vs normal from video frames + title + transcript (audio optional). Multimodal and multilingual (English + Chinese). Caption is NOT a model input (annotation-only).

**Constraints**

Compute: SLURM cluster (login=compute node foscsmlprd01), conda env HateVideo, ALL GPU work via sbatch with no --time; per-user 16 CPU / 128 GB / 2 GPU; artifacts pushed to Backblaze B2 (b2:junyi-data/RGCL_video). Method backbone: FROZEN CLIP ViT-L/14-336 encoders -> precomputed embeddings -> lightweight ~5M-param MLP (HateClipper-style element-wise "align" fusion) -> FAISS retrieval-guided contrastive (triplet) + BCE hybrid loss -> retrieval kNN evaluation. CLIP is frozen; only the MLP head trains. faiss is CPU-only here (--Faiss_GPU False).

**Direction**

Bring **retrieval-guided contrastive learning** to hateful VIDEO detection. Inspired by RGCL (Mei et al., ACL 2024) which did this for hateful MEME detection; we adapt the paradigm to video. **Novelty is scoped strictly within the hateful-video-detection field**: prior work in OTHER domains (including meme detection and the RGCL we adapt from) is treated as *inspiration* ("inspired by"), NOT as prior art that defeats our novelty. Target novelty = being the FIRST in hateful-video detection to do [retrieval-guided contrastive / cross-lingual retrieval / temporal retrieval / audio-prosody / implicit-hate retrieval], validated by measurable gains + ablation attribution.

**Background**

The RGCL mechanism is modality-agnostic (embedding-space contrastive mining + kNN retrieval inference over learned fused embeddings), which is why it ported to video by only swapping the encoder front-end (meme image+OCR -> video 8-frame CLIP + title+transcript CLIP). This also means the bare port has NO methodological novelty by itself — novelty must be added along a video-specific axis.

**Non-goals**

No video-LMM LoRA SFT (the "Route C" / real RA-HMD, e.g. Qwen2-VL) for now unless explicitly chosen later. Not chasing novelty relative to non-video domains. Not adding caption as a model input.

**Domain Knowledge**

Candidate novelty axes to validate against the literature survey: (1) cross-lingual / language-agnostic retrieval-guided contrastive — leverages existing EN+ZH assets and the documented Chinese text-tower weakness; current front-runner. (2) temporal / moment-level retrieval-contrastive — video != image; the current pipeline mean-pools 8 frames and discards all temporal structure. (3) audio / prosody as a third stream into the retrieval space — hate is often in delivery/tone; current pipeline uses transcript text only, discards the waveform. (4) implicit-hate-focused retrieval (ImpliHateVid) — retrieval of same-context/opposite-label confounders may suit no-slur implicit hate.

**Existing Results**

Final verified test metrics (binary harmful-vs-normal). **Goal acc>=0.85 met on HateMM + ImpliHateVid; MHClip EN/ZH remain OPEN** (near field ceiling on tiny test splits). HateMM (EN): frozen-Qwen acc 0.870 (crosses 0.85). ImpliHateVid (EN, implicit+explicit hate, balanced test): ~0.90 acc / ~0.90 macro-F1 on both encoders (crosses 0.85). MultiHateClip English (MHClip-EN / MHC): frozen-CLIP 0.783 acc / 0.711 macroF1; frozen-Qwen 0.789 / 0.738; LoRA-adapted Qwen 0.7516 / 0.6916 — LoRA REGRESSES below both frozen floors; below 0.85. MultiHateClip Chinese (MHClip-ZH / MHC_zh): frozen-CLIP 0.805 acc / 0.771 macroF1; LoRA-adapted Qwen 0.8322 / 0.8023 (best-ever ZH, +0.027 acc vs CLIP); below 0.85 (English CLIP text tower on Chinese is a documented handicap). Novelty stance: cross-dataset updatable kNN memory = validated headline novelty vs MoRE; multi-granularity / segment retrieval = DEMOTED negative analysis; LoRA = mixed performance lever, not novelty. Datasets: HateMM, MultiHateClip (EN+ZH), ImpliHateVid.
## Open Gaps
# Gap Map

_Field gaps with stable IDs._

_Field = HATEFUL / HARMFUL / TOXIC VIDEO detection (short social-media clips). Novelty judged ONLY within hateful-VIDEO. Meme / text / audio-only / general-video work is inspiration/ingredient, never field state. Built 2026-07-01 from 26 ingested deep-read records (papers/). Verdicts are honest: an "open" gap has NO hateful-video paper doing it; where a thing IS already done in video, it says so plainly._

_**Experimental updates (2026-07-02):** G4's segment/multi-granularity retrieval sub-gap is now a tested NEGATIVE (sign-flips by language, no acc≥0.85 — Phase 3 iter1/iter2); G3's cross-dataset/cross-lingual retrieval-memory opportunity is now ADDRESSED/VALIDATED vs MoRE as an updatable kNN memory (Phase 3b). See the per-axis UPDATE notes below._

_**LoRA lever vs the 0.85 ceiling (2026-07-02):** LoRA-adapted Qwen2.5-VL-7B RGCL+kNN (first LoRA-adapted RGCL runs on disk; jobs 2723309/2794237) does NOT cross the field acc≥0.85 ceiling on either MHClip split. **EN (MHC):** test acc 0.7516 / macro-F1 0.6916 (selEp26); gap to 0.85 = 0.0984 — and LoRA REGRESSES below both the frozen-CLIP (0.7826) and frozen-Qwen (0.7888) floors, i.e. it moved E
## Failed Ideas (avoid repeating)
- **LoRA-SFT of the Qwen2.5-VL encoder (prediction still via RGCL contrastive + kNN head)**: **Lesson:** LoRA-SFT of the Qwen2.5-VL encoder is a MIXED performance lever, not novelty — best-ever ZH (0.8322 acc / 0.8023 macroF1, +0.027 acc vs frozen-CLIP floor) but REGRESSES EN below both froze
- **Multi-granularity / segment-level temporal retrieval (AUTO sub-clip FAISS + MIL drifting hard-negative)**: **Lesson:** Segment/multi-granularity retrieval is a tested NEGATIVE — language sign-flips (EN vs ZH) and noisy MIL pseudo-positives (no gold spans) mean no seg_mode beats the whole-video baseline on 
- **Retrieval-consensus segment denoising (memory washes its own sub-clip labels)**: 
- **Multi-granularity annotation-free temporal retrieval + updatable kNN memory for hateful video (MLLM-encoded)**: **Lesson:** This umbrella node over-bundled three mechanisms; resolved into split nodes with honest outcomes — multi-granularity temporal retrieval = NEGATIVE, cross-dataset kNN memory = POSITIVE head
## Key Papers (26 total)
- [paper:cspedessarrias2025_mmhsd_multimodal_hate] MM-HSD: Multi-Modal Hate Speech Detection in Videos
- [paper:das2023_hatemm_multimodal_dataset] HateMM: A Multi-Modal Dataset for Hate Video Classification
- [paper:gupta2022_adima_abuse_detection] ADIMA: Abuse Detection In Multilingual Audio
- [paper:jing2025_hvguard_utilizing_multimodal] HVGuard: Utilizing Multimodal Large Language Models for Hateful Video Detection
- [paper:koushik2025_towards_robust_framework] Towards a Robust Framework for Multimodal Hate Detection: A Study on Video vs. Image-based Content
- [paper:koushik2026_tandem_temporalaware_neural] TANDEM: Temporal-Aware Neural Detection for Multimodal Hate Speech
- [paper:lang2025_biting_off_more] Biting Off More Than You Can Detect: Retrieval-Augmented Multimodal Experts for Short Video Hate Detection (MoRE)
- [paper:li2026_shedding_facades_connecting] Shedding the Facades, Connecting the Domains: Detecting Shifting Multimodal Hate Video with Test-Time Adaptation
- [paper:lu2026_decoding_multimodal_cues] Decoding Multimodal Cues: Unveiling the Implicit Meaning Behind Hateful Videos
- [paper:maity2025_multimodal_approach_hate] A Multimodal Approach for Hate and Offensive Content Detection in Tamil: From Corpus Creation to Model Development
- [paper:mei2023_improving_hateful_meme] Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning
- [paper:mei2025_robust_adaptation_large] Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection
## Recent Relationships (35 total)
  idea:multigranularity-temporal-retrieval --addresses_gap--> gap:G1
  idea:multigranularity-temporal-retrieval --addresses_gap--> gap:G4
...(truncated)
