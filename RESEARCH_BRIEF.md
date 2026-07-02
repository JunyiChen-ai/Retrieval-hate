# RESEARCH_BRIEF — Hateful Video Detection

## Problem
Hateful/harmful video detection: binary-classify short videos as harmful vs normal from video frames + title + transcript (audio optional). Multimodal and multilingual (English + Chinese). Caption is NOT a model input (annotation-only).

## What I'm Looking For
Bring **retrieval-guided contrastive learning** to hateful VIDEO detection. Inspired by RGCL (Mei et al., ACL 2024) which did this for hateful MEME detection; we adapt the paradigm to video. **Novelty is scoped strictly within the hateful-video-detection field**: prior work in OTHER domains (including meme detection and the RGCL we adapt from) is treated as *inspiration* ("inspired by"), NOT as prior art that defeats our novelty. Target novelty = being the FIRST in hateful-video detection to do [retrieval-guided contrastive / cross-lingual retrieval / temporal retrieval / audio-prosody / implicit-hate retrieval], validated by measurable gains + ablation attribution.

## Constraints
Compute: SLURM cluster (login=compute node foscsmlprd01), conda env HateVideo, ALL GPU work via sbatch with no --time; per-user 16 CPU / 128 GB / 2 GPU; artifacts pushed to Backblaze B2 (b2:junyi-data/RGCL_video). Method backbone: FROZEN CLIP ViT-L/14-336 encoders -> precomputed embeddings -> lightweight ~5M-param MLP (HateClipper-style element-wise "align" fusion) -> FAISS retrieval-guided contrastive (triplet) + BCE hybrid loss -> retrieval kNN evaluation. CLIP is frozen; only the MLP head trains. faiss is CPU-only here (--Faiss_GPU False).

## Background
The RGCL mechanism is modality-agnostic (embedding-space contrastive mining + kNN retrieval inference over learned fused embeddings), which is why it ported to video by only swapping the encoder front-end (meme image+OCR -> video 8-frame CLIP + title+transcript CLIP). This also means the bare port has NO methodological novelty by itself — novelty must be added along a video-specific axis.

## Existing Results
Final verified test metrics (binary harmful-vs-normal). **Goal acc>=0.85 met on HateMM + ImpliHateVid; MHClip EN/ZH remain OPEN** (near field ceiling on tiny test splits). HateMM (EN): frozen-Qwen acc 0.870 (crosses 0.85). ImpliHateVid (EN, implicit+explicit hate, balanced test): ~0.90 acc / ~0.90 macro-F1 on both encoders (crosses 0.85). MultiHateClip English (MHClip-EN / MHC): frozen-CLIP 0.783 acc / 0.711 macroF1; frozen-Qwen 0.789 / 0.738; LoRA-adapted Qwen 0.7516 / 0.6916 — LoRA REGRESSES below both frozen floors; below 0.85. MultiHateClip Chinese (MHClip-ZH / MHC_zh): frozen-CLIP 0.805 acc / 0.771 macroF1; LoRA-adapted Qwen 0.8322 / 0.8023 (best-ever ZH, +0.027 acc vs CLIP); below 0.85 (English CLIP text tower on Chinese is a documented handicap). Novelty stance: cross-dataset updatable kNN memory = validated headline novelty vs MoRE; multi-granularity / segment retrieval = DEMOTED negative analysis; LoRA = mixed performance lever, not novelty. Datasets: HateMM, MultiHateClip (EN+ZH), ImpliHateVid.

## Domain Knowledge
Candidate novelty axes to validate against the literature survey: (1) cross-lingual / language-agnostic retrieval-guided contrastive — leverages existing EN+ZH assets and the documented Chinese text-tower weakness; current front-runner. (2) temporal / moment-level retrieval-contrastive — video != image; the current pipeline mean-pools 8 frames and discards all temporal structure. (3) audio / prosody as a third stream into the retrieval space — hate is often in delivery/tone; current pipeline uses transcript text only, discards the waveform. (4) implicit-hate-focused retrieval (ImpliHateVid) — retrieval of same-context/opposite-label confounders may suit no-slur implicit hate.

## Non-Goals
No video-LMM LoRA SFT (the "Route C" / real RA-HMD, e.g. Qwen2-VL) for now unless explicitly chosen later. Not chasing novelty relative to non-video domains. Not adding caption as a model input.
