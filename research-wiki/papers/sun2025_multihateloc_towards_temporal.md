---
type: paper
node_id: paper:sun2025_multihateloc_towards_temporal
title: "MultiHateLoc: Towards Temporal Localisation of Multimodal Hate Content in Online Videos"
authors: ["Qiyue Sun", "Tailin Chen", "Yinghui Zhang", "Yuchen Zhang", "Jiangbei Yue", "Jianbo Jiao", "Zeyu Fu"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2512.10408"
  doi: null
  s2: null
tags: ["hateful-video", "temporal-localization", "weakly-supervised", "multiple-instance-learning", "cross-modal-contrastive", "dynamic-modality-fusion", "HateMM", "MultiHateClip", "ViT", "VGGish", "BERT", "Whisper", "frame-level", "WWW2026", "multimodal", "audio", "implicit-hate", "is_core_hateful_video"]
added: 2026-07-01T09:39:47Z
---

# MultiHateLoc: Towards Temporal Localisation of Multimodal Hate Content in Online Videos

## One-line thesis
MultiHateLoc is the first weakly-supervised framework that localizes WHEN hate occurs in videos at frame level using only video-level labels, by combining modality-aware temporal encoders, dynamic cross-modal fusion, cross-modal contrastive alignment, and a modality-aware multiple-instance-learning (MIL) objective.

## Problem / Gap
Hateful-video research is almost all video-LEVEL binary; temporal localization (which frames are hateful) under weak supervision (video-level labels only) is unaddressed, and hate cues emerge asynchronously across modalities.

## Method
(1) Modality-aware temporal encoders + sentence-wise text encoding. (2) Dynamic Cross-Modal Fusion = Dynamic Modality Selection (per-timestep sigmoid gating) + Cross-Modal Attention. (3) **Cross-modal contrastive alignment (CM-Contrast)**: same video+timestamp across modalities = positives, else negatives — frame-level multimodal consistency. (4) Modality-Aware MIL (Top-K frame selection, K=3). L_total = L_MA-MIL + 0.1·L_smooth + 0.2·L_con. Encoders ViT/VGGish/BERT(Whisper). **Has FRAME-LEVEL contrastive but NO retrieval/kNN.**

## Key Results
Frame-level mAP/AUC. HateMM full (V+A+T): **mAP 0.645 / AUC 0.799** (vs Late Fusion 0.578/0.779, CMFusion 0.596/0.763). MultiHateClip: **mAP 0.445 / AUC 0.750** (vs CMFusion 0.420/0.672). Incremental ablation: +DCM-Fusion, +CM-Contrast, +MA-MIL each help. Sentence-wise text > naive.

## Limitations / Failure Modes
**NO retrieval/kNN** (orthogonal to RGCL). Only HateMM + MHC; no ImpliHateVid. Crosslingual only implicit via MHC-Chinese content, no per-language breakdown. Frame-level GT only on HateMM. Only frame mAP/AUC, no per-IoU AP.

## Reusable Ingredients
(1) Frame-level weakly-supervised MIL Top-K (adaptive K) — turns video-level labels into segment predictions. (2) **Frame-level cross-modal contrastive alignment** — a temporal contrastive variant to combine with RGCL retrieval-guided contrastive. (3) Per-timestep modality gating (DMS). (4) Sentence-wise text encoding. (5) Smoothness reg. (6) Frame-level mAP/AUC protocol + baselines (VAD-CLIP, Early/Late Fusion, CMFusion).

## Relevance to This Project
Core TEMPORAL axis; already fuses temporal + contrastive. Defines the frame-level SOTA/protocol we'd compete on if we add a localization head; its frame contrastive is directly comparable to RGCL's retrieval contrastive. Clear opening: **NO retrieval/kNN**, so retrieval-guided contrastive localization is still novel.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
- Contrastive (frame-level) already in hateful video; adds no retrieval — RGCL retrieval-guided contrastive remains open.

## Abstract (original)

> The rapid growth of video content on platforms such as TikTok and YouTube has intensified the spread of multimodal hate speech, where harmful cues emerge subtly and asynchronously across visual, acoustic, and textual streams. Existing research primarily focuses on video-level classification, leaving the practically crucial task of temporal localisation, identifying when hateful segments occur, largely unaddressed. This challenge is even more noticeable under weak supervision, where only video-level labels are available, and static fusion or classification-based architectures struggle to capture cross-modal and temporal dynamics. To address these challenges, we propose MultiHateLoc, the first framework designed for weakly-supervised multimodal hate localisation. MultiHateLoc incorporates (1) modality-aware temporal encoders to model heterogeneous sequential patterns, including a tailored text-based preprocessing module for feature enhancement; (2) dynamic cross-modal fusion to adaptively emphasise the most informative modality at each moment and a cross-modal contrastive alignment strategy to enhance multimodal feature consistency; (3) a modality-aware MIL objective to identify discriminative segments under video-level supervision. Despite relying solely on coarse labels, MultiHateLoc produces fine-grained, interpretable frame-level predictions. Experiments on HateMM and MultiHateClip show that our method achieves state-of-the-art performance in the localisation task.

