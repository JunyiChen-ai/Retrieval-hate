---
type: paper
node_id: paper:lang2025_biting_off_more
title: "Biting Off More Than You Can Detect: Retrieval-Augmented Multimodal Experts for Short Video Hate Detection (MoRE)"
authors: ["Jian Lang", "Rongpei Hong", "Jin Xu", "Yili Li", "Xovee Xu", "Fan Zhou"]
year: 2025
venue: "ACM Web Conference (WWW) 2025"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["hateful-video-detection", "SVHD", "retrieval-augmentation", "mixture-of-experts", "multimodal-fusion", "sample-adaptive-router", "bipolar-attention", "HateMM", "MultiHateClip", "MHClip-B-Chinese", "Bilibili", "YouTube", "BitChute", "WWW2025", "audio-text-vision", "LVLM-baselines", "cross-dataset-generalization", "evolving-hate", "baseline-for-us", "is_core_hateful_video"]
added: 2026-07-01T09:40:40Z
---

# Biting Off More Than You Can Detect: Retrieval-Augmented Multimodal Experts for Short Video Hate Detection (MoRE)

## One-line thesis
MoRE is a retrieval-augmented mixture-of-multimodal-experts framework for short-video hate detection that retrieves relevant hateful/non-hateful videos via a joint (audio+text+vision) video retriever, injects that contextual knowledge into per-modality experts, and fuses experts with a per-sample adaptive soft router, beating SOTA by ~6.91% macro-F1 across three benchmarks. FIRST retrieval-augmented hateful-VIDEO method (retrieval-for-experts, not retrieval-guided contrastive).

## Problem / Gap
Short-Video Hate Detection: (1) hate evolves over time (static classifiers stale); (2) signals dispersed across audio/text/vision; (3) modality contribution varies per-sample (75.3% of MHClip-B hateful videos carry hate in only 1-2 modalities). Claims to be FIRST to bring MoE + retrieval augmentation to video hate.

## Method
Three-part end-to-end. (1) **Joint Multimodal Video Retriever**: Memory Bank of (audio,text,vision) triples from train+val; per-modality query vectors (Whisper->BERT audio, BERT title+desc text, ViT keyframes vision); weighted-cosine video-to-video retrieval; top-K=50 hateful + top-L=50 non-hateful (bipolar). (2) **Contextual experts** (audio/text/vision FFN) + **Bipolar Hateful Attention Network (BHAN)**: two cross-attentions AttHat (alpha=0.7) + AttNon (1-alpha), 'inspired by contrastive learning' but implemented as attention. (3) **Sample-Sensitive Integration** + Modality-mixture Soft Router (MSR). Losses ALL BCE (L_exp per-expert + L_ovl fused, epoch-annealed). **NO contrastive/InfoNCE loss; retriever is frozen weighted-cosine, not learned.**

## Key Results
Binary (offensive merged into hateful). MoRE ACC/M-F1 — HateMM 0.8341/**0.8235**; MHClip-Y 0.7750/**0.7519**; MHClip-B (Chinese) 0.7850/**0.7475**. Avg +5.27% ACC, **+6.91% M-F1** over best baseline (HateMM +7.59%, MHClip-Y +11.13%, MHClip-B +2.03%, p<0.01). Beats 9 baselines incl. HTMM, MHCL, LLaVA-OV, Qwen2-VL. Ablation: removing retriever drops HateMM M-F1 0.8235->0.7355. Router: text>vision>audio.

## Limitations / Failure Modes
**'Contrastive' is nominal** — BHAN is attention, all supervision BCE (the exact opening for a genuine retrieval-guided contrastive method). Retriever frozen/heuristic (not contrastively learned). Small datasets, offensive+hateful merged binary, video-level only (no temporal despite motivation). Memory bank includes val (+ target train in cross-dataset). Monolingual-per-dataset, not joint cross-lingual.

## Reusable Ingredients
(1) Joint multimodal video-to-video retriever (weighted-cosine, bipolar top-K hateful + top-L non-hateful) — the seed for retrieval-guided contrastive framing. (2) Memory bank from train+val (leak-safe). (3) BHAN bipolar attention (contrast target). (4) Sample-adaptive soft router. (5) Epoch-annealed loss schedule. (6) SVHD eval protocol (HateMM+MHClip-Y+MHClip-B, macro-F1, cross-dataset generalization). (7) Whisper EN+ZH ASR pipeline.

## Relevance to This Project
CENTRAL prior on the retrieval axis: the ONLY published retrieval-augmented hateful-VIDEO method. Instantiates retrieval-**for-experts** (attention + MoE, BCE) NOT retrieval-guided **contrastive** embedding — precisely our RGCL/RA-HMD niche. Defines our head-to-head baseline on the same three datasets/protocol. Leaves retrieval-guided contrastive representation learning open.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
- RGCL/RA-HMD → this project: extends MoRE's retrieval-for-experts to retrieval-guided contrastive embedding.

