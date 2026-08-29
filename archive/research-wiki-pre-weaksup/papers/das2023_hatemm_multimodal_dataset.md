---
type: paper
node_id: paper:das2023_hatemm_multimodal_dataset
title: "HateMM: A Multi-Modal Dataset for Hate Video Classification"
authors: ["Mithun Das", "Rohit Raj", "Punyajoy Saha", "Binny Mathew", "Manish Gupta", "Animesh Mukherjee"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2305.03915"
  doi: null
  s2: null
tags: ["hateful-video", "dataset", "benchmark", "multimodal-fusion", "BitChute", "English", "video+audio+transcript", "ViT", "BERT", "HateXplain", "MFCC", "late-fusion", "frame-span-rationale", "target-labels", "Vosk-ASR", "ICWSM2023", "SOTA-anchor", "binary-classification", "is_core_hateful_video"]
added: 2026-07-01T09:38:36Z
---

# HateMM: A Multi-Modal Dataset for Hate Video Classification

## One-line thesis
HateMM introduces the first large public multi-modal (video+audio+transcript) hate-video benchmark of 1,083 BitChute videos with frame-span rationales and target labels, and shows that fusing all three modalities (BERT + ViT + MFCC) beats every unimodal model at hate-vs-non-hate classification.

## Problem / Gap
No public multi-modal hate-VIDEO benchmark existed; prior offensive-video efforts were <500 videos, text-only/text+audio, not public. HateMM fills this for low-moderation BitChute.

## Method
Binary Z(F,A,T)->y. 1 fps (100 frames, pad/subsample); Vosk ASR transcripts. TEXT fastText/LASER/BERT/HateXplain; AUDIO MFCC/AudioVGG19; VISION 3D-CNN/InceptionV3+LSTM/ViT+LSTM. Fusion = trainable late-concatenation (192d) -> 2-way head. Loss log-softmax+NLL; Adam 1e-4, batch 10, 20 epochs, 5-fold CV. **No cross-modal attention, no contrastive, no retrieval.**

## Key Results
Best M1 (BERT⊙ViT⊙MFCC): **acc 0.798, macro-F1 0.790**, hate-F1 0.749 — the SOTA anchor. Best unimodals: HateXplain 0.757/0.733, ViT 0.748/0.733, AudioVGG19 0.690/0.669. All modalities contribute (~2-3% each). kappa 0.625.

## Limitations / Failure Modes
Small (1,083), English-only, single-platform (BitChute far-right skew), late-fusion only, noisy ASR (~22% OOV), binary only, frame-span rationales collected but unused.

## Reusable Ingredients
(1) **HateMM dataset** (1,083 videos, ~144K frames, Vosk transcripts, frame-span rationales, target labels) — English anchor + retrieval corpus. (2) ViT+LSTM / BERT-HateXplain / MFCC stack. (3) Numbers to beat (0.790/0.798). (4) 5-fold CV + ablation template. (5) Frame-span rationales as retrieval keys / segment-level positives.

## Relevance to This Project
THE foundational hateful-video benchmark; English anchor to beat (M-F1 0.790). Naive late-fusion, NO retrieval, NO contrastive = clear opening. ViT-frame + transcript matches our Route A. Frame-span rationales/target labels give temporal + bias hooks.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
- RGCL/RA-HMD (memes) → HateMM adaptation: retrieval-guided contrastive never applied here.

## Abstract (original)

> Hate speech has become one of the most significant issues in modern society, having implications in both the online and the offline world. Due to this, hate speech research has recently gained a lot of traction. However, most of the work has primarily focused on text media with relatively little work on images and even lesser on videos. Thus, early stage automated video moderation techniques are needed to handle the videos that are being uploaded to keep the platform safe and healthy. With a view to detect and remove hateful content from the video sharing platforms, our work focuses on hate video detection using multi-modalities. To this end, we curate ~43 hours of videos from BitChute and manually annotate them as hate or non-hate, along with the frame spans which could explain the labelling decision. To collect the relevant videos we harnessed search keywords from hate lexicons. We observe various cues in images and audio of hateful videos. Further, we build deep learning multi-modal models to classify the hate videos and observe that using all the modalities of the videos improves the overall hate speech detection performance (accuracy=0.798, macro F1-score=0.790) by ~5.7% compared to the best uni-modal model in terms of macro F1 score. In summary, our work takes the first step toward understanding and modeling hateful videos on video hosting platforms such as BitChute.

