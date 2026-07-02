---
type: paper
node_id: paper:mei2023_improving_hateful_meme
title: "Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning"
authors: ["Jingbiao Mei", "Jinghong Chen", "Weizhe Lin", "Bill Byrne", "Marcus Tomalin"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2311.08110"
  doi: null
  s2: null
tags: ["retrieval-augmented", "contrastive-learning", "hard-negative-mining", "pseudo-gold-positive", "knn-inference", "faiss", "CLIP", "HateCLIPper", "hateful-meme", "frozen-encoder", "project-base-method", "meme-domain", "ACL2024", "inspiration", "NOT-core-hateful-video"]
added: 2026-07-01T09:39:52Z
---

# Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning

## One-line thesis
RGCL constructs a hatefulness-aware embedding space via retrieval-guided contrastive training (hard-negative + pseudo-gold-positive mining) plus kNN majority-vote inference, yielding SOTA hateful-MEME detection with a lightweight trainable head. INSPIRATION/PROJECT-BASE-METHOD, meme domain, NOT hateful-video.

## Problem / Gap
CLIP-based hateful-MEME embedding spaces are insensitive to subtle differences deciding hatefulness (near-duplicate memes with opposite labels sit close). Also can't be updated with new examples without retraining. (Domain = MEMES, not video.)

## Method
FROZEN CLIP ViT-L/14 encodes image + OCR text -> HateCLIPper-style fused joint embedding -> trainable projection MLP outputs classification logit + retrieval embedding. Losses = **retrieval-guided contrastive (RGCL, InfoNCE-style)** pulling anchor toward FAISS-retrieved pseudo-gold positive (same-label NN) + pushing hard negatives (opposite-label NN) + BCE. FAISS DB refreshed each epoch. Inference: **kNN majority vote over K=10 retrieved neighbors** (supports adding new labeled examples at test time, no retraining).

## Key Results
HatefulMemes AUROC 87.0/acc 78.8 (beats HateCLIPper 85.5, LLaVA-13B, Flamingo-80B). HarMeme AUROC 91.8. Cross-domain kNN transfer 66.6 AUROC (>> zero-shot LMMs). A ~few-M-param head beats fine-tuned multi-billion-param LMMs.

## Limitations / Failure Modes
**Meme (single-image) domain only** — no temporal/moment, no audio/prosody, text is OCR only, English-only, no crosslingual/Chinese. Implicit hate not explicitly targeted.

## Reusable Ingredients
Modality-agnostic core (exactly what this project ports to video): (1) FAISS mining of hard negatives (opposite-label NN) + pseudo-gold positives (same-label NN), per-epoch refresh. (2) Hybrid loss = retrieval-guided contrastive (triplet/InfoNCE) + BCE over frozen encoder + light MLP. (3) kNN majority-vote inference over swappable DB (training-free updates, cross-domain transfer). (4) HateCLIPper element-wise fusion. (5) RPG removes need for augmentation.

## Relevance to This Project
**Direct base method + namesake** (this repo IS the RGCL codebase). INSPIRATION, not prior art in our field: retrieval-guided contrastive + kNN has NOT been done in hateful-VIDEO — our opening. Bare port to video inherits RGCL's (not our) novelty, so we must add a video-specific axis (cross-lingual retrieval, temporal/moment retrieval, audio-prosody, implicit-hate retrieval).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
- Base method → hateful-video adaptation (this project). Modality-agnostic; swap encoder front-end.

## Abstract (original)

> Hateful memes have emerged as a significant concern on the Internet. Detecting hateful memes requires the system to jointly understand the visual and textual modalities. Our investigation reveals that the embedding space of existing CLIP-based systems lacks sensitivity to subtle differences in memes that are vital for correct hatefulness classification. We propose constructing a hatefulness-aware embedding space through retrieval-guided contrastive training. Our approach achieves state-of-the-art performance on the HatefulMemes dataset with an AUROC of 87.0, outperforming much larger fine-tuned large multimodal models. We demonstrate a retrieval-based hateful memes detection system, which is capable of identifying hatefulness based on data unseen in training. This allows developers to update the hateful memes detection system by simply adding new examples without retraining, a desirable feature for real services in the constantly evolving landscape of hateful memes on the Internet.

