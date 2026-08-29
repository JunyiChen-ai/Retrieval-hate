---
type: paper
node_id: paper:gupta2022_adima_abuse_detection
title: "ADIMA: Abuse Detection In Multilingual Audio"
authors: ["Vikram Gupta", "Rini Sharon", "Ramit Sawhney", "Debdoot Mukherjee"]
year: 2022
venue: "arXiv"
external_ids:
  arxiv: "2202.07991"
  doi: null
  s2: null
tags: ["audio-abuse", "multilingual", "cross-lingual", "Indic", "wav2vec2", "XLSR", "CLSRIL-23", "log-mel-VGG", "AudioSet", "prosody", "benchmark-dataset", "ICASSP-2022", "ASR-free", "zero-shot-transfer", "profanity-detection", "audio-only", "adjacent", "NOT-core-hateful-video"]
added: 2026-07-01T09:39:52Z
---

# ADIMA: Abuse Detection In Multilingual Audio

## One-line thesis
ADIMA shows abusive/profane content in spoken social-media audio can be detected end-to-end in the audio domain (no ASR needed), and releases a 10-language Indic profanity-detection audio benchmark with monolingual and zero-shot cross-lingual baselines. ADJACENT audio-axis, audio-only, NOT hateful-video.

## Problem / Gap
_TODO._

## Method
_TODO._

## Key Results
_TODO._

## Assumptions
_TODO._

## Limitations / Failure Modes
_TODO._

## Reusable Ingredients
_TODO._

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
_TODO._

## Abstract (original)

> Abusive content detection in spoken text can be addressed by performing Automatic Speech Recognition (ASR) and leveraging advancements in natural language processing. However, ASR models introduce latency and often perform sub-optimally for profane words as they are underrepresented in training corpora and not spoken clearly or completely. Exploration of this problem entirely in the audio domain has largely been limited by the lack of audio datasets. Building on these challenges, we propose ADIMA, a novel, linguistically diverse, ethically sourced, expert annotated and well-balanced multilingual profanity detection audio dataset comprising of 11,775 audio samples in 10 Indic languages spanning 65 hours and spoken by 6,446 unique users. Through quantitative experiments across monolingual and cross-lingual zero-shot settings, we take the first step in democratizing audio based content moderation in Indic languages and set forth our dataset to pave future work.

