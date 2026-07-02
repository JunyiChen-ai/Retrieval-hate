---
type: paper
node_id: paper:wang2024_multihateclip_multilingual_benchmark
title: "MultiHateClip: A Multilingual Benchmark Dataset for Hateful Video Detection on YouTube and Bilibili"
authors: ["Han Wang", "Tan Rui Yang", "Usman Naseem", "Roy Ka-Wei Lee"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2408.03468"
  doi: null
  s2: null
tags: ["hateful-video", "benchmark-dataset", "multilingual", "crosslingual", "chinese", "bilibili", "youtube", "gender-based-hate", "VLM", "GPT-4V", "Qwen-VL", "late-fusion", "segment-level", "target-victim", "modality-attribution", "implicit-hate", "ACM-MM-2024", "is_core_hateful_video"]
added: 2026-07-01T09:39:43Z
---

# MultiHateClip: A Multilingual Benchmark Dataset for Hateful Video Detection on YouTube and Bilibili

## One-line thesis
MultiHateClip is the first cross-cultural English+Chinese (YouTube+Bilibili) benchmark of 2,000 gender-based hateful short videos with fine-grained 3-class labels plus segment/target/modality annotations, and benchmarking shows current VLMs and fusion models struggle to separate hateful from offensive content and perform markedly worse on Chinese than English.

## Problem / Gap
Prior hate-video datasets (HateMM, OffVidPT) are English/Western-only, small, coarse binary, lacking segment/target/modality labels. No multilingual (esp. Chinese) benchmark existed.

## Method
Dataset: 80 EN/ZH gender-hate lexicon pairs -> keyword search YouTube+Bilibili (clips <=60s) -> ChatGPT prefilter -> 2,000 (1,000/lang) manually annotated (Hateful/Offensive/Normal + segment timestamps + target victim + contributing modality). Benchmarks: mBERT/GPT-4V-text/Qwen (text), MFCC/Wav2Vec2-BERT (audio), ViViT/ViT+LSTM (vision), GPT-4V/Qwen-VL (VLM), mBERT⊙MFCC⊙ViViT late-fusion (M1). Multiclass + collapsed-binary. **No contrastive, no retrieval.**

## Key Results
2,000 videos (EN: 82H/256O/662N; ZH: 128H/194O/678N). EN best multiclass M-F1 **0.63** (GPT-4V), binary 0.79. ZH best multiclass **0.50** (M1), binary 0.78. Clear **EN>ZH gap (0.63 vs 0.50)**. Hateful-class F1 often collapses to 0.00 (5/11 EN models). YOLOv3 fails on 704 ZH vs 289 EN videos (Western detector bias). kappa 0.62 EN / 0.51 ZH.

## Limitations / Failure Modes
Few hateful positives, imbalanced toward Normal, gender-hate only, annotators all Asian 18-24, Western-biased tooling fails on Chinese, off-the-shelf baselines, no implicit-hate handling, no temporal model trained despite segment labels.

## Reusable Ingredients
(1) **MHC dataset** (2,000 EN+ZH, 3-class + segment-timestamp + target + modality) — crosslingual/Chinese + temporal + modality-attribution anchor. (2) Split (70/10/20) + Macro-F1 multiclass/binary protocol. (3) M1 late-fusion + GPT-4V 4-frame prompt baselines. (4) 80 lexicon pairs + ChatGPT prefilter pipeline. (5) Documented hard cases: hateful-vs-offensive confusion + implicit hate.

## Relevance to This Project
Core anchor: only widely-used EN+ZH hateful-video benchmark; defines crosslingual/Chinese axis (ZH under-solved 0.50 vs 0.63). NO baseline uses retrieval or contrastive — RGCL/RA-HMD sweet spot. Segment/target/modality labels give temporal + attribution eval targets. Published split + Tables 8/9 = reproducible baseline to beat.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Abstract (original)

> Hate speech is a pressing issue in modern society, with significant effects both online and offline. Recent research in hate speech detection has primarily centered on text-based media, largely overlooking multimodal content such as videos. Existing studies on hateful video datasets have predominantly focused on English content within a Western context and have been limited to binary labels (hateful or non-hateful), lacking detailed contextual information. This study presents MultiHateClip1 , an novel multilingual dataset created through hate lexicons and human annotation. It aims to enhance the detection of hateful videos on platforms such as YouTube and Bilibili, including content in both English and Chinese languages. Comprising 2,000 videos annotated for hatefulness, offensiveness, and normalcy, this dataset provides a cross-cultural perspective on gender-based hate speech. Through a detailed examination of human annotation results, we discuss the differences between Chinese and English hateful videos and underscore the importance of different modalities in hateful and offensive video analysis. Evaluations of state-of-the-art video classification models, such as VLM, GPT-4V and Qwen-VL, on MultiHateClip highlight the existing challenges in accurately distinguishing between hateful and offensive content and the urgent need for models that are both multimodally and culturally nuanced. MultiHateClip represents a foundational advance in enhancing hateful video detection by underscoring the necessity of a multimodal and culturally sensitive approach in combating online hate speech.

