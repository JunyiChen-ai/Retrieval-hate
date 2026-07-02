---
type: paper
node_id: paper:yang2025_revealing_temporal_label
title: "Revealing Temporal Label Noise in Multimodal Hateful Video Classification"
authors: ["Shuonan Yang", "Tailin Chen", "Rahul Singh", "Jiangbei Yue", "Jianbo Jiao", "Zeyu Fu"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2508.04900"
  doi: null
  s2: null
tags: ["hateful-video-detection", "temporal-label-noise", "moment-level", "HateMM", "MultiHateClip", "multimodal", "BERT", "MFCC", "ViT", "weak-supervision", "segment-level", "benchmark-analysis", "MUWS-2025", "University-of-Exeter", "is_core_hateful_video"]
added: 2026-07-01T09:39:47Z
---

# Revealing Temporal Label Noise in Multimodal Hateful Video Classification

## One-line thesis
Video-level hateful labels in current benchmarks are temporally noisy because hateful videos contain large stretches of non-hateful content, and training/evaluating on timestamp-trimmed clean hateful segments dramatically improves classification, proving coarse labels are a systematic source of label noise that motivates temporally-aware models.

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

> The rapid proliferation of online multimedia content has intensified the spread of hate speech, presenting critical societal and regulatory challenges. While recent work has advanced multimodal hateful video detection, most approaches rely on coarse, video-level annotations that overlook the temporal granularity of hateful content. This introduces substantial label noise, as videos annotated as hateful often contain long non-hateful segments. In this paper, we investigate the impact of such label ambiguity through a fine-grained approach. Specifically, we trim hateful videos from the HateMM and MultiHateClip English datasets using annotated timestamps to isolate explicitly hateful segments. We then conduct an exploratory analysis of these trimmed segments to examine the distribution and characteristics of both hateful and non-hateful content. This analysis highlights the degree of semantic overlap and the confusion introduced by coarse, video-level annotations. Finally, controlled experiments demonstrated that time-stamp noise fundamentally alters model decision boundaries and weakens classification confidence, highlighting the inherent context dependency and temporal continuity of hate speech expression. Our findings provide new insights into the temporal dynamics of multimodal hateful videos and highlight the need for temporally aware models and benchmarks for improved robustness and interpretability. Code and data are available at https://github.com/Multimodal-Intelligence-Lab-MIL/HatefulVideoLabelNoise.

