---
type: paper
node_id: paper:muhumuza2026_majority_vote_silences
title: "Majority Vote Silences Minority Values: Annotator Disagreement at the Hate/Offensive Boundary in HateXplain"
authors: ["Joshua Muhumuza", "Joab Ezra Agaba", "Mercy Amiyo"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2606.28772"
  doi: null
  s2: null
tags: ["hate-speech", "annotator-disagreement", "label-noise", "hate-offensive-boundary", "inspiration"]
added: 2026-08-07T13:00:22Z
---

# Majority Vote Silences Minority Values: Annotator Disagreement at the Hate/Offensive Boundary in HateXplain

## One-line thesis
Majority-vote aggregation at the hate/offensive boundary in HateXplain silences minority annotator values, making the boundary label itself contested rather than noisy-but-recoverable.

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

> Hate speech annotation pipelines routinely collapse annotator disagreement into majority vote labels before training. We show that this aggregation is not neutral: 42.6% of all annotator disagreement in HateXplain concentrates specifically at the hate/offensive boundary, a pattern consistent with annotators applying different thresholds for where hate begins (chi-squared = 135.199, df = 2, p < 0.0001). Both a hard-label BERT model (Model A) and a soft-label model (Model B) drop 22 percentage points in accuracy from agreed posts (~80%) to disagreement posts (~58%), confirmed at p < 0.0001. A per-annotator multi-head model (Model C) widens this gap further to 28 points while collapsing offensive disagreement accuracy to 0.245. Critically, Model A expresses significantly higher confidence on boundary case errors than Model C (0.710 vs. 0.495, p < 0.0001), meaning standard evaluation metrics will not detect the failure. Three downstream interventions of increasing sophistication all fail to recover boundary accuracy. We argue the problem is structural. Majority vote presents a contested judgment as ground truth, and models inherit that false certainty. The intervention must be upstream in annotation design.

