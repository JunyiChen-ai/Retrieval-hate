---
type: paper
node_id: paper:vivierardisson2026_differentiable_knapsack_topk
title: "Differentiable Knapsack and Top-k Operators via Dynamic Programming"
authors: ["Germain Vivier-Ardisson", "Michaël E. Sander", "Axel Parmentier", "Mathieu Blondel"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2601.21775"
  doi: null
  s2: null
tags: ["differentiable-top-k", "discrete-selection", "dynamic-programming", "mechanism-inspiration", "NOT-core-hateful-video"]
added: 2026-08-07T12:59:58Z
---

# Differentiable Knapsack and Top-k Operators via Dynamic Programming

## One-line thesis
Differentiable relaxations of knapsack and top-k operators via smoothed dynamic programs, with entropy shown to be the unique permutation-equivariant regularizer.

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

> Knapsack and Top-k operators are useful for selecting discrete subsets of variables. However, their integration into neural networks is challenging as they are piecewise constant, yielding gradients that are zero almost everywhere. In this paper, we propose a unified framework casting these operators as dynamic programs, and derive differentiable relaxations by smoothing the underlying recursions. On the algorithmic side, we develop efficient parallel algorithms supporting both deterministic and stochastic forward passes, and vector-Jacobian products for the backward pass. On the theoretical side, we prove that Shannon entropy is the unique regularization choice yielding permutation-equivariant operators, and characterize regularizers inducing sparse selections. Finally, on the experimental side, we demonstrate our framework on a decision-focused learning benchmark, a constrained dynamic assortment RL problem, and an extension of discrete VAEs.

