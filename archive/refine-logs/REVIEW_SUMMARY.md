# Review Summary

**Problem:** Meaningfully and novelly integrate an MLLM into RGCL and obtain ≥+0.030 accuracy and ≥+0.030 macro-F1 on at least two datasets, paired seeds 0/1/2, with statistical and remove/shuffle causality gates.  
**Initial approach:** SSR-MemRGCL: video-level labels sign real hard-pair edges; a frozen MLLM supplies only stance–target–mechanism relation pseudo-types that shape final kNN memory geometry.  
**Date:** 2026-07-10  
**Rounds:** 4 / 5  
**Final score:** 9.03 / 10  
**Final verdict:** READY for experiment handoff

## Problem Anchor

- Only video-level binary labels are gold; no segment-level gold exists.
- All MLLM semantic outputs are confidence-bearing weak/privileged train-only pseudo-signals with deterministic no-edge fallback and no test-time annotation role.
- Success remains two datasets × three paired seeds × both final kNN metrics ≥+3 points, with corrected statistics and causal controls.

## Round-by-Round Resolution Log

| Round | Main concern | What changed | Solved? | Remaining risk |
|---|---|---|---|---|
| 1 | Direction ambiguity, cross-seed drift, no reliability/noise null, ornamental types | Directed seed-isolated arcs; agreement reliability; MI+/SC−; matched shuffle/noise; bounded graph | Yes | split/headroom rules |
| 2 | B1 could touch validation; accuracy-only headroom; dataset-adaptive families | Strict train-only OOF; dual-metric oracle; common cross-dataset family; exact-null stop | Yes | mixed OOF/full geometry |
| 3 | B1 mixed geometries; shuffle direction dependence unclear | Fully OOF diagnostic universe; canonical record+direction-mask shuffle; Wilson audit rule | Yes | empirical signal existence |
| 4 | No proposal blocker | READY; freeze exact MI event/statistic for handoff | Yes | B0/B1 may fast-fail |

## Overall Evolution

- The method contracted from three relation names to two operative constraint families and one familiar ranking loss.
- MLLM responsibility is narrow: label-blind train-only cross-video relation typing; video labels alone determine sign.
- OOF diagnostics, exact canonical shuffle and calibrated noise isolate semantic pair assignment from labels, difficulty, graph statistics, missingness and compute.
- Drift was removed: no validation/test relation endpoints and no cross-seed active arcs.

## Final Status

- **Anchor:** preserved (10/10).
- **Focus:** one dominant contribution; no module pile.
- **Modernity:** appropriately frontier-aware and non-decorative.
- **Strongest parts:** exact final-memory locus; no new parameters/inference path; rigorous causal null; explicit fast-fail gates.
- **Remaining weaknesses:** novelty is narrow and evidence-dependent; common-family reliability, dual-metric headroom and exact-null feasibility are unproven until B0/B1.
