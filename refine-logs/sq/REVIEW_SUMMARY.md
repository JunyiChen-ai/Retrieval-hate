# SQ-RGCL Review Summary

**Problem:** Make an MLLM a meaningful, removable train-only component that substantially improves final ordinary full-video RGCL kNN.  
**Initial approach:** Whole-video presentation/context posterior drives class-conditional quotient geometry.  
**Date:** 2026-07-11  
**Rounds:** 4 / 5  
**Final score:** 9.12 / 10  
**Final verdict:** READY for experiment planning, not performance success.

## Problem Anchor

The immutable anchor is preserved verbatim in `PROBLEM_ANCHOR.md` and every proposal/refinement. Only parent-video binary labels are gold; no segment/timestamp/span/localization gold exists. Final success remains two datasets × paired seeds 0/1/2 × at least +0.030 accuracy and +0.030 macro-F1 with statistics and MLLM removal/permutation attribution.

## Round-by-Round Resolution

| Round | Main concern | Focused change | Resolution |
|---:|---|---|---|
| 1 | proxy P0 replaced learned gate; nuisance leakage; generic NCA; weak top-20 alignment | restored strict-OOF +.05/+.05 SQ-0; presentation schema/audits; vote-exposed crossed ranking; explicit P2/P4/prior controls | partial, 6.88 REVISE |
| 2 | q→y and minority-share gates ill-posed; pilot pairs not closed; free exposure parameters | q→y diagnostic; environment×class ESS; graph-closed common-edge pilot; repository rank/signed exposure | partial, 7.90 REVISE |
| 3 | harmonic tail not evaluator-backed; edge pseudoreplication; hub selection; aggregation undefined | exact top-20-only exposure; pre-call anchor power gate; representative closure/IPW; frozen mean-q/min-confidence artifact | resolved, 8.46 REVISE pending recheck |
| 4 | closure audit | all scientific blockers closed; only experiment-plan details remain | READY, 9.12 |

## Overall Evolution

- The generic environment-weighted NCA became one coupled crossed triplet tied to actual harmful top-20 vote exposure.
- The route recovered an actual learned strict-OOF final-kNN capacity gate rather than relying on proxy correlations.
- Nuisance validity is based on provenance, blind presentation audit, semantic exclusions, and positivity/ESS—not a false independence assumption.
- Pilot inference treats anchors, not edges, as independent and stops before calls if a representative powered closure cannot fit the cap.
- MLLM use remains a single confidence-bearing train-only posterior; test inference is unchanged.

## Final Status

- Anchor: preserved.
- Focus: tight—one posterior, one loss, one inference path.
- Modernity: appropriate privileged MLLM relation teacher; no forced extra module.
- Novelty: narrow but method-level; must stay phrased as MLLM-defined presentation crossing × exact-vote-exposed RGCL ranking.
- Remaining risk: actual q/archive provenance or SQ-0 may fail; no performance has been measured.
