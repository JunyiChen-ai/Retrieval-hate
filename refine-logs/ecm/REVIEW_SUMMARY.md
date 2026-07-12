# ECM-RGCL Review Summary

**Problem:** integrate an MLLM meaningfully and novelly into final RGCL and obtain `>=+0.030` accuracy and macro-F1 on MHC-EN and MHC-ZH, paired seeds 0/1/2.  
**Initial approach:** strict-OOF MLLM whole-video failure-mode posteriors controlling a mode-level projected/minimax RGCL update.  
**Date:** 2026-07-11  
**Rounds:** 1 / 5  
**Final score:** 4.98 / 10  
**Final verdict:** **RETHINK — ECM ABANDONED**

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causally removable part of hateful-video RGCL and do not stop until the unchanged ordinary full-video train-memory kNN endpoint improves by at least `+0.030` absolute in both accuracy and macro-F1 on MHC-EN and MHC-ZH, paired seeds `0/1/2`, under the complete statistical and mechanism-attribution protocol.
- **Must-solve bottleneck:** The MLLM must diagnose dense whole-video failure mechanisms from a strict-OOF, label-blind prediction trace for every training video, and those modes must directly alter the optimizer of the shared final RGCL embedding. The route must reach errors outside SSR's sparse edge universe and EDCM's frozen top-64/two-swap universe, without becoming sample reweighting, a router, a renamed GroupDRO/JTT/EIIL method, or generic gradient surgery.
- **Non-goals:** No localization endpoint; no segment weighting or segment loss; no rationale/schema/summary concatenation; no teacher-selected key, score fusion, reranking, veto, auxiliary/native-head claim, test-time MLLM/mode/teacher, MoE/router, model/data/epoch/ensemble scaling, or rescue of frozen SSR/EDCM/CTE routes. SQ-RGCL is still at formal S0/S1 plan status and is not declared a performance failure here.
- **Constraints:** The only gold supervision that exists is the parent video's binary label. There is no segment, timestamp, span, stance, target, mechanism, rationale, or localization gold. MLLM modes are confidence-bearing train-only weak/privileged pseudo-signals. The teacher sees neither gold label nor any correctness/error/loss/true-class-margin indicator. Every train video is processed under the same strict-OOF rule; validation/test have no teacher, trace, mode, or extra head. All eventual computation must use SLURM in `HateVideo`, with at most 2 GPU / 16 CPU / 128 GB and no `--time`.
- **Success condition:** Relative to `max(historical strongest non-MLLM point, paired same-seed strongest non-MLLM mean)`, FULL gains `>=+0.030` accuracy and `>=+0.030` macro-F1 on both datasets; all three paired deltas are positive; hierarchical paired-bootstrap lower bounds exceed zero and four dataset-by-metric tests survive Holm correction. FULL significantly beats REMOVE and within-train MODE-SHUFFLE, and must beat margin-bin, standard GroupDRO, JTT, EIIL and generic gradient-surgery controls under matched capacity. Only final ordinary full-video kNN counts.

## Round-by-Round Resolution Log

| Round | Main reviewer concerns | Author processing | Result | Remaining risk |
|---:|---|---|---|---|
| 1 | soft mode risks + QP reduce to dynamic sample reweighting and generic gradient surgery; raw-gradient constraints do not constrain AdamW; teacher can reconstruct error propensity; exact-vote geometry and selection were incomplete | Accepted the mechanism critique. Preserved a non-canonical proximal-bank sketch only as a future distinct-hypothesis pointer, then terminated ECM rather than rename it | **RETHINK / ABANDONED** | A new route still must prove non-reducibility, actual-update semantics, and value beyond scalar error propensity |

## What Was Resolved

- Supervision fidelity and no-segment-gold audit passed.
- The teacher input contained no direct gold/correctness/error/loss/true-class-margin field, but the stronger “correctness firewall” wording was rejected because the MLLM may infer error propensity.
- SQ ordering is now terminally known from later authoritative evidence: job 12730 wrote S0 STOP for provenance/QC, with no learned SQ performance result.
- ECM did not proceed to code, SLURM, cache generation or teacher calls.

## Final Status

- **Anchor:** supervision and endpoint preserved; core mechanism failed the non-reweighting/non-generic-surgery anchor.
- **Focus:** terminally tight—one frozen route rejected, not expanded.
- **Modernity:** train-only MLLM role was appropriate, but operationally only semantic pseudo-group annotation.
- **Strongest valid lesson:** a teacher posterior over whole-video residual modes is not enough; if downstream optimization is GroupDRO/JTT/PCGrad/CAGrad-equivalent, the MLLM integration is not method-novel.
- **No-segment-gold:** PASS. Only parent-video binary labels are gold; no segment/timestamp/span/localization annotation exists or was used.
