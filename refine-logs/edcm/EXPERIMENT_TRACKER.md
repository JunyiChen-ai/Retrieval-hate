# EDCM-RGCL Iteration 2 Experiment Tracker

**Status date:** 2026-07-10  
**Execution authorization:** A0 only; A0 is complete and the frozen EDCM route stopped.  
**Hard lock:** `EDCM-A0-DECISION-v1` is verified `STOP`, so A1--A3 remain locked. Only video-level binary labels are gold; segment gold does not exist.

| Run ID / exact namespace | Stage | Purpose | Dataset / seed / arm | Decisive metrics or checks | Resources / dependency | Priority | Status | Expected artifact / note |
|---|---|---|---|---|---|---|---|---|
| `EDCM-A0-REUSE-AUDIT-v1` | A0.0 | Validate SSR OOF recipe/hash suitability | MHC + MHC_zh, OOF S0 | config/code/data/fold/output hashes; train-only disjointness; full ranking; exact top-20 vote reproduction; no segment gold | CPU 4 / 16 GB; first | MUST | **COMPLETED_GO · 12710** | All frozen provenance/vote checks passed; `reuse_audit.json` payload verified |
| `EDCM-A0-REACH-MHC-v1` | A0.1 | Top-64, <=2-swap frozen-geometry reachability | MHC, OOF S0 | support >=80%; reachable >=28; delta acc >=.050; delta mF1 >=.050; all hashes pass | CPU 4 / 16 GB; `afterok` reuse audit | MUST | **COMPLETED_STOP · 12711** | support 202/549=0.3679; reachable 15; delta acc +.0273; delta mF1 +.0394; provenance only gate passed |
| `EDCM-A0-REACH-MHC_zh-v1` | A0.1 | Top-64, <=2-swap frozen-geometry reachability | MHC_zh, OOF S0 | support >=80%; reachable >=29; delta acc >=.050; delta mF1 >=.050; all hashes pass | CPU 4 / 16 GB; `afterok` reuse audit; may parallel MHC | MUST | **COMPLETED_STOP · 12712** | support 364/579=0.6287; reachable 22; delta acc +.0380; delta mF1 +.0444; provenance only gate passed |
| `EDCM-A0-DECISION-v1` | A0.2 | Joint pre-MLLM cost decision | both datasets | every A0 cell true; `edcm_mllm_calls_before_decision=0`; provenance verifies | CPU 4 / 16 GB; `afterok` both reach jobs | MUST | **COMPLETED_STOP · 12713** | Authoritative row/witness/metric recomputation passed; eight binding dataset cells failed; `A1_unlocked=false` |
| `EDCM-A1-TEACHER-FREEZE-v1` | A1 | Freeze model/OCR/prompts/schema/canonicalization | train only | manifest hashes; label-blind schema; no timestamp/span/segment fields | CPU; only after verified A0 GO | CONDITIONAL | **NOT_RUN_LOCKED_A0_STOP** | No teacher process was started |
| `EDCM-A1-SMOKE-Q25VL7B-v1` | A1 | Strict parser/throughput/fallback smoke | frozen train subset | 2 prompts x 2 orders; deterministic JSON; failures map to missing | 1 A100 / 8 CPU / 64 GB; after teacher freeze | CONDITIONAL | **NOT_RUN_LOCKED_A0_STOP** | No MLLM call was made |
| `EDCM-A1-EXTRACT-MHC-v1` | A1 | Extract all train coalition distributions | MHC train | complete/reliable signature records + confidence/fallback | 1 A100 / 8 CPU / 64 GB; after smoke; sequential | CONDITIONAL | **NOT_RUN_LOCKED_A0_STOP** | No teacher cache exists |
| `EDCM-A1-EXTRACT-MHC_zh-v1` | A1 | Extract all train coalition distributions | MHC_zh train | complete/reliable signature records + confidence/fallback | 1 A100 / 8 CPU / 64 GB; after MHC extraction; sequential | CONDITIONAL | **NOT_RUN_LOCKED_A0_STOP** | No teacher cache exists |
| `EDCM-A1-GATE-MHC-v1` | A1 | Teacher-active + proxy-strength gate | MHC five-fold OOF | coverage/Wilson; reliable 8+8; TV/`R`; reachable `DeltaD`; shared proxy grid | CPU 8 / 32 GB; after both extractions | CONDITIONAL | **NOT_RUN_LOCKED_A0_STOP** | Predecessor failed |
| `EDCM-A1-GATE-MHC_zh-v1` | A1 | Teacher-active + proxy-strength gate | MHC_zh five-fold OOF | same frozen gates as MHC | CPU 8 / 32 GB; after both extractions | CONDITIONAL | **NOT_RUN_LOCKED_A0_STOP** | Predecessor failed |
| `EDCM-A1-DECISION-v1` | A1 | Joint A1 decision | both datasets | every reliability/activity/direction/proxy gate passes | CPU; after both A1 gates | CONDITIONAL | **NOT_RUN_LOCKED_A0_STOP** | Predecessor failed |
| `EDCM-A2-{MHC,MHC_zh}-S0-{REMOVE,LABEL,PROXY,FULL,SHUFFLE,NOISE1X,NOISE2X}-v1` | A2 | Seed-0 causal mechanism controls | 2 datasets x 7 frozen arms | dev kNN acc/mF1; >=.010 full-control gaps; purity/wrong-neighbour; monotone noise | 1 A100 / 16 CPU / 120 GB each; strictly sequential; after A1 GO | CONDITIONAL | **NOT_RUN_LOCKED_A0_STOP** | Exact expansion is 14 runs; test not evaluated |
| `EDCM-A2-DECISION-v1` | A2 | Joint seed-0 gate | both datasets | every control, kNN-locus, corruption, isolation gate passes | CPU; after 14 arms | CONDITIONAL | **NOT_RUN_LOCKED_A0_STOP** | No A2 checkpoint exists |
| `EDCM-A3-{MHC,MHC_zh}-S{0,1,2}-{REMOVE,FULL,SHUFFLE,NOISE1X}-v1` | A3 | Paired final endpoint and causal controls | 2 datasets x 3 seeds x 4 arms | final test acc/mF1; seed deltas; removal/shuffle; noise survival | 1 A100 / 16 CPU / 120 GB each; sequential; after A2 GO | CONDITIONAL | **NOT_RUN_LOCKED_A0_STOP** | No test evaluation was run |
| `EDCM-A3-{MHC,MHC_zh}-S0-NOISE2X-v1` | A3 | Twice-noise endpoint check | 2 datasets, seed 0 | must not exceed NOISE1X in either metric | same budget; after A2 GO | CONDITIONAL | **NOT_RUN_LOCKED_A0_STOP** | No test evaluation was run |
| `EDCM-A3-FINAL-STATS-v1` | A3 | Verify complete objective | all A3 outputs | +.030/+.030 vs moving bar; 3/3 signs; hierarchical bootstrap; Holm; full mechanism contract | CPU-only; after all final runs | CONDITIONAL | **NOT_RUN_LOCKED_A0_STOP** | Global project target remains unmet |

## A0 Submission DAG (for the later authorized executor)

```text
EDCM-A0-REUSE-AUDIT-v1
          |
          +--afterok--> EDCM-A0-REACH-MHC-v1 ----+
          |                                       +--afterok--> EDCM-A0-DECISION-v1
          +--afterok--> EDCM-A0-REACH-MHC_zh-v1 -+

A1 MLLM jobs: absent unless A0_DECISION is verified GO.
```

## Current Evidence State

- SSR's terminal failure is authoritative negative evidence: its optimistic one-neighbour candidate universe reached only 2/7 MHC and 3/15 MHC-ZH MI/SC errors and failed every `+.05/+.05` gate. EDCM therefore changes the correctable unit to a bounded list-level two-swap screen before spending teacher compute.
- Existing SSR OOF assets passed strict reuse audit job 12710; provenance was not the blocker.
- The frozen two-swap cost screen failed on both datasets. MHC support is 36.79%, with 15 reachable errors and +.0273/+.0394 oracle gains; MHC-ZH support is 62.87%, with 22 reachable errors and +.0380/+.0444 gains. Every binding geometry/headroom gate failed except provenance.
- Joint job 12713 independently reconstructed every authoritative ranking, canonical witness, metric, and gate before writing `STOP`. It records zero EDCM MLLM/OCR/teacher calls, `A1_unlocked=false`, and `A2_A3_locked=true`.
- The EDCM route is stopped before teacher spending. The global target remains active and unmet; a successor must change the video-level correctable mechanism rather than tune EDCM prompts/teacher/loss or assume segment gold.
