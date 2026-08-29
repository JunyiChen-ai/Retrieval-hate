# EDCM-RGCL A0 Initial Experiment Results

**Date:** 2026-07-10 (Pacific/Auckland)  
**Plan:** `refine-logs/edcm/EXPERIMENT_PLAN.md`  
**Authorization used:** A0 only  
**Final A0 decision:** **STOP**

## Contract

- The only gold supervision used was the parent video's binary label.
- No segment-level gold exists or was assumed. K4 subclip labels occur only in the audited historical comparator recipe and are inherited parent-video labels, not segment annotations.
- A0 consumed strict five-fold train OOF full-video rankings. Validation/test files were not read as sources or endpoints.
- No MLLM, OCR, coalition, proxy, or teacher cache was read, generated, or called. The joint decision records `edcm_mllm_calls_before_decision=0`.

## Implementation and integrity

- Independent code review initially found two HIGH issues (non-overwrite TOCTOU and insufficient decision lineage/gate verification), then one HIGH on authoritative row/witness binding.
- Fixes added persistent `O_CREAT|O_EXCL` namespace locks, fail-if-exists publication, current-audit and implementation binding, complete supervision/zero-call checks, authoritative ranking reconstruction, canonical witness recomputation, and independent metric/gate recomputation.
- Final targeted review: **0 CRITICAL / 0 HIGH**.
- SLURM sanity/freezing jobs: 12708 (`NEEDS_FREEZE`, produced hashes) and 12709 (`GO`, verified frozen config and implementation).

## Results by milestone

### A0.0 Reuse audit — GO

Job 12710 verified the frozen repository/config/data/fold/output hashes, full fold-local rankings, exact top-20 arithmetic similarity vote, repository vote agreement, train-only query/key partitions, video-label-only supervision, and zero teacher artifacts.

### A0.1 Frozen-geometry reachability — STOP on both datasets

| Dataset | Job | N | Baseline acc / mF1 | Support | Reachable errors | Oracle acc / mF1 | Oracle delta acc / mF1 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MHC | 12711 | 549 | 0.7687 / 0.6934 | 202 (0.3679) | 15 / required 28 | 0.7960 / 0.7328 | +0.0273 / +0.0394 | STOP |
| MHC-ZH | 12712 | 579 | 0.7599 / 0.7194 | 364 (0.6287) | 22 / required 29 | 0.7979 / 0.7638 | +0.0380 / +0.0444 | STOP |

For each dataset, provenance passed but all four binding geometry/headroom cells failed: all-video support `<0.80`, reachable count below the preregistered minimum, accuracy delta `<0.050`, and macro-F1 delta `<0.050`.

### A0.2 Joint decision — STOP

Job 12713 independently reconstructed every query from the authoritative frozen ranking, recomputed exact votes, support, the minimum-swap / maximum-margin / lexicographically tied canonical witness, both metric vectors, and every gate. It wrote:

- `decision=STOP`
- `all_binding_gates_pass=false`
- `A1_unlocked=false`
- `A2_A3_locked=true`
- `edcm_mllm_calls_before_decision=0`
- `segment_gold_exists=false`, `segment_gold_used=false`

Decision payload SHA-256: `8256b454a7ae4a8a4cff1726851bd450999f8c5c814047dedfc9ffdff1e3c159`.

## Interpretation and next action

This is a failure of the frozen EDCM correctable-unit/headroom prerequisite, not a provenance failure and not evidence about teacher quality. Even an optimistic video-label oracle with two exact list substitutions cannot reach the preregistered `+0.05/+0.05` screen on either dataset. Therefore teacher extraction cannot rescue this registered route and was correctly skipped.

Do not tune the EDCM prompt, teacher scale, confidence filter, loss weight, search depth, or swap count on this route. A successor hypothesis must change the video-level causal correctable mechanism while keeping the only-gold/video-label contract and must pass a new preregistered two-dataset dual-metric oracle cost screen before any MLLM call.

The global project target remains active and unmet.

## Evidence

- `artifacts/edcm/v1/a0/reuse_audit.json`
- `artifacts/edcm/v1/a0/MHC/{reachability.jsonl,metrics.json,manifest.json}`
- `artifacts/edcm/v1/a0/MHC_zh/{reachability.jsonl,metrics.json,manifest.json}`
- `artifacts/edcm/v1/A0_DECISION.json`
- `slurm/logs/edcm_a0_cpu_{12708,12709,12710,12711,12712,12713}.out`

