# EDCM-RGCL A0: pre-MLLM frozen-geometry cost screen

**Date:** 2026-07-10  
**Route:** EDCM-RGCL / Iteration 2  
**Outcome:** **STOP before any teacher spend**  
**Global +3 acc / +3 macro-F1 target:** not met

## Why this experiment existed

SSR failed because its one-neighbour relation universe touched too few unique OOF errors. EDCM proposed a denser train-only MLLM coalition signature acting through the full memory-list gradient. Before paying for an MLLM teacher, A0 asked whether the frozen strongest RGCL OOF geometry even contained enough nearby video-label-only list corrections.

The preregistered screen used the exact full-video top-20 arithmetic cosine vote, searched only ranks 1--64, and allowed at most two optimistic label-oracle substitutions. A query could remove only opposite-video-label keys from top 20 and add only same-video-label keys from ranks 21--64. This is a reachability/cost diagnostic, not segment supervision and not a learned EDCM result.

## Supervision and isolation

- Sole gold: parent-video binary label.
- Segment gold: nonexistent and unused.
- K4 comparator labels: inherited parent-video labels only, never segment annotations.
- Inputs: five-fold train OOF full-video rankings; no validation/test source file or label endpoint was read.
- MLLM/OCR/teacher calls or caches: zero.
- A1--A3: never started.

## Code and provenance review

The A0 implementation is `scripts/analysis/edcm_a0.py` with `configs/edcm/edcm_v1.json` and `scripts/slurm/edcm_a0_cpu.sbatch`.

An independent reviewer found and closed three HIGH-level integrity gaps across two repair rounds:

1. replaced check-then-`os.replace` publication with persistent exclusive namespace locks and no-clobber publication;
2. bound joint decision to the current audit, frozen implementation, supervision/zero-call fields, and independently recomputed numeric gates;
3. bound every output row to its authoritative ranking and recomputed the exact canonical witness.

Final review result: **0 HIGH / 0 CRITICAL**. Sanity/freezing jobs 12708--12709 passed.

## SLURM execution

| Job | Run | State | Key result |
|---:|---|---|---|
| 12708 | `EDCM-A0-SANITY-FREEZE-v1` | COMPLETED | generated implementation/config hashes |
| 12709 | `EDCM-A0-SANITY-VERIFY-v1` | COMPLETED | sanity `GO` |
| 12710 | `EDCM-A0-REUSE-AUDIT-v1` | COMPLETED | reuse audit `GO` |
| 12711 | `EDCM-A0-REACH-MHC-v1` | COMPLETED | dataset `STOP` |
| 12712 | `EDCM-A0-REACH-MHC_zh-v1` | COMPLETED | dataset `STOP` |
| 12713 | `EDCM-A0-DECISION-v1` | COMPLETED | joint `STOP`; A1 locked |

All jobs used `HateVideo`, `slurmpartition`, 4 CPU / 16 GB, no GPU, no `--time`. Initial `JobHeldUser` states cleared automatically; no job was manually released.

## Results

| Gate | MHC | Required | Pass | MHC-ZH | Required | Pass |
|---|---:|---:|---|---:|---:|---|
| All-video support | 202/549 = 0.3679 | >=0.80 | no | 364/579 = 0.6287 | >=0.80 | no |
| Unique reachable errors | 15 | >=28 | no | 22 | >=29 | no |
| Oracle accuracy gain | +0.0273 | >=+0.050 | no | +0.0380 | >=+0.050 | no |
| Oracle macro-F1 gain | +0.0394 | >=+0.050 | no | +0.0444 | >=+0.050 | no |
| Fold/output/vote/metric provenance | verified | all pass | yes | verified | all pass | yes |

Baseline and oracle endpoints:

| Dataset | Baseline acc | Baseline mF1 | Oracle acc | Oracle mF1 | Baseline errors | Reachable errors |
|---|---:|---:|---:|---:|---:|---:|
| MHC | 0.7687 | 0.6934 | 0.7960 | 0.7328 | 127 | 15 |
| MHC-ZH | 0.7599 | 0.7194 | 0.7979 | 0.7638 | 139 | 22 |

Joint artifact `artifacts/edcm/v1/A0_DECISION.json` has payload SHA-256 `8256b454a7ae4a8a4cff1726851bd450999f8c5c814047dedfc9ffdff1e3c159`, `A1_unlocked=false`, and zero MLLM/OCR/teacher counts.

## Conclusion and anti-repeat rule

EDCM fails its registered pre-MLLM necessary cost screen on both datasets. The failure is stronger than a negative teacher result: the optimistic frozen-geometry two-swap oracle itself is below every required support/headroom threshold. No prompt, teacher, reliability, or loss tuning can change this registered A0 bound.

Do not rerun this route with tuned top-64 depth, swap count, thresholds, prompt, model size, or EDCM auxiliary weight. The next hypothesis must alter the video-level correctable unit and preregister a new two-dataset, accuracy-plus-macro-F1 oracle screen before creating any teacher artifact. It must continue to treat all future MLLM outputs as weak train-only pseudo-signals and must never assume segment gold.

