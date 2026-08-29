# Experiment Tracker

**Scope now**: B0/B1 only. `LOCKED` rows must not launch until the named upstream decision artifact is `GO`.  
**Global invariant**: only video-level labels are gold; every MLLM relation is a train-only weak pseudo-signal; all compute is SLURM-only, no `--time`.

| Run ID | Milestone | Purpose / System | Split | Decisive output | Priority | Status | Depends on / stop note |
|---|---|---|---|---|---|---|---|
| `SSR-B0-FREEZE-v1` | M0 | Freeze config, folds, schema, prompts, events, hashes | train metadata | frozen manifest; disjoint split assertions | MUST | GO (jobs 12686 bootstrap, 12687 frozen; static 12688 GO) | Config `b32e57…718e`; dev/test ID-only, no labels consumed; strict JSON/payload/BA/four-call static checks GO |
| `SSR-B0-OOF-MHC-F0-S0` | M0 | MHC-EN exact comparator OOF fold 0 | train F0 query / F1-4 memory | checkpoint, ranking, prediction manifest | MUST | COMPLETED (12691) | query 110 / memory 439; fixed epoch 25; all assertions pass |
| `SSR-B0-OOF-MHC-F1-S0` | M0 | MHC-EN exact comparator OOF fold 1 | train F1 / rest | same | MUST | COMPLETED (12692) | fixed epoch 25; hash/fold/vote checks pass |
| `SSR-B0-OOF-MHC-F2-S0` | M0 | MHC-EN exact comparator OOF fold 2 | train F2 / rest | same | MUST | COMPLETED (12693) | same |
| `SSR-B0-OOF-MHC-F3-S0` | M0 | MHC-EN exact comparator OOF fold 3 | train F3 / rest | same | MUST | COMPLETED (12694) | same |
| `SSR-B0-OOF-MHC-F4-S0` | M0 | MHC-EN exact comparator OOF fold 4 | train F4 / rest | same | MUST | COMPLETED (12695) | same |
| `SSR-B0-OOF-MHC_zh-F0-S0` | M0 | MHC-ZH exact comparator OOF fold 0 | train F0 / rest | checkpoint, ranking, prediction manifest | MUST | COMPLETED (12696) | fixed epoch 28; hash/fold/vote checks pass |
| `SSR-B0-OOF-MHC_zh-F1-S0` | M0 | MHC-ZH exact comparator OOF fold 1 | train F1 / rest | same | MUST | COMPLETED (12697) | same |
| `SSR-B0-OOF-MHC_zh-F2-S0` | M0 | MHC-ZH exact comparator OOF fold 2 | train F2 / rest | same | MUST | COMPLETED (12698) | same |
| `SSR-B0-OOF-MHC_zh-F3-S0` | M0 | MHC-ZH exact comparator OOF fold 3 | train F3 / rest | same | MUST | COMPLETED (12699) | same |
| `SSR-B0-OOF-MHC_zh-F4-S0` | M0 | MHC-ZH exact comparator OOF fold 4 | train F4 / rest | same | MUST | COMPLETED (12700) | same |
| `SSR-B0-MINE-MHC-v1` | M0 | Exact MI/SC candidates, references, events, allocator | train OOF only | <=1,200 canonical pairs + event ledger | MUST | COMPLETED (12701) | 1,200 pairs / 1,219 arcs; MI events 6, SC 18 |
| `SSR-B0-MINE-MHC_zh-v1` | M0 | Exact MI/SC candidates, references, events, allocator | train OOF only | <=1,200 canonical pairs + event ledger | MUST | COMPLETED (12702) | 1,200 pairs / 1,222 arcs; MI events 6, SC 29 |
| `SSR-B0-SMOKE-Q25VL7B-v1` | M0 | Frozen 16-pair/dataset, four-call extraction smoke | train pairs only | replay/input/schema assertions | MUST | SKIPPED BY B1 STOP | All-candidates oracle upper bound already fails; extraction cannot change the bound |
| `SSR-B0-REL-MHC-Q25VL7B-v1` | M0 | Full four-call weak relation extraction | MHC train pairs | canonical call/record/failure ledger | MUST | SKIPPED BY B1 STOP | Avoided scientifically futile 7B spend after verified necessary-gate failure |
| `SSR-B0-REL-MHC_zh-Q25VL7B-v1` | M0 | Full four-call weak relation extraction | MHC_zh train pairs | canonical call/record/failure ledger | MUST | SKIPPED BY B1 STOP | same |
| `SSR-B1-AGG-MHC-v1` | M1 | Reliability, family predicates, fallback and audit pack | MHC OOF | counts, rho/missing coverage, audit IDs | MUST | NOT RUN — MOOT AFTER NECESSARY GATE FAIL | relation quality cannot overcome failed all-candidates headroom ceiling |
| `SSR-B1-AGG-MHC_zh-v1` | M1 | Reliability, family predicates, fallback and audit pack | MHC_zh OOF | counts, rho/missing coverage, audit IDs | MUST | NOT RUN — MOOT AFTER NECESSARY GATE FAIL | same |
| `SSR-B1-AUDIT-MHC-MI-{A1,A2,ADJ}` | M1 | Blinded pair-level MI audit; no segment annotation | 80 MHC accepted MI records | valid/n=80, Wilson 95% interval | MUST | NOT RUN — MOOT | failed optimistic headroom ceiling precedes relation/audit quality |
| `SSR-B1-AUDIT-MHC-SC-{A1,A2,ADJ}` | M1 | Blinded pair-level SC audit | 80 MHC accepted SC records | valid/n=80, Wilson interval | MUST | NOT RUN — MOOT | same |
| `SSR-B1-AUDIT-MHC_zh-MI-{A1,A2,ADJ}` | M1 | Blinded pair-level MI audit | 80 MHC_zh accepted MI records | valid/n=80, Wilson interval | MUST | NOT RUN — MOOT | same |
| `SSR-B1-AUDIT-MHC_zh-SC-{A1,A2,ADJ}` | M1 | Blinded pair-level SC audit | 80 MHC_zh accepted SC records | valid/n=80, Wilson interval | MUST | NOT RUN — MOOT | same |
| `SSR-B1-COND-MHC-MI-v1` | M1 | Reduced vs +MI grouped logistic, 10,000 canonical permutations | all MHC MI candidate arcs | `DeltaNLL`, `DeltaAUC`, `p_raw` | MUST | NOT RUN — MOOT | no accepted subset can pass headroom |
| `SSR-B1-COND-MHC-SC-v1` | M1 | Reduced vs +SC grouped logistic | all MHC SC candidate arcs | same | MUST | NOT RUN — MOOT | same |
| `SSR-B1-COND-MHC_zh-MI-v1` | M1 | Reduced vs +MI grouped logistic | all MHC_zh MI candidate arcs | same | MUST | NOT RUN — MOOT | same |
| `SSR-B1-COND-MHC_zh-SC-v1` | M1 | Reduced vs +SC grouped logistic | all MHC_zh SC candidate arcs | same | MUST | NOT RUN — MOOT | same |
| `SSR-B1-ORACLE-MHC-MI-v1` | M1 | Correct only event-touched baseline errors | MHC OOF predictions | acc/mF1 oracle gains | MUST | UPPER-BOUND FAIL (12704) | all-candidate upper Δacc +0.0036 / ΔmF1 +0.0048; 2 touched |
| `SSR-B1-ORACLE-MHC-SC-v1` | M1 | Same for SC | MHC OOF | acc/mF1 oracle gains | MUST | UPPER-BOUND FAIL (12704) | +0.0128 / +0.0176; 7 touched |
| `SSR-B1-ORACLE-MHC_zh-MI-v1` | M1 | Same for MI | MHC_zh OOF | acc/mF1 oracle gains | MUST | UPPER-BOUND FAIL (12704) | +0.0052 / +0.0065; 3 touched |
| `SSR-B1-ORACLE-MHC_zh-SC-v1` | M1 | Same for SC | MHC_zh OOF | acc/mF1 oracle gains | MUST | UPPER-BOUND FAIL (12704) | +0.0259 / +0.0307; 15 touched |
| `SSR-B1-SHUFFLE-OOF-MHC-v1` | M1 | Exact canonical record derangement | MHC OOF graph | assignment + all equality checks | MUST | NOT RUN — LOCKED | requires nonempty common family; upper-bound common set is empty |
| `SSR-B1-SHUFFLE-OOF-MHC_zh-v1` | M1 | Exact canonical record derangement | MHC_zh OOF graph | assignment + all equality checks | MUST | NOT RUN — LOCKED | same |
| `SSR-B1-DECISION-v1` | M1 | Machine-check all B1 gates and common family | train OOF only | `B1_DECISION.json: GO|STOP` | MUST | STOP (12705) | Verified preflight 12704: empty common family under optimistic all-candidates dual-metric oracle upper bound |
| `SSR-B2-MHC-S0-{REMOVE,LABEL,FULL,SHUFFLE,NOISE1,NOISE2}-v1` | M2 | Seed-0 dev causal-geometry arms | MHC train/dev | dev kNN acc/mF1 + topology | CONDITIONAL | LOCKED | Requires B1 GO and seed-0 exact shuffle; FULL `>=+0.010` over REMOVE/LABEL/SHUFFLE in both metrics |
| `SSR-B2-MHC_zh-S0-{REMOVE,LABEL,FULL,SHUFFLE,NOISE1,NOISE2}-v1` | M2 | Seed-0 dev causal-geometry arms | MHC_zh train/dev | same | CONDITIONAL | LOCKED | same |
| `SSR-B3-MHC-S{0,1,2}-{REMOVE,LABEL,FULL,SHUFFLE,NOISE1}-v1` | M3 | Frozen final paired campaign | MHC train/dev/test | final acc/mF1, paired CIs/causality | CONDITIONAL | LOCKED | Requires both B2 dataset gates; no adaptive test rerun |
| `SSR-B3-MHC_zh-S{0,1,2}-{REMOVE,LABEL,FULL,SHUFFLE,NOISE1}-v1` | M3 | Frozen final paired campaign | MHC_zh train/dev/test | same | CONDITIONAL | LOCKED | final target requires both datasets, all seeds, both metrics |

## Gate Snapshot

| Gate | Required evidence | Current state |
|---|---|---|
| B0 integrity | strict train-only OOF, deterministic hashes, no label-bearing MLLM payload, no segment gold | STATIC + 10 OOF + mining PASS; relation extraction intentionally skipped after necessary B1 STOP |
| B1 reliability | `N_accepted>=80` and audit Wilson lower bound `>=0.80` for a common family on both datasets | NOT RUN — cannot rescue failed headroom upper bound |
| B1 conditional information | `DeltaNLL>0`, `DeltaAUC>0`, Holm `p<0.05` | NOT RUN — cannot rescue failed headroom upper bound |
| B1 headroom | oracle acc and macro-F1 gains each `>=+0.050` | FAIL (12704): optimistic all-candidate upper bound fails all 4 cells |
| B1 causal-null feasibility | exact canonical shuffle verified on both dataset OOF graphs | NOT RUN — empty common family at prior necessary gate |
| B2 unlock | `B1_DECISION=GO` | LOCKED (`B1_DECISION=STOP`, 12705) |
| Goal completion | two datasets x seeds 0/1/2 x final acc/mF1 `>=+0.030`, statistics and causal controls | NOT MET |
