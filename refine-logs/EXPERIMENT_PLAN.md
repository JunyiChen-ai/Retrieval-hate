# Experiment Plan

**Problem**: Meaningfully and novelly integrate an MLLM into RGCL/RA-HMD hateful-video detection and obtain a substantial improvement in the unchanged final train-memory kNN classifier.  
**Method Thesis**: A frozen label-blind MLLM supplies only reliable train-only stance-target-mechanism relation types for real RGCL hard pairs; video-level labels alone sign directed constraints, and one parameter-free ranking loss repairs the exact geometry used by final kNN.  
**Date**: 2026-07-10  
**Active scope**: Implement and execute **B0/B1 only**. B2/B3 are locked specifications, not current authorization.

## 0. Frozen Contract and Stop Meaning

1. The **only gold supervision is the video-level binary label**. No segment-level gold exists. No segment boundary, segment hate label, stance, target, proposition, mechanism, rationale, or localization span may be treated as gold.
2. MLLM relation fields are confidence-bearing **weak, privileged, train-only pseudo-signals**. Human relation checks below are a blinded aggregate audit only; they are not training labels, are never attached to segments, and cannot be used to edit individual graph records.
3. No validation/test video is an endpoint of B0/B1 relation extraction. Validation is first used only in B2 model selection; test is first consumed only after B2 is frozen and B3 is launched.
4. Low-agreement, unclear, invalid, missing, or failed MLLM outputs deterministically become one canonical `missing/no_edge` record. They never inherit a label-derived semantic type.
5. Edge polarity comes only from video labels. The MLLM never receives labels, predictions, ranks, margins, fold IDs, seed IDs, or intended loss role.
6. The final target remains unchanged: on MHC-EN and MHC-ZH, full SSR must improve accuracy and macro-F1 each by at least `+0.030` over `max(historical strongest point, paired same-seed non-MLLM mean)`, with seeds 0/1/2, 3/3 positive deltas, corrected statistics, and remove/shuffle causality. B0/B1 passing is only permission to try B2, not target completion.
7. All computation, including CPU analysis, runs through SLURM in `conda activate HateVideo`; no `#SBATCH --time`; aggregate concurrent use must stay within 2 GPU, 16 CPU, and 128 GB.

Historical lower bounds remain MHC-EN `0.7888 acc / 0.7262 mF1` (`full`, `lambda_seg=0.5`) and MHC-ZH `0.8255 / 0.7875` (`milmax`, `lambda_seg=0.5`). Their binding final targets are at least `0.8188 / 0.7562` and `0.8555 / 0.8175`, respectively, and move upward if paired baselines are stronger.

## 1. Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| **C1 (primary)**: reliable MLLM MI+/SC- relation types identify correctable RGCL neighbour failures beyond labels, cosine, rank, margin, fold, and error status. | Without conditional information, SSR is P2/P4 under a new name. | On both MHC-EN and MHC-ZH, at least one **common** family has >=80 accepted canonical records, audit Wilson lower bound >=0.80, positive held-out `DeltaNLL` and `DeltaAUC`, Holm-corrected one-sided permutation `p<0.05`, and >=+0.05 oracle gain in **both** accuracy and macro-F1. | B0, B1 |
| **C2 (supporting, still unproven)**: correct semantic assignment, not label/difficulty/degree structure, can causally repair the final kNN geometry. | This makes the MLLM meaningful rather than decorative. | B1 proves an exact canonical semantic derangement is feasible without relaxing any matching constraint; later, and only if B1 passes, B2 full beats remove, label-only, and shuffle in dev kNN accuracy and macro-F1. | B1, locked B2 |
| **Anti-claim**: gains arise from segment annotation, extra capacity, test-time judging, prompt/model scaling, or more training. | Any one invalidates the intended contribution. | No segment gold anywhere; no new trainable parameters; fixed 7B teacher and four calls; unchanged epochs/checkpoint/vote; no relation artifacts at inference; matched workload controls. | all |

## 2. Paper Storyline

- **Main paper must prove**: strict train-only conditional information/headroom; causal final-kNN geometry; final two-dataset three-seed `+3/+3` result.
- **Appendix can support**: prompt/order agreement, schema/fallback breakdown, relation subtype counts, BIP diagnostics, gradient/topology diagnostics, and corruption curves.
- **Intentionally cut**: segment localization evaluation, test-time MLLM, generated summaries/counterfactuals, relation adapters, routers, larger teachers, prompt sweeps, dataset-specific relation families, SSR+EDCM stacking, and extra epochs/data.

## 3. Frozen B0/B1 Geometry

### 3.1 Dataset and fold construction

- Dataset codes are `MHC` (MHC-EN) and `MHC_zh` (MHC-ZH).
- Use only each dataset's current **train** split. Assert that every emitted ID is in train and that `train_ids`, `dev_ids`, and `test_ids` are pairwise disjoint.
- Create five stratified outer folds with `StratifiedKFold(n_splits=5, shuffle=True, random_state=20260710)`, after sorting records by canonical video ID. Save the complete ID-to-fold map and SHA-256.
- For outer fold `f`, its videos are queries. Train the exact dataset-specific comparator on the other four folds only. The outer fold is never used for optimization, checkpoint choice, thresholds, or prompt decisions.
- To avoid outer-fold label leakage through checkpoint selection, use a **frozen epoch index**, not the outer fold: MHC-EN epoch 25 and MHC-ZH epoch 28, inherited from the historical strongest recipes. All other optimizer/backbone/loss settings remain those recipes. These OOF heads are diagnostics, not replacement final baselines.
- MHC-EN OOF uses existing `lambda_seg=0.5, seg_mode=full`; MHC-ZH uses `lambda_seg=0.5, seg_mode=milmax`. Existing subclip handling inherits only the parent video label and is not segment gold.
- Project outer queries and four-fold memory with that fold's single checkpoint. L2-normalize embeddings and use `IndexFlatIP`, `topk=20`, arithmetic rank weights `w_r=21-r`, similarity weighting, and threshold at vote score `0`, matching `retrieve_evaluate_RAC_` plus `compute_metrics_retrieval(... majority_voting="arithmetic", topk=20, use_sim=True)`.
- Self matches are impossible because query and memory folds are disjoint. No embedding, neighbour, score, or label from one outer-fold geometry may be mixed with another.

For query `i`, let the full memory ranking be `R_i=(r_1,...,r_M)` in descending cosine with ties resolved by canonical neighbour ID. Let `N20(i)` be the first 20. The exact vote is

```text
V(S) = sum_{r=1}^{20} (21-r) * cosine(i,S_r) * (2*y_{S_r}-1)
yhat(S) = 1[V(S) >= 0]
```

The denominator used in the repository is positive and therefore omitted without changing the decision.

### 3.2 Exact candidate arcs

For every query, define two pre-MLLM directed candidate lists:

- `C_SC(i)`: the first at most three members of `N20(i)` whose video label differs from `y_i`.
- `C_MI(i)`: scanning ranks 21 onward, the first at most three members whose video label equals `y_i`. A candidate is retained only when `N20(i)` contains at least one opposite-label reference.

For each MI candidate `j`, choose one reference

```text
n*(i,j) = argmin over n in N20(i), y_n != y_i of
          (abs(cos(i,j)-cos(i,n)), rank_i(n), SHA256(dataset|fold|i|j|n)).
```

This choice is deterministic and never chosen by whether it corrects the vote.

Candidate allocation is frozen:

1. Sort queries by `(baseline_error first, SHA256("ssr-v1"|dataset|fold|query_id))`.
2. Pass A adds each error query's highest-ranked unused `C_SC` arc.
3. Pass B cycles candidate position 1, 2, 3; within each query it adds unused SC then MI arcs, following the same query order.
4. Canonicalize each unordered pair as `(min_id,max_id)`, prompt it once, and retain the selected direction mask. Duplicate arcs do not consume budget twice.
5. Stop at 1,200 unique canonical pairs per dataset OOF universe. Do not fill from dev/test or relax the allocator.

### 3.3 Exact neighbour events

These definitions are frozen before extraction and are computed from the same fold geometry:

```text
Y_SC(i,j)=1 iff
  baseline yhat(N20(i)) != y_i,
  j in C_SC(i),
  remove j, promote original rank-21 item, re-sort by original cosine,
  and the exact vote becomes y_i.

Y_MI(i,j,n*)=1 iff
  baseline yhat(N20(i)) != y_i,
  j in C_MI(i), n*=n*(i,j),
  replace n* with j, re-sort the 20 items by original cosine,
  and the exact vote becomes y_i.
```

`Y_MI` is a counterfactual **missing-helpful-neighbour event**, not a segment label and not an observed gold relation. If the baseline is correct, or the specified single replacement does not flip it, the event is 0. No alternate reference may be searched after seeing the outcome.

## 4. B0: Freeze, Build, and Extract (MUST-RUN)

### 4.1 Frozen MLLM interface

Teacher: `Qwen/Qwen2.5-VL-7B-Instruct`. Each video contributes four uniform full-video frames plus timestamped ASR/OCR, head/tail truncated deterministically to 2,048 Unicode characters. There is no segment annotation or span target.

Each canonical pair receives exactly four deterministic calls:

```text
P0_AB, P0_BA, P1_AB, P1_BA
do_sample=false, temperature=0, top_p=1, max_new_tokens=256
```

`P0/P1` are two frozen paraphrases; `AB/BA` are the two orders. Save model, processor, input-builder, prompt, schema, source-artifact, and decoding hashes. BA outputs are canonicalized back to AB, including swapping per-video fields.

Allowed JSON fields and values are exactly those in `FINAL_PROPOSAL.md`. There is no hate verdict, confidence self-score, label, free rationale, localization, or segment field.

Field reliability is modal agreement across the four canonicalized calls. Ties resolve to `unclear`. A field is usable only at agreement `rho_field in {0.75,1.0}`. Family creation uses deterministic predicate priority:

- **MI+**: equal video labels; `mechanism_relation=same`; then first usable true predicate in `[topic_surface_relation=different, target_relation=different]`.
- **SC-**: different video labels; `topic_surface_relation=same`; then first usable true predicate in `[stance_relation=opposed, target_relation=different, proposition_relation in {opposed,unrelated}]`.

Record weight `rho` is the minimum agreement of fields actually used by its predicate. Any required `unclear`, agreement below 0.75, invalid JSON, missing call, or failed call yields canonical `missing/no_edge` for all directions. Only video labels apply the plus/minus sign after the label-blind relation record is frozen.

### 4.2 B0 sanity gate

- Run the complete frozen pipeline on the first 16 hash-selected canonical pairs per dataset, all four calls.
- This checks ID/split assertions, order canonicalization, JSON parsing, four-call completeness, deterministic replay, and no-label prompt payload. It is not a prompt-selection experiment.
- **Pass**: identical rerun hashes, zero split leakage, zero label/prediction/rank fields in serialized MLLM inputs, and all valid parses conform to the schema. Parse failures are allowed only as recorded `missing/no_edge`; no repair prompt is issued.
- **Fail**: any leakage, nondeterministic serialization, BA canonicalization error, or output overwrite. Fix implementation and rerun the same ID; do not alter scientific thresholds or compare prompts.

## 5. B1: Reliability, Conditional Information, Headroom, and Null Feasibility (MUST-RUN)

### 5.1 Accepted count and blinded audit

For each dataset x family cell (`MHC/MI`, `MHC/SC`, `MHC_zh/MI`, `MHC_zh/SC`):

- `N_accepted` is the number of unique canonical records satisfying the frozen family predicate and reliability rule; report directed arcs and unique query coverage separately.
- The cell fails immediately if `N_accepted < 80`.
- Audit exactly 80 accepted records selected by ascending `SHA256("audit-v1"|dataset|family|canonical_pair_id)`.
- Two EN/ZH-capable annotators independently see the raw pair evidence and the claimed relation predicate, while blinded to video labels, baseline outputs, ranks, event values, family loss role, and MLLM identity. They mark `valid/invalid/unclear`; `unclear` counts invalid. A third adjudicates disagreements.
- Compute the two-sided 95% Wilson interval on adjudicated valid count (`z=1.959963984540054`). **Pass only if the lower bound is >=0.80.** Report numerator, denominator, point precision, lower/upper bounds, and agreement. Audit judgments are aggregate diagnostics only and are never loaded by graph construction.

### 5.2 Frozen conditional statistic

Analyze MI and SC separately using **all selected pre-MLLM directed candidate arcs**, not only accepted arcs. For each dataset x family:

- Outcome is the exact `Y_MI` or `Y_SC` above.
- Reduced features are: cosine; normalized full-memory rank `rank/M`; absolute exact vote margin `abs(V)/sum_r w_r*abs(cos_r)`; query video label; one-hot outer OOF fold; baseline-error indicator.
- Full features are the reduced features plus `A_F`, a binary indicator that the canonical record is an accepted reliable MI or SC relation for that directed arc.
- Create diagnostic grouped folds by `g(i)=uint64(SHA256("cond-v1"|dataset|query_id)[:16]) mod 5`; every arc of a query stays together. For each `g`, fit on the other four groups and predict the held-out group. Continuous-feature standardization is fit on the training groups only.
- Both models use deterministic L2 logistic regression: intercept on, `C=1.0`, `solver=lbfgs`, `max_iter=5000`, `tol=1e-9`, no class weighting. A one-class outcome or convergence failure is an automatic failure.
- Clip held-out probabilities to `[1e-6,1-1e-6]`. Pool all held-out predictions and compute:

```text
NLL = mean[-Y*log(p) -(1-Y)*log(1-p)]
DeltaNLL = NLL_reduced - NLL_full
DeltaAUC = AUC_full - AUC_reduced
```

The one-sided permutation test uses `DeltaNLL` as the preregistered statistic; `DeltaAUC>0` is an additional sign gate. At canonical-record level, permute the whole pseudo-relation record plus direction mask within the exact signature

```text
(video-label polarity, direction-mask pattern,
 sorted multiset over directions of [cosine decile, rank bin,
 margin quartile, baseline-error stratum, outer fold])
```

where cosine deciles and margin quartiles are frozen from all candidate arcs, and rank bins are `[1-5,6-10,11-15,16-20,>20]`. Both directions move together. Refit both models for 10,000 seeded permutations (`seed=20260710+p`).

```text
p_raw = (1 + count(DeltaNLL_perm >= DeltaNLL_observed)) / 10001
```

Apply Holm step-down correction across the four preregistered dataset x family cells, including failed/degenerate cells as non-significant. A cell passes only when `DeltaNLL>0`, `DeltaAUC>0`, and `p_Holm<0.05`.

### 5.3 Dual-metric oracle headroom

For each dataset x family independently, define touched queries as baseline OOF errors having at least one accepted reliable family arc with that family's exact event equal to 1. Set the oracle prediction to `y_i` for all and only those queries; leave every other OOF prediction unchanged. Concatenate the five outer-fold query predictions once per video and recompute accuracy and macro-F1 from video-level labels.

Report baseline/oracle values, absolute gains, touched errors, unique-query coverage, and required realized fraction `0.03/oracle_gain`. **Pass only if both oracle accuracy gain and oracle macro-F1 gain are >=+0.050.** This is a feasibility ceiling, never a model result.

### 5.4 Common family and exact canonical shuffle feasibility

First form

```text
F_common = intersection across MHC and MHC_zh of families passing
           accepted-count + Wilson + conditional + acc-headroom + mF1-headroom.
```

The family set is common and frozen across datasets; there is no EN-only/ZH-only fallback. Empty `F_common` stops SSR.

For each dataset's complete OOF graph, solve the preregistered binary assignment/derangement over **all** canonical records, including missing records. Every source record must move to a different canonical target pair in the same dataset OOF universe. Source and target must match label polarity, direction-mask pattern, and every per-direction outer-fold/cosine-decile/rank-bin/margin-error stratum; thus each projected direction stays comparable to its own fold-local geometry even when a canonical pair has directions mined from two folds. Projecting records to all target directions must reproduce exactly: query out-degree vector, neighbour indegree vector, common-family counts, reliability histogram, missing count, and canonical direction coupling. Original-pair assignment is forbidden.

- **Pass**: solver status `OPTIMAL` or a separately verified feasible complete assignment with every equality assertion true and zero fixed points.
- **Fail/stop**: `INFEASIBLE`, timeout without a verified assignment, or any failed equality. Bins, degrees, direction masks, family counts, and missingness may not be relaxed or approximately matched.

Only if both dataset graphs have a verified exact shuffle is `B1_DECISION=GO` written. Otherwise SSR stops and B2/B3 remain locked.

## 6. Experiment Blocks

### B0: Strict train-only geometry and weak-relation extraction

- **Claim tested**: the proposed interface can be instantiated without leakage, segment gold, or silent missing-signal repair.
- **Compared systems**: none; this is frozen artifact construction.
- **Metrics**: split assertions, artifact/hash completeness, schema validity, agreement/reliability distribution, accepted/fallback counts.
- **Success criterion**: B0 sanity passes and complete <=1,200-pair/dataset OOF artifacts exist.
- **Failure interpretation**: interface is not safely executable; do not interpret relation quality.
- **Target**: appendix artifact/QC table.
- **Priority**: MUST-RUN.

### B1: Conditional information, headroom, and exact-null feasibility

- **Claim tested**: C1 and the feasibility part of C2.
- **Compared systems**: reduced conditional model vs +reliable-family indicator; observed relations vs exact canonical shuffle feasibility.
- **Metrics**: accepted count, Wilson audit, `DeltaNLL`, `DeltaAUC`, Holm-corrected permutation p, accuracy/macro-F1 oracle headroom, BIP assertions.
- **Success criterion**: a nonempty common family passes every gate on both datasets and exact shuffle is feasible on both.
- **Failure interpretation**: relation signal is too sparse/unreliable/redundant, cannot support +3/+3, or cannot be causally isolated. Stop SSR without scaling/tuning.
- **Target**: main-paper feasibility table if later stages pass; otherwise a documented negative result.
- **Priority**: MUST-RUN.

### B2: Seed-0 causal geometry (LOCKED)

- **Unlock**: only a signed, hash-complete `B1_DECISION=GO`.
- **Runs**: per dataset seed 0, `REMOVE`, `LABEL`, `FULL`, `SHUFFLE`, `NOISE1`, `NOISE2`; identical initialization, data, batches, epochs, checkpoint rule, and tuple workload. `NOISE1=e_hat`; `NOISE2=2e_hat`.
- **Success**: FULL exceeds REMOVE, LABEL, and SHUFFLE by >=+0.010 in dev kNN accuracy **and** macro-F1, repairs neighbour topology, and does not merely redistribute native-head vs memory accuracy. Full-seed exact shuffle infeasibility stops before training.
- **Priority**: CONDITIONAL; do not implement/run during B0/B1.

### B3: Frozen final test (LOCKED)

- **Unlock**: both dataset B2 gates pass before any test result is inspected.
- **Runs**: MHC/MHC_zh x seeds 0/1/2 x `REMOVE,LABEL,FULL,SHUFFLE,NOISE1`; seed 0 additionally retains `NOISE2` from B2. The exact common family, hashes, graphs, teacher, prompts, loss, statistics, seeds, and comparator recipe are frozen.
- **Success**: target contract in Section 0, including `+0.030` in both metrics on both datasets, 3/3 positive deltas, mean+-std, Holm-corrected hierarchical paired-bootstrap lower bounds >0, and significant full-minus-remove/full-minus-shuffle costs in both metrics.
- **Priority**: CONDITIONAL; one terminal test campaign, no adaptive rerun.

## 7. Exact Run Order and Run IDs

Current authorized order is intentionally compact:

| Order | Exact Run ID(s) | Purpose | Decision Gate | Resources / Cost |
|---:|---|---|---|---|
| 1 | `SSR-B0-FREEZE-v1` | Write config, fold/prompt/schema hashes and split assertions. | Hash-complete, no overlap. | SLURM CPU, 1 CPU/8 GB, <1 h. |
| 2 | `SSR-B0-OOF-MHC-F0-S0` ... `F4-S0`; `SSR-B0-OOF-MHC_zh-F0-S0` ... `F4-S0` | Ten exact OOF geometry heads and rankings. | Each fold has one query prediction per train ID; no cross-geometry record. | 1 A100/16 CPU/120 GB each; estimated 5-15 GPU-h total; sequential because of CPU cap. |
| 3 | `SSR-B0-MINE-MHC-v1`, `SSR-B0-MINE-MHC_zh-v1` | Apply exact SC/MI events and 1,200-pair allocator. | Deterministic replay/hash, <=1,200 pairs/dataset. | SLURM CPU, 4 CPU/16 GB, <1 h total. |
| 4 | `SSR-B0-SMOKE-Q25VL7B-v1` | Frozen 16-pair/dataset four-call smoke. | B0 sanity in Section 4.2. | 1 A100/8 CPU/48 GB, <1 GPU-h. |
| 5 | `SSR-B0-REL-MHC-Q25VL7B-v1`, `SSR-B0-REL-MHC_zh-Q25VL7B-v1` | Complete four-call relation extraction. | Four-call/missing record for every canonical pair. | 1 A100/8 CPU/48 GB each; estimated 8-20 GPU-h total after measured smoke throughput; at most two together. |
| 6 | `SSR-B1-AGG-MHC-v1`, `SSR-B1-AGG-MHC_zh-v1` | Canonicalize, compute reliability/family/fallback reports and audit packs. | Counts/hashes complete; cells with <80 fail. | SLURM CPU, 4 CPU/16 GB, <1 h. |
| 7 | `SSR-B1-AUDIT-{MHC,MHC_zh}-{MI,SC}-A1`, `...-A2`, `...-ADJ` | Blinded human audit IDs. | Wilson lower bound >=0.80 per cell. | 160-320 records; estimated 12-20 person-h; packaging/scoring through SLURM CPU. |
| 8 | `SSR-B1-COND-MHC-MI-v1`, `SSR-B1-COND-MHC-SC-v1`, `SSR-B1-COND-MHC_zh-MI-v1`, `SSR-B1-COND-MHC_zh-SC-v1` | Frozen grouped logistic and 10,000 canonical permutations. | Positive deltas and Holm `p<0.05`. | SLURM CPU, up to 16 CPU/64 GB; estimated 1-8 wall-h/cell, run one cell at a time. |
| 9 | `SSR-B1-ORACLE-MHC-{MI,SC}-v1`, `SSR-B1-ORACLE-MHC_zh-{MI,SC}-v1` | Dual-metric ceiling. | Each surviving cell >=+0.05 acc and mF1. | SLURM CPU, 2 CPU/8 GB, <1 h. |
| 10 | `SSR-B1-SHUFFLE-OOF-MHC-v1`, `SSR-B1-SHUFFLE-OOF-MHC_zh-v1` | Exact canonical derangement. | Verified feasible both datasets; no relaxation. | SLURM CPU, up to 16 CPU/64 GB; solver-dependent 0.5-8 wall-h/dataset. |
| 11 | `SSR-B1-DECISION-v1` | Machine-check all manifests and emit STOP/GO. | GO only if every required assertion passes. | SLURM CPU, 1 CPU/8 GB, <1 h. |

Conditional IDs, reserved but locked:

- B2: `SSR-B2-{MHC,MHC_zh}-S0-{REMOVE,LABEL,FULL,SHUFFLE,NOISE1,NOISE2}-v1`.
- B3: for each dataset and seed 0/1/2, `SSR-B3-<DS>-S<seed>-{REMOVE,LABEL,FULL,SHUFFLE,NOISE1}-v1`; no additional arms or rescue runs.

At most one 16-CPU job, or two 8-CPU/one-GPU jobs, may run concurrently. `JobHeldUser` is expected; wait for automatic release and never force-release.

## 8. Planned File and Script Interfaces

These are implementation targets and **do not yet exist unless noted**. B0/B1 implementation must use these names so the tracker is executable.

| Interface | Inputs | Required outputs / assertions |
|---|---|---|
| `configs/ssr/ssr_v1.yaml` | frozen values in this plan | Canonical serialization plus `config_sha256`; no dataset-specific family switch. |
| `scripts/analysis/ssr_make_folds.py --config ... --dataset <DS>` | current split metadata | `artifacts/ssr/v1/folds/<DS>.json`; sorted IDs, labels, fold, split hashes; disjointness assertions. |
| `scripts/slurm/ssr_b01_oof.sbatch` | env `RUN_ID,DATASET,FOLD`; no `--time` | checkpoint, train/query embeddings, `ranking.jsonl`, `predictions.json`, manifest. Must call the current model/vote implementation, with fold-local inputs only. |
| `scripts/analysis/ssr_mine_pairs.py --config ... --dataset <DS>` | five fold rankings/predictions | `pairs.jsonl`, `arcs.jsonl`, `events.jsonl`, allocator report; exact `Y_MI/Y_SC` and reference IDs. |
| `scripts/slurm/ssr_b01_extract.sbatch` | env `RUN_ID,DATASET,MODE=smoke|full`; no `--time` | append-safe `calls.jsonl`, one canonical `records.jsonl`, failure ledger, input/prompt/model hashes; never labels in payload. |
| `scripts/analysis/ssr_aggregate.py` | pairs/arcs/calls | accepted/missing graph, reliability histogram, audit pack, coverage JSON; canonical directions coupled. |
| `artifacts/ssr/v1/audit/<DS>/<F>/` | audit pack | immutable `A1.csv`, `A2.csv`, `ADJ.csv`; columns contain pair ID and validity only, no event/label/rank. |
| `scripts/analysis/ssr_b1_gate.py --gate conditional|oracle` | frozen OOF candidates, records, audit | cell JSON with exact counts, Wilson interval, CV predictions, deltas, permutations, p-values, oracle metrics; deterministic seeds. |
| `scripts/analysis/ssr_exact_shuffle.py --universe oof --dataset <DS>` | all canonical records and graph bins | solver model, assignment, solver log, equality-audit JSON; nonzero exit on infeasible/unverified result. |
| `scripts/analysis/ssr_verify_b1.py --config ...` | all manifests/cell results/shuffles | `artifacts/ssr/v1/B1_DECISION.json` with `GO|STOP`, common families, hashes, and every gate result. |

Future B2/B3 code may add to existing `src/run_rac.py` only the inert CLI `--ssr_graph`, `--ssr_lambda`, `--ssr_margin`, and `--ssr_control`; `--ssr_graph none` must be bit-identical to the comparator. Planned graph loading must assert train-only IDs, dataset/seed/fold hashes, and identical tuple/no-op workload. This code is **not part of current B0/B1 implementation**.

## 9. Control Contract for Locked Training

- **REMOVE**: exact comparator, `lambda_ssr=0`; represents remove-MLLM/baseline.
- **LABEL**: identical signs, family-sized tuple slots, direction masks, query degree, difficulty, and workload, but endpoints chosen without pseudo-relations.
- **FULL**: accepted common-family graph, `margin=0.1`, `lambda_ssr=0.2`, max two arcs/family/query and 1,600 arcs/seed/dataset, eight tuples/surviving family/step.
- **SHUFFLE**: exact canonical assignment; every record moves; all equality constraints verified. Infeasible means stop, never approximate.
- **NOISE1/2**: canonical-record corruption at adjudicated invalid rate `e_hat` / `2e_hat`; missing at observed fallback `m_hat`; direction coupling and no-op workload retained.

No arm changes epochs, batches, optimizer, checkpoint rule, trainable parameters, data volume, retrieval `topk=20`, or final vote. The same initialization checkpoint and batch-order manifest are shared across arms.

## 10. Compute and Data Budget

- **B0/B1 estimated GPU**: 13-36 A100 GPU-hours (10 OOF heads plus <=9,600 deterministic 7B calls; smoke throughput determines the reported estimate, not the gate).
- **B0/B1 estimated CPU**: light preprocessing plus four 10,000-permutation cells and two exact BIPs; approximately 10-50 SLURM wall-hours depending on solver/permutation throughput, never more than 16 CPUs/64 GB for analysis.
- **Human audit**: up to 320 accepted pair relations, approximately 12-20 person-hours. It is pair-level relation QA only; there is no segment labeling.
- **Data preparation**: reuse existing frozen CLIP/video/subclip and ASR/OCR artifacts; create train-only OOF folds, rankings, pairs, relation records, and audit/solver manifests. No new videos or labels.
- **Biggest bottlenecks**: relation coverage under reliability filtering; rare positive neighbour events; exact derangement feasibility.

## 11. Risks and Mitigations

- **Relation semantics are label-redundant**: conditional model and LABEL control; stop on failure.
- **Coverage cannot move overall metrics**: dual-metric +0.05 oracle gate; stop on either metric.
- **MLLM output instability/missingness**: four-call agreement and deterministic no-edge fallback; report all missing records.
- **Cross-language ontology drift**: require the same common family on both datasets; no per-language rescue.
- **Pair-selection confound**: canonical conditional permutation and exact degree/difficulty/missingness-matched shuffle.
- **Test leakage/adaptive test use**: B0/B1 train-only, B2 dev-only, B3 one frozen terminal campaign.
- **Segment-gold drift**: no segment target is present in any schema, event, metric, or audit. Four uniform frames are model inputs, not segment annotations.
- **Resource violation**: every compute entry has an sbatch wrapper without `--time`; scheduler caps and concurrency assertions are checked in manifests.

## 12. Final Checklist

- [x] Claims and anti-claim are frozen.
- [x] No segment-level gold is assumed; only video label signs edges/events.
- [x] MI missing-helpful-neighbour event and reference are exact.
- [x] Conditional statistic, permutation unit, p-value, and Holm family are exact.
- [x] Strict train-only OOF geometry is isolated by fold.
- [x] Common EN/ZH family rule is frozen.
- [x] Accepted-count, Wilson, dual-metric +0.05 headroom, and exact-shuffle stop gates are explicit.
- [x] B0/B1 run IDs, interfaces, costs, and order are specified.
- [x] B2/B3 are locked behind machine-readable gates.
- [ ] B0/B1 implementation exists and passes tests.
- [ ] B0/B1 empirical gates pass.
- [ ] B2/B3 evidence exists; the final +3/+3 goal is **not yet met**.
