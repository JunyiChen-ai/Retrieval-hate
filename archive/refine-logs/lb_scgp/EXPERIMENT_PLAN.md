# LB-SCGP Experiment Plan

**Problem:** 让 label-blind、train-only MLLM 通过一个可移除、可归因的 full-bank geometry 接口进入 RGCL，并最终显著提升 ordinary full-video train-memory kNN 的 accuracy 与 macro-F1。  
**Method thesis:** `label-blind structural reflection -> exact-vote-safe proximal full-bank target -> uniform encoder fit -> ordinary kNN`。  
**Date:** 2026-07-11  
**Authoritative scientific inputs:** `PROBLEM_ANCHOR.md`, `FINAL_PROPOSAL.md`, `REVIEW_SUMMARY.md`, `REFINEMENT_REPORT.md`, and this `EXPERIMENT_PLAN.md`. Mutable progress records such as `EXPERIMENT_TRACKER.md`, root `TARGET_LOOP.md`, `TARGET_STATE.json`, target findings, execution records, and handoff notes are audit-trail records, not formal rehashed freeze inputs.  
**Current authorization:** **仅 G0**。G1--G4 全部锁定；`mllm_teacher_call_count=0`。
**Current status:** v1 sanitizer and v1 freeze were executed, but `LBSCGP-G0-FREEZE-v1` failed formal code audit because it hash-bound mutable progress records and the pre-freeze sanitizer schema was underspecified. v1 artifacts and locks are immutable and must not be edited. The registered repair path is a no-clobber v2 freeze lineage only; no synthetic, realfold, replay, decision, G1, teacher, MLLM, or OCR job is authorized by this clarification.

## 0. Immutable Contract

1. 唯一 gold 是 **parent-video binary label**。`segment_gold_exists=false`、`segment_gold_used=false`；不存在也不得假定 segment/timestamp/span/localization/stance/target/mechanism/rationale gold。G0/G1 不读取 subclip artifact，不构造 segment cache，不传入 segment objective；继承 parent label 的 subclip row 不是 segment gold，且不得作为监督。
2. frames、完整可用 ASR/OCR/title 只是 whole-video input。未来 certificate atom 始终是 confidence-bearing weak/privileged pseudo-signal，不是 annotation 或 ground truth。
3. G0/G1 禁止任何 teacher/new-OCR call、teacher/certificate cache 读写。G0 只能读取 sealed outer-train full-video bank/labels；outer-held label/content、validation、test 均不得读取。
4. 最终 endpoint 固定为 MHC-EN（代码名 `MHC`）与 MHC-ZH（`MHC_zh`）的 ordinary top-20 arithmetic-rank、similarity-signed full-video kNN；无 teacher、certificate、compiler、target、rerank、fusion 或 native head。
5. 固定五折为 `artifacts/ssr/v1/folds/{MHC,MHC_zh}.json`，只复用 IDs/hashes 与 baseline checkpoints，不复用 SSR pairs/events/action universe。
6. 全部计算经 SLURM、`conda activate HateVideo`，不得写 `--time`，不得手工 release `JobHeldUser`，最多 2 GPU / 16 CPU / 128 GB。
7. 正式 namespace no-clobber；失败后不得放宽 tolerance、pivot/orientation budget、metric gate、teacher/schema、epoch或scale来救同一路线。

## 1. Claim Map

| Claim | Minimum convincing evidence | Stage |
|---|---|---|
| C1: exact-vote-safe product-space projection 是可实现、可核验的 full-bank intervention | independent implementation audit；sealed synthetic + one-real-fold 的 projector/KKT/Dykstra/rank-cell/exact-vote/Farkas/factor/rollback 全过；十折上界 `<160 GPU-h` | G0 |
| C2: 该 action family 可被 encoder 学到并改变真正 OOF kNN | zero-teacher LABEL-ONLY 在两库 strict OOF acc 与 mF1 均 `>=+0.050` vs REMOVE，五折每个 metric delta 同号为正 | G1, LOCKED |
| C3: MLLM structural certificate 有 label-only 之外的因果价值 | powered label-blind pilot 后 FULL 胜 label-only/proxy/direct/shuffle/noise controls | G2--G3, LOCKED |
| Final claim | 两库 seeds 0/1/2 相对 moving strongest comparator acc/mF1 均 `>=+0.030`，归因和统计门全部通过 | G4, LOCKED |

Historical anti-repeat constraints are binding: no verdict/score fusion, neighbour rerank, segment weighting, schema auxiliary head, summary concat, counterfactual twin, pseudo-group reweighting/gradient surgery, native-head substitution, teacher key selection, or segment-gold route.

## 2. G0 — Authorized Implementation Audit and Sealed Numerical Microbenchmark

### 2.1 Freeze and sealed fixtures

- Historical namespace: `artifacts/lb_scgp/v1/` with `LBSCGP-G0-FREEZE-v1`; it is preserved as a failed-audit predecessor and is never edited, deleted, overwritten, or reused for v2 claims.
- Current repair namespace: `artifacts/lb_scgp/v2/`.
- `LBSCGP-G0-FREEZE-v2` freezes `configs/lb_scgp/lb_scgp_v2.json`, implementation hashes, dependency versions, supervision counters, immutable scientific protocol documents, the dedicated pre-freeze sanitizer contract snapshot, and immutable data/config/code artifacts. It explicitly excludes mutable progress logs (`EXPERIMENT_TRACKER.md`, `TARGET_LOOP.md`, `TARGET_STATE.json`, target findings, handoff notes, execution records, and runtime logs) from formal `input_files` and dirty-state predecessor checks so required documentation updates cannot invalidate later verification.
- Synthetic cases are generated only from canonical seed `20260711`; their expected feasibility/status ledger is committed in the freeze before solver execution.
- The single real fixture is **`MHC_zh`, outer fold 4**, chosen before outcomes because it has the largest outer-train memory (`N=464`): checkpoint `artifacts/ssr/v1/oof/MHC_zh/fold4/checkpoint_epoch28.pt` SHA256 `c135924c87d1b12218f332ed8b955795cc34d6da972a576c011820572c3d0a39`; allowed bank members are `memory_ids`, `memory_z`, `memory_labels`, and `query_ids` sentinel only. Its 115 outer-held IDs are exclusion sentinels only; `query_z`, `query_labels`, held labels/content and all val/test content stay unopened by formal G0.
- Round3 data-isolation decision: no physically separated fold4 train-only whole-video feature source was found by metadata/path inspection, so byte-level non-opening is revised **before formal freeze** to a dedicated quarantine sanitizer. `LBSCGP-G0-SANITIZE-MHC_zh-F4-v1` may open only the mixed whole-video feature cache in the quarantine stage, with no model/optimizer/evaluator/teacher/OCR import or call, select rows solely by `memory_ids`, inherit parent-video labels only from `memory_labels`, and write one whitelist-only train-only whole-video feature PT artifact plus sanitized provenance. It must not open, produce, hash or publish a subclip/segment artifact. `LBSCGP-G0-SANITIZER-VERIFY-MHC_zh-F4-v1` independently verifies exact ID order, schema, inherited parent-video labels, zero query overlap, no segment artifact/objective and no-clobber before formal freeze. The quarantine manifest discloses source paths/hashes; formal G0 freezes only sanitized provenance/decision and the train-only whole-video feature output, not source, mixed-cache, fold-JSON, quarantine-manifest or subclip locators/hashes.
- G0 real-fold outputs are numerical/cost evidence only and cannot be reused as a G1 result or endpoint.

### 2.2 Independent implementation audit

Before any numerical job, a separate reviewer must inspect the frozen implementation and write `LBSCGP-G0-CODE-AUDIT-v1` with **0 CRITICAL / 0 HIGH**. Binding checks:

- full real `N x N` Frobenius ambient space; separate symmetry projection; PSD input explicitly symmetrized;
- exact row/class-mean/semantic preimage-ball projections and persistent Dykstra corrections;
- complete 19 internal top-20 inequalities plus 20th-vs-all-outsiders, self exclusion, canonical-ID ties and globally coupled boundary orientations;
- only `LOCAL_STATIONARY_CERTIFIED` trains; incomplete enumeration, `>8` independent orientations, `>32` pivots or unresolved tie maps deterministically to REMOVE;
- independent verifier does not import solver/projector/evaluator implementations under test;
- no forbidden dataset reader, teacher/cache path, segment object, val/test endpoint, or silent fallback; deterministic rollback restores model/optimizer/scheduler/scaler/RNG/sampler/epoch cursor.

### 2.3 Synthetic numerical suite

`LBSCGP-G0-SYNTH-v1` runs isolated projectors and composed product-space cases, including feasible interior/boundary cells, simultaneous ties, infeasible/over-budget cells, `semantic radius=0`, rank swaps, Farkas in-cone/out-of-cone witnesses and repeated/null eigenspaces.

Binding thresholds from `FINAL_PROPOSAL.md`:

- operator/adjoint dot error `<=1e-10`;
- for every nontrivial projector: feasibility, KKT stationarity, complementarity, scalar-root residual, idempotence, finite-difference optimum and independent dense-reference error `<=1e-7`; variational inequality `<=1e-8` over 1,000 frozen feasible probes;
- Dykstra stops only after a full cycle with max independent set violation `<=1e-6` and relative iterate change `<=1e-7`, max 500 cycles;
- independent float64 exact evaluator matches canonical IDs, ranks, cosine values, weighted signed vote, tie decision and prediction; synthetic overflow/NaN/Inf count is zero;
- feasible cases finish `LOCAL_STATIONARY_CERTIFIED`; deliberately capped cases return the preregistered fail-closed status and exact REMOVE replay, never a trainable target;
- abstract/factored registered-cone NNLS and independent Farkas witness agree: out-of-cone residual/separation `>=0.25`, duality gap `<=1e-5`; known in-cone controls are detected;
- factor rejects eigenvalue `<-1e-7`; factor/row/Gram reconstruction and Procrustes errors `<=1e-6`, with deterministic repeated-eigenspace/nullspace basis.

### 2.4 One-real-fold microbenchmark

`LBSCGP-G0-REAL-MHC_zh-F4-S0-v1` uses the sealed 464-row bank and frozen LABEL-ONLY constraints. It measures one complete target refresh, every projector, PSD eigensolve, Dykstra cycle, rank-cell pivots, exact-vote verifier, abstract/factored and one-block-realized Farkas audit, factorization, one uniform target-fit block, failure rollback and direct REMOVE replay hash. No outer-held prediction or metric is produced.

Real-fold GO additionally requires:

- all float64/numerical thresholds above pass; target status is `LOCAL_STATIONARY_CERTIFIED` (not bounded fallback);
- all rank/vote/slack/trust/PSD/box/semantic-absent constraints independently recompute within tolerance;
- realized-bank displacement cosine `>=0.80`, relative target residual `<=0.50`, no collapse/duplicate, abstract and realized registered-cone separation pass;
- actual fit and separate replay use whole-video memory only, `lambda_seg=0`, `segment_cache=None`; any segment cache/objective is fail-closed;
- rollback replay is byte/hash-identical to a direct REMOVE clone, and a separate GPU replay artifact independently reruns batch order, target-fit steps, realized bank, and live AdamW/scheduler/scaler/RNG/cursor rollback before CPU decision;
- peak GPU allocation `<=24 GiB`, host peak RSS `<=64 GiB`.

The ten-fold G1 cost upper bound is computed before seeing any G1 endpoint:

`H10_upper = 1.25 * 10 * [2*H_REMOVE_fullfold + 5*p95(H_bank + H_target + H_factor + H_fitblock + H_verify + H_farkas) + H_final_bank]`.

`p95` is computed on the preregistered sealed timing-sample set; when G0 has one observation, `p95` is the conservative max/upper statistic of that single sample. `H_final_bank` is outside the refresh term. `H_REMOVE_fullfold` is independently reconstructed from the frozen 29-epoch real-fold ledger and checked by the real dry block; factor `1.25` is mandatory contingency. **GO requires finite `H10_upper <160 GPU-hours`** and verified one-GPU SLURM allocation/runtime; equality or missing timing is STOP.

### 2.5 G0 decision

`LBSCGP-G0-DECISION-v1` is a separate CPU verifier which re-hashes and independently recomputes every gate; producer `pass` flags are ignored.

G0 is GO iff all are true: code audit 0 CRITICAL/HIGH; freeze valid; synthetic and real numerical/KKT/float64/Dykstra/local-stationary/exact-vote/Farkas/factor/rollback gates pass; resource and `<160 GPU-h` gates pass; `only_gold_supervision=parent_video_binary_label`; `segment_gold_exists=false`; `segment_gold_used=false`; all teacher/OCR/cache, outer-held-label/content, val-content and test-content counters are zero. Any missing cell is `INVALID/STOP`; G1 and every teacher stage remain locked.

## 3. Exact G0 Execution Interface

Round3 implementation files prepared for independent review:

```text
configs/lb_scgp/lb_scgp_v2.json
configs/lb_scgp/lb_scgp_v1.json
scripts/analysis/lb_scgp_common.py
scripts/analysis/lb_scgp_sanitize_inputs.py
scripts/analysis/lb_scgp_verify_sanitizer.py
scripts/analysis/lb_scgp_g0.py
scripts/analysis/lb_scgp_real_replay.py
scripts/analysis/lb_scgp_independent_verify.py
scripts/slurm/lb_scgp_sanitize_inputs.sbatch
scripts/slurm/lb_scgp_g0_cpu.sbatch
scripts/slurm/lb_scgp_g0_gpu.sbatch
```

Frozen Python CLI:

```text
lb_scgp_sanitize_inputs.py --config configs/lb_scgp/lb_scgp_v1.json --source-config configs/lb_scgp/lb_scgp_sanitizer_sources.json --task build --run-id LBSCGP-G0-SANITIZE-MHC_zh-F4-v1
lb_scgp_verify_sanitizer.py --config configs/lb_scgp/lb_scgp_v1.json --task verify --run-id LBSCGP-G0-SANITIZER-VERIFY-MHC_zh-F4-v1
lb_scgp_g0.py --config configs/lb_scgp/lb_scgp_v2.json --task freeze --run-id LBSCGP-G0-FREEZE-v2
lb_scgp_g0.py --config configs/lb_scgp/lb_scgp_v2.json --task synthetic --run-id LBSCGP-G0-SYNTH-v2
lb_scgp_g0.py --config configs/lb_scgp/lb_scgp_v2.json --task realfold --dataset MHC_zh --outer-fold 4 --run-id LBSCGP-G0-REAL-MHC_zh-F4-S0-v2
lb_scgp_real_replay.py --config configs/lb_scgp/lb_scgp_v2.json --task replay --run-id LBSCGP-G0-REAL-REPLAY-MHC_zh-F4-S0-v2
lb_scgp_independent_verify.py --config configs/lb_scgp/lb_scgp_v2.json --task decide --run-id LBSCGP-G0-DECISION-v2
```

Frozen SLURM interface:

```text
TASK=build sbatch scripts/slurm/lb_scgp_sanitize_inputs.sbatch
TASK=verify sbatch scripts/slurm/lb_scgp_sanitize_inputs.sbatch
CONFIG=configs/lb_scgp/lb_scgp_v2.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v2 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
CONFIG=configs/lb_scgp/lb_scgp_v2.json TASK=synthetic RUN_ID=LBSCGP-G0-SYNTH-v2 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
CONFIG=configs/lb_scgp/lb_scgp_v2.json TASK=realfold DATASET=MHC_zh OUTER_FOLD=4 RUN_ID=LBSCGP-G0-REAL-MHC_zh-F4-S0-v2 sbatch scripts/slurm/lb_scgp_g0_gpu.sbatch
CONFIG=configs/lb_scgp/lb_scgp_v2.json TASK=replay DATASET=MHC_zh OUTER_FOLD=4 RUN_ID=LBSCGP-G0-REAL-REPLAY-MHC_zh-F4-S0-v2 sbatch scripts/slurm/lb_scgp_g0_gpu.sbatch
CONFIG=configs/lb_scgp/lb_scgp_v2.json TASK=decide RUN_ID=LBSCGP-G0-DECISION-v2 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
```

CPU jobs: 8 CPU / 64 GB. GPU job: 1 GPU / 8 CPU / 64 GB. Scripts reject absent `SLURM_JOB_ID`, wrong conda env, wrong predecessor, nonfrozen config, existing namespace or any teacher/certificate argument.

## 4. Artifact and JSON Provenance Contract

```text
artifacts/lb_scgp/v1/
  CONFIG_FREEZE.json
  g0/code_audit/{review.md,audit.json}
  g0/synthetic/{cases.jsonl,projectors.jsonl,dykstra.jsonl,rank_cells.jsonl,exact_vote.jsonl,farkas.jsonl,factor.jsonl,manifest.json}
  g0/real/MHC_zh/fold4/{timings.json,numerics.json,projectors.jsonl,rank_cells.jsonl,exact_vote.jsonl,farkas.json,factor.json,fit_rollback.json,fit_replay.json,resource.json,manifest.json}
  G0_DECISION.json
```

Pre-freeze sanitizer artifacts live outside the formal namespace:

```text
artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json
artifacts/lb_scgp/inputs/MHC_zh/fold4/{outer_train_features.pt,sanitized_provenance.json,sanitizer_decision.json}
```

Dedicated pre-freeze sanitizer schema for existing v1 sanitizer physical records, prospectively clarified before v2 freeze:

- Applies only to `LBSCGP-G0-SANITIZE-MHC_zh-F4-v1`, `LBSCGP-G0-SANITIZER-VERIFY-MHC_zh-F4-v1`, and their disclosure record before any formal G0 freeze. It is narrower than the generic manifest schema because the sanitizer precedes the formal namespace and does not run model, optimizer, evaluator, teacher, MLLM, OCR, validation, or test code.
- `sanitized_provenance.json` mandatory fields: `schema_version,run_id,stage,slurm_job_id,dataset,outer_fold,artifact_namespace,feature_cache_path,feature_cache_sha256,memory_id_count,query_id_sentinel_count,memory_ids_sha256,memory_labels_sha256,query_ids_sha256,row_selection_rule,parent_label_rule,input_cache_labels_ignored,zero_overlap_with_query_ids,payload_sha256,sanitizer_code_sha256,segment_cache_path,segment_cache_sha256,segment_artifact_created,segment_objective_allowed,teacher_mllm_ocr_calls,network_external_calls,formal_query_z_read_count,formal_query_labels_read_count,formal_model_optimizer_evaluator_outer_held_read_count,pre_freeze_disclosure_record_external,no_clobber_locks_present`.
- `sanitizer_decision.json` mandatory fields: `schema_version,run_id,stage,status,slurm_job_id,dataset,outer_fold,feature_cache_path,feature_cache_sha256,sanitized_provenance_path,sanitized_provenance_sha256,safe_contract_sha256,memory_id_count,query_id_sentinel_count,memory_ids_sha256,memory_labels_sha256,query_ids_sha256,payload_sha256,segment_cache_path,segment_cache_sha256,segment_artifact_created,segment_objective_allowed,teacher_mllm_ocr_calls,network_external_calls,formal_query_z_read_count,formal_query_labels_read_count,formal_model_optimizer_evaluator_outer_held_read_count,gates`.
- The non-formal disclosure record must have `formal_g0_input=false`, `payload_sha256`, `access_ledger_sha256`, `feature_cache_sha256`, `output_hashes`, `memory_id_count=464`, `query_id_sentinel_count=115`, `zero_overlap_with_query_ids=true`, `input_cache_labels_ignored=true`, `parent_label_rule=output labels inherited only from memory_labels`, `segment_cache_path=null`, `segment_cache_sha256=null`, `segment_artifact_created=false`, `segment_objective_allowed=false`, `teacher_mllm_ocr_calls=0`, `network_external_calls=0`, `formal_query_z_read_count=0`, `formal_query_labels_read_count=0`, `formal_model_optimizer_evaluator_outer_held_read_count=0`, and persistent no-clobber locks for every published sanitizer output.
- v2 formal freeze binds the train-only feature artifact, safe sanitizer provenance, safe sanitizer decision, and a safe immutable sanitizer-contract snapshot. The disclosure record hash is recorded only inside that safe snapshot; its protected source locators/hashes remain outside formal `input_files`.
- From `G0_FREEZE` onward, every formal manifest/decision uses the full generic schema below.

Every formal manifest/decision from `G0_FREEZE` onward contains at least: `schema_version,run_id,stage,status,slurm_job_id,git_head,dirty_diff_sha256,conda_env,python/torch/numpy/scipy/cuda versions,gpu_name,config_canonical_sha256,implementation_sha256,independent_verifier_sha256,input_files[{path,sha256}],fold_ids_sha256,checkpoint_sha256,output_files[{path,sha256}],payload_sha256,only_gold_supervision,segment_gold_exists,segment_gold_used,mllm_call_count,ocr_call_count,teacher_cache_read_count,teacher_cache_write_count,outer_held_label_read_count,outer_held_content_read_count,val_content_read_count,test_content_read_count,val_test_teacher_artifact_count`.

JSON uses sorted-key UTF-8 compact canonical serialization and forbids NaN/Inf. `payload_sha256` excludes itself. Formal outputs use persistent `O_CREAT|O_EXCL` lock and temp+fsync+atomic no-clobber publish.

## 5. Locked G1--G4 Skeleton

- **G1 / SCGP-0, LOCKED:** after verified G0 GO and new authorization only. Runs `LBSCGP-G1-OOF-{MHC|MHC_zh}-F{0..4}-S0-v1` plus `LBSCGP-G1-DECISION-v1`; each fold restores identical initialization for REMOVE and LABEL-ONLY. Zero teacher/certificate calls. Five strict outer folds concatenate actual ordinary-kNN ledgers. Both datasets require LABEL-ONLY minus REMOVE `>=+0.050` accuracy and macro-F1, and every fold delta is positive for both metrics; all target-fit/collapse/Farkas/numerical gates pass. Otherwise terminal STOP with teacher still zero.
- **G2 / certificate pilot, LOCKED:** runs begin `LBSCGP-G2-SAMPLE-*`, `LBSCGP-G2-TEACHER-*`, `LBSCGP-G2-DECISION-v1`. At most 128 unique whole-train videos/dataset, sampled by parent label × strict-OOF prediction × margin quartile with stored inclusion probability and hash-fixed A/B cross-fit. At most four label-blind calls/video (`<=512/dataset`); cache Merkle-closes before train labels enter compiler. No segment fields/gold. All design-based support/selection/reflection and partial-OOF control gates from `FINAL_PROPOSAL.md` are binding.
- **G3 / seed-0 controls, LOCKED:** `LBSCGP-G3-{MHC|MHC_zh}-S0-{ARM}-v1`, `LBSCGP-G3-DECISION-v1`; FULL must beat REMOVE, moving LABEL-ONLY, CERT-SHUFFLE/NOISE, scalar propensity, P4, TextTeacher, pair/triplet/SupCon, DIRECT-AEXC and STATE-MOMENT by `+0.010` in both metrics on both datasets; corruption `{0,.25,.50,.75,1}` monotonically removes gain. Freeze one global strongest direct control.
- **G4 / final, LOCKED:** `LBSCGP-G4-{MHC|MHC_zh}-S{0,1,2}-{FULL|REMOVE|SHUFFLE|DIRECT}-v1` plus `LBSCGP-G4-FINAL-STATS-v1`. FULL must exceed `max(historical strongest point, paired same-seed strongest comparator mean)+0.030` in acc and mF1 on both datasets; all paired seed deltas positive; hierarchical paired-bootstrap lower bounds `>0`; four tests survive Holm FWER `.05`; FULL also significantly beats REMOVE, SHUFFLE and frozen strongest direct control. Only this stage can close the project goal.

## 6. STOP / GO Summary

```text
quarantine sanitizer build -> independent sanitizer verify
  -> G0 freeze -> independent audit
  -> sealed synthetic
  -> sealed MHC_zh/F4 real-fold microbenchmark
  -> sealed GPU fit replay
  -> independent G0 decision
  -> only verified GO + fresh authorization can unlock G1

G1 GO -> G2 pilot -> G3 seed0 controls -> G4 two datasets x three seeds
```

No G0 result is a performance result or an MLLM success. No G1 result is an MLLM result. The route remains active only while the exact frozen gates pass; the global goal remains unachieved until G4 proves the final +3/+3, removability, novelty, statistics and supervision contract jointly.
