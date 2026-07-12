# CTE-RGCL Experiment Plan

**Problem:** 让 label-blind、train-only MLLM 以可移除且 assignment-specific 的方式改变 RGCL 共享 full-video retrieval geometry，并最终在两个数据集的普通 full-video train-memory kNN 上取得 substantial accuracy/macro-F1 提升。  
**Method thesis:** whole-modality withholding relation 不是片段标注或因果真值；只有当它对冻结、train-supported prototype tangent 的 class-conditional ordinal transfer 通过实证门槛时，才可作为 privileged weak supervision。  
**Date:** 2026-07-10  
**Authoritative method:** `refine-logs/cte/FINAL_PROPOSAL.md`  
**Current authorization:** **C0 + C1 only**. C2 teacher pilot、C3、C4 全部锁定。

## 0. Immutable Contract

1. 唯一 gold 是 parent-video binary label。`segment_gold_exists=false`；不存在也不得假定 segment/timestamp/span/localization/stance/target/mechanism/rationale gold。
2. 固定数据集为 MHC-EN（代码名 `MHC`）和 MHC-ZH（代码名 `MHC_zh`）；固定 train folds 复用 `artifacts/ssr/v1/folds/{MHC,MHC_zh}.json`，但不得复用 SSR relation/action universe。
3. C0/C1 的 `mllm_call_count=0`、`ocr_call_count=0`、`teacher_cache_read_count=0`、`teacher_cache_write_count=0`。出现任何非零值即 `INVALID/STOP`。
4. C0/C1 仅使用 train split。validation/test 不作为 source、target、checkpoint selector 或 endpoint；除已冻结 split-ID/hash disjointness 证明外，执行脚本不得打开 val/test 内容。
5. CTE 不产生 segment view、segment target、segment weight 或 segment endpoint。若 REMOVE 复现现有 strongest RGCL 的 K4 base loss，K4 标签只能是 parent-video label 的机械继承，必须记录为 `inherited_parent_video_label_not_segment_gold`；CTE auxiliary path 始终是 whole-video full modality path。
6. validation/test 以及最终 inference 永远只接收 full video，使用 rebuilt full-video train bank 和 repository ordinary top-20 arithmetic cosine kNN；无 relation、confidence、neutralized view、prototype view、teacher key、rerank、score fusion 或 MLLM。
7. `CTE-0` 是该精确 loss/path/bank 的 **zero-teacher bounded empirical cost/capacity screen**；它不是理论上界、不是 MLLM 结果，也不能否定其他表示学习路线。
8. 不允许失败后更换 anchor、半径、dataset-specific prompt/hyperparameter、加 adapter/head/EMA/router，或放宽门槛。所有正式产物 no-clobber；禁止在正式 namespace 使用 `--force`。

## 1. Claim Map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1（dominant）固定 supported tangent 与 exact epoch-refreshed full-bank loss 具有足够 action capacity，可改变旧 top-64 外的共享 query/key geometry | SSR/EDCM 失败来自冻结邻域动作覆盖不足；CTE 必须先证明连续表示路径能实际修复足够 train-OOF errors | C0 numerics/support/runtime 全过；C1 在 `MHC`、`MHC_zh` 的 strict train-OOF full-video kNN 同时达到 `>=+0.050 acc` 和 `>=+0.050 mF1` vs REMOVE，且 error correction/churn/control gates 全过 | B0, B1 |
| C2（supporting，后续条件性）label-blind MLLM relation 提供超出 video label、generic multiview 与随机 assignment 的 class-specific ordinal information | 否则 MLLM 只是装饰或 label redundancy | 仅在 C1 GO 后，≤128 videos/dataset 的 two-radius、two-class ordinal-transfer 与 held-out pilot update 同时胜过 label-only/energy/multiview/random/SHUFFLE/NOISE | B2（LOCKED） |
| Anti-claim | Gain 仅来自额外优化、generic multiview、片段标注、旧邻域 edit 或 endpoint 改动 | C1 label-only > matched multiview/random；后续 FULL > REMOVE/SHUFFLE/NOISE；所有记录明确 no-segment-gold，val/test no teacher | B1–B4 |

## 2. Paper Storyline

- Main paper must prove: final ordinary kNN +3/+3；MLLM relation assignment 的 removability；shared full-bank geometry 而非 frozen-neighbour edit；无 inference dependency。
- Appendix can support: C0 numerical parity/runtime、support/drift distributions、nested-probe details、teacher parse/coverage（仅后续解锁后）。
- Intentionally cut: localization、segment classification/weighting、teacher key、summary/score concat、reranking、native-head claim、scaling、SSR/EDCM retune。

## 3. Frozen Method Definitions

### 3.1 Exact representation spaces

在同一 baseline checkpoint 的 `eval()` mode：

- `p_i^V = normalize(img_proj(v_i))`，`p_i^L = normalize(text_proj(l_i))`；归一化 epsilon `1e-6`，pre-normalization norm `<1e-4` 为 hard numerical failure。
- `g_i = p_i^V * p_i^L`（`align` 的 pre-MLP fused vector）；support 距离计算前也 L2-normalize。
- 每个 modality 单独选 spherical medoid video ID：最小化该 outer-train/full-train bank 内 summed cosine distance；精确并列按 canonical UTF-8 video ID 升序。记为 `anchor_id^V`、`anchor_id^L`。
- `p_tilde_i^m(a)=normalize((1-a)p_i^m+a p_anchor^m)`，候选 `a={0.05,0.10,0.20,0.30}`。
- projected-pair space：`concat(p_tilde_i^V,p_i^L)` 或 `concat(p_i^V,p_tilde_i^L)`；fused space：两者 elementwise product。两空间均以 cosine distance `1-cos` 做 exact 5-NN。
- 每空间的 support threshold 是 unperturbed train points 的 leave-one-out 5-NN radius 的 95th percentile（NumPy linear quantile）；perturbed query 对 unperturbed train bank 搜索，不排除 same-ID unperturbed point。
- `S_i^m(a)=1` 当 projected-pair 与 fused 两个 radius 同时不超过各自 threshold。相邻 pair `(a1,a2)` 必须同时满足：所有 `i,m,a` 的 mean support `>=0.95`，且 `mean_i 1[all m in {V,L}, a in {a1,a2}: S_i^m(a)=1] >=0.80`。从 `(0.05,0.10),(0.10,0.20),(0.20,0.30)` 中选 `a2` 最大者；无候选即 STOP。
- anchor IDs/radii 一经选择即 hash 冻结；后续只允许重编码相同 anchor ID 和重算 support mask，不得重选。

### 3.2 Full-bank tangent and numerical constants

对 detached、normalized、epoch-start full train bank，self ID 排除：

`M_i(z)=tau*LSE_same(s/tau)-tau*LSE_other(s/tau)`，`s=z^T k`；每个 query 必须至少有一个 non-self same-label key 与一个 opposite-label key。

`T_i^m(a)=tanh((M_i(z_i^{m,a})-M_i(z_i))/(a*max(MAD,sMin)+1e-6))`。

- Grid only: `tau in {0.05,0.10}`, `lambda in {0.05,0.10}`, `sMin in {0.05,0.10}`；共 8 tuples。
- Teacher intervals（后续阶段才使用）固定：`Ip=[-0.05,0.05]`、`Iw=[-0.50,-0.20]`、`Ir=[-1.00,-0.50]`；`dist=max(l-T,0,T-u)`，cost=`dist^2/4`。
- bank 与 tangent query 都用 `eval()` stochastic semantics；base RGCL loss 保持原训练语义。每个 train video 每 epoch 恰好成为一次 CTE query；batch-64 base step 内以 32-row CTE microbatch 累积，optimizer step 数不变。
- epoch boundary 与 checkpoint load 重建 exact detached full bank。若任一 arm 首 epoch的 median same-ID epoch-start/end cosine `<0.95` 或 p95 angular drift `>0.25 rad`，整个 dataset/fold namespace 从共同初始 checkpoint 重启为 half-epoch refresh；仍失败即 STOP。不得 arm-specific fallback。
- 每次 refresh 重算固定路径 support。aggregate joint support `<0.90`、median frozen-direction cosine `<0.90`、p10 `<0.70`、nonfinite、norm failure均 STOP。

## 4. Experiment Blocks

### B0 — C0 vectorized full-bank microbenchmark, numerics and support audit

- **Claim tested:** kernel 可执行、数值正确，固定 whole-video prototype path 在 train geometry 中受支持。
- **Inputs:** frozen SSR fold IDs/hashes、full-video CLIP caches、现有 strongest RGCL fold checkpoints仅作 C0 audit fixture；不读取 SSR pairs/events/relations。
- **Runs:** `CTE-C0-MICRO-MHC-S0-v1`、`CTE-C0-MICRO-MHC_zh-S0-v1`、联合 `CTE-C0-DECISION-v1`。
- **Numerical tests:** synthetic double-precision scalar reference vs vectorized FP32（masked self、两类、noncontiguous IDs）；actual-bank scalar/vectorized margin、T、cost parity；autograd finite/directional finite-difference check；stable LSE under logits shifted by ±100；all four radii、both modalities、all five fold memories。
- **Frozen gates:** max absolute margin error `<=1e-5`，T/cost error `<=2e-5`，relative gradient error `<=1e-3`，100% finite，minimum pre-normalization norm `>=1e-4`；每个 dataset×fold 至少一个 adjacent pair 通过 §3.1；batch32 no OOM 且 peak allocated memory `<=24 GiB`。记录 20 warmup + 200 timed iterations的 median/p95 ms、peak memory、estimated C1 GPU-hours；测量值替代目前 20–40 GPU-hour 暂估。
- **Decision:** 任一 dataset/fold/numerical/resource gate失败即 `C0=STOP`，C1不启动。C0 GO 只证明 implementation/support feasibility。
- **Target:** appendix numerical/support table；priority MUST。

### B1 — C1 / CTE-0 strict nested train-OOF zero-teacher screen

- **Claim tested:** exact fixed-tangent/epoch-bank path 能在不使用 teacher 的最佳受控 target 下产生足够 full-video kNN action capacity，并非 generic extra optimization。
- **Splits:** authoritative five outer folds。每个 outer fold `f` 的 held-out fold只做 outer OOF endpoint；其 label 不进入训练、probe、hyperparameter selection。
- **Complete nested probe:** 对每个 outer-train `I_f`，先用 stratified 3-fold（canonical sorted IDs，`random_state=20260710+f`）得到 inner folds。对每个 inner validation `J_r`，只在 `P_r=I_f\J_r` 内再做 stratified 3-way `(A,B,C)` rotations：对 `C in {0.01,0.1,1}` 的 `StandardScaler(A)+L2 LogisticRegression(lbfgs,max_iter=5000,tol=1e-9)` fit A，以 B binary log-loss最小选择（tie: smaller C），在 A∪B refit，预测 C；三轮覆盖 `P_r`。因此 `J_r` 的 label 不可能流入其候选训练 target。outer refit 时在 `I_f` 内同样三轮生成全覆盖 strict OOF targets，outer heldout fold完全未参与。
- **Probe target:** 同一个 full-fused probe `h` 应用于两条 modality tangent；`q=2y-1`，在冻结 `a1`：`b_i^m=min(0,tanh(q_i*(h(g_i^{m,a1})-h(g_i))/max(MAD,0.05)))`，interval=`[max(-1,b-0.05),min(0,b+0.05)]`。targets在任何 pilot ID selection 前缓存并 hash。
- **Hyperparameter selection:** 每个 outer fold的 8 tuples 分别在三个 `P_r -> J_r` 上训练/评估。REMOVE 与所有 arms 同 initialization、base loss、optimizer steps、epoch/refresh/checkpoint rule。对每 tuple，跨三 inner folds聚合每个 dataset的 acc/mF1；一个 tuple 同时用于两数据集，最大化四个 dataset×metric gain cells 中的最小值；tie依次 smaller `lambda`、larger `sMin`、lexicographic `(tau,lambda,sMin)`。outer outcomes、dev/test、teacher均不得选择 tuple。
- **Outer arms:** `REMOVE`；selected `LABEL_ONLY`；`MULTIVIEW`（所有 support-valid views uniform `Ip`、weight1、无 target/mask/assignment，只有一个使用 LABEL_ONLY clean-training-fold first-step aggregate gradient norm 的全局 strength scalar）；`RANDOM`（LABEL_ONLY interval multiset 的固定 no-fixed-point whole-record permutation，无 per-video assignment，再以一个全局 scalar匹配 first-step gradient norm）。所有 arms same steps/bank/checkpoint。
- **Endpoint:** 每 outer heldout query 只用 full video；memory 是该 outer-train arm 的 rebuilt full bank；repository top20 arithmetic cosine vote。五 fold 拼接为每 dataset 一份 strict OOF prediction ledger。
- **Binding C1 GO，须全部满足:**  
  1. initial/final support、direction drift、bank drift、numerics均过；  
  2. `LABEL_ONLY - REMOVE >= +0.050` accuracy 且 `>=+0.050` macro-F1，在 `MHC` 与 `MHC_zh` 各自成立；  
  3. corrected baseline errors：MHC `>=28`、MHC-ZH `>=29`；每个 class 的 `corrected_y-broken_y >0`；  
  4. held-out top20 set Jaccard churn 的 `LABEL_ONLY-RANDOM >=0.10`，video-paired 10,000 bootstrap 95% lower bound `>0`，两数据集均成立；  
  5. LABEL_ONLY 在每个 dataset 的 accuracy 和 mF1 都严格高于 MULTIVIEW 与 RANDOM；  
  6. 五个 outer-selected tuples 的 deterministic mode（tie按上规则）成为 C2 onward 唯一 frozen tuple；无 mode/缺 fold/manifest mismatch即 STOP。
- **Interpretation:** PASS 仅说明 label-only target 可驱动该精确 action family；不能声称 MLLM 成功。FAIL 是 cost STOP，不是理论不可能性。
- **Conditional full-train freeze after GO:** 以 modal tuple、seed0、dataset frozen epoch recipe各训练一次 LABEL_ONLY full-train checkpoint，选择并冻结 post-C1 `anchor_id^V/L` 与 adjacent radii，写入 teacher-before-call manifest。该步骤完成且独立 verify 前，C2仍锁定。
- **Target:** main/appendix action-capacity and control table；priority MUST。

### B2 — C2 ≤128/dataset teacher pilot（LOCKED skeleton）

只有 `C0=GO`、`C1=GO`、full-train anchor manifest verified 后才可另行实施。冻结 ≤128 IDs/dataset（label×strict-OOF-error×margin-tertile allocation + ID hash order），teacher 不见 label，只输出 `preserve/weaken/reverse/unclear + confidence`；上限 2048 calls。必须分别对 y=0/y=1、两个冻结 radii 做 reliability-weighted ordinal slope/ordered means/ESS gate，并以 held-out 20-step pilot 胜过 cached label-only、energy、multiview、random、feasible whole-record SHUFFLE；parse/coverage/agreement/kappa/noise gates沿 FINAL_PROPOSAL。当前无 run ID、无调用授权。

### B3 — C3 seed-0 controls（LOCKED skeleton）

C2双数据集 GO 后才设计正式 runs。train-only teacher cache一次冻结；seed0 dev full-video kNN 必须在两数据集 acc/mF1 各 `>=+0.010` 胜过 REMOVE、LABEL_ONLY、MULTIVIEW、RANDOM、ENERGY、SHUFFLE、NOISE，并满足 clean>eta1>eta2。validation 无 teacher/view；test保持锁定。

### B4 — C4 final 2 datasets × 3 seeds（LOCKED skeleton）

C3 GO 后才解锁。seeds 0/1/2；FULL 相对 `max(historical strongest point, paired same-seed strongest non-MLLM mean)` 在两数据集 acc/mF1各 `>=+0.030`，12个 paired seed deltas全正；shared-video hierarchical paired bootstrap 10,000、95% LB>0、四个 p-values Holm FWER .05；FULL-minus-REMOVE/SHUFFLE 同样 uncertainty-qualified。test仍是 ordinary full-video train-bank kNN，无 teacher artifact。

## 5. Exact Execution Interface for Authorized C0/C1

### 5.1 Planned files（implementation handoff；本计划不创建它们）

- `configs/cte/cte_v1.json`
- `scripts/analysis/cte_common.py`
- `scripts/analysis/cte_c0.py`
- `scripts/analysis/cte_c1.py`
- `scripts/slurm/cte_c0_gpu.sbatch`
- `scripts/slurm/cte_c0_cpu.sbatch`
- `scripts/slurm/cte_c1_gpu.sbatch`
- `scripts/slurm/cte_c1_cpu.sbatch`

Python CLI 固定为：

```text
cte_c0.py --config configs/cte/cte_v1.json --phase micro --dataset {MHC|MHC_zh} --run-id RUN_ID
cte_c0.py --config configs/cte/cte_v1.json --phase decide --run-id CTE-C0-DECISION-v1
cte_c1.py --config configs/cte/cte_v1.json --phase inner --dataset DATASET --outer-fold F --run-id RUN_ID
cte_c1.py --config configs/cte/cte_v1.json --phase select --run-id CTE-C1-SELECT-v1
cte_c1.py --config configs/cte/cte_v1.json --phase outer --dataset DATASET --outer-fold F --run-id RUN_ID
cte_c1.py --config configs/cte/cte_v1.json --phase decide --run-id CTE-C1-DECISION-v1
cte_c1.py --config configs/cte/cte_v1.json --phase full-freeze --dataset DATASET --run-id RUN_ID
cte_c1.py --config configs/cte/cte_v1.json --phase verify-freeze --run-id CTE-C1-FREEZE-VERIFY-v1
```

SLURM positional interface固定为：

```text
sbatch scripts/slurm/cte_c0_gpu.sbatch micro DATASET RUN_ID
sbatch scripts/slurm/cte_c0_cpu.sbatch decide CTE-C0-DECISION-v1
sbatch scripts/slurm/cte_c1_gpu.sbatch inner DATASET OUTER_FOLD RUN_ID
sbatch scripts/slurm/cte_c1_cpu.sbatch select CTE-C1-SELECT-v1
sbatch scripts/slurm/cte_c1_gpu.sbatch outer DATASET OUTER_FOLD RUN_ID
sbatch scripts/slurm/cte_c1_cpu.sbatch decide CTE-C1-DECISION-v1
sbatch scripts/slurm/cte_c1_gpu.sbatch full-freeze DATASET CTE-C1-FULLFREEZE-DATASET-S0-v1
sbatch scripts/slurm/cte_c1_cpu.sbatch verify-freeze CTE-C1-FREEZE-VERIFY-v1
```

不得加 `--time`，不得手工 release `JobHeldUser`，统一 `conda activate HateVideo`。GPU scripts: `--partition=slurmpartition --gres=gpu:1 --cpus-per-task=4 --mem=32G`；CPU scripts: `--partition=slurmpartition --cpus-per-task=4 --mem=16G`。同一用户最多并发两个 GPU jobs；C0/C1 dependencies 只在 predecessor verified GO 后提交。

### 5.2 Exact official run IDs

- C0: `CTE-C0-MICRO-{MHC|MHC_zh}-S0-v1`；`CTE-C0-DECISION-v1`。
- C1 inner: `CTE-C1-INNER-{MHC|MHC_zh}-F{0..4}-S0-v1`（10 runs）。
- C1 selection: `CTE-C1-SELECT-v1`，内部必须输出 `fold0.json`…`fold4.json`。
- C1 outer: `CTE-C1-OUTER-{MHC|MHC_zh}-F{0..4}-S0-v1`（10 runs）。
- C1 decision: `CTE-C1-DECISION-v1`。
- Conditional after GO: `CTE-C1-FULLFREEZE-{MHC|MHC_zh}-S0-v1`；`CTE-C1-FREEZE-VERIFY-v1`。

## 6. Artifact and JSON Provenance Contract

```text
artifacts/cte/v1/
  CONFIG_FREEZE.json
  c0/{MHC,MHC_zh}/{microbenchmark.json,numerics.json,support.json,manifest.json}
  C0_DECISION.json
  c1/inner/<dataset>/fold<F>/{inner_splits.json,probe_targets.jsonl,grid_metrics.jsonl,manifest.json}
  c1/selection/{fold0.json,...,fold4.json,manifest.json}
  c1/outer/<dataset>/fold<F>/{anchors.json,probe_targets.jsonl,diagnostics.jsonl,predictions_<arm>.json,neighbors_<arm>.jsonl,manifest.json}
  C1_DECISION.json
  c1/fullfreeze/<dataset>/{checkpoint.pt,anchors.json,manifest.json}
  C1_FREEZE_VERIFY.json
```

每个 manifest/decision 至少含：`schema_version,run_id,stage,status,slurm_job_id,git_head,conda_env,cuda_version,gpu_name,config_canonical_sha256,implementation_sha256,input_files[{path,sha256}],fold_ids_sha256,checkpoint_sha256,output_files[{path,sha256}],payload_sha256,only_gold_supervision,segment_gold_exists,segment_gold_used,mllm_call_count,ocr_call_count,teacher_cache_{read,write}_count,val_endpoint_count,test_endpoint_count,val_test_teacher_artifact_count`。训练 manifest 另含 exact params、initialization hash、batch/optimizer step counts、epoch/half-epoch bank refresh ledger、anchor IDs/radii、support/drift/norm/gradient diagnostics。

所有 JSON canonical serialization 为 sorted keys、UTF-8、compact separators、禁止 NaN/Inf；`payload_sha256` 对移除自身字段后的 canonical payload 计算。每个正式 namespace 先以 persistent `O_CREAT|O_EXCL` lock 占用，最终文件 temp+fsync+no-clobber publish。decision/verify phase 必须从 prediction ledgers 和 authoritative inputs独立重算 metrics/gates，不能信任 producer 的 pass flag；任一 hash、row count、ID partition、run ID、zero-call 或 supervision field 不符即 `INVALID/STOP`。

## 7. Run Order, Gates and Cost

| Milestone | Goal | Runs | Gate | Provisional cost | Main risk |
|---|---|---|---|---|---|
| M0 | implementation/hash/static audit | implementation bridge before execution | reviewer 0 HIGH/CRITICAL；CONFIG_FREEZE verified | CPU only | hidden leakage/no-clobber gap |
| M1 | C0 kernel/numerics/support/runtime | 2 GPU + 1 CPU decision | all §B0 gates | <1 GPU-h expected；measured value binding | unsupported tangent or unstable normalization |
| M2 | C1 nested inner selection | 10 GPU + 1 CPU selection | all inner manifests valid；one tuple/fold | C0 estimate replaces 20–40 GPU-h provisional total | nested target leakage or cost |
| M3 | C1 outer OOF screen | 10 GPU + 1 CPU decision | all six §B1 gates on both datasets | included above | +5/+5 not reachable or generic controls match |
| M4 | conditional full-train freeze | 2 GPU + 1 CPU verify | C1 GO + anchors/radii verified before teacher | C0-scaled | path drift/full-train no supported pair |
| M5 | C2–C4 | LOCKED | separate authorization only | not estimated | teacher cells/assignment value/final gain fail |

Human annotation cost为零；MLLM outputs（后续若解锁）始终是 weak pseudo-relations，不是 annotations。

## 8. Stop/Go and Anti-Interpretation Rules

- C0 STOP：不提交 C1，不调半径/anchor/kernel threshold，不调用 teacher。
- C1 STOP：不提交 C2，不做 prompt/teacher/scale rescue；记录为该 exact label-only tangent action family 的 empirical cost failure。
- C1 GO：只解锁“准备 C2 plan/review”，不等于 teacher value、novelty evidence 或 project target success。
- 任何 val/test teacher/view artifact、segment gold wording/field、非 SLURM compute、formal namespace overwrite、outer/inner leakage 都使结果 invalid，而不是可忽略 warning。
- SSR/EDCM negatives 只排除各自 frozen sparse-edge/two-swap action spaces；不得把它们写成对 shared representation motion 的理论界。

## 9. Final Checklist

- [x] Claims linked to decisive evidence and stop gates
- [x] C0/C1 exact run IDs, interfaces, resources and provenance frozen
- [x] Complete nested probes and train-only outer OOF defined
- [x] Separate modality medoid IDs, adjacent radii and joint projected/fused support defined
- [x] Exact epoch-refreshed full bank and numerical/drift rules defined
- [x] Two-dataset `+0.05 acc/+0.05 mF1` C1 GO retained
- [x] CTE-0 explicitly not a theoretical upper bound or MLLM evidence
- [x] No segment gold; val/test no teacher/view artifact
- [x] Teacher pilot locked; C2–C4 conditional skeleton only
- [ ] Implementation bridge and independent code audit complete
- [ ] C0 evidence complete
- [ ] C1 evidence complete
