# SQ-RGCL Experiment Plan

**Problem:** 让 train-only MLLM 以有意义、可移除且方法级新颖的方式改变 RGCL 最终 full-video memory geometry，并最终显著提高 ordinary full-video kNN accuracy / macro-F1。  
**Method thesis:** whole-video presentation posterior 只有在它能识别并修复 exact top-20 中的 nuisance-conditioned wrong-class attraction，且 crossed ranking 本身具有 learned OOF capacity 时，才值得支付新 teacher 成本。  
**Date:** 2026-07-11  
**Authoritative method:** `refine-logs/sq/FINAL_PROPOSAL.md`  
**Current execution authorization:** **S0 + S1 only**. S2--S4 are locked. This document is a plan only: it creates no implementation, job, cache, or teacher call.

## 0. Immutable Contract

1. 唯一 gold supervision 是 parent-video binary label。`segment_gold_exists=false`，`segment_gold_used=false`；不存在也不得假定 segment/timestamp/span/localization/stance/target/mechanism/rationale gold。
2. 固定数据集为 MHC-EN（代码名 `MHC`）和 MHC-ZH（代码名 `MHC_zh`）。只使用 train split 的五个 frozen folds：`artifacts/ssr/v1/folds/{MHC,MHC_zh}.json`。outer-held train videos 只作 OOF endpoint；其 label、posterior、archive 或 loss 不进入对应 outer model 的训练。
3. S0/S1 必须记录 `new_teacher_call_count=0`、`teacher_cache_read_count=0`、`teacher_cache_write_count=0`。现有 archive 只可按本计划审计/读取；不得重新生成、补问、repair 或调用任何 MLLM。
4. validation/test 内容、label、teacher/archive/presentation artifact 均不得被 S0/S1 打开或加载；只可复用 SSR freeze manifest 中已存在的 split-ID/hash disjointness 声明。S0/S1 的所有 endpoint 都是 train-OOF。
5. validation/test 以及最终推理永远只有 full video -> shared embedding -> rebuilt full train bank -> ordinary repository top-20 arithmetic cosine kNN；没有 posterior、archive、teacher、presentation head、rerank 或 score fusion。
6. `q^arch` 只有在原始生成 provenance 和 presentation-only blind audit 都通过时，才可标记为 `PROMOTED_ARCHIVE_WEAK_MLLM`。否则它只能标记为 `PROXY_ONLY_CHEAP_FORMAT`，不能支持 MLLM value/novelty claim；S1 可把它用于 action-capacity screen，但不得把 S1 写成 teacher 成功。
7. S0/P0 与 S1/SQ-0 是 empirical fast-fails，不是理论上界。S1 任一数据集失败即终止 frozen SQ route，S2 不得调用 teacher；不得据结果改 ontology、prototype、exposure、loss、top-k、fold、epoch 或 gate。
8. 所有实际计算必须通过 SLURM、`conda activate HateVideo`、不设 `--time`，不手工 release `JobHeldUser`。正式 namespace no-clobber；禁止 `--force`。

## 1. Current Evidence Inventory and Its Meaning

### 1.1 Existing archive facts to freeze in S0

| Dataset | v2 train archive | Current SHA-256 | Rows / unique IDs | parse / schema / summary |
|---|---|---|---:|---:|
| MHC | `data/Archive/MHC/v2/train_Qwen2.5-VL-7B-Instruct_archive.jsonl` | `fa179621b49eefdfdd5f42eee40a2019dfb9918215a2c47e6677ff925ccbafd6` | 549 / 549 | 549 parse; 548 schema; 549 nonempty summary |
| MHC_zh | `data/Archive/MHC_zh/v2/train_Qwen2.5-VL-7B-Instruct_archive.jsonl` | `7c83e3e7f21ea7848d4981c4b4d233df3820ddcc2d2d0e8604ba3ac44df47a27` | 579 / 579 | 579 parse; 573 schema; 579 nonempty summary |

The current files contain outer `label` and `raw_output` plus harm-related archive fields. The only permitted semantic payload for SQ is `archive.neutral_summary`; every other key must produce a reader access count of zero. The current generator code path appears to build the model prompt from frames plus `item["text"]` and writes `item["label"]` only to the output record, but neither archive JSONL nor old SLURM logs embed generation-time prompt/model-revision/input/code hashes. Therefore the inventory does **not yet prove promotion**. S0 must fail closed to `PROXY_ONLY_CHEAP_FORMAT` unless cryptographic, original-run evidence closes that gap.

### 1.2 SSR assets conditionally reusable

- Fold IDs, full rankings, OOF embeddings, checkpoints and prediction ledgers exist for 5 folds x 2 datasets under `artifacts/ssr/v1/oof/`.
- Stored OOF manifests assert train-only queries, query/memory disjointness, video-label-only supervision, and no segment gold; rankings cover the complete outer-train memory.
- S0 must nevertheless hash every consumed file and independently reproduce ranking/vote arithmetic. SSR relations/pairs/events are never training inputs. Their old corrected-error universes may be read only for the S1 reach-beyond diagnostic.
- If a source/hash/fold/evaluator check fails, S0 is `INVALID/STOP`; no silent rebuild from different folds or comparator is allowed.

## 2. Claim Map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1 (dominant, eventual) MLLM presentation crossing x exact-vote-exposed ranking is a causally useful method component | A presentation label that is merely plausible, label-redundant, or decorative cannot satisfy the project goal | S2 teacher posterior beats common-edge archive/base/shuffle/random alternatives; S3/S4 FULL beats REMOVE and SHUFFLE; final ordinary kNN reaches the frozen +.030/+ .030 target | B2--B4, locked |
| C2 (current capacity claim) the exact crossed loss can densely improve shared query/key geometry rather than edit a sparse old universe | SSR/EDCM failed on sparse/frozen correctable units; teacher spend is unjustified without learned capacity | S0 evaluator/P0/cost gates pass; S1 actual train-OOF ordinary kNN gains >=+.050 accuracy and >=+.050 macro-F1 on both datasets, with controls, fold signs and anchor bootstrap | B0, B1 |
| Anti-claim | gain is extra optimization, label-only structure, a cheap/base cluster, archive leakage, changed vote, or segment supervision | S1 matched REMOVE/LABEL_ONLY/BASE_CLUSTER/SHUFFLE/RANDOM; exact evaluator parity; only parent labels; no val/test or teacher artifact | B0, B1 |

## 3. Frozen Signal, Loss, and Evaluator Definitions

### 3.1 Six-way zero-new-call `q_proxy`

S0 constructs one deterministic posterior from **only** `archive.neutral_summary`; it never reads `label`, `raw_output`, target/mechanism/modality/explicitness fields. The six fixed English prototype strings and their order are:

1. `a news report, reportage, interview, documentary, broadcast, or current-events coverage`
2. `a staged comedy sketch, satire, parody, meme performance, role-play, or acted skit`
3. `an educational explanation, tutorial, lecture, analysis, demonstration, or informational presentation`
4. `a personal story, first-person narrative, opinion discussion, conversation, vlog, or direct-to-camera commentary`
5. `gameplay, gaming, music performance, dance, sports, entertainment montage, or edited amusement clip`
6. `presentation style is mixed, ambiguous, or cannot be determined`

Using local `openai/clip-vit-large-patch14-336`, encode each summary alone and each prototype, L2-normalize, and set `q=softmax(cosine/0.10)`. Set `r=max(0,1-H(q)/log(6))`; missing/invalid/nonfinite summary maps to uniform `q` and `r=0`. No CLIP cache that concatenates archive fields may be used. The taxonomy, text, temperature and confidence formula are frozen before outcome computation.

If S0 provenance+blind audit pass, this exact artifact is named `q_arch` with status `PROMOTED_ARCHIVE_WEAK_MLLM`. Otherwise the identical numeric artifact remains `q_proxy` with status `PROXY_ONLY_CHEAP_FORMAT`. Renaming cannot change its scientific status.

### 3.2 Exact repository vote and harmful exposure

For normalized float32 query/key embeddings, memory is inserted in canonical UTF-8 video-ID order. Rank is descending inner product/cosine with exact ties resolved by canonical ID. S0 must verify this tie rule against the repository; any disagreement is STOP pending a new reviewed plan.

For the first 20 neighbors, rank weights are `w_r=21-r` and

`V_i = sum_{r=1}^{20} w_r s_ir (2 y_r - 1)`, `prediction_i = 1[V_i >= 0]`.

The denominator used only for diagnostics is `sum w_r |s_ir|`. Macro-F1 is sklearn binary macro-F1 with `zero_division=0`. There is no similarity threshold and no rank>20 vote or exposure.

For train anchor `i`, exclude self. Let `t_ij=+1` for same video label and `-1` otherwise. The **only** exposure is

`E_i(j)=(21-rho_i(j))*max(0,-t_ij*s_ij)` for `rho<=20`, else `0`.

Thus a negative is eligible only when it is a different-label current top-20 key with positive cosine and nonzero harmful signed contribution. S0 parity must expose the actual top-20 IDs, ranks, cosines, labels, weights, signed contributions and `E` in JSONL—not an approximate neighbor score.

### 3.3 One SQ loss

On each detached epoch-start full outer-train bank:

- `A_ij=sum_e sqrt(q_i[e]q_j[e])`.
- positive candidate weights: `u_ip=r_i r_p(1-A_ip)` over every non-self `y_p=y_i` key;
- negative sampling weights: `v_in=r_i r_n A_in` over only eligible harmful top-20 `y_n!=y_i` keys;
- normalize each side separately and require Kish ESS >=8 on each side; otherwise the anchor has exact REMOVE fallback;
- draw 64 `(p,n)` triplets using stateless RNG SHA-256 keyed by `(config_sha256,seed,epoch,anchor_id,draw_index)`;
- `L_SQ(i)=mean E_i(n)*softplus((s_in-s_ip+0.10)/0.10)`.

The full loss is `L_RGCL + lambda_Q L_SQ`. Keys/exposure/sampling are detached within an epoch; gradients flow through the current anchor embedding. Every video becomes an anchor once per epoch, so the shared encoder and all keys co-move at the next exact bank refresh. No trainable module or inference path is added.

`lambda_Q` is selected in S0 without accuracy/F1: candidates `{0.025,0.05,0.10,0.20}`; choose one global value for both datasets whose pooled fold median first-step auxiliary/base gradient-norm ratio is closest to `0.20`, subject to every dataset median lying in `[0.05,0.50]`; tie -> smaller lambda. No candidate satisfying the band is STOP. This value is frozen before P0/S1 outcomes.

### 3.4 S1 arms and strength matching

Every fold trains from the same serialized initialization and uses identical batches, optimizer steps, epochs, checkpoint rule and bank refreshes.

- `REMOVE`: exact repository base RGCL.
- `BASE_CLUSTER`: six-way label-blind spherical k-means soft posterior fitted only on outer-train frozen base embeddings; seed `20260711`, k-means++ then 100 iterations, canonical centroid order; same SQ loss.
- `LABEL_ONLY`: same exposure and 64-triplet workload, with uniform non-self same-label positives and eligible harmful different-label negatives; no q/r.
- `FULL`: exact promoted `q_arch` or proxy-only `q_proxy` according to S0 status.
- `SHUFFLE`: one no-fixed-point permutation of complete `(q,r)` records within outer-train `class x r-quartile`, hash seed `sq-v1-shuffle`; singleton strata merge deterministically with the adjacent lower quartile, then upper if needed.
- `RANDOM`: logistic-normal posteriors calibrated once within outer-train to match FULL marginal mean, entropy distribution, `r`, missingness and active-anchor mass; record seed `sq-v1-random`.

LABEL_ONLY/BASE_CLUSTER/SHUFFLE/RANDOM each receive one scalar fixed at the first step so their aggregate auxiliary gradient norm equals FULL's, capped to `[0.5,2.0]`; cap activation is reported and is a S0 microbenchmark failure if it occurs for >20% folds. No per-example or later-step matching is allowed.

## 4. Experiment Blocks

### B0 / S0 — provenance, exact evaluator, P0 relevance, power and microbenchmark

- **Claim tested:** existing inputs are auditable; the frozen signal is presentation-like and conditionally vote-relevant; the exact loss/evaluator is correct and feasible before learned OOF training.
- **Data:** train split and frozen SSR OOF assets only.
- **Priority:** MUST-RUN; current execution authority includes S0.

#### S0-A: provenance and reader fail-closed audit

For each archive, freeze JSONL, train-GT ID/text, video manifest, generator source, system/user prompt, sbatch, job log, model identifier/revision/cache snapshot and all hashes that can be tied to the original run. Static data-flow plus a label-poison replay fixture must prove that model messages contain only frames and title/transcript text; label/prediction/margin/fold/seed and all dev/test fields must never enter. The SQ reader runs under an allowlist and must report nonzero access only to `id`, `split`, `parse_ok`, `archive.neutral_summary`; outer `label` is joined later from the frozen train fold file, never read through the archive reader.

Promotion requires original-run cryptographic linkage for prompt, exact model revision, generator code and input manifest. Timestamp adjacency, current clean source, filename, or output plausibility are insufficient. Missing linkage yields `PROXY_ONLY_CHEAP_FORMAT`, not fabricated provenance. ID mismatch, duplicate, non-train row, reader forbidden-key access, or split leakage yields `INVALID/STOP`.

#### S0-B: blind whole-video presentation QC

Freeze 64 unique train videos/dataset by posterior argmax strata: allocate 10 per each of six categories plus four extras to the four largest strata; shortages redistribute by descending stratum size; choose IDs by salted hash `sq-v1-audit`. The exported rater sheet contains random audit ID, whole video, six probabilities and category definitions, but no dataset label/prediction/margin/fold, archive raw output, target/mechanism/explicitness or localization information.

Two independent raters mark `presentation_appropriate {yes,no}`, `semantic_contamination {yes,no}` and one fixed reason code; a third blinded rater adjudicates disagreements. These are QC judgments, never training supervision. Gates per dataset: at most 3/64 contaminated and 95% Wilson lower bound of appropriateness >=0.90. Audit records include rater hashes, timestamps, adjudication and no gold-label column.

#### S0-C: archive positivity and P0 conditional relevance

Required before S1 on both datasets:

1. parse/usable coverage >=90%; each environment x class has `sum r_i q_i[e]>=10` and Kish ESS >=8; each class effective environment count >=2.5; >=80% anchors/class have ESS >=8 on both crossed relations; class-pure cells STOP.
2. On each frozen strict-OOF actual top-20 directed graph, fit five rotations: four folds train, fifth evaluates. Base features are cosine, rank one-hot, anchor class, normalized vote margin, frozen modality energies and six-way base-cluster affinity. Adding posterior affinity must increase held-out wrong-class-neighbor AUC by >=0.030, all five fold deltas positive, and 10,000 anchor-cluster bootstrap 95% lower bound >0.
3. An anchor-level quotient-pressure summary added to the corresponding base controls must increase held-out baseline-error AUC by >=0.030, with the same sign/bootstrap gates.
4. Let the frozen-rank correct-class signed soft-vote margin be `m_i=(2y_i-1)V_i`. First-step alignment is `1[grad(m_i)^T(-grad L_SQ(i))>0]`. FULL alignment rate must exceed both SHUFFLE and RANDOM by >=0.10 with 10,000 anchor bootstrap lower bounds >0.

Edges are never independent replicates. Every bootstrap resamples anchors within `outer_fold x video_class`; all outgoing top-20 edges follow their anchor. Seed is `20260711`; percentile 95% interval; exactly 10,000 replicates. Undefined single-class AUC anchors remain in the graph but do not become fake edge observations.

#### S0-D: S2 power pre-registration (no teacher selection/call)

Power is computed now so the later pilot cannot adapt its sample after teacher outputs. On the complete q-free strict-OOF top-20 graph, compute per-anchor paired `cheap-minus-shuffle` differences for (i) within-anchor wrong-vs-correct neighbor AUC and (ii) first-step positive-alignment indicator. For every dataset x class x estimand, use

`variance_U=min(1,(N-1)*s^2 / chi2_ppf(0.05,N-1))`.

If `N<3`, variance is nonfinite, or the estimand lacks both outcomes, S2 is `STOP_INFEASIBLE`. Freeze conservative `alpha*=0.05/8=0.00625` (the smallest Holm threshold for 2 datasets x 2 classes x 2 estimands), power 0.80, effects `delta_AUC=0.02`, `delta_align=0.10`, and

`n0=ceil((z_(1-alpha*/2)+z_0.80)^2 * variance_U / delta^2)`,  
`n_FPC=ceil(N*n0/(N-1+n0))`.

The class requirement is the maximum of its two `n_FPC` values. S2 must later sample exactly this many anchors uniformly without replacement within class x OOF-margin quartile, proportional to full stratum sizes, then add every sampled anchor's frozen top-20 endpoint. If graph closure exceeds 128 unique videos/dataset, any stratum is unsampleable, or the achieved FPC power is <0.80, write `STOP_INFEASIBLE` before calls. No shrinking, hub optimization or alpha/effect relaxation is permitted.

#### S0-E: evaluator/loss parity and GPU microbenchmark

- Synthetic and every actual OOF query: helper ranking/vote must match `src/model/evaluate_rac.py` + `src/utils/metrics.py` IDs, float32 cosines, ranks, signed vote, prediction, accuracy and macro-F1. Include exact-cosine ties, negative cosine, vote exactly zero and noncontiguous IDs.
- Scalar float64 versus vectorized float32 parity: max cosine/vote/exposure/loss error <=`2e-5`; gradient relative error <=`1e-3`; 100% finite.
- Batch 64, 64 triplets/active anchor, complete bank, six arms: 20 warmups + 200 timed iterations; peak allocated <=24 GiB; FULL median step <=2.0x REMOVE; measured fold and total S1 GPU-hours become binding estimates.
- Freeze global `lambda_Q` and report active-anchor coverage, ESS, gradient ratios, top-20 exposure counts and cap activations.

#### S0 joint gate

`SQ-S0-DECISION-v1` independently rehashes/recomputes every gate. S1 may run when SSR reuse, reader, positivity, P0, evaluator, numerics and resource gates all pass. Archive provenance/audit determines `q_signal_status`; a provenance-only failure may route to `PROXY_ONLY_CHEAP_FORMAT`, but a reader/semantic/positivity/P0/evaluator/resource failure is STOP. `new_teacher_call_count` must be zero.

### B1 / S1 — learned strict-OOF SQ-0 actual ordinary-kNN screen

- **Claim tested:** the frozen crossed action family has enough learned capacity to justify a representative teacher pilot and is not explained by label-only, base clustering, shuffle or matched random optimization.
- **Setup:** 5 outer folds x 2 datasets, seed 0. Each fold job serially trains the six arms in the order `REMOVE, BASE_CLUSTER, LABEL_ONLY, FULL, SHUFFLE, RANDOM`, restoring the identical initialization before each arm. Dataset epoch recipes remain MHC epoch index 25 / `full,lambda_seg=.5`, MHC_zh epoch index 28 / `milmax,lambda_seg=.5`, matching the frozen SSR comparator. No outcome-based hyperparameter selection occurs.
- **Endpoint:** outer-held query uses full video only; memory is the complete rebuilt outer-train full-video bank; exact ordinary top-20 arithmetic cosine vote. The evaluator process rejects all q/archive/teacher arguments and checks zero such reads.
- **Priority:** MUST-RUN; current execution authority includes S1 only after verified S0 GO.

#### Binding S1 GO gates, all required

1. All 10 fold manifests and six-arm ledgers pass hashes, row partitions, identical initialization/order/steps/epochs/refresh and zero-call/zero-val-test audits.
2. For each dataset and metric separately, `FULL - max(REMOVE,BASE_CLUSTER) >= +0.050` for accuracy and macro-F1 on the concatenated train-OOF predictions. This moving non-MLLM comparator is metric-specific and cannot be replaced by a weaker historical arm.
3. For each dataset and metric, FULL exceeds each of LABEL_ONLY, SHUFFLE and RANDOM by >=`+0.010`.
4. For FULL versus the moving comparator, each of five fold accuracy deltas and each of five fold macro-F1 deltas is strictly positive.
5. Paired 10,000-replicate anchor bootstrap, stratified by outer fold x video class and recomputing nonlinear macro-F1, has 95% lower bound >0 for every required comparison in gates 2--3. The four primary dataset x metric FULL-vs-moving-comparator p-values pass Holm FWER 0.05.
6. Actual wrong-class top-20 signed mass `sum_{wrong,r<=20} w_r max(s_ir,0)` is lower than the moving comparator with anchor-bootstrap lower bound >0 for the paired reduction; net corrected-minus-broken errors are positive in both classes.
7. At least one corrected baseline error/dataset lies outside the union of the frozen SSR candidate-correctable and EDCM top64/two-swap reachable error IDs, and outside-union corrected-minus-broken count is positive. This is a diagnostic of wider action, not an oracle claim.
8. No archive/teacher/native-head/localization/segment metric may substitute for an ordinary-kNN failure. If `q_signal_status=PROXY_ONLY_CHEAP_FORMAT`, even a GO is only action-capacity evidence; it unlocks S2 to test a fresh graph-closed teacher, not an MLLM claim.

Any missing/failed cell writes `SQ-0=STOP`, `S2_unlocked=false`. No partial-dataset rescue, prompt/model scaling, alternate q construction, ontology edit, lambda sweep or additional seed is allowed.

## 5. Exact Authorized Execution Interface

### 5.1 Planned files (implementation handoff; this plan does not create them)

```text
configs/sq/sq_v1.json
scripts/analysis/sq_common.py
scripts/analysis/sq_s0.py
scripts/analysis/sq_s1.py
scripts/slurm/sq_s0_cpu.sbatch
scripts/slurm/sq_s0_gpu.sbatch
scripts/slurm/sq_s1_gpu.sbatch
scripts/slurm/sq_s1_cpu.sbatch
```

Frozen Python interface:

```text
sq_s0.py --config configs/sq/sq_v1.json --task freeze --run-id SQ-S0-FREEZE-v1
sq_s0.py --config ... --task provenance --dataset DATASET --run-id RUN_ID
sq_s0.py --config ... --task qproxy --dataset DATASET --run-id RUN_ID
sq_s0.py --config ... --task audit-freeze --run-id SQ-S0-AUDIT-FREEZE-v1
sq_s0.py --config ... --task audit-ingest --dataset DATASET --audit-csv PATH --run-id RUN_ID
sq_s0.py --config ... --task parity-power-p0 --dataset DATASET --run-id RUN_ID
sq_s0.py --config ... --task micro --dataset DATASET --run-id RUN_ID
sq_s0.py --config ... --task decide --run-id SQ-S0-DECISION-v1
sq_s1.py --config configs/sq/sq_v1.json --task outer --dataset DATASET --outer-fold F --run-id RUN_ID
sq_s1.py --config ... --task decide --run-id SQ-S1-DECISION-v1
```

Frozen SLURM interface:

```text
TASK=freeze RUN_ID=SQ-S0-FREEZE-v1 sbatch scripts/slurm/sq_s0_cpu.sbatch
TASK=provenance DATASET=D RUN_ID=... sbatch scripts/slurm/sq_s0_cpu.sbatch
TASK=qproxy DATASET=D RUN_ID=... sbatch scripts/slurm/sq_s0_gpu.sbatch
TASK=audit-freeze RUN_ID=... sbatch scripts/slurm/sq_s0_cpu.sbatch
TASK=audit-ingest DATASET=D AUDIT_CSV=P RUN_ID=... sbatch scripts/slurm/sq_s0_cpu.sbatch
TASK=parity-power-p0 DATASET=D RUN_ID=... sbatch scripts/slurm/sq_s0_cpu.sbatch
TASK=micro DATASET=D RUN_ID=... sbatch scripts/slurm/sq_s0_gpu.sbatch
TASK=decide RUN_ID=SQ-S0-DECISION-v1 sbatch scripts/slurm/sq_s0_cpu.sbatch
TASK=outer DATASET=D OUTER_FOLD=F RUN_ID=... sbatch scripts/slurm/sq_s1_gpu.sbatch
TASK=decide RUN_ID=SQ-S1-DECISION-v1 sbatch scripts/slurm/sq_s1_cpu.sbatch
```

All scripts reject absence of `SLURM_JOB_ID`, wrong conda env, nonfrozen config, existing output namespace, or a non-GO predecessor. CPU: partition `slurmpartition`, 4 CPU / 16 GB. qproxy/micro GPU: 1 A100 / 4 CPU / 32 GB. S1 fold GPU: 1 A100 / 8 CPU / 64 GB. No `--time`; at most two GPU jobs concurrently.

### 5.2 Exact run IDs

- S0 freeze: `SQ-S0-FREEZE-v1`.
- Provenance: `SQ-S0-PROVENANCE-{MHC|MHC_zh}-v1`.
- Signal construction: `SQ-S0-QPROXY-{MHC|MHC_zh}-v1`.
- Blind audit: `SQ-S0-AUDIT-FREEZE-v1`; `SQ-S0-AUDIT-{MHC|MHC_zh}-v1`.
- Evaluator/P0/power: `SQ-S0-PARITY-POWER-P0-{MHC|MHC_zh}-v1`.
- Microbenchmark: `SQ-S0-MICRO-{MHC|MHC_zh}-S0-v1`.
- Joint S0: `SQ-S0-DECISION-v1`.
- S1 folds: `SQ-S1-OOF-{MHC|MHC_zh}-F{0..4}-S0-v1` (10 SLURM jobs; each emits six arm-specific subrun manifests named `<RUN_ID>-<ARM>`).
- Joint S1: `SQ-S1-DECISION-v1`.

## 6. Artifact / JSON Provenance Contract

```text
artifacts/sq/v1/
  CONFIG_FREEZE.json
  s0/provenance/{MHC,MHC_zh}.json
  s0/qproxy/{MHC,MHC_zh}/{posterior.jsonl,manifest.json}
  s0/audit/{sample_manifest.json,MHC.csv,MHC_zh.csv,MHC_result.json,MHC_zh_result.json}
  s0/p0/{MHC,MHC_zh}/{edge_ledger.jsonl,anchor_ledger.jsonl,power.json,metrics.json,manifest.json}
  s0/micro/{MHC,MHC_zh}/{timings.json,numerics.json,exposure_examples.jsonl,manifest.json}
  S0_DECISION.json
  s1/oof/<dataset>/fold<F>/<arm>/{checkpoint.pt,predictions.json,neighbors.jsonl,training.jsonl,manifest.json}
  S1_DECISION.json
```

Every manifest/decision contains at least:

`schema_version,run_id,stage,status,slurm_job_id,git_head,dirty_diff_sha256,conda_env,python/torch/faiss/sklearn/cuda versions,gpu_name,config_canonical_sha256,implementation_sha256,input_files[{path,sha256}],fold_ids_sha256,initialization_sha256,output_files[{path,sha256}],payload_sha256,only_gold_supervision,segment_gold_exists,segment_gold_used,new_teacher_call_count,teacher_cache_read_count,teacher_cache_write_count,archive_forbidden_key_access_count,outer_held_q_read_count,val_content_read_count,test_content_read_count,val_test_teacher_artifact_count`.

S0 provenance additionally contains original-run evidence and `q_signal_status`; parity contains all actual top-20 components; power contains `N,s2,variance_U,alpha_star,delta,n0,n_FPC,closure_upper_bound`; training contains exact params, batches, optimizer steps, epoch-bank hashes, active/fallback anchors, ESS, sampled-triplet hash, gradient strength, q IDs read and evaluator reject-log.

JSON is sorted-key UTF-8 canonical serialization with compact separators and no NaN/Inf. `payload_sha256` is computed after removing itself. Formal outputs use persistent `O_CREAT|O_EXCL` locks and temp+fsync+atomic no-clobber publish. Decision phases independently recompute metrics from row ledgers and authoritative inputs; producer pass flags are not trusted.

## 7. Run Order, Gates, and Cost

| Milestone | Goal | Runs | Decision gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | implementation/static review | planned files above | independent review 0 CRITICAL/HIGH; frozen config/hash | CPU only | leakage/provenance semantics |
| M1 | freeze + provenance + q proxy | freeze, 2 provenance, 2 qproxy | valid reader; promoted or explicit proxy-only status | 2 short GPU + CPU jobs | old archive lacks original hashes |
| M2 | blind QC | audit freeze + 2 ingests | contamination/Wilson gates | human QC + CPU ingest; not gold annotation | presentation categories collapse |
| M3 | exact P0/power/parity/micro | 2 CPU analyses + 2 GPU micro + decision | every non-provenance S0 gate passes | measured in micro; expected <2 GPU-h total | q is vote-orthogonal; loss too costly |
| M4 | learned S1 OOF | 10 fold GPU jobs + decision | every eight-part S1 gate on both datasets | micro estimate binding; likely 10--30 GPU-h | +.05/+ .05 not reachable |
| M5 | S2--S4 | LOCKED | only after verified S1 GO | not authorized | teacher value/final gain may fail |

## 8. Locked S2--S4 Skeleton

### S2 representative graph-closed teacher pilot — LOCKED

Only after `SQ-S1-DECISION-v1` verifies GO. Use S0 frozen powered anchor counts and q-free graph; sample by class x OOF-margin quartile, close all top-20 endpoints, and STOP before calls if >128 unique videos/dataset or power <80%. At most four invocations/video (two prompts x two input orders), hence <=512 invocations/dataset. The actual teacher sees whole-video presentation only, never label/prediction/margin/neighbor/fold/segment fields. Every teacher/archive/base/shuffle/random comparison uses the identical closed vertices/edges and inverse-probability weighted anchor estimand. No val/test call.

### S3 seed-0 controls — LOCKED

Only after S2 GO. On both datasets, dev ordinary kNN FULL must exceed REMOVE, LABEL_ONLY, SHUFFLE, RANDOM, BASE_CLUSTER, CHEAP-FORMAT, ENV-SUPCON, Yang-style and P4-PREDICT by >=.010 accuracy and >=.010 macro-F1; posterior corruption/masking must degrade monotonically. Dev evaluator loads no teacher artifact.

### S4 final two datasets x three seeds — LOCKED

Only after S3 GO. Seeds 0/1/2; FULL ordinary test kNN must improve both metrics by >=.030 over `max(historical strongest,paired same-seed strongest non-MLLM mean)` on both datasets; all paired signs positive; 10,000 hierarchical bootstrap lower bounds >0; four primary tests pass Holm FWER .05; FULL significantly beats REMOVE and SHUFFLE. No test-time teacher/archive/presentation artifact. Only S4 can complete the project objective.

## 9. Final Checklist

- [x] S0/S1 claims, exact signal status, loss, vote/exposure and controls are frozen.
- [x] Archive promotion requires original provenance plus blind presentation QC; proxy status cannot masquerade as MLLM evidence.
- [x] Power worst-case variance, finite-population correction, graph closure and anchor bootstrap are frozen before teacher outputs.
- [x] Exact run IDs, planned scripts, SLURM resources, JSON schemas and no-clobber rules are specified.
- [x] Only parent-video labels are gold; no segment gold is assumed or used.
- [x] Validation/test cannot load teacher/archive/presentation artifacts.
- [x] Current authorization is S0/S1 only and new teacher calls are forbidden.
- [ ] S0/S1 implementation exists and passes independent code review.
- [ ] `SQ-S0-DECISION-v1` verifies GO.
- [ ] `SQ-S1-DECISION-v1` verifies both-dataset +.05/+ .05 GO.
- [ ] S2 is unlocked; currently false.
- [ ] Final two-dataset x three-seed +.03/+ .03 target is proven.
