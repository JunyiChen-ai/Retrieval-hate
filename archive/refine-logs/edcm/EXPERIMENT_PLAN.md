# EDCM-RGCL Iteration 2 Experiment Plan

**Problem:** Integrate an MLLM into hateful-video RGCL as a meaningful, novel, removable train-only mechanism and obtain at least `+0.030` absolute final test accuracy and macro-F1 on two datasets under the frozen paired three-seed protocol.  
**Method thesis:** Relative, label-blind MLLM judgments over deterministic `V/S/O` coalitions define a privileged train-only measure that changes the listwise gradient of the ordinary full-video retrieval memory and is completely discarded at validation/test.  
**Date:** 2026-07-10 (Pacific/Auckland)  
**Current authorization:** Plan A0--A3; implement/execute **A0 only**. No MLLM call, teacher cache, `src/` edit, or SLURM submission is authorized by this planning task.

## 0. Non-Negotiable Contract

1. The only gold supervision is the **parent video's binary label**. No segment-level gold annotation exists or may be assumed.
2. Uniform frames are whole-video input samples, transcripts have no temporal gold, and OCR is deterministic input extraction. Coalition ranks, confidence, signatures, stance-like interpretations, or any later MLLM field are weak/privileged train-only pseudo-signals, never gold, annotation, dense supervision, or oracle evidence.
3. A0 uses only five-fold train OOF full-video geometry and video-level labels. A0 must reach `GO` on both `MHC` (MHC-EN) and `MHC_zh` (MHC-ZH) before any MLLM process is launched or any teacher artifact is created.
4. Validation/test must not load or generate MLLM, OCR, coalition, signature, confidence, proxy-fit, or teacher-reliability artifacts. Teacher/proxy artifacts may contain train IDs only. Development selects checkpoints under the unchanged comparator rule; test is evaluated only after the corresponding gate unlocks it.
5. The strongest dataset-specific RGCL comparator is frozen: `MHC: lambda_seg=0.5, seg_mode=full, epoch index 25 for OOF`; `MHC_zh: lambda_seg=0.5, seg_mode=milmax, epoch index 28 for OOF`; CLIP model, align fusion, optimizer, loss, top-20 arithmetic similarity vote, splits, preprocessing, and label space remain unchanged.
6. Every artifact is canonical JSON/JSONL with finite numbers only, deterministic key/order rules, SHA-256 provenance, `run_id`, `slurm_job_id`, source hashes, config/code hashes, supervision declaration, and a payload hash. Missing, malformed, hash-mismatched, overlapping, incomplete, or unverifiable evidence is `STOP`, never an implicit pass.
7. All computation is SLURM-only in `HateVideo`, with no `--time`, and within the per-user `2 GPU / 16 CPU / 128 GB` limit. `JobHeldUser` is allowed to clear automatically and is never manually released.

## 1. Claim Map

| Claim | Why it matters | Minimum convincing evidence | Linked blocks |
|---|---|---|---|
| **C1 (primary):** MLLM coalition semantics create a broad, teacher-specific listwise gradient that is not reducible to video labels, generic modality/content statistics, or extra optimization strength. | Prior routes were sparse, label-redundant, absorbed by fusion, or moved accuracy between head and memory. | A0 reaches target-sized nearby correction support before teacher cost; then A1 passes reliability, TV/`R`, reachable-error `DeltaD`, and a strength-matched semantic-free proxy gate on both datasets. | B0, B1 |
| **C2 (primary):** Internalizing that gradient causally repairs the same full-video kNN geometry used at inference and yields the frozen final `+3/+3` endpoint. | Localization, native-head-only, or audit gains do not meet the project objective. | A2 full beats remove, Label-only, proxy, and shuffle in both dev metrics while kNN topology improves; A3 passes two datasets x seeds 0/1/2, statistics, removal/shuffle costs, and corruption survival. | B2, B3 |
| **Anti-claim:** Gains come from label-only ListNCA, low-level modality availability, extra loss strength, changed protocol, teacher artifacts at validation/test, or nonexistent segment gold. | Any of these makes the MLLM decorative or invalidates the result. | Exact controls and provenance; no validation/test teacher rows; parent-label inheritance explicitly distinguished from segment gold; identical schedules and final vote. | B0--B3 |

## 2. Paper Storyline

- **Main paper must prove:** one retrieval-specific mechanism; final paired EN/ZH `+3/+3`; remove/shuffle/corruption attribution; improvement in the actual kNN readout.
- **Appendix can support:** A0 reachability distributions, A1 TV/`R`/`DeltaD`, proxy strength match, coverage/fallback, class recall, embedding variance, and detailed provenance.
- **Intentionally cut:** localization, teacher-selected memory keys, student coalition branches, generated content, relation graphs/SSR stacking, routers/adapters/extra heads, rationale or score concatenation, test-time MLLM reranking/veto, teacher/model-size sweeps, segment objectives, and any segment-gold claim.

## 3. A0 Is the Only Currently Executable Stage

### 3.1 Authoritative SSR OOF reuse candidates

The existing assets are **eligible candidates, not yet authorized inputs**. They match the intended comparator recipe on inspection and contain full fold-local rankings, but must pass `EDCM-A0-REUSE-AUDIT-v1` before reuse.

Frozen high-level identity:

| Item | Expected value |
|---|---|
| Repository HEAD at planning time (SSR scripts are untracked, so HEAD alone is not provenance) | `a1b1922bc970bb831526b4d21c911380ec871248` |
| SSR canonical config digest | `b32e57e8361392516c0e5087ea8c7bb5c85a6544ba384c5ea4bb3e9a4d64718e` |
| `configs/ssr/ssr_v1.yaml` file SHA-256 | `dde8fdd0f0d62632a2b462c969cffb195883f81219a58af0180d0fbbc0bd9d5d` |
| MHC fold artifact SHA-256 | `bc7e3088faeeb6fad091b97fd1d10a506ed4a53826d2580b9a92891646a3584d` |
| MHC-ZH fold artifact SHA-256 | `b3aaeb2af131316302e1c8f6d249e1770c70b2ce23f9a83fc250916979e9887f` |
| OOF runs | `SSR-B0-OOF-MHC-F0-S0` ... `F4-S0` (jobs 12691--12695); `SSR-B0-OOF-MHC_zh-F0-S0` ... `F4-S0` (jobs 12696--12700) |
| MHC train rows / fold bank rows | `549 / 439--440` |
| MHC-ZH train rows / fold bank rows | `579 / 463--464` |
| Retrieval/vote | canonical `(-cosine, id)` order; `k=20`; arithmetic weights `20..1`; cosine-similarity signed vote; prediction `vote >= 0` |
| Search radius / swaps | first 64 fold-local keys; at most 2 replacements |

Execution-recipe hashes that A0 must record and compare before trusting outputs:

| File | Expected SHA-256 |
|---|---|
| `scripts/analysis/ssr_oof.py` | `274e2a5c32ded808dd6c7cb4a3ccf256090662a39281e5a5b1bb9fef36dc0710` |
| `scripts/analysis/ssr_common.py` | `4971498b5e145ef88bff6c5d10768d57fd1bd8a8bd3af88c0da15938e8b69786` |
| `src/data_loader/rac_dataloader.py` | `3545cffc7e6c1e136bd25ceed547b62c4983d37cabb77aecae317f29b1dbef2e` |
| `src/model/classifier.py` | `82c0d89a8b788c12d08a702cf9aebb88bddffbaea8f9d785c6e24644a94b04d9` |
| `src/model/loss.py` | `89f497b07bdeabcbce14b31941b82e8fe4df8628224b22e79ae1b4b0b80dbfa9` |
| `src/utils/retrieval.py` | `b2877a84421e860ef8313fd2e708ad10c783911b801cbe39ad1531a5706dd025` |
| `src/utils/metrics.py` | `1dd00feaad5c5b917971567291dede5a85f1362c8a3fcebd07350569527b9f35` |

These code hashes supplement rather than replace per-fold manifest/output hashes. Any expected code, config, data, fold, manifest, or output hash mismatch is `STOP`; a new strict OOF rerun then requires separate authorization. Reproducing only the final metric cannot waive a hash mismatch.

The old comparator uses K4 subclips internally because those are part of the strongest RGCL recipe. Their values are inherited **parent-video labels**, explicitly not segment labels or segment gold. A0 never treats a subclip as a labeled endpoint and consumes only the final full-video query/key embeddings plus parent-video labels.

### 3.2 Exact A0 two-swap semantics

Constants are frozen route-wide: `k=20`, search depth `64`, `max_swaps=2`, both datasets, five folds, seed 0 OOF, and video-level binary labels only.

For every OOF query:

1. Verify the query belongs to exactly one held-out fold `F`; all ranked keys belong to `T\F`; query/key overlap is empty; the list has at least 64 unique keys; IDs, labels, and ranking hashes agree with the fold manifest.
2. Sort by descending stored cosine and then canonical ID. Recompute the exact baseline top-20 signed vote
   `v = sum_{r=1}^{20} (21-r) * cosine_r * (2*y_r-1)` and predict `1[v>=0]`. It must match both `predictions.json` and the repository retrieval metric implementation.
3. Candidate support is evaluated on **all** OOF videos: count same-parent-label and opposite-parent-label keys among ranks 1--64. A query is supported iff both counts are at least four.
4. Reachability is evaluated only for baseline-wrong queries. For `m=1` and then `m=2`, exhaustively choose `m` distinct opposite-query-label keys from ranks 1--20 and `m` distinct same-query-label keys from ranks 21--64. Replace the selected top-20 records with the selected outside records, retaining each outside record's stored ID, video label, and cosine. Canonically re-sort the resulting 20-record set by `(-cosine, id)`, then apply the same exact vote. No label is edited and no key beyond rank 64 is used.
5. A query is reachable iff any enumerated edited list predicts its video label. The canonical witness is selected by: fewer swaps; then largest true-class signed margin `(2*y_q-1)*v`; then lexicographically smallest sorted removed-ID tuple; then added-ID tuple. This is an optimistic label-oracle **cost/reachability diagnostic**, not evidence that the MLLM can choose a witness and not an upper bound on learned EDCM.
6. The oracle prediction vector changes all and only reachable baseline errors to their canonical-witness correct predictions. Accuracy and macro-F1 use the repository definitions (`zero_division=0` for macro-F1). Correct baseline predictions are never changed.

Per-dataset binding gates:

| Gate | MHC | MHC_zh |
|---|---:|---:|
| All-video supported fraction | `>=0.80` | `>=0.80` |
| Unique reachable errors | `>=ceil(.05*549)=28` | `>=ceil(.05*579)=29` |
| Oracle accuracy gain | `>=0.050` | `>=0.050` |
| Oracle macro-F1 gain | `>=0.050` | `>=0.050` |
| Fold/output/vote/metric/provenance checks | all pass | all pass |

Every inequality is inclusive. A0 is joint `GO` only if every row passes independently on both datasets. No averaging across datasets, no metric trade-off, no threshold/prompt/model tuning, and no teacher call is permitted after a failed cell.

### 3.3 Planned A0 interfaces and artifacts

These files are implementation targets and do not yet exist:

```text
configs/edcm/edcm_v1.json
scripts/analysis/edcm_a0.py
scripts/slurm/edcm_a0_cpu.sbatch
artifacts/edcm/v1/a0/
```

Frozen Python interface:

```bash
python scripts/analysis/edcm_a0.py \
  --config configs/edcm/edcm_v1.json \
  --task reuse-audit \
  --run-id EDCM-A0-REUSE-AUDIT-v1

python scripts/analysis/edcm_a0.py \
  --config configs/edcm/edcm_v1.json \
  --task reachability \
  --dataset MHC \
  --run-id EDCM-A0-REACH-MHC-v1

python scripts/analysis/edcm_a0.py \
  --config configs/edcm/edcm_v1.json \
  --task reachability \
  --dataset MHC_zh \
  --run-id EDCM-A0-REACH-MHC_zh-v1

python scripts/analysis/edcm_a0.py \
  --config configs/edcm/edcm_v1.json \
  --task decision \
  --run-id EDCM-A0-DECISION-v1
```

No `--force`, alternate top-k, search-depth, swap-count, threshold, or dataset alias is exposed. Existing nonempty outputs are never overwritten.

Frozen SLURM interface:

```bash
TASK=reuse-audit RUN_ID=EDCM-A0-REUSE-AUDIT-v1 \
  sbatch scripts/slurm/edcm_a0_cpu.sbatch

TASK=reachability DATASET=MHC RUN_ID=EDCM-A0-REACH-MHC-v1 \
  sbatch --dependency=afterok:<audit_job> scripts/slurm/edcm_a0_cpu.sbatch

TASK=reachability DATASET=MHC_zh RUN_ID=EDCM-A0-REACH-MHC_zh-v1 \
  sbatch --dependency=afterok:<audit_job> scripts/slurm/edcm_a0_cpu.sbatch

TASK=decision RUN_ID=EDCM-A0-DECISION-v1 \
  sbatch --dependency=afterok:<mhc_job>:<mhc_zh_job> scripts/slurm/edcm_a0_cpu.sbatch
```

`edcm_a0_cpu.sbatch` is frozen at partition `slurmpartition`, `--cpus-per-task=4`, `--mem=16G`, no GPU, no `--time`, `HateVideo`, offline mode, and disk guard. The two reachability jobs may run together (8 CPU / 32 GB total). The decision job is never submitted before both dataset jobs complete successfully.

Required outputs:

```text
artifacts/edcm/v1/a0/reuse_audit.json
artifacts/edcm/v1/a0/MHC/reachability.jsonl
artifacts/edcm/v1/a0/MHC/metrics.json
artifacts/edcm/v1/a0/MHC/manifest.json
artifacts/edcm/v1/a0/MHC_zh/reachability.jsonl
artifacts/edcm/v1/a0/MHC_zh/metrics.json
artifacts/edcm/v1/a0/MHC_zh/manifest.json
artifacts/edcm/v1/A0_DECISION.json
```

Each `reachability.jsonl` row must include query/fold/video label, baseline vote/prediction, support counts, reachable flag, minimal swaps, canonical witness IDs and post-swap vote/prediction, plus hashes of the exact top-64 records. `metrics.json` must include `N`, confusion matrices, accuracy/macro-F1 before and after, both deltas, supported/reachable counts and fractions, and each gate boolean.

`A0_DECISION.json` must contain at least:

```json
{
  "schema_version": 1,
  "run_id": "EDCM-A0-DECISION-v1",
  "stage": "A0_pre_mllm_frozen_geometry_reachability_cost_screen",
  "decision": "GO_or_STOP",
  "A1_unlocked": false,
  "edcm_mllm_calls_before_decision": 0,
  "only_gold_supervision": "video_level_binary_label",
  "segment_gold_exists": false,
  "segment_gold_used": false,
  "validation_test_teacher_artifact_count": 0,
  "config_sha256": "...",
  "source_manifest_sha256": {"MHC": "...", "MHC_zh": "..."},
  "dataset_gates": {"MHC": {}, "MHC_zh": {}},
  "all_binding_gates_pass": false,
  "payload_sha256": "...",
  "slurm_job_id": "..."
}
```

`A1_unlocked` may be `true` only when `decision="GO"`, both dataset manifests and payload hashes verify, and `edcm_mllm_calls_before_decision=0`. JSON keys may be extended but not removed or reinterpreted.

### 3.4 A0 STOP / GO interpretation

- **STOP:** any reuse/provenance discrepancy, fold leakage, candidate/vote mismatch, candidate-support failure, reachable-count failure, or either metric gain below `0.050` on either dataset. Record the exact failed cell. Do not call an MLLM, rebuild teacher inputs, change search depth/swaps, or tune the comparator. The global project goal remains active, but this frozen EDCM route fails its cost screen.
- **GO:** only proves that the existing nearby full-video OOF geometry contains enough video-label-only correction opportunities to justify teacher cost. It does not prove teacher reliability, MLLM novelty/usefulness, learned improvement, or final `+3/+3`.

## 4. Conditional A1--A3 Skeleton (Locked Until A0 GO)

This section freezes the conditional story without authorizing implementation or execution.

### B1 / A1: coalition extraction and teacher-active gate

- **Exact run IDs:** `EDCM-A1-TEACHER-FREEZE-v1`; `EDCM-A1-SMOKE-Q25VL7B-v1`; `EDCM-A1-EXTRACT-MHC-v1`; `EDCM-A1-EXTRACT-MHC_zh-v1`; `EDCM-A1-GATE-MHC-v1`; `EDCM-A1-GATE-MHC_zh-v1`; `EDCM-A1-DECISION-v1`.
- **Mechanism:** frozen Qwen2.5-VL-7B, 2 prompts x 2 orders, four uniform whole-video frames, dataset transcript, title+frozen OCR, seven `V/S/O` coalitions, strict JSON ordinals only. No hate label, rationale, timestamp, span, segment, or localization output.
- **Gate:** on each dataset, reliable signature coverage `>=85%` with Wilson lower bound `>=0.80`; reliable 8+8 lists `>=80%`; teacher-active `>=70%` of all OOF videos and `>=60%` of A0-reachable errors; mean reachable-error `DeltaD>0` with 10,000-query-bootstrap lower bound `>0`; individual positive `DeltaD>=60%`. Select one shared EN+ZH proxy temperature only from the frozen 65-point grid by pooled median-`R` match within 5%; no accuracy/F1/`DeltaD` selection.
- **Resources/order:** teacher freeze CPU; one 1xA100, 8 CPU, 64 GB smoke; then one extraction job at a time with the same resources; fold-local gate jobs CPU 8 / 32 GB; joint decision last. No `--time`.
- **STOP:** any reliability, activity, directional, provenance, or proxy-strength gate fails on either dataset. No A2 tuning rescue.

### B2 / A2: seed-0 causal mechanism gate

- **Exact run namespace:** for every `D in {MHC,MHC_zh}`, exactly `EDCM-A2-D-S0-{REMOVE,LABEL,PROXY,FULL,SHUFFLE,NOISE1X,NOISE2X}-v1` plus `EDCM-A2-DECISION-v1`.
- **Frozen arms/order:** REMOVE -> LABEL -> PROXY -> FULL -> SHUFFLE -> NOISE1X -> NOISE2X; identical seed-0 initialization, data order, epochs, checkpoint rule, and 8+8 workload. Only the registered auxiliary weights differ.
- **Gate:** using dev kNN only, FULL exceeds REMOVE, LABEL, PROXY, and SHUFFLE by `>=0.010` accuracy and `>=0.010` macro-F1 separately on both datasets; same-label purity rises and wrong-neighbour rate falls vs REMOVE; native-head movement cannot substitute for kNN improvement; `clean > noise1x > noise2x` in both metrics and `noise1x > remove` in both metrics.
- **Artifact isolation:** training may load train-ID teacher records; dev/test evaluators must reject any teacher/proxy artifact argument. A2 does not evaluate test. Seed-0 selected checkpoints are sealed for potential A3 test evaluation without retraining.
- **Resources/order:** 1xA100, 16 CPU, 120 GB per arm, strictly one training job at a time; no `--time`.

### B3 / A3: two-dataset, paired three-seed endpoint

- **Exact run namespace:** for `D in {MHC,MHC_zh}`, `S in {0,1,2}`, and `A in {REMOVE,FULL,SHUFFLE,NOISE1X}`, exactly `EDCM-A3-D-S-A-v1`; add `EDCM-A3-D-S0-NOISE2X-v1` and joint `EDCM-A3-FINAL-STATS-v1`. Seed-0 training provenance must point to sealed A2 checkpoints; A3 performs their first test evaluation rather than retraining them.
- **Historical lower bounds:** MHC uses `0.7888 accuracy / 0.7262 macro-F1`, hence no final claim below `0.8188 / 0.7562`; MHC-ZH uses `0.8255 / 0.7875`, hence no claim below `0.8555 / 0.8175`. If the paired REMOVE mean is higher for a metric, its mean plus `0.030` replaces that historical target.
- **Final gate:** on both datasets, FULL accuracy and macro-F1 each exceed `max(historical strongest point, paired REMOVE mean)` by `>=0.030`; all 3/3 paired seed deltas are positive; report mean+/-std and hierarchical paired-bootstrap CIs; all four dataset x metric primary tests pass Holm FWER `0.05` with lower bounds `>0`; FULL beats REMOVE and SHUFFLE with same-direction paired effects and CIs excluding zero in both metrics; NOISE1X has positive per-seed deltas over REMOVE, retains at least 50% of clean mean gain, and remains below clean; seed-0 NOISE2X does not exceed NOISE1X in either metric.
- **Resources/order:** same exact per-arm budget as A2, one job at a time; one-time test evaluation only after all configs/checkpoints are sealed; CPU-only final statistics last.
- **STOP:** any dataset, metric, seed sign, statistical, removability, corruption, kNN-locus, supervision, novelty, or protocol item is missing or fails. A promising subset is not completion.

## 5. Run Order and Milestones

| Milestone | Goal | Runs | Decision gate | Cost | Risk / mitigation |
|---|---|---|---|---|---|
| **A0.0 MUST** | Prove SSR OOF assets are recipe/hash suitable | `EDCM-A0-REUSE-AUDIT-v1` | Every source/output/fold/vote/provenance check passes | CPU SLURM, <1 hour expected | Old manifests lack full code provenance; supplement with frozen code hashes and independent output reproduction |
| **A0.1 MUST** | Measure frozen-geometry support/reachability | `EDCM-A0-REACH-MHC-v1`, `EDCM-A0-REACH-MHC_zh-v1` | Per-dataset support, reachable count, accuracy gain, and macro-F1 gain all pass | 2 CPU jobs, may run in parallel | Ambiguous swaps would invalidate preregistration; exact exhaustive semantics are frozen above |
| **A0.2 MUST** | Jointly lock or stop teacher spend | `EDCM-A0-DECISION-v1` | Both datasets pass; zero prior MLLM calls | CPU SLURM | Partial pass cannot unlock A1 |
| **A1 CONDITIONAL** | Establish reliable teacher-specific gradient and beat matched proxy | seven A1 IDs above | Every A1 gate passes on both datasets | Estimated 10--30 GPU-hours | Reliable but gradient-inert or proxy-explained teacher -> STOP |
| **A2 CONDITIONAL** | Causal seed-0 memory repair | 14 arms + decision | Full beats four controls and corruption is monotone | Part of estimated 30--60 GPU-hours | Head/memory redistribution -> STOP |
| **A3 CONDITIONAL** | Prove final objective | frozen namespace above | Entire `+3/+3`, statistics, removability, corruption contract | Part of estimated 30--60 GPU-hours | Two-dataset/three-seed endpoint may still fail |

## 6. Compute and Data Budget

- **A0 authorized design:** three CPU-stage submissions (audit, two parallel reachability jobs, decision); no GPU and no MLLM.
- **A1 conditional:** about 4 deterministic teacher calls per train video, roughly 4,512 calls across 549 EN + 579 ZH videos; estimated 10--30 A100 GPU-hours after smoke.
- **A2/A3 conditional:** estimated 30--60 A100 GPU-hours with baseline/cache reuse; one 16-CPU/120-GB training job at a time to remain under the user cap.
- **New human/gold annotation:** none. No segment annotation is requested, inferred, fabricated, or evaluated.
- **Biggest bottleneck:** teacher signatures may be reliable yet induce too little sample-specific gradient beyond Label-only and the strength-matched low-level proxy.

## 7. Risks and Mitigations

- **Reuse drift:** Stored SSR artifacts may no longer match source/config/data. **Mitigation:** A0 reuse audit is a hard predecessor and stores every hash.
- **Gold-language drift:** Parent-label-inherited subclips may be misdescribed as segment gold. **Mitigation:** A0 consumes only full-video endpoints; every JSON repeats `segment_gold_exists=false` and `segment_gold_used=false`; documentation calls inherited labels parent-video labels only.
- **A0 overclaim:** Reachability may be mistaken for an EDCM upper bound or MLLM evidence. **Mitigation:** decision schema and paper wording call it a fixed-geometry cost screen only.
- **Leakage:** Teacher/proxy statistics may reach dev/test. **Mitigation:** train-ID allowlists, fold-local fits, manifest ID hashes, and evaluators that reject teacher arguments.
- **Decorative MLLM:** A low-level proxy, Label-only, shuffle, or noise may match full. **Mitigation:** binding A1/A2 STOP gates; no tuning rescue.
- **Head/memory redistribution:** Native head may improve while final kNN does not. **Mitigation:** all promotion gates use actual kNN metrics/topology; head is diagnostic only.
- **Multiple testing/story bloat:** Many controls may become post-hoc. **Mitigation:** fixed two primary claims, fixed arms, exact run namespace, and Holm-corrected final family.

## 8. Final Checklist

- [x] Primary and supporting claims are frozen and linked to evidence.
- [x] A0 exact top-64 / <=2-swap / accuracy+macro-F1 gates are frozen.
- [x] A0 precedes every MLLM call and teacher artifact.
- [x] SSR OOF reuse is conditional on recipe, manifest, output, fold, vote, and hash verification.
- [x] Only video-level binary gold is permitted; no segment gold is assumed.
- [x] Validation/test teacher-artifact prohibition is explicit.
- [x] Exact run IDs/namespaces, script interfaces, SLURM resources, order, outputs, and STOP/GO gates are specified.
- [x] A1--A3 are conditional skeletons; current execution authority remains A0 only.
- [ ] A0 implementation exists and passes static review.
- [ ] A0 SLURM jobs complete and `A0_DECISION.json` verifies.
- [ ] A1 is unlocked (only possible after A0 joint GO).
- [ ] Final two-dataset x three-seed `+3/+3` target is proven.
