# CTE C0/C1 Independent Code Review

**Review date:** 2026-07-11  
**Scope:** `configs/cte/cte_v1.json`, `scripts/analysis/cte_{common,c0,c1}.py`, `scripts/slurm/cte_c{0,1}_{cpu,gpu}.sbatch`  
**Authority:** `refine-logs/cte/{EXPERIMENT_PLAN.md,FINAL_PROPOSAL.md}` and repository `AGENTS.md`  
**Mode:** static/read-only review; no experiment was submitted or executed; this report is the only file written.

## Executive verdict

**NOT READY FOR CONFIG_FREEZE / C0 execution.** No CRITICAL defect was found, but four HIGH defects remain. The current code correctly enforces the most important supervision boundary: the only gold consumed is the parent-video binary label; K4 labels are audited as mechanical inheritance and are never represented as segment gold. C0/C1 contain no MLLM, OCR, or teacher-cache path and do not open validation/test content. SLURM scripts use the frozen resources, activate `HateVideo`, request at most one GPU, and do not set `--time`.

The repository endpoint was separately checked: `cte_common.ordinary_knn` uses rank weight × cosine similarity × signed label, which agrees with `scripts/analysis/ssr_common.py::exact_vote` and with the authoritative SSR comparator's explicit `compute_metrics_retrieval(..., majority_voting="arithmetic", use_sim=True)` parity check. It is therefore compliant with the frozen repository arithmetic-cosine endpoint and is not a finding.

## CRITICAL

None.

## HIGH

### H1 — C0 does not audit the frozen numerical kernel over all required radii, and its gradient check stops at the margin

- **Location:** `scripts/analysis/cte_c0.py:112-140`, `scripts/analysis/cte_c0.py:143-196`
- **Contract:** EXPERIMENT_PLAN §B0 requires actual-bank scalar/vectorized margin, tangent and cost parity, an autograd/directional finite-difference check, and coverage of all four radii, both modalities and all five fold memories.
- **Observed:** `actual_numerics` builds prototype paths only at `selected_pair["a1"]` and therefore checks two modality paths at one radius, not all `0.05/0.10/0.20/0.30` radii. The only directional finite-difference objective is `vectorized_margin(...).mean()`; it does not exercise `tangent_values`, MAD scaling, `tanh`, support masking or `interval_cost`. The shifted-logit check is likewise a standalone double-precision LSE difference rather than the masked FP32 full kernel.
- **Impact:** C0 can report GO even if a non-selected frozen radius or the actual tangent/cost autograd path is numerically wrong. That invalidates C0's role as the binding feasibility/numerics gate before the expensive C1 grid.
- **Minimum fix:** make actual-bank cases iterate every fold × modality × configured radius; compare margin, T and interval cost against a double scalar reference for each case; add finite/directional-FD checks for the complete supported CTE loss (including mask and MAD handling), and have `phase_decide` require complete cardinality and every case to pass.

### H2 — matched-control strength uses one minibatch, not the clean training-fold aggregate first-step gradient norm

- **Location:** `scripts/analysis/cte_c1.py:563-585`
- **Contract:** EXPERIMENT_PLAN §B1 and FINAL_PROPOSAL “Assignment Controls” freeze a single strength scalar based only on the clean training fold's **aggregate first-step gradient norm** so MULTIVIEW/RANDOM are capacity-matched without per-video information.
- **Observed:** the reference and each control norm are computed from `next(iter(make_loader(...)))`, i.e. one shuffled batch of at most 64 videos. The remaining training-fold videos do not contribute. Which videos land in that batch can materially change the scalar, especially when support/target activity is heterogeneous.
- **Impact:** MULTIVIEW and RANDOM are not the specified matched controls. A LABEL_ONLY win could be caused by an accidentally weak control scalar, so the anti-claim “not generic multiview/random extra optimization” would not be supported.
- **Minimum fix:** at the shared initialization and frozen context, deterministically accumulate each arm's CTE gradient over the entire clean training fold with the same per-video normalization used by training, then take the norm of that aggregate gradient and derive exactly one scalar per control. Persist the covered-ID hash, active counts, raw aggregate norms and resulting scalars; verify them independently.

### H3 — conditional full-train freeze publishes pre-CTE anchors/radii as the post-C1 teacher-before-call identity

- **Location:** `scripts/analysis/cte_c1.py:1193-1219`
- **Contract:** EXPERIMENT_PLAN §B1 “Conditional full-train freeze after GO” requires training the seed-0 LABEL_ONLY full-train checkpoint and then selecting/freezing the **post-C1** `anchor_id^V/L` and adjacent radii before any teacher call.
- **Observed:** anchors/support are selected on `initialization` before LABEL_ONLY training (`1194-1199`). After training, `anchors.json` copies `package["fixed"]` (`1213-1218`), which is the same pre-training identity used inside the C1 run. The final checkpoint is never re-encoded for a fresh post-C1 medoid/adjacent-radius selection.
- **Impact:** `C1_FREEZE_VERIFY` may authorize C2 planning with an anchor/radius identity that was not selected on the published post-C1 checkpoint. This breaks the teacher-before-call freeze contract and can make later transfer/support claims refer to the wrong tangent.
- **Minimum fix:** after the final LABEL_ONLY state is produced, rebuild a model from the exact saved state, encode the full train bank, independently select the two modality medoids and largest passing adjacent pair on that post-C1 geometry, and publish/hash those values as the C2 identity. Keep the pre-training training-path identity separately named for C1 diagnostics. `verify-freeze` must recompute the post-C1 selection from the checkpoint and authoritative train cache and require exact equality.

### H4 — selection/decision verification is not fail-closed on provenance and frozen ID partitions

- **Location:** `scripts/analysis/cte_common.py:728-741`; `scripts/analysis/cte_c1.py:765-778`, `scripts/analysis/cte_c1.py:787-831`, `scripts/analysis/cte_c1.py:1039-1076`
- **Contract:** EXPERIMENT_PLAN §6 requires decision/verify phases to independently recompute from ledgers and authoritative inputs, and to STOP on any config/implementation/hash/row-count/ID-partition mismatch.
- **Observed:** `validate_manifest_common` checks payload hash, run/stage, zero-call fields and listed file bytes, but does not compare `config_canonical_sha256`, `implementation_sha256`, `fold_ids_sha256`, checkpoint provenance, required status, or required-file membership against the current frozen authority. `phase_select` reads `inner_splits.json` only as a hashed file and never checks that each grid row's IDs/labels equal that row's frozen `J_ids`, that its probe targets are exactly the corresponding `P_ids`, or that the three `inner_fold` values form `{0,1,2}`. `phase_decide` checks only the five-fold aggregate query-ID union; it does not require each fold ledger to equal that fold's authoritative held-out IDs or its manifest's train/query hashes.
- **Impact:** a stale, permuted, incomplete, or semantically mismatched producer ledger can retain valid self-consistent metrics and pass selection/GO despite violating the nested train-OOF boundary. File hashing alone proves byte stability, not that the bytes satisfy the frozen split/provenance contract.
- **Minimum fix:** strengthen common validation with exact current config/implementation/fold/checkpoint/status and required-artifact checks. In selection, reconstruct authoritative outer-train and `P/J` partitions, require exact ID/label equality per inner fold and tuple, verify target coverage/namespace and unique grid cardinality. In decision, require each outer directory's queries to equal only that fold's held IDs, its train hash to equal the complementary IDs, all prediction/neighbor arms to share that exact ordered ledger, and anchors/diagnostics payloads to pass their own hashes and cardinality/gate recomputation.

## MEDIUM

### M1 — optimizer settings are not frozen as exact numeric parameters

- **Location:** `configs/cte/cte_v1.json:28-30`; `scripts/analysis/cte_c1.py:99`, `scripts/analysis/cte_c1.py:430`
- **Impact:** the config records `AdamW_repository_default_weight_decay`, while code passes only `lr` and inherits library defaults for weight decay, betas, epsilon and AMSGrad. This weakens exact reproducibility and the manifest's “exact params” claim.
- **Minimum fix:** freeze all AdamW numeric arguments in config, pass each explicitly in baseline and arm training, and record them in training manifests.

### M2 — the binding C1 GPU-hour estimate is not derived from the implemented workload

- **Location:** `scripts/analysis/cte_c0.py:239-248`
- **Impact:** the estimate multiplies one auxiliary-kernel median by a hard-coded `2 × steps × epochs × 9 arms × 3`, while the implemented inner workload is 8 tuples × 2 arms × 3 splits per outer fold and also pays full-bank refresh, support audits, base loss and endpoint encoding. The value cannot safely replace the plan's provisional 20–40 GPU-hour estimate.
- **Minimum fix:** measure a representative end-to-end epoch including refresh/base/aux/endpoint costs, or derive the estimate from the exact scheduled tuple/arm/split/fold counts and report components plus assumptions.

### M3 — no-clobber publication does not fsync the containing directory

- **Location:** `scripts/analysis/cte_common.py:123-180`
- **Impact:** file contents are fsynced and publication uses an exclusive hard link, but the directory entry is not fsynced after link/unlink. A host/filesystem crash can leave a persistent namespace lock without its supposedly published artifact.
- **Minimum fix:** after exclusive publication and temporary unlink, open and fsync the parent directory; handle publication failures without weakening no-clobber semantics.

## LOW

### L1 — SLURM wrappers do not uniformly reject surplus positional arguments

- **Location:** `scripts/slurm/cte_c0_gpu.sbatch:16-22`, `scripts/slurm/cte_c1_gpu.sbatch:19-26`, `scripts/slurm/cte_c0_cpu.sbatch:15-31`, `scripts/slurm/cte_c1_cpu.sbatch:15-20`
- **Impact:** some phases validate required arguments but ignore extras, making an operator typo harder to detect and weakening the frozen positional interface.
- **Minimum fix:** assert exact `$#` for every case before launching Python.

## Confirmed compliant items

- **Gold boundary:** only parent-video binary labels are accepted. `segment_gold_exists=false`; K4 labels are checked against parent labels and recorded as `inherited_parent_video_label_not_segment_gold`. No segment/timestamp/span/localization/stance/target/mechanism/rationale gold is assumed.
- **Zero-resource boundary:** C0/C1 have no MLLM/OCR/teacher-cache call path; config and manifests freeze all corresponding counters at zero.
- **Train-only boundary:** only frozen train caches and train fold records are loaded; val/test content is not opened. Producer-side nested probe construction keeps each C row out of both probe fitting and C selection, and outer-held videos are used only as OOF endpoints.
- **Geometry:** separate V/L spherical medoid IDs, adjacent radii, and joint projected-pair plus fused-space support are implemented. Anchors are re-encoded rather than replaced at refresh.
- **Loss/bank:** tangent is bounded by `tanh`; interval cost is bounded; the full detached epoch bank excludes self and requires both classes. Current code fails closed on half-epoch interval drift and post-first-epoch drift.
- **Endpoint:** exact top-20 rank-weighted arithmetic-cosine vote matches the authoritative SSR/repository `use_sim=True` endpoint; no relation, teacher, rerank or score fusion is used.
- **SLURM:** requested CPU/memory/GPU resources match the plan, `HateVideo` is activated, and no script sets `--time` or releases held jobs.
- **No-clobber/canonical basics:** persistent `O_CREAT|O_EXCL` locks, sorted compact UTF-8 JSON with NaN/Inf forbidden, payload hashes, file hashes, temp-file fsync and exclusive publication are present, subject to H4/M3.

## Required disposition

Do not create `CONFIG_FREEZE.json`, submit C0, or unlock C1 until H1–H4 are fixed and this review is repeated against the final implementation hashes. C2–C4 and all teacher calls remain locked regardless of C0/C1 code status.

CRITICAL: 0
HIGH: 4

---

## Re-review — 2026-07-11

This section supersedes the initial execution disposition above while retaining it as an audit trail. The re-review was performed against the revised implementation with these reviewed SHA-256 values:

- `configs/cte/cte_v1.json`: `e21266520550edf7d624c67bf695567990591da2333ef12f1a3b30cf1584e274`
- `scripts/analysis/cte_common.py`: `d14ab9b9f938928c38f4a22fbcb41bd829c6483432f827881b07b7f175639692`
- `scripts/analysis/cte_c0.py`: `a0cbb96e4fa88a8623ac954a7174578c4632d47ade4c68cd37d084282130ca0b`
- `scripts/analysis/cte_c1.py`: `da17b43f718531987d35551db73f41bcf8e784a06419a3265bb8840c69fe1bfe`
- SLURM wrappers: `cte_c0_cpu=628832dbc6f934ef207279c56d0487d8a50f6a79f5fdd4bbfe14061925e590f7`, `cte_c0_gpu=ebd9ee18e577b706f93b22cb187e115ee22fc210c11b186ee4f3435eb0c6c677`, `cte_c1_cpu=e741369c668551af1bebec47540e5c7095b0e27ec506ef63841057571e69ac56`, `cte_c1_gpu=53ff3b65a9a999e8693076f509541459756a4169c7e6e9cd84a8baa9db0d88fd`.

### Closure of original HIGH findings

- **H1 — CLOSED.** Actual-bank numerics now cover exactly five distinct outer folds and the complete `2 tau × 2 modalities × 4 radii × 2 sMin = 32` case product per fold. Each case records margin/T/cost errors and a complete T→interval-cost directional derivative. C0 decision reconstructs the expected keys, rejects missing/duplicate cases, checks synthetic tau coverage, and recomputes maxima/finite gates directly from case rows rather than trusting producer summaries. Support evidence is likewise required to contain five folds and all eight modality/radius cells.
- **H2 — CLOSED.** Strength matching now accumulates the auxiliary objective over the entire canonical clean training fold before taking the gradient norm. It records exact covered-video count/hash, active modality rows, raw aggregate norms and scalars for LABEL_ONLY/MULTIVIEW/RANDOM. C1 decision requires full authoritative outer-train coverage and independently checks `control_strength = label_norm / control_norm`.
- **H3 — CLOSED.** The teacher-before-call V/L medoids and adjacent radii are selected only after loading the final C1 LABEL_ONLY state. `verify-freeze` reloads the saved post-C1 checkpoint with the authoritative full-train cache on CPU, independently recomputes both medoids and the supported adjacent pair, and requires exact equality with the published freeze artifact.
- **H4 — CLOSED.** Common manifest validation now requires `COMPLETED`, current config/implementation hashes, expected fold/checkpoint provenance, exact required output membership, required inputs, unique file records and file hashes. Selection reconstructs authoritative P/J partitions and validates target/query ID-label coverage and the exact 3×8 grid. Outer decision validates each fold's train/held-out partition, video labels, memory-only unique neighbors, self exclusion, exact arithmetic-cosine vote, anchor/radius consistency, target coverage, control matching, diagnostics cardinality and fail-closed drift records.

### New CRITICAL/HIGH audit

No new CRITICAL or HIGH defect was found in the revised paths. The unique gold remains the parent-video binary label; no segment/timestamp/span gold is assumed. C0/C1 remain train-only and zero-call for MLLM/OCR/teacher cache. SLURM resources and the no-`--time` rule remain compliant.

### Final disposition

The static M0 code-review gate is clear for the exact hashes above. This does not constitute C0 empirical GO: CONFIG_FREEZE and all C0/C1 runs must still follow the frozen SLURM order and runtime gates. C2–C4 and all teacher calls remain locked.

CRITICAL: 0
HIGH: 0
