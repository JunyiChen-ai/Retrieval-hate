# C01 A0 — Policy-Contrastive Endpoint Audit

**Current status:** `V2_READY_NOT_RUN_PENDING_REVIEW`; next action is a fresh independent static review of the v2 contract. No v2 SLURM job has been submitted and no v2 result/decision artifact exists. The v1 job and empty namespace remain immutable historical evidence. This is a **pre-Stage0 kill-only diagnostic**: survival permits only the next same-pooling preparation step and does not advance Gate 0.

## Prospective preregistration

### Claim and interpretation boundary

C01 A0 tests one narrow claim: a **block-normalized paired-endpoint contrast** may retain label-relevant structure beyond either endpoint and beyond equally normalized paired-endpoint controls. The endpoints are existing Qwen2.5-VL L24 caches:

- standard policy: baseline prompt with image prefix-mean / text response-mean pooling;
- one-word policy: one-word prompt with last-token pooling.

Prompt and pooling change together. A0 therefore identifies only a **readout-policy endpoint contrast**, not a prompt-only effect. Without per-block normalization, `[standard, oneword]` and `[standard+oneword, oneword-standard]` are orthogonal reparameterizations containing identical information; A0 can measure only the effect of the frozen block normalization/equalization. A positive A0 authorizes only extraction of new **same-pooling neutral/policy caches** and a later capacity-matched fold-head pilot. It cannot establish safety, stance, discourse disentanglement, or an end-to-end gain. A negative A0 kills only this existing-endpoint route, not same-pooling policy contrast in general.

### Inputs and hard access guards

The only readable inputs are the exact `train` and `dev_seen` standard/one-word L24 caches for `MHC_zh` and `HateMM`. Existing provenance contains only exact byte sizes plus historical SHA256 **16-hex prefixes** in `refine-logs/READOUT_SCREEN_OUT.json`; no prior full 64-hex cache provenance was found, so these are explicitly called `size+sha16 provenance guards`, not full hashes.

Before A0 may load any cache, `scripts/slurm/c01_hash_inputs.sbatch` must run the read-only `scripts/analysis/c01_hash_inputs.py` preflight. It hashes exactly the eight allowlisted train/dev files, opens no test path, checks size+sha16 provenance, and exclusive-creates `artifacts/c01_policy_contrastive/v1/hash_preflight/C01-HASH-v1/full_sha256_manifest.json`. The only approved manifest is exact SHA256 `083275d39a1026bde3b6583bd5608d41cec5b431da9ffda87ae8ab1046cf2305`; the consumer hashes and exact-compares the whole manifest before parsing it or opening any cache. A0 then requires the manifest's frozen schema/run/source-set, eight exact file identities, clean access ledger and 64-hex values. The consumer freezes the nine-key ledger-entry schema and zip-compares every entry in ordinal order with the same `expected_manifest_records`: ordinal/path/dataset/split/policy must match exactly, `test_like=false`, `open_attempted=true`, `opened=true`, and `bytes_read` must equal the registered byte count. It also requires a non-empty positive-integer Slurm job ID and the exact independent hash-job environment `CUDA_VISIBLE_DEVICES=""` with `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`. A0 rehashes every cache and requires exact equality with its manifest 64-hex SHA256—not merely the historical sha16 prefix—**before** `torch.load`.

The runtime loader additionally requires the exact `{ids,img_feats,text_feats,labels}` contract; `[N,3584]` finite image/text tensors; finite binary labels; exact standard/one-word ID, order, and label agreement; unique IDs; expected row counts; and disjoint train/dev IDs. It fails closed on any test-like split or input path. Test paths/splits, L28 caches, other datasets, external pools, and cross-dataset memory are forbidden. Runtime access counts are derived from the actual ledger rather than asserted as a literal zero.

Historical strict train-memory→dev-query endpoint accuracy must reproduce `refine-logs/READOUT_SCREEN_OUT.json` within `1e-12`: MHC-ZH standard/one-word `0.8589743589743589 / 0.8589743589743589`; HateMM `0.8411214953271028 / 0.8411214953271028`. The deployed L28 R0 accuracies (`0.8589743589743589`, `0.8504672897196262`) are context-only comparison floors and are not reloaded.

### Frozen transforms and controls

Each endpoint and modality is row-L2-normalized. With `epsilon=1e-12`, any norm at/below epsilon aborts the run. Per modality:

- `common = L2(standard + oneword)`;
- `displacement = L2(oneword - standard)`;
- `common_interaction = L2(common * displacement)`.

Every two-block modality representation individually normalizes both blocks, concatenates and normalizes the modality, then concatenates normalized image/text modalities and normalizes again. Frozen arms are `endpoint_std`, `endpoint_ow`, score-level `avg_score`, equally normalized `endpoint_concat`, `common`, `displacement`, primary `common_displacement`, secondary `common_interaction`, and the shuffled-pair null.

The strong algebra control applies the same block-L2 rule after orthogonal endpoint rotations
`u=cos(theta) standard+sin(theta) oneword`,
`v=-sin(theta) standard+cos(theta) oneword`.
The ex-ante angle set is `{8.3,17.6,29.1,60.4,72.7,83.8}` degrees; 45 degrees is excluded because it is the primary transform. No angle may be selected after seeing dev results. The maximum across the complete frozen set is the rotation-distribution upper bound, and every paired comparison is multiplicity-corrected.

### Retrieval and statistics

All arms use only strict train-memory→dev-query retrieval: rank-weighted signed-cosine top-20 with descending integer rank weights and fixed score cutoff zero. No LOO arm, threshold tuning, top-k selection, routing, fitting, or dev-memory use is allowed. Each arm reports accuracy, macro-F1, ROC-AUC, confusion counts, and fixed/broken/net items.

Paired bootstrap uses 2,000 dev resamples and reports observed delta, mean, 5th/95th percentiles, one-sided p, and Holm correction. The primary is compared with all declared ordinary controls and every frozen rotation; the secondary `common_interaction` is descriptive and separately corrected. The shuffled-pair null uses 256 independent deterministic SHA256 ID orderings in train and dev, with no labels entering the pairing; both real `common_displacement` and real `displacement` are compared with their respective null distributions and jointly Holm-corrected.

Before normalizing displacement, the run reports `||L2(oneword)-L2(standard)||` distributions for image/text and both splits, the fraction at/below the frozen tiny threshold `1e-3`, and a small-displacement gain-concentration audit. “Small” uses the train-only 10th percentile of `min(image_norm,text_norm)` and is applied unchanged to dev. The binding reference is the same strongest ordinary control used by the net-fix decision: maximum accuracy, then macro-F1, then the earliest entry in the frozen `gain_controls` order. More than half of primary fixes versus that selected control arising in the small subset is flagged as dominated. The corresponding calculation versus `endpoint_concat` is retained only as a mechanism diagnostic and never carries the gate. The selector reads only frozen ordinary-control metrics, excludes `common_displacement`, and does not read the small-displacement outcome, so no baseline/gate circular dependency is introduced. Tiny fraction above 5% or an epsilon violation fails closed.

### Decision

The code contains one canonical binding object and requires exact equality for run identity, namespace, datasets/splits, complete arms and controls, primary/secondary, metrics, transforms, thresholds, `+0.020`, net-fix counts, bootstrap/permutation seed and counts, Holm alpha, fixed angles, preflight manifest, output paths/budget, all required-true flags, and decision schema. The decision artifact records each dataset's selected small-displacement gate reference, the frozen selection rule, and the `endpoint_concat=diagnostic_only` role. Every aggregate check rejects an empty family before calling `all`.

`CONTINUE_SAME_POOLING_CACHE_ONLY` requires the same primary arm to pass on **both** datasets:

1. accuracy and macro-F1 each gain at least `+0.020` over the strongest frozen ordinary control (accuracy also clears the historical deployed-R0 context);
2. every registered primary/control paired-bootstrap lower bound is positive with Holm rejection;
3. primary metrics exceed the complete fixed rotation upper bound, with positive corrected paired-bootstrap comparisons against every angle;
4. real primary and real displacement accuracy/macro-F1 each exceed shuffled-pair p95 with Holm rejection;
5. net fixes versus the strongest ordinary control are at least `+2` MHC-ZH and `+3` HateMM;
6. cache/history/access/algebra/tiny-displacement guards pass and fixes are not dominated by the train-defined small-displacement subset.

Any failed binding condition yields `KILL_CURRENT_ENDPOINT_ROUTE_ONLY`. LOO-only gains, a single-dataset result, one-item movement, secondary-only gains, an angle chosen after observation, threshold tuning, or a pooling-confounded safety claim cannot survive. Even `CONTINUE` is not a Stage0 pass or Gate0→Gate1 promotion.

### Execution and publication contract

Execution order is frozen: first `sbatch scripts/slurm/c01_hash_inputs.sbatch`; only after its immutable manifest exists may `sbatch scripts/slurm/c01_a0_cpu.sbatch` be considered for a runtime smoke. A0 uses conda `HateVideo`, CPU-only, 8 CPU / 32 GB, fixed `OMP/MKL/OPENBLAS/NUMEXPR=8`, `CUDA_VISIBLE_DEVICES=""`, no `--time`, and no `disk_guard`. Logs are `slurm/logs/c01_a0_%j.{out,err}`. Both scripts refuse non-SLURM execution.

Results publish under `artifacts/c01_policy_contrastive/v1/a0/C01-A0-v1/`. The CLI has no force option. The entire run namespace is exclusive/no-clobber; any pre-existing namespace—including a result without a decision—fails closed and requires a new run ID. Result and frozen-schema decision JSON are fsync+atomic exclusive-create. The job does not edit or append `TARGET_LOOP.md`, `TARGET_STATE.json`, `TARGET_FINDINGS.md`, or this record.

## Implementation card

- Config: `configs/c01/c01_a0_v1.json`
- Analysis: `scripts/analysis/c01_policy_contrast_a0.py`
- SLURM wrapper: `scripts/slurm/c01_a0_cpu.sbatch`
- Hash preflight: `scripts/analysis/c01_hash_inputs.py`, `scripts/slurm/c01_hash_inputs.sbatch`
- State: **READY_NOT_RUN_PENDING_REVIEW_CONFIRMATION; pre-Stage0 kill-only**
- Next: **fresh_external_static_review_then_hash_preflight_then_runtime_smoke**
- Jobs/results/new metrics: **none**

## V2 structural-null contract amendment (2026-07-28)

### Historical lineage retained

The preceding preregistration is the v1 record. It is not rewritten into a successful run. Job `13712` failed closed on `HateMM/train/standard/img` row 355 before any metric or decision was published, and `artifacts/c01_policy_contrastive/v1/a0/C01-A0-v1/` remains an empty, non-reusable namespace. `configs/c01/c01_a0_v1.json` and `scripts/slurm/c01_a0_cpu.sbatch` remain historical v1 entry points; the canonical analysis now accepts only v2.

### Exact evidence and sole authorization

V2 exact-binds:

- the eight-cache full-hash manifest SHA256 `083275d39a1026bde3b6583bd5608d41cec5b431da9ffda87ae8ab1046cf2305`;
- zero-probe artifact `artifacts/c01_policy_contrastive/v2/zero_contract_probe/C01-ZERO-PROBE-v1/zero_contract_probe.json`, SHA256 `bee4964ce7e4ca81cfdb72c3859f78196568badf982aef587bc14ee6dbe63526`;
- the sole structural-null tuple `HateMM / train / hate_video_95 / row 355 / expected-label-integrity-only 1`, aligned under `standard` and `oneword` for both `img` and `text`.

Only that tuple may be numeric exact-zero in every component at input and after normalization. Every raw and derived modality block, fused arm, algebra control, and frozen orthogonal rotation must preserve exactly the registered row mask. This exact-zero mask contract is distinct from the byte-level with/remove comparison below. Any other exact-zero row, any positive norm at or below `1e-12`, non-finite row, endpoint/modality mask mismatch, identity drift, or new derived zero halts before publication.

The expected label is checked only as immutable row-integrity evidence. It never selects, creates, normalizes, shuffles, retrieves, or scores the exception.

### Mandatory HALT-only validity guards

All 256 label-blind ID-hash shuffles keep the registered train null at index 355 and form a bijection over the remaining indices; labels remain absent from permutation construction. For every ordinary/scientific arm and every frozen rotation, the registered null must be absent from every dev top-20 FAISS neighbor list. The implementation then removes the registered null, rebuilds retrieval, maps neighbor indices back to the original memory, and requires identical dtype, shape, and C-order bytes for neighbor indices, similarities, scores, and predictions. Each side's byte SHA256 is persisted. Metrics use canonical sorted typed serialization whose float values are IEEE-754 binary64 big-endian hex; canonical payload SHA256 must match. `+0.0` and `-0.0` are distinct, and NaN/Inf is forbidden. The same checks apply to both shuffled scientific arms on every draw.

These guards can only abort with HALT. They cannot contribute evidence to `CONTINUE_SAME_POOLING_CACHE_ONLY`, relax a scientific gate, or turn an invalid run into `KILL_CURRENT_ENDPOINT_ROUTE_ONLY`.

### Scientific contract unchanged

V2 retains the v1 strict train-memory→dev-query R0 parity contract, complete rotation family, 256 shuffle draws, 2,000 paired bootstraps, Holm families, small/tiny-displacement checks, minimum `+0.020` accuracy and macro-F1 gain, and minimum net fixes `+2` MHC-ZH / `+3` HateMM. It preserves the same claim and interpretation boundary. The structural-null amendment strengthens validity only.

### V2 publication and review state

- Run/config: `C01-A0-v2`, `configs/c01/c01_a0_v2.json`
- Analysis: `scripts/analysis/c01_policy_contrast_a0.py` (v2 canonical only)
- SLURM wrapper: `scripts/slurm/c01_a0_cpu_v2.sbatch`
- Exclusive namespace: `artifacts/c01_policy_contrastive/v2/a0/C01-A0-v2/`
- Schemas: `c01_a0_result_v2`, `c01_a0_decision_v2`
- State: **V2_READY_NOT_RUN_PENDING_REVIEW**
- Execution authorization: **none**
- Python / SLURM / result / decision / new metric from this amendment: **none**

### V2 review repair: displacement null exclusion and exact-comparison semantics

The registered structural null is excluded from every scientific displacement reduction. In particular, the HateMM train joint displacement quantile, per-modality tiny counts/fractions and distributions use 743 ordinary rows, not the 744-row array containing the structural null. The implementation independently recomputes the complete displacement gate in two ways:

1. retain the source array and mask the registered null before every reduction;
2. physically delete the registered null before every reduction.

The two routes must produce byte-identical IEEE-754 binary64 thresholds and tiny fractions, an identical dtype/shape/C-order-byte dev small-row mask, identical tiny counts/denominators, the same `small_rows_dominate_fixes` boolean, and the same final displacement gate boolean. Hashes/encodings are reported, and any discrepancy HALTs before bootstrap, Holm, net-fix, or decision publication.

Retrieval with/remove equivalence already makes all ordinary/scientific/rotation/shuffle scores, predictions and metrics independent of the registered null. Combined with the displacement dual-path guard, the null cannot affect any bootstrap/Holm input, net-fix count, train-derived quantile, or final decision. These remain validity guards only and cannot supply positive scientific evidence.
