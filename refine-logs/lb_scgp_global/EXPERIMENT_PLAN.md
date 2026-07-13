# Experiment Plan: LB-SCGP Global

**Problem**: hateful video detection on MHC-EN (`MHC`) and MHC-ZH (`MHC_zh`) with a meaningful train-only label-blind MLLM role, while final inference remains ordinary full-video train-memory top20 kNN.
**Method thesis**: a sealed train-only label-blind MLLM structural cache defines one certifiable encoder-realizable PSD/unit-diagonal global target; the existing encoder fits it uniformly, and success is accepted only if final paired ordinary kNN beats the frozen moving strongest same-protocol non-MLLM comparator and attribution controls.
**Date**: 2026-07-12
**Revision**: one formal revision round after `EXPERIMENT_PLAN_REVIEW.md`, plus Run2-v2 amendment and two static implementation repair/freeze passes after fresh `M0_RUN2_RESULT_TO_CLAIM_REVIEW_FRESH.md`.
**Evidence status**: Run1 is frozen; Run2-v1 is immutable FAIL_STOP infrastructure evidence only; Run2-v2 amendment review passed 0C/0H, the first implementation fix/freeze was reviewed and failed 0C/2H/1M/1L, and Run2-v2 fix2 is locked for a fresh 0C/0H implementation/code review, exact hashes/no-clobber review, and separate execution authorization. Planning and static repair readiness are not success.

## Immutable Contract

- Claims: exactly two primary claims are frozen, C1 and C2.
- Core blocks: exactly five core blocks, B0-B4. Milestones M0-M5 map onto these blocks.
- Baseline families: exactly three families. Arms inside a family do not create extra families.
- Datasets and seeds: `MHC`/MHC-EN and `MHC_zh`/MHC-ZH, seeds `0,1,2`.
- Only gold supervision: `parent_video_binary_label`.
- `segment_gold_exists=false`, `segment_gold_used=false`; no segment, frame, timestamp, span, localization, stance, target, mechanism, rationale, or fragment gold exists or is used.
- Uniform frames and full transcripts may be cache inputs. They are evidence inputs, not annotations.
- Train-only label-blind MLLM cache. Cache sees train videos only and no labels, predictions, margins, losses, held content, validation content, test content, teacher artifacts, heads, rerankers, keys, neighbors, memory IDs, compiler targets, or target banks.
- Labels first enter only after cache seal, inside compiler/evaluator code. Validation/test labels are evaluator-only.
- Validation/test inference must have zero cache reads, certificate reads, compiler-target reads, auxiliary-head loads, reranker loads, key-selector loads, and teacher-artifact reads.
- Final endpoint: ordinary full-video train-memory top20 kNN; no MLLM/cache/teacher/head/rerank/key selection.
- No sample weighting, key selection, pair/triplet/SupCon objective, segment route, teacher verdict/rationale route, local rank-cell route, or local v8.
- Retired local rank-cell evidence, including local v7, cannot be reused as global PASS evidence.
- All future compute must use SLURM with `conda activate HateVideo`, no `--time`, at most 16 CPU / 128 GB / 2 GPU. `PENDING (JobHeldUser)` waits for automatic release.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: executable, isolated, certifiable, encoder-realizable global geometry | The contribution fails if the cache-to-target map is not closed, verifiable, resource-feasible, or isolated | M0 G0 gates pass: implementation contract freeze, synthetic closed-convex projection with serialized H-metric KKT-only payload, independent verification, actual train-bank resource/replay decision, common-basis operator, closed strong convexity, PSD/unit diagonal, coordinate/row/class trust, rank-tail `<= d`, nondegeneration, REMOVE/null parity, cache/isolation injection failures, robust interval coverage report, and no local-v7 PASS reuse | B0 |
| C2: final ordinary-kNN performance and attribution | The method must improve the real endpoint and show the MLLM geometry is not replaceable by simpler routes | On both datasets and seeds 0/1/2, FULL beats the frozen moving strongest same-protocol non-MLLM comparator by `>= +0.030` accuracy and `>= +0.030` macro-F1, every paired seed delta is positive, hierarchical paired bootstrap one-sided lower bound is `> 0`, Holm passes over four dataset-by-metric tests, and decisive controls support attribution | B1-B4 |

Anti-claim to rule out: the gain is only stronger training, REMOVE replay drift, corrupted/identity cache artifacts, direct structural-moment use, direct certificate-feature distillation, scalar difficulty/error propensity, sample weighting, pair mining, or a hidden teacher/reranker.

## Baseline Families

| Family | Arms | Role |
|---|---|---|
| F1 moving strongest same-protocol comparator / REMOVE | frozen moving strongest same-protocol non-MLLM comparator; paired REMOVE replay | Final performance comparator and removability anchor |
| F2 certificate identity/corruption | SHUFFLE; covariance-matched NOISE | Tests identity and corruption sensitivity without treating certificate atoms as gold |
| F3 direct/replaceability | DIRECT-MOMENT; DIRECT-CERT-FEATURE; SCALAR-PROPENSITY | Tests whether direct structural moment, direct certificate-feature distillation, or scalar propensity replaces the global geometry |

## Data and Hash Provenance

- Operational train hashes may be used by G0/M1/compiler planning because train is the only cache/compile data scope.
- Validation/test hashes in plan artifacts are provenance-only until the relevant evaluator gate. They are not permission to open validation/test content for cache construction, comparator selection outside validation, control construction, hyperparameter rescue, MLLM prompts, matching, or G0.
- Final test hashes become evaluator provenance only after code/cache/compiler/statistics freeze. Final test labels, predictions, margins, and errors are never inputs to comparator selection or control construction.

## Comparator Freeze

Run `LBSCGP-GLOBAL-M2-COMPARATOR-FREEZE-v1` is mandatory before any M2 validation comparator/FULL run.

Eligible comparator candidates must satisfy all conditions:

- non-MLLM, no teacher/verdict/rationale/head/reranker/key-selector/test-time artifact;
- ordinary full-video train-memory top20 kNN with the same vote rule and train labels as the final endpoint;
- no segment route, no sample weighting, no key selection, no pair/triplet/SupCon objective, and no fragment gold;
- same train/validation split, seed accounting, preprocessing provenance, and SLURM/HateVideo environment constraints as FULL;
- selection evidence comes only from train/OOF and official validation ledgers frozen before final-test evaluation.

Candidate source ledger entries must include `candidate_id`, candidate family name, config path, config SHA256, script path, script SHA256, train/OOF ledger SHA256, validation prediction ledger SHA256, code commit or source tree hash, split hashes, `topk=20`, vote rule, seed list, and explicit zero counters for final-test labels/predictions/margins/errors and adaptive query-label access.

Allowed initial source roots for candidate discovery are the repository non-MLLM full-video retrieval baselines and static ledgers, including `scripts/slurm/train_archive_baseline.sbatch`, `scripts/slurm/train_archive.sbatch`, `scripts/slurm/train_transcript.sbatch`, `data/CLIP_Embedding/*/train_openai_clip-vit-large-patch14-336_HF.pt`, and prior static comparator ledgers used only as provenance. Any source containing MLLM verdicts, segment routes, pair/triplet/SupCon routes, key selectors, sample weights, heads, rerankers, or test-adaptive decisions is logged as rejected and cannot be selected.

Selection criterion:

1. For each dataset, rank eligible candidates by validation macro-F1 averaged over available preregistered seeds.
2. Break ties by validation accuracy.
3. Break remaining ties by lower estimated GPU-hours.
4. Break remaining ties lexicographically by `candidate_id`.

The freeze artifact serializes the selected comparator per dataset and the rejected candidates with reasons. After freeze, no final-test result and no new adaptive validation query can change the comparator.

## Core Blocks

### B0: G0 Contract, Synthetic KKT, and Train-Bank Resource Gate

- Claim tested: C1.
- Dataset/scope: synthetic fixtures plus actual train-bank manifests for `MHC` and `MHC_zh`; no validation/test/outer-held content or labels.
- Metrics/gates: common-basis operator correctness, closed strong convexity, PSD/unit diagonal, coordinate/row/class trust, KKT-only payload verification, replay/hash parity, rank-tail `<= d`, nondegeneration, REMOVE/null parity, cache/isolation injection failures, robust interval coverage.
- Robust coverage rule: low robust coverage disables only the robust safety claim and robust constraints. It does not fail global geometry.
- Priority: MUST-RUN.

### B1: Train-Only Label-Blind MLLM Cache Seal

- Claim tested: C1 prerequisite and C2 input integrity.
- Dataset/scope: train videos only. `MHC` train `N=549`, `MHC_zh` train `N=579`.
- Inputs: deterministic uniform frames, title if available, ASR/OCR/full transcript text if available and deterministically truncated.
- Forbidden inputs: labels, held/validation/test content or labels, predictions, correctness, margins, losses, gradients, neighbors, keys, memory IDs, teacher artifacts, target banks, split statistics beyond train allowlist.
- Schema: `scgp_global_cert_v2`, restricted JSON only. Extra free text, target names, proposition text, mechanism text, timestamps, spans, localization, verdicts, rationales, or stance labels are parse failures.
- Call formula: for dataset `D`, `U_D = unique(evidence_pack_sha256)` among train videos after dedup. Base calls `C_D = 4 * U_D`; no-dedup upper calls are `2196` for MHC and `2316` for MHC-ZH, total `4512`. Retry calls are only for transport/no-response failures, `R_D <= 4 * U_D`; hard contingency cap is `9024` total calls. The cache is dataset-level and seed-independent, so folds/seeds do not multiply MLLM calls.
- Cost honesty: no provider prices or throughput are fabricated; measured throughput/cost fields remain `TBD` until actual cache jobs run.
- Outputs: JSONL replicas, consensus, invalid/unresolved fallback ledger, prompt/input/model/processor/schema hashes, access counters, ID allowlist, Merkle root, seal decision.
- Priority: MUST-RUN.

### B2: Comparator Freeze and Directional Final-Target Pilot

- Claim tested: C2 early direction, not final success.
- Dataset/scope: comparator freeze uses train/OOF plus official validation ledgers only; validation pilot uses official train -> validation for both datasets, seed 0. Test remains unopened.
- Compared systems: frozen moving strongest same-protocol non-MLLM comparator, paired REMOVE replay, FULL.
- Metrics/gates: accuracy, macro-F1, paired deltas, target/rank/fit gates, comparator freeze ledger integrity, zero validation-time cache/certificate/compiler-target/teacher/head/reranker/key-selector reads.
- GO: both datasets have positive FULL-minus-frozen-comparator validation deltas in accuracy and macro-F1, all C1 target/rank/fit/isolation gates pass, and no direct cache artifact is present at validation inference.
- STOP: any nonpositive directional delta, leakage, rank failure, missing KKT certificate, test access, or comparator-freeze ambiguity. No hyperparameter rescue is allowed except preregistered synthetic-only or nested train-only settings.
- Priority: MUST-RUN.

### B3: Final Paired Main Runs

- Claim tested: C2 final performance against the frozen moving strongest comparator.
- Dataset/scope: final train/test protocol for `MHC` and `MHC_zh`, seeds 0/1/2, after M2 GO and all code/cache/compiler/statistics freeze. Test labels are read only by the final evaluator.
- Compared systems: FULL and the frozen strongest paired moving same-protocol non-MLLM comparator. Each paired run uses identical initialization, schedule, split, and ordinary kNN inference where applicable.
- Metrics/gates: accuracy, macro-F1, all paired seed deltas positive. Aggregate `>= +0.030`, bootstrap, and Holm are decided in M5.
- Decision node: `LBSCGP-GLOBAL-M3-PAIRED-MAIN-DECISION-v1` must pass before M4. It verifies all 12 M3 ledgers, paired seed identities, isolation counters, and hash-locked test evaluator provenance; it does not tune or select.
- Priority: MUST-RUN.

### B4: Attribution Controls, Statistics, Tables, and Non-Blocking Diagnostics

- Claim tested: C2 attribution and reporting.
- Dataset/scope: separate per-arm control runs for every dataset x seed cell, followed by CPU statistics and figure/table production.
- Compared systems: REMOVE, SHUFFLE, covariance-matched NOISE, DIRECT-MOMENT, DIRECT-CERT-FEATURE, SCALAR-PROPENSITY.
- SHUFFLE construction: after cache seal, build the shuffle map using train-only/OOF bins, or a preregistered validation-frozen binning artifact generated before final-test evaluation. Bins may use train labels, train/OOF predictions, train/OOF margins, and train/OOF scalar error-propensity estimates only. Final test labels, predictions, margins, and errors are evaluator-only and have construction counters fixed to zero.
- NOISE construction: covariance is estimated from sealed train certificate encodings only, with fixed random seed per dataset/seed/arm and no final-test statistics.
- DIRECT-MOMENT: uses the structural moment representation directly under the same split/init/schedule/inference accounting; it tests whether the moment alone replaces the global projection.
- DIRECT-CERT-FEATURE: distills direct certificate features under the same split/init/schedule/inference accounting; it is distinct from DIRECT-MOMENT.
- SCALAR-PROPENSITY: uses matched scalar difficulty/error propensity from train/OOF or preregistered validation-frozen estimates only.
- Decisive gates: FULL must beat REMOVE, SHUFFLE, DIRECT-MOMENT, DIRECT-CERT-FEATURE, and SCALAR-PROPENSITY; NOISE must degrade or erase the gain. Certificate-field predictivity for parent label, correctness, or margin is diagnostic only and never value evidence or selection.
- Priority: MUST-RUN except the appendix diagnostic, which is NICE-TO-HAVE and never blocks.

## Statistics Gate

M5 implements the final four tests: `MHC/accuracy`, `MHC/macro_f1`, `MHC_zh/accuracy`, `MHC_zh/macro_f1`.

- Paired prediction unit: the same test video under FULL and the frozen comparator for a fixed dataset and seed.
- Hierarchy: for each dataset and metric, a bootstrap replicate first samples the three seeds with replacement, then within each sampled seed samples test videos with replacement, stratified by `parent_video_binary_label` using evaluator-only labels. The video record carries paired FULL/comparator predictions for that seed.
- Statistic: recompute the metric from resampled paired predictions. Accuracy is recomputed as correct/total. Macro-F1 is recomputed from the resampled confusion counts for both classes, not averaged from per-video scores.
- Delta: `delta_b = metric_b(FULL) - metric_b(frozen_comparator)`. The observed delta is computed from all three seeds and original test videos with the same recomputation rule.
- One-sided lower bound: use the 5th percentile of `delta_b` over 10,000 bootstrap replicates with preregistered RNG seed `20260712`; the lower bound must be `> 0`.
- P-value source: one-sided bootstrap sign p-value `p = (1 + count(delta_b <= 0)) / (B + 1)`, with `B=10000`.
- Holm: order the four p-values ascending; at rank `k` with `m=4`, require `p_(k) <= 0.05/(m-k+1)`. Rejecting all four tests is required for the final success claim.
- Joint enforcement: each dataset x metric must satisfy observed delta `>= +0.030`, bootstrap lower bound `> 0`, Holm rejection, and every paired seed delta positive. Failure of any component means no final success claim for C2.

## Run2-v2 Amendment Boundary

`LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2` is the only planned new lineage after the Run2-v1 fail-stop. It has planned config `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json` and planned artifact namespace `artifacts/lb_scgp_global/v2/m0/synth_kkt/`. All v2 schemas, locks, source manifests, access ledgers, payloads, and publish locks must be v2-lineage bound and must not reuse or overwrite v1 paths.

Run2-v1 remains one historical planned run record with zero remaining budget. Both attempted SLURM jobs are preserved under the same approved v1 run ID:

- `12902`: 8 CPU / 64 GB / 0 GPU, elapsed `00:00:04`, terminal `FAILED`, exit `1:0`, older producer path, `KeyError: finite_vi_diagnostic`, no accepted artifact.
- `12904`: 8 CPU / 64 GB / 0 GPU, elapsed `00:00:01`, terminal `FAILED`, exit `1:0`, newer validator path, `KeyError: payload_schema`, no accepted artifact.

These are infrastructure/interface failures before publish. They are not scientific, numerical, KKT, rank, factor, mechanism, dataset, MLLM, OCR, GPU, training, validation/test, or performance evidence. They must never be deleted, overwritten, reused as v2, or called PASS.

Run2-v2 changes only implementation/config/schema/wrapper/verifier alignment needed to make the frozen KKT/rank contract internally checkable. It freezes the v1 science contract: resource request 8 CPU / 64 GB / 0 GPU in `HateVideo` with no `--time`; planned envelope 32 CPU-h, 0 GPU-h, 0 API calls, 5 GB; schema intent, thresholds, fixture identities/counts, expected decisions, KKT tolerances, movement/nondegeneration target, rank/factor rules, intended claim, and failure transition unchanged. No tuning, tolerance change, fixture shrinkage, rank-gate weakening, rescue, or post-hoc scientific claim change is allowed.

Unreviewed partial v2 files currently present in the workspace are unauthorized. They are not implementation evidence, not review evidence, and not execution permission.

Authorization boundary: amendment review has passed, but the first fresh implementation/code review failed and fix2 is only a static repair/freeze state. Execution requires a new fresh independent v2 code review with 0 Critical / 0 High, exact hashes and no-clobber check, and separate execution authorization. Run3 and all later runs, including MLLM/cache, remain locked until v2 PASS and fresh independent v2 artifact review.

## Run Order and Gates

The G0 prefix through the v2 supplement never emits performance claims:

1. `LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1`
2. `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`
3. `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2`
4. `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v2`

| Milestone | Block | Goal | Runs | Decision Gate | Cost Estimate | Risk |
|---|---|---|---|---|---|---|
| M0 | B0 | Contract, synthetic KKT, v2 repair supplement, and actual train-bank resource/replay gate | first 4 G0 records through Run2-v2 and realbank | Run1 FROZEN, v1 failure evidence preserved, v2 PASS plus independent artifact review before realbank; no performance claim | 132 lifetime lineage CPU-h, 0 GPU-h | rank/KKT/resource failure stops before cache |
| M1 | B1 | Build and seal train-only label-blind cache | cache MHC, cache MHC-ZH, seal decision | Merkle seal, zero forbidden counters, schema/QC pass | 24 CPU-h, 4512 base API calls, 9024 retry cap | invalid cache becomes unresolved or STOP, not rescue |
| M2 | B2 | Freeze comparator and run validation directional pilot | comparator-freeze, comparator/FULL validation per dataset, decision | reproducible comparator, then positive directional deltas | 60 CPU-h, 48 GPU-h | ambiguity or nonpositive delta stops expansion |
| M3 | B3 | Final paired FULL vs comparator | 12 paired main runs plus completion decision | complete paired ledgers and isolation; final stats wait for M5 | 160 CPU-h, 216 GPU-h | incomplete or nonpositive seed deltas block M4/M5 |
| M4 | B4 | Decisive controls | 36 per-arm control runs plus decision | FULL beats controls; NOISE erases/degrades gain | 312 CPU-h, 420 GPU-h | direct/scalar match invalidates mechanism claim |
| M5 | B4 | Statistics, paper tables/figures | final stats, tables/figures, optional appendix | corrected inference and artifacts complete | 80 CPU-h, 0 GPU-h | appendix never blocks |

## Compute and Data Budget

Budget conventions are mechanically reproducible from `EXPERIMENT_PLAN.machine.json`:

- Lineage run records: 65 MUST-RUN, 1 NICE-TO-HAVE, 66 total. This is one more record than the original R2 plan because v1 is retained and v2 is added.
- Status-aware counts: 1 MUST FROZEN, 1 MUST FAIL_STOP with zero remaining budget, 1 MUST locked prospective v2, 62 downstream MUST TODO locked until v2 PASS, and 1 NICE TODO locked until final stats.
- Original approved R2 paper-plan envelope before the v2 supplement: 64 MUST plus 1 NICE; MUST 704 CPU-h / 684 GPU-h / 4512 base API calls / 9024 retry-cap API calls / 786 GB; total with NICE 720 CPU-h / 684 GPU-h / 4512 base API calls / 9024 retry-cap API calls / 791 GB.
- Substitution paper-plan view: v2 replaces exhausted v1 as the synthetic KKT scientific slot, so aggregate paper-plan totals remain unchanged from original R2: MUST 704 CPU-h / 684 GPU-h / 4512 base API calls / 9024 retry-cap API calls / 786 GB; total with NICE 720 CPU-h / 684 GPU-h / 4512 base API calls / 9024 retry-cap API calls / 791 GB.
- Lifetime lineage envelope: retain v1 historical planned 32 CPU-h / 0 GPU-h / 0 API / 5 GB and add v2 planned 32 CPU-h / 0 GPU-h / 0 API / 5 GB. Machine run-row totals therefore become MUST 736 CPU-h / 684 GPU-h / 4512 base API calls / 9024 retry-cap API calls / 791 GB; total with NICE 752 CPU-h / 684 GPU-h / 4512 base API calls / 9024 retry-cap API calls / 796 GB.
- Remaining prospective budget after Run1 FROZEN and v1 FAIL_STOP: MUST 700 CPU-h / 684 GPU-h / 4512 base API calls / 9024 retry-cap API calls / 785 GB; total with NICE 716 CPU-h / 684 GPU-h / 4512 base API calls / 9024 retry-cap API calls / 790 GB. Execution-authorized remaining budget now is 0.
- Actual v1 diagnostic spend from jobs `12902` and `12904`: 2 jobs, 5 wall-seconds total, 40 allocated CPU-seconds = 0.0111111111 CPU-hours, 0 GPU-hours, 0 API calls, and no scientific artifact storage. These attempts are actual diagnostic spend under one failed v1 record, not additional planned scientific runs.
- API calls: 4512 base MLLM calls if no evidence-pack dedup and no retries; 9024 hard retry contingency cap. Exact formula is `4 * (U_MHC + U_MHC_zh) + R_MHC + R_MHC_zh`, with `R_MHC + R_MHC_zh <= 4512`.
- Human time: schema/QC review only, estimated 8-12 hours. No segment annotation or new gold labeling.

Concrete two-GPU / 16-CPU schedule:

- Wave 0: after fix2, allow only a fresh independent v2 implementation/code review and exact hashes/no-clobber review. A v2 CPU SLURM execution is not authorized by this plan state; it requires fresh independent v2 code review with 0 Critical / 0 High and separate execution authorization. Realbank can run only after v2 PASS and fresh independent v2 artifact review. Do not submit M1 until G0 decision is GO.
- Wave 1: run M1 cache jobs for MHC and MHC-ZH in parallel if API/CPU policy allows; then run seal decision.
- Wave 2: run comparator-freeze CPU job, then M2 comparator/FULL validation jobs as at most two 1-GPU jobs, one dataset per GPU, followed by CPU decision.
- Wave 3: run M3 final paired jobs as at most two concurrent 1-GPU jobs, paired by dataset/seed, with comparator and FULL schedule matched; then run the M3 completion decision.
- Wave 4: run M4 per-arm controls as at most two concurrent 1-GPU jobs. Each control row is auditable and contributes its own CPU/GPU/storage budget; then run CPU control decision.
- Wave 5: run M5 CPU statistics and table/figure jobs. NICE appendix can run only after final stats and never delays decisions.

All SLURM submissions use `conda activate HateVideo`, no `--time`, and no manual release for `JobHeldUser`.

## Artifact Contract

All formal outputs must use canonical sorted-key JSON where applicable, no NaN/Inf, temp+fsync+atomic publish, no-clobber locks, and payload hashes excluding self-hash fields.

Required schema IDs:

- `scgp_global_contract_freeze_v1`
- `scgp_global_synth_kkt_payload_v1`
- `scgp_global_synth_kkt_payload_v2`
- `scgp_global_realbank_resource_v2`
- `scgp_global_cache_replica_v2`
- `scgp_global_cache_seal_v1`
- `scgp_global_comparator_freeze_v1`
- `scgp_global_target_manifest_v1`
- `scgp_global_fit_eval_ledger_v1`
- `scgp_global_control_arm_v1`
- `scgp_global_stats_v1`
- `scgp_global_table_figure_manifest_v1`

Validation/test target, evaluator, and control schemas must include zero counters for `cache_read_count`, `certificate_read_count`, `compiler_target_read_count`, `auxiliary_head_load_count`, `reranker_load_count`, `key_selector_load_count`, and `teacher_artifact_read_count`. Control-construction schemas must additionally include `final_test_label_read_count_for_control_construction=0`, `final_test_prediction_read_count_for_control_construction=0`, `final_test_margin_read_count_for_control_construction=0`, and `final_test_error_read_count_for_control_construction=0`.

KKT payload must serialize primal values, H metric, affine normals, box/coordinate normals, SOC normals, PSD normal, halfspace normals, stationarity residual, dual feasibility, complementarity, optional duality gap, and hashes. PSD sign convention is explicit: for feasible constraint `G in S_+`, the normal contribution in `0 = H(X*-X0) + sum_j v_j` is `v_psd = -S_psd`, where `S_psd in S_+` and `tr(S_psd G*)` is complementary.

## Tables and Figures

- Main Table 1: paired final frozen comparator/FULL by dataset and seed, with accuracy, macro-F1, deltas, bootstrap lower bounds, p-values, Holm order, and corrected inference.
- Table 2: decisive controls, including REMOVE, SHUFFLE, NOISE, DIRECT-MOMENT, DIRECT-CERT-FEATURE, and SCALAR-PROPENSITY.
- G0 Table: contract, synthetic KKT, replay/hash, rank-tail, resource, leakage counters, robust coverage.
- Figure 1: method and run order from train-only cache to ordinary kNN.
- Figure 2: paired deltas and hierarchical bootstrap intervals.
- Optional appendix: mechanism/failure diagnostics only if non-blocking.

## Preimplementation Checklist

- [ ] Freeze `scgp_global_cert_v2` and reject extra keys/free text/verdict/rationale/localization/span/timestamp/target/mechanism/stance fields.
- [ ] Freeze prompt/input/model/processor/schema hashes before any cache call.
- [ ] Verify `N` and `d` from manifests during preflight, not from prose estimates.
- [ ] Implement KKT-only acceptance; solver traces and finite probes are diagnostics only.
- [ ] Explicitly serialize PSD normal sign as `v_psd=-S_psd`.
- [ ] Keep robust constraints disabled unless coverage passes and the report replays.
- [ ] Run comparator-freeze before M2 validation; selection must use only train/OOF and validation ledgers and zero final-test/adaptive query-label access.
- [ ] Guarantee validation/test inference cannot import cache/certificate/compiler-target/teacher/head/reranker/key-selector artifacts.
- [ ] Build SHUFFLE, NOISE, DIRECT-MOMENT, DIRECT-CERT-FEATURE, and SCALAR controls from train/OOF or preregistered validation-frozen construction artifacts only.
- [ ] Register all hyperparameters before outcomes; otherwise use synthetic-only or nested train-only selection.
- [ ] Confirm no sample weighting, key selection, pair/triplet/SupCon, segment route, or local rank-cell route.
- [ ] Confirm local v7 evidence is never cited as global PASS evidence.
- [ ] Validate run-ID parity/order/dependencies between tracker and machine JSON before execution.
- [ ] Complete independent amendment review before any v2 implementation audit.
- [ ] Complete static contract matrix and negative tests, fresh independent Run2-v2 code review with 0 Critical / 0 High, exact hashes/no-clobber check, and separate execution authorization before any Run2-v2 SLURM submission.
- [ ] Keep all Run2-v2 Python validation inside the future single SLURM wrapper; no login-node Python preflight.

## G0 Runs Through Run2-v2 Supplement

1. `LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1`: implementation contract audit/freeze.
2. `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`: historical FAIL_STOP evidence; no artifact is accepted and no PASS claim is made.
3. `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2`: prospective synthetic KKT supplement at `artifacts/lb_scgp_global/v2/m0/synth_kkt/manifest.json`, fix2-complete but locked pending fresh 0C/0H implementation/code review, exact hashes/no-clobber review, separate execution authorization, and later artifact review gates.
4. `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v2`: actual train-bank static/resource microbenchmark plus replay/decision, without training or performance claim; depends on Run2-v4 PASS and fresh independent v4 artifact review (`M0_RUN2_V4_ARTIFACT_REVIEW.md` = ARTIFACT_ACCEPTED, 2026-07-13), not Run2-v1/v2/v3. The realbank input protocol (A frozen CLIP-L/336 train bank as `Z0`, B synthetic label-blind placeholder `b_struct` at `m=36`, C two-stage producer/verifier GO criterion) is pinned in `EXPERIMENT_PLAN.machine.json` `runs[3].realbank_protocol` and `REALBANK_RESOURCE_V1_PLAN_AMENDMENT.md`. The v1 single submit burned preflight on a wrapper `$TMPDIR` path escape; v2 is a byte-clone of the eight v1 entities plus the audit-specified in-repo `slurm/tmp/` handoff fix, with run_id/schema-id v1→v2 REPLACED in place (`REALBANK_FULLCHAIN_STATIC_AUDIT.md`, `REALBANK_RESOURCE_V2_CLONE_FREEZE.md`).

## Run2-v4 Amendment Note (2026-07-13)

The v2 and v3 single-submit lineages are both closed. v2 spent its budget (job `12971`, missing `jsonschema`); v3 was consumed and CLOSED (job `12974`, machine `run_order[2]` v2/v3 plan-document/code drift). Per `M0_RUN2_V3_RESULT_TO_CLAIM_REVIEW.md` §5, a v4 lineage is opened by `M0_RUN2_V4_PLAN_AMENDMENT.md`: v4 is a byte-exact clone of v3, and the authoritative `EXPERIMENT_PLAN.machine.json` REPLACES index `[2]` in place (v2 content -> v4) — array length and every downstream index unchanged, so realbank stays at `[3]` and now depends on `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v4`. The v2/v3 records above are preserved as closed-lineage evidence and are NOT rewritten. v4 remains locked pending independent v4 amendment review, a fresh independent 0C/0H v4 code review (including the mandatory runtime cross-check static-simulation table, every row PASS) with dependency-availability evidence, exact hashes/no-clobber check, and separate execution authorization before any SLURM submission. No PASS claim and no execution are authorized.
