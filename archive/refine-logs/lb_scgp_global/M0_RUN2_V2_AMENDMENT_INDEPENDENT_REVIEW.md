# M0 Run2-v2 Amendment Independent Review

Date: 2026-07-12

Reviewer boundary: fresh independent amendment review only. I read `AGENTS.md` first and did not run Python, SLURM, experiments, data processing, model/API calls, training, evaluation, or artifact generation. Shell use was read-only inspection only. This report is the only write.

Reviewed primary source:

- `refine-logs/lb_scgp_global/M0_RUN2_RESULT_TO_CLAIM_REVIEW_FRESH.md`
- SHA256: `b02e5a0f7839e8b215c15197f6f120e3bebfe498ab89d2120c26b50585707a0c`
- Fresh review verdict used here: Run2-v1 does not support the executability claim; v2 may be authorized in principle only as a new non-overwriting infrastructure repair lineage, with no execution authorization.

## Required Hash Checkpoints

Exact paths from the review request, checkpoint 1:

- `EXPERIMENT_PLAN.md`: MISSING at repo root (`sha256sum: No such file or directory`)
- `EXPERIMENT_TRACKER.md`: MISSING at repo root (`sha256sum: No such file or directory`)
- `EXPERIMENT_PLAN.machine.json`: MISSING at repo root (`sha256sum: No such file or directory`)
- `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.md`: `95b4b839017479bad4d2ed7a48455864c765e8f91997adab0fa4a7ca66aa57fb`
- `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.machine.json`: `56603297b2a1f6bcf12e941d96ce95911bb2ab8bfd0de967586430d38959f3d8`
- `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.HASHES`: MISSING (`sha256sum: No such file or directory`)

Exact paths from the review request, checkpoint 2:

- `EXPERIMENT_PLAN.md`: MISSING at repo root (`sha256sum: No such file or directory`)
- `EXPERIMENT_TRACKER.md`: MISSING at repo root (`sha256sum: No such file or directory`)
- `EXPERIMENT_PLAN.machine.json`: MISSING at repo root (`sha256sum: No such file or directory`)
- `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.md`: `95b4b839017479bad4d2ed7a48455864c765e8f91997adab0fa4a7ca66aa57fb`
- `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.machine.json`: `56603297b2a1f6bcf12e941d96ce95911bb2ab8bfd0de967586430d38959f3d8`
- `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.HASHES`: MISSING (`sha256sum: No such file or directory`)

Resolved LB-SCGP global paths, checkpoint 1:

- `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.md`: `8ef1417fe16f8a556c82394705259f13447e7747d600c0f8f8061e237ecb8994`
- `refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md`: `d0c7e6ff291b6ae4ba6d5661b2e8ddc2bb8a80ac7f2f6d1e37ab8494782df483`
- `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json`: `838370c8eee68f568b6d133f80b305fed23bc14342d4c4bf08df976fd4d73d07`
- `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.md`: `95b4b839017479bad4d2ed7a48455864c765e8f91997adab0fa4a7ca66aa57fb`
- `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.machine.json`: `56603297b2a1f6bcf12e941d96ce95911bb2ab8bfd0de967586430d38959f3d8`
- `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT_HASHES.sha256`: `a88fc1c75c10b653deebf03c1792733d2bb412f13dc021567b221d009cc38390`

Resolved LB-SCGP global paths, checkpoint 2:

- `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.md`: `8ef1417fe16f8a556c82394705259f13447e7747d600c0f8f8061e237ecb8994`
- `refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md`: `d0c7e6ff291b6ae4ba6d5661b2e8ddc2bb8a80ac7f2f6d1e37ab8494782df483`
- `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json`: `838370c8eee68f568b6d133f80b305fed23bc14342d4c4bf08df976fd4d73d07`
- `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.md`: `95b4b839017479bad4d2ed7a48455864c765e8f91997adab0fa4a7ca66aa57fb`
- `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.machine.json`: `56603297b2a1f6bcf12e941d96ce95911bb2ab8bfd0de967586430d38959f3d8`
- `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT_HASHES.sha256`: `a88fc1c75c10b653deebf03c1792733d2bb412f13dc021567b221d009cc38390`

Checkpoint result: no hash drift was observed. The exact root-level plan/tracker/machine paths and exact `.HASHES` path from the prompt are consistently absent; the corresponding repository-local LB-SCGP global files and the actual `_HASHES.sha256` ledger are stable.

## Hash Parity

PASS with a path-reference note.

- `M0_RUN2_V2_PLAN_AMENDMENT_HASHES.sha256` records:
  - `M0_RUN2_V2_PLAN_AMENDMENT.md`: `95b4b839017479bad4d2ed7a48455864c765e8f91997adab0fa4a7ca66aa57fb`
  - `M0_RUN2_V2_PLAN_AMENDMENT.machine.json`: `56603297b2a1f6bcf12e941d96ce95911bb2ab8bfd0de967586430d38959f3d8`
- Both match the current checkpoint hashes.
- `M0_RUN2_V2_PLAN_AMENDMENT.machine.json` output hashes also match current hashes for:
  - `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.md`: `8ef1417fe16f8a556c82394705259f13447e7747d600c0f8f8061e237ecb8994`
  - `refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md`: `d0c7e6ff291b6ae4ba6d5661b2e8ddc2bb8a80ac7f2f6d1e37ab8494782df483`
  - `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json`: `838370c8eee68f568b6d133f80b305fed23bc14342d4c4bf08df976fd4d73d07`
  - `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.md`: `95b4b839017479bad4d2ed7a48455864c765e8f91997adab0fa4a7ca66aa57fb`

Verified v1 log hashes:

- `slurm/logs/lbscgp_global_r2_run2_synth_kkt_12902.err`: `77ba892b49d7c6262bc0a6165188c173b8f0619a7cbbfe9fdcc881fc8ec5f71c`
- `slurm/logs/lbscgp_global_r2_run2_synth_kkt_12902.out`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `slurm/logs/lbscgp_global_r2_run2_12904.err`: `93e8515cad0d89ec65d3a1844d497694324143b9d836bc304b732765f1ead306`
- `slurm/logs/lbscgp_global_r2_run2_12904.out`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## Audit Findings

PASS: v1 `12902` / `12904` preservation.

- Amendment lines 15-31 retain `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1` as one historical planned run with two failed jobs, zero remaining budget, no artifact, and no scientific evidence.
- Tracker lines 15 and 91-97 preserve both jobs and their failure classes.
- Machine plan lines 474-508 preserve job IDs, names, resources, elapsed seconds, exit codes, artifact_published=false, and log hashes.

PASS: unique non-overwrite v2.

- Amendment lines 32-44 introduce exactly one new MUST run: `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2`.
- Planned config: `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json`.
- Planned namespace: `artifacts/lb_scgp_global/v2/m0/synth_kkt/`.
- Machine plan lines 523-621 encode v2 as a separate `LOCKED_PENDING_AMENDMENT_REVIEW` record with v2-bound config, schema, locks, source manifest, access ledger, publish lock, no-clobber lock, and manifest.

PASS: frozen science.

- Amendment lines 46-58 restrict v2 to interface-key/contract alignment only.
- Plan lines 152-156 and machine plan lines 585-600 keep resource request, budget, schema intent, thresholds, fixtures, expected decisions, KKT tolerances, rank/factor rules, solver/math/verifier logic, intended claim, and failure transition unchanged.
- Solver fixes, tuning, tolerance changes, fixture changes, rescue, and post-hoc scientific changes are forbidden.

PASS: DAG `Run1 -> v1 FAIL_STOP -> v2 -> realbank`.

- Amendment lines 60-69 and plan lines 162-165 encode the G0 prefix:
  1. `LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1`
  2. `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`
  3. `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2`
  4. `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1`
- Machine plan lines 289-293 encode the same run order.
- Machine plan lines 638-640 make realbank depend on v2, not v1.

PASS: M1 / MLLM / Run3 locked.

- Amendment line 13 and plan line 156 keep Run3 and later runs, including MLLM/cache work, locked until v2 PASS and fresh independent v2 artifact review.
- Tracker lines 4-7 record Run2-v2 locked pending amendment review, Run3 and later locked, and 62 downstream MUST records locked until v2 PASS.
- Machine plan lines 3743-3746 explicitly lock Run3 and MLLM cache until v2 PASS plus fresh artifact review.

PASS: budget/status parity and execution budget 0.

- Amendment lines 71-85, tracker lines 5-7, plan lines 180-186, and machine plan lines 3788-3868 agree on:
  - 65 MUST, 1 NICE, 66 total lineage records.
  - v1 FAIL_STOP with zero remaining budget.
  - v2 as one locked prospective MUST record.
  - original/substitution paper-plan envelope unchanged.
  - lifetime lineage envelope increased by one retained v1 plus one v2 record.
  - remaining prospective budget: MUST 700 CPU-h / 684 GPU-h / 4512 base API / 9024 retry-cap API / 785 GB.
  - execution-authorized remaining budget: 0.

PASS: parent-video-label-only and no segment gold.

- Amendment lines 87-93, plan lines 15-16, tracker line 10, machine plan lines 74-79, and machine counters lines 3987-4035 all preserve `parent_video_binary_label` as the only gold supervision.
- `segment_gold_exists=false`, `segment_gold_used=false`, and segment/frame/timestamp/span/localization/stance/target/mechanism/rationale/fragment gold are absent or zero-access.

PASS: amendment authorizes at most implementation audit after review, never execution or SLURM.

- Amendment lines 7-13 state the amendment is planning only and does not authorize v2 implementation or v2 execution.
- Plan lines 156, 192, and 248-249 require implementation, static contract matrix/negative tests, fresh independent v2 code review with 0 Critical / 0 High, exact hashes/no-clobber, and separate execution authorization before any Run2-v2 SLURM submission.
- Machine amendment lines 266-274 and machine plan lines 4166-4170/4215-4222 encode `ready_for_execution=false` and execution authorization false.

PASS: partial v2 code is not evidence.

- Read-only file listing found v2 scaffolding paths including `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json`, `scripts/wrappers/lb_scgp_global_r2_run2_v2.sh`, and `scripts/analysis/lb_scgp_global_r2_run2_v2_*.py`.
- No `artifacts/lb_scgp_global/v2/...` artifact was listed.
- Amendment line 42, plan line 154, and machine plan lines 612-616 mark unreviewed partial v2 files as unauthorized, not implementation evidence, and not execution permission.

## Severity Findings

- Critical: 0
- High: 0
- Medium: 0
- Low: 1

L1: Prompt-path mismatch in the required hash checkpoint list. The exact root-level `EXPERIMENT_PLAN.*` / `EXPERIMENT_TRACKER.md` paths and exact `M0_RUN2_V2_PLAN_AMENDMENT.HASHES` path are absent. The repository-local LB-SCGP global files and actual `M0_RUN2_V2_PLAN_AMENDMENT_HASHES.sha256` ledger are present, stable across two checkpoints, and hash-parity checked. This is a path-reference issue, not observed hash drift.

## Decision

- claim_supported: yes
- route: authorize_v2_implementation_audit_only
- confidence: high
- next_boundary: v2 implementation audit only. No execution, SLURM submission, Python validation, experiments, data processing, model/API calls, MLLM/OCR/cache work, training, evaluation, Run3, or realbank unlock is authorized by this review.

Because there is no hash drift and severity is 0 Critical / 0 High, the next step may be a v2 implementation audit only. Execution remains locked until completed v2 implementation, static contract matrix and negative tests, fresh independent v2 code review with 0 Critical / 0 High, exact hashes/no-clobber check, and separate execution authorization.

Review SHA256: not embedded; compute after file creation and report in final message.
