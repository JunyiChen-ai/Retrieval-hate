# LB-SCGP Experiment Tracker

**Status date:** 2026-07-11
**Current phase:** G0 v5 stopped at the synthetic numerical gate. The independent formal code audit remains published and post-verified, but synthetic job `12833` returned `FAIL` (`expected_statuses_ok=false`, `dykstra_gate=false`) and therefore realfold, replay, decision, G1, teacher, MLLM, OCR, held, val, test, and performance workloads remain locked and were not submitted.
**Authorization:** G0 only. G1--G4 locked. Teacher/new-OCR calls = 0.
**Supervision:** only parent-video binary labels are gold; `segment_gold_exists=false`, `segment_gold_used=false`. Subclips are not G0/G1 inputs, and inherited parent labels are not segment gold or segment supervision.

| Run ID / namespace | Stage | Purpose | Binding gate | Resource / dependency | Status | Artifact / note |
|---|---|---|---|---|---|---|
| `LBSCGP-G0-SANITIZE-MHC_zh-F4-v1` | Pre-G0 quarantine | Build one physically train-only whole-video feature artifact under the explicit sanitizer exception | select solely by `memory_ids`; labels only from `memory_labels`; whitelist-only feature schema; no subclip/segment artifact; no model/optimizer/evaluator/teacher/OCR import/call; no-clobber | CPU 4 / 32 GB; separate sanitizer namespace before freeze | COMPLETED_AFTER_12737_FAILURE | Job `12737` failed before artifacts with the relative path bug. Path fix review then passed 0C/0H. Job `12738` completed and produced `outer_train_features.pt` SHA256 `ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496`, sanitized provenance SHA256 `b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007`, and quarantine manifest SHA256 `055dffed9b61053293741ec5ba0ce3577daf458ceef4a3f81143a81d937c684b`. |
| `LBSCGP-G0-SANITIZER-VERIFY-MHC_zh-F4-v1` | Pre-G0 quarantine verifier | Independently verify sanitizer outputs and write formal sanitized decision | exact ID equality/order; zero overlap with `query_ids`; inherited parent-video labels; no segment artifact/objective; sanitized provenance/decision carry no mixed/quarantine locator/hash | CPU 4 / 32 GB; after sanitizer build | PASS_ARTIFACT_LEVEL | Job `12739` completed and wrote `sanitizer_decision.json` SHA256 `172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954`, embedded payload `8685c805995f97ad0513658c1345b6320dc794e2bad02b907d9a2d07ff16cb1f`, status `PASS`, feature IDs `464`, held-query sentinels `115`, all gates true. Physical review: `refine-logs/lb_scgp/G0_SANITIZER_PHYSICAL_REVIEW.md`. |
| `LBSCGP-G0-FREEZE-v1` | G0 | Historical first freeze | v1 hashes; no-clobber; zero forbidden readers/calls; supervision contract | CPU 8 / 64 GB | COMPLETED_FAILED_AUDIT_PREDECESSOR | Job `12742` completed; `artifacts/lb_scgp/v1/CONFIG_FREEZE.json` SHA256 `b6697472b61a61706c694a67b21618d618fcad6e7f59265d8696aee79dc46889`, payload SHA256 `b3b33090b39b3b975c2cf213aab669041b345c6ef3a3f7c200366a506bcebfd5`, log SHA256 `70cd194ed6a811be6956644e832055907f94750ccec6876352a8c1d6b5e98628`. Preserved immutable; superseded by v2 repair. |
| `LBSCGP-G0-CODE-AUDIT-v1` | G0 | Independent implementation review | 0 CRITICAL / 0 HIGH required | separate reviewer after v1 freeze | FAIL_REVIEW_ONLY | `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW.md` reported FAIL: C1 mutable freeze inputs and H1 pre-freeze sanitizer schema gap. No PASS artifact was created. |
| `LBSCGP-G0-FREEZE-v2` | G0 | No-clobber repaired freeze lineage | stable scientific protocol inputs only; dedicated sanitizer schema snapshot; mutable progress docs excluded; zero forbidden readers/calls; supervision contract | CPU 8 / 64 GB; after C1/H1 repair | COMPLETED_CODE_AUDIT_NEXT | Job `12746` completed; `artifacts/lb_scgp/v2/CONFIG_FREEZE.json` SHA256 `4c6f0199a429bbf766cb284b9ecaedd9dcaf38733834dd54574245ab86b633ae`, payload SHA256 `bda18d7dfda00ab5808595afec04e5b925d3a4920f55fe963dec5e9d99795c0b`, log SHA256 `e95d167af054b91582e1d1f8fbf66fb57a3b5cd67e298883940d385a65ccf563`, lock SHA256 `22426f693ba3f72d52928b0a86d08fe439d7e7fd43f9c74c61aedb710443e211`. Execution record: `refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V2.md`. |
| `LBSCGP-G0-CODE-AUDIT-v2` | G0 | Fresh independent v2 implementation review / formal audit artifact | 0 CRITICAL / 0 HIGH required | separate reviewer after v2 freeze, before runs | FAIL_REVIEW_ONLY | `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V2.md` reported one Critical: mandatory post-freeze formal review path was not excluded from dirty-state predecessor checks. No PASS artifact was created. |
| `LBSCGP-G0-FREEZE-v3` | G0 | No-clobber v3 freeze lineage for the v2 C1 dirty-state repair | config-explicit dirty-state policy; v3 formal artifact namespace exclusion; stable scientific protocol inputs only; zero forbidden readers/calls; supervision contract | CPU 8 / 64 GB; after v2 audit C1 repair | COMPLETED_CODE_AUDIT_NEXT | Job `12748` completed; `artifacts/lb_scgp/v3/CONFIG_FREEZE.json` SHA256 `9fba7f1649dd67d4bb0fcc193e555d8246d7a4966307732e87d5e9fca7346dd9`, payload SHA256 `352ec2215e2225b1768a13f39f96ef935b91966606d8db874d6de4410b1a9f3d`, log SHA256 `f60d2301e7460ab91b25f6c323c578e49a00001b46c67a590f9f3c3d58abf545`, lock SHA256 `9c32d07c524e466ad06c06fc2a472829764cd1facff22390eac8b2879d329b8f`. Execution record: `refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V3.md`. |
| `LBSCGP-G0-CODE-AUDIT-v3` | G0 | Fresh independent v3 implementation review / formal audit artifact | 0 CRITICAL / 0 HIGH; independent verifier separation; fail-closed paths; v3 frozen inputs rehash clean after docs; mandatory review path dirty-hash stable | separate reviewer after v3 freeze, before runs | PASS_REVIEW_ONLY_NO_ARTIFACT | `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V3.md` reports 0 Critical / 0 High / 2 Important and closes the exact v2 Critical for v3. No `artifacts/lb_scgp/v3/g0/code_audit/{review.md,audit.json}` was created because no authorized schema/no-clobber producer task exists; hand-written PASS artifacts remain forbidden. |
| `LBSCGP-G0-FREEZE-v4` | G0 | No-clobber v4 tooling-lineage freeze | strict independent-verifier audit-publish path; exact review/report sidecar dirty exclusions; v4 formal schema consumers; stable scientific protocol inputs only; zero forbidden readers/calls; supervision contract | CPU 8 / 64 GB; after v3 tooling-gap review | COMPLETED_BUT_OPERATIONAL_LINEAGE_FAILED | Job `12759` completed; `artifacts/lb_scgp/v4/CONFIG_FREEZE.json` SHA256 `dcf65eceba04e7c4f08145b2012653705f7347c6e96ebc8b2b769280dff48fd0`, payload SHA256 `92301ad95870b8a1af41e7e69e45054a2e63fe80a021cf4c1c22906aea0872bf`, log SHA256 `6203cab3eded38f22638980c4828020a10ddbd8421819a0d5f5059eca6faa6da`, lock SHA256 `09003ce9e741d7c0310045f854479deb8fecff74bfddd33f6b9d80dc6df9572a`. Superseded by v5 repair because v4 producer-consumer failed after publication. |
| `LBSCGP-G0-CODE-AUDIT-v4` | G0 | Independent v4 review and formal audit publication | strict machine review record + report hash; publisher recomputed frozen inputs, dirty equality, v1-v3 non-overwrite, zero counters, no-segment-gold, wrapper/run-ID gates; atomic no-clobber publication | fresh independent GPT-5.5 xhigh auditor after v4 freeze | FAIL_OPERATIONAL_AFTER_PUBLICATION | Formal files exist under `artifacts/lb_scgp/v4/g0/code_audit/`, but publication verification recorded `producer_consumer_ok=false` and `NameError: name 'git_state' is not defined`. v4 formal outputs are failed-lineage evidence only and never valid unlock evidence. |
| `LBSCGP-G0-SYNTH-v4` | G0 | Projector/KKT/Dykstra/rank-cell/exact-vote/Farkas/factor sealed suite | all frozen parity/tolerance/status gates | CPU 8 / 64 GB | BLOCKED_BY_V4_OPERATIONAL_FAIL | no synthetic run was authorized or submitted |
| `LBSCGP-G0-REAL-MHC_zh-F4-S0-v4` | G0 | One sealed worst-size real fold numerical and cost microbenchmark | `LOCAL_STATIONARY_CERTIFIED`; float64/KKT/Farkas/factor/fit/rollback; whole-video memory only; `lambda_seg=0`; peak limits; no endpoint | 1 GPU / 8 CPU / 64 GB | BLOCKED_BY_V4_OPERATIONAL_FAIL | outer-held labels/content unopened |
| `LBSCGP-G0-REAL-REPLAY-MHC_zh-F4-S0-v4` | G0 | Separate GPU replay of actual fit/rollback | independently rerun batch order, target-fit steps, realized bank hash, live-state rollback vs direct REMOVE; fail closed if any segment cache/objective appears | 1 GPU / 8 CPU / 64 GB | BLOCKED_BY_V4_OPERATIONAL_FAIL | no replay was run |
| `LBSCGP-G0-DECISION-v4` | G0 | Independently recompute joint G0 decision | every G0 gate; exact H10 formula; replay artifact; zero-call/test-clean/no-segment-gold; strict v4 code-audit schema | CPU 8 / 64 GB | BLOCKED_BY_V4_OPERATIONAL_FAIL | no decision run; G1 remains locked |
| `LBSCGP-G0-FREEZE-v5` | G0 | No-clobber v5 NameError repair freeze | import `git_state`; v4/v5 strict audit schema support; exact v5 namespace/run IDs; v4 formal artifacts as prior no-clobber evidence only; stable scientific protocol inputs only | CPU 8 / 64 GB; after focused v5 regression | COMPLETED_CODE_AUDIT_NEXT | Job `12823` completed; `artifacts/lb_scgp/v5/CONFIG_FREEZE.json` SHA256 `254e45afe9c0355892824c0c26bc73b4b0854cb20c67c3703982762fad010931`, payload SHA256 `d89f7cd4ad43c8ef83a04b7530ec19186ec1cb0e96958d239f1ce5c2b146bb4d`, log SHA256 `c43a2b16fc8c95bdfafb5c48c674fa4b778dd9c3d3c3764e24fe50ed038a0526`, lock SHA256 `54dc06d236d5fc1f3ac96400f1a81faeb1d2c0c8e5af075065d1964260de98a9`. |
| `LBSCGP-G0-CODE-AUDIT-v5` | G0 | Independent v5 review and formal audit publication | strict machine record + report hash; v5 publisher/producer/decision contract; no-clobber publication | fresh independent auditor after v5 freeze | PASS_POST_VERIFIED | Review SHA256 `495b5f3bc453034ae5f9830a77bc9b4a2b04af181b0d4365e95bdbaf450bd36b`; record SHA256 `4cb399a0209025581cef094f9f339b6617d9a0ad1d22d5925c131107118a3770`; audit-publish job `12830` completed; post-verifier job `12831` reports `producer_consumer_ok=true`, `decision_consumer_ok=true`, `all_ok=true`, `dirty_equal_frozen=true`. |
| `LBSCGP-G0-SYNTH-v5` | G0 | Projector/KKT/Dykstra/rank-cell/exact-vote/Farkas/factor sealed suite | all frozen parity/tolerance/status gates; strict v5 code-audit PASS/post-verify evidence | CPU 8 / 64 GB; only after valid v5 code-audit PASS artifacts | FAIL_STOP | Job `12833` failed `2:0` after `00:00:44`; log SHA256 `a8a249101ebf8ebe3ab56d5b152b8df35a8f593c2271e26e111b04364726ce49`. Manifest SHA256 `07dc7d5d17194cd7a2b5d42d539adb9e8248e78b4dc629bbcdaf9d4f64719242`, payload `751b5ede4cdd6f05032768b4c9295b56ba62fbe370be11436f7ca3f7dbec3fc5`, `thresholds_ok=true`, `expected_statuses_ok=false`, `dykstra_gate=false`. Dykstra `feasible_interior`, `feasible_boundary`, and `feasible_oriented_boundary` returned `BOUNDED_SEARCH_FEASIBLE` despite expected `LOCAL_STATIONARY_CERTIFIED`. Dedicated record: `refine-logs/lb_scgp/G0_V5_NUMERICAL_EXECUTION.md`. |
| `LBSCGP-G1-OOF-MHC-F{0..4}-S0-v1` | G1 | Zero-teacher REMOVE vs LABEL-ONLY strict OOF | pooled acc/mF1 `>=+0.050`; 5/5 fold signs positive; all numerical/fit/Farkas gates | 1 GPU/fold; verified G0 GO + fresh authorization | NOT_RUN_LOCKED | no certificate or MLLM result |
| `LBSCGP-G1-OOF-MHC_zh-F{0..4}-S0-v1` | G1 | Same | same | same | NOT_RUN_LOCKED | actual ordinary full-video kNN only |
| `LBSCGP-G1-DECISION-v1` | G1 | Two-dataset SCGP-0 gate | all dataset x metric x fold gates | CPU; all 10 folds | NOT_RUN_LOCKED | failure keeps teacher at zero |
| `LBSCGP-G2-SAMPLE-{MHC|MHC_zh}-v1` | G2 | Freeze powered probability sample and A/B halves | <=128 unique/dataset; inclusion probabilities; test-clean | CPU; only after G1 GO | NOT_RUN_LOCKED | still zero calls |
| `LBSCGP-G2-TEACHER-{MHC|MHC_zh}-v1` | G2 | Label-blind whole-video certificate pilot | <=512 calls/dataset; strict schema; Merkle closure before labels; design gates | LOCKED | NOT_RUN_LOCKED | all outputs weak pseudo-signals, never gold |
| `LBSCGP-G2-DECISION-v1` | G2 | Pilot conditional-value decision | support/selection/reflection + partial OOF beats all controls | CPU; after both pilots | NOT_RUN_LOCKED | no full-cache calls unless GO |
| `LBSCGP-G3-{MHC|MHC_zh}-S0-{ARM}-v1` | G3 | Seed-0 causal controls and corruption | FULL `+0.010` vs every binding arm in both metrics/datasets; monotone corruption | LOCKED after G2 GO | NOT_RUN_LOCKED | freezes strongest direct control |
| `LBSCGP-G3-DECISION-v1` | G3 | Joint seed-0 freeze | all control/attribution/test-clean gates | CPU | NOT_RUN_LOCKED | test remains unopened until GO |
| `LBSCGP-G4-{MHC|MHC_zh}-S{0,1,2}-{FULL|REMOVE|SHUFFLE|DIRECT}-v1` | G4 | Final endpoint and removability | moving-bar `+0.030/+0.030`; all signs; bootstrap/Holm; direct attribution | LOCKED after G3 GO | NOT_RUN_LOCKED | ordinary kNN, no teacher at inference |
| `LBSCGP-G4-FINAL-STATS-v1` | G4 | Completion audit | every project target, novelty, statistics and supervision item verified | CPU; all final ledgers | NOT_RUN_LOCKED | only this run may support goal completion |

## Submission DAG

```text
SANITIZE-v1 (done) -> SANITIZER-VERIFY-v1 (done) -> G0-FREEZE-v1 (failed audit predecessor)
  -> G0-FREEZE-v2 (failed audit predecessor)
  -> G0-FREEZE-v3 (done) -> independent CODE-AUDIT-v3 review-only
  -> G0-FREEZE-v4 -> independent CODE-AUDIT-v4 publication -> v4 operational FAIL/BLOCKED
  -> G0-FREEZE-v5 (done) -> independent CODE-AUDIT-v5 audit-publish + post-verify (done) -> G0-SYNTH-v5 (FAILED/STOP, job 12833) -> G0-REAL-MHC_zh-F4-v5 (not run, locked) -> G0-REAL-REPLAY-v5 (not run, locked) -> G0-DECISION-v5 (not run, locked)

verified G0 GO + fresh authorization -> G1 OOF 5 folds x 2 datasets -> G1-DECISION
verified G1 GO -> G2 sample/teacher pilot -> G2-DECISION
verified G2 GO -> G3 seed0 controls -> G3-DECISION
verified G3 GO -> G4 2 datasets x 3 seeds -> FINAL-STATS
```

## Current Evidence State

- LB-SCGP method specification is READY (9.15). Canonical Round3 independent review originally failed only because physical sanitizer artifacts did not exist; code review had 0 High and the sole Critical was physical C1.
- Sanitizer build job `12737` failed before artifacts with exit code `1:0`; this failure history is preserved in `refine-logs/lb_scgp/G0_SANITIZER_EXECUTION.md`.
- Path-normalization repair review reported 0 Critical / 0 High and authorized a fresh sanitizer build under SLURM.
- Build job `12738` completed. Verifier job `12739` completed and wrote status `PASS`.
- Physical artifact review `refine-logs/lb_scgp/G0_SANITIZER_PHYSICAL_REVIEW.md` reports 0 Critical / 0 High and closes C1 at artifact level only.
- Physical evidence: feature SHA256 `ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496`; sanitized provenance file SHA256 `b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007`; quarantine manifest file SHA256 `055dffed9b61053293741ec5ba0ce3577daf458ceef4a3f81143a81d937c684b`; verifier log SHA256 `40f333bb82884e9ad412fc87a20165e1640238815c3c6b6951d003e1cb0ec247`; decision file SHA256 `172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954`; decision payload SHA256 `8685c805995f97ad0513658c1345b6320dc794e2bad02b907d9a2d07ff16cb1f`.
- The verifier decision records `memory_id_count=464`, `query_id_sentinel_count=115`, all gates true, no segment artifact/objective, zero teacher/MLLM/OCR/network calls, and zero formal query reads.
- No `segment` or `subclip` artifact exists under `artifacts/lb_scgp`; formal provenance and decision both carry `segment_cache_path=null`, `segment_cache_sha256=null`, `segment_artifact_created=false`, `segment_objective_allowed=false`.
- G0 v1 freeze job `12742` completed but formal audit failed on C1/H1; v1 remains immutable and superseded.
- G0 v2 freeze job `12746` completed, but formal audit v2 failed with one Critical on dirty-state exclusion of mandatory post-freeze review records. No v2 PASS artifact was created.
- G0 v3 freeze job `12748` completed after the narrow dirty-state repair. G0 PASS is not claimed. The one-real-fold fixture remains MHC-ZH fold 4 for maximum outer-train size, not outcome.
- Independent v3 formal code audit review `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V3.md` reports `PASS_REVIEW_ONLY`, 0 Critical / 0 High / 2 Important, no-segment-gold PASS, and v2 Critical closed for v3. Formal PASS artifacts remain absent because no existing authorized producer was found.
- G0 v4 formal code-audit artifacts exist and are internally consistent, but `publication_verification.json` records `all_ok=false` because the producer consumer raised `NameError: name 'git_state' is not defined`. No v4 downstream stage is unlocked.
- G0 v5 freeze job `12823` completed after the narrow NameError repair. Independent audit-only negative checks job `12825`, strict record validation job `12829`, formal publisher job `12830`, and post-publication verifier job `12831` all completed with the required PASS conditions. Synthetic-only G0 was then submitted exactly once as job `12833` and failed closed.
- Synthetic failure evidence: `artifacts/lb_scgp/v5/g0/synthetic/manifest.json` has `status=FAIL`, `thresholds_ok=true`, `expected_statuses_ok=false`, `dykstra_gate=false`, `rank_gate=true`, `farkas_gate=true`, `factor_gate=true`, `rollback_gate=true`, and `overflow_nan_inf_count=0`. Dykstra cases `feasible_interior`, `feasible_boundary`, and `feasible_oriented_boundary` returned `BOUNDED_SEARCH_FEASIBLE` instead of the frozen expected `LOCAL_STATIONARY_CERTIFIED`.
- G1 and all teacher stages are locked; `mllm_call_count=0`, `ocr_call_count=0`, teacher cache read/write count = 0.
- No segment-level gold exists or is assumed. G0/G1 use only parent-video binary labels; G2+ certificate fields remain weak train-only pseudo-signals.
- The global two-dataset, three-seed final `+0.030 accuracy / +0.030 macro-F1` goal remains active and unmet. This LB-SCGP v5 G0 branch is stopped at synthetic and supplies no performance evidence.
- v2 freeze audit note: `artifacts/lb_scgp/v2/CONFIG_FREEZE.json` excludes this tracker, `TARGET_LOOP.md`, `TARGET_STATE.json`, target findings, handoff notes and execution records from formal `input_files` and dirty-state predecessor checks. Its frozen immutable inputs were rehashed successfully immediately after freeze. This tracker update is an audit-trail record and is intentionally excluded from the v2 formal input set.

## 2026-07-11 G0 v5 Formal Code Audit Publication And Post-Verification

- Final review: `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.md`, SHA256 `495b5f3bc453034ae5f9830a77bc9b4a2b04af181b0d4365e95bdbaf450bd36b`; verdict PASS with Critical `0`, High `0`, Important `3`, and no-segment-gold PASS.
- Strict record: `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.record.json`, SHA256 `4cb399a0209025581cef094f9f339b6617d9a0ad1d22d5925c131107118a3770`; record payload SHA256 `b11589d4a892afa3f05982b29dd93276bbde20969804e4c853734f76e6be63c0`.
- Audit-only negative job `12825` completed `0:0` and verified fail-closed wrong hashes, dirty drift, run-ID/path/schema drift, v4 fallback attempts, prior-lineage hash drift, segment-gold drift, no-clobber and transaction residue cleanup.
- Publisher job `12830` completed `0:0` with payload SHA256 `46c3eece9f51b285749d7b70cab863be44eaf3acb930ca7c8c91a47353997016` and created only the formal files and locks under `artifacts/lb_scgp/v5/g0/code_audit/`.
- Post-publication verifier job `12831` completed `0:0` and really called producer `_load_freeze_and_audit` plus the strict decision-consumer verifier against the published v5 bundle. It records `producer_consumer_ok=true`, `decision_consumer_ok=true`, `all_ok=true`, and `dirty_equal_frozen=true` for frozen/current dirty hash `1c8284781fb57e90714b390fdbef362e978b70789632b19df3d8161dfe8827b7`.
- Supervision remains only `parent_video_binary_label`; `segment_gold_exists=false`, `segment_gold_used=false`, forbidden `query_z`/`query_labels` NPZ members were not opened, and MLLM/OCR/teacher/cache/held-label/held-content/val/test/formal-held counters remain `0`.
- Unlock state before numerical execution: only `LBSCGP-G0-SYNTH-v5` was unlocked. Realfold, replay, decision, G1, teacher, MLLM, OCR, held, val, test, and performance workloads remained locked.

## 2026-07-11 G0 v5 Synthetic Numerical Gate

- Exact command: `CONFIG=configs/lb_scgp/lb_scgp_v5.json TASK=synthetic RUN_ID=LBSCGP-G0-SYNTH-v5 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch`.
- SLURM job `12833`: `FAILED`, exit `2:0`, elapsed `00:00:44`, allocation `8 CPU / 64G`, MaxRSS `161708K`; no `--time`, no manual release/requeue/cancel.
- Log: `slurm/logs/lbscgp_g0_cpu_12833.out`, SHA256 `a8a249101ebf8ebe3ab56d5b152b8df35a8f593c2271e26e111b04364726ce49`.
- Artifact: `artifacts/lb_scgp/v5/g0/synthetic/manifest.json`, file SHA256 `07dc7d5d17194cd7a2b5d42d539adb9e8248e78b4dc629bbcdaf9d4f64719242`, payload SHA256 `751b5ede4cdd6f05032768b4c9295b56ba62fbe370be11436f7ca3f7dbec3fc5`.
- Manifest verdict: `status=FAIL`, `thresholds_ok=true`, `expected_statuses_ok=false`, `dykstra_gate=false`, `rank_gate=true`, `farkas_gate=true`, `factor_gate=true`, `rollback_gate=true`, `overflow_nan_inf_count=0`.
- Dykstra mismatches: `feasible_interior`, `feasible_boundary`, and `feasible_oriented_boundary` returned `BOUNDED_SEARCH_FEASIBLE` while frozen v5 expected `LOCAL_STATIONARY_CERTIFIED`.
- Access/supervision facts: only `parent_video_binary_label` is gold; `segment_gold_exists=false`, `segment_gold_used=false`; MLLM/OCR/teacher/cache/held-label/held-content/val/test counters remain `0`.
- Stopped stages: realfold, replay, decision, G1, teacher, MLLM, OCR, held/validation/test evaluation, and final performance training were not submitted. No replacement synthetic run was submitted.
- Dedicated execution record: `refine-logs/lb_scgp/G0_V5_NUMERICAL_EXECUTION.md`.
- Next authorization boundary: fresh repair/review authorization is required before any further LB-SCGP G0 work. This failed synthetic artifact does not unlock realfold, replay, decision, G1, or teacher stages.

## 2026-07-11 Round3 Repair Handoff Update

- Worker launcher metadata recorded: model `gpt-5.5`, reasoning `xhigh`, `--strict-config`.
- Implementation was patched and later path-fix reviewed at 0C/0H. The original Round3 review's only open Critical was physical C1.
- G0/G1 subclips are not inputs. No subclip/segment artifact is produced by the sanitizer, frozen, loaded by producer/replay, or passed to `compute_loss`.
- Formal G0 fails closed unless the train-only whole-video feature artifact, sanitized provenance and sanitizer decision exist and hash-match. Source locators/hashes live only in the quarantine source config/manifest, not in formal G0 freeze inputs.
- Handoff document: `refine-logs/lb_scgp/G0_IMPLEMENTATION_HANDOFF.md`.

## 2026-07-11 Sanitizer Job History

- `12737`: submitted with `sbatch --export=ALL,TASK=build scripts/slurm/lb_scgp_sanitize_inputs.sbatch`; failed before artifacts due to the relative-vs-absolute path bug; log SHA256 `80d071490eb9f799ed119dce04d9004aad30a5fd2189ccc6b2c09d7d4fa4b4d7`.
- `12738`: submitted with `sbatch --export=ALL,TASK=build scripts/slurm/lb_scgp_sanitize_inputs.sbatch`; completed with `status=SANITIZED`; log SHA256 `3314b3a7ea4f4b602cb28357258a5edf716cfa0abcd1bc8440d80fc0f222c978`.
- `12739`: submitted with `sbatch --export=ALL,TASK=verify scripts/slurm/lb_scgp_sanitize_inputs.sbatch`; completed with `status=PASS`; log SHA256 `40f333bb82884e9ad412fc87a20165e1640238815c3c6b6951d003e1cb0ec247`.
- At the end of the sanitizer-verification step, freeze/synthetic/realfold/replay/decision/G1/teacher/MLLM/OCR work had not run. Later freeze, audit, and synthetic execution are recorded above/below; synthetic eventually failed at job `12833`, no realfold/replay/decision/G1/teacher/MLLM/OCR work has run, no performance result exists, and G1/teacher remain locked.

## 2026-07-11 G0 Freeze Execution

- Exact command: `TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v1 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch`.
- SLURM job `12742`: `COMPLETED`, exit `0:0`, elapsed `00:00:03`, allocation `8 CPU / 64G`; no `--time`, no manual release/requeue/cancel.
- Output: `artifacts/lb_scgp/v1/CONFIG_FREEZE.json`, file SHA256 `b6697472b61a61706c694a67b21618d618fcad6e7f59265d8696aee79dc46889`; payload SHA256 `b3b33090b39b3b975c2cf213aab669041b345c6ef3a3f7c200366a506bcebfd5`; publish-lock SHA256 `34a05bde46775bcb75384c1c853cef7985f5c05efcdf3afec5fad6d85ef57d8d`.
- Log: `slurm/logs/lbscgp_g0_cpu_12742.out`, SHA256 `70cd194ed6a811be6956644e832055907f94750ccec6876352a8c1d6b5e98628`.
- Freeze artifact records `status=FROZEN`, `stage=G0_FREEZE`, `conda_env=HateVideo`, `only_gold_supervision=parent_video_binary_label`, `segment_gold_exists=false`, `segment_gold_used=false`, all teacher/MLLM/OCR/cache/held-label/held-content/val/test counters at `0`, `G1_G4_locked=true`, and held IDs as exclusion sentinels only.
- No segment/subclip artifact exists under `artifacts/lb_scgp`. No synthetic/realfold/replay/decision/G1/teacher/MLLM/OCR job was submitted.
- Explicit non-claims: no G0 PASS, no formal code audit, no numerical result, no performance result, no G1 unlock, and no teacher unlock.
- Dedicated execution record: `refine-logs/lb_scgp/G0_FREEZE_EXECUTION.md`.

## 2026-07-11 G0 v2 Repair And Freeze Execution

- Formal audit of v1: `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW.md` reported FAIL with C1 mutable freeze inputs and H1 sanitizer schema underspecification. No PASS audit artifacts were created, and no synthetic/realfold/replay/decision/G1/teacher stage was authorized from v1.
- Repair files: `configs/lb_scgp/lb_scgp_v2.json`, `refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json`, `refine-logs/lb_scgp/G0_V2_REPAIR_HANDOFF.md`, and code updates to config-driven v2 identities/freeze inputs. `refine-logs/lb_scgp/EXPERIMENT_PLAN.md` was amended before v2 freeze with the dedicated pre-freeze sanitizer schema and the rule that the full generic manifest/decision schema applies from `G0_FREEZE` onward.
- v2 immutable inputs exclude mutable records: this tracker, `TARGET_LOOP.md`, `TARGET_STATE.json`, `TARGET_FINDINGS.md`, `TARGET_REVIEW_RAW.md`, `G0_V2_REPAIR_HANDOFF.md`, and `G0_FREEZE_EXECUTION_V2.md`.
- Exact command: `CONFIG=configs/lb_scgp/lb_scgp_v2.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v2 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch`.
- SLURM job `12746`: initially `PENDING (JobHeldUser)`, automatically released, then `COMPLETED`, exit `0:0`, elapsed `00:00:03`, allocation `8 CPU / 64G`; no `--time`, no manual release/requeue/cancel.
- Output: `artifacts/lb_scgp/v2/CONFIG_FREEZE.json`, file SHA256 `4c6f0199a429bbf766cb284b9ecaedd9dcaf38733834dd54574245ab86b633ae`; payload SHA256 `bda18d7dfda00ab5808595afec04e5b925d3a4920f55fe963dec5e9d99795c0b`; publish-lock SHA256 `22426f693ba3f72d52928b0a86d08fe439d7e7fd43f9c74c61aedb710443e211`.
- Log: `slurm/logs/lbscgp_g0_cpu_12746.out`, SHA256 `e95d167af054b91582e1d1f8fbf66fb57a3b5cd67e298883940d385a65ccf563`.
- Freeze artifact records `status=FROZEN`, `lineage_version=v2`, `conda_env=HateVideo`, `config_canonical_sha256=3c7e391ca8e35fffa76ebcfc76a1c9c6e7c76c92bcc2dc08faa7e9a72be7cb1b`, `implementation_sha256=51fc1cee40f489e98e82c4aac93799015ac0ad7918e2847c6dbb7e0596890aef`, `independent_verifier_sha256=8ab99bad45daea1963dd030c24f91c28c87b46631d46e6fcafa4b3e3e102a4f6`, sanitizer provenance SHA256 `b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007`, sanitizer decision SHA256 `172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954`, and pre-freeze sanitizer contract snapshot SHA256 `cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b`.
- Supervision/counters: `only_gold_supervision=parent_video_binary_label`, `segment_gold_exists=false`, `segment_gold_used=false`; no segment/subclip artifact exists; MLLM/OCR/teacher/cache/held-label/held-content/val/test/formal-model-held counters all `0`.
- Frozen input rehash immediately after freeze: all 12 file rows and all 4 allowed bank NPZ members matched the hashes recorded in `CONFIG_FREEZE.json`.
- Explicit non-claims: no G0 PASS, no formal code audit, no synthetic, no realfold, no replay, no decision, no G1, no teacher/MLLM/OCR work, no performance result, and no unlock.
- Dedicated execution record: `refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V2.md`.
- Next gate: `LBSCGP-G0-CODE-AUDIT-v2`, a fresh independent formal code audit artifact with 0 Critical / 0 High. G1 and teacher remain locked.

## 2026-07-11 G0 v3 Repair And Freeze Execution

- Formal audit of v2: `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V2.md` reported FAIL with one Critical: `G0_FORMAL_CODE_AUDIT_REVIEW_V2.md` itself demonstrated that mandatory post-freeze audit records must be excluded from the dirty-state predecessor hash before a downstream decision gate can be safe. No v2 PASS audit artifacts were created.
- Repair files: `configs/lb_scgp/lb_scgp_v3.json`, `refine-logs/lb_scgp/G0_V3_REPAIR_HANDOFF.md`, `refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V3.md`, and narrow code updates to `lb_scgp_common.py`, `lb_scgp_g0.py`, `lb_scgp_independent_verify.py`, and `lb_scgp_g0_gpu.sbatch`.
- v3 immutable inputs exclude only the explicit mutable progress/audit records: this tracker, `TARGET_LOOP.md`, `TARGET_STATE.json`, `TARGET_FINDINGS.md`, `TARGET_REVIEW_RAW.md`, `G0_V3_REPAIR_HANDOFF.md`, `G0_FREEZE_EXECUTION_V3.md`, and future `G0_FORMAL_CODE_AUDIT_REVIEW_V3.md`.
- v3 formal artifact exclusion prefixes are exactly `artifacts/lb_scgp/v1/`, `artifacts/lb_scgp/v2/`, and `artifacts/lb_scgp/v3/`. The only mutable dirty prefix is `refine-logs/lb_scgp/runtime/`.
- Exact command: `CONFIG=configs/lb_scgp/lb_scgp_v3.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v3 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch`.
- SLURM job `12748`: `COMPLETED`, exit `0:0`, elapsed `00:00:03`, allocation `8 CPU / 64G`; no `--time`, no manual release/requeue/cancel.
- Output: `artifacts/lb_scgp/v3/CONFIG_FREEZE.json`, file SHA256 `9fba7f1649dd67d4bb0fcc193e555d8246d7a4966307732e87d5e9fca7346dd9`; payload SHA256 `352ec2215e2225b1768a13f39f96ef935b91966606d8db874d6de4410b1a9f3d`; publish-lock SHA256 `9c32d07c524e466ad06c06fc2a472829764cd1facff22390eac8b2879d329b8f`.
- Log: `slurm/logs/lbscgp_g0_cpu_12748.out`, SHA256 `f60d2301e7460ab91b25f6c323c578e49a00001b46c67a590f9f3c3d58abf545`.
- Freeze artifact records `status=FROZEN`, `lineage_version=v3`, `conda_env=HateVideo`, `dirty_diff_sha256=91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1`, `config_canonical_sha256=84227b68eaa496da6e307ce5c5ef3469e1b7c68e350f0d62d1677d01f07645bf`, `implementation_sha256=b8759436a6c5e2a67bf7125cbd1ab57cb05187e764e837373abfdf1a92916e75`, `independent_verifier_sha256=d1e50057b4c166a71426f89474b6526e3eab11da5547e15368112b6620dbf5ce`, sanitizer provenance SHA256 `b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007`, sanitizer decision SHA256 `172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954`, and pre-freeze sanitizer contract snapshot SHA256 `cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b`.
- Supervision/counters: `only_gold_supervision=parent_video_binary_label`, `segment_gold_exists=false`, `segment_gold_used=false`; no segment/subclip artifact exists; MLLM/OCR/teacher/cache/held-label/held-content/val/test/formal-model-held counters all `0`.
- Explicit non-claims: no G0 PASS, no formal code audit PASS, no synthetic, no realfold, no replay, no decision, no G1, no teacher/MLLM/OCR work, no performance result, and no unlock.
- Dedicated execution record: `refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V3.md`.
- Next gate: `LBSCGP-G0-CODE-AUDIT-v3`, a fresh independent formal code audit artifact with 0 Critical / 0 High. G1 and teacher remain locked.

## 2026-07-11 G0 v3 Formal Code Audit Review

- Review record: `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V3.md`.
- Verdict: `PASS_REVIEW_ONLY`; Critical `0`, High `0`, Important `2`; no-segment-gold audit `PASS`; exact v2 Critical closed for v3.
- Audit-only SLURM hash checks before report creation, after report creation, after report update, and after required documentation updates all matched frozen dirty hash `91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1`. Final successful recompute was job `12756`; wrapper attempts `12753`--`12755` failed before producing audit values and did not run any experiment stage.
- No `artifacts/lb_scgp/v3/g0/code_audit/{review.md,audit.json}` PASS artifact was created. Repository search found consumers for that artifact but no authorized schema/no-clobber producer task; hand-invented PASS JSON is forbidden.
- No synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, or performance job was submitted. No G0 success, performance result, G1 unlock, or teacher unlock is claimed.
- Next authorization boundary: supply or explicitly authorize a real no-clobber formal code-audit artifact producer, then re-verify schema/hash locks before any synthetic stage.

## 2026-07-11 G0 v4 Tooling-Lineage Repair

- v4 repairs only the v3 tooling gap: no authorized formal code-audit PASS producer.
- Added `configs/lb_scgp/lb_scgp_v4.json`, namespace `artifacts/lb_scgp/v4`, exact v4 run IDs, strict future review path `G0_FORMAL_CODE_AUDIT_REVIEW_V4.md`, and strict future machine record path `G0_FORMAL_CODE_AUDIT_REVIEW_V4.record.json`.
- Added dedicated later wrapper `scripts/slurm/lb_scgp_g0_audit_publish.sbatch` for `TASK=audit-publish`, 2 CPU / 4G, no `--time`, `conda activate HateVideo`, offline flags, and strict `CONFIG/RUN_ID/REVIEW/REVIEW_RECORD` checks.
- Added `audit-publish` to the independent verifier. It must recompute v4 freeze/config/implementation/verifier/dirty hashes, rehash all frozen inputs and allowed NPZ members without opening `query_z` or `query_labels`, verify v1-v3 freeze/lock hashes, enforce zero forbidden counters and no-segment-gold, validate the strict review record, and atomically publish `review.md`, `review_record.json`, `audit.json`, `publication_index.json`, and persistent locks only after every check passes.
- Producer `_load_freeze_and_audit` and the final decision verifier now consume the same strict v4 code-audit schema and reject missing/additional/drifted fields.
- Exact freeze command: `CONFIG=configs/lb_scgp/lb_scgp_v4.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v4 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch`.
- SLURM job `12759`: initially `PENDING (JobHeldUser)`, automatically released, then `COMPLETED`, exit `0:0`, elapsed `00:00:03`, allocation `8 CPU / 64G`; no `--time`, no manual release/requeue/cancel.
- Output: `artifacts/lb_scgp/v4/CONFIG_FREEZE.json`, file SHA256 `dcf65eceba04e7c4f08145b2012653705f7347c6e96ebc8b2b769280dff48fd0`; payload SHA256 `92301ad95870b8a1af41e7e69e45054a2e63fe80a021cf4c1c22906aea0872bf`; publish-lock SHA256 `09003ce9e741d7c0310045f854479deb8fecff74bfddd33f6b9d80dc6df9572a`; log SHA256 `6203cab3eded38f22638980c4828020a10ddbd8421819a0d5f5059eca6faa6da`.
- Freeze-bound hashes: `dirty_diff_sha256=8ca10aec315f800959e7869beb200f4bbc5f5d27841d8c307d896f1644803e7a`, `config_canonical_sha256=9e99cba37486e2511b0e37fb7d2c3b59053fbac8aca577ba05b36c138aa67c56`, `implementation_sha256=c7e9371494f991d88a7ab93cc64769fa1e6a92913df3afd2f647201d0eef1bf1`, `independent_verifier_sha256=03a78a89867d3cea468b5319463ccabcefa4b4a589a61863bffd3e14c9df5402`, `access_ledger_sha256=ef67ad3b6521a9b8e9b73dd27260917c531e4ec72a84e04c52afed0c34ba72a7`.
- At v4 freeze time, no v4 formal PASS artifact, no v4 formal review, and no v4 review machine record was created in this repair. A later v4 audit publication produced internally consistent formal files, but publication verification failed operationally because the producer consumer raised `NameError: name 'git_state' is not defined`. No synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, held-label/content, val/test, or performance task was run or unlocked from v4.

## 2026-07-11 G0 v5 NameError Repair And Freeze Execution

- v5 repairs only the missing `git_state` import in the producer strict schema verifier plus v4/v5 lineage/path/schema/no-clobber support in the publisher, producer, and decision consumer. Scientific method, thresholds, supervision, data protocol, numerical logic, and evaluation behavior are unchanged.
- Config and namespace: `configs/lb_scgp/lb_scgp_v5.json`, `artifacts/lb_scgp/v5`, future review sidecar `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.md`, future record sidecar `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.record.json`.
- Focused audit-only regression job `12820` completed under SLURM with result `PASS`, proving producer `_load_freeze_and_audit` consumes a strict v5 schema and fails closed for wrong review-record hash, wrong dirty hash, wrong code-audit run ID, and wrong review/publication path. Runtime fixtures were removed and no formal v5 namespace was created by regression.
- Exact freeze command: `CONFIG=configs/lb_scgp/lb_scgp_v5.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v5 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch`.
- SLURM job `12823`: initially `PENDING (JobHeldUser)`, automatically released, then `COMPLETED`, exit `0:0`, elapsed `00:00:02`, allocation `8 CPU / 64G`; no `--time`, no manual release/requeue/cancel.
- Output: `artifacts/lb_scgp/v5/CONFIG_FREEZE.json`, file SHA256 `254e45afe9c0355892824c0c26bc73b4b0854cb20c67c3703982762fad010931`; payload SHA256 `d89f7cd4ad43c8ef83a04b7530ec19186ec1cb0e96958d239f1ce5c2b146bb4d`; publish-lock SHA256 `54dc06d236d5fc1f3ac96400f1a81faeb1d2c0c8e5af075065d1964260de98a9`.
- Log: `slurm/logs/lbscgp_g0_cpu_12823.out`, SHA256 `c43a2b16fc8c95bdfafb5c48c674fa4b778dd9c3d3c3764e24fe50ed038a0526`.
- Freeze-bound hashes: `dirty_diff_sha256=1c8284781fb57e90714b390fdbef362e978b70789632b19df3d8161dfe8827b7`, `config_canonical_sha256=4a45fb6c66884b6b8aa4571961dff3ef7751c2b9f97e2df1584521cfe1eb3dba`, `implementation_sha256=939acffbafbd9204fc654972cd73f174393c8466c61f4af045b1c20948a6b687`, `independent_verifier_sha256=f0f49f41de4efee9abf2267b27b75be440f0020583baef565955c3d0c2988b2d`, `access_ledger_sha256=ce6898035cf25dbe53f4b258a7b792796e4388a282384185f13a84926397ea0f`.
- Supervision/counters: `only_gold_supervision=parent_video_binary_label`, `segment_gold_exists=false`, `segment_gold_used=false`, `G1_G4_locked=true`, outer-held labels/content not opened, protected storage not read, and MLLM/OCR/teacher/cache/held-label/held-content/val/test/formal-model-held counters all `0`.
- Explicit non-claims: no G0 PASS, no v5 formal audit PASS, no v5 review/record, no audit-publish, no synthetic, no realfold, no replay, no decision, no G1, no teacher/MLLM/OCR work, no performance result, and no unlock.
- Dedicated execution record: `refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V5.md`.
- Next boundary: fresh independent GPT-5.5 xhigh v5 audit only.

## 2026-07-11 Path-Normalization Repair

- Repair scope: shared ROOT-relative path canonicalization for relative or absolute config/artifact inputs, plus static hardening of reachable analogous LB-SCGP path surfaces.
- Handoff record: `refine-logs/lb_scgp/G0_SANITIZER_PATH_FIX.md`.
- Review record: `refine-logs/lb_scgp/G0_SANITIZER_PATH_FIX_REVIEW.md`, final verdict PASS, 0 Critical / 0 High.
- No segment path/objective was reintroduced; `lambda_seg=0` and `segment_cache=None` remain binding.

## 2026-07-11 Physical Sanitizer Review

- Review record: `refine-logs/lb_scgp/G0_SANITIZER_PHYSICAL_REVIEW.md`.
- Open Critical: `0`.
- Open High: `0`.
- C1 status: closed at artifact level only.
- Explicit non-claims: no G0 PASS, no G0 freeze, no performance result, no G1 unlock, no teacher unlock.
