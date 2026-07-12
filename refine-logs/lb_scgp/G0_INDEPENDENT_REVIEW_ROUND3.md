# LB-SCGP G0 Independent Review Round 3

**Launcher / Sole Reviewer / Non-Actions**

Launcher metadata supplied: `GPT-5.5`, `model_reasoning_effort=xhigh`.  
Sole reviewer: no subagent, sidecar, dynamic workflow, network, Python/import execution, data/model execution, SLURM, teacher/MLLM/OCR, edits, or artifact/cache creation.  
Static inspection only: `rg`, `sed`, `jq empty`, `bash -n`. `jq empty` passed for both LB-SCGP configs and `TARGET_STATE.json`; `bash -n` passed for the three LB-SCGP sbatch wrappers.

**Supersession note:** the review below is preserved verbatim as the pre-sanitizer Round3 review. Its sole open Critical was physical C1 before artifacts existed. Subsequent sanitizer build job `12738`, verifier job `12739`, and `refine-logs/lb_scgp/G0_SANITIZER_PHYSICAL_REVIEW.md` close C1 at artifact level only. G0 PASS is still not claimed, and G1/teacher remain locked.

**Round2 Closure Table**

| Round2 item | Status | Evidence |
|---|---:|---|
| CRITICAL: no-segment closure end-to-end | CLOSED | Formal contract says only parent-video binary labels and no segment gold at [lb_scgp_v1.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:11). G0 returns `segment=None` at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1915), sets `lambda_seg=0.0` at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1939), asserts no segment objective at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1942), and calls `compute_loss(..., segment_cache=None)` at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1998). `compute_loss` only enters segment loss if `lambda_seg > 0 and segment_cache is not None` at [loss.py](/data/jehc223/RGCL/src/model/loss.py:556). Replay enforces the same at [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:188). |
| CRITICAL: physical train-only sanitizer output and non-null binding | OPEN | Code is fail-closed, but the physical artifact is absent now: `rg --files artifacts/lb_scgp` returns “No such file or directory”. Config hash slots remain null at [lb_scgp_v1.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:45). Tracker says no sanitizer/G0/replay/decision has run at [EXPERIMENT_TRACKER.md](/data/jehc223/RGCL/refine-logs/lb_scgp/EXPERIMENT_TRACKER.md:4). Freeze can bind sanitizer hashes at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1546) and fails missing/stale artifacts at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1568), but current state is not physically closed. |
| CRITICAL: real Dykstra/rank-cell replay independence | CLOSED | Producer records correction-state and projector transition hashes at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1067), [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1089), and [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1183). Adjacent cells carry objective, target, trace, transition, and final correction hashes at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1223) and [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1243). Verifier independently reconstructs cells/traces/objectives at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1304) through [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1378). |
| HIGH: formal config mixed/fold legacy lineage | CLOSED | Formal config no longer has legacy/mixed/subclip slots; it names sanitized whole-video outputs at [lb_scgp_v1.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:20) and allowed/forbidden bank members at [lb_scgp_v1.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:51). Mixed source lineage is only in the quarantine source config, marked `formal_g0_input=false`, at [lb_scgp_sanitizer_sources.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_sanitizer_sources.json:4). |
| HIGH: sanitizer decision can carry quarantine locator | CLOSED | Sanitizer verifier reads the quarantine manifest but rejects it as formal G0 input at [lb_scgp_verify_sanitizer.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_verify_sanitizer.py:80) and emits a decision containing sanitized paths/hashes only at [lb_scgp_verify_sanitizer.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_verify_sanitizer.py:162). G0 recursively scans provenance and decision at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1510). |
| HIGH: real verifier forbidden-path scan no-op | CLOSED | Verifier has populated protected prefixes at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:31), recursive scanners at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:114), and real manifest/access-ledger scanning at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1579). |
| MEDIUM: Farkas definition pinning | CLOSED | Producer pins the registered singleton/pair/triplet/SupCon definition at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1355); verifier pins the same definition at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1429) and gates reported definition/universe/family errors at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1636). |

**New Findings**

CRITICAL:
1. **Physical sanitizer output is still absent, so G0 cannot be certified GO.**  
   Failure path: a reviewer could treat Round3 code repair as equivalent to a physical train-only artifact, but no `artifacts/lb_scgp` tree exists, config hash slots are still null, and tracker records no sanitizer/G0 run. The code fails closed, but the gate condition is not satisfied.

HIGH: none found.

MEDIUM: none found.

LOW:
1. The formal `g0/code_audit/audit.json` artifact is not created by this read-only review; later pipeline stages require it before synthetic execution at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1609). This is a process/DAG requirement, not a code defect in the reviewed repair.

**Explicit Gate Verdicts**

| Gate | Verdict |
|---|---|
| no-segment-gold | PASS: no segment cache/objective path remains live in G0/replay. |
| isolation | PASS for code; physical output still pending. |
| sanitizer/hash binding | FAIL: binding logic exists, but no physical sanitizer output exists now. |
| Dykstra replay | PASS: selected and adjacent cells are independently replayed with trace/transition/correction hashes. |
| fit/rollback | PASS for code: actual fit and separate replay are live and segment-disabled. |
| Farkas | PASS: pinned definition plus singleton/pair/triplet/SupCon oracle checks. |
| H10/resource | PASS statically: exact formula and one-GPU resource gates are implemented; sbatch constraints are valid. |
| DAG | FAIL for current state: G1 remains locked, but G0 cannot proceed without sanitizer artifacts and formal audit artifact. |

**Open Counts**

Open Critical: **1**  
Open High: **0**

**Final Verdict: FAIL**

PASS requires 0 Critical / 0 High. At this review time, the sole blocker was physical, not a new leakage route: the sanitizer and G0 artifacts had not been produced, so C1 remained open.

**Exact Next Actions**

1. Run the pre-freeze sanitizer build under SLURM, then sanitizer verifier under SLURM.
2. Re-check that `outer_train_features.pt`, `sanitized_provenance.json`, and `sanitizer_decision.json` exist, are non-clobbered, and hash-bind.
3. Only after that, create the formal code-audit artifact if this review is accepted, then proceed to freeze/synthetic/realfold/replay/decision in the registered DAG.
