# LB-SCGP G0 v5 Numerical Gate Execution Record

**Date:** 2026-07-11
**Executor identity:** fresh LB-SCGP G0 v5 numerical-gate executor. The externally visible thread/session ID is not exposed to this process.
**Scope:** v5 synthetic numerical correctness/executability gate only. This is not final accuracy or macro-F1 evidence.

## Frozen Inputs Read Before Execution

- Config: `configs/lb_scgp/lb_scgp_v5.json`
- Freeze: `artifacts/lb_scgp/v5/CONFIG_FREEZE.json`
- Freeze lock SHA256: `54dc06d236d5fc1f3ac96400f1a81faeb1d2c0c8e5af075065d1964260de98a9`
- Code-audit bundle: `artifacts/lb_scgp/v5/g0/code_audit/`
- Post-publication verifier: `refine-logs/lb_scgp/runtime/v5_independent_audit/post_publication_verification_12831.json`
- Producer: `scripts/analysis/lb_scgp_g0.py`
- Independent verifier: `scripts/analysis/lb_scgp_independent_verify.py`
- Replay code: `scripts/analysis/lb_scgp_real_replay.py`
- CPU wrapper: `scripts/slurm/lb_scgp_g0_cpu.sbatch`
- GPU wrapper: `scripts/slurm/lb_scgp_g0_gpu.sbatch`

Frozen run IDs:

```text
freeze   LBSCGP-G0-FREEZE-v5
audit    LBSCGP-G0-CODE-AUDIT-v5
synth    LBSCGP-G0-SYNTH-v5
real     LBSCGP-G0-REAL-MHC_zh-F4-S0-v5
replay   LBSCGP-G0-REAL-REPLAY-MHC_zh-F4-S0-v5
decision LBSCGP-G0-DECISION-v5
```

Post-publication verifier job `12831` recorded `producer_consumer_ok=true`, `decision_consumer_ok=true`, `all_ok=true`, `dirty_equal_frozen=true`, zero forbidden counters, and no-segment-gold OK. This unlocked **synthetic only**. Realfold, replay, decision, G1, teacher, MLLM, OCR, held, validation, test, and performance work remained locked.

## Synthetic Submission

Exact command:

```text
CONFIG=configs/lb_scgp/lb_scgp_v5.json TASK=synthetic RUN_ID=LBSCGP-G0-SYNTH-v5 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
```

No `--time` was set. No manual release, requeue, cancel, or bypass was used.

SLURM:

```text
job_id=12833
job_name=lbscgp_g0_cpu
state=FAILED
exit=2:0
elapsed=00:00:44
start=2026-07-11T21:36:21
end=2026-07-11T21:37:05
alloc=8 CPU / 64G
MaxRSS=161708K
```

Log:

```text
slurm/logs/lbscgp_g0_cpu_12833.out
sha256=a8a249101ebf8ebe3ab56d5b152b8df35a8f593c2271e26e111b04364726ce49
```

Terminal log line:

```text
{"expected_statuses_ok":false,"run_id":"LBSCGP-G0-SYNTH-v5","status":"FAIL","thresholds_ok":true}
```

## Synthetic Artifact Set

The producer created the expected synthetic files and persistent locks under `artifacts/lb_scgp/v5/g0/synthetic/`. Because the manifest status is `FAIL`, these are failure evidence only and do not unlock any later stage.

```text
cases.jsonl                         8d619f3579886fc9e6a515c5c6c1215a09f4bc628a89ba5e418c18887d27b698
cases.jsonl.publish.lock            f97fa43cf254437bb42bd682a9e8bbeb3f588b703ea5496f203d91657fd8415b
dykstra.jsonl                       c28090bbd26da0d6ba89ca67340355c8cebc6e24e20899954282dfeed02a92f6
dykstra.jsonl.publish.lock          f97fa43cf254437bb42bd682a9e8bbeb3f588b703ea5496f203d91657fd8415b
exact_vote.jsonl                    bc0c1f4b5823f153c49a9cfb1efb7de531a7a7550890c553810f468b8d00690b
exact_vote.jsonl.publish.lock       f97fa43cf254437bb42bd682a9e8bbeb3f588b703ea5496f203d91657fd8415b
factor.jsonl                        75df4129030cd528bee20f0de2c750898a4eeaf90fd49af5698b2b64cc0af6dc
factor.jsonl.publish.lock           f97fa43cf254437bb42bd682a9e8bbeb3f588b703ea5496f203d91657fd8415b
farkas.jsonl                        b7abd457d44ffcc07b701590d9302e293019842e298ece4aaa70611823c30e90
farkas.jsonl.publish.lock           f97fa43cf254437bb42bd682a9e8bbeb3f588b703ea5496f203d91657fd8415b
manifest.json                       07dc7d5d17194cd7a2b5d42d539adb9e8248e78b4dc629bbcdaf9d4f64719242
manifest.json.publish.lock          f97fa43cf254437bb42bd682a9e8bbeb3f588b703ea5496f203d91657fd8415b
projectors.jsonl                    ad5e26c63125c8d5db78ed254a8d1b6112735eaa72a39bc0287d8eb45fc79091
projectors.jsonl.publish.lock       f97fa43cf254437bb42bd682a9e8bbeb3f588b703ea5496f203d91657fd8415b
rank_cells.jsonl                    aad6d4d573b18043059476049279ba5919084b6b220d56cafaf6924efb215a98
rank_cells.jsonl.publish.lock       f97fa43cf254437bb42bd682a9e8bbeb3f588b703ea5496f203d91657fd8415b
```

Manifest evidence:

```text
status=FAIL
stage=G0_SYNTHETIC
payload_sha256=751b5ede4cdd6f05032768b4c9295b56ba62fbe370be11436f7ca3f7dbec3fc5
thresholds_ok=true
expected_statuses_ok=false
dykstra_gate=false
rank_gate=true
farkas_gate=true
factor_gate=true
rollback_gate=true
overflow_nan_inf_count=0
dirty_diff_sha256=1c8284781fb57e90714b390fdbef362e978b70789632b19df3d8161dfe8827b7
implementation_sha256=939acffbafbd9204fc654972cd73f174393c8466c61f4af045b1c20948a6b687
config_canonical_sha256=4a45fb6c66884b6b8aa4571961dff3ef7751c2b9f97e2df1584521cfe1eb3dba
independent_verifier_sha256=f0f49f41de4efee9abf2267b27b75be440f0020583baef565955c3d0c2988b2d
access_ledger_sha256=5786035f51f7fe78a8d31e9dfdffc78bce7661c2d840af2c112bda07eab2af99
```

## Numerical Failure Evidence

All projectors reported `PASS`. Rank-cell/exact-vote reported the expected `PASS`/`REMOVE` outcomes. Farkas, factor, and rollback gates reported `PASS`. The failing surface is Dykstra expected-status parity:

```text
case=feasible_interior
status=BOUNDED_SEARCH_FEASIBLE
expected=LOCAL_STATIONARY_CERTIFIED
cycles=1
max_cycles=5
max_set_violation=3.1086244689504383e-15
relative_iterate_change=6.704483993371942e-16
rank_cell_stable=true
search_reason=unresolved_cell
independent_orientations=0
adjacent_cells_total=1
adjacent_cells_checked=1
pivots=0

case=feasible_boundary
status=BOUNDED_SEARCH_FEASIBLE
expected=LOCAL_STATIONARY_CERTIFIED
cycles=84
max_cycles=500
max_set_violation=9.771865633061666e-7
relative_iterate_change=7.032126641120063e-9
rank_cell_stable=false
search_reason=unresolved_cell
independent_orientations=0
adjacent_cells_total=1
adjacent_cells_checked=1
pivots=0

case=feasible_oriented_boundary
status=BOUNDED_SEARCH_FEASIBLE
expected=LOCAL_STATIONARY_CERTIFIED
cycles=500
max_cycles=500
max_set_violation=6.752627108152793e-6
relative_iterate_change=5.1725509722744595e-8
rank_cell_stable=false
search_reason=base_cell_not_converged
independent_orientations=1
adjacent_cells_total=1
adjacent_cells_checked=1
pivots=0
```

The frozen manifest therefore fails closed. No tuning, hot-fix, alternate same-stage run, or later-stage submission was performed.

## Read-Only Failure Attribution

This is a mixed implementation/certification defect and numerical-certificate failure, not merely an incorrect expected-status fixture:

- `feasible_interior` is not a numerical non-convergence. The base projection meets both frozen tolerances after one cycle and the emitted top-20 `rank_cell_stable` value is `true`. However, the zero-orientation branch in `_rank_search_controller` compares `stable_rankings(..., topk=n-1)` (23 entries for this 24-item fixture) against `fixture["rankings"]`, which `_product_fixture` / `_refresh_fixture_rank_fields` constructs with `topk=20`. That length-inconsistent comparison forces `unresolved_cell` even when the emitted top-20 comparison passes.
- `feasible_boundary` is affected by the same 23-entry versus 20-entry certification comparison. Its projection meets both scalar tolerances, but the independently emitted top-20 `rank_cell_stable=false` means this case cannot be attributed only to that comparison defect.
- `feasible_oriented_boundary` is a genuine frozen-budget numerical/certification failure: it exhausts `500/500` cycles and has `max_set_violation=6.752627108152793e-6`, above the frozen `1e-6` tolerance. Its relative change is within tolerance, but the joint certificate is not achieved.

Therefore the branch STOP remains correct. The available evidence does not justify changing the frozen expected statuses, thresholds, cycle budgets, or implementation in this execution. This attribution is also not accuracy or macro-F1 evidence.

## Access And Supervision Facts

Manifest and v5 freeze/audit evidence retain:

```text
only_gold_supervision=parent_video_binary_label
segment_gold_exists=false
segment_gold_used=false
mllm_call_count=0
ocr_call_count=0
teacher_cache_read_count=0
teacher_cache_write_count=0
outer_held_label_read_count=0
outer_held_content_read_count=0
val_content_read_count=0
test_content_read_count=0
val_test_teacher_artifact_count=0
```

The synthetic task did not open `query_z`, `query_labels`, held content, held labels, validation content, test content, teacher artifacts, MLLM, OCR, or segment artifacts. `query_ids` remained only an exclusion-sentinel concept in the frozen predecessor evidence.

## Stopped Stages

The following stages were **not submitted** because synthetic did not PASS:

```text
LBSCGP-G0-REAL-MHC_zh-F4-S0-v5
LBSCGP-G0-REAL-REPLAY-MHC_zh-F4-S0-v5
LBSCGP-G0-DECISION-v5
G1/G2/G3/G4
teacher / MLLM / OCR
held / validation / test evaluation
final performance training
```

## Protected Evidence Rehash

End-of-run rehash of protected v1-v5 freeze/formal evidence matched the frozen/indexed hashes, including:

```text
v1 CONFIG_FREEZE.json                                      b6697472b61a61706c694a67b21618d618fcad6e7f59265d8696aee79dc46889
v1 CONFIG_FREEZE.json.publish.lock                         34a05bde46775bcb75384c1c853cef7985f5c05efcdf3afec5fad6d85ef57d8d
v2 CONFIG_FREEZE.json                                      4c6f0199a429bbf766cb284b9ecaedd9dcaf38733834dd54574245ab86b633ae
v2 CONFIG_FREEZE.json.publish.lock                         22426f693ba3f72d52928b0a86d08fe439d7e7fd43f9c74c61aedb710443e211
v3 CONFIG_FREEZE.json                                      9fba7f1649dd67d4bb0fcc193e555d8246d7a4966307732e87d5e9fca7346dd9
v3 CONFIG_FREEZE.json.publish.lock                         9c32d07c524e466ad06c06fc2a472829764cd1facff22390eac8b2879d329b8f
v4 CONFIG_FREEZE.json                                      dcf65eceba04e7c4f08145b2012653705f7347c6e96ebc8b2b769280dff48fd0
v4 CONFIG_FREEZE.json.publish.lock                         09003ce9e741d7c0310045f854479deb8fecff74bfddd33f6b9d80dc6df9572a
v4 code_audit/audit.json                                   098f5e02b9df17c14a5b43df3ffe15e7fc8e45638b89f6e9d5f3ab84239b290f
v4 code_audit/publication_index.json                       b3b5c86250dd9675c1bfac0d63adbe8b45ea734f459495586a13f4296ab8604b
v4 code_audit/review.md                                    616bd9093334b08c067208c7de7a2d54649b19be131d8f8a49a2ce48b84bb53d
v4 code_audit/review_record.json                           b2e8ef2ac9c5d3fede5ace5f678e65a5d1c0f6a565356b19fb6143e186a44f5e
v5 CONFIG_FREEZE.json                                      254e45afe9c0355892824c0c26bc73b4b0854cb20c67c3703982762fad010931
v5 CONFIG_FREEZE.json.publish.lock                         54dc06d236d5fc1f3ac96400f1a81faeb1d2c0c8e5af075065d1964260de98a9
v5 code_audit/audit.json                                   6aeaeae22bd52d13ade2d178c068b527cf3c67a6e424c944ab175a175e5799c8
v5 code_audit/publication_index.json                       0b29b1425d49bfe75331b5e392620dd4472d4329d8a824749ace3c6ab4a2a517
v5 code_audit/review.md                                    495b5f3bc453034ae5f9830a77bc9b4a2b04af181b0d4365e95bdbaf450bd36b
v5 code_audit/review_record.json                           4cb399a0209025581cef094f9f339b6617d9a0ad1d22d5925c131107118a3770
```

## Final Boundary

Final v5 G0 numerical outcome for this execution: **STOP at synthetic**.

Exact next authorization boundary: a fresh repair/review authorization is required before any further LB-SCGP G0 work. Re-running synthetic, running realfold, replay, decision, G1, teacher, MLLM, OCR, held/validation/test evaluation, or performance training is not authorized from this failed synthetic artifact.
