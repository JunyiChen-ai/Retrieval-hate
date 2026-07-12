# LB-SCGP G0 Round3 Fix Handoff

**Date:** 2026-07-11  
**Launcher metadata:** model `gpt-5.5`, `model_reasoning_effort=xhigh`.  
**Worker mode:** sole implementation worker; no subagent, sidecar, dynamic workflow, network, teacher, MLLM, OCR, SLURM job, Python execution/import/test, data processing, or artifact/cache creation.  
**Status:** Original Round3 repairs were prepared for independent review, then the path-normalization repair review passed at 0 Critical / 0 High. Later sanitizer build `12738` and verifier `12739` completed; physical C1 is closed at artifact level only. This is not G0 PASS and does not unlock G1 or teacher.

## Closure Mapping

1. **Critical A, no segment gold/objective:** closed for implementation handoff by removing formal subclip artifacts from config/freeze/sanitizer/replay, enforcing `lambda_seg=0`, returning `segment=None`, and passing `segment_cache=None` into `compute_loss`. Fail-closed assertions remain in producer, replay and verifier. Inherited parent labels are documented as not segment gold.
2. **Critical B, C1 physical train-only input:** sanitizer now emits only `outer_train_features.pt` plus sanitized provenance/decision. Formal config no longer carries subclip or quarantine manifest paths. Freeze binds the non-null whole-video feature hash from the sanitizer decision and fails closed on missing/stale output.
3. **Critical C, C2 real Dykstra/rank-cell independence:** producer emits selected per-cycle projector transition hashes and persistent correction-state hashes; adjacent-cell ledgers carry trace/transition/final-correction hashes. Independent verifier reconstructs selected and adjacent traces without producer imports and matches trace hashes, objectives, selected cell, rank/tie/vote invariance and exact-vote evidence.
4. **High D1, formal mixed/fold lineage:** legacy fold and mixed whole-artifact hashes/locators and formal subclip locator/hash slots are removed from `configs/lb_scgp/lb_scgp_v1.json`. Formal config allows only allowed bank-member hashes and sanitized train-only output hash slots.
5. **High D2/D3, sanitizer formal decision and forbidden paths:** sanitizer decision contains no quarantine manifest locator/hash and no source access ledger. Shared and independent recursive scanners reject quarantine/mixed/protected locators and prohibited source/mixed/legacy hash surfaces. Real verifier denylist is populated and scans manifest `input_files`/`access_ledger` recursively, including resolved paths where feasible.
6. **Medium E, Farkas definition hardening:** producer and verifier pin the registered singleton/pair/triplet/SupCon cone definition from config and fail on drift while preserving full oracle checks.

## No-Segment Audit

- `configs/lb_scgp/lb_scgp_v1.json` has no `outer_train_subclip_cache` path or hash slot.
- `configs/lb_scgp/lb_scgp_sanitizer_sources.json` has no mixed subclip source.
- `scripts/analysis/lb_scgp_sanitize_inputs.py` writes only the whole-video feature cache; manifest records no subclip source/output.
- `scripts/analysis/lb_scgp_verify_sanitizer.py` verifies no segment artifact/objective and writes no subclip path/hash.
- `scripts/analysis/lb_scgp_g0.py` sets `lambda_seg=0`, rejects non-null segment cache, and passes `segment_cache=None`.
- `scripts/analysis/lb_scgp_real_replay.py` independently enforces the same no-segment contract.
- Docs/state explicitly say G0/G1 subclips are not inputs and inherited parent labels are not segment gold.

## Static Validations

Allowed checks run:

```text
jq empty configs/lb_scgp/lb_scgp_v1.json configs/lb_scgp/lb_scgp_sanitizer_sources.json TARGET_STATE.json
bash -n scripts/slurm/lb_scgp_sanitize_inputs.sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch scripts/slurm/lb_scgp_g0_gpu.sbatch
rg stale no-segment callsites: only intentional denylist/fail-closed hits remained
rg formal legacy/mixed/subclip surfaces: only intentional denylist/fail-closed hits remained
sha256sum changed-file ledger below
```

Not run: Python syntax/import tests, sanitizer, sanitizer verifier, freeze, synthetic, realfold, replay, decision, training, evaluation, data processing, `sbatch`, `srun`, teacher/MLLM/OCR/network calls, or formal artifact/cache writes.

## Process Note

Round1 and Round2 independent review files are preserved verbatim. The first aborted Round2 review attempt was rejected because it spawned sidecars without xhigh proof. The canonical Round2 report was produced by a sole explicit GPT-5.5 xhigh reviewer with no delegation. This is process provenance only, not a scientific finding.

## SHA256 Ledger

```text
e6aaf5d66399cdbbe7fcc2c811931277b0ed4a24b592ffa5cbb60315b29ea23c  AGENTS.md
82e32bfa9fa552744106fd8a5b9c2e07de8e55ab0ad29c35cb5a6907ca744b43  configs/lb_scgp/lb_scgp_v1.json
ceb1676bfe81ad62373911458570a681d05b632beebe3b5b757c9c7e83eb5aae  configs/lb_scgp/lb_scgp_sanitizer_sources.json
94c43d0c37f6e13d6b5cdbd277cbf72b951a3f03d01550c0a72f5f4738499a70  scripts/analysis/lb_scgp_common.py
5801d609a9227fd058e76dbe55f8c9fbb2af1e676954cd40c068b460e4b16573  scripts/analysis/lb_scgp_sanitize_inputs.py
2a581ae8c7ef54534dca4ff1ee8f4d34b436763cff91ba7a5f0b5629bf9f42d1  scripts/analysis/lb_scgp_verify_sanitizer.py
97c4913be719c9ebcb67e1da5be80c4f05adb54a0afe313b21ab4d956d39e796  scripts/analysis/lb_scgp_g0.py
6f881f2e2a2e3b039e78586e4455b9e221cae3905f192f726f18bddab9e7f40a  scripts/analysis/lb_scgp_real_replay.py
2cc7a73f7af795f9d35b98772d397ba65645cd418d035339e06d667267dbbf49  scripts/analysis/lb_scgp_independent_verify.py
350b9b946470c9afbe4fcbbfb4b742d9f24a984e2c983089276d2ec5f48fff17  scripts/slurm/lb_scgp_sanitize_inputs.sbatch
8722f7d71070b9711a45aa4074c128b516e08e53c556940660ab1ac980229241  scripts/slurm/lb_scgp_g0_cpu.sbatch
932f14edebf115d5f307ed2438530eef8f774e2fadb90b98b136eecdfa56ff04  scripts/slurm/lb_scgp_g0_gpu.sbatch
3bdce884e907ed865c8048d4e31fc01bd4dab1de4bd38fe34c307eb346afd49c  refine-logs/lb_scgp/EXPERIMENT_PLAN.md
b2db41090938d1e8b6a6b7cbbc3d7cdfa911f2867e0774da48bf66d6861f139c  refine-logs/lb_scgp/EXPERIMENT_TRACKER.md
68d24df979c36d2fb36d5dfa598d3174f3fac3058e042d2f53dfbe7f1158d99a  refine-logs/lb_scgp/G0_IMPLEMENTATION_HANDOFF.md
94d7b6e9305e8c6095e0a9f20351bb4cafff042f7aa048f2a934ac6d1a3a0a0c  refine-logs/lb_scgp/FINAL_PROPOSAL.md
254f4c68fdf578239b952222f4e72378b8a110df21e85158c32b3cbc3b90200d  refine-logs/lb_scgp/PROBLEM_ANCHOR.md
c692fe13615ac167fa1b7583b38627c4519fadd897ce436df1bc910b3a62f350  TARGET_FINDINGS.md
d21c8563db46c5eb8b1162c08f4c2d08287ae43f960d9351360572694bd3ed47  TARGET_LOOP.md
94900876036ddfad0f6b10e3bdbac130d244b9d5ca1511a156d451fb248448d3  TARGET_STATE.json
```

This handoff omits its own hash to avoid self-reference; compute it after write.

## 2026-07-11 Post-Handoff Sanitizer Evidence

Preserve the original `12737` failure: build job `12737` failed before artifacts with log SHA256 `80d071490eb9f799ed119dce04d9004aad30a5fd2189ccc6b2c09d7d4fa4b4d7`.

The fresh build and verifier are now complete:

```text
12738|lbscgp_sanitize|COMPLETED|0:0|00:00:05|sbatch --export=ALL,TASK=build scripts/slurm/lb_scgp_sanitize_inputs.sbatch
12739|lbscgp_sanitize|COMPLETED|0:0|00:00:05|sbatch --export=ALL,TASK=verify scripts/slurm/lb_scgp_sanitize_inputs.sbatch
```

Physical evidence:

```text
ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496  artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt
b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007  artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json
055dffed9b61053293741ec5ba0ce3577daf458ceef4a3f81143a81d937c684b  artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json
40f333bb82884e9ad412fc87a20165e1640238815c3c6b6951d003e1cb0ec247  slurm/logs/lbscgp_sanitize_12739.out
172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954  artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json
8685c805995f97ad0513658c1345b6320dc794e2bad02b907d9a2d07ff16cb1f  sanitizer_decision payload_sha256
```

The verifier decision records `memory_id_count=464`, `query_id_sentinel_count=115`, status `PASS`, all gates true, no segment artifact/objective, zero teacher/MLLM/OCR/network calls, and zero formal query reads. No `segment` or `subclip` artifact exists under `artifacts/lb_scgp`.

Physical artifact review: `refine-logs/lb_scgp/G0_SANITIZER_PHYSICAL_REVIEW.md` reports 0 Critical / 0 High and closes remaining C1 at artifact level only.

Next registered gate is G0 freeze/formal audit preparation. G0 PASS is not claimed; G1 and teacher remain locked.
