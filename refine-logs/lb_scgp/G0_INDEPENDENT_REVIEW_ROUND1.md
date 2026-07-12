**Verdict: FAIL**

CRITICAL findings: 5  
HIGH findings: 2  
Ready for freeze/SLURM: no.

**Critical Findings**

1. Missing physical train-only artifacts block G0.
Evidence: [lb_scgp_v1.json:21](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:21), [lb_scgp_v1.json:50](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:50), [lb_scgp_g0.py:1321](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1321). The required feature/subclip hashes are `null`, and `task_freeze` fails before publish when they are missing. I also found no `artifacts/lb_scgp/inputs/MHC_zh/fold4` files.
Why it invalidates a gate: freeze cannot pass; real fit cannot be safely sourced.
Minimal fix: STOP until physically outer-train-only artifacts are generated from a separated train-only source with hashes frozen. Do not extract from the mixed caches.

2. Real Dykstra/rank-cell verification is not independently replayable.
Evidence: producer stores real cell summaries without per-cell result/trace at [lb_scgp_g0.py:1196](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1196), emits only cycle summaries at [lb_scgp_g0.py:1920](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1920), and verifier only checks rank-search hashes/statuses at [lb_scgp_independent_verify.py:1027](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1027) plus `len(projectors)==cycles` at [lb_scgp_independent_verify.py:1154](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1154).
Why it invalidates a gate: the real numerical/KKT/persistent-correction/adjacent-cell gate is not genuinely recomputed.
Minimal fix: emit full real per-set/per-cycle state sufficient for replay, or implement a separate independent real Dykstra/rank-cell solver that recomputes every candidate cell and selected objective.

3. Realized fit and rollback gates trust producer fields and do not prove actual checkpoint rollback.
Evidence: verifier reads `realized_bank` from producer JSON at [lb_scgp_independent_verify.py:1119](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1119) and its fit gate only checks displacement/residual/rollback boolean at [lb_scgp_independent_verify.py:1159](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1159). Producer creates fresh optimizer/scheduler/scaler inside the epoch at [lb_scgp_g0.py:1753](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1753), but actual rollback snapshots only model/RNG, not a live optimizer/scheduler/scaler, at [lb_scgp_g0.py:1794](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1794), while claiming AdamW/scheduler/scaler restored at [lb_scgp_g0.py:1819](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1819).
Why it invalidates a gate: the actual checkpoint/model/epoch fit and rollback gate is not closed.
Minimal fix: snapshot and restore the real continuation optimizer/scheduler/scaler/cursors, and have an independent replay verify batch order, fit steps, realized bank hash, and rollback hash.

4. H10 cost formula is inconsistent with the registered plan.
Evidence: plan requires `1.25 * 10 * [2*H_REMOVE + 5*p95(...) + H_final_bank]` at [EXPERIMENT_PLAN.md:77](/data/jehc223/RGCL/refine-logs/lb_scgp/EXPERIMENT_PLAN.md:77). Producer instead computes `contingency*folds*refreshes*(2*remove_fullfold_seconds+per_refresh_seconds)/3600` at [lb_scgp_g0.py:1905](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1905), and verifier enforces the same shortened formula at [lb_scgp_independent_verify.py:1121](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1121).
Why it invalidates a gate: resource/cost approval can pass a non-registered accounting rule.
Minimal fix: implement the exact registered H10 formula, including p95 semantics and final-bank placement, then independently recompute it.

5. Farkas/non-reweighting audit is partial and not independently dual-verified for the real cone.
Evidence: producer’s NNLS matrix uses only pair/singleton/SupCon generated columns at [lb_scgp_g0.py:1257](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1257) while only counting triplets at [lb_scgp_g0.py:1261](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1261). Verifier repeats the same NNLS construction at [lb_scgp_independent_verify.py:1057](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1057) and compares witness equality at [lb_scgp_independent_verify.py:1078](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1078).
Why it invalidates a gate: it does not prove separation from the full registered cone by an independent dual/oracle solve.
Minimal fix: define the full registered cone explicitly, use a true independent separation oracle/dual solve, and report feasibility against all registered singleton/pair/triplet/SupCon columns.

**High Findings**

1. Formal provenance still depends on a manifest carrying mixed-cache lineage.
Evidence: config forbids combined caches at [lb_scgp_v1.json:23](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:23), but SSR manifest records mixed source cache paths/hashes at [manifest.json:25](/data/jehc223/RGCL/artifacts/ssr/v1/oof/MHC_zh/fold4/manifest.json:25), and freeze hashes that manifest at [lb_scgp_g0.py:1334](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1334).
Why it matters: this does not read mixed bytes now, but it keeps mixed-cache hashes in the formal input surface.
Minimal fix: freeze a sanitized train-only provenance manifest that excludes mixed-cache locator/hash fields.

2. Resource gate relies partly on self-report.
Evidence: producer hardcodes `"one_gpu": True` at [lb_scgp_g0.py:1955](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1955), while verifier checks only peak memory thresholds at [lb_scgp_independent_verify.py:1174](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1174).
Why it matters: one-GPU/resource accounting is not independently established.
Minimal fix: record and verify SLURM allocation, `CUDA_VISIBLE_DEVICES`, GPU name/count, and peak stats from runtime metadata.

**No-Segment-Gold Audit**

The written contract is consistent: config says only parent-video binary labels and no segment gold at [lb_scgp_v1.json:10](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:10); the problem anchor repeats it at [PROBLEM_ANCHOR.md:6](/data/jehc223/RGCL/refine-logs/lb_scgp/PROBLEM_ANCHOR.md:6). Code enforces subclip labels as inherited parent labels at [lb_scgp_g0.py:1684](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1684). I found no segment/timestamp/span/localization/stance/target/mechanism/rationale gold interface in the G0 code.

**Data-Access Audit**

Formal code avoids fold JSON reads, whole-bank `.npz` hashing, `query_z`, `query_labels`, and combined cache loads. It reads `.npz` allowed members plus `query_ids` as sentinel only at [lb_scgp_g0.py:1837](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1837). The known mixed CLIP caches are ignored by `.gitignore`, but the safe train-only artifacts do not currently exist. Opening the SSR manifest does not expose held labels/content directly, but it does expose mixed-cache lineage and should be sanitized.

**Safe Artifact Path**

No safe path is established from existing separated sources in the current handoff. STOP. The only concrete current source named for those tensors is the mixed train/subclip cache lineage, and extracting from those is forbidden. A valid next path requires a new audited generator that reads only an outer-train ID allowlist plus a physically train-only raw/content source, then writes the two PT artifacts and freezes their hashes.

**Recommended Next Action**

Do not submit freeze, synthetic, realfold, or decision jobs. First fix the five critical gates, create/sanitize the physically train-only input provenance, and re-review before any SLURM execution.