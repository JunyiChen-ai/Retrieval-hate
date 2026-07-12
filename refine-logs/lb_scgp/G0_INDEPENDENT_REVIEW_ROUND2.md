# G0_INDEPENDENT_REVIEW_ROUND2

**Launcher / Reviewer Declaration**

Launcher-specified metadata: `gpt-5.5`, `model_reasoning_effort=xhigh`. This is recorded as supplied metadata; no runtime model introspection was performed.  
Sole reviewer: no subagent, sidecar, dynamic workflow, network, teacher, MLLM, OCR, SLURM job, Python execution, file edit, or artifact creation was used. Static shell reads/searches only; `jq empty` and `bash -n` passed.

**Verdict: FAIL**

Open Critical: **3**  
Open High: **3**  
PASS requires 0 Critical / 0 High; this review is not 0C/0H.

**Round1 Closure Table**

| ID | Status | Evidence |
|---|---:|---|
| C1 train-only physical artifacts | OPEN | Config still has null authoritative artifact hashes at [lb_scgp_v1.json:62](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:62)-[65](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:65); tracker says sanitizer/G0 not run and freeze blocked pending sanitizer outputs at [EXPERIMENT_TRACKER.md:42](/data/jehc223/RGCL/refine-logs/lb_scgp/EXPERIMENT_TRACKER.md:42)-[43](/data/jehc223/RGCL/refine-logs/lb_scgp/EXPERIMENT_TRACKER.md:43). Code can bind non-null hashes from sanitizer decision at [lb_scgp_g0.py:1522](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1522)-[1534](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1534), but no physical closure exists now. |
| C2 real Dykstra/rank-cell replay | OPEN | Adjacent cells are recomputed, but selected real projector evidence is still barely checked: verifier only requires `len(projectors)==cycles` at [lb_scgp_independent_verify.py:1450](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1450)-[1456](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1456), and per-cell `trace_sha256` is only required non-null, not matched to independent trace, at [lb_scgp_independent_verify.py:1221](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1221)-[1226](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1226). |
| C3 actual fit/rollback | CLOSED for rollback mechanics | Live model/AdamW/scheduler/scaler/RNG/cursor are snapshotted at [lb_scgp_g0.py:2019](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2019)-[2025](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2025), restored at [lb_scgp_g0.py:2031](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2031)-[2036](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2036), and checked against direct REMOVE at [lb_scgp_g0.py:2040](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2040)-[2058](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2058). Separate GPU replay exists at [lb_scgp_real_replay.py:379](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:379)-[456](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:456). New segment-loss issue below invalidates fit semantics, not rollback mechanics. |
| C4 H10 | CLOSED | Registered formula is in config at [lb_scgp_v1.json:163](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:163)-[169](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:169), producer computes final-bank-outside-refresh H10 at [lb_scgp_g0.py:2148](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2148)-[2154](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2154), verifier recomputes it at [lb_scgp_independent_verify.py:1417](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1417)-[1423](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1423) and gates it at [1490](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1490)-[1495](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1495). |
| C5 Farkas/cone | CLOSED with medium caveat | Producer registers singleton/pair/triplet/SupCon definition at [lb_scgp_g0.py:1409](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1409)-[1415](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1415), implements pair/triplet oracles at [1275](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1275)-[1308](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1308), and verifier independently regenerates/gates them at [lb_scgp_independent_verify.py:1266](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1266)-[1373](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1373), [1472](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1472)-[1483](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1483). |
| H1 formal mixed lineage | OPEN | Formal config still contains `legacy_mixed_bank_sha256_not_formal_input` at [lb_scgp_v1.json:55](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:55), despite formal zero-count claims at [66](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:66)-[68](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:68). Freeze hashes the whole config as a formal input at [lb_scgp_g0.py:1495](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1495)-[1496](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1496). |
| H2 resource self-report | CLOSED | GPU wrapper requests one A100 and 8 CPU / 64 GB at [lb_scgp_g0_gpu.sbatch:72](/data/jehc223/RGCL/scripts/slurm/lb_scgp_g0_gpu.sbatch:72)-[74](/data/jehc223/RGCL/scripts/slurm/lb_scgp_g0_gpu.sbatch:74); runtime records CUDA/SLURM device evidence at [lb_scgp_g0.py:2204](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2204)-[2221](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2221); verifier requires one visible CUDA device and GPU name at [lb_scgp_independent_verify.py:1484](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1484)-[1489](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1489). |

**New Findings**

1. **CRITICAL: G0 actual fit/replay uses segment-level loss from inherited subclip labels.**  
Failure path: docs forbid segment loss and segment gold at [FINAL_PROPOSAL.md:7](/data/jehc223/RGCL/refine-logs/lb_scgp/FINAL_PROPOSAL.md:7)-[8](/data/jehc223/RGCL/refine-logs/lb_scgp/FINAL_PROPOSAL.md:8), but G0 sets `lambda_seg=0.5`, `seg_mode=milmax` at [lb_scgp_g0.py:1921](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1921)-[1927](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1927), builds a `segment` object with inherited labels at [1897](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1897)-[1899](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1899), and passes it into `compute_loss` at [1978](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1978)-[1980](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1980). The called loss adds segment loss when `lambda_seg > 0` at [loss.py:553](/data/jehc223/RGCL/src/model/loss.py:553)-[563](/data/jehc223/RGCL/src/model/loss.py:563) and mines segment positives/negatives using `sc_label` at [loss.py:800](/data/jehc223/RGCL/src/model/loss.py:800)-[806](/data/jehc223/RGCL/src/model/loss.py:806). This is an implicit segment-supervision route.

2. **HIGH: formal config still binds mixed/fold legacy hashes.**  
Failure path: the formal config carries legacy fold and mixed-bank hashes at [lb_scgp_v1.json:53](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:53)-[55](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:55); freeze includes the config hash as formal input at [lb_scgp_g0.py:1495](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1495)-[1496](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1496). The scanner only catches `source_*` keys or `data/CLIP_Embedding/` strings at [lb_scgp_g0.py:1434](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1434)-[1447](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1447), so this mixed hash surface is not rejected.

3. **HIGH: formal sanitizer decision can carry a quarantine-manifest locator.**  
Failure path: verifier records the quarantine manifest path into its ledger at [lb_scgp_verify_sanitizer.py:98](/data/jehc223/RGCL/scripts/analysis/lb_scgp_verify_sanitizer.py:98)-[99](/data/jehc223/RGCL/scripts/analysis/lb_scgp_verify_sanitizer.py:99), embeds `access_ledger` in the formal decision at [193](/data/jehc223/RGCL/scripts/analysis/lb_scgp_verify_sanitizer.py:193)-[194](/data/jehc223/RGCL/scripts/analysis/lb_scgp_verify_sanitizer.py:194), and formal G0 reads/freezes that decision at [lb_scgp_g0.py:1456](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1456)-[1457](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1457), [1500](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1500)-[1501](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1501). The mixed-locator scanner in the sanitizer verifier does not reject `artifacts/lb_scgp/quarantine/` strings at [lb_scgp_verify_sanitizer.py:58](/data/jehc223/RGCL/scripts/analysis/lb_scgp_verify_sanitizer.py:58)-[72](/data/jehc223/RGCL/scripts/analysis/lb_scgp_verify_sanitizer.py:72).

4. **HIGH: real verifier’s forbidden-path intersection is a no-op.**  
Failure path: `forbidden_paths=set()` is empty at [lb_scgp_independent_verify.py:1424](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1424), so the manifest/access checks at [1432](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1432)-[1436](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1436) cannot catch mixed/quarantine/protected paths if they appear.

5. **MEDIUM: real cone verifier hashes the reported definition text rather than an independently fixed registered definition.**  
Failure path: verifier sets `definition=reported.get(...)` at [lb_scgp_independent_verify.py:1362](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1362)-[1363](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1363). Universe counts and family maxima are recomputed, but the definition text itself is not independently pinned.

**Explicit Gate Verdicts**

| Gate | Verdict |
|---|---|
| no-segment-gold | **FAIL**: enabled segment loss uses inherited subclip labels. |
| isolation | **FAIL**: sanitizer selection is mostly mechanical, but formal config/decision still expose mixed/quarantine hash/path surfaces. |
| numerical replay | **FAIL**: rank-cell target replay improved, but real persistent projector/correction evidence remains under-verified. |
| Farkas | **PASS with MEDIUM caveat**: full families/oracles exist; definition pinning should be hardened. |
| H10 | **PASS**: exact registered formula and p95 semantics are implemented and independently recomputed. |
| rollback | **PASS for rollback mechanics**: live state and separate GPU replay are implemented. |
| resource | **PASS**: SLURM/HateVideo/no-`--time`/one-GPU checks are present. |
| DAG | **FAIL**: missing artifacts fail closed, but the formal audit cannot unlock G1 while no-segment and isolation High/Critical issues remain. |

**Next Actions**

1. Disable all segment-loss and segment-cache paths for LB-SCGP G0/G1, or prove the called loss is a strict no-op; inherited subclip labels must not enter a segment objective.
2. Remove legacy mixed/fold whole-artifact hashes from formal config and replace them with allowed-member or sanitized train-only hashes only.
3. Strip quarantine-manifest paths from the formal sanitizer decision, or make formal G0 reject any `artifacts/lb_scgp/quarantine/` locator in formal inputs.
4. Populate the real verifier forbidden-path denylist and fail on mixed/quarantine/protected paths in manifests and access ledgers.
5. Bind real Dykstra projector traces/correction evidence to independent recomputation, not merely count cycles or require non-null trace hashes.
6. After code fixes, rerun independent review before any sanitizer, G0, G1, or teacher stage.