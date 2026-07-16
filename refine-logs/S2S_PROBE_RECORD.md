# S2S Stage-P G0-cond probe — RAW RESULTS record (Modal CPU triage)

**Author:** s2s-implementer (executor). **Date:** 2026-07-16. **Repo HEAD at run:** `cc3d90e`.

RAW numbers only — **NO pass/fail interpretation by the executor**. The mechanical gate arithmetic
below is the probe's own pre-registered threshold arithmetic, reproduced verbatim and explicitly **NOT a
verdict**; the independent verdict reviewer renders the binding ruling (house rule; prereg §6.6, review
§5.6/N6). **These are CLOUD triage numbers (~1.4pt cross-hardware drift) — triage-only, NEVER mixed into
a local paper table** (CLAUDE.md cloud policy; G-repro is local-only).

---

## 1. Provenance

| item | value |
|---|---|
| route | Modal cloud CPU (`scripts/cloud/modal_probe_runner.py::run`, CPU function) |
| Modal app (REAL run) | **`ap-iE3OEkUra7f37UuzDk2uz3`** (rgcl-probe, CPU, `--detach`) |
| launch / duration | ~2026-07-16T03:37Z → COMPLETED returncode 0, **265.0s** probing time |
| frozen probe | `scripts/analysis/s2s_probe.py` sha256 **`141a0441845d6175646d642a57b4534f78a48d96521ef3dc3a2d9fcf0f2301b3`** — verified in-container by the adapter's hash-gate (== frozen r4, design §10) |
| path-adapter shim | `scripts/analysis/s2s_probe_cloud_adapter.py` sha256 **`48e5b3752206b2e59a9f55d37a707c07386dc7ac9bdeb872ecff2cf679965e5e`** (PLUMBING-ONLY: hash-gate + symlink `/data/jehc223/RGCL/{data,src}`→`/root/{data,src}` + `runpy` the frozen probe UNCHANGED; mirrors approved W2-A adapter `fb609d4b…`) |
| runner | `scripts/cloud/modal_probe_runner.py` sha256 `fe49d86a7b27feff97fdabede89a22443e934551c68246a5648683637f5ce045` (W2-A timeout-edited; `MODAL_PROBE_TIMEOUT=7200` set locally for this run) |
| in-container invocation | `s2s_probe.py --datasets HateMM,MHC --frameset_dir frameset_qwen7b_8f --n_boot 1000 --n_perframe_null 100 --out_md /root/data/S2S_PROBE_RESULTS.md --out_json /root/data/s2s_probe_results.json` |
| prereg config confirmed in-container | `topk=20`, `null_seeds=100` (N1 0..99), `n_boot=1000`, `n_perframe_null=100` |
| N4 fail-closed (0 test-touch) | PASS — `HateMM memory N=851 (train∪val)`, `MHC memory N=629 (train∪val)`; test_seen never constructed |
| probe self-test | `synthetic shared-segment: MMS 0.2594 > POOLED 0.2479 OK` |
| synced datasets | HateMM (2196 files/504 MB) + MHC (1665 files/466 MB) → volume `rgcl-features` (all video-guard-passed; derived .pt + label json only) |
| retrieved raw artifacts | `refine-logs/s2s_probe_results_cloud.json` sha256 `9998ba00…05dd0f`; `refine-logs/S2S_PROBE_RESULTS_cloud.md` sha256 `22d7cec7…9ec8074` (via `modal volume get`, committed on the run's clean exit) |
| dry-run (THROWAWAY, DISCARDED) | app `ap-9arL8gpDuMYdDZNpo5E1Mk`, `DRYRUN 20` (n_boot=20, n_perframe_null=0), 52.1s — plumbing validation only; its numbers are **not evidence** and appear nowhere below |
| sync apps | `ap-UYJt6fCMnpT6iJ0j7FXk48` (HateMM), `ap-F29p2fl3htnrNTIrCTORbS` (MHC) |

---

## 2. RAW per-arm results (verbatim from the probe's committed output)

### HateMM (memory N=851, T=4, zero-guard rows=1)

| arm | acc | macro_f1 | roc |
|---|---|---|---|
| POOLED | 0.7662 | 0.7552 | 0.8257 |
| SET | 0.7697 | 0.7555 | 0.8297 |
| SET_CHAMFER | 0.7697 | 0.7572 | 0.8317 |
| PIPELINE_ANCHOR | 0.7673 | 0.7568 | 0.8259 |
| WITH_TEXT_POOLED | 0.8073 | 0.8049 | 0.8904 |
| WITH_TEXT_SET | 0.8073 | 0.8040 | 0.8897 |
| POOLED_RANKONLY | 0.7673 | 0.7568 | 0.8253 |
| SET_RANKONLY | 0.7744 | 0.7611 | 0.8285 |
| ASYM | 0.7603 | 0.7464 | 0.8215 |
| POOLED_NEARDUP_EXCL | 0.7662 | 0.7549 | 0.8229 |
| SET_NEARDUP_EXCL | 0.7744 | 0.7613 | 0.8274 |

- **Primary paired Δ(SET−POOLED):** acc **+0.0035**, macro_f1 **+0.0003**.
- **Rank-only (A2):** acc +0.0071, macro_f1 +0.0042; obs Δacc +0.0071 vs rank-only null-95th +0.0188, rank-only bootstrap-5th −0.0071 (corroborates=False: sign=True null=False boot=False).
- **C2 ASYM (pooled-query × set-memory):** acc 0.7603, macro_f1 0.7464. Δ(ASYM−SET): acc −0.0094, mF1 −0.0091 (beats_set=False); obs Δacc −0.0094 vs null-95th +0.0212, bootstrap-5th −0.0223.
- **Fano (±1 gold-label key) acc:** 1.0000.
- **Oracle ceiling (A4):** acc 0.8578 (Δ vs POOLED acc **+0.0917**, mF1 +0.0953).
- **Near-dup (A3):** flagged pairs (≥0.995 pooled-OR-MMS) = 120; excluded-retrieval Δ(SET−POOLED) acc +0.0082, mF1 +0.0064. Dist: pooled≥0.980=211, mms≥0.980=144, maxframe≥0.980=187; pooled≥0.990=146, mms≥0.990=116, maxframe≥0.990=128; pooled≥0.995=120, mms≥0.995=109, maxframe≥0.995=116.
- **Permutation null (N1, 100 seeds):** obs Δacc +0.0035 vs null-95th +0.0189; obs ΔmF1 +0.0003 vs null-95th +0.0298.
- **Per-frame null (optional, 100 seeds):** Δacc-95th −0.1832, ΔmF1-95th −0.2480.
- **Bootstrap (1000):** Δacc [5/50/95]=[−0.0106/+0.0035/+0.0165]; ΔmF1 [5/50/95]=[−0.0145/+0.0009/+0.0146].
- **Stage-E gates surfaced:** train decomp_max=5.96e-08 grecon_cos_min=0.9999995232 grecon_maxabs_max=0.0; dev_seen decomp_max=5.96e-08 grecon_cos_min=0.9999997020 grecon_maxabs_max=0.0.

### MHC (memory N=629, T=4, zero-guard rows=0)

| arm | acc | macro_f1 | roc |
|---|---|---|---|
| POOLED | 0.7027 | 0.5694 | 0.6601 |
| SET | 0.6630 | 0.5463 | 0.6736 |
| SET_CHAMFER | 0.6900 | 0.5682 | 0.6828 |
| PIPELINE_ANCHOR | 0.7027 | 0.5671 | 0.6614 |
| WITH_TEXT_POOLED | 0.7695 | 0.7193 | 0.8227 |
| WITH_TEXT_SET | 0.7615 | 0.7258 | 0.8322 |
| POOLED_RANKONLY | 0.7027 | 0.5694 | 0.6595 |
| SET_RANKONLY | 0.6630 | 0.5463 | 0.6720 |
| ASYM | 0.6773 | 0.5607 | 0.6547 |
| POOLED_NEARDUP_EXCL | 0.7027 | 0.5694 | 0.6587 |
| SET_NEARDUP_EXCL | 0.6630 | 0.5463 | 0.6721 |

- **Primary paired Δ(SET−POOLED):** acc **−0.0397**, macro_f1 **−0.0231**.
- **Rank-only (A2):** acc −0.0397, macro_f1 −0.0231; obs Δacc −0.0397 vs rank-only null-95th +0.0145, rank-only bootstrap-5th −0.0636 (corroborates=False: sign=True null=False boot=False).
- **C2 ASYM (pooled-query × set-memory):** acc 0.6773, macro_f1 0.5607. Δ(ASYM−SET): acc +0.0143, mF1 +0.0145 (beats_set=True); obs Δacc +0.0143 vs null-95th +0.0159, bootstrap-5th −0.0079.
- **Fano (±1 gold-label key) acc:** 1.0000.
- **Oracle ceiling (A4):** acc 0.8426 (Δ vs POOLED acc **+0.1399**, mF1 +0.2104).
- **Near-dup (A3):** flagged pairs (≥0.995 pooled-OR-MMS) = 2; excluded-retrieval Δ(SET−POOLED) acc −0.0397, mF1 −0.0231. Dist: pooled≥0.980=9, mms≥0.980=5, maxframe≥0.980=8; pooled≥0.990=5, mms≥0.990=1, maxframe≥0.990=2; pooled≥0.995=2, mms≥0.995=0, maxframe≥0.995=0.
- **Permutation null (N1, 100 seeds):** obs Δacc −0.0397 vs null-95th +0.0130; obs ΔmF1 −0.0231 vs null-95th +0.0267.
- **Per-frame null (optional, 100 seeds):** Δacc-95th −0.0127, ΔmF1-95th −0.0935.
- **Bootstrap (1000):** Δacc [5/50/95]=[−0.0636/−0.0397/−0.0175]; ΔmF1 [5/50/95]=[−0.0576/−0.0237/+0.0100].
- **Stage-E gates surfaced:** train decomp_max=5.96e-08 grecon_cos_min=0.9999995232 grecon_maxabs_max=0.0; dev_seen decomp_max=5.96e-08 grecon_cos_min=0.9999997020 grecon_maxabs_max=0.0.

---

## 3. Mechanical gate arithmetic (pre-registered thresholds — reproduced verbatim; NOT a verdict)

Reproduced from the probe's `mechanical_gate_check` block. The executor renders **no** pass/fail
judgment; the independent verdict reviewer rules (and must cross-check the local Stage-E K0–K2 gates from
the extraction record `cc3d90e` before honoring any downstream escalation).

| gate | value | threshold | op | result |
|---|---|---|---|---|
| Fano[HateMM] | 1.0 | 0.99 | >= | ABOVE |
| Fano[MHC] | 1.0 | 0.99 | >= | ABOVE |
| OracleDacc[HateMM] | 0.0916568742655699 | 0.04 | >= | ABOVE |
| OracleDacc[MHC] | 0.13990461049284575 | 0.04 | >= | ABOVE |
| OracleKillSwitch(all-datasets) | False | all < 0.04 |  | SURVIVES |
| RawDacc[HateMM] | 0.003525264394829586 | 0.05 | >= | BELOW |
| RawDmF1[HateMM] | 0.00028856783156994137 | 0.05 | >= | BELOW |
| RankOnlyCorroborates[HateMM] (A2) | False | True |  | BELOW |
| RankOnlyObsDacc>null95[HateMM] (A2) | 0.007050528789659283 | 0.018801410105757872 | > | BELOW |
| RankOnlyBoot5th>0[HateMM] (A2) | -0.007050528789659172 | 0.0 | > | BELOW |
| ObsDacc>null95[HateMM] | 0.003525264394829586 | 0.018860164512338465 | > | BELOW |
| Bootstrap5th>0[HateMM] | -0.010575793184488763 | 0.0 | > | BELOW |
| NearDupExclSurvives[HateMM] (A3) | 0.008225616921269108 | 0.0 | > | ABOVE |
| C2 ASYM beats SET (acc AND mF1) [HateMM] | False | True |  | BELOW |
| C2 ASYM beats SET (acc AND mF1) [MHC] | True | True |  | ABOVE |
| C2 route adjudication (ASYM beats SET on >=1 dataset) | True | True |  | SURVIVES(escalate-to-§11-asym) |

Full machine JSON (all arms, roc, null/bootstrap percentiles, c2_asym fields): `refine-logs/s2s_probe_results_cloud.json`.
Probe's own raw markdown: `refine-logs/S2S_PROBE_RESULTS_cloud.md`.

---

## 4. Notes (non-interpretive)

- Zero test-touch: N4 fail-closed guard confirmed memory = train∪val exactly (851 / 629) in-container;
  no `test_seen` file was constructed or opened.
- Cloud triage only: these numbers carry ~1.4pt cross-hardware drift and are **not** to be placed in any
  local/paper table; a formal stage (if ever authorized) re-runs locally under G-repro.
- 16-frame arm remains HELD (ratified) — not run.
- Executor stops here: this record is raw evidence for the independent verdict reviewer; no downstream
  head-training or further GPU is authorized by this run.
