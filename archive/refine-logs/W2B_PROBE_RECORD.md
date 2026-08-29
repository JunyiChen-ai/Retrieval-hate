# W2-B Probe — RAW EXECUTION RECORD (Modal CPU, features-only)

**Status:** executed 2026-07-15; **RAW numbers only — NO pass/fail interpretation. The independent verdict
reviewer renders the binding ruling** (house rule). The `mechanical_gate_check` block below is
pre-registered threshold arithmetic quoted verbatim and is **NOT the binding verdict**.

## Provenance
- **Design (r1, hash-frozen):** `refine-logs/W2B_PROBE_DESIGN.md` (B1–B3 + N1–N5 folded);
  `refine-logs/W2B_FORENSIC_RECON.md`; `refine-logs/W2B_PREREG_REVIEW.md` (APPROVED-WITH-AMENDMENTS; §8
  conditional authorization; §10 code re-check → CLOUD EXECUTION CLEARED).
- **Probe script:** `scripts/analysis/w2b_probe.py`, **sha256 `d22aac02b4c50f2952e1aa06b4609dd158d69ff54dd184cd9885fec1d3a15776`** (unchanged pre/post run).
- **Platform:** Modal, app **`ap-qRhIPZPGASmeO9JZVuJmMQ`** (ephemeral, stopped), function `run_probe_cpu`
  (CPU, no GPU), image pinned to the HateVideo env (torch 2.6.0 / transformers 4.49.0 / faiss-cpu / numpy
  1.26.4). Volume **`rgcl-features`** mounted at `/root/data`; features-only.
- **Runner plumbing patch (this session):** `scripts/cloud/modal_probe_runner.py` `_execute` gained a
  post-subprocess `features.commit()` so `/root/data` output writes persist to the volume for
  `modal volume get`. Plumbing only — probe logic and its sha256 unchanged.
- **Invocation:** `modal run scripts/cloud/modal_probe_runner.py::run --script scripts/analysis/w2b_probe.py
  --args "--data_root /root/data --datasets HateMM,MHC --k30_sensitivity 0 --mm_sensitivity 1
  --n_perframe_null 30 --out_md /root/data/W2B_PROBE_RESULTS.md --out_json /root/data/w2b_probe_results.json"`.
- **Outputs retrieved via** `modal volume get rgcl-features /w2b_probe_results.json` and `/W2B_PROBE_RESULTS.md`.
- **Config (from `meta`):** topk 20; null_seeds 100 (0..99); n_boot 1000; expected_mem_primary
  {HateMM 851, MHC 629} (**N4/B1 video-count guards PASSED**); raw_bar 0.05; mhc_survival_bar 0.03;
  oracle_bar 0.04; fano_bar 0.99; near_dup_threshold 0.995. Deterministic (`CUDA_VISIBLE_DEVICES=""`,
  torch/np seed 20260714, bootstrap seed 20260714).
- **TERM-1:** outputs wired to the volume root and committed (retrieved above).
- **TERM-2:** `--k30_sensitivity 0` on cloud — **the K30 granularity sensitivity (HateMM train-only 744) was
  NOT run on cloud; it is DEFERRED to a local run** (non-survival-determining breadth-modifier, B2).
- **`--n_perframe_null 30`:** the OPTIONAL, non-gating per-sub-clip shuffle null was run at 30 seeds
  (reduced from the default 100) to bound the single authorized run; all GATING statistics
  (permutation null 0..99, bootstrap 1000) are at pre-registered defaults.
- **Cost:** CPU minutes on Modal; ~$0 (within free credits).

---

## HateMM — K4 PRIMARY (memory V=851, K=4, zero-guard=1)

| arm | acc | macro_f1 | roc |
|---|---|---|---|
| POOLED | 0.7568 | 0.7518 | 0.8301 |
| SET | 0.7521 | 0.7441 | 0.8285 |
| SET_CHAMFER | 0.7532 | 0.7438 | 0.8279 |
| ASYM | 0.7509 | 0.7419 | 0.8273 |
| PIPELINE_ANCHOR | 0.7532 | 0.7485 | 0.8375 |
| WITH_TEXT_POOLED | 0.7756 | 0.7742 | 0.8750 |
| WITH_TEXT_SET | 0.7814 | 0.7787 | 0.8707 |
| POOLED_RANKONLY | 0.7532 | 0.7485 | 0.8318 |
| SET_RANKONLY | 0.7532 | 0.7452 | 0.8303 |
| POOLED_NEARDUP_EXCL | 0.7626 | 0.7582 | 0.8276 |
| SET_NEARDUP_EXCL | 0.7485 | 0.7401 | 0.8265 |

- **Primary paired Δ(SET−POOLED):** acc **-0.0047**, macro_f1 **-0.0077**.
- **Rank-only (A2):** acc +0.0000, macro_f1 -0.0033; obs Δacc +0.0000 vs rank-only null-95th +0.0213,
  rank-only bootstrap-5th -0.0153 (**corroborates=False**).
- **ASYM (pooled-query × set-memory):** acc 0.7509, macro_f1 0.7419; Δ(ASYM−SET) acc -0.0012, mF1 -0.0022
  (**beats_set=False**).
- **Fano (±1 gold-label key) acc:** **1.0000**.
- **Oracle ceiling (A4, K4 primary):** acc 0.8343 (Δ vs POOLED acc **+0.0776**, mF1 **+0.0754**).
- **Near-dup (A3):** flagged pairs (≥0.995) = 125; excluded-retrieval Δ(SET−POOLED) acc -0.0141, mF1 -0.0181.
  Distribution: `{"pooled>=0.980": 126, "mms>=0.980": 128, "maxframe>=0.980": 169, "pooled>=0.990": 125,
  "mms>=0.990": 126, "maxframe>=0.990": 166, "pooled>=0.995": 122, "mms>=0.995": 122, "maxframe>=0.995": 162}`.
- **Permutation null (N1, 100 seeds):** obs Δacc -0.0047 vs null-95th +0.0235; obs ΔmF1 -0.0077 vs
  null-95th +0.0271.
- **Bootstrap (1000 resamples):** Δacc [5/50/95]=[-0.0188/-0.0047/+0.0094]; ΔmF1 [5/50/95]=[-0.0227/-0.0078/+0.0072].
- **Per-frame null (optional, 30 seeds):** Δacc-95th -0.1838, ΔmF1-95th -0.2355.

## MHC-EN — K4 PRIMARY (memory V=629, K=4, zero-guard=0)

| arm | acc | macro_f1 | roc |
|---|---|---|---|
| POOLED | 0.7186 | 0.6081 | 0.7536 |
| SET | 0.7202 | 0.6112 | 0.7544 |
| SET_CHAMFER | 0.7266 | 0.6253 | 0.7548 |
| ASYM | 0.7250 | 0.6170 | 0.7550 |
| PIPELINE_ANCHOR | 0.7250 | 0.6239 | 0.7498 |
| WITH_TEXT_POOLED | 0.7456 | 0.7019 | 0.8001 |
| WITH_TEXT_SET | 0.7456 | 0.7001 | 0.8060 |
| POOLED_RANKONLY | 0.7186 | 0.6081 | 0.7542 |
| SET_RANKONLY | 0.7266 | 0.6218 | 0.7550 |
| POOLED_NEARDUP_EXCL | 0.7186 | 0.6081 | 0.7536 |
| SET_NEARDUP_EXCL | 0.7202 | 0.6112 | 0.7544 |

- **Primary paired Δ(SET−POOLED):** acc **+0.0016**, macro_f1 **+0.0031**.
- **Rank-only (A2):** acc +0.0079, macro_f1 +0.0137; obs Δacc +0.0079 vs rank-only null-95th +0.0175,
  rank-only bootstrap-5th -0.0048 (**corroborates=False**).
- **ASYM (pooled-query × set-memory):** acc 0.7250, macro_f1 0.6170; Δ(ASYM−SET) acc +0.0048, mF1 +0.0057
  (**beats_set=True**).
- **Fano (±1 gold-label key) acc:** **1.0000**.
- **Oracle ceiling (A4, K4 primary):** acc 0.7886 (Δ vs POOLED acc **+0.0700**, mF1 **+0.1015**).
- **Near-dup (A3):** flagged pairs (≥0.995) = 1; excluded-retrieval Δ(SET−POOLED) acc +0.0016, mF1 +0.0031.
  Distribution: `{"pooled>=0.980": 4, "mms>=0.980": 2, "maxframe>=0.980": 3, "pooled>=0.990": 1,
  "mms>=0.990": 0, "maxframe>=0.990": 2, "pooled>=0.995": 1, "mms>=0.995": 0, "maxframe>=0.995": 2}`.
- **Permutation null (N1, 100 seeds):** obs Δacc +0.0016 vs null-95th +0.0160; obs ΔmF1 +0.0031 vs
  null-95th +0.0201.
- **Bootstrap (1000 resamples):** Δacc [5/50/95]=[-0.0127/+0.0024/+0.0159]; ΔmF1 [5/50/95]=[-0.0188/+0.0042/+0.0266].
- **Per-frame null (optional, 30 seeds):** Δacc-95th -0.0302, ΔmF1-95th -0.1367.

## _mm modality SENSITIVITY — MHC-EN train-only (modality report only, B2)

- V=549, K=4, text-coverage=0.711. POOLED acc 0.7086 / mF1 0.5833; VIS-SET acc 0.7286 / mF1 0.6195;
  MM-SET acc 0.7213 / mF1 0.6375. Δ(MM−POOLED) acc +0.0128 / mF1 +0.0542; Δ(MM−VIS) acc -0.0073 / mF1 +0.0180;
  mm-vs-vis obs>null95=False, boot5th -0.0328.

## K30 granularity SENSITIVITY — DEFERRED (TERM-2)
Not run on cloud (`--k30_sensitivity 0`); the HateMM train-only K4-vs-K30 (744) granularity contrast is a
non-survival-determining breadth-modifier (B2) and is **deferred to a local run** if the verdict reviewer
requests the breadth read.

---

## Mechanical gate arithmetic (quoted verbatim — NOT the binding verdict; the independent verdict reviewer rules)

| gate | value | threshold | op | result |
|---|---|---|---|---|
| Fano[HateMM] | 1.0 | 0.99 | >= | ABOVE |
| Fano[MHC] | 1.0 | 0.99 | >= | ABOVE |
| OracleDacc_K4primary[HateMM] | 0.07755581668625144 | 0.04 | >= | ABOVE |
| OracleDacc_K4primary[MHC] | 0.06995230524642293 | 0.04 | >= | ABOVE |
| OracleKillSwitch(K4-primary,all-datasets) | False | all < 0.04 |  | SURVIVES |
| RawDacc_K4[HateMM] | -0.004700352526439522 | 0.05 | >= | BELOW |
| RawDmF1_K4[HateMM] | -0.0077158816713917 | 0.05 | >= | BELOW |
| RankOnlyCorroborates[HateMM] (A2) | False | True |  | BELOW |
| ObsDacc>null95[HateMM] | -0.004700352526439522 | 0.023501762632197387 | > | BELOW |
| Bootstrap5th>0[HateMM] | -0.018801410105757976 | 0.0 | > | BELOW |
| NearDupExclSurvives[HateMM] (A3) | -0.014101057579318454 | 0.0 | > | BELOW |
| SurvivalDacc_K4[MHC-EN] | 0.0015898251192368873 | 0.03 | >= | BELOW |
| SurvivalDmF1_K4[MHC-EN] | 0.0031269227530910104 | 0.03 | >= | BELOW |
| DatasetRule (K4-primary determined, B2) | (d) NEGATIVE (neither raw bar; weak-negative family update) | a/b/c/d |  | (d) NEGATIVE (neither raw bar; weak-negative family update) |
| _mm modality report[MHC-EN] (NON-determining, B2) | mm-vs-pool Dacc=+0.0128 / mm-vs-vis Dacc=-0.0073 | reported only |  | REPORTED |

_The above `mechanical_gate_check` is pre-registered threshold arithmetic quoted from
`w2b_probe_results.json`. It is explicitly **NOT the binding verdict** — the independent verdict reviewer
renders the ruling against the four pre-declared dataset-rule rows (post-B2, K4-primary determined). Raw
JSON + MD retrieved from the `rgcl-features` volume and staged; numbers transcribed by copy, not recall._
