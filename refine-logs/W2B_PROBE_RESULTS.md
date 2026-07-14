# W2-B Sub-clip Set-Matching Probe — RAW RESULTS (no pass/fail interpretation)

_Executor writes raw numbers only; the independent verdict reviewer renders the binding ruling (house rule). The mechanical gate arithmetic in `w2b_probe_results.json` is NOT a verdict. K4 primary is the sole survival-determining arm (B2); K30/_mm are breadth/modality reports._


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

**Primary paired Δ(SET−POOLED):** acc -0.0047, macro_f1 -0.0077. **Rank-only (A2):** acc +0.0000, macro_f1 -0.0033; obs Δacc +0.0000 vs rank-only null-95th +0.0213, rank-only bootstrap-5th -0.0153 (corroborates=False).

**ASYM (pooled-query × set-memory):** acc 0.7509, macro_f1 0.7419; Δ(ASYM−SET) acc -0.0012, mF1 -0.0022 (beats_set=False).

**Fano (±1 gold-label key) acc:** 1.0000.
**Oracle ceiling (A4, K4 primary):** acc 0.8343 (Δ vs POOLED acc +0.0776, mF1 +0.0754).
**Near-dup (A3):** flagged pairs (≥0.995) = 125; excluded-retrieval Δ(SET−POOLED) acc -0.0141, mF1 -0.0181. Distribution: {"pooled>=0.980": 126, "mms>=0.980": 128, "maxframe>=0.980": 169, "pooled>=0.990": 125, "mms>=0.990": 126, "maxframe>=0.990": 166, "pooled>=0.995": 122, "mms>=0.995": 122, "maxframe>=0.995": 162}.
**Permutation null (N1, 100 seeds):** obs Δacc -0.0047 vs null-95th +0.0235; obs ΔmF1 -0.0077 vs null-95th +0.0271.
**Bootstrap (1000 resamples):** Δacc [5/50/95]=[-0.0188/-0.0047/+0.0094]; ΔmF1 [5/50/95]=[-0.0227/-0.0078/+0.0072].
**Per-frame null (optional, 30 seeds):** Δacc-95th -0.1838, ΔmF1-95th -0.2355.

## MHC — K4 PRIMARY (memory V=629, K=4, zero-guard=0)

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

**Primary paired Δ(SET−POOLED):** acc +0.0016, macro_f1 +0.0031. **Rank-only (A2):** acc +0.0079, macro_f1 +0.0137; obs Δacc +0.0079 vs rank-only null-95th +0.0175, rank-only bootstrap-5th -0.0048 (corroborates=False).

**ASYM (pooled-query × set-memory):** acc 0.7250, macro_f1 0.6170; Δ(ASYM−SET) acc +0.0048, mF1 +0.0057 (beats_set=True).

**Fano (±1 gold-label key) acc:** 1.0000.
**Oracle ceiling (A4, K4 primary):** acc 0.7886 (Δ vs POOLED acc +0.0700, mF1 +0.1015).
**Near-dup (A3):** flagged pairs (≥0.995) = 1; excluded-retrieval Δ(SET−POOLED) acc +0.0016, mF1 +0.0031. Distribution: {"pooled>=0.980": 4, "mms>=0.980": 2, "maxframe>=0.980": 3, "pooled>=0.990": 1, "mms>=0.990": 0, "maxframe>=0.990": 2, "pooled>=0.995": 1, "mms>=0.995": 0, "maxframe>=0.995": 2}.
**Permutation null (N1, 100 seeds):** obs Δacc +0.0016 vs null-95th +0.0160; obs ΔmF1 +0.0031 vs null-95th +0.0201.
**Bootstrap (1000 resamples):** Δacc [5/50/95]=[-0.0127/+0.0024/+0.0159]; ΔmF1 [5/50/95]=[-0.0188/+0.0042/+0.0266].
**Per-frame null (optional, 30 seeds):** Δacc-95th -0.0302, ΔmF1-95th -0.1367.

## _mm modality SENSITIVITY — MHC-EN train-only (modality report only, B2)

- V=549, K=4, text-coverage=0.711. POOLED acc 0.7086/mF1 0.5833; VIS-SET acc 0.7286/mF1 0.6195; MM-SET acc 0.7213/mF1 0.6375. Δ(MM−POOLED) acc +0.0128/mF1 +0.0542; Δ(MM−VIS) acc -0.0073/mF1 +0.0180; mm-vs-vis obs>null95=False, boot5th -0.0328.

## Mechanical gate arithmetic (NOT a verdict — see JSON)

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
