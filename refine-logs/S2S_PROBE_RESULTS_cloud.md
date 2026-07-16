# S2S G0-cond Probe — RAW RESULTS (no pass/fail interpretation)

_Executor writes raw numbers only; the independent verdict reviewer renders the binding ruling (house rule; prereg §6.6, review §5.6/N6). The mechanical gate arithmetic is in `s2s_probe_results.json` and is NOT a verdict._


## HateMM  (memory N=851, T=4, zero-guard rows=1)

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

**Primary paired Δ(SET−POOLED):** acc +0.0035, macro_f1 +0.0003. **Rank-only (A2):** acc +0.0071, macro_f1 +0.0042; obs Δacc +0.0071 vs rank-only null-95th +0.0188, rank-only bootstrap-5th -0.0071 (corroborates=False: sign=True null=False boot=False).

**C2 ASYM (r3, pooled-query × set-memory):** acc 0.7603, macro_f1 0.7464. Adjudication Δ(ASYM−SET): acc -0.0094, mF1 -0.0091 (beats_set=False); obs Δacc -0.0094 vs null-95th +0.0212, bootstrap-5th -0.0223.

**Fano (±1 gold-label key) acc:** 1.0000.
**Oracle ceiling (A4):** acc 0.8578 (Δ vs POOLED acc +0.0917, mF1 +0.0953).
**Near-dup (A3):** flagged pairs (≥0.995 pooled-OR-MMS) = 120; excluded-retrieval Δ(SET−POOLED) acc +0.0082, mF1 +0.0064. Distribution: {"pooled>=0.980": 211, "mms>=0.980": 144, "maxframe>=0.980": 187, "pooled>=0.990": 146, "mms>=0.990": 116, "maxframe>=0.990": 128, "pooled>=0.995": 120, "mms>=0.995": 109, "maxframe>=0.995": 116}.
**Permutation null (N1, 100 seeds):** obs Δacc +0.0035 vs null-95th +0.0189; obs ΔmF1 +0.0003 vs null-95th +0.0298.
**Per-frame null (optional, 100 seeds):** Δacc-95th -0.1832, ΔmF1-95th -0.2480.
**Bootstrap (1000 resamples):** Δacc [5/50/95]=[-0.0106/+0.0035/+0.0165]; ΔmF1 [5/50/95]=[-0.0145/+0.0009/+0.0146].
**Stage-E gates:** train: decomp_max=5.960464477539063e-08 grecon_cos_min=0.9999995231628418 grecon_maxabs_max=0.0; dev_seen: decomp_max=5.960464477539063e-08 grecon_cos_min=0.9999997019767761 grecon_maxabs_max=0.0.

## MHC  (memory N=629, T=4, zero-guard rows=0)

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

**Primary paired Δ(SET−POOLED):** acc -0.0397, macro_f1 -0.0231. **Rank-only (A2):** acc -0.0397, macro_f1 -0.0231; obs Δacc -0.0397 vs rank-only null-95th +0.0145, rank-only bootstrap-5th -0.0636 (corroborates=False: sign=True null=False boot=False).

**C2 ASYM (r3, pooled-query × set-memory):** acc 0.6773, macro_f1 0.5607. Adjudication Δ(ASYM−SET): acc +0.0143, mF1 +0.0145 (beats_set=True); obs Δacc +0.0143 vs null-95th +0.0159, bootstrap-5th -0.0079.

**Fano (±1 gold-label key) acc:** 1.0000.
**Oracle ceiling (A4):** acc 0.8426 (Δ vs POOLED acc +0.1399, mF1 +0.2104).
**Near-dup (A3):** flagged pairs (≥0.995 pooled-OR-MMS) = 2; excluded-retrieval Δ(SET−POOLED) acc -0.0397, mF1 -0.0231. Distribution: {"pooled>=0.980": 9, "mms>=0.980": 5, "maxframe>=0.980": 8, "pooled>=0.990": 5, "mms>=0.990": 1, "maxframe>=0.990": 2, "pooled>=0.995": 2, "mms>=0.995": 0, "maxframe>=0.995": 0}.
**Permutation null (N1, 100 seeds):** obs Δacc -0.0397 vs null-95th +0.0130; obs ΔmF1 -0.0231 vs null-95th +0.0267.
**Per-frame null (optional, 100 seeds):** Δacc-95th -0.0127, ΔmF1-95th -0.0935.
**Bootstrap (1000 resamples):** Δacc [5/50/95]=[-0.0636/-0.0397/-0.0175]; ΔmF1 [5/50/95]=[-0.0576/-0.0237/+0.0100].
**Stage-E gates:** train: decomp_max=5.960464477539063e-08 grecon_cos_min=0.9999995231628418 grecon_maxabs_max=0.0; dev_seen: decomp_max=5.960464477539063e-08 grecon_cos_min=0.9999997019767761 grecon_maxabs_max=0.0.

## Mechanical gate arithmetic (NOT a verdict — see JSON)

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
