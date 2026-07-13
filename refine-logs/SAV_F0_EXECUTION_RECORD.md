# SAV (C2) F-G0/F-G1 FULL CHAIN — execution record

**Executor:** SAV smoke executor (acting under the conditional execution authorization).
**Authorization:** `refine-logs/SAV_F0_EXECUTION_AUTHORIZATION.md` §4 — ONE submission of
`sbatch scripts/slurm/sav_f0.sbatch`, effective on (1) SMOKE_PASS + (2) 7-hash re-match.
**Date:** 2026-07-14 (NZST).

## Authorization conditions — BOTH satisfied at submit time

1. **SMOKE_PASS recorded** — `refine-logs/SAV_F0_SMOKE_RECORD.md` (this executor,
   2026-07-14): job 13058 COMPLETED 0:0; all four prescription criteria met, incl.
   criterion 3 guard-preview cosines ≥ 0.999 per dataset for BOTH streams on all 12/12
   smoke videos (minima: cos_img 0.999999934 @ HateMM non_hate_video_58, cos_text
   0.999999986 @ MHC 5snzFreG79c — fp32-storage-precision-level, no pipeline drift).
2. **Seven frozen sha256 hashes re-verified** at 2026-07-14 ~03:07 NZST, immediately before
   submission (`cd /data/jehc223/RGCL && sha256sum <7 files>`; diff vs the §2 freeze =
   **empty → ALL 7 MATCH**):

```
0a580a5db752e908d02d35ada72ae5b0a156f04b6115348d56221be92ed34d5e  scripts/analysis/sav_f0_common.py
c92ae952bfa73ebbc236599659921616fe8f9dff7cfe943aec5bf324e57fa776  scripts/analysis/sav_f0_extract.py
23ad8d41606dc2ceec4e25376d03bde1610919a4eadbdf89744f6c7ec81f88ff  scripts/analysis/sav_f0_guard.py
597101ef9d82a93e670520373ca04132b3b3e0d62e7caa564c6f901e826b98f9  scripts/analysis/sav_f0_probe.py
26c5cdedf614e244731fb94f1e9727d5055c233485d707cd3fa96eab9ee97b2c  scripts/wrappers/sav_f0.sh
197e7db77b11107ef66d95e3d6f7fe8e305db3d9a801517374426fb1368e74a2  scripts/slurm/sav_f0.sbatch
c65e40bf2a1a1789fe296321736e7dba76d04598609ebf15af5ffb4f2fedd11a  research-wiki/experiments/exp-sav-f0.md
```

## Submission (the ONE authorized submission — no resubmits under any circumstance)

- **Command:** `cd /data/jehc223/RGCL && sbatch scripts/slurm/sav_f0.sbatch`
- **Submit timestamp:** `2026-07-14 03:07:50 NZST` (sacct Submit=2026-07-14T03:07:50)
- **Job id:** **13099** (`sav_f0`)
- **State at +4 s:** `RUNNING` on `foscsmlprd01` (no JobHeldUser hold this time).
- Chain: extract (warm-starts over the 12 smoke caches) → gate1 (manifests complete,
  full-count) → guard (PRIMARY min-cosine ≥ 0.999, all train+val, img+text) → gate2 →
  F-G1 statistics engine → gate3 (`verdict.json .status=="COMPLETE"`).
- RUN_ID ceremony: sbatch passes `RUN_ID=EXPECTED=SAV-F0-FG0-FG1`; wrapper refuses mismatch.
- Logs: `slurm/logs/sav_f0_13099.out` / `.err`.
- Estimated ~1–1.5 h extraction (IMPL_NOTES) + guard + F-G1 CPU probes; watcher cap 6 h.

## Executor duties on terminal state (per authorization §4.1)

- COMPLETED → verify: 6 extraction manifests `.complete==true and .n==.n_expected`
  (744/107, 549/80, 579/78); 3 guard JSONs `.pass==true`; probe `verdict.json`
  `.status=="COMPLETE"` (read `.verdict`). Report with provenance. The F-G1 verdict is
  conclusion-bearing → MUST go to an independent verdict reviewer before any F-G2 step
  (authorization §4.2); this record does NOT process the verdict.
- FAILED / non-zero exit (2=RUN_ID, 3=extraction-manifest, 4=guard-PRIMARY, 5=verdict) →
  collect evidence (sacct line, `.out`/`.err`, whichever JSONs were written); **do NOT
  resubmit**; route to a fresh result-to-claim review.
- `JobHeldUser` (if it re-enters queue) = wait; never force, never resubmit.

## Outcome

*(recorded 2026-07-14 ~05:15 NZST — RAW; NO interpretation here)*

**The F-G1 verdict is conclusion-bearing and goes to an INDEPENDENT VERDICT REVIEWER
(authorization §4.2) — this record transcribes raw artifacts only.**

### Terminal state

```
sacct -j 13099 -X: 13099  sav_f0  COMPLETED  0:0  01:24:35  2026-07-14T03:07:59  2026-07-14T04:32:34
```
Chain log (`slurm/logs/sav_f0_13099.out`) shows all three fail-closed gates passed and the
chain footer printed:
```
[gate1] all extraction manifests complete.
[guard] HateMM: PRIMARY min_cos=1.000000 (>= 0.999) pass=True | SECONDARY |Δacc|=0.0000 pass=True => PASS
[guard] MHC: PRIMARY min_cos=1.000000 (>= 0.999) pass=True | SECONDARY |Δacc|=0.0000 pass=True => PASS
[guard] MHC_zh: PRIMARY min_cos=1.000000 (>= 0.999) pass=True | SECONDARY |Δacc|=0.0000 pass=True => PASS
[gate2] reproduction guard PRIMARY passed on all datasets.
[probe] VERDICT = KILL -> /data/jehc223/RGCL/artifacts/sav_f0/probe/verdict.json
[gate3] F-G1 verdict: KILL
########## SAV F-G0/F-G1 chain COMPLETE ##########
```
`.err` contains only the known-benign shard-loading progress + the expected
`hate_video_95` partial-file NAL-unit decode errors (the banked zero-guard video). No anomalies.

### Artifact verification table (all verified by this executor with jq/torch)

| artifact | check | measured | verdict |
|---|---|---|---|
| extract manifests ×6 | `.complete==true and .n==.n_expected` | HateMM 744/744 & 107/107; MHC 549/549 & 80/80; MHC_zh 579/579 & 78/78; every split `n_skipped_resumed=2` (smoke warm-start consumed); `limit=0` | ✅ |
| per-video `.pt` counts | files == gt counts | 744/107/549/80/579/78 | ✅ |
| zero-guard | only the banked zero | HateMM/train `zero_guard_ids=["hate_video_95"]`; all others empty | ✅ |
| guard `HateMM` | `.pass==true` | PRIMARY min_cos img 0.999999999999968 (`train:hate_video_249`) / text 0.9999999999999858 (`train:non_hate_video_283`), n=1700 compared, zero_matched=2 (hate_video_95 img+text), zero_mismatch=0; SECONDARY acc_fresh=acc_cached=0.794392523364486, abs_delta=0.0 | ✅ PASS |
| guard `MHC` | `.pass==true` | PRIMARY min_cos img 0.9999999999999858 (`train:4V0KGql_fUI`) / text 0.9999999999999858 (`train:2CiUE95cOr8`), n=1258, zero_matched=0; SECONDARY acc_fresh=acc_cached=0.65, abs_delta=0.0 | ✅ PASS |
| guard `MHC_zh` | `.pass==true` | PRIMARY min_cos img 0.9999999999999858 (`train:BV1f8411b7Xz`) / text 0.9999999999999858 (`train:BV14K411t7QQ`), n=1314, zero_matched=0; SECONDARY acc_fresh=acc_cached=0.7692307692307693, abs_delta=0.0 | ✅ PASS |
| probe `verdict.json` | present, parseable, `.status=="COMPLETE"`, all arms | 30566 bytes, sha256 `46f6a8d596f298248abdc233684ac63b916d02172bbab304b85a3b1b781343fa`; `status=COMPLETE`; `arms_present=true` for all 3 datasets; 14 per-arm blocks each; `required_arms` all present | ✅ |

**Cosine-precision reconciliation (for future auditors):** the smoke record's preview cosines
(~1−6.6e-8) vs the guard's (~1−1.4e-14) are the SAME underlying fact measured through different
float paths. Empirically re-checked on 3 ids (incl. both guards' min-ids): the fresh `img_pooled`
vectors are **bitwise identical** to the banked fp32 cache vectors (`np.array_equal` on raw
bytes = True; symmetric float64 cosine = 1.0000000000000000). The preview's 1e-8-level deficit
came from its float32 `np.linalg.norm(cached)` term; the guard's 1e-14-level deficit is float
division noise in its own `_cos`. Reproduction is exact at fp32 storage level — stronger than
the ≥0.999 gate requires.

### F-G1 verdict.json — raw content summary (verbatim numbers)

- `schema=sav_f0_probe_verdict_v1`, `status=COMPLETE`, **`verdict="KILL"`**,
  `authority="research-wiki/experiments/exp-sav-f0.md (Rev-2a) F-G1"`.
- `config` echo: seeds [0-4], topk [10,20,40], selection_per_class 20, probe_train_frac 0.8,
  cv_folds 5, lambdas 1e-4..1e2, bootstrap_draws 10000, prob_clip 1e-6,
  projected_gain_bar **0.04** (base 0.03 + noise 0.01), hatemm_noharm_dacc −0.01,
  probe_stream img, carrying=MHC, noharm=HateMM, secondary=MHC_zh,
  probe "StandardScaler+L2-LogisticRegressionCV".
- `decision` (raw booleans):
  - k=10: mhc `pass_deltaL=true`, `pass_projected_gain=false` → `mhc_pass=false`;
    hatemm `noharm_deltaL=true`, `noharm_delta_acc=false` → `hatemm_noharm=false`; `proceed=false`
  - k=20: same pattern (mhc deltaL true / projected-gain false; hatemm delta_acc false); `proceed=false`
  - k=40: mhc `pass_deltaL=false`, `pass_projected_gain=false`; hatemm `noharm_delta_acc=false`; `proceed=false`
  - `proceed_k=[]` → **KILL**. `oracle_below_bar=false`
    (`oracle_max_projected_gain_mhc=0.22632248620269702`).
- Per-dataset probe context: chosen_lambda_mean = 100.0 for EVERY arm on all 3 datasets
  (λ-CV saturated at the grid top); SAV majority-vote acc: MHC .59/.6125/.6325,
  HateMM .7383/.7308/.7271, MHC_zh .6359/.6487/.6513 (k=10/20/40); head-set stability
  (5-seed, top-k intersection / mean pairwise Jaccard): MHC 0/0/0 & .022/.069/.111,
  HateMM 0/0/1 & .060/.082/.107, MHC_zh 1/1/2 & .100/.104/.106.

**Per-arm table — MHC (carrying; pooled L̄=88.8742 bits, acc .6375):**
| arm | L̄_arm | ΔL mean [CI] (x0) | Δacc mean [CI] (x0) | projGain mean [CI] (x0) |
|---|---|---|---|---|
| SAV@10 | 67.312 | .269527 [.094138,.452777] (T) | .0875 [.0175,.1625] (T) | .2300 [0,.2945] (F) |
| SAV@20 | 74.0046 | .18587 [.010074,.364825] (T) | .0525 [−.015,.12] (F) | .1598 [0,.266] (F) |
| SAV@40 | 79.8413 | .112912 [−.079045,.302119] (F) | .0375 [−.0325,.1075] (F) | .0262 [−.0443,.2443] (F) |
| C-pos | 78.8913 | .124787 [−.04921,.306035] (F) | −.01 [−.0725,.055] (F) | .0692 [−.0338,.2273] (F) |
| C-sparse@10 | 71.3852 | .218612 [.083461,.365143] (T) | .0625 [−.005,.13] (F) | .1907 [0,.2663] (F) |
| C-sparse@20 | 74.7581 | .176451 [.04238,.317312] (T) | .055 [−.015,.125] (F) | .1495 [0,.2513] (F) |
| C-sparse@40 | 77.2773 | .144961 [.00006,.293622] (T) | .05 [−.015,.115] (F) | .1082 [0,.2472] (F) |
| U-1 | 111.9196 | −.288068 [−.67368,.058263] (F) | .02 [−.055,.0975] (F) | 0 [−.2083,.0014] (F) |
| U-2@10/20/40 | 68.4696 | .255057 [.045179,.489036] (T) | .0575 [−.0275,.145] (F) | .2197 [.0126,.2744] (T) |
| oracle@10 | 67.7354 | .264235 [.068637,.463467] (T) | .0675 [−.01,.145] (F) | .2263 [0,.3032] (F) |
| oracle@20 | 73.6418 | .190405 [−.006361,.392759] (F) | .06 [−.02,.1375] (F) | .1644 [0,.2812] (F) |
| oracle@40 | 78.9202 | .124425 [−.112805,.358956] (F) | .0775 [−.005,.16] (F) | .0683 [−.0702,.2676] (F) |

**Per-arm table — HateMM (no-harm; pooled L̄=78.8726 bits, acc .7720):**
| arm | L̄_arm | ΔL mean [CI] (x0) | Δacc mean [CI] (x0) | projGain mean [CI] (x0) |
|---|---|---|---|---|
| SAV@10 | 84.1542 | −.04936 [−.176106,.074753] (F) | −.0187 [−.0729,.0355] (F) | −.0272 [−.1274,.0502] (F) |
| SAV@20 | 82.5855 | −.0347 [−.152697,.078914] (F) | −.0037 [−.0486,.043] (F) | −.0187 [−.1178,.0509] (F) |
| SAV@40 | 90.9186 | −.112579 [−.261236,.026777] (F) | −.0262 [−.0785,.0262] (F) | −.0681 [−.248,.0137] (F) |
| C-pos | 101.5123 | −.211586 [−.422983,−.012048] (T) | −.0449 [−.1159,.0262] (F) | −.1597 [−.3191,0] (F) |
| C-sparse@10 | 77.0982 | .016583 [−.077138,.112898] (F) | −.0037 [−.0449,.0393] (F) | .0084 [−.036,.09] (F) |
| C-sparse@20 | 75.8478 | .02827 [−.052866,.113144] (F) | .0019 [−.0411,.0486] (F) | .0142 [−.0265,.0799] (F) |
| C-sparse@40 | 81.0402 | −.020258 [−.11635,.071482] (F) | −.0019 [−.0449,.043] (F) | −.0107 [−.094,.0413] (F) |
| U-1 | 122.2487 | −.405384 [−.73145,−.117189] (T) | −.0093 [−.0617,.043] (F) | −.2923 [−.3484,−.0423] (T) |
| U-2@10/20/40 | 83.4731 | −.042995 [−.169864,.103596] (F) | −.0224 [−.0804,.0374] (F) | −.0235 [−.0772,.0933] (F) |
| oracle@10 | 82.9024 | −.037662 [−.185553,.096382] (F) | −.0168 [−.0766,.043] (F) | −.0204 [−.1494,.0623] (F) |
| oracle@20 | 83.664 | −.044779 [−.194249,.093704] (F) | −.0206 [−.0804,.0393] (F) | −.0245 [−.169,.0563] (F) |
| oracle@40 | 84.7486 | −.054916 [−.234874,.111006] (F) | −.0112 [−.0729,.0505] (F) | −.0305 [−.2133,.063] (F) |

**Per-arm table — MHC_zh (secondary, non-gating; pooled L̄=63.2083 bits, acc .7564):**
| arm | L̄_arm | ΔL mean [CI] (x0) | Δacc mean [CI] (x0) | projGain mean [CI] (x0) |
|---|---|---|---|---|
| SAV@10 | 66.543 | −.042753 [−.190079,.114731] (F) | −.0667 [−.1231,−.0128] (T) | −.0289 [−.151,.1258] (F) |
| SAV@20 | 69.3025 | −.078131 [−.216065,.064257] (F) | −.041 [−.0949,.0103] (F) | −.0566 [−.2104,.0571] (F) |
| SAV@40 | 73.6144 | −.133412 [−.291142,.026626] (F) | −.0538 [−.1179,.0077] (F) | −.1119 [−.2622,.0165] (F) |
| C-pos | 70.5636 | −.0943 [−.271809,.093088] (F) | −.0077 [−.0821,.0692] (F) | −.0708 [−.2474,.0767] (F) |
| C-sparse@10 | 61.8775 | .017061 [−.133906,.18961] (F) | −.0615 [−.1179,−.0077] (T) | .0105 [−.073,.1952] (F) |
| C-sparse@20 | 64.6004 | −.017848 [−.174417,.144343] (F) | −.0154 [−.0667,.0385] (F) | −.0115 [−.1488,.1459] (F) |
| C-sparse@40 | 69.3251 | −.078422 [−.245374,.078439] (F) | −.0333 [−.0821,.0179] (F) | −.0568 [−.2325,.0604] (F) |
| U-1 | 111.5408 | −.619647 [−1.031478,−.256196] (T) | −.0744 [−.1385,−.0128] (T) | −.2506 [−.3568,0] (F) |
| U-2@10/20/40 | 66.3827 | −.040698 [−.240535,.19135] (F) | −.0846 [−.1667,−.0051] (T) | −.0274 [−.1407,.1987] (F) |
| oracle@10 | 69.0299 | −.074636 [−.234055,.096441] (F) | −.0897 [−.1513,−.0308] (T) | −.0536 [−.2161,.0902] (F) |
| oracle@20 | 70.8608 | −.098109 [−.266404,.075284] (F) | −.0846 [−.1487,−.0231] (T) | −.0744 [−.2501,.064] (F) |
| oracle@40 | 73.4374 | −.131143 [−.320868,.051897] (F) | −.0718 [−.1333,−.0128] (T) | −.1092 [−.2718,.0361] (F) |

(x0 = the bootstrap CI `excludes_zero` boolean, verbatim; ΔL in bits/example, seed-averaged
example-level clustered bootstrap, n_effective = n_val: 80 / 107 / 78. U-2@k blocks are
identical within a dataset because U-2 = best single head — k-independent by construction.)

### Handoff

Chain artifacts are frozen in place (`artifacts/sav_f0/{extract,guard,probe}`); verdict.json
sha256 recorded above. **Next step per authorization §4.2: independent verdict review before
any F-G2 action.** No further submission is authorized under this ceremony; the smoke and
the full chain each consumed their single submission (13058, 13099).
