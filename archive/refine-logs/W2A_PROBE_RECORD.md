# W2-A Stage-P' Probe — EXECUTION RECORD (RAW numbers only)

**Scope:** RAW transcription of the probe's own output. The mechanical gate arithmetic below is
pre-registered threshold ARITHMETIC quoted VERBATIM and is explicitly **NON-binding** — an
independent verdict reviewer renders the ruling. No pass/fail interpretation is made here.

Method line: W2-A = transcript-first grounded-key extraction (joint forward
`[transcript][frames][instruction]` through Qwen2.5-VL-7B; vision-span pool = grounded retrieval
key). Probe = LOO retrieval + the SOLE binding K9 conditional-info adjudicator of A=grd on
`Z_best` (8960-d) with a ≥150-perm null and the +0.040 triple rule; kNN GROUNDED−CONCAT is ADVISORY.

---

## 1. Provenance

| item | value |
|---|---|
| Frozen probe `scripts/analysis/w2a_probe.py` | sha256 `af4a2f9f5b35461173fd82c176bd52c6fc84bf8fc0d09736f938d38d8f6fe06d` — matches r2b freeze; re-verified byte-identical AND gated in-container (adapter aborts on mismatch; the gate printed the matching hash every chunk) |
| Adapter shim `scripts/analysis/w2a_probe_cloud_adapter.py` | sha256 `fb609d4be12dd96162634a8c463fdb68134215f530e8297de550b2f4ee2bccdd` (Modal path-adapter: symlinks `/data/jehc223/RGCL/{data,src}`→`/root/{data,src}`; runpy-execs the frozen probe unchanged) |
| Runner `scripts/cloud/modal_probe_runner.py` | sha256 `d246668f1d357bca8a6aa96bbcce537a764be3499ad7330f0efe37c82a91747d` (plumbing only: env-gated timeout + app name, soft-budget/periodic-commit chunk mode; **video-guard intact**) |
| SLURM driver `scripts/slurm/w2a_probe_chunkloop.sbatch` | sha256 `7b60c9fe19a1160d9db61abdde43fda4a84c81a65f68c7db6bc83cc03a7f32a1` (CPU-only, no GPU) |
| Probe config (from results JSON `meta`) | `ci_nseed_perm=150`, `Zbest_dim=8960`, `ci_bar=0.040`, `oracle_bar=0.040`, `fano_bar=0.99`, `expected_mem={HateMM:851, MHC:629}`, `grounded_dir=grounded_qwen7b_8f` |
| CI_NSEED=150 | CONFIRMED in-container ("CI_NSEED=150" banner every chunk); final checkpoint holds 150 perms in each of the 3 run_perm cells |
| Compute | Modal cloud CPU (probe compute); triage-only per CLAUDE.md (~1.4pt cross-hardware drift — NEVER mixed into local tables) |

The FROZEN PROBE was never edited. All engineering was launch/runner plumbing (adapter + runner
timeout/app-name/chunk-mode + SLURM driver), each additive and backward-compatible; the
`modal_probe_runner.py` video-upload hard-guard was preserved.

---

## 2. Interruption → resume → chunk-loop lineage (full)

The probe's K9 permutation-null is multi-hour (450 perms across 3 run_perm variants). Modal clamps
the effective function timeout to **~3600 s server-side** (VERIFIED: `MODAL_PROBE_TIMEOUT=43200`
reaches the child and computes 43200 — no stray 3600 in code — yet every single-container attempt
was killed at ~3600 s function-time). So one container could never finish; the run was completed as
a resumable `--ci_ckpt` chunk loop under a soft budget below the cap.

Timeline (all UTC):
- **Cancellation 1** — app `ap-93KHpJNP9yDSui6fhIlOLs` (first detached, launch ~02:36) killed ~03:38:35 (~62 min).
- **Cancellation 2** — app `ap-qNF5v5HekPTrGvwjPWGIp0` (distinct name `rgcl-w2a-probe-e`, launch ~04:51) killed 05:53:41 (~62 min). (Both initially mis-attributed to a cross-agent sweep; corrected to the server timeout by the identical 62-min bound.)
- **`rm`-race** — `modal volume rm /w2a_ci_ckpt.json` (~06:00) RACED with Modal volume eventual-consistency; the checkpoint persisted and the next run RESUMED it (a valid deterministic resume, NOT clean-from-scratch — corrected provenance).
- **Validation chunk** `ap-nkulMcYIzCYyVHmuLGY4pL` (budget 1200 s, ~06:1x) — proved soft-budget→periodic-commit→resume; HateMM Z_best 40→60 perms, point-arms cached.
- **Reaped client driver** — chunk `ap-R8EPtV3V1aQsBgmj3TElRl` advanced 60→140 perms then the client-side chain driver was REAPED ~07:31 (login=compute node reaps non-SLURM processes); no chunk chained.
- **SLURM-wrapped loop — job 13212** (CPU-only, non-detached modal client, reap-proof), `sacct` COMPLETED, Elapsed 02:38:57, start 2026-07-16T08:49:58Z → loop finished 2026-07-16T11:28:54Z:

| in-job chunk | app id | launch ts (UTC) | perms-after (sum across cells) |
|---|---|---|---|
| 1 | `ap-DHOcJQWtHAVEU2g54M1xyq` | 2026-07-16T08:49:58Z | 230 |
| 2 | `ap-7SspMdpMBRJjtfzSWqvnUu` | 2026-07-16T09:43:38Z | 310 |
| 3 | `ap-RPjOTBQsELsdyyltlbGTxp` | 2026-07-16T10:37:19Z | COMPLETE (results on volume) |

Full ledger: `refine-logs/W2A_CHUNK_LOG.md`. Final checkpoint cell perm-counts: `HateMM|Z_best_8960`=150,
`HateMM|Z_best_covered`=150, `MHC|Z_best_8960`=150, `HateMM|Qwen_only_7168`=0 and `MHC|Qwen_only_7168`=0
(run_perm=False by design — point-arms only).

### Determinism / equivalence of the chunked run

The chunked run is equivalent to one uninterrupted run: every chunk shared ONE checkpoint whose
`_meta` pins `probe_sha=af4a2f9f…` (frozen probe) and `grd_sha` = the extraction c013884
grounded-cache sha256s (HateMM `1cae1f83…+41bda7de…`, MHC `9f8da7a1…+7c1a1a4f…` — verified MATCH), so
a config/data drift would force a fresh checkpoint. Each perm seed is
`np.random.default_rng(CI_PERM_BASE+si)` (container-independent); point-arms are computed once and
cached; resume continues the perm loop from `len(maxk)`. Any cross-container float difference is far
below the triage-tier drift and does not affect the +0.040 triple-rule arithmetic.

---

## 3. RAW results

### 3.1 HateMM (memory N=851, zero-guard=1, empty-transcript=48)

| arm | acc | macro_f1 | roc |
|---|---|---|---|
| POOLED_IMG | 0.7673 | 0.7568 | 0.8259 |
| CONCAT | 0.8026 | 0.8003 | 0.8905 |
| GROUNDED | 0.7767 | 0.7708 | 0.8627 |
| GROUNDED_TEXT | 0.7873 | 0.7859 | 0.8979 |
| GROUNDED_PFX | 0.7756 | 0.7695 | 0.8632 |
| CONCAT_PCA | 0.8249 | 0.8185 | 0.8946 |
| CONCAT_ALPHA | 0.8108 | 0.8076 | 0.8848 |

- **BINDING K9 — grd on Z_best (8960-d):** calib accZA 1.0000 (PASS=True); best decision-k=8 Δacc −0.0000 CI[−0.0052,+0.0049]; C1(≥+0.040)=False C2(CI>0)=False C3(real>all perm)=False; perm-null maxk mean −0.0033 max +0.0085 (n=150); **VERDICT=GROUNDED_DEAD_AT_ZBEST**.
- **Secondary — grd on Qwen-only (7168-d):** best Δacc −0.0038 CI[−0.0092,+0.0014] VERDICT=SECONDARY_NO_PERM (non-binding).
- **K5 oracle-ceiling (grd vs CONCAT, tie→CONCAT):** acc 0.8660 (Δ vs CONCAT acc +0.0635, mF1 +0.0628); chose-grd fraction 0.395.
- **K4 Fano (±1 gold-label key):** 1.0000.
- **ADVISORY kNN Δ(GROUNDED−CONCAT):** acc −0.0259, macro_f1 −0.0295; beat CONCAT-PCA(dim=850)-sign=False, beat CONCAT-α(α*=0.6)-sign=False.
- **ADVISORY rank-only (sim→1.0):** Δacc −0.0235, ΔmF1 −0.0267 (sign-matches sim-weighted=True); obs Δacc −0.0235 vs rank-only null-95th +0.0331 (gt=False); rank-only bootstrap-5th −0.0470.
- **ADVISORY perm-null (100 seeds):** obs Δacc −0.0259 vs null-95th +0.0330 (gt=False); obs ΔmF1 −0.0295 vs null-95th +0.0483.
- **ADVISORY bootstrap (1000):** Δacc [5/50/95]=[−0.0470/−0.0247/−0.0035]; ΔmF1 [5/50/95]=[−0.0510/−0.0280/−0.0067].
- **A3 near-dup:** flagged pairs (≥0.995 grd-OR-concat)=36; excluded-retrieval Δacc −0.0259, mF1 −0.0295. Distribution: grd≥0.980=79, concat≥0.980=108, grd≥0.990=39, concat≥0.990=40, grd≥0.995=31, concat≥0.995=29.
- **Amdt-5 covered-rows-only (non-empty transcript, n=802):** advisory kNN Δacc −0.0262 ΔmF1 −0.0286; binding K9 best Δacc −0.0032 CI[−0.0075,+0.0012] VERDICT=GROUNDED_DEAD_AT_ZBEST.

### 3.2 MHC (memory N=629, zero-guard=0, empty-transcript=0)

| arm | acc | macro_f1 | roc |
|---|---|---|---|
| POOLED_IMG | 0.7027 | 0.5671 | 0.6614 |
| CONCAT | 0.7647 | 0.7149 | 0.8276 |
| GROUNDED | 0.7138 | 0.6024 | 0.7176 |
| GROUNDED_TEXT | 0.7838 | 0.7388 | 0.8442 |
| GROUNDED_PFX | 0.7091 | 0.5986 | 0.7152 |
| CONCAT_PCA | 0.7440 | 0.6969 | 0.8218 |
| CONCAT_ALPHA | 0.7965 | 0.7628 | 0.8490 |

- **BINDING K9 — grd on Z_best (8960-d):** calib accZA 1.0000 (PASS=True); best decision-k=8 Δacc −0.0038 CI[−0.0099,+0.0019]; C1(≥+0.040)=False C2(CI>0)=False C3(real>all perm)=False; perm-null maxk mean −0.0065 max +0.0076 (n=150); **VERDICT=GROUNDED_DEAD_AT_ZBEST**.
- **Secondary — grd on Qwen-only (7168-d):** best Δacc +0.0032 CI[−0.0067,+0.0137] VERDICT=SECONDARY_NO_PERM (non-binding).
- **K5 oracle-ceiling (grd vs CONCAT, tie→CONCAT):** acc 0.8617 (Δ vs CONCAT acc +0.0970, mF1 +0.1082); chose-grd fraction 0.353.
- **K4 Fano (±1 gold-label key):** 1.0000.
- **ADVISORY kNN Δ(GROUNDED−CONCAT):** acc −0.0509, macro_f1 −0.1125; beat CONCAT-PCA(dim=628)-sign=False, beat CONCAT-α(α*=0.3)-sign=False.
- **ADVISORY rank-only (sim→1.0):** Δacc −0.0445, ΔmF1 −0.1066 (sign-matches sim-weighted=True); obs Δacc −0.0445 vs rank-only null-95th +0.0238 (gt=False); rank-only bootstrap-5th −0.0779.
- **ADVISORY perm-null (100 seeds):** obs Δacc −0.0509 vs null-95th +0.0254 (gt=False); obs ΔmF1 −0.1125 vs null-95th +0.0431.
- **ADVISORY bootstrap (1000):** Δacc [5/50/95]=[−0.0843/−0.0525/−0.0207]; ΔmF1 [5/50/95]=[−0.1540/−0.1145/−0.0694].
- **A3 near-dup:** flagged pairs (≥0.995 grd-OR-concat)=0; excluded-retrieval Δacc −0.0509, mF1 −0.1125. Distribution: grd≥0.980=6, concat≥0.980=5, grd≥0.990=2, concat≥0.990=2, grd≥0.995=0, concat≥0.995=0.
- (MHC 100% transcript coverage → no covered-rows-only secondary.)

---

## 4. K2 / K3 LIVE cross-reference to extraction record c013884 (MANDATORY for the verdict reviewer)

The probe's own Stage-E' gate read-back matches extraction record c013884 exactly and both K2/K3 are
LIVE on both datasets (the verdict reviewer MUST confirm this before honoring any K9 PROCEED — here
both K9 verdicts are GROUNDED_DEAD_AT_ZBEST, so the point is moot, but it is recorded per protocol):

| dataset / split | grecon_cos_min | grounding_present_median | grounding_void(≥0.999) | placebo_median | placebo_void(≥0.999) |
|---|---|---|---|---|---|
| HateMM / train | 0.9999995231628418 | 0.9368228316307068 | False (LIVE) | 0.9804498553276062 | False (LIVE) |
| HateMM / dev_seen | 0.9999997019767761 | 0.9474989771842957 | False (LIVE) | 0.9812073409557343 | False (LIVE) |
| MHC / train | 0.9999995231628418 | 0.9604964256286621 | False (LIVE) | 0.9711186289787292 | False (LIVE) |
| MHC / dev_seen | 0.9999997019767761 | 0.9608643352985382 | False (LIVE) | 0.9708898663520813 | False (LIVE) |

These grounding/placebo medians are bit-consistent with c013884's saved-lines (HateMM 0.9368/0.9475
grounding, 0.9804/0.9812 placebo; MHC 0.9605/0.9609 grounding, 0.9711/0.9709 placebo), and every
VOID flag is False (LIVE). K2 GroundingLive LIVE + K3 Placebo LIVE, both datasets.

---

## 5. Mechanical gate arithmetic — VERBATIM, NON-BINDING (the verdict reviewer rules)

Copied verbatim from the probe's `mechanical_gate_check` (this is pre-registered threshold
arithmetic, NOT a verdict):

| gate | value | threshold | op | result |
|---|---|---|---|---|
| Fano[HateMM] (K4) | 1.0 | 0.99 | >= | ABOVE |
| Fano[MHC] (K4) | 1.0 | 0.99 | >= | ABOVE |
| OracleDacc[HateMM] (K5) | 0.06345475910693299 | 0.04 | >= | ABOVE |
| OracleDacc[MHC] (K5) | 0.09697933227345001 | 0.04 | >= | ABOVE |
| OracleKillSwitch(all-datasets) (K5) | False | all < 0.04 |  | SURVIVES |
| GroundingLive[HateMM] (K2) | LIVE | LIVE |  | LIVE |
| Placebo[HateMM] (K3) | LIVE | LIVE |  | LIVE |
| GroundingLive[MHC] (K2) | LIVE | LIVE |  | LIVE |
| Placebo[MHC] (K3) | LIVE | LIVE |  | LIVE |
| CondInfo Z_best VERDICT[HateMM] (K9 BINDING) | GROUNDED_DEAD_AT_ZBEST | CONDINFO_PROCEED |  | BELOW |
| CondInfo Z_best VERDICT[MHC] (K9 BINDING) | GROUNDED_DEAD_AT_ZBEST | CONDINFO_PROCEED |  | BELOW |
| CondInfo BINDING (any LIVE dataset PROCEED) (K9) | False | True |  | BELOW |
| RawDacc[HateMM] (K6 adv) | -0.025851938895417148 | 0.05 | >= | BELOW |
| RawDmF1[HateMM] (K6 adv) | -0.029508777130594965 | 0.05 | >= | BELOW |
| BeatConcatPCA+alpha sign[HateMM] (K6 adv) | False | True |  | BELOW |
| ObsDacc>null95[HateMM] (K7 adv) | -0.025851938895417148 | 0.03301997649823741 | > | BELOW |
| NearDupExclSurvives[HateMM] (K7b adv) | -0.025851938895417148 | 0.0 | > | BELOW |
| Bootstrap5th>0[HateMM] (K8 adv) | -0.047003525264394774 | 0.0 | > | BELOW |
| RankOnlyCorroborates[HateMM] (adv) | False | True |  | BELOW |

---

## 6. Artifacts

- Probe raw outputs (from volume `rgcl-features`): `/W2A_PROBE_RESULTS.md`, `/w2a_probe_results.json` (61466 B), final `/w2a_ci_ckpt.json` (150/150/150 perms). Retrieved via `modal volume get`.
- SLURM job log `slurm/logs/w2a_probe_loop_13212.log`; per-chunk modal logs `slurm/logs/w2a_probe_chunk_13212_{1,2,3}.log`.
- No pass/fail interpretation is made in this record — the independent verdict reviewer rules next
  (prereg verbatim; must check the K2/K3-LIVE row above before honoring any K9 PROCEED; +0.040 triple
  rule; chunked-run equivalence; triage-only tier).
