# B2 Execution Record — frozen Qwen2.5-VL-32B encoder-scale test (Stage-D/E/C/T)

**Executor:** B2 prep/executor agent · **Started:** 2026-07-14 11:18 NZST
**Authority:** C3 GO from orchestrator under the goal directive's standing authority;
conditional authorization `refine-logs/B2_PREREG_REVIEW.md` (C0/C1 done, C2 delta-check
PASS incl. R-1 fix). Pre-registration: `research-wiki/experiments/exp-encoder-32b-b2.md`
(rev r1, `DRAFT-REV1-AWAITING-DELTA-CHECK` frozen at hash 56588dc1…).

**Discipline:** single submit per stage; JobHeldUser = wait, never force; any FAILED =
HALT with evidence, no resubmit; raw numbers only, transcribed from raw trainlogs with
line provenance; no interpretation/gate application in this record beyond the
pre-registered mechanical checks (hash match, blob counts, G-dims).

---

## Stage-0 — pre-submit hash gate (2026-07-14 11:18 NZST)

Re-hash vs the C2-PASS set — **ALL 4 MATCH, gate PASS**:

```
817a951d717be56e7329ccb894c2f6ffb1edeb85e656d91286a57b34bd35284a  scripts/slurm/b2_stage_d_download.sbatch
532a8a3458f84862919d625da17b3e7e33d437b465d9bde13e93a475c5a1ff1c  scripts/slurm/b2_stage_e_extract.sbatch
9c312da639dba0ee8061b1bb3e22b4a4a074db1812e043763732e666ef04564c  scripts/slurm/b2_stage_t_train.sbatch
56588dc1b2f492e002948e9844f5059ba4bab1a156589bc67ca75b082833eb0b  research-wiki/experiments/exp-encoder-32b-b2.md
```

Disk/quota at submit (2026-07-14 11:18): `df /data` = 409G avail (98%); user quota
382G* used / 290G soft / 3000G hard, **grace 23:07 remaining**.
**Quota-grace context:** the 290G soft quota is already exceeded (382G, pre-existing);
adding the ~66G transient weights (→ ~448G) is **by design a transient overage** — the
grace window (~23h) far exceeds the planned few-hour download→extract→delete lifecycle,
and the hard cap (3000G) is never approached. Recorded per the authorization.

## Stage-D — download (job 13131) — **FAILED → HALT (kill-rule compliant, no resubmit)**

- **Submitted:** 2026-07-14 11:18 NZST, `sbatch scripts/slurm/b2_stage_d_download.sbatch`
  → **job 13131**. Terminal: **FAILED, exit 1:0, elapsed 00:00:05** (sacct via monitor).
- Stdout/evidence: `slurm/logs/b2_dl32b_13131.out` (full traceback preserved).

### Failure evidence (transcribed from the raw log + follow-up probes)

- **What worked:** job env correct (conda HateVideo, `HF_HUB_OFFLINE=0`); `df` before
  recorded (409G avail); `huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct` started;
  **all 8 small repo files downloaded fine** (.gitattributes, README, config.json,
  generation_config, configuration, chat_template, added_tokens, merges.txt — served from
  the regular hub CDN).
- **What failed:** the first safetensors shard GET →
  `requests.exceptions.HTTPError: 403 Client Error: Forbidden` at
  `https://cas-bridge.xethub.hf.co/xet-bridge-us/67dd8463…` →
  `huggingface_hub.errors.HfHubHTTPError: 403 Forbidden … <Error><Code>AccessDenied</Code>`
  (log tail, `b2_dl32b_13131.out`). All 18 shards route through the HF **xet CAS bridge**;
  the 12 shard downloads in flight all died with the same 403.
- **Diagnosis probes (2026-07-14 ~11:20-11:25 NZST, login node, read-only/1-byte):**
  1. `huggingface-cli whoami` → `Jakcey` — **stored token is valid** (not an auth expiry).
  2. Anonymous HEAD on the 32B shard resolve URL → redirects to cas-bridge with
     `X-Xet-Cas-Uid=public` → **403** (denied even without any token).
  3. 1-byte ranged GET, anonymous AND authenticated → **403 / 403** (rules out
     HEAD-signature artifacts; the GET itself is denied).
  4. **Control:** 1-byte ranged GET on `Qwen2.5-VL-7B-Instruct/model-00001-of-00005.safetensors`
     (the exact repo downloaded successfully 2026-07-02) → **403 as well.**
  5. HF status endpoint probe returned empty from this host.
- **Conclusion:** the HF xet CAS bridge is currently denying **ALL safetensors blob GETs
  from this host** (any repo, any auth state), while hub metadata/small files work. This is
  a **transient upstream/infrastructure condition** (HF xet-bridge incident or an IP-level
  block affecting the cluster), NOT a defect in the sbatch, the token, or the repo. The
  identical mechanism succeeded for this exact model on ~Jul 8 (`dl_qwen25vl_32b.log`).
- **Residue:** partial HF cache
  `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct` = **1.7 MB**
  (8 small files + 15 zero-progress `.incomplete` blob stubs). Left in place — negligible
  size; a future authorized retry would resume over it. No weights on disk; disk state
  effectively unchanged (df still 409G avail).
- **Kill-rule action:** **HALT. No resubmit** (per authorization: "any FAILED = HALT with
  evidence, no resubmit"). Stage-E/C/T NOT started; no GPU used; no test touch. Escalated
  to coordinator with this evidence — a retry-when-upstream-clears requires a coordinator
  ruling (it would be a resubmit of a burned single-submit).

### Coordinator ruling on the Stage-D failure (received 2026-07-14, recorded verbatim per its item 1)

> Ruling on Stage-D (orchestrator): option (a)+(b) combined — PROBE-GATED INFRASTRUCTURE
> RETRY, authorized as follows:
>
> 1. This is an environmental upstream failure (your evidence: auth-independent 403 across
>    repos incl. the previously-successful 7B shard), not a science/code failure — the
>    single-submit discipline's intent (no knob-tweaking after negative evidence) is not
>    violated by re-submitting the UNMODIFIED artifact once the environment recovers.
>    Record this ruling verbatim in B2_EXECUTION_RECORD.md.
> 2. Arm a probe loop (run_in_background): every ~30 min, one 1-byte authenticated ranged
>    GET against the 32B shard URL; exit the loop when it returns 200/206 (or after 24
>    probes ≈ 12h). Keep it cheap and log each probe result to the execution record's
>    Stage-D appendix.
> 3. When the probe succeeds: re-hash b2_stage_d_download.sbatch (must still be 817a951d —
>    any change voids this ruling), then ONE re-submit. Document as "infra-retry #1,
>    probe-gated, artifact unmodified". Then continue the staged plan (E -> C -> T) exactly
>    as authorized.
> 4. If 12h of probes all fail: HALT, report back — the alternative-transport route (runner
>    modification, e.g. xet-bypass/mirror) would need a delta-check, and hub 0.29.3 may not
>    support a client-side xet disable, so that decision comes back to me with your
>    findings on what transports ARE available (e.g. does the snapshot resolve non-xet
>    URLs? git-lfs clone? hf_transfer install?).
> 5. Quota-grace note: the ~23h grace clock only matters once weights land; if the retry
>    lands late in the grace window, note it but proceed — lifecycle is still ~3h and hard
>    limit is 3000G.
> Nothing else changes: JobHeldUser=wait, no further resubmits beyond infra-retry #1, all
> stage gates as authorized.

## Stage-D infra-retry #1 (probe-gated, artifact unmodified) — job 13138

- **Probe gate:** probe #2 of the loop returned **HTTP 206 at 11:53:24 NZST** (see appendix);
  pre-submit re-verify immediately before sbatch: **206 again at 12:25 NZST** — bridge serving.
- **Hash gate:** `sha256(b2_stage_d_download.sbatch) = 817a951d…284a` — **matches the
  C2-PASS/R-1 hash exactly; artifact unmodified.** Ruling conditions satisfied.
- **Submitted:** 2026-07-14 12:25:12 NZST → **job 13138** ("infra-retry #1, probe-gated,
  artifact unmodified" per coordinator ruling item 3; the ~35 min gap between probe success
  and submit was a lost executor wake, nudged by orchestrator).
- Stdout: `slurm/logs/b2_dl32b_13138.out`.
- **Terminal: COMPLETED, exit 0:0, elapsed 00:06:51** (sacct; ~12:32 NZST).

### Stage-D blob verification — PASS (2026-07-14 12:45 NZST)

| check | result |
|---|---|
| safetensors shard count (job log + on-disk re-count) | **18 / 18** ✅ |
| snapshot dir | `snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da/` present ✅ |
| total cache size | **64G** (`du -sh` on `models--Qwen--Qwen2.5-VL-32B-Instruct`) |
| `.incomplete` blobs remaining | **0** ✅ (the 15 stubs from job 13131 all resolved) |
| config.json (cheap-load from snapshot) | `hidden_size=5120`, `num_hidden_layers=64`, `torch_dtype=bfloat16`, `vision_config.out_hidden_size=5120` — **matches the prereg's pinned 5120** ✅ |
| df before → after (job log) | 409G avail → **344G avail** (66G landed) |

**Quota-grace note:** the ~66G transient is now on disk — the grace clock is consuming
from ~12:32 NZST. Per ruling item 5: lifecycle ~3h ≪ grace (~22h at landing); proceeding
without idle gaps.

## Stage-E — extraction — job 13139 (single submit)

- **Submitted:** 2026-07-14 12:47:13 NZST, `sbatch scripts/slurm/b2_stage_e_extract.sbatch`
  → **job 13139**, state RUNNING within seconds (no hold).
- Stdout: `slurm/logs/b2_ext32b_13139.out`. Order: HateMM → MHC → MHC_zh (fail-closed).
- **Terminal: COMPLETED, exit 0:0, elapsed 01:50:29** (sacct; 14:37:40 NZST) — well under
  the 5-7 GPU-h budget (realized ≈ 2.4 s/video incl. both forward passes, 2662 videos).

### Stage-E G-dims verification — PASS (2026-07-14 14:45 NZST)

**(1) Sbatch echo output (job log `b2_ext32b_13139.out`):** all 9 caches echoed
`img=(N,5120) text=(N,5120)` with N exactly 744/107/215 (HateMM), 549/80/161 (MHC),
579/78/149 (MHC_zh); `Saved` lines confirm Dv=Dt=5120 for every split.

**(2) Independent CPU `torch.load` of ALL 9 caches (not just one per dataset):**

| cache | rows (expect) | img / text dims | ids == 7B arm | labels == 7B arm | zero-vec |
|---|---|---|---|---|---|
| HateMM train | 744 (744) ✅ | 5120 / 5120 ✅ | ✅ | ✅ | 1 (see note) |
| HateMM dev_seen | 107 (107) ✅ | 5120 / 5120 ✅ | ✅ | ✅ | 0 |
| HateMM test_seen | 215 (215) ✅ | 5120 / 5120 ✅ | ✅ | ✅ | 0 |
| MHC train | 549 (549) ✅ | 5120 / 5120 ✅ | ✅ | ✅ | 0 |
| MHC dev_seen | 80 (80) ✅ | 5120 / 5120 ✅ | ✅ | ✅ | 0 |
| MHC test_seen | 161 (161) ✅ | 5120 / 5120 ✅ | ✅ | ✅ | 0 |
| MHC_zh train | 579 (579) ✅ | 5120 / 5120 ✅ | ✅ | ✅ | 0 |
| MHC_zh dev_seen | 78 (78) ✅ | 5120 / 5120 ✅ | ✅ | ✅ | 0 |
| MHC_zh test_seen | 149 (149) ✅ | 5120 / 5120 ✅ | ✅ | ✅ | 0 |

- **Paired-arm id audit:** id lists identical (set AND order) to the 7B caches on all 9
  splits; labels bit-identical. **G-dims (HARD): PASS.**
- **Zero-vector note:** the single zero-guard is HateMM train idx 355 = `hate_video_95` —
  the **same video at the same index is zero in the 7B cache** (pre-existing decode
  failure, pipeline-consistent; NOT a 32B extraction defect).
- **7B-cache mtime tripwire: PASS** — all 9 `*Qwen2.5-VL-7B-Instruct_HF.pt` mtimes remain
  2026-07-02 (untouched); the Rev-2 overwrite risk did not materialize.
- **b2_push confirmation:** 3/3 datasets pushed (`[b2_push] done -> …/embeddings/{HateMM,MHC,MHC_zh}` in job log).
- 32B cache sizes: HateMM 30.5/4.4/8.8 MB, MHC 22.5/3.3/6.6 MB, MHC_zh 23.7/3.2/6.1 MB
  (train/dev/test) — as predicted (tens of MB).

## Stage-C — weight cleanup — DONE (2026-07-14 14:51 NZST, after G-dims PASS)

- df BEFORE: 342G avail → deleted
  `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct` (the 64G weights) →
  df AFTER: **406G avail**. **7B weights dir verified KEPT.**
- Quota after: 382G* used, grace 19:34 remaining — the 66G transient overage lasted
  ~2h20m (12:32 → 14:51), well inside grace, as designed.

## Stage-T — training — job 13146 (single submit)

- **Submitted:** 2026-07-14 14:51:41 NZST, `sbatch scripts/slurm/b2_stage_t_train.sbatch`
  → **job 13146**, initial state **PENDING (JobHeldUser)** — waiting for auto-release per
  rule (never force).
- Stdout: `slurm/logs/enc3seed_13146.out`; per-run trainlogs
  `slurm/logs/enc3s_<ds>_Qwen2.5-VL-32B-Instruct_HF_seed<s>_13146.trainlog` (9 expected).
- Tracked waiter armed: until-loop on sacct terminal state, sleep 60, cap 1h.
- **Terminal: COMPLETED, exit 0:0, elapsed 00:09:03** (sacct; 15:26:47 NZST; JobHeldUser
  hold auto-released after ~26 min).

### Stage-T run verification (2026-07-14 15:35 NZST)

- **All 9 trainlogs exist** with the expected naming:
  `slurm/logs/enc3s_{HateMM,MHC,MHC_zh}_Qwen2.5-VL-32B-Instruct_HF_seed{0,1,2}_13146.trainlog`
  (24-27 KB each). Sbatch stdout: `slurm/logs/enc3seed_13146.out` (9 `RESULT_ROW` lines).
- **Dims trained = 5120 in every run:** each trainlog prints
  `Image feature dimension:  5120` / `Text feature dimension:  5120` (lines 2-3) and the
  head builds `Linear(in_features=5120, out_features=1024)` for both projections (log
  lines 7/11). All 9 runs completed 30 epochs (30 `Test_Retrieval` readouts each).
- **G-repro (Rev-4: config #1 = HateMM s0) — sanity readout:** loads the 5120-d caches
  without error, trains 30 epochs with well-formed `Val_/Test_Retrieval` lines, readouts
  non-degenerate (val-sel Test F1 0.8724 / acc 0.8791; final 0.8197 / 0.8279 — not
  0.5-band, no NaN). Recorded as observed; formal gate application is the verdict
  reviewer's task.
- **Namespace line 1** (spot-read, HateMM s0): `model='Qwen2.5-VL-32B-Instruct_HF'`,
  `dataset='HateMM'`, `seed=0`, `archive_feats=None`, `lambda_seg=0.0`, `warmup=5`,
  `group_name='RAC_video_archive_seeds'`, `force=False`, all campaign-era extra flags at
  inert defaults. Full Namespace-diff gate vs 12850/12275-12276/13115 left to the verdict
  reviewer per prereg.

### RAW per-seed table — the 9 NEW 32B runs (transcription only; NO deltas, NO gates, NO interpretation)

Parsing: sbatch's own `RESULT_ROW` output CROSS-CHECKED by an independent fresh re-parse
of each raw trainlog (regex over full text; val-sel = epoch ≥ warmup 5 maximizing
Val_Retrieval acc, roc tie-break; final = epoch 29). **All 18 readings (9 runs × 2
protocols) agree to all 4 printed decimals between the two parsers.** Line numbers below
are newline-based (`grep -n` semantics) positions of the `Test_Retrieval Epoch NN …` line
in the named trainlog (tqdm `\r` segments not counted as lines).

**Arm: frozen Qwen2.5-VL-32B-Instruct_HF (5120-d), archive OFF, job 13146**

| dataset | seed | val-sel: ep / Test F1 / acc / roc (line) | final: ep / Test F1 / acc / roc (line) |
|---|---|---|---|
| HateMM | 0 | e25 / 0.8724 / 0.8791 / 0.9210 (`…HateMM…seed0…:291`) | e29 / 0.8197 / 0.8279 / 0.9195 (`:332`) |
| HateMM | 1 | e26 / 0.8552 / 0.8605 / 0.9234 (`…seed1…:301`) | e29 / 0.8638 / 0.8698 / 0.9197 (`:332`) |
| HateMM | 2 | e23 / 0.8547 / 0.8605 / 0.9193 (`…seed2…:273`) | e29 / 0.8301 / 0.8372 / 0.9151 (`:334`) |
| MHC (EN) | 0 | e14 / 0.5665 / 0.7081 / 0.7566 (`…MHC…seed0…:163`) | e29 / 0.6674 / 0.7516 / 0.8271 (`:299`) |
| MHC (EN) | 1 | e28 / 0.6972 / 0.7578 / 0.8203 (`…seed1…:288`) | e29 / 0.7070 / 0.7640 / 0.8398 (`:298`) |
| MHC (EN) | 2 | e13 / 0.6618 / 0.7391 / 0.7861 (`…seed2…:157`) | e29 / 0.6940 / 0.7640 / 0.8302 (`:302`) |
| MHC_zh | 0 | e15 / 0.7016 / 0.7584 / 0.8581 (`…MHC_zh…seed0…:174`) | e29 / 0.7245 / 0.7785 / 0.8498 (`:301`) |
| MHC_zh | 1 | e23 / 0.7221 / 0.7718 / 0.8498 (`…seed1…:244`) | e29 / 0.7517 / 0.7919 / 0.8626 (`:299`) |
| MHC_zh | 2 | e6 / 0.7006 / 0.7785 / 0.8476 (`…seed2…:90`) | e29 / 0.7296 / 0.7651 / 0.8598 (`:298`) |

(Full filenames: `slurm/logs/enc3s_<dataset>_Qwen2.5-VL-32B-Instruct_HF_seed<s>_13146.trainlog`.)

Reference arms for the paired comparison (NOT re-run, NOT transcribed here): HateMM
CLIP/7B + MHC-EN CLIP + MHC-EN-7B-s0 = job 12850 (`exp-encoder-3seed.md:150-170`);
MHC-EN 7B s1/s2 = arcbase 12275/12276; MHC-ZH CLIP/7B = job 13115
(`B1_VERDICT_REVIEW.md:29-40`).

**Executor deliverable complete (2026-07-14 15:40 NZST).** Delta computation and gate
application (formal G-repro, Namespace-diff, decision rule — 32B-vs-CLIP primary /
32B-vs-7B secondary, both protocols) now pass to the independent verdict reviewer per the
pre-registration. Test-touch: this one 9-run evaluation, as budgeted; no re-runs.

---

## Stage-D appendix — probe log (1-byte authenticated ranged GET on the 32B shard; ~30-min cadence, max 24 probes ≈ 12h; appended live by the background probe loop)

- probe #1 2026-07-14 11:23:22 NZST: http_code=403
- probe #2 2026-07-14 11:53:24 NZST: http_code=206
- **PROBE SUCCESS at probe #2** — CAS bridge serving again; proceeding to re-hash + infra-retry #1.
