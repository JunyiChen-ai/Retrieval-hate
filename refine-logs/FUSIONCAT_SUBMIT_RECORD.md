# FUSIONCAT — SUBMIT EXECUTION RECORD

Executor: fusioncat submit executor (Opus 4.8). Working tree `/data/jehc223/RGCL`, conda `HateVideo`.
Frozen family: `refine-logs/FUSIONCAT_FREEZE.md` (APPROVED-WITH-NOTES). Prereg: `refine-logs/FUSIONCAT_PREREG.md`.
Chain script = prereg §4 / §6. NO state/ mutation, NO push. Raw facts only; verdict is an independent reviewer's job.

---

## Authorization (S0) — VERIFIED @ HEAD `aa132f9`

Re-ran `sha256sum` + `git status --porcelain src/` at submit-executor start. ALL match the freeze block byte-exact:

| artifact | expected (freeze) | measured | match |
|---|---|---|---|
| `refine-logs/FUSIONCAT_PREREG.md` | `c88332b8…433c0830` | `c88332b8972e3270081600d0a8cb892a8d24afefbc73e378a5a3104a433c0830` | ✓ |
| `scripts/slurm/fusioncat_family.sbatch` | `62bfb773…2517fc` | `62bfb773beeec325096362c86bae1aca8b90c94804ba3798973bbc88e82517fc` | ✓ |
| `src/model/classifier.py` | `e7b61df4…` | `e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378` | ✓ |
| `src/run_rac.py` | `b85eb72a…` | `b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3` | ✓ |
| `src/model/loss.py` | `2ae7a73f…` | `2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b` | ✓ |
| `src/utils/retrieval.py` | `d43e3bc4…` | `d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57` | ✓ |
| `git status --porcelain src/` | empty | empty (CLEAN) | ✓ |

VOID-on-edit clause (freeze §26-39): none of the 3 VOID conditions triggered ⇒ authorization VALID.
NO codex gate (prereg §4.5: zero-code flag-only family, reviewer-confirmed exemption).

---

## S1 — CPU smoke (prereg §4.4.1) — PASS

- `bash -n scripts/slurm/fusioncat_family.sbatch` → **SYNTAX_OK**.
- CONFIGS word-count = **6 rows** (MHC_zh×{0,1,2} + HateMM×{0,1,2}). ✓
- Collision re-check (all must be ABSENT):
  - `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_fuscat*` → **ABSENT** ✓
  - `slurm/logs/*fuscat*.trainlog` → **ABSENT** ✓
  - `logging/Retrieval/*/RAC_video_smoke_fuscat*` → **ABSENT** ✓
- Banked LoRA feature caches present (read-only inputs):
  - ZH `data/CLIP_Embedding/MHC_zh/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` ✓
  - HateMM `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` ✓
- Banked floors present (paired controls):
  - ZH `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog` ✓
  - HateMM `slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog` ✓
- Queue at S1: `squeue -u jehc223` = EMPTY (no 16-CPU concurrency concern; never-2×16-CPU trivially clear).

S1 verdict: **PASS** — proceed to S2 GPU smoke.

---

## S2 — GPU smoke (prereg §4.4.2) — PASS (all asserts)

**Throwaway job `13496`** (`fuscat_smoke`), submitted from a scratchpad-only sbatch (NOT in repo; never committed).
`sacct`: `13496|fuscat_smoke|COMPLETED|0:0|00:35:12` (elapsed dominated by the start-of-job `disk_guard.sh`
B2-verify/prune pass; the 2 head runs themselves are seconds). Initial state was `PENDING (JobHeldUser)` → waited for
auto-release, never forced (DEV-B).

Smoke sbatch = the frozen `run_one` python command with **exactly 2 command-line deltas** (verified by `diff` of the
extracted command block): `--epochs 30`→`3` and `--exp_comment "_${MODEL}_fuscat"`→`"_smoke"`; plus the throwaway
`GROUP_NAME=RAC_video_smoke_fuscat` (a variable definition outside the command block). `--fusion_mode "concat"`
UNCHANGED, `--force False` UNCHANGED. 1 seed × 2 datasets = 2 short runs. **ZERO source edit** (shas re-verified
post-smoke, below).

### S2 asserts — raw evidence

| assert (prereg §4.4.2) | ZH (`MHC_zh`, `…-LoRA_HF`, seed0) | HateMM (`…-LoRA-curric_HF`, seed0) |
|---|---|---|
| (i) completes, no shape error | 3/3 epochs; `Val_Retrieval`=6, `Test_Retrieval`=6 lines; job COMPLETED 0:0 | 3/3 epochs; `Val_Retrieval`=6, `Test_Retrieval`=6 lines; job COMPLETED 0:0 |
| (ii) finite losses (no nan/inf) | nan-count **0**; train Loss 0.786433 → 0.602962 → 0.440625; dev loss 0.6907/0.6952/0.6948 | nan-count **0**; train Loss 0.847019/0.567058 → 0.470125/0.416021 → 0.422306/0.380741; dev loss 0.6944/0.6942/0.6930 |
| (iii)a `grep -m1 "fusion_mode='concat'"` MUST match | **line 1: `fusion_mode='concat'`** ✓ | **line 1: `fusion_mode='concat'`** ✓ |
| (iii)b `grep "fusion_mode='align'"` MUST be empty | **empty** ✓ | **empty** ✓ |

Branch-assert source = the existing `run_rac.py:1065` `print(args)` Namespace echo (line 1 of each trainlog); NO code
edit was made to produce it. The same echo line confirms the inert-key state (F0.8): `sam=False, mod_dropout=False,
head_loss='triplet', mixup=False, nca_tau=0.1, lambda_seg=0.0, archive_feats=None, tarc_target_source='off'` — the
flags-off floor path, plus `proj_dim=1024, map_dim=1024, epochs=3(smoke), warmup=5, topk=20, metric='cos',
loss='triplet', hybrid_loss=True, ce_weight=0.5, majority_voting='arithmetic', force=False`.

`grep -inE "Traceback|RuntimeError|shape|size mismatch|CUDA error|Killed|OOM"` over the smoke `.out` → **no hits**
(DEV-E concat first-Linear 2048→1024 instantiates and trains cleanly).

### S2 cleanup + non-contamination (verified)

- Deleted: `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_smoke_fuscat` (both), both smoke trainlogs, `fuscat_smoke_13496.out`.
- Post-clean re-check: `logging/Retrieval/*/RAC_video_smoke_fuscat*` → **ABSENT**; `slurm/logs/*fuscat*` → **ABSENT**
  (clean slate for the real submit; §4.3 collision conditions restored exactly).
- Banked LoRA caches **untouched** — mtimes bit-identical to the S1 reading (ZH Jul 2 12:08/12:11/12:17; HateMM
  Jul 18 12:26/12:29/12:34), sizes unchanged.
- `git status --porcelain src/` **still empty**; the 4 reused shas re-verified **unchanged** post-smoke
  (`e7b61df4…`/`b85eb72a…`/`2ae7a73f…`/`d43e3bc4…`) ⇒ §4.6 not triggered, freeze still valid.

### S2 observations (transcribed, non-blocking, NOT deviations from the prereg)

- **O-1 (disk_guard, DEV-C expected).** The smoke's `disk_guard.sh` pass pruned B2-**verified** ckpt `.pt` files only
  (e.g. an `RAC_video_zhp` seed2 `last_model_29_*.pt`, sha1-verified at B2 before local delete) and ended with
  `WARNING: still over target (usage=256G > target 250G) after all SAFE steps … MANUAL INTERVENTION required`. It
  explicitly refuses to touch raw datasets. Non-fatal (`|| true`), expected wall-time padding — **not a failure**.
- **O-2 (ZH floor dir absent; floor numbers unaffected).** `logging/Retrieval/MHC_zh/RAC_video_b3_lora` does not
  exist. This is **pre-existing** (the earlier documented `lora_p9`+`Retrieval` deletion after verified B2 backup),
  **not** caused by this smoke: disk_guard prunes individual B2-verified ckpt files, never whole group dirs, and the
  dir was already absent at the first post-smoke listing. The prereg's ZH paired control is derived from the **raw
  trainlogs** `enc3s_MHC_zh_…_13150.trainlog` (§2.1) — all 3 present and byte-intact (verified S1). Likewise the
  HateMM floor trainlogs `…_13241.trainlog` (§2.2) are present; `logging/Retrieval/HateMM/RAC_video_lora_curric`
  also still exists. No floor evidence was lost.

S2 verdict: **PASS** on every declared assert (branch-taken both datasets, finite losses both datasets, no shape
error, throwaways removed, banked inputs untouched) ⇒ authorized to proceed to S3 real submission.

---

## S3 — REAL SUBMISSION — job `13514`

**Submit-instant gates (all re-run immediately before `sbatch`; ALL PASS):**

```
c88332b8972e3270081600d0a8cb892a8d24afefbc73e378a5a3104a433c0830  refine-logs/FUSIONCAT_PREREG.md      ✓ freeze
62bfb773beeec325096362c86bae1aca8b90c94804ba3798973bbc88e82517fc  scripts/slurm/fusioncat_family.sbatch ✓ freeze
e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378  src/model/classifier.py              ✓ freeze
b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3  src/run_rac.py                       ✓ freeze
2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b  src/model/loss.py                    ✓ freeze
d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57  src/utils/retrieval.py               ✓ freeze
git status --porcelain src/  → empty (CLEAN)                                                           ✓
```

- **Never-two-16-CPU (DEV-F):** `squeue -u jehc223` at submit instant = **EMPTY** (zero jobs queued/running). This
  job is **8-CPU** ⇒ the submit-time aggregate wedge rule clears trivially.
- **Collision (DEV-D) re-check at submit instant:** `logging/Retrieval/*/RAC_video_fuscat*` **ABSENT**,
  `slurm/logs/*fuscat*` **ABSENT** (smoke residue fully removed) ⇒ `--force False` cannot trip the
  `run_rac.py:1059-1062` hard-abort, and nothing banked can be overwritten.

**Command (ONE bite, UNMODIFIED frozen artifact, exactly one `sbatch`):**

```
$ sbatch scripts/slurm/fusioncat_family.sbatch
Submitted batch job 13514
$ squeue -u jehc223
     13514  fuscat  PENDING  8  (JobHeldUser)
```

**JOB ID = `13514`.** 6 head runs = {ZH `Qwen2.5-VL-7B-Instruct-LoRA_HF`, HateMM
`Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`} × seed{0,1,2}, `--fusion_mode concat`, group `RAC_video_fuscat`, ~0.1 GPU-h.
Initial `PENDING (JobHeldUser)` = expected (DEV-B) → **waited for auto-release, never forced**.
No second submission was made under this prereg (§3.6 one bite).

---

## S4 — RAW TRANSCRIPTION (job `13514`) — NO GATES, NO INTERPRETATION APPLIED

> Per prereg §6/§3.7 the executor transcribes raw both-protocol per-seed numbers and applies **no** decision rule.
> Everything below is a measurement or an arithmetic difference. **No pass/fail/kill language appears in this
> section by design** — the verdict is rendered by an independent 0-context reviewer against the frozen prereg.

**Terminal state:** `sacct 13514` → `13514|fuscat|COMPLETED|0:0|00:07:18` (and `13514.batch|COMPLETED|0:0`).
6/6 trainlogs written. Job `.out` = `slurm/logs/fuscat_13514.out`: 6 `########## RUN:` banners (lines 28, 311, 590,
872, 1181, 1487), `======== fuscat ALL DONE (13514) ========` at line 1798, `[b2_push] done -> b2:junyi-data/
RGCL_video/logs/fuscat` at line 6918. Error scan (`Traceback|RuntimeError|size mismatch|CUDA error|Killed|OOM|already
exists`) over the full `.out` → **no hits**. nan/inf count in all 6 trainlogs → **0** each.

### S4.0 Branch-assert evidence — the concat branch was taken in ALL 6 runs

Source = the existing `run_rac.py:1065` `print(args)` Namespace echo, line **1** of every trainlog (no code edit).

| run (trainlog, all `…_13514.trainlog`) | `grep -c "fusion_mode='concat'"` | first-match line | `grep -c "fusion_mode='align'"` |
|---|---|---|---|
| `fuscat_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed0` | **1** | 1 | **0** |
| `fuscat_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed1` | **1** | 1 | **0** |
| `fuscat_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed2` | **1** | 1 | **0** |
| `fuscat_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed0` | **1** | 1 | **0** |
| `fuscat_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed1` | **1** | 1 | **0** |
| `fuscat_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed2` | **1** | 1 | **0** |

(Same echo re-confirms the inert-key/flags-off state per F0.8 and the pinned knobs: `epochs=30, warmup=5, topk=20,
proj_dim=1024, map_dim=1024, metric='cos', loss='triplet', hybrid_loss=True, ce_weight=0.5,
majority_voting='arithmetic', force=False, group_name='RAC_video_fuscat'`.)

### S4.1 Parsing + provenance method

- **Primary (embedded) parser:** the frozen sbatch's own readout (`fusioncat_family.sbatch:70-89`), whose
  `RESULT_ROW` lines are at `.out` lines **309, 588, 870, 1179, 1485, 1796**.
- **Independent cross-parse:** a separately written line-oriented parser (scratchpad, not committed) implementing the
  same protocol by token-splitting rather than one regex. It reproduced **all 6 runs × both protocols ×
  acc/mF1/roc/epoch bit-exactly** (agreement 12/12 readouts, 4dp) — independent confirmation of every number below.
- **Line numbers:** every value was re-read from the raw trainlog with `grep -n`. **Provenance note:** these
  trainlogs contain tqdm `\r` carriage returns, so a Python universal-newline reader numbers lines differently from
  `grep`/`sed`. **All line numbers cited below are `grep -n` (`\n`-based) numbers of the `macroF1`-bearing
  `Test_Retrieval` line** and were verified by re-reading each cited line.
- **Protocol (prereg §2 / §3.1):** val-sel = among epochs ≥ warmup 5, max **Val**`_Retrieval` acc with **roc**
  tie-break, read Test at that epoch; final-ep = max epoch (29). **Selection used Val only** (DEV-A no-peek).
- **Selection audit (Val-only, per run):** ZH s0 ep25 (val 0.8590/roc 0.9207 — roc-max among six ep≥5 rows tied at
  val acc 0.8590), ZH s1 ep8 (val 0.8590/0.9200), ZH s2 ep26 (val 0.8718/0.9171, unique max), HateMM s0 ep15
  (val 0.8598/0.9135, roc-max over ep20's 0.9055 at the same acc), HateMM s1 ep5 (val 0.8411/0.9193, roc-max among
  five rows tied at 0.8411), HateMM s2 ep18 (val 0.8598/0.8943, unique max). All 6 tie-breaks re-derived from the raw
  Val rows.

### S4.2 ZH (`MHC_zh`, `Qwen2.5-VL-7B-Instruct-LoRA_HF`) — concat 13514 vs banked align floor 13150 (§2.1)

| seed | protocol | ep | concat acc/mF1 | trainlog line | floor acc/mF1 | Δ(concat−floor) acc/mF1 |
|---|---|---|---|---|---|---|
| 0 | val-sel | 25 | 0.8389 / 0.8135 | 241 | 0.8322 / 0.8023 | +0.0067 / +0.0112 |
| 1 | val-sel | 8 | 0.8456 / 0.8133 | 101 | 0.8255 / 0.7956 | +0.0201 / +0.0177 |
| 2 | val-sel | 26 | 0.8322 / 0.8068 | 248 | 0.8389 / 0.8065 | −0.0067 / +0.0003 |
| **mean** | **val-sel** | | **0.8389 / 0.8112** | | **0.8322 / 0.8015** | **+0.0067 / +0.0097** (sign acc 2/3 pos, mF1 3/3 pos) |
| 0 | final-ep | 29 | 0.8389 / 0.8135 | 274 | 0.8456 / 0.8181 | −0.0067 / −0.0046 |
| 1 | final-ep | 29 | 0.8456 / 0.8181 | 270 | 0.8389 / 0.8113 | +0.0067 / +0.0068 |
| 2 | final-ep | 29 | 0.8389 / 0.8135 | 273 | 0.8523 / 0.8226 | −0.0134 / −0.0091 |
| **mean** | **final-ep** | | **0.8411 / 0.8150** | | **0.8456 / 0.8173** | **−0.0045 / −0.0023** (sign acc 1/3 pos, mF1 1/3 pos) |

Test roc (transcribed, not part of any bar): val-sel 0.9325 / 0.9348 / 0.9158; final-ep 0.8944 / 0.9124 / 0.8989.

### S4.3 HateMM (`Qwen2.5-VL-7B-Instruct-LoRA-curric_HF`) — concat 13514 vs banked align floor 13241 (§2.2)

| seed | protocol | ep | concat acc/mF1 | trainlog line | floor acc/mF1 | Δ(concat−floor) acc/mF1 |
|---|---|---|---|---|---|---|
| 0 | val-sel | 15 | 0.8791 / 0.8730 | 173 | 0.8791 / 0.8730 | +0.0000 / +0.0000 |
| 1 | val-sel | 5 | 0.8698 / 0.8632 | 80 | 0.8744 / 0.8678 | −0.0046 / −0.0046 |
| 2 | val-sel | 18 | 0.8744 / 0.8672 | 202 | 0.8791 / 0.8724 | −0.0047 / −0.0052 |
| **mean** | **val-sel** | | **0.8744 / 0.8678** | | **0.8775 / 0.8711** | **−0.0031 / −0.0033** (sign acc 0/3 pos, mF1 0/3 pos) |
| 0 | final-ep | 29 | 0.8698 / 0.8626 | 300 | 0.8791 / 0.8730 | −0.0093 / −0.0104 |
| 1 | final-ep | 29 | 0.8791 / 0.8724 | 297 | 0.8791 / 0.8724 | +0.0000 / +0.0000 |
| 2 | final-ep | 29 | 0.8791 / 0.8724 | 302 | 0.8791 / 0.8724 | +0.0000 / +0.0000 |
| **mean** | **final-ep** | | **0.8760 / 0.8691** | | **0.8791 / 0.8726** | **−0.0031 / −0.0035** (sign acc 0/3 pos, mF1 0/3 pos) |

Test roc (transcribed): val-sel 0.9401 / 0.9162 / 0.9242; final-ep 0.9246 / 0.9239 / 0.9239.

Floor columns are the prereg's own §2.1/§2.2 banked values (re-derived there from the 13150/13241 trainlogs, which
this executor verified present and untouched — mtimes unchanged Jul 14 18:55-18:56 / Jul 18 12:43-12:44). Δ columns
are plain subtraction; means are the 3-seed arithmetic means of the concat column and of the per-seed Δ.

### S4.4 Post-run integrity

- `git status --porcelain src/` **empty**; all 6 frozen shas **re-verified unchanged after the run**
  (`c88332b8…`, `62bfb773…`, `e7b61df4…`, `b85eb72a…`, `2ae7a73f…`, `d43e3bc4…`) ⇒ §4.6 never triggered.
- Banked LoRA caches unchanged (mtimes identical to S1/S2 readings); banked floor trainlogs 13150/13241 unchanged.
- New artifacts written by this job (expected): `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_fuscat`, the 6
  `fuscat_*_13514.trainlog` files, `slurm/logs/fuscat_13514.out`, and the B2 push of derived `logging` only
  (`b2:junyi-data/RGCL_video/logs/fuscat`; raw videos never leave the machine).
- Test-touch spent: exactly the **6** budgeted dataset×seed evaluations. No re-run, no second bite.

---

## S5 — CLOSEOUT

### 5.1 Stage outcomes

| stage | outcome | evidence |
|---|---|---|
| S0 authorization | all 6 shas match freeze byte-exact; `src/` git-clean; no VOID condition | §"Authorization (S0)" |
| S1 CPU smoke | PASS — SYNTAX_OK, CONFIGS=6, 3 collision classes ABSENT, caches + floors present | §S1 |
| S2 GPU smoke | PASS — throwaway job **13496** COMPLETED 0:0; concat branch-assert + finite losses on BOTH datasets; throwaways deleted; banked inputs untouched | §S2 |
| S3 real submission | ONE unmodified `sbatch` → job **13514**; submit-instant shas match; queue empty (8-CPU) | §S3 |
| S4 raw transcription | job **13514** COMPLETED 0:0 (07:18); 6/6 runs; branch-assert 6/6; dual-protocol per-seed numbers with `grep -n` line refs; independent cross-parse agrees 12/12 bit-exactly | §S4 |

### 5.2 Job IDs

- **13496** — throwaway GPU smoke (2 short concat runs, 3 epochs). COMPLETED 0:0, 00:35:12. Artifacts deleted.
- **13514** — the pre-registered family (6 head runs). COMPLETED 0:0, 00:07:18. **This is the family's single bite.**

### 5.3 Full sha chain (identical at freeze → submit-instant → post-run)

```
refine-logs/FUSIONCAT_PREREG.md        c88332b8972e3270081600d0a8cb892a8d24afefbc73e378a5a3104a433c0830
scripts/slurm/fusioncat_family.sbatch  62bfb773beeec325096362c86bae1aca8b90c94804ba3798973bbc88e82517fc
src/model/classifier.py                e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378
src/run_rac.py                         b85eb72a690bc8fccc2ff5d5358fd6523359bf6596d2b2a0d6d0701bec9e53e3
src/model/loss.py                      2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b
src/utils/retrieval.py                 d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57
git status --porcelain src/            empty at all three checkpoints
```

ZERO source-code diff across the whole chain ⇒ the prereg's F0.7 zero-code premise held; §4.6 (code-fix ⇒ re-freeze)
was never triggered; the §4.5 codex-gate exemption stands as pre-declared.

### 5.4 Deviations from the frozen chain

**None material.** Full disclosure of every executor choice not literally spelled out in the prereg:

1. **D-1 (smoke sbatch location).** Prereg §4.4.2 specifies the throwaway commands but not a file. The executor
   wrapped them in a scratchpad-only sbatch (SLURM policy: no GPU work outside `sbatch`), never added to the repo,
   never committed. Its `run_one` was `diff`-verified against the frozen command block = exactly 2 deltas
   (`--epochs 30`→`3`, `--exp_comment`→`"_smoke"`) plus the throwaway `GROUP_NAME`. `--fusion_mode "concat"` and
   `--force False` untouched.
2. **D-2 (smoke breadth).** Ran 1 seed × 2 datasets (= the prereg's "GPU throwaway concat run **per dataset**"),
   3 epochs, as specified. No extra seeds.
3. **D-3 (line-number convention).** Cited line numbers are `grep -n` (`\n`-based), not Python universal-newline
   numbers, because tqdm `\r` makes the two differ; stated inline in §S4.1 so any reader can reproduce each number
   with `sed -n '<N>p'`. The metric values themselves are unaffected and were double-parsed.
4. **D-4 (watcher restart).** The first background watcher for 13514 expired silently; a persistent monitor was
   re-armed and caught the terminal transition. No effect on the job (SLURM-side only); no forcing of the
   `JobHeldUser` hold at any point (both 13496 and 13514 waited for auto-release).

### 5.5 Discipline statements

- **NO `state/` mutation**, **NO `research-wiki/` mutation**, **NO push** — commits are local on `main` only.
- **NO verdict rendered here.** §S4 is raw transcription + arithmetic only; the KS-arm-dead and FORMAL gates (§3.2,
  §3.3) and any pass/fail wording are for the **independent 0-context reviewer** against the frozen prereg VERBATIM.
- One pre-registered family, **one** submission, **6** budgeted test reads consumed; scope frozen per §3.6 (no
  `cross`/gated arm, no param-matched control, no third dataset — each would be a new prereg and a new bite).
- Standing vetoes honoured: own-train-split only, no cross-dataset mixing, no OCR, no cross-seed ensemble, raw video
  never left the machine (B2 push carried derived `logging` only).
