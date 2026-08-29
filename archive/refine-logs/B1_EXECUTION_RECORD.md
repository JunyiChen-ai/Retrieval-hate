# B1 Execution Record — `enc3seed_zh_b1.sbatch`

**Executor:** B1 prep/implementation agent · **Date:** 2026-07-14

## Authorization citation

Executed under the **CONDITIONAL EXECUTION AUTHORIZATION (B1)** granted 2026-07-14 by
the B1 pre-registration reviewer (`refine-logs/B1_PREREG_REVIEW.md`, final section):
scope = exactly ONE submission of `sbatch scripts/slurm/enc3seed_zh_b1.sbatch`, under
conditions (i) hash pinning, (ii) FORCE=False no-collision re-check at submit time,
(iii) single-submit discipline. Delta-check verdict D.5: APPROVED; cosmetic remnants
ruled IMMATERIAL, left as-is per ruling.

## (i) Submit-time re-hash — MATCH

Run 2026-07-14 ~10:04 NZST, cwd `/data/jehc223/RGCL`:

| file | sha256 (measured at submit) | matches authorization |
|---|---|---|
| `research-wiki/experiments/exp-encoder-zh-b1.md` | `91982eb333e61efc34e62794031f6b3f8b672e34ffee7d558fa03e1b2b57972b` | ✅ identical |
| `scripts/slurm/enc3seed_zh_b1.sbatch` | `9504dba00ad1ae8351bedbe1ebcd1b5bf1382374273c27df7db7d521f0cbd762` | ✅ identical |

## (ii) Submit-time FORCE=False no-collision re-check — PASS

- `logging/Retrieval/MHC_zh/RAC_video_archive_seeds/` contains **9 dirs, all
  `*-LoRA_HF*`** (seeds 0-4 LoRA-only + seeds 1-4 `_arc-knn-a0.25`); dirs matching the
  frozen tags `*_hybrid_loss_openai_clip-vit-large-patch14-336_HF` or
  `*_hybrid_loss_Qwen2.5-VL-7B-Instruct_HF`: **NONE** (`ls` no-such-file; non-LoRA
  count = 0).
- `slurm/logs/enc3s_MHC_zh*`: **NONE** (no log collision).
- No smoke run was performed between authorization and submit (no partial dirs).

## (iii) Submission

| field | value |
|---|---|
| command | `sbatch scripts/slurm/enc3seed_zh_b1.sbatch` (single invocation, no `--time`) |
| submit timestamp | **2026-07-14T10:05:07+12:00** |
| job id | **13115** |
| job name / stdout | `enc3seed` → `slurm/logs/enc3seed_13115.out` |
| initial state | **RUNNING** from 10:05:16 (no `JobHeldUser` hold occurred) |
| sacct Submit/Start | 2026-07-14T10:05:07 / 2026-07-14T10:05:16 |

Discipline: no resubmission after any terminal state; no mid-run intervention; waiter
polls `sacct` every 120 s, cap ~2 h (expected wall < 20 min).

## Remnant note (per delta-check ruling, D.2)

**Remnant #1:** the runner's header comment says "Runs 10 configs serially" — this is a
verbatim-copy remnant from the parent `enc3seed.sbatch`; **job 13115 actually runs 6
configs** (`MHC_zh × {CLIP frozen, Qwen2.5-VL-7B-Instruct_HF frozen} × seeds {0,1,2}`).
Comment-only, zero runtime effect; ruled IMMATERIAL and left as-is because editing the
file would void authorization hash (i). A future reader of `enc3seed_13115.out` should
expect exactly 6 `########## RUN:` blocks.
(Remnant #2, job-name `enc3seed` instead of `enc3seed_zh_b1`, likewise immaterial —
`%j` prevents any .out collision with `enc3seed_12850.out`.)

## Expected outputs (for terminal-state verification)

```
slurm/logs/enc3s_MHC_zh_openai_clip-vit-large-patch14-336_HF_seed{0,1,2}_13115.trainlog
slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed{0,1,2}_13115.trainlog
```

Each with 30 epochs and parseable VALSEL/FINAL `RESULT_ROW` readout lines.

## Terminal state (sacct, read back 2026-07-14)

| field | value |
|---|---|
| State / ExitCode | **COMPLETED / 0:0** |
| Start / End / Elapsed | 2026-07-14T10:05:16 / 2026-07-14T10:16:37 / **00:11:21** |

Stdout `slurm/logs/enc3seed_13115.out`: exactly **6** `########## RUN:` blocks (remnant
#1 note applies: header comment says 10), each followed by parseable VALSEL/FINAL
readout lines; ends `======== enc3seed ALL DONE (13115) ========` (`.out:1698`).
No `NO_PARSE` occurrences.

## Trainlog verification — all 6 exist, sane

All at `slurm/logs/`, each with 30 Test_Retrieval epoch lines (0..29), correct
Namespace header (dataset='MHC_zh', expected model=, seed=, topk=20, epochs=30,
lambda_seg=0.0, archive_feats=None) and correct feature dims (CLIP 1024/768;
Qwen 3584/3584):

```
enc3s_MHC_zh_openai_clip-vit-large-patch14-336_HF_seed0_13115.trainlog  (25,681 B)
enc3s_MHC_zh_openai_clip-vit-large-patch14-336_HF_seed1_13115.trainlog  (25,463 B)
enc3s_MHC_zh_openai_clip-vit-large-patch14-336_HF_seed2_13115.trainlog  (25,147 B)
enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed0_13115.trainlog             (25,528 B)
enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed1_13115.trainlog             (25,223 B)
enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed2_13115.trainlog             (24,902 B)
```

## Raw results (transcribed from the raw trainlogs; verified against the .out readouts — identical on all 12 readings)

**RAW DATA ONLY — no interpretation, no gate application. Gates 1a/1b + decision rule
= verdict processing (orchestrator + independent review), not this record.**

Selection rule as registered: val-sel = epoch >= warmup 5 maximizing Val_Retrieval acc
(roc tie-break); final = epoch 29. Re-derived independently from the raw logs by this
executor; matched the sbatch parser's VALSEL/FINAL lines in `enc3seed_13115.out`
exactly.

### frozen-CLIP arm (`openai_clip-vit-large-patch14-336_HF`)

| seed | protocol | epoch | Test macroF1 | Test acc | Test roc | provenance (`slurm/logs/`) |
|---|---|---|---|---|---|---|
| 0 | val-sel | 29 | 0.7706 | 0.8054 | 0.8382 | `enc3s_MHC_zh_openai_clip-vit-large-patch14-336_HF_seed0_13115.trainlog:305` (val line `:304`, val acc 0.8077) |
| 0 | final | 29 | 0.7706 | 0.8054 | 0.8382 | same file `:305` (val-sel epoch = final epoch) |
| 1 | val-sel | 28 | 0.7579 | 0.8054 | 0.8346 | `..._seed1_13115.trainlog:294` (val line `:293`, val acc 0.7821) |
| 1 | final | 29 | 0.7542 | 0.8054 | 0.8342 | same file `:304` |
| 2 | val-sel | 25 | 0.7742 | 0.8121 | 0.8419 | `..._seed2_13115.trainlog:264` (val line `:263`, val acc 0.8205) |
| 2 | final | 29 | 0.7913 | 0.8322 | 0.8444 | same file `:301` |

### frozen-Qwen arm (`Qwen2.5-VL-7B-Instruct_HF`)

| seed | protocol | epoch | Test macroF1 | Test acc | Test roc | provenance (`slurm/logs/`) |
|---|---|---|---|---|---|---|
| 0 | val-sel | 22 | 0.7412 | 0.7919 | 0.8838 | `enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed0_13115.trainlog:241` (val line `:240`, val acc 0.8205) |
| 0 | final | 29 | 0.7864 | 0.8188 | 0.8906 | same file `:305` |
| 1 | val-sel | 25 | 0.7871 | 0.8121 | 0.8874 | `..._seed1_13115.trainlog:265` (val line `:264`, val acc 0.8718) |
| 1 | final | 29 | 0.7759 | 0.8054 | 0.8951 | same file `:302` |
| 2 | val-sel | 28 | 0.7759 | 0.8054 | 0.8940 | `..._seed2_13115.trainlog:289` (val line `:288`, val acc 0.8462) |
| 2 | final | 29 | 0.7514 | 0.7852 | 0.8806 | same file `:299` |

### Reference readings relevant to gates 1a/1b (recorded for the verdict processor; NOT applied here)

- Gate 1a reference (frozen-Qwen ZH s0, old code, `rgcl_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_1151518.trainlog`): val-sel e22 = 0.7412 F1 / 0.7919 acc; final e29 = 0.7864 F1 / 0.8188 acc.
- Gate 1b reference (frozen-CLIP ZH λ=0 floor s0, job 12130, cross-runner): val-sel = final = 0.7706 F1 / 0.8054 acc (ep29).
