# B3 Execution Record — `enc3seed_zh_b3.sbatch`

**Executor:** B3 prep/execution agent · **Date:** 2026-07-14

## Authorization citation

Executed under the **CONDITIONAL EXECUTION AUTHORIZATION (B3)** granted 2026-07-14 by the
B3 pre-registration reviewer (`refine-logs/B3_PREREG_REVIEW.md` §9, verdict §8 APPROVED):
scope = exactly ONE submission of `sbatch scripts/slurm/enc3seed_zh_b3.sbatch`, under
conditions (i) hash pinning, (ii) FORCE=False no-collision re-check at submit time,
(iii) single-submit discipline. Erratum **F-1** (§7d −0.0113 → −0.0112) and the **§2.2
binding marginal-pass reporting ruling** are handled at verdict-processing time, NOT by
this executor; both hashed files were left untouched as authorized (editing either would
void the hashes). Optional smoke skipped per authorization ("if in doubt, skip the smoke —
the cache dims are already CPU-verified").

## (i) Submit-time re-hash — MATCH

Run 2026-07-14 16:33:08 NZST, cwd `/data/jehc223/RGCL`:

| file | sha256 (measured at submit) | matches authorization |
|---|---|---|
| `research-wiki/experiments/exp-lora-zh-b3.md` | `71745cf29de7f03a2bd4d351b30b02637a8d250f493dfb7f49d3459c44f7d802` | ✅ identical |
| `scripts/slurm/enc3seed_zh_b3.sbatch` | `4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad` | ✅ identical |

## (ii) Submit-time FORCE=False no-collision re-check — PASS (all three clauses)

- `logging/Retrieval/MHC_zh/RAC_video_b3_lora*`: **does not exist** (`ls` no-such-file). ✅
- `slurm/logs/enc3s_MHC_zh_*LoRA_HF*`: **does not exist** (no trainlog collision). ✅
- Arcbase anchor dirs **intact** — all three of
  `logging/Retrieval/MHC_zh/RAC_video_archive_seeds/RAC_..._seed{0,1,2}_hybrid_loss_Qwen2.5-VL-7B-Instruct-LoRA_HF`
  exist (the G-repro anchors 12223-25 are preserved; the fresh group writes elsewhere). ✅
- No smoke run was performed between authorization and submit (no partial dirs).

## (iii) Submission

| field | value |
|---|---|
| command | `sbatch scripts/slurm/enc3seed_zh_b3.sbatch` (single invocation, no `--time`) |
| submit timestamp | **2026-07-14T16:33:20+12:00** |
| job id | **13150** |
| job name / stdout | `enc3seed` → `slurm/logs/enc3seed_13150.out` (job-name remnant per impl-notes §a; `%j` prevents overwrite of `enc3seed_{12850,13115}.out`) |
| initial state | **PENDING (JobHeldUser)** at 16:33 — per project rule, WAIT for auto-release; never force |

Discipline: single submit; no resubmission after any terminal state; no mid-run
intervention; tracked background waiter (see waiter log below).

| waiter event | value |
|---|---|
| waiter 1 armed | 2026-07-14 ~16:34, poll 60 s, cap 1 h |
| waiter 1 expired | job still `PENDING (JobHeldUser)` at 1h08m (17:41) — cap reached, NO terminal state, no action taken |
| waiter 2 re-armed (orchestrator nudge) | 2026-07-14 17:42, poll 120 s, cap 4 h (historical `JobHeldUser` holds have run ~4.5 h; NEVER force-release, per project rule) |

## Expected outputs (for terminal-state verification)

```
slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog
```

Each with 30 Test_Retrieval epochs (0..29) and parseable VALSEL/FINAL `RESULT_ROW` lines.
New output dirs expected under `logging/Retrieval/MHC_zh/RAC_video_b3_lora/` only.

## Terminal state — PENDING (waiter armed; to be filled on completion)

*(to be appended on terminal state: sacct State/ExitCode/Start/End/Elapsed, trainlog
verification, and the raw both-protocol per-seed transcription table — RAW numbers only,
no gates, no interpretation, per executor obligations. Any FAILED state = HALT + evidence,
no resubmit.)*

## Results (job 13150) — raw transcription

**RAW DATA ONLY — no verdict, no gate application, no comparison against the 13115 CLIP
arm. Delta/gate/decision-rule processing is the independent verdict reviewer's job.**

Terminal state (per orchestrator completion notification): **COMPLETED, exit 0, elapsed
2m46s**. Stdout `slurm/logs/enc3seed_13150.out` ends `======== enc3seed ALL DONE (13150)
========` (`.out:847`); three `########## RUN:` blocks, one per seed; no traceback / NaN /
OOM / `NO_PARSE`.

### Completion-integrity per trainlog

- `enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed0_13150.trainlog` — **completed
  normally**: Namespace echoes `model='Qwen2.5-VL-7B-Instruct-LoRA_HF'`, `dataset='MHC_zh'`,
  `seed=0`, `group_name='RAC_video_b3_lora'`, `epochs=30`, `warmup=5`; feature dims 3584/3584
  (head builds `Linear(in_features=3584,…)`); reached epoch 29 + `Last Epoch, saving...`
  (`:274`); no traceback.
- `…seed1_13150.trainlog` — **completed normally**: same config echo with `seed=1`; reached
  epoch 29 + `Last Epoch, saving...` (`:275`); no traceback.
- `…seed2_13150.trainlog` — **completed normally**: same config echo with `seed=2`; reached
  epoch 29 + `Last Epoch, saving...` (`:270`); no traceback.

### Selection rule (as registered, re-derived from the raw logs)

Val-sel = epoch ≥ warmup 5 maximizing **Val_Retrieval acc** (roc tie-break); final = epoch
29. Independently re-derived here from the raw `Current Epoch Val_Retrieval … Best model so
far, saving...` markers; matches the sbatch parser's `VALSEL`/`FINAL`/`RESULT_ROW` lines in
`enc3seed_13150.out` **exactly to all 4 printed decimals on all 12 readings (3 seeds × 2
protocols × acc,F1)**. "Test F1" = **macroF1** (the metric the `RESULT_ROW` parser emits),
i.e. the `Test_Retrieval Epoch NN macroF1: …` line; the binary `f1:` from the sibling
`Test_Retrieval Epoch NN acc: … f1:` line is also quoted below for completeness.

### RAW per-seed table — LoRA arm (`Qwen2.5-VL-7B-Instruct-LoRA_HF`, 3584-d), group `RAC_video_b3_lora`, job 13150

| seed | protocol | epoch | Test macroF1 | Test acc | Test roc | Test binary-f1 | provenance (`slurm/logs/…seed<s>_13150.trainlog`) |
|---|---|---|---|---|---|---|---|
| 0 | val-sel | 20 | 0.8023 | 0.8322 | 0.8825 | 0.7253 | seed0 `:199` (macro) / `:197` (binary); save marker `:201` (Val acc 0.8462… **8718** → see note) |
| 0 | final | 29 | 0.8181 | 0.8456 | 0.9036 | 0.7473 | seed0 `:272` (macro) / `:270` (binary); `Last Epoch, saving...` `:274` |
| 1 | val-sel | 26 | 0.7956 | 0.8255 | 0.9004 | 0.7174 | seed1 `:248` (macro) / `:246` (binary); save marker `:250` (Val acc 0.8718) |
| 1 | final | 29 | 0.8113 | 0.8389 | 0.8955 | 0.7391 | seed1 `:273` (macro) / `:271` (binary); `Last Epoch, saving...` `:275` |
| 2 | val-sel | 19 | 0.8065 | 0.8389 | 0.8838 | 0.7273 | seed2 `:187` (macro) / `:185` (binary); save marker `:189` (Val acc 0.8718) |
| 2 | final | 29 | 0.8226 | 0.8523 | 0.9115 | 0.7500 | seed2 `:268` (macro) / `:266` (binary); `Last Epoch, saving...` `:270` |

(Note on the seed-0 val-sel epoch: the last `Best model so far, saving...` marker is at
seed0 `:201`, printed **after** the epoch-20 block, and its `Current Epoch Val_Retrieval
acc: 0.8717948717948718` equals the epoch-20 `Val_Retrieval Epoch 20 acc: 0.8718` — so the
val-selected checkpoint is epoch 20, not the epoch-9 marker `:112` that also read 0.8718;
epoch 20 wins the roc tie-break 0.9229 > 0.9207.)

### Exact source lines (blockquotes)

**seed 0** — `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed0_13150.trainlog`

> `:197` `Test_Retrieval Epoch 20 acc: 0.8322 roc: 0.8825 pre: 0.7174 recall: 0.7333 f1: 0.7253`
> `:199` `Test_Retrieval Epoch 20 macroF1: 0.8023 macroP: 0.8004 macroR: 0.8042 acc: 0.8322 roc: 0.8825`
> `:201` `Current Epoch Val_Retrieval acc:  0.8717948717948718 roc:  0.9228571428571429 Best model so far, saving...`
> `:270` `Test_Retrieval Epoch 29 acc: 0.8456 roc: 0.9036 pre: 0.7391 recall: 0.7556 f1: 0.7473`
> `:272` `Test_Retrieval Epoch 29 macroF1: 0.8181 macroP: 0.8162 macroR: 0.8201 acc: 0.8456 roc: 0.9036`
> `:274` `Last Epoch, saving...`

**seed 1** — `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed1_13150.trainlog`

> `:246` `Test_Retrieval Epoch 26 acc: 0.8255 roc: 0.9004 pre: 0.7021 recall: 0.7333 f1: 0.7174`
> `:248` `Test_Retrieval Epoch 26 macroF1: 0.7956 macroP: 0.7922 macroR: 0.7994 acc: 0.8255 roc: 0.9004`
> `:250` `Current Epoch Val_Retrieval acc:  0.8717948717948718 roc:  0.9128571428571428 Best model so far, saving...`
> `:271` `Test_Retrieval Epoch 29 acc: 0.8389 roc: 0.8955 pre: 0.7234 recall: 0.7556 f1: 0.7391`
> `:273` `Test_Retrieval Epoch 29 macroF1: 0.8113 macroP: 0.8078 macroR: 0.8153 acc: 0.8389 roc: 0.8955`
> `:275` `Last Epoch, saving...`

**seed 2** — `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed2_13150.trainlog`

> `:185` `Test_Retrieval Epoch 19 acc: 0.8389 roc: 0.8838 pre: 0.7442 recall: 0.7111 f1: 0.7273`
> `:187` `Test_Retrieval Epoch 19 macroF1: 0.8065 macroP: 0.8108 macroR: 0.8027 acc: 0.8389 roc: 0.8838`
> `:189` `Current Epoch Val_Retrieval acc:  0.8717948717948718 roc:  0.9085714285714286 Best model so far, saving...`
> `:266` `Test_Retrieval Epoch 29 acc: 0.8523 roc: 0.9115 pre: 0.7674 recall: 0.7333 f1: 0.7500`
> `:268` `Test_Retrieval Epoch 29 macroF1: 0.8226 macroP: 0.8271 macroR: 0.8186 acc: 0.8523 roc: 0.9115`
> `:270` `Last Epoch, saving...`

### sbatch `RESULT_ROW` cross-check (`slurm/logs/enc3seed_13150.out`, verbatim)

> `:286` `RESULT_ROW	…seed0…	valsel	20	0.8023	0.8322	0.8825	final	29	0.8181	0.8456	0.9036`
> `:568` `RESULT_ROW	…seed1…	valsel	26	0.7956	0.8255	0.9004	final	29	0.8113	0.8389	0.8955`
> `:845` `RESULT_ROW	…seed2…	valsel	19	0.8065	0.8389	0.8838	final	29	0.8226	0.8523	0.9115`

All three `RESULT_ROW` lines agree with the trainlog macro-line transcription above to all
4 decimals. **Transcription complete — no interpretation applied.**
