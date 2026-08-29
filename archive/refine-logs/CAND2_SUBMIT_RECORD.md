# CAND-2 Curriculum LoRA-SFT — SUBMIT RECORD (submit executor)

**Role:** submit executor. ZERO user interaction. NO push. NO test metric read. NO verdict produced.
**Date:** 2026-07-18.
**Freeze:** `refine-logs/CAND2_FREEZE.md`, commit `7804324` ("freeze: cand-2 curriculum prereg + artifacts
locked"). Freeze = PASS (all freeze-block + reused-machinery + DEV-1 fork-source shas match; builder bit-exact
idempotent; K-C2-0 PASS both).
**Prereg:** `CAND2_CURRICULUM_PREREG.md` commit `76ef0e2`, sha256 `e5a689d9…f939790e`.
**Review:** `CAND2_PREREG_REVIEW.md` commit `c1315cb`, APPROVED-WITH-NOTES.

---

## 1. SFT smoke (prereg §4.4 step 1) — PASS, cleaned up

- **Job 13236** (`lora_sft_smoke_curric`), throwaway config `smoke_curric_sft.yaml` (max_steps 20, save_steps 20,
  `output_dir logging/lora/_smoke_curric`), submitted via `sbatch` (NO `--time`). State: **COMPLETED** (exit 0:0),
  runtime 22m36s wall (12m01s train after ~10m hold+load).
- **Dataset echoed:** `mhc_zh_lora_curric_train` → `Num examples = 579` (== ZH curric N); `max_steps` override
  applied (`Total optimization steps = 20`); recipe r16/alpha32 (frozen curric config).
- **Loss finite + decreasing (raw `trainer_log.jsonl`):** step5 `0.2868` → step10 `0.2240` → step15 `0.1930` →
  step20 `0.2238`; `train_loss 0.23189724`. No NaN/inf/traceback/OOM (clean scan). Well inside the §4.1a
  ~0.12–0.18 band expectation for a partial (0.28-epoch) run; monotone-ish downtrend, final-step wiggle is normal.
- **Checkpoint written:** `logging/lora/_smoke_curric/checkpoint-20/adapter_model.safetensors` (161 MB) +
  `optimizer.pt`/`scheduler.pt`/`trainer_state.json`; final adapter + `train_results.json` at output root.
- **Cleanup:** `logging/lora/_smoke_curric` **deleted** (prereg §4.4); §4.3 collision targets re-verified clean
  after deletion. (Throwaway `smoke_curric_sft.yaml` + `lora_sft_smoke_curric.sbatch` live only in the
  session scratchpad; smoke slurm log retained at `logging/slurm/lora_sft_smoke_curric_13236.out` as evidence.)
- **Step-2 head smoke: SKIPPED** — prereg §4.4 step 2 explicitly permits skipping ("if in doubt, skip — the
  same-code guarantee and cache dims are CPU-verified"); the reviewer independently confirmed `run_one`
  byte-identical across the three enc3seed sbatch and CPU-verified cache dims. Not run to avoid needless queue
  contention.

## 2. Collision re-check (prereg §4.3) — CLEAN (at submit; re-verified after smoke cleanup)

- `logging/lora/{MHC_zh_curric,HateMM_curric}` — absent (fresh SFT, no clobber of generic adapters).
- `data/CLIP_Embedding/{MHC_zh,HateMM}/*LoRA-curric*.pt` — absent (fresh extraction; frozen + generic caches untouched).
- `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_lora_curric*` — absent (fresh group; `force=False`).
- `slurm/logs/enc3s_*LoRA-curric*_seed*_*.trainlog` — absent.
- `squeue -u` — empty before submit.

## 3. Real chain — single-submitted (prereg §6; NO `--time`; afterok-wired)

| job | id | script + args | dependency | GPU | ~cost |
|---|---|---|---|---|---|
| J1 SFT ZH | **13237** | `lora_sft_curric.sbatch MHC_zh` → `logging/lora/MHC_zh_curric` | (none) | 1×A100 | ~2.8–3.3 h |
| J2 SFT HateMM | **13238** | `lora_sft_curric.sbatch HateMM` → `logging/lora/HateMM_curric` | (none) | 1×A100 | ~3.1–3.5 h |
| J3 extract ZH | **13239** | `gen_embed_lora.sbatch MHC_zh logging/lora/MHC_zh_curric Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` | `afterok:13237` | 1×A100 | ~0.35 h |
| J4 extract HateMM | **13240** | `gen_embed_lora.sbatch HateMM logging/lora/HateMM_curric Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` | `afterok:13238` | 1×A100 | ~0.4 h |
| J5 head (6 rows) | **13241** | `enc3seed_lora_curric.sbatch` (MHC_zh×3 + HateMM×3) | `afterok:13239:13240` | 1×A100 | ~2 min |

Dependency graph (scontrol-verified): `13237 → 13239`, `13238 → 13240`, `{13239,13240} → 13241`. J5 gates on
BOTH extractions (`Dependency=afterok:13239(unfulfilled),afterok:13240(unfulfilled)`). Peak concurrent GPU = 2
(J1+J2), within the 2-GPU user cap. All five submitted with `sbatch --parsable` (recipe sbatch carry NO `--time`).

## 4. Queue state at submit

All five **PENDING (JobHeldUser)** immediately post-submit — normal per CLAUDE.md; awaiting auto-release, holds
NEVER forced.

## 5. Review note carried forward (non-blocking)

Review NON-BLOCKING NOTE (2) echoed for the verdict reviewer: HateMM KC20 `n_train_cache = 744` vs
`n_train_sft = 743`, `n_anchor_missing_from_cache = 0` — all 743 SFT anchors present in the frozen cache; one
cache-only train video is a potential LOO neighbor only. Train-only, no leakage, predates cand-2. Benign.
(Notes 1 and 3 likewise non-blocking.)

## 6. Healthy-start verification (STAGE 5) — HEALTHY

**J1 = 13237 (ZH SFT) auto-released from hold and reached RUNNING** (node foscsmlprd01); J2/J3/J4/J5 remain
`PENDING (JobHeldUser)` behind their holds/dependencies (normal; never forced).

Sane first-log-line confirmation for 13237:
- **Correct curric config echoed:** `DATASET=MHC_zh CONFIG=…/mhc_zh_qwen25vl_lora_curric_sft.yaml
  OUTDIR=…/logging/lora/MHC_zh_curric` — the curriculum arm, fresh output dir (no clobber).
- **Recipe r16/alpha32 (hash-frozen config B, sha `ac1c5962…`, verified at freeze):** `lora_rank: 16`,
  `lora_alpha: 32`, `dataset: mhc_zh_lora_curric_train`, `eval_dataset: mhc_zh_lora_val`,
  `output_dir: …/MHC_zh_curric`.
- **Clean preamble:** at the healthy-start checkpoint the job was executing the normal `disk_guard.sh`
  B2-housekeeping loop (pruning old, already-backed-up HateMM Retrieval checkpoints) before STEP 1a/1b builds +
  model load; no NaN/traceback/OOM/ABORT. No loss line yet (first loss logs at step 5, once the training loop
  starts after the preamble).
- **Loss-finite pre-evidence:** the SFT smoke (§1) drove the **identical recipe on the identical 579-row ZH
  curriculum data** and produced finite, decreasing loss (0.2868→0.1930, `train_loss 0.2319`) with a checkpoint
  written — so the real ZH arm's finite-loss behaviour is directly pre-demonstrated. STEP 1b in-job re-build will
  re-emit `train_curric.json sha256 c8260dd3…` (RNG-free builder; smoke + freeze both reproduced F bit-exact).

STAGE-5 bar ("first SFT RUNNING with sane first log lines") **met**. The first real-run loss line + STEP 1b sha
re-verification land during the normal preamble→training transition and are monitored by the orchestrator from
here (executor does not read any test metric).

---

**Confirmations:** nothing pushed; no held-out test metric read; no gate/interpretation/verdict applied by the
executor (verdict is rendered independently against the prereg verbatim). Smoke throwaway output deleted. The
orchestrator monitors the chain from here.
