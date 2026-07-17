# LoRA-HateMM — SINGLE-SUBMIT RECORD (real 3-job chain queued)

**Executor:** submit executor. ZERO user interaction. No verdict produced; no held-out test metric read. Not pushed.
**Submit timestamp (`date -u`):** `Fri Jul 17 03:19:07 UTC 2026`
**Prereg:** `LORA_HATEMM_PREREG.md` commit `3ebd880` · **Review:** `2e41332` (APPROVED-WITH-NOTES) · **Freeze:** `8de0991` (`refine-logs/LORA_HATEMM_FREEZE.md`; confirmed still an ancestor of HEAD at submit time).

---

## 1. Freeze re-verification at submit time (prereg §6.4)

The repo HEAD advanced past the freeze commit (concurrent multi-agent commits), so per §6.4 the frozen
artifacts were re-hashed immediately before treating the submission as authorized. **All match byte-for-byte
→ authorization intact:**

```
da1759a4b18fd7689c9086d5343a0e6805dedd24b76c3f2933bdc83a1941cd4b  refine-logs/LORA_HATEMM_PREREG.md
d2f415cd93fa6f7b439fd4b4573a536baf48ad42186dc8bd50f3fab20553e36a  hatemm_qwen25vl_lora_sft.yaml            (A)
e767eba0ca6ff40679857e5efb759d72aa985629a9ece6584ea424ac2baba62f  scripts/slurm/lora_sft.sbatch           (B)
19c76b177f7dc883a9e03524450ad2e6cb302cdd0a6704d69da68a62188a06fc  scripts/slurm/enc3seed_lora_hatemm.sbatch (C)
c76bb42240feaa300c8b89cdb1fdba1c2d0dbb7360b0ffe53d32fc260a46f386  scripts/slurm/gen_embed_lora.sbatch
93c6d3d1bffbca22b2dd8beba57a33575a48d8ca61d8d56e3148fecdbb93973a  data/lora_sft/HateMM/train.json
9e103ed35a014af81eb3aa6af0d51a28707efd66a606c5bf0459db570a9cc9ef  data/lora_sft/HateMM/val.json
c12ad356aa2917ed80ef17ba93e7854cd36751f770f05a3b19956cfbfdce8462  data/lora_sft/HateMM/test.json
ebf14b472744b0ca2007695033026b9dde4538aa37ccf019b9482a1ab07681b5  RA-HMD/.../data/dataset_info.json
```
Matches the freeze block in `LORA_HATEMM_FREEZE.md` (commit `8de0991`) and the reviewer's freeze block (`2e41332`).

---

## 2. Smoke verification (STAGE 3)

### SFT smoke — job 13228 — **PASS** (the binding §7-step-1 / §4.1(a) gate)

Throwaway smoke: a copy of the frozen recipe with `max_steps:20`, `save_steps:20`, `output_dir=logging/lora/_smoke_hatemm`
(never wrote into `logging/lora/HateMM`); STEP-1 data build skipped (data already built + hash-frozen); env block a
verbatim mirror of `lora_sft.sbatch`. `sacct`: **COMPLETED, ExitCode 0:0, Elapsed 24:17.**

- **Dataset wiring:** `Num examples = 743` (HateMM own train split), eff train batch size 8 (per-device 1 × grad-accum 8),
  trainable params 40,370,176 (backbone-only LoRA).
- **Recipe (adapter_config.json):** `r=16`, `lora_alpha=32`, `lora_dropout=0.0`, `task_type=CAUSAL_LM`, base
  `Qwen/Qwen2.5-VL-7B-Instruct`; target_modules = q/k/v/o_proj + all-layer gate/up/down_proj, **no vision_tower / visual /
  merger modules** (confirms `freeze_vision_tower`/`freeze_multi_modal_projector` — the F0.4 mechanism basis). Matches the
  MHC adapter precedent.
- **Loss finite + decreasing (§4.1a):** step5 `0.3359` → step10 `0.1782` → step15 `0.1542` → step20 `0.1355`; grad_norm
  1.19→0.48→0.66→0.65 (finite, no explosion); mean `train_loss 0.2010`. Trends toward the MHC anchor band (train_loss 0.0964
  / eval_loss 0.1620 at 204 steps). **No NaN / error / traceback.** `[smoke_lora_sft] SMOKE DONE`.
- **Checkpoint written:** `checkpoint-20/` with adapter_model.safetensors + optimizer.pt + scheduler.pt + trainer_state.json.
- Throwaway dir `logging/lora/_smoke_hatemm` deleted after verification (per §4.4 "delete the smoke dir"). StdOut retained in
  session scratchpad `smoke_lora_sft_13228.out`.

### Head smoke — job 13229 — **CANCELLED as redundant** (ruling below)

Was still `PENDING (JobHeldUser)` (approval-daemon backlog) when the SFT gate passed. Per the ruling it is not a submit
gate; cancelled with `scancel` (my own throwaway — cancellation ≠ force-release) before it ran, so it left no
`logging/Retrieval/MHC/_smoke*` artifact.

---

## 3. RULING — Stage 4 does NOT require the head smoke; SFT smoke sufficient

Rests on the **frozen prereg text**:
- **§7 (execution plan), step 1** names the SFT smoke as the *only* pre-submit gate: "Gate: SFT smoke (§4.4.1) BEFORE this
  real submit; on COMPLETE, apply the G-repro SFT-loss sanity (§4.1a)." The head smoke is not listed as a submit gate.
- **§4.1(b):** head-run validity rests on the byte-identical same-code guarantee (CPU-verified `diff`, §4.2) — "this retires
  the code-version confound" — NOT on a runtime head smoke. Independently corroborated: the 12850 head code was reproduced
  bit-exact 12/12 by the router gate (ROUTER_GATE_RECORD.md §1), and the EN-LoRA align-fusion path was already exercised by
  the B4 seed0 anchor.
- **§4.4:** "If in doubt, skip the smokes — cache dims and the same-code guarantee are already CPU-verified."
- **Sequencing:** the real head job (Stage 3) is `afterok`-chained behind extraction (~3.5 h out); it cannot run before
  Stages 1–2 finish regardless, so blocking the whole chain submit on a head smoke gates nothing immediate.

---

## 4. Submit-time collision re-check (STAGE 2 / prereg §4.3) — CLEAN

- `logging/lora/HateMM` — ABSENT (only `MHC` + `MHC_zh` adapters present, untouched).
- `data/CLIP_Embedding/HateMM/*LoRA*.pt` — ABSENT (frozen HateMM caches untouched).
- `logging/Retrieval/HateMM/RAC_video_lora_hm*` and `logging/Retrieval/MHC/RAC_video_lora_hm*` — ABSENT.
- `logging/Retrieval/MHC/_smoke*` — ABSENT (head smoke cancelled pre-run).
- `slurm/logs/enc3s_{HateMM,MHC}_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed*_*.trainlog` — ABSENT.
- `squeue -u jehc223` — no conflicting jobs before submit.

---

## 5. STAGE 4 — real chain submitted (dependency order; NO `--time`)

| step | job id | script + args | dependency | StdOut |
|---|---|---|---|---|
| 1 SFT | **13233** | `sbatch scripts/slurm/lora_sft.sbatch HateMM` | (none) | `logging/slurm/lora_sft_13233.out` |
| 2 EMB | **13234** | `sbatch scripts/slurm/gen_embed_lora.sbatch HateMM logging/lora/HateMM` | `afterok:13233` | `slurm/logs/lora_embed_13234.out` |
| 3 HEAD | **13235** | `sbatch scripts/slurm/enc3seed_lora_hatemm.sbatch` | `afterok:13234` | `slurm/logs/enc3seed_13235.out` |

Dependency graph: **13233 → 13234 → 13235** (SFT → extraction → 3-seed head + bundled B4-EN arm). Exact invocations per
prereg §1.2/§1.3/§1.4 and §7.

---

## 6. STAGE 5 — queue state at hand-off

`squeue` immediately after submit:
```
     JOBID                 NAME      STATE           DEPENDENCY     NODELIST(REASON)
     13235             enc3seed    PENDING afterok:13234(unfulf        (JobHeldUser)
     13234           lora_embed    PENDING afterok:13233(unfulf        (JobHeldUser)
     13233             lora_sft    PENDING               (null)        (JobHeldUser)
```

All three `PENDING (JobHeldUser)` — expected. The orchestrator flagged a cluster-wide approval-daemon backlog that may hold
jobs for hours; **not** force-released (CLAUDE.md: WAIT for auto-release, never `scontrol release`/`requeue`). Correctly-chained
queued state is the healthy-submission evidence at hand-off; the orchestrator monitors SFT start (~3–3.5 h once released) from here.

---

## 7. Required statements

- Freeze re-verified at submit time (all frozen artifacts byte-unchanged); freeze commit `8de0991` still in history.
- SFT smoke gate PASSED; head smoke ruled non-gating (frozen prereg §7/§4.1b/§4.4) and cancelled as redundant.
- No held-out test metric read; NO verdict rendered (verdict = independent 0-context reviewer, post-completion).
- Nothing pushed to any remote (this record + freeze are local commits only).
