# BIDIR-STAGE1 — SUBMIT RECORD (submit executor)

**Role:** submit executor for the frozen BIDIR-STAGE1 cell. ZERO user interaction. NO push. NO test metric
read for decisions. NO verdict / NO gates / NO deltas / NO pass-fail language on the head numbers. RAW-ONLY at
the head stage: the executor transcribes raw both-protocol per-seed numbers (line-numbered); the verdict is
rendered by an independent 0-context reviewer against the prereg VERBATIM.
**Date:** 2026-07-25 NZST.
**Prereg:** `refine-logs/BIDIR_STAGE1_PREREG.md`, FROZEN sha256
`3c532e5370e52b2ed53e0bcc2ad63d2958823f5aca0e6f710495d8cf55565142` (commit `a7bb2a1`).
**Freeze:** `refine-logs/BIDIR_STAGE1_FREEZE.md` (reviewer verdict APPROVED-WITH-NOTES, 4 non-blocking notes).
**Review:** `refine-logs/BIDIR_STAGE1_PREREG_REVIEW.md`.
**House precedent:** `refine-logs/READOUT_SUBMIT_RECORD.md`, `refine-logs/FRAME16_SUBMIT_RECORD.md`.

Authorization derives from the freeze and is VOID on any sha mismatch. The readout chain (parallel cell) has
CLEARED the queue (screen-killed, no head job) ⇒ clear to submit per prereg §6 ordering.

---

## 1. Sha re-verification at submit time — ALL MATCH (authorization intact)

Re-ran `sha256sum` on the prereg (self-sha), the four new artifacts A/A2/B/C, and the reused-unchanged
machinery. **Every hash matches the frozen block in `BIDIR_STAGE1_FREEZE.md`; authorization is intact.**

### Prereg self-sha + frozen artifacts A/A2/B/C
```
FROZEN 3c532e5370e52b2ed53e0bcc2ad63d2958823f5aca0e6f710495d8cf55565142  refine-logs/BIDIR_STAGE1_PREREG.md                       [MATCH]
A      36cedbac365b2b13c945adbe3437efdc61d8be15ecc85878eb9614225abe367b  src/utils/bidir_patch.py                                 [MATCH]
A2     03f39e09c417bbea291f3c06b787f5220693568cd613705693df1c2bf23e020d  src/utils/generate_VideoMLLM_embedding_bidir_HF.py       [MATCH]
B      0f17fce6910981bbc4c5942eae3b18947151bc6990ceee401fc86b252a287ecd  scripts/slurm/gen_embed_mllm_bidir.sbatch                [MATCH]
C      82a69e74d570df59a1b686891814c7756b15755901d2a645bb1d3f0164a51264  scripts/slurm/enc3seed_bidir.sbatch                      [MATCH]
```

### Reused-unchanged machinery (NOT edited)
```
causal extractor  b6b61a3fa4214f28d2098ca305ca9a981934445afc81ccc5ac3939f8c1fb0ec6  src/utils/generate_VideoMLLM_embedding_lora_HF.py   [MATCH]
head anchor       dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d  scripts/slurm/enc3seed.sbatch                        [MATCH]
ZH adapter cfg    f9384d8dbdb8c1e315bb40a96952f068830c9a98cd6107f3b2ac2458e7fc477b  logging/lora/MHC_zh/adapter_config.json              [MATCH]
ZH adapter wts    35a510f4ad84542c798939cfdb340b00317a5b8a2c670b07ced8d1869dd7b438  logging/lora/MHC_zh/adapter_model.safetensors        [MATCH]
HM curric cfg     eaca36dd5cef2a4ff866a0398680d420adb157be815fe335500a387bbf9037b8  logging/lora/HateMM_curric/adapter_config.json       [MATCH]
HM curric wts     6571d132ef3218e4bdfcee98aab468df21f8aa83b16d623dd2098f8486394efa  logging/lora/HateMM_curric/adapter_model.safetensors [MATCH]
```

**Same-code guarantee (§4.2).** The exact `run_one()`…`PY` head block (41 lines) of `enc3seed_bidir.sbatch`,
`enc3seed.sbatch`, and `enc3seed_lora_curric.sbatch` all hash to the same block sha `13e34e4e…` — BYTE-IDENTICAL
to the freeze block sha `13e34e4e93c6a76988557e1c609fd54e0353c627fd36eb1c5b9e26ed187c3feb`. Only the manipulated
head variables vs the banked causal controls are `--model` (`-bidir` cache tag) and `--group_name`
(`RAC_video_bidir`) plus derived `--exp_comment`. `bash -n` on both new sbatch = SYNTAX_OK.

Header verification (prereg §6 resource plan): `gen_embed_mllm_bidir.sbatch` and `enc3seed_bidir.sbatch` each
request `--cpus-per-task=8`, `--mem=64G`, `--gres=gpu:a100:1`, and carry **NO `--time`** (L2-8 of each). The head
is `afterok`-chained to the extraction ⇒ peak footprint 8 CPU / 64 G / 1 GPU — within the 16/128/2 cap, never
two 16-CPU jobs.

## 2. Collision-safety re-check at submit — CLEAN (all ABSENT); banked causal caches PRESENT + mtimes recorded

- `data/CLIP_Embedding/MHC_zh/*LoRA-bidir*.pt` — ABSENT ⇒ fresh extraction.
- `data/CLIP_Embedding/HateMM/*LoRA-curric-bidir*.pt` — ABSENT ⇒ fresh extraction.
- `logging/Retrieval/{MHC_zh,HateMM}/RAC_video_bidir*` — ABSENT ⇒ fresh verdict group.
- `slurm/logs/enc3s_*bidir*seed*.trainlog` — ABSENT ⇒ no trainlog collision.
- `logging/_smoke_bidir` — ABSENT before smoke.

**Banked causal paired-floor caches PRESENT + sha16/mtimes recorded BEFORE the run (must be UNTOUCHED after —
distinct `-bidir` out-tags cannot clobber the banked causal tags):**

| dataset | split | sha16 | bytes | mtime (before) |
|---|---|---|---|---|
| ZH (13150 `-LoRA_HF`) | train | b2e8e78d19c71d2c | 16619871 | 2026-07-02 12:08:59.501321227 +1200 |
| ZH | dev_seen | 4c07af75098391c9 | 2240628 | 2026-07-02 12:11:47.839858186 +1200 |
| ZH | test_seen | 4e107bf65f58745a | 4278267 | 2026-07-02 12:17:25.706949549 +1200 |
| HateMM (13241 `-LoRA-curric_HF`) | train | 5e80f39327a74314 | 21358864 | 2026-07-18 12:26:57.405769081 +1200 |
| HateMM | dev_seen | 46ee4fd9fcaec80b | 3073381 | 2026-07-18 12:29:24.237503621 +1200 |
| HateMM | test_seen | b50ae4ecb077a833 | 6173356 | 2026-07-18 12:34:15.123051972 +1200 |

Banked sha16/mtimes to be re-checked after the run (§6).

## 3. CPU non-causality self-test (prereg §4.4.1, LOAD-BEARING) — PASS (reproduced)

Env: `transformers 4.49.0`, `HateVideo` (matches freeze pin). `CUDA_VISIBLE_DEVICES="" python
src/utils/bidir_patch.py` (login-node CPU, seconds; tiny random Qwen2.5-VL decoder — no test data, no GPU):

```
[BIDIR self-test] patched mask shape=(1, 1, 6, 6) all-zero=True
[BIDIR self-test] d_causal(pos0, future perturbed) = 0.000e+00  (expect < 1e-05)
[BIDIR self-test] d_causal(last pos, sanity)        = 1.042e+01  (expect > 1e-04)
[BIDIR self-test] d_bidir (pos0, future perturbed)  = 6.387e-02  (expect > 1e-04)
[BIDIR self-test] VERDICT: PASS
```

Reproduces the prereg §4.4.1 and review Check-1 numbers exactly: under causal, position 0 is invariant to a
future-token perturbation (`0.000e+00`); under the bidir patch it changes (`6.387e-02`); the sanity leg confirms
the perturbation is real (`1.042e+01`). Non-causality propagates backward iff the patch is active.

## 4. GPU smoke (prereg §4.4.2 + review note 4) — job 13469 (throwaway; PENDING at write time)

Throwaway smoke sbatch (session scratchpad `smoke_bidir.sbatch`; env block mirrors artifact B — conda HateVideo,
HF/TRANSFORMERS offline; disk_guard OMITTED as a throwaway per the readout house pattern; writes ONLY to
`logging/_smoke_bidir` via redirected `--EXP_FOLDER`, never `data/CLIP_Embedding/`, no B2 push). Three parts in
one GPU job:
- **Part A** — the frozen runner verbatim (prereg §4.4.2 command): `generate_VideoMLLM_embedding_bidir_HF.py
  --dataset HateMM --lora_dir logging/lora/HateMM_curric --num_frames 8 --splits test --limit 2 --out_model_tag
  _smoke_bidir --EXP_FOLDER logging/_smoke_bidir --device cuda`. Exercises the real merged 7B + `apply_bidir_mask`
  (its sdpa assert + `is_causal=False` loop → the `[BIDIR] … installed on N decoder attention module(s)` line).
- **Part B** — cache sanity + the **bidir ≠ causal DIFFER check** (CPU): shapes `(2,3584)`, finite, L2 norms,
  id-matched rows vs the banked HateMM-curric causal cache; **ABORT if any smoke row is bit-identical to the
  banked causal row** (would mean the patch silently failed).
- **Part C** — the real-model non-causality belt (review note 4): reload base + merge HateMM_curric + apply
  patch, `assert type(model.model).__name__ == "Qwen2_5_VLModel"`, and confirm the bound `_update_causal_mask`
  returns a NON-None all-zeros `(1,1,S,S)` mask.
- **Cleanup** — `rm -rf logging/_smoke_bidir`; then re-verify collision targets ABSENT and banked caches
  UNTOUCHED.

**Smoke job 13469: COMPLETED exit 0:0, Elapsed 00:01:03** (A100-SXM4-80GB, `foscsmlprd01`; auto-released from
`JobHeldUser`, never forced; running aggregate was zero). Full log `slurm/logs/smoke_bidir_13469.out`.

- **Namespace (echoed):** `dataset='HateMM', lora_dir='logging/lora/HateMM_curric', out_model_tag='_smoke_bidir',
  EXP_FOLDER='logging/_smoke_bidir', num_frames=8, max_pixels=151200, splits='test', limit=2, device='cuda'` —
  frozen runner unedited; distinct `_smoke_bidir` tag + throwaway EXP_FOLDER (never wrote `data/CLIP_Embedding/`).
- **Part A (apply_bidir_mask on the real merged 7B, review note 4 coverage):**
  `[BIDIR] mask-flip patch installed on model.model; is_causal=False on 28 decoder attention module(s);
  attention is now bidirectional.` — the sdpa assert PASSED (no AssertionError) and the `is_causal=False` loop
  ran over all **28** decoder attention modules. Extraction: `Saved 'test_seen': N=2, Dv=3584, Dt=3584,
  zero-vector videos=0`; no OOM, no traceback, the L306 masked-scatter invariant held at 8 frames.
- **Part B (shape/finite/norm + bidir ≠ causal DIFFER):** smoke ids `['hate_video_1','non_hate_video_4']`,
  img/text shape `(2,3584)`, L2 norms `[1.0,1.0]`/`[1.0,1.0]`, NaN/Inf=0. **DIFFER PASS** — id-matched vs the
  banked HateMM-curric causal cache: `hate_video_1` (banked row 0) img max|Δ|=2.452e-01, text max|Δ|=3.166e-01;
  `non_hate_video_4` (row 1) img max|Δ|=1.315e-01, text max|Δ|=2.592e-01. **No row bit-identical to causal ⇒ the
  mask flip genuinely changed the representation end-to-end (patch did NOT silently fail).**
- **Part C (real-model non-causality belt):** `TYPE_CHECK: model.model is Qwen2_5_VLModel`;
  `MASK_CHECK: non-None all-zeros mask shape=(1, 1, 10, 10) max|mask|=0.0`; `BELT: PASS` — the merge preserves
  `model.model` as the decoder and the bound `_update_causal_mask` returns the NON-None all-zeros mask.
- **Cleanup:** `logging/_smoke_bidir` **deleted**; collision targets re-verified ABSENT after deletion; banked
  causal caches re-checked UNTOUCHED (sha16 + mtimes bit-identical to the §2 pre-run table for all 6).

**SMOKE_VERDICT: PASS.** Cleared to submit the real chain.

## 5. Real chain — single-submitted (prereg §6; NO `--time`; SEQUENTIAL, afterok-wired)

Final `sha256sum` re-verified at the submit instant — prereg `3c532e53…`, A `36cedbac…`, A2 `03f39e09…`,
B `0f17fce6…`, C `82a69e74…` [ALL MATCH]; `bash -n` B and C = SYNTAX_OK; authorization intact.

| job | id | script + args | dependency | CPU/mem/GPU | ~cost |
|---|---|---|---|---|---|
| J1 extract bidir | **13470** | `gen_embed_mllm_bidir.sbatch` (ZH `logging/lora/MHC_zh`→`-LoRA-bidir_HF`, then HateMM `logging/lora/HateMM_curric`→`-LoRA-curric-bidir_HF`; 8 frames) → `data/CLIP_Embedding/{MHC_zh,HateMM}/{train,dev_seen,test_seen}_…-bidir…_HF.pt` | (none) | 8 CPU / 64 G / 1×A100 | ~0.5–0.7 GPU-h |
| J2 head 3-seed | **13471** | `enc3seed_bidir.sbatch` (ZH-bidir + HateMM-curric-bidir seeds 0/1/2, group `RAC_video_bidir`) | `afterok:13470` | 8 CPU / 64 G / 1×A100 | ~2 min |

**Dependency graph (scontrol-verified):** `13470 → 13471` (J2 `Dependency=afterok:13470(unfulfilled)`). The head
cannot start until extraction succeeds ⇒ the two jobs **never run concurrently**; peak footprint 8 CPU / 64 G /
1 GPU (within 16/128/2 cap; never two 16-CPU jobs). Both submitted `sbatch --parsable`, carry NO `--time`. The
readout chain had already cleared the queue (prereg §6 sequencing satisfied).

## 5.1 Queue state at submit — both PENDING (JobHeldUser); WAIT never force

- J1 **13470 PENDING (JobHeldUser)** (Dependency null).
- J2 **13471 PENDING (JobHeldUser)** + `afterok:13470(unfulfilled)`.

Running aggregate ZERO at submit ⇒ favorable for auto-release (the smoke 13469 auto-released from the same hold).
Per CLAUDE.md the `JobHeldUser` hold is **waited out, NEVER forced**. If J1 stays held > 2 h, a status line is
committed and the turn ends PENDING-JOB (orchestrator resumes).

**Expected split sizes for §6 cache sanity:** ZH 579/78/149, HateMM 744/107/215 (gt line counts + banked causal
cache dims, both re-confirmed this submit). **On J1 (13470) COMPLETE:** cache sanity (§6 — counts/dims/NaN,
bidir caches DIFFER from causal, banked untouched by mtime). **On J2 (13471) COMPLETE:** transcribe RAW per-seed
both-protocol numbers (line-numbered) into §7 — NO gates, NO deltas, NO pass/fail (independent 0-context verdict
reviewer rules).
