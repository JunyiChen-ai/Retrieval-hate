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
