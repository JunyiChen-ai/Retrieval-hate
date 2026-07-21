# FRAME16 stage-1 — SUBMIT RECORD (submit executor)

**Role:** submit executor for the frozen FRAME16 stage-1 cell. ZERO user interaction. NO push. NO test metric
read for decisions. NO verdict / NO gates / NO deltas / NO pass-fail language. RAW-ONLY: the executor transcribes
raw both-protocol per-seed numbers (line-numbered); the verdict is rendered by an independent 0-context reviewer
against the prereg VERBATIM.
**Date:** 2026-07-21 NZST.
**Prereg:** `refine-logs/FRAME16_PREREG.md`, FROZEN sha256
`5c240518217e1ab69cbd52de34c8849450d8b37ad163d68b4247c5f2c791c725`.
**Freeze:** `refine-logs/FRAME16_FREEZE.md` (reviewer `7164cbb`, verdict APPROVED-WITH-NOTES, 3 non-blocking notes).
**Review:** `refine-logs/FRAME16_PREREG_REVIEW.md` (commit `7164cbb`).
**House precedent:** `refine-logs/VISION_UNFREEZE_SUBMIT_RECORD.md`.

Authorization derives from the freeze and is VOID on any sha mismatch.

---

## 1. Sha re-verification at submit time — ALL MATCH (authorization intact)

Re-ran `sha256sum` on the prereg, artifacts B/C, and the reused-unchanged machinery (extractor + both fork/anchor
sources) at submit time. **Every hash matches the frozen block; authorization is intact.**

### Prereg (self-sha)
```
5c240518217e1ab69cbd52de34c8849450d8b37ad163d68b4247c5f2c791c725  refine-logs/FRAME16_PREREG.md   [MATCH]
```

### Frozen artifacts B, C
```
B a600e74c0a6483095329f9ce15a3df19c842554362f7a3ef1f6e76e26fe3c750  scripts/slurm/gen_embed_mllm_16f.sbatch  [MATCH]
C 99e7e8b10286e22d7913e85c14141c8fa02c90ae27adc0da6facaceeb703864a  scripts/slurm/enc3seed_fb16.sbatch       [MATCH]
```

### Reused-unchanged machinery (NOT edited)
```
extractor    d89a912602d763aa055a54f50b0188e302e554b70ff6c0eb872f250bd454b67c  src/utils/generate_VideoMLLM_embedding_HF.py  [MATCH]
fork source  9357fa1087e775d059779e6c5f86e19e71b78b2d166f904fa3c71a1a1cbb3268  scripts/slurm/gen_embed_mllm.sbatch           [MATCH]
head anchor  dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d  scripts/slurm/enc3seed.sbatch                 [MATCH]
```

Header verification (prereg §6 resource plan): both new sbatch request `--cpus-per-task=8`, `--mem=64G`,
`--gres=gpu:a100:1`, and carry **NO `--time`** (verified L2-8 of each). These are **8-CPU** jobs; the head is
`afterok`-chained to the extraction ⇒ peak footprint 8 CPU / 64 G / 1 GPU (never two 16-CPU jobs in flight —
the standing infra rule after the 13303 wedge, VISION §7).

## 2. Collision-safety re-check at submit — CLEAN (all ABSENT)

- `data/CLIP_Embedding/HateMM/*Qwen2.5-VL-7B-Instruct_HF-16f*.pt` — ABSENT (fresh extraction).
- `logging/Retrieval/HateMM/RAC_video_fb16*` — ABSENT (fresh group; `--force False` never trips).
- `slurm/logs/enc3s_*HF-16f_seed*.trainlog` — ABSENT (no trainlog collision).
- `logging/_smoke_fb16` — ABSENT before smoke (created + deleted by smoke; re-verified ABSENT after).

**Banked 8f caches PRESENT + untouched (distinct out-tag `…_HF-16f` cannot clobber `…_HF`).** mtimes recorded
BEFORE the run (to re-confirm untouched AFTER):

| file | bytes | mtime (before) |
|---|---|---|
| `data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct_HF.pt` | 21358780 | 2026-07-02 00:11:19.293608963 +1200 |
| `data/CLIP_Embedding/HateMM/dev_seen_Qwen2.5-VL-7B-Instruct_HF.pt` | 3073233 | 2026-07-02 00:13:44.933286504 +1200 |
| `data/CLIP_Embedding/HateMM/test_seen_Qwen2.5-VL-7B-Instruct_HF.pt` | 6173272 | 2026-07-02 00:18:33.187669051 +1200 |

(bytes match the freeze block 21358780 / 3073233 / 6173272.) Queue empty at submit (`squeue -u jehc223` = 0 rows;
running aggregate ZERO ⇒ favorable for JobHeldUser auto-release).

## 3. 16f extraction smoke (prereg §4.4.1) — PASS (all load-bearing checks), cleaned up

Throwaway smoke sbatch (session scratchpad `smoke_fb16_extract.sbatch`, mirroring `gen_embed_mllm_16f.sbatch`'s
env block: conda HateVideo, HF offline, disk_guard) running the prereg §4.4.1 command verbatim:
`generate_VideoMLLM_embedding_HF.py --dataset HateMM --num_frames 16 --splits test --limit 3
--out_model_tag _smoke16f --EXP_FOLDER logging/_smoke_fb16 --device cuda`.

- **Job 13352? no — smoke job 13349** (`smoke_fb16`): submitted `sbatch` (NO `--time`); auto-released from
  `JobHeldUser` (never forced); **COMPLETED** (exit 0:0, Elapsed 00:40:20 — of which ~30 min was the disk_guard
  B2 backup/prune phase at sbatch start, `|| true`, benign; the extraction proper is ~1-2 min).
- **Namespace (echoed):** `dataset='HateMM', num_frames=16, out_model_tag='_smoke16f',
  EXP_FOLDER='logging/_smoke_fb16', splits='test', limit=3, device='cuda'` — exactly the pinned single changed
  variable (`--num_frames 16`) + distinct out-tag; no extractor code edit.
- **Extractor line:** `Saved 'test_seen': N=3, Dv=3584, Dt=3584, zero-vector videos=0 ->
  logging/_smoke_fb16/HateMM/test_seen__smoke16f.pt` (redirected EXP_FOLDER ⇒ never wrote into
  `data/CLIP_Embedding/HateMM/`).
- **(1) Shapes:** `img_feats (3, 3584)` + `text_feats (3, 3584)`, both `float32` — dual-stream 3584-d confirmed.
- **(2) Finite:** img NaN=0 inf=0; text NaN=0 inf=0; zero-vector videos=0.
- **(3) Sane norms:** per-row L2 norm img `[1.0, 1.0, 1.0]`, text `[1.0, 1.0, 1.0]` (the `_encode` L2-normalize).
- **(4) No OOM / no assertion:** clean scan — no `OOM`/`CUDA error`/`Traceback`/`AssertionError`; the L283
  masked-scatter invariant held at 16 frames.
- **`ids` structure sanity:** the saved `ids` is a length-1 list wrapping the inner id list (`len(ids)==1`,
  inner list = the N ids) — **identical structure to the banked 8f cache** (`test_seen_…_HF.pt`: same
  `list len=1`, inner list of 215; smoke inner list of 3); the id ordering matches deterministically
  (`hate_video_1, non_hate_video_4, non_hate_video_8, …`). This is the normal cache format the head loader
  unwraps — not a smoke artifact.
- **Cleanup:** `logging/_smoke_fb16` **deleted** (prereg §4.4); the four §2 collision targets re-verified ABSENT
  after deletion; banked 8f caches re-checked UNTOUCHED (mtimes bit-identical to the §2 pre-run table). Throwaway
  `smoke_fb16_extract.sbatch` lives only in the session scratchpad; smoke slurm log retained as evidence at
  `slurm/logs/smoke_fb16_13349.out`.

## 4. Real chain — single-submitted (prereg §6; NO `--time`; afterok-wired; SEQUENTIAL)

Final `sha256sum` re-verified at the submit instant — B `a600e74c…`, C `99e7e8b1…` [MATCH]; authorization intact.

| job | id | script + args | dependency | CPU/mem/GPU | ~cost |
|---|---|---|---|---|---|
| J1 extract 16f | **13352** | `gen_embed_mllm_16f.sbatch HateMM` → `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF-16f.pt` | (none) | 8 CPU / 64 G / 1×A100 | ~0.4–0.6 GPU-h |
| J2 head 3-seed | **13353** | `enc3seed_fb16.sbatch` (HateMM-16f seeds 0/1/2, group `RAC_video_fb16`) | `afterok:13352` | 8 CPU / 64 G / 1×A100 | ~1.5 min |

**Dependency graph (scontrol-verified):** `13352 → 13353` (J2 `Dependency=afterok:13352(unfulfilled)`). The head
cannot start until extraction succeeds ⇒ the two jobs **never run concurrently**; peak footprint = 8 CPU / 64 G /
1 GPU (never two 16-CPU jobs in flight). Both submitted `sbatch --parsable`, carry NO `--time`.

## 5. Queue state at submit — both PENDING (JobHeldUser); WAIT never force

- J1 **13352 PENDING (JobHeldUser)** (Dependency null).
- J2 **13353 PENDING (JobHeldUser)** + `afterok:13352(unfulfilled)`.

Running aggregate ZERO at submit ⇒ favorable for auto-release (the smoke 13349 auto-released from the same hold).
Per CLAUDE.md the `JobHeldUser` hold is **waited out, NEVER forced**. If J1 stays held > 2 h, a status line is
committed and the turn ends PENDING-JOB (orchestrator resumes).

**On J1 (13352) COMPLETE:** cache sanity (train 744 / dev 107 / test 215 counts, dim 3584, 0 NaN, distinct
`…_HF-16f` tag; 8f caches untouched by mtime). **On J2 (13353) COMPLETE:** transcribe RAW per-seed both-protocol
numbers (line-numbered) into §6 — NO gates, NO deltas, NO pass/fail (independent 0-context verdict reviewer rules).
