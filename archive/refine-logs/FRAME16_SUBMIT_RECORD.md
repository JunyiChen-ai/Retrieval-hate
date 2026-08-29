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

**On J1 (13352) COMPLETE:** cache sanity (§6). **On J2 (13353) COMPLETE:** transcribe RAW per-seed both-protocol
numbers (line-numbered) into §7 — NO gates, NO deltas, NO pass/fail (independent 0-context verdict reviewer rules).

## 5.1 Chain outcome — both COMPLETED (exit 0:0)

Both jobs auto-released from `JobHeldUser` (never forced; running aggregate was zero) and ran sequentially:
- **J1 13352 mllm_embed_16f (extract):** COMPLETED, exit 0:0, Elapsed 00:59:57.
- **J2 13353 enc3seed_fb16 (head):** COMPLETED, exit 0:0, Elapsed 00:04:13.

## 6. Extraction cache sanity (prereg §4.1c) — PASS; banked 8f UNTOUCHED

New `…_HF-16f` caches written (distinct tag; `data/CLIP_Embedding/HateMM/`):
`train_…_HF-16f.pt` 21358808 B @ 2026-07-21 20:35:23, `dev_seen_…_HF-16f.pt` 3073261 B @ 20:41:19,
`test_seen_…_HF-16f.pt` 6173300 B @ 20:53:07.

| split | N | img_feats | text_feats | labels | img NaN/Inf | text NaN/Inf | ids == 8f id-order |
|---|---|---|---|---|---|---|---|
| train | 744 | (744, 3584) | (744, 3584) | 744 | 0 / 0 | 0 / 0 | yes |
| dev_seen | 107 | (107, 3584) | (107, 3584) | 107 | 0 / 0 | 0 / 0 | yes |
| test_seen | 215 | (215, 3584) | (215, 3584) | 215 | 0 / 0 | 0 / 0 | yes |

Counts 744/107/215 as expected; dual-stream 3584-d; zero NaN/Inf; the `ids` (length-1-list-wrapped inner list)
match the banked 8f cache id-order **exactly** for all three splits ⇒ the within-seed pairing is on identical
cache membership + order.

**Banked 8f caches re-checked AFTER the run — mtimes bit-identical to the §2 pre-run table (UNTOUCHED):**
`train_…_HF.pt` 21358780 B @ 2026-07-02 00:11:19.293608963; `dev_seen_…_HF.pt` 3073233 B @ 00:13:44.933286504;
`test_seen_…_HF.pt` 6173272 B @ 00:18:33.187669051. The distinct `-16f` out-tag did not clobber the 8f floor.

## 7. RAW head numbers — frozen-Qwen-16f per-seed both-protocol (job 13353) — TRANSCRIPTION ONLY

**RAW-ONLY.** Verbatim from the frozen embedded parser readout in `slurm/logs/enc3seed_fb16_13353.out` (the
`run_one…PY` block is byte-identical to `enc3seed.sbatch`, verified at freeze — same parser that produced the
banked 8f floor); each value cross-checked against the underlying `Test_Retrieval … macroF1 … acc … roc` line in
the per-run `enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF-16f_seed{0,1,2}_13353.trainlog`. val-sel = epoch ≥ warmup 5
with max `Val_Retrieval` acc (roc tie-break); final = max epoch (29). **NO Δ-vs-8f-floor, NO KS-16f-dead, NO
CONTINUE gate, NO FORMAL bar, NO mean, NO sign count, NO pass/fail — every gate + the 3-seed aggregation is left
to the independent 0-context verdict reviewer, who re-parses the raw logs itself against the prereg VERBATIM.**

### 7.1 HateMM — frozen-Qwen-16f (model `Qwen2.5-VL-7B-Instruct_HF-16f`, group `RAC_video_fb16`)
| seed | protocol | epoch | acc | macroF1 | roc | `enc3seed_fb16_13353.out` ln | trainlog macroF1 ln |
|---|---|---|---|---|---|---|---|
| 0 | val-sel | 23 | 0.8698 | 0.8606 | 0.9247 | 330 | seed0:250 |
| 0 | final-ep | 29 | 0.8605 | 0.8514 | 0.9136 | 331 | seed0:305 |
| 1 | val-sel | 28 | 0.8651 | 0.8567 | 0.9228 | 641 | seed1:292 |
| 1 | final-ep | 29 | 0.8744 | 0.8666 | 0.9312 | 642 | seed1:302 |
| 2 | val-sel | 20 | 0.8605 | 0.8514 | 0.9282 | 953 | seed2:221 |
| 2 | final-ep | 29 | 0.8744 | 0.8653 | 0.9307 | 954 | seed2:303 |

RESULT_ROW lines (verbatim, tab-separated `…\tvalsel\t<ep>\t<F1>\t<acc>\t<roc>\tfinal\t<ep>\t<F1>\t<acc>\t<roc>`),
`enc3seed_fb16_13353.out` ln 332 (seed0) / 643 (seed1) / 955 (seed2):
```
RESULT_ROW  enc3s_…_HF-16f_seed0_13353.trainlog  valsel 23 0.8606 0.8698 0.9247  final 29 0.8514 0.8605 0.9136
RESULT_ROW  enc3s_…_HF-16f_seed1_13353.trainlog  valsel 28 0.8567 0.8651 0.9228  final 29 0.8666 0.8744 0.9312
RESULT_ROW  enc3s_…_HF-16f_seed2_13353.trainlog  valsel 20 0.8514 0.8605 0.9282  final 29 0.8653 0.8744 0.9307
```

The banked frozen-Qwen-8f floor for the paired comparison lives in prereg §2.1 (re-derived from the 12850
trainlogs); the executor does **not** compute the pairing — that is the verdict reviewer's step.

## 8. Closeout — CHAIN COMPLETE

Sha re-verify (submit + submit-instant) ALL MATCH; 16f extraction smoke PASS (shapes (3,3584) img+text, 0 NaN,
L2 norms 1.0, no OOM, L283 assert held), smoke artifacts deleted; collisions CLEAN throughout; chain submitted
sequential afterok (13352 → 13353), both COMPLETED exit 0:0; extraction cache sanity PASS (744/107/215, 3584-d,
0 NaN/Inf, ids == 8f id-order); banked 8f caches UNTOUCHED (mtimes bit-identical before/after); 3 head runs
COMPLETED; raw both-protocol per-seed numbers transcribed (§7). The verdict (G-repro → single test-touch →
KS-16f-dead → CONTINUE gate → FORMAL bar, both protocols) is rendered by a fresh independent 0-context reviewer
against the frozen prereg VERBATIM — the executor applied NO gates and NO pass/fail language. No `state/`
mutation. Nothing pushed.
