# READOUT-GRID (R0–R3) — SUBMIT RECORD (submit executor)

**Role:** submit executor for the frozen READOUT-GRID cell. ZERO user interaction. NO push. NO test
metric read for decisions. NO verdict / NO gates / NO deltas / NO pass-fail language on the head numbers.
RAW-ONLY at the head stage: the executor transcribes raw both-protocol per-seed numbers (line-numbered);
the verdict is rendered by an independent 0-context reviewer against the prereg VERBATIM. The $0 CPU screen
IS run by the executor (it is a pre-declared deterministic decision rule, not a test-touch — it hard-blocks
the test split) and its KS-dead-vs-promote output is reported per cell.
**Date:** 2026-07-25 NZST.
**Prereg:** `refine-logs/READOUT_PREREG.md`, FROZEN sha256
`f6d43096332acb2e9f9743f5e51967f6a93b3006770be3520effc13929aac543` (commit `1b3e0c6`).
**Freeze:** `refine-logs/READOUT_FREEZE.md` (reviewer verdict APPROVED-WITH-NOTES, 3 non-blocking notes).
**Review:** `refine-logs/READOUT_PREREG_REVIEW.md`.
**House precedent:** `refine-logs/FRAME16_SUBMIT_RECORD.md`.

Authorization derives from the freeze and is VOID on any sha mismatch.

---

## 1. Sha re-verification at submit time — ALL MATCH (authorization intact)

Re-ran `sha256sum` on the prereg, artifacts A/B/C, and the reused-unchanged machinery. **Every hash matches
the frozen block in `READOUT_FREEZE.md`; authorization is intact.**

### Prereg (self-sha) + frozen artifacts A/B/C
```
FROZEN f6d43096332acb2e9f9743f5e51967f6a93b3006770be3520effc13929aac543  refine-logs/READOUT_PREREG.md              [MATCH]
A      ef05f3d45a3e8c31f8dc198ba41e18c2e525cd29e9ba0ed539dfd9b4c6d869c3  src/utils/generate_VideoMLLM_embedding_readout_HF.py  [MATCH]
B      948db8514c9e4b02d6d20ceed3e6a63104893c8a6e623def75e4c22bc9419e29  scripts/slurm/gen_embed_readout.sbatch    [MATCH]
C      f56badb64b9dc8a4d18fbbcbbff99994234df3812dccd7334f8827e100d35547  scripts/analysis/readout_screen.py        [MATCH]
```

### Reused-unchanged machinery (NOT edited; git-clean fork verified)
```
fork source  b6b61a3fa4214f28d2098ca305ca9a981934445afc81ccc5ac3939f8c1fb0ec6  src/utils/generate_VideoMLLM_embedding_lora_HF.py  [MATCH]
vote source  d4adf545125a5a08d78ec9198947dc44f6c6abeec158ed308e138fc9d3d96a5d  scripts/analysis/cross_channel_router_gate.py      [MATCH]
head clone   00d9e9956549bdf97c6b8913d42d87811f4a2f150e9f459e8b8b86978b306f02  scripts/slurm/enc3seed_lora_curric.sbatch          [MATCH]
```
`git status --porcelain` + `git diff --stat HEAD` on the fork source = **empty** ⇒ byte-untouched.

Header verification (prereg §6 resource plan): `gen_embed_readout.sbatch` requests `--cpus-per-task=8`,
`--mem=64G`, `--gres=gpu:a100:1`, and carries **NO `--time`** (L2-8). One combined job (ZH then HateMM
sequential); peak footprint 8 CPU / 64 G / 1 GPU — within the 16/128/2 cap, never two 16-CPU jobs.

## 2. Collision-safety re-check at submit — CLEAN (all ABSENT); banked R0 PRESENT + mtimes recorded

- `data/CLIP_Embedding/{MHC_zh,HateMM}/*-ro_*.pt` — ABSENT (0 found each) ⇒ fresh extraction.
- `logging/Retrieval/*/RAC_video_readout*` — ABSENT (0) ⇒ fresh verdict group; tiebreak group fresh.
- `slurm/logs/*-ro_*.trainlog` + `enc3s_*ro_*seed*.trainlog` — ABSENT (0) ⇒ no trainlog collision.

**Banked R0 caches PRESENT + sha16 matches freeze §5.2; mtimes recorded BEFORE the run (must be UNTOUCHED
after — distinct `-ro_*` suffix cannot clobber the un-suffixed deployed tag):**

| dataset | split | sha16 (freeze) | sha16 (disk) | bytes | mtime (before) |
|---|---|---|---|---|---|
| ZH | train | b2e8e78d19c71d2c | b2e8e78d19c71d2c | 16619871 | 2026-07-02 12:08:59.501321227 +1200 |
| ZH | dev_seen | 4c07af75098391c9 | 4c07af75098391c9 | 2240628 | 2026-07-02 12:11:47.839858186 +1200 |
| ZH | test_seen | 4e107bf65f58745a | 4e107bf65f58745a | 4278267 | 2026-07-02 12:17:25.706949549 +1200 |
| HateMM | train | 5e80f39327a74314 | 5e80f39327a74314 | 21358864 | 2026-07-18 12:26:57.405769081 +1200 |
| HateMM | dev_seen | 46ee4fd9fcaec80b | 46ee4fd9fcaec80b | 3073381 | 2026-07-18 12:29:24.237503621 +1200 |
| HateMM | test_seen | b50ae4ecb077a833 | b50ae4ecb077a833 | 6173356 | 2026-07-18 12:34:15.123051972 +1200 |

All banked sha16 MATCH. mtimes to be re-checked after the run (§6).

## 3. Codex review of the extractor (prereg §6 step 0 / freeze obligation #1) — reads model internals

Artifact A reads model internals (hidden-state layer index + generation-position token), so per CLAUDE.md /
the prereg's own step 0 it is codex-review-gated BEFORE any GPU. `codex exec -m gpt-5.4 -c model_reasoning_effort=xhigh`
via CLI (session `019f9454-…`; 122,924 tokens). Codex read the installed transformers 4.49.0 Qwen2.5-VL
modeling + processing source and the local tokenizer snapshot; it did NOT run a full forward (source-based
review + offline tokenizer/template spot-check).

**VERDICT: NO P1/P2/P3 findings** — "I do not see a readout-grid change that breaks the deployed extractor's
non-readout behavior." All 7 requested invariants confirmed with source citations:

1. **Layer indexing CORRECT.** `Qwen2_5_VLModel` appends embeddings once, per-layer states in the decoder
   loop, then the final post-`self.norm` state ⇒ a 28-layer model yields 29 hidden-state entries, so
   `hidden_states[28] == hidden_states[-1]` (the deployed read) and `hidden_states[24]` is a real intermediate.
   `model.config.num_hidden_layers` is the LLM depth (vision_config uses `depth` separately); the
   `assert num_hidden_layers == 28` guard is correctly placed after load/merge, before extraction.
2. **R0 bit-exact.** `_pool_span()` matches the deployed `_encode()` math for both spans (last-`<|im_start|>`
   boundary, `.float()`, `F.normalize(dim=0)`, `.detach().cpu()`); baseline prompts + Title/Transcript/(none)
   assembly identical; only the one-word passes + `-ro_*` suffix differ.
3. **Masked-scatter invariant valid.** The processor expands each `<|video_pad|>` into
   `video_grid_thw.prod()//merge_size**2` token slots before tokenization; forward checks
   `n_video_tokens == n_video_features` before `masked_scatter` ⇒ decoder seq length == `input_ids` length, so
   `last_hidden.shape[0] == input_ids.numel()` holds at 8 frames.
4. **`last_token` reads a sane deterministic prompt-EOL position** (assistant-header newline after
   `add_generation_prompt=True`; processor default `padding=False`; single example) — not a pad token.
5. **Determinism matches the deployed path** (load kwargs, LoRA merge, bf16/sdpa, `device_map=None`, processor
   `max_pixels`, `np.linspace` sampling, `output_hidden_states=True/use_cache=False`). 4 sequential forwards/item
   (~2× wall time) but peak memory ~1 deployed forward (layers 24/28 harvested from one forward; prompt passes
   not concurrent).
6. **Clobber-safe:** every save uses `out_tag = base_tag + "-" + suffix`, every `CELLS` suffix is `ro_*` ⇒ the
   un-suffixed deployed cache is never written.
7. No other variant-specific crash / silent-corruption path; cache keys + `(N,3584)` shape aligned with the
   deployed contract.

Both the executor's own read and Codex agree — **no P1s; gate CLEARED**. The extractor was **NOT edited**
(sha `ef05f3d4…` still matches the frozen block; authorization intact). Cleared for GPU.
Full transcript: session scratchpad `codex_readout_review.txt`.

## 4. Readout extraction smoke (prereg §4.4.1) — PASS (all load-bearing checks), cleaned up

Throwaway smoke sbatch (session scratchpad `smoke_readout.sbatch`, mirroring `gen_embed_readout.sbatch`'s
env block: conda HateVideo, HF/TRANSFORMERS offline; disk_guard OMITTED as it is a throwaway — the frozen
artifact B keeps it) running the prereg §4.4.1 command verbatim:
`generate_VideoMLLM_embedding_readout_HF.py --dataset HateMM --lora_dir logging/lora/HateMM_curric
--out_model_base_tag Qwen2.5-VL-7B-Instruct-LoRA-curric_HF --splits test --limit 3 --EXP_FOLDER
logging/_smoke_ro --device cuda`.

- **Smoke job 13467** (`smoke_readout`): `sbatch` (NO `--time`); auto-released from `JobHeldUser` (never
  forced; running aggregate was zero); **COMPLETED** exit 0:0, Elapsed 00:00:48 (A100-SXM4-80GB).
- **Namespace (echoed):** `dataset='HateMM', lora_dir='logging/lora/HateMM_curric',
  out_model_base_tag='Qwen2.5-VL-7B-Instruct-LoRA-curric_HF', num_frames=8, splits='test', limit=3,
  device='cuda'` — extractor code unedited; distinct `-ro_*` tags + throwaway EXP_FOLDER.
- **4 caches written** to `logging/_smoke_ro/HateMM/test_seen_..-ro_{L28,L24,ow_L28,ow_L24}.pt`, each
  `Saved ... N=3, Dv=3584, Dt=3584, zero-vector videos=0` (redirected EXP_FOLDER ⇒ never wrote into
  `data/CLIP_Embedding/`).
- **(1) Shapes:** all 4 cells `img (3,3584)` + `text (3,3584)`, ids `[hate_video_1, non_hate_video_4,
  non_hate_video_8]` (first 3 test items in gt order).
- **(2) Finite:** every cell img NaN=0/Inf=0, text NaN=0/Inf=0; zero-vector videos=0.
- **(3) One-word prompts tokenized:** Pass B ran and produced the `ow_L28`/`ow_L24` caches (last-token span)
  with finite (3,3584) tensors — the one-word img/text prompts tokenized as pinned; no exception.
- **(4) No OOM / no assert:** clean scan — no `OOM`/`CUDA error`/`Traceback`/`AssertionError`; the
  L360 masked-scatter invariant (`last_hidden.shape[0]==input_ids.numel()`) held at 8 frames × 4 forwards.
- **(5) R0 BIT-EXACT vs banked (the G-repro anchor):** the `ro_L28` 3-row slice reproduces the **banked
  deployed HateMM cache** rows for those 3 videos **bit-exact** — `img max|Δ| = 0.0`, `text max|Δ| = 0.0`,
  id-order match — on this exact GPU/library stack. R0_BIT_EXACT = **True** (the DEV-5 banked-R0 pairing
  basis + F0.2 determinism gate hold in the smoke).
- **Cleanup:** `logging/_smoke_ro` **deleted**; collision targets (§2) re-verified ABSENT after deletion;
  banked ZH + HateMM caches re-checked UNTOUCHED (sha16 + mtimes bit-identical to the §2 pre-run table).
  Throwaway `smoke_readout.sbatch` lives only in the session scratchpad; smoke slurm log retained at
  `slurm/logs/smoke_readout_13467.out`.

**SMOKE_VERDICT: PASS.** Cleared to submit the real combined extraction job.

## 5. Real extraction — single-submitted (prereg §6 / DEV-2; NO `--time`; combined ZH+HateMM)

Final `sha256sum` re-verified at the submit instant — A `ef05f3d4…`, B `948db851…`, C `f56badb6…` [MATCH];
`bash -n` B = SYNTAX_OK; authorization intact.

| job | id | script | cells | CPU/mem/GPU | ~cost |
|---|---|---|---|---|---|
| extract R0–R3 | **13468** | `gen_embed_readout.sbatch` (ZH then HateMM sequential, hardcoded CONFIGS) → `data/CLIP_Embedding/{MHC_zh,HateMM}/{train,dev_seen,test_seen}_<BASE>-ro_{L28,L24,ow_L28,ow_L24}.pt` | 4 grid caches/dataset | 8 CPU / 64 G / 1×A100 | ~2 GPU-h |

Peak footprint 8 CPU / 64 G / 1 GPU (within the 16/128/2 cap; never two 16-CPU jobs). The readout chain
submits BEFORE the bidirectional-encoder chain (prereg §6). Queue was EMPTY at submit (running aggregate
zero ⇒ favorable for auto-release; the smoke 13467 auto-released from the same hold).

## 5.1 Queue state at submit — PENDING (JobHeldUser); WAIT never force

- **13468 PENDING (JobHeldUser)** (no dependency).

Per CLAUDE.md the hold is **waited out, NEVER forced**. If held > 2 h, a status line is committed and the
turn ends PENDING-JOB (orchestrator resumes). **On COMPLETE:** cache sanity (§6 — row counts/dims/NaN, R0
full-cache bit-exact vs banked, banked mtimes unchanged), then the `$0` CPU screen (§7).
