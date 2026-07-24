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
the prereg's own step 0 it is codex-review-gated BEFORE any GPU. `gpt-5.4` `xhigh` via CLI.

_(filled below on completion)_
