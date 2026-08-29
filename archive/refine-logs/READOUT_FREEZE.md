# READOUT-GRID (R0–R3) — HASH-FREEZE

Recorded by the independent 0-context pre-registration reviewer at freeze time.
**Prereg NOT modified** (per review mandate); this file is the authoritative freeze record.
**Verdict:** APPROVED-WITH-NOTES (see `refine-logs/READOUT_PREREG_REVIEW.md`).
**Date:** 2026-07-25 NZST. Prereg commit `1b3e0c6`; recon `61a9f4a`.

```
FROZEN f6d43096332acb2e9f9743f5e51967f6a93b3006770be3520effc13929aac543  refine-logs/READOUT_PREREG.md (commit 1b3e0c6)
A ef05f3d45a3e8c31f8dc198ba41e18c2e525cd29e9ba0ed539dfd9b4c6d869c3  src/utils/generate_VideoMLLM_embedding_readout_HF.py
B 948db8514c9e4b02d6d20ceed3e6a63104893c8a6e623def75e4c22bc9419e29  scripts/slurm/gen_embed_readout.sbatch
C f56badb64b9dc8a4d18fbbcbbff99994234df3812dccd7334f8827e100d35547  scripts/analysis/readout_screen.py
```

## Reused-unchanged machinery (verify at submit; do NOT edit)

```
b6b61a3fa4214f28d2098ca305ca9a981934445afc81ccc5ac3939f8c1fb0ec6  src/utils/generate_VideoMLLM_embedding_lora_HF.py   (fork source — git-clean, byte-untouched)
d4adf545125a5a08d78ec9198947dc44f6c6abeec158ed308e138fc9d3d96a5d  scripts/analysis/cross_channel_router_gate.py       (vote source; _weighted_signed_vote L73-79 byte-verbatim in C)
00d9e9956549bdf97c6b8913d42d87811f4a2f150e9f459e8b8b86978b306f02  scripts/slurm/enc3seed_lora_curric.sbatch           (verdict-head clone source, §1.3)
```

## Banked paired-floor caches (NOT clobbered; sha16 matches prereg §5.2)

```
ZH  R0 (Qwen2.5-VL-7B-Instruct-LoRA_HF)        train b2e8e78d19c71d2c  dev 4c07af75098391c9  test 4e107bf65f58745a
HateMM R0 (Qwen2.5-VL-7B-Instruct-LoRA-curric_HF) train 5e80f39327a74314  dev 46ee4fd9fcaec80b  test b50ae4ecb077a833
```

## Executor obligations at submit time

1. **Codex-review the extractor** (artifact A reads model internals: hidden-state layer index +
   generation-position token) BEFORE any GPU, per CLAUDE.md / precedent (the prereg's own step 0).
2. Re-run `sha256sum` on the prereg + A/B/C and confirm the fork-source extractor sha `b6b61a3f…` unchanged;
   **any mismatch = authorization VOID.**
3. Re-check collisions absent: `data/CLIP_Embedding/{MHC_zh,HateMM}/*-ro_*.pt`,
   `logging/Retrieval/*/RAC_video_readout*`, `slurm/logs/*-ro_*.trainlog`.
4. **R0 bit-exact gate is the G-repro anchor** — if `max|Δ|` between `-ro_L28` and the banked deployed
   cache is nonzero on train or dev, it is **VOID / investigate**; do NOT relax the gate to an epsilon and
   do NOT proceed to any Δ or verdict (a nonzero R0 Δ invalidates the winner-vs-banked-R0 pairing). See
   Note 1 of the review.
5. `sbatch` with **NO `--time`**; `PENDING (JobHeldUser)` → wait for auto-release, never force. The readout
   chain submits BEFORE the bidirectional-encoder chain.
6. `$0` screen: `CUDA_VISIBLE_DEVICES="" python scripts/analysis/readout_screen.py`. Flat (no cell ≥ +0.020
   dev on either dataset in either arm) ⇒ **KS-readout-dead**, STOP, no test-touch. Promote ⇒ escalate to
   the normal ceremony (author verdict head → 0-context review → freeze → single-test-touch 3-seed verdict
   paired vs the banked R0).

*Reviewer spent zero GPU/SLURM/Modal. CPU-only verification. Prereg not modified. Not pushed.*
