# FRAME16 Pre-Registration — HASH-FREEZE

**Frozen by:** independent 0-context pre-registration reviewer (CPU-only; no GPU/SLURM/Modal spent; no job
submitted; prereg NOT modified).
**Date:** 2026-07-21 NZST.
**Verdict:** `APPROVED-WITH-NOTES` (see `refine-logs/FRAME16_PREREG_REVIEW.md`; 3 notes, all non-blocking).
**Prereg commit:** `0b5cbb5` ("prereg: frozen-Qwen-16f stage-1 (HateMM) DRAFT, unreviewed").
**Review commit:** `7164cbb` ("review: frame16 prereg APPROVED-WITH-NOTES (0-context)").

All values below were re-computed on disk at freeze time and **match** the prereg §5 block. Any mismatch at
submit time = **authorization VOID**.

```
FROZEN 5c240518217e1ab69cbd52de34c8849450d8b37ad163d68b4247c5f2c791c725  refine-logs/FRAME16_PREREG.md (commit 0b5cbb5)
B a600e74c0a6483095329f9ce15a3df19c842554362f7a3ef1f6e76e26fe3c750  scripts/slurm/gen_embed_mllm_16f.sbatch
C 99e7e8b10286e22d7913e85c14141c8fa02c90ae27adc0da6facaceeb703864a  scripts/slurm/enc3seed_fb16.sbatch
```

**Reused-unchanged machinery (re-verify at submit; do NOT edit):**

```
extractor    d89a912602d763aa055a54f50b0188e302e554b70ff6c0eb872f250bd454b67c  src/utils/generate_VideoMLLM_embedding_HF.py
fork source  9357fa1087e775d059779e6c5f86e19e71b78b2d166f904fa3c71a1a1cbb3268  scripts/slurm/gen_embed_mllm.sbatch
head anchor  dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d  scripts/slurm/enc3seed.sbatch
```

**Banked 8f floor (paired anchor; NOT clobbered — distinct out-tag):** present and untouched at freeze time —
`data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF.pt`
(21358780 / 3073233 / 6173272 bytes, dated Jul 2). The 16f extraction writes `…_HF-16f.pt` (hardcoded out-tag,
`gen_embed_mllm_16f.sbatch:37`), so these files cannot be overwritten.

## Independently verified at freeze (summary; full detail in the review)

- **Floors (§2.1) re-parsed to 4dp** from the raw `enc3s_HateMM_…_HF_seed{0,1,2}_12850.trainlog` with an
  independently written enc3seed parser: val-sel 0.8698/0.8651/0.8837 (sel ep 28/22/29), final 0.8605/0.8605/0.8837
  (ep 29); means val-sel **0.8729/0.8648**, final **0.8682/0.8591**. Exact.
- **Head `run_one` byte-identical** to `enc3seed.sbatch` (block sha256 `286a9e44953ff2b2f17af3821f3ed3e254569cb68893fefe6b451b04d6ab9101`).
- **Extractor 16f is one-arg, no-code-edit:** `--num_frames`/`--out_model_tag` pre-existing (L84-95); L283 assert
  holds at 16; `_encode` pooled operator unchanged; out-path `{EXP_FOLDER}/{ds}/{outname}_{out_model_tag}.pt`.
- **Clobber-catch real & defused:** fork source cannot set `--out_model_tag`; the new sbatch hardcodes it.
- **Loader** routes `…_HF-16f` → 16f cache (`dataset.py:499-503` / `:605-608`).
- **Collisions** (`…_HF-16f.pt`, `RAC_video_fb16*`, `…_HF-16f_seed*.trainlog`, `_smoke_fb16`) verified **ABSENT**.
- **Resources within cap:** each sbatch 8 CPU / 64 G / 1 A100; `afterok` chain ⇒ peak 8 CPU / 64 G / 1 GPU; NO
  `--time`; JobHeldUser wait-never-force.

## Submit gate (executor)

Re-run `sha256sum` on `refine-logs/FRAME16_PREREG.md` + B + C and confirm the extractor sha `d89a9126…` unchanged
at submit time; re-confirm the four collision paths ABSENT and the banked 8f caches present. Any mismatch =
authorization VOID. Then: smoke (§4.4.1) → `sbatch scripts/slurm/gen_embed_mllm_16f.sbatch HateMM` → shape sanity
(§4.1c) → `sbatch --dependency=afterok:<1> scripts/slurm/enc3seed_fb16.sbatch`. Executor transcribes raw
both-protocol per-seed numbers (line-numbered) and applies NO gates; the verdict is rendered by an independent
0-context reviewer against the prereg verbatim.

*Freeze performed CPU-only; prereg not modified; not pushed.*
