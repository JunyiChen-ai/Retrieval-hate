# B5 HEADS SAFEKEEP MANIFEST — step-0 checkpoint snapshot (disk-guard mitigation)

**Date:** 2026-07-14 (copy verified 2026-07-14 08:40 UTC).
**Task:** `B5_PROBE_DESIGN.md` §2.3 executor step 0 — snapshot the 12 required head checkpoints
to a guard-excluded path before the concurrent `disk_guard.sh` oldest-first cleanup can prune them.
**Destination (verbatim from §2.3):** `/data/jehc223/RGCL/refine-logs/b5_ckpt_snapshot/`
(NOT under `logging/` ⇒ outside disk-guard scope). Source tree `logging/…` was READ-ONLY (copy only).
**No git commit** (large `.pt`; manifest to be committed later with the B5 batch).

## Verification result

- **12/12 logical (arm × protocol) slots present and verified: YES.**
- **11 distinct physical files** (CLIP s0's final-epoch and val-selected are the SAME file — both
  epoch 29 — so its two protocol slots share one `.pt`; the `{model}_s{S}_e{E}.pt` naming dedupes it).
- **sha256 source-vs-copy: 11/11 MATCH.** No missing files. No mismatches.
- **Total distinct bytes: 351,600,470 (335.3 MiB / 351.6 MB).** (§2.3 estimated "~372 MB" because it
  counts CLIP s0 twice across the 12 logical slots; physically distinct footprint is 335.3 MiB.)
- Each source glob (`epoch_model_{E}_*.pt`) matched exactly one file, and every resolved filename
  matches the §2.1 table float-suffix exactly (transcription cross-check passed).
- Copies made with `cp -p` (mtime/mode preserved).

## Slot → file map (12 logical slots)

| arm | seed | protocol | epoch | destination file |
|---|---|---|---|---|
| CLIP | 0 | B (final) & A (val-sel) | 29 | `CLIP_s0_e29.pt` (one file serves both slots) |
| CLIP | 1 | B (final) | 29 | `CLIP_s1_e29.pt` |
| CLIP | 1 | A (val-sel) | 28 | `CLIP_s1_e28.pt` |
| CLIP | 2 | B (final) | 29 | `CLIP_s2_e29.pt` |
| CLIP | 2 | A (val-sel) | 25 | `CLIP_s2_e25.pt` |
| Qwen | 0 | B (final) | 29 | `Qwen_s0_e29.pt` |
| Qwen | 0 | A (val-sel) | 22 | `Qwen_s0_e22.pt` |
| Qwen | 1 | B (final) | 29 | `Qwen_s1_e29.pt` |
| Qwen | 1 | A (val-sel) | 25 | `Qwen_s1_e25.pt` |
| Qwen | 2 | B (final) | 29 | `Qwen_s2_e29.pt` |
| Qwen | 2 | A (val-sel) | 28 | `Qwen_s2_e28.pt` |

## Full manifest (11 distinct files: source path, size, sha256, copy)

Source base dir: `logging/Retrieval/MHC_zh/RAC_video_archive_seeds/RAC_lr0.0001_Bz64_Ep30_cosSim_triplet_drop[0.2, 0.4, 0.1]_topK20__PseudoGold_positive_1_hard_negative_1_seed{S}_hybrid_loss_{MODEL}/ckpt/`

### CLIP (model = `openai_clip-vit-large-patch14-336_HF`)

1. **CLIP s0 e29** (serves protocol B + A) — 19,952,722 bytes
   - src: `…seed0_hybrid_loss_openai_clip-vit-large-patch14-336_HF/ckpt/epoch_model_29_0.8076923076923077.pt`
   - dst: `refine-logs/b5_ckpt_snapshot/CLIP_s0_e29.pt`
   - sha256: `a8ccc5eecb39fee379a03ba6c138cff68b74bb41a545adbb632dc56d0c237367`
2. **CLIP s1 e29** (protocol B) — 19,952,722 bytes
   - src: `…seed1_…openai_clip-vit-large-patch14-336_HF/ckpt/epoch_model_29_0.7692307692307693.pt`
   - dst: `refine-logs/b5_ckpt_snapshot/CLIP_s1_e29.pt`
   - sha256: `3d07a26fd4ab9a62261e9b77f28656af386f4bc66b0af0c4c2d536c488d8d35b`
3. **CLIP s1 e28** (protocol A) — 19,952,706 bytes
   - src: `…seed1_…openai_clip-vit-large-patch14-336_HF/ckpt/epoch_model_28_0.782051282051282.pt`
   - dst: `refine-logs/b5_ckpt_snapshot/CLIP_s1_e28.pt`
   - sha256: `7f547e0828d19035457f8a8f50af60b3c77149c7d846ae16e033abf92b88978f`
4. **CLIP s2 e29** (protocol B) — 19,952,722 bytes
   - src: `…seed2_…openai_clip-vit-large-patch14-336_HF/ckpt/epoch_model_29_0.7948717948717948.pt`
   - dst: `refine-logs/b5_ckpt_snapshot/CLIP_s2_e29.pt`
   - sha256: `5c5779e51d317dd6024a8defe628e75e99a7aa707aab29d5c9f0cd6ee6554719`
5. **CLIP s2 e25** (protocol A) — 19,952,722 bytes
   - src: `…seed2_…openai_clip-vit-large-patch14-336_HF/ckpt/epoch_model_25_0.8205128205128205.pt`
   - dst: `refine-logs/b5_ckpt_snapshot/CLIP_s2_e25.pt`
   - sha256: `8532add57e3af25dcac7a9329a6043b88b5d30b9374b0a97c10799fed97bdba8`

### Qwen (model = `Qwen2.5-VL-7B-Instruct_HF`)

6. **Qwen s0 e29** (protocol B) — 41,972,802 bytes
   - src: `…seed0_hybrid_loss_Qwen2.5-VL-7B-Instruct_HF/ckpt/epoch_model_29_0.782051282051282.pt`
   - dst: `refine-logs/b5_ckpt_snapshot/Qwen_s0_e29.pt`
   - sha256: `ce2919e6b1c7377498fd81cc6e1498dd316a0335394374700c74b310d1dc6ea8`
7. **Qwen s0 e22** (protocol A) — 41,972,818 bytes
   - src: `…seed0_…Qwen2.5-VL-7B-Instruct_HF/ckpt/epoch_model_22_0.8205128205128205.pt`
   - dst: `refine-logs/b5_ckpt_snapshot/Qwen_s0_e22.pt`
   - sha256: `b4601d7b26ccf83bdbb69dcb99cf65e966ec3a7bab745770760c625312215202`
8. **Qwen s1 e29** (protocol B) — 41,972,818 bytes
   - src: `…seed1_…Qwen2.5-VL-7B-Instruct_HF/ckpt/epoch_model_29_0.8205128205128205.pt`
   - dst: `refine-logs/b5_ckpt_snapshot/Qwen_s1_e29.pt`
   - sha256: `c790de95e1e0c2a422ffad0c93df258aa69a6eb517f92ce9af38af09082a1856`
9. **Qwen s1 e25** (protocol A) — 41,972,818 bytes
   - src: `…seed1_…Qwen2.5-VL-7B-Instruct_HF/ckpt/epoch_model_25_0.8717948717948718.pt`
   - dst: `refine-logs/b5_ckpt_snapshot/Qwen_s1_e25.pt`
   - sha256: `a9e3b66f8722c5ee3c6252c456a7c5a3deab4c57320ac50be8643e22be90a750`
10. **Qwen s2 e29** (protocol B) — 41,972,802 bytes
    - src: `…seed2_…Qwen2.5-VL-7B-Instruct_HF/ckpt/epoch_model_29_0.782051282051282.pt`
    - dst: `refine-logs/b5_ckpt_snapshot/Qwen_s2_e29.pt`
    - sha256: `0898389628a94160c632cb27a2d4a4289aa0d22e417098a5a46fcd7210ceb871`
11. **Qwen s2 e28** (protocol A) — 41,972,818 bytes
    - src: `…seed2_…Qwen2.5-VL-7B-Instruct_HF/ckpt/epoch_model_28_0.8461538461538461.pt`
    - dst: `refine-logs/b5_ckpt_snapshot/Qwen_s2_e28.pt`
    - sha256: `643212cbb6ac843973d649467e803d8264b0cbb39af43cb2d2162b4e899d5ab0`

## Notes for the executor

- The snapshot is the reload source for `scripts/analysis/b5_calibration_probe.py` (§3.1
  `CKPT_PATH`). For CLIP s0 both protocols load `CLIP_s0_e29.pt`.
- If the originals under `logging/` are pruned by the guard before the probe runs, this snapshot is
  authoritative; the sha256 column here is the integrity check. Deeper fallback = B2 mirror or the §6
  GPU regeneration.
- Snapshot dir holds exactly these 11 `.pt` files, nothing else.
