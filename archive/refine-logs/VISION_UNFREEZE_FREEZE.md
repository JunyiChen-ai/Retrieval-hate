# VISION-UNFREEZE Pre-Registration — HASH-FREEZE

**Frozen by:** independent 0-context reviewer, after `refine-logs/VISION_UNFREEZE_PREREG_REVIEW.md`
(verdict **APPROVED-WITH-NOTES**, review commit `b93a4be`).
**Date:** 2026-07-20 NZST. **CPU-only; no job submitted; prereg NOT modified; not pushed.**

This locks `refine-logs/VISION_UNFREEZE_PREREG.md` and its 5 authored artifacts. The prereg body's §5.3
placeholder is intentionally left as-is (reviewer does not edit the prereg); the authoritative frozen values are
recorded here. **The executor MUST re-run `sha256sum` on the prereg + A–E (and re-verify the reused-machinery +
SFT-data shas) at submit time; any mismatch = authorization VOID.**

## Git provenance

| object | commit |
|---|---|
| prereg + artifacts C/D/E (parent repo) | `c1592bb` ("prereg: vision-unfreeze LoRA-SFT (EN + HateMM) DRAFT, unreviewed") |
| configs A/B (submodule `RA-HMD/LLAMA-FACTORY-Ver202512`) | `a912747c` ("hatevideo: vision-unfreeze LoRA-SFT configs (EN + HateMM) — freeze_vision_tower false + lora_target all") |
| review | `b93a4be` ("review: vision-unfreeze prereg APPROVED-WITH-NOTES (0-context)") |
| this freeze | *(the commit that adds this file)* |

All four parent-repo artifacts are tracked at `c1592bb` with **no on-disk drift** (`git diff HEAD` empty); both
configs are tracked inside the submodule at `a912747c`.

## FROZEN hashes (re-verified on disk at freeze time)

```
FROZEN a2bb1c45b44de35fdfb959fdf07d1b1146546b0944f37fe95434e49e4ed2be2d  refine-logs/VISION_UNFREEZE_PREREG.md
A 7d551460239aaf537ecbb62f4c77d859cfeea3403867ccb99b34d31eeeb7fd3f  RA-HMD/.../my_configs/hatevideo/mhc_qwen25vl_lora_vis_sft.yaml
B 634bd0bb02789a1728728be19efdf91b69b36aab27a5f1dd9eab229e3041700b  RA-HMD/.../my_configs/hatevideo/hatemm_qwen25vl_lora_vis_sft.yaml
C 3e895420e308b30d8371c54a7a03ab9cf033ebe4804143a511989e68f3ef7946  scripts/slurm/lora_sft_vis.sbatch
D ca7749149fd836bd84404cad8436fd868c51c1ff2930c3ed9e91657c6933e2fb  scripts/slurm/enc3seed_lora_vis.sbatch
E 719ab1fe837ad4c9f75c750b8e8e5d5853bd64cdcf3c526da35fe0177944c4a6  scripts/analysis/vis_image_moved_probe.py
```

## Reused-unchanged machinery (re-verify at submit; do NOT edit)

```
974771775e15fd58c31bd07bfd26d6dac43eab304b5fd888235a8449009190f6  scripts/analysis/encoder_swap_geometry.py   (F58 kNN-AUC operator, imported by E)
c76bb42240feaa300c8b89cdb1fdba1c2d0dbb7360b0ffe53d32fc260a46f386  scripts/slurm/gen_embed_lora.sbatch          (extraction; out-tag arg 3)
dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d  scripts/slurm/enc3seed.sbatch                (same-code anchor for run_one)
db371c18f306c5a3a00eeef8550964c3ddacf9e20400439324009ef2e69b1b52  RA-HMD/.../mhc_qwen25vl_lora_sft.yaml         (EN generic fork source)
d2f415cd93fa6f7b439fd4b4573a536baf48ad42186dc8bd50f3fab20553e36a  RA-HMD/.../hatemm_qwen25vl_lora_sft.yaml      (HateMM generic fork source)
```

## SFT data (== the banked generic comparator trained on; re-verify at submit)

```
7fe4c654b19a30bb48f6a7e6479ea8c009d6ce4df3406c14c241d68b987e1bba  data/lora_sft/MHC/train.json   (549)
575c84f254ebdfa90edc9be572d4cdb592afafeca54330c2b1b266ed24976571  data/lora_sft/MHC/val.json     (80)
93c6d3d1bffbca22b2dd8beba57a33575a48d8ca61d8d56e3148fecdbb93973a  data/lora_sft/HateMM/train.json (743)
9e103ed35a014af81eb3aa6af0d51a28707efd66a606c5bf0459db570a9cc9ef  data/lora_sft/HateMM/val.json   (107)
```
`HateMM/train.json 93c6d3d1…` is byte-identical to the pin in `LORA_HATEMM_PREREG.md` and
`CAND2_CURRICULUM_PREREG.md`.

## Independent re-derivations banked at freeze (CPU-only)

- **Floors (§2.1/§2.2):** all 24 per-seed acc/F1 pairs + 12 means re-parsed to 4dp from raw
  `enc3s_*_{12850,13235}.trainlog` + `arcbase_MHC_*_1227{5,6}.trainlog` with the `enc3seed.sbatch` embedded rule.
- **§2.3 EN image anchor (committed F58 operator, train n=549 + dev_seen n=80, `CUDA_VISIBLE_DEVICES=""`):**
  CLIP 0.7338/0.7367 · frozen-Qwen 0.5992/0.6865 · **generic-LoRA 0.6236/0.6756** — all exact.
  DEV-4 machinery-validation delta (frozen→generic): **+0.024450 train / −0.010909 dev** ⇒ FLAT (reproduced).
- **Generic adapters:** 88 config target modules / 392 safetensors tensors / 40,370,176 params, **ZERO `visual.*`**
  (K-V2 clean-superset premise) — both `logging/lora/{HateMM,MHC}`.
- **eval_loss anchors (§3.7b):** HateMM 0.10844 / MHC 0.16196 (→ 0.1084 / 0.1620).

## Authorization

Frozen and **cleared to single-submit** under the CLAUDE.md ceremony (single-submit discipline, no `--time`,
`JobHeldUser` wait-never-force). Execution order: SFT → extract → EN image-MOVED gate ($0 CPU) → head, with the
EN early-kill as a submit-time `DATASETS` argument to `enc3seed_lora_vis.sbatch` (no frozen-file edit). The
verdict on the resulting test-touch reads is rendered by a fresh independent 0-context reviewer against this
prereg VERBATIM. **Notes 1–3 of the review are non-blocking**; the verdict reviewer should apply Note 1 (wider
EN generic between-seed spread ⇒ 3/3-sign carries the EN K-V2 discrimination; F0.2 single-draw caveat prominent)
when interpreting any EN K-V2 pass.
