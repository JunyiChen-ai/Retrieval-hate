# VISION-UNFREEZE LoRA-SFT — SUBMIT RECORD (submit executor)

**Role:** submit executor. ZERO user interaction. NO push. NO test metric read. NO verdict produced.
RAW-ONLY: the executor transcribes no interpretive language; the verdict is rendered by an independent
0-context reviewer against the prereg VERBATIM.
**Date:** 2026-07-20 NZST.
**Prereg:** `refine-logs/VISION_UNFREEZE_PREREG.md`, FROZEN sha256
`a2bb1c45b44de35fdfb959fdf07d1b1146546b0944f37fe95434e49e4ed2be2d`.
**Freeze:** `refine-logs/VISION_UNFREEZE_FREEZE.md` (reviewer `b93a4be`, verdict APPROVED-WITH-NOTES).
**Review:** `refine-logs/VISION_UNFREEZE_PREREG_REVIEW.md` (commit `b93a4be`).

---

## 1. Sha re-verification at submit time — ALL MATCH (authorization intact)

Re-ran `sha256sum` on the prereg, artifacts A–E, reused-unchanged machinery, and the SFT data at submit time.
**Every hash matches the frozen block; authorization is intact.**

### Prereg
```
a2bb1c45b44de35fdfb959fdf07d1b1146546b0944f37fe95434e49e4ed2be2d  refine-logs/VISION_UNFREEZE_PREREG.md   [MATCH]
```

### Frozen artifacts A–E
```
A 7d551460239aaf537ecbb62f4c77d859cfeea3403867ccb99b34d31eeeb7fd3f  mhc_qwen25vl_lora_vis_sft.yaml        [MATCH]
B 634bd0bb02789a1728728be19efdf91b69b36aab27a5f1dd9eab229e3041700b  hatemm_qwen25vl_lora_vis_sft.yaml     [MATCH]
C 3e895420e308b30d8371c54a7a03ab9cf033ebe4804143a511989e68f3ef7946  scripts/slurm/lora_sft_vis.sbatch     [MATCH]
D ca7749149fd836bd84404cad8436fd868c51c1ff2930c3ed9e91657c6933e2fb  scripts/slurm/enc3seed_lora_vis.sbatch [MATCH]
E 719ab1fe837ad4c9f75c750b8e8e5d5853bd64cdcf3c526da35fe0177944c4a6  scripts/analysis/vis_image_moved_probe.py [MATCH]
```

### Reused-unchanged machinery
```
974771775e15fd58c31bd07bfd26d6dac43eab304b5fd888235a8449009190f6  scripts/analysis/encoder_swap_geometry.py   [MATCH]
c76bb42240feaa300c8b89cdb1fdba1c2d0dbb7360b0ffe53d32fc260a46f386  scripts/slurm/gen_embed_lora.sbatch          [MATCH]
dbe3fb81800897cb7bac56d71f5d881d54d46421fdbda214df00d4deb0815c3d  scripts/slurm/enc3seed.sbatch                [MATCH]
db371c18f306c5a3a00eeef8550964c3ddacf9e20400439324009ef2e69b1b52  mhc_qwen25vl_lora_sft.yaml   (EN fork source) [MATCH]
d2f415cd93fa6f7b439fd4b4573a536baf48ad42186dc8bd50f3fab20553e36a  hatemm_qwen25vl_lora_sft.yaml (HateMM src)   [MATCH]
```

### SFT data (== banked generic comparator)
```
7fe4c654b19a30bb48f6a7e6479ea8c009d6ce4df3406c14c241d68b987e1bba  data/lora_sft/MHC/train.json   (549)  [MATCH]
575c84f254ebdfa90edc9be572d4cdb592afafeca54330c2b1b266ed24976571  data/lora_sft/MHC/val.json     (80)   [MATCH]
93c6d3d1bffbca22b2dd8beba57a33575a48d8ca61d8d56e3148fecdbb93973a  data/lora_sft/HateMM/train.json (743) [MATCH]
9e103ed35a014af81eb3aa6af0d51a28707efd66a606c5bf0459db570a9cc9ef  data/lora_sft/HateMM/val.json   (107) [MATCH]
```

## 2. Collision-safety re-check at submit — CLEAN (all ABSENT)

- `logging/lora/{MHC_vis,HateMM_vis,_smoke_vis}` — absent (fresh SFT; generic MHC/HateMM adapters not clobbered).
- `data/CLIP_Embedding/{MHC,HateMM}/*LoRA-vis*.pt` — absent (fresh extraction; frozen + `-LoRA_HF` +
  `-LoRA-curric_HF` caches untouched).
- `logging/Retrieval/{MHC,HateMM}/RAC_video_lora_vis*` — absent (fresh group; `force=False` never trips).
- `slurm/logs/enc3s_*LoRA-vis*.trainlog` — absent.
- `squeue -u jehc223` — empty before smoke submit.

Environment prerequisites verified: `mhc_lora_train`/`hatemm_lora_train` (+`_val`) registered in LF
`dataset_info.json` pointing at the frozen `data/lora_sft/*/{train,val}.json`; MHC frames cached; base model
`Qwen/Qwen2.5-VL-7B-Instruct` present locally (HF offline cache); `.cuda_home_shim` present; disk free 672G
(>20G guard).

## 3. SFT smoke (prereg §4.4.1 / §4.1a) — PASS (all 3 LOAD-BEARING checks), cleaned up

- **Job 13299** (`lora_sft_smoke_vis`): throwaway smoke config (scratchpad `smoke_vis_sft.yaml` = frozen MHC vis
  config A with `max_steps: 20`, `save_strategy: steps` + `save_steps: 20`, `eval_strategy: "no"`,
  `output_dir: logging/lora/_smoke_vis`) via a throwaway smoke sbatch mirroring `lora_sft_vis.sbatch`'s env
  block (conda HateVideo, HF offline, `CUDA_HOME` shim, `DISABLE_VERSION_CHECK`). Submitted with `sbatch`
  (NO `--time`); auto-released from `JobHeldUser` (never forced); **COMPLETED** (train_runtime 1958.9 s over 20
  steps; the ViT-unfreeze backward makes each step ~98 s — expected).
- **(1) Loss finite + decreasing (raw `trainer_log.jsonl` / stdout):** step5 `0.3401` → step10 `0.1409` →
  step15 `0.1387` → step20 `0.1561`; `train_loss 0.19394605`. No NaN/inf/traceback/OOM (clean scan). Downtrend
  with a normal final-step wiggle; lands in the §3.7b ~0.10–0.18 band for the trained-step region.
- **(2) Checkpoint written:** `logging/lora/_smoke_vis/checkpoint-20/adapter_model.safetensors` (206 MB) +
  `optimizer.pt` etc., and the final root adapter `logging/lora/_smoke_vis/adapter_model.safetensors` (206 MB;
  vs the generic adapter's 161 MB — the +44.6 MB ≈ 11.15M fp32 ViT-LoRA params).
- **(3) LOAD-BEARING ViT-LoRA-present census (prereg §4.1a exact command, on the root smoke adapter):**
  - **`n_visual_lora_tensors = 320` (> 0 ⇒ PASS; the arm genuinely reaches the ViT, is NOT degenerate-to-generic).**
  - Structure: 320 visual = 32 ViT blocks (indices 0..31) × 5 Linears
    {`attn.qkv`, `attn.proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`} × {lora_A, lora_B}.
  - `n_llm_lora_tensors = 392` (byte-identical count to the banked generic adapter's 392 tensors ⇒ clean-superset
    premise confirmed: vis = generic ⊕ ViT-LoRA; total 712).
  - `n_merger_lora_tensors = 0` (projector frozen ✓); `n_patchembed_lora_tensors = 0` (Conv3d / lora_conflict ✓).
- **Cleanup:** `logging/lora/_smoke_vis` **deleted** (prereg §4.4); §2 collision targets re-verified ABSENT after
  deletion. Throwaway `smoke_vis_sft.yaml` + `lora_sft_smoke_vis.sbatch` live only in the session scratchpad;
  smoke slurm log retained at `logging/slurm/lora_sft_smoke_vis_13299.out` as evidence.
- **Step-2 head smoke: SKIPPED** — prereg §4.4 step 2 permits skipping; `run_one` is byte-identical to the banked
  controls (freeze §4.2) and cache dims are CPU-verified. Not run to avoid needless queue contention.

## 4. Real chain — single-submitted (prereg §6; NO `--time`; afterok-wired)

| job | id | script + args | dependency | GPU | ~cost |
|---|---|---|---|---|---|
| J1 SFT MHC (EN) | **13301** | `lora_sft_vis.sbatch MHC` → `logging/lora/MHC_vis` | (none) | 1×A100 | ~4–5.5 h |
| J2 extract MHC | **13302** | `gen_embed_lora.sbatch MHC logging/lora/MHC_vis Qwen2.5-VL-7B-Instruct-LoRA-vis_HF` | `afterok:13301` | 1×A100 | ~0.4 h |
| J3 SFT HateMM | **13303** | `lora_sft_vis.sbatch HateMM` → `logging/lora/HateMM_vis` | (none) | 1×A100 | ~4–5.5 h |
| J4 extract HateMM | **13304** | `gen_embed_lora.sbatch HateMM logging/lora/HateMM_vis Qwen2.5-VL-7B-Instruct-LoRA-vis_HF` | `afterok:13303` | 1×A100 | ~0.4 h |

Dependency graph (squeue-verified): `13301 → 13302`, `13303 → 13304`. Peak concurrent GPU = 2 (J1+J3), within
the 2-GPU user cap. All submitted `sbatch --parsable` (recipe sbatch carry NO `--time`).

**Deferred (submitted after the EN image-MOVED $0 gate read + both extracts):**
- **EN image-MOVED gate (§3.4):** `python scripts/analysis/vis_image_moved_probe.py --dataset MHC --context`
  ($0 CPU, after J2). Records `en_head_proceeds`; BRANCH POINT: MOVED → EN head rows IN; FLAT/DEGRADED → EN head
  CANCELLED (bank raw gate output), head job runs `"HateMM"` only.
- **J5 head:** `enc3seed_lora_vis.sbatch "$DSLIST"` (`"HateMM MHC"` if gate MOVED, else `"HateMM"`), after J2 AND
  J4. Arg-driven DATASETS = NO frozen-file edit.

## 5. Queue state at submit

J1 = **13301 RUNNING** (auto-released immediately, node foscsmlprd01); J2/J3/J4 **PENDING (JobHeldUser)** or
dependency-held (normal per CLAUDE.md; holds NEVER forced). J3 (2nd SFT) auto-releases when the 2nd GPU frees.
