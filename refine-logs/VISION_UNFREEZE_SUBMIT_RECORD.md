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

## 6. EN leg COMPLETE — G-repro + cache sanity + image-MOVED gate (raw)

**Job states (sacct):** J1 13301 SFT-EN **COMPLETED** (06:21:10, exit 0:0); J2 13302 extract-EN **COMPLETED**
(00:34:37, exit 0:0). J3 13303 SFT-HateMM PENDING (JobHeldUser — waiting, never forced); J4 13304 extract-HateMM
PENDING (dependency).

### 6.1 J1 (MHC_vis) SFT G-repro — PASS
- **eval_loss = 0.17310** (`logging/lora/MHC_vis/all_results.json`) — inside the §3.7b **0.10–0.18** band
  (generic MHC anchor 0.1620). It is HIGHER, not much-lower, so the §3.7b overfit tripwire (much-lower
  eval_loss + widening gap) does NOT fire. train_loss 0.10396; train_runtime 20773 s.
- **Real-adapter ViT-tensor census** (`logging/lora/MHC_vis/adapter_model.safetensors`, 206 MB):
  **n_visual_lora_tensors = 320** (32 ViT blocks 0..31 × 5 Linears × {A,B}), **n_llm = 392** (== generic ⇒
  clean-superset), **n_merger = 0**, **n_patchembed = 0**. Total 712. Matches the smoke census bit-for-bit.

### 6.2 J2 (MHC_vis extract) cache sanity — PASS (tag `Qwen2.5-VL-7B-Instruct-LoRA-vis_HF`)
| split | N | img_feats | text_feats | labels | img NaN | text NaN | N/dim == generic | ids == generic |
|---|---|---|---|---|---|---|---|---|
| train | 549 | (549, 3584) | (549, 3584) | 549 | 0 | 0 | yes | yes |
| dev_seen | 80 | (80, 3584) | (80, 3584) | 80 | 0 | 0 | yes | yes |
| test_seen | 161 | (161, 3584) | (161, 3584) | 161 | 0 | 0 | yes | yes |

Dual-stream 3584-d (§1.3); zero NaN; N and ids identical to the banked generic-LoRA EN cache (the id alignment
the gate asserts). Frozen / `-LoRA_HF` / `-LoRA-curric_HF` caches untouched (distinct tag). (test_seen inspected
for STRUCTURE only — shape/count/NaN/id — no test metric, no `Test_*` line read; RAW-ONLY holds.)

### 6.3 EN image-MOVED $0 gate (§3.4) — MOVED → EN-HEAD-PROCEEDS (raw; committed F58 operator, train+dev only)
`python scripts/analysis/vis_image_moved_probe.py --dataset MHC --context` (CUDA_VISIBLE_DEVICES="", K=20);
JSON `scripts/analysis/vis_image_moved_MHC_out.json`.

| footing | generic-LoRA img AUC | vis-LoRA img AUC | dAUC(vis−gen) | threshold | same-sign? |
|---|---|---|---|---|---|
| train-LOO | 0.6236 | 0.6556 | **+0.0320** | ≥ +0.010 | ✓ |
| dev | 0.6756 | 0.6822 | **+0.0065** | ≥ +0.005 | ✓ |

- generic-LoRA anchor **reproduces the frozen §2.3 value exactly** (0.6236 / 0.6756). Context (image-only):
  CLIP 0.7338 / 0.7367, frozen-Qwen 0.5992 / 0.6865. Text-stream dAUC(vis−gen): trLOO −0.0102, dev +0.0051.
- **IMG MOVED** (both footings clear the F58 thresholds, same positive sign) ⇒ **EN-HEAD-PROCEEDS**; gate exit
  code 0. This is the pre-declared mechanical branch point (train+dev only, zero test-touch), NOT a
  head-accuracy verdict.
- **DSLIST decision: `"HateMM MHC"`** (both legs; EN head rows retained).

### 6.4 Head job J5 submitted (arg-driven; NO frozen-file edit)
- First submit **13306** used `--dependency=afterok:13302:13304`; since 13302 had already COMPLETED and was
  ~4 min from the 300-s MinJobAge purge (a `DependencyNeverSatisfied` race), it was cancelled and resubmitted.
- **J5 = 13307** `enc3seed_lora_vis.sbatch "HateMM MHC"`, `--dependency=afterok:13304` (the EN vis cache is
  already on disk + verified §6.2, so the head keys only on the pending HateMM extract). Runs 6 head rows
  (HateMM seeds 0/1/2 + MHC-EN seeds 0/1/2), `--model Qwen2.5-VL-7B-Instruct-LoRA-vis_HF`,
  `--group_name RAC_video_lora_vis`. PENDING behind 13303→13304.

### 6.5 Status — PENDING-JOB (HateMM leg)
Remaining chain: **13303 (SFT-HateMM) → 13304 (extract-HateMM) → 13307 (head)**. On 13303 COMPLETE: HateMM_vis
SFT G-repro (eval_loss band; anchor 0.1084) + adapter census. On 13304 COMPLETE: HateMM vis cache sanity. On
13307 COMPLETE: transcribe RAW per-seed both-protocol numbers (line-numbered, NO gates) for the independent
verdict reviewer. No test metric read by the executor; nothing pushed.
