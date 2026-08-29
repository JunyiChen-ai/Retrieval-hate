# B1 Implementation Notes — `enc3seed_zh_b1.sbatch`

**Author:** B1 prep agent · **Date:** 2026-07-14 · **Status:** implementation artifacts
ready; NO submission (awaiting reviewer delta-check of Rev-1/2/3 + conditional
authorization).

- Pre-registration: `research-wiki/experiments/exp-encoder-zh-b1.md`
  (status `DRAFT-REV1-AWAITING-DELTA-CHECK`; Rev-1/2/3 from
  `refine-logs/B1_PREREG_REVIEW.md` applied).
- Runner: `scripts/slurm/enc3seed_zh_b1.sbatch` (NEW file; parent
  `scripts/slurm/enc3seed.sbatch` untouched).

---

## (a) Runtime cross-check table

### a.1 Input caches — all 6 exist; sizes, dims, row counts, id-pairing VERIFIED

Verified 2026-07-14 by CPU `torch.load` (HateVideo env python; no GPU). All under
`/data/jehc223/RGCL/data/CLIP_Embedding/MHC_zh/`:

| file | size | mtime | img dims | text dims | rows (expect) | labels |
|---|---|---|---|---|---|---|
| `train_openai_clip-vit-large-patch14-336_HF.pt` | 4,169,097 B | Jul 1 18:16 | (579, 1024) | (579, 768) | 579 (579) ✅ | (579,) |
| `dev_seen_openai_clip-vit-large-patch14-336_HF.pt` | 563,358 B | Jul 1 18:17 | (78, 1024) | (78, 768) | 78 (78) ✅ | (78,) |
| `test_seen_openai_clip-vit-large-patch14-336_HF.pt` | 1,074,213 B | Jul 1 18:19 | (149, 1024) | (149, 768) | 149 (149) ✅ | (149,) |
| `train_Qwen2.5-VL-7B-Instruct_HF.pt` | 16,619,836 B | Jul 2 01:37 | (579, 3584) | (579, 3584) | 579 (579) ✅ | (579,) |
| `dev_seen_Qwen2.5-VL-7B-Instruct_HF.pt` | 2,240,593 B | Jul 2 01:40 | (78, 3584) | (78, 3584) | 78 (78) ✅ | (78,) |
| `test_seen_Qwen2.5-VL-7B-Instruct_HF.pt` | 4,278,232 B | Jul 2 01:46 | (149, 3584) | (149, 3584) | 149 (149) ✅ | (149,) |

- Row counts match the ZH splits exactly (`data/gt/MHC_zh/{train,val,test}.jsonl` =
  579/78/149; `EXPECTED_TRAIN_N["MHC_zh"]=579`,
  `scripts/analysis/lb_scgp_global_r2_m1_cache_v1_common.py:46`).
- Dims match the primary-log record: Qwen ZH 3584/3584
  (`rgcl_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_1151518.trainlog:2-3`), CLIP 1024 img / 768
  text (`mhc_train_seg_12130` lineage; same as EN CLIP).
- Keys per file: `{ids, img_feats, text_feats, labels}` — exactly the schema
  `src/data_loader/dataset.py:501` documents and `load_feats_MHC`
  (`dataset.py:587-589`) loads for `MHC_zh` (`dataset.py:499`).
- `ids` field is a single-element list wrapping the id list (`ids[0]` = 579 ids).
  **Same structure in the EN caches the parent test consumed successfully** (checked
  `MHC/train_{CLIP,Qwen}` side-by-side) → pipeline-standard format, not corruption.
- **Paired-arm id audit:** CLIP-vs-Qwen id lists are identical (same set AND same
  order) on all three splits — the two arms see the exact same videos.
- **IN-1 (review):** model tag in the runner is `QWEN=Qwen2.5-VL-7B-Instruct_HF`
  (**frozen**, inherited verbatim from the parent) — NOT `-LoRA_HF`. The `-LoRA_HF`
  caches also exist in the same dir but are never referenced.

### a.2 Output log naming — no collisions

Runner derives per-run logs as
`slurm/logs/enc3s_${DATASET}_${MODEL}_seed${SEED}_${SLURM_JOB_ID}.trainlog`, i.e.:

```
enc3s_MHC_zh_openai_clip-vit-large-patch14-336_HF_seed{0,1,2}_<JID>.trainlog
enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF_seed{0,1,2}_<JID>.trainlog
```

- **Verified 2026-07-14: NO `slurm/logs/enc3s_MHC_zh*` files exist** (`ls` returns
  no-such-file). Existing enc3s logs are all `enc3s_MHC_*` (EN) / `enc3s_HateMM_*`
  from job 12850; `MHC_zh` vs `MHC` prefixes cannot collide, and the fresh
  `$SLURM_JOB_ID` suffix double-protects.
- Sbatch stdout = `slurm/logs/enc3seed_<JID>.out` (`%x_%j`, job-name inherited as
  `enc3seed`); only existing file is `enc3seed_12850.out` — new JID, no overwrite.
- **IN-2 (review, `FORCE=False` output-dir collision):** verified
  `logging/Retrieval/MHC_zh/RAC_video_archive_seeds/` contains **only**
  `*_Qwen2.5-VL-7B-Instruct-LoRA_HF*` dirs (seeds 0-4, ± `arc-knn-a0.25`). The B1 arms
  create fresh `..._hybrid_loss_openai_clip-vit-large-patch14-336_HF` /
  `..._hybrid_loss_Qwen2.5-VL-7B-Instruct_HF` dirs — no collision. **Re-check at
  submit time** in case a smoke run creates a partial dir first (per review IN-2).

### a.3 Run matrix (matches prereg table verbatim)

| # | dataset | model | seed | role |
|---|---|---|---|---|
| 1 | MHC_zh | CLIP | 0 | control + cross-runner confirmatory check vs 12130 (kill rule 1b) |
| 2 | MHC_zh | CLIP | 1 | control |
| 3 | MHC_zh | CLIP | 2 | control |
| 4 | MHC_zh | Qwen | 0 | treatment + HARD reproduction gate vs 1151518 (kill rule 1a) |
| 5 | MHC_zh | Qwen | 1 | treatment |
| 6 | MHC_zh | Qwen | 2 | treatment |

Archive OFF by construction: the runner passes no `--archive_feats` →
`archive_feats=None` (run_rac gate), `--lambda_seg 0`, `GROUP=RAC_video_archive_seeds`,
`--force False` — all inherited verbatim from the parent command block.

---

## (b) Diff vs parent — CONFIGS-only (verified)

`diff -u scripts/slurm/enc3seed.sbatch scripts/slurm/enc3seed_zh_b1.sbatch` produces a
**single hunk** replacing the 10-row EN/HateMM `CONFIGS` array with the 6-row ZH array:

```diff
@@ -29,16 +29,12 @@
 # config list: "DATASET MODEL SEED"
 CONFIGS=(
-  "HateMM $CLIP 1"
-  "HateMM $CLIP 2"
-  "HateMM $QWEN 1"
-  "HateMM $QWEN 2"
-  "MHC $CLIP 1"
-  "MHC $CLIP 2"
-  "HateMM $CLIP 0"
-  "HateMM $QWEN 0"
-  "MHC $CLIP 0"
-  "MHC $QWEN 0"
+  "MHC_zh $CLIP 0"
+  "MHC_zh $CLIP 1"
+  "MHC_zh $CLIP 2"
+  "MHC_zh $QWEN 0"
+  "MHC_zh $QWEN 1"
+  "MHC_zh $QWEN 2"
 )
```

- Everything else — SBATCH headers, env setup, `CLIP`/`QWEN`/`GROUP_NAME`/`WARMUP`
  vars, the full `run_one()` python command, the val-sel/final-epoch readout parser,
  the loop, and the b2 push — is **byte-identical** to the parent. `bash -n` syntax
  check: OK.
- Therefore the python command is byte-identical to the parent test's (and hence to
  `train_archive_baseline.sbatch`'s) except `--dataset MHC_zh` — satisfying the
  prereg's config-match requirement and review checklist row "Config-edit scope".
- **Two cosmetic verbatim-copy remnants, kept deliberately** (the instruction was
  "change ONLY the CONFIGS block"); flagged for the delta-checker, either may be
  authorized as a comment-only touch-up or left as-is:
  1. Header comment line "Runs 10 configs serially" (now 6).
  2. `#SBATCH --job-name=enc3seed` unchanged → stdout `enc3seed_<JID>.out` (no
     functional collision; a distinct name like `enc3seed_zh_b1` would only aid
     `squeue` readability).

## (c) Deferred-import note

**n/a — reused pipeline.** B1 introduces **zero new Python code**: no new imports, no
new modules, no code edits. The runner invokes the existing `src/run_rac.py` through
the exact parent command; `MHC_zh` is an already-supported dataset branch
(`src/data_loader/dataset.py:499`), and both feature families load through the
long-exercised `load_feats_MHC` path. The only new artifact is the sbatch CONFIGS
array (shell data, not code). Nothing to audit for deferred/lazy imports.

## (d) Runtime estimate

- **Extraction: 0 GPU-s** — all 6 caches exist (§a.1).
- **Per-run training:** parent job 12850 tqdm end-bars with cached frozen features:
  MHC-EN CLIP 20-36 s, MHC-EN Qwen 33 s, HateMM 26-52 s, one I/O-contended outlier
  2:55. ZH train n=579 ≈ EN 549 (+5%), Qwen dim identical (3584) → same band,
  **~20-60 s/run**.
- **Total: 6 runs serial ≈ 2-6 min compute; < ~20 min wall** including conda
  activation, disk_guard, per-run readout parsing, and b2 push. 1× A100 / 8 CPU /
  64 GB (inherited parent headers; within the 16 CPU / 128 GB / 2 GPU user cap).
- No `--time` set (project rule). Expect initial `PENDING (JobHeldUser)`; wait for
  auto-release, never force.

---

## Submission status

**NOT SUBMITTED.** Next steps per prereg readiness list: reviewer delta-check of
Rev-1/2/3 + this impl package → (optional smoke) → conditional authorization → single
`sbatch scripts/slurm/enc3seed_zh_b1.sbatch`.
