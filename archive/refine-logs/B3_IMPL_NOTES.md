# B3 Implementation Notes — `enc3seed_zh_b3.sbatch`

**Author:** B3 prep agent · **Date:** 2026-07-14 · **Status:** implementation artifacts
ready; NO submission (awaiting fresh pre-registration review + conditional authorization).

- Pre-registration: `research-wiki/experiments/exp-lora-zh-b3.md` (status `DRAFT-UNREVIEWED`).
- Runner: `scripts/slurm/enc3seed_zh_b3.sbatch` (NEW file; parents
  `scripts/slurm/enc3seed_zh_b1.sbatch` and `scripts/slurm/enc3seed.sbatch` untouched).
- Authoritative recon: `refine-logs/B3_FORENSIC_RECON.md`.

---

## (a) Sbatch diff vs `enc3seed_zh_b1.sbatch` — 3 hunks (functional = model-tag + GROUP; header = inert)

`diff -u scripts/slurm/enc3seed_zh_b1.sbatch scripts/slurm/enc3seed_zh_b3.sbatch` (re-run
this prep; `bash -n` = SYNTAX_OK):

```diff
@@ header comment (INERT — corrects a substantively-wrong description) @@
- 3-seed paired encoder-swap test: frozen-CLIP vs frozen-Qwen, archive OFF.
- Runs 10 configs serially ...
+ B3: LoRA-Qwen encoder vs frozen-CLIP on MHC-ZH ... runs ONLY the LoRA treatment arm (3
+ head-seeds); control = B1 job 13115 (not re-run); GROUP_NAME fresh so nothing is
+ overwritten (force stays False).

@@ variable block (FUNCTIONAL: model tag + GROUP) @@
- QWEN=Qwen2.5-VL-7B-Instruct_HF
- GROUP_NAME=RAC_video_archive_seeds
+ LORA=Qwen2.5-VL-7B-Instruct-LoRA_HF
+ GROUP_NAME=RAC_video_b3_lora
  CLIP=...                     # kept as a breadcrumb to the control arm (unused; not re-run)

@@ CONFIGS (FUNCTIONAL: 6 ZH rows -> 3 LoRA rows) @@
- "MHC_zh $CLIP 0/1/2" + "MHC_zh $QWEN 0/1/2"      (6 rows: B1 CLIP+frozen-Qwen)
+ "MHC_zh $LORA 0" / "$LORA 1" / "$LORA 2"          (3 rows: B3 LoRA treatment)
```

**Everything else is byte-identical to B1** — SBATCH resource headers
(`gpu:a100:1`, 8 CPU, 64 GB, `--job-name=enc3seed`, `%x_%j.out`, no `--time`), env setup
(`HF_HUB_OFFLINE`/`WANDB_MODE`/`PYTHONUNBUFFERED`), `WARMUP=5`, the **full `run_one()`
python command**, the VALSEL/FINAL readout parser (identical regex + selection rule), the
loop, and the b2 push. So the python invocation is byte-identical to B1's (hence to
`train_archive_baseline.sbatch`'s) **except `--model Qwen2.5-VL-7B-Instruct-LoRA_HF` and
`--group_name RAC_video_b3_lora`** — matching the recon's "CONFIGS+GROUP-only" scope.

**Relation to the requested "CONFIGS+GROUP-or-FORCE only" scope.** Functionally the change
IS CONFIGS + GROUP: (i) CONFIGS now names the LoRA model tag directly (in B1 the model came
from `$CLIP`/`$QWEN` vars — I renamed `QWEN`→`LORA` so the tag reads correctly, and left
`CLIP` as an unused breadcrumb), and (ii) `GROUP_NAME` is the fresh group. FORCE is NOT
touched (stays `--force False`, the safe default — see (c)). The header-comment hunk is
comment-only (inert); I updated it rather than leave B1's "frozen-CLIP vs frozen-Qwen"
text, which would be substantively wrong for a LoRA test (B1 precedent tolerates stale
comments as immaterial, but this one mis-describes the experiment, so correcting it is the
safer choice — zero runtime effect either way).

**Immaterial remnants (flagged, left as-is):**
1. `#SBATCH --job-name=enc3seed` (not `enc3seed_zh_b3`) → stdout `enc3seed_<JID>.out`; `%j`
   guarantees no overwrite (only `enc3seed_{12850,13115}.out` exist). squeue readability is
   the only cost. Immaterial.
2. `CLIP=openai_clip-vit-large-patch14-336_HF` is defined but unused (the control arm is not
   re-run). Kept as a breadcrumb to job 13115; zero runtime effect.

## (b) Cache existence check — the 3 LoRA `.pt` (dims/rows) VERIFIED

Verified 2026-07-14 by CPU `torch.load` (HateVideo env; no GPU). All under
`/data/jehc223/RGCL/data/CLIP_Embedding/MHC_zh/`:

| file | size | mtime | img dims | text dims | rows (expect) | labels | keys |
|---|---|---|---|---|---|---|---|
| `train_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` | 16,619,871 B | Jul 2 12:08 | (579, 3584) | (579, 3584) | 579 (579) ✅ | (579,) | ids,img_feats,text_feats,labels |
| `dev_seen_..._LoRA_HF.pt` | 2,240,628 B | Jul 2 12:11 | (78, 3584) | (78, 3584) | 78 (78) ✅ | (78,) | same |
| `test_seen_..._LoRA_HF.pt` | 4,278,267 B | Jul 2 12:17 | (149, 3584) | (149, 3584) | 149 (149) ✅ | (149,) | same |

- **Single file per split, no seed suffix** — the single shared LoRA-SFT draw
  (`B3_FORENSIC_RECON.md:86-95`). The 3 B3 seeds all read these same 3 files ⇒ head-seed
  variance only (prereg §0 limitation 2).
- Row counts = ZH splits 579/78/149; dim 3584 (Qwen hidden), consistent with the arcbase
  primary logs. `ids` = single-element list wrapping the id list (`ids[0]` = the id list),
  pipeline-standard, loaded by `load_feats_MHC` (`src/data_loader/dataset.py:499,587-589`)
  for `MHC_zh`.
- LoRA-vs-CLIP test ids set- AND order-identical (`B3_FORENSIC_RECON.md:94-95`) ⇒ the B3
  LoRA arm and the 13115 CLIP control see the identical 149 test videos.
- Output trainlogs derive as
  `slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_<JID>.trainlog` — **no
  such files exist** (only `arcbase_*-LoRA_HF*` logs); no log collision (the enc3s log name
  omits `group_name`, so this holds regardless of the group).

### G-repro anchors (final-epoch, re-read from primary logs this prep)

| seed | anchor job | final-ep acc | final-ep F1 | trainlog |
|---|---|---|---|---|
| 0 | 12223 | 0.8456 | 0.8181 | `arcbase_MHC_zh_..._LoRA_HF_seed0_12223.trainlog` (ep29) |
| 1 | 12224 | 0.8389 | 0.8113 | `..._seed1_12224.trainlog` (ep29) |
| 2 | 12225 | 0.8523 | 0.8226 | `..._seed2_12225.trainlog` (ep29) |

The G-repro HARD gate (prereg §10 rule 1) requires the fresh LoRA runs to match these to
4dp. Features cached + argv byte-identical except inert `group_name`/`output_path` ⇒ exact
match is EXPECTED; any mismatch = HALT + code-drift investigation.

## (c) Collision semantics — distinct GROUP `RAC_video_b3_lora`, `FORCE=False` (NON-DESTRUCTIVE)

**Verified in `src/run_rac.py`** (`:855` local `group_name` is dead; `:899-900` builds
`output_path=.../Retrieval/MHC_zh/<group_name>/<exp_name>/`; `:901-908` collision handler):

```
:901  if not os.path.exists(output_path):  os.makedirs(output_path); os.makedirs(.../ckpt/)
:904  else:
:905      if not args.force:
:908          raise Exception("Output path already exists, aborting...")   # HARD ABORT
          # (force=True would instead proceed and OVERWRITE the existing dir in place)
```

- **The collision that forced the decision.** `exp_name` for a LoRA seed-s run is
  seed+model-derived and **byte-identical to the existing arcbase dir**
  `RAC_..._seed{s}_hybrid_loss_Qwen2.5-VL-7B-Instruct-LoRA_HF` under
  `RAC_video_archive_seeds/MHC_zh/` (seeds 0-4 exist, 2026-07-04, verified `ls`). Under the
  arcbase group with `force=False` → **hard abort on all 3 seeds**; with `force=True` →
  **overwrites the arcbase 12223-25 output artifacts** (ckpt/metrics).
- **Decision = fresh GROUP `RAC_video_b3_lora`, `force=False`.** Three reasons:
  1. **Non-destructive:** preserves the arcbase 12223-25 artifacts — the very G-repro
     anchors — untouched. `FORCE=True` would destroy them.
  2. **Cleaner Namespace:** `force` stays `False`, matching BOTH the 13115 CLIP control
     (`force=False`, verified in its Namespace) and the arcbase anchors (`force=False`,
     verified). So the only Namespace deltas vs 13115 are `{model, exp_comment, group_name,
     output_path}` (all inert); `FORCE=True` would add a `force` divergence.
  3. **`group_name` is computationally inert** — feeds ONLY `output_path` at `:900`
     (`:855` local is unused). It cannot change any result ⇒ the G-repro reproduction
     expectation is unaffected by the group swap.
- **No-collision guarantee (explicit):** `logging/Retrieval/MHC_zh/RAC_video_b3_lora*` **does
  not exist** (verified `ls` → none) ⇒ `:901-902` creates fresh dirs; `force=False` never
  trips `:908`; **nothing anywhere is overwritten.** New output dirs will be
  `RAC_video_b3_lora/MHC_zh/RAC_..._seed{0,1,2}_hybrid_loss_Qwen2.5-VL-7B-Instruct-LoRA_HF/`.
- **Namespace-diff gate note (prereg §10 rule 2):** `group_name` joins the parent's
  benign-field whitelist (`model`/`exp_comment`/`output_path`) because it is inert — a
  `group_name` difference between the B3 LoRA arm and the 13115 CLIP arm is NOT a substantive
  divergence. `exp_comment` = `_${MODEL}` = `_Qwen2.5-VL-7B-Instruct-LoRA_HF`, matching the
  arcbase anchors' `exp_comment` exactly (so vs 12223-25 the ONLY delta is `group_name`).

## (d) Deferred-import / new-code note

**n/a — reused pipeline.** B3 introduces zero new Python: no new imports, modules, or code
edits. The runner invokes existing `src/run_rac.py` via the exact B1 python command;
`MHC_zh` + `-LoRA_HF` caches load through the long-exercised `load_feats_MHC` path. The only
new artifact is the sbatch CONFIGS/GROUP data (shell, not code). Nothing to audit for
deferred/lazy imports.

## (e) Runtime estimate

- Extraction: 0 GPU-s (3 caches exist, §b). Training: 3 runs × ~20-25 s cached (recon §6;
  job-12850 corroboration ~20-52 s). Total ≈ 1-2 min compute; < ~10 min wall incl.
  conda/disk_guard/parse/b2-push. 1× A100 / 8 CPU / 64 GB (within 16 CPU / 128 GB / 2 GPU
  cap). No `--time`; expect initial `PENDING (JobHeldUser)`, wait for auto-release, never
  force.

## (f) sha256 (this prep, `sha256sum`)

```
71745cf29de7f03a2bd4d351b30b02637a8d250f493dfb7f49d3459c44f7d802  research-wiki/experiments/exp-lora-zh-b3.md
4379224671defe7dafb638c4f0c8b69295a27d11646b685912a249e2385e29ad  scripts/slurm/enc3seed_zh_b3.sbatch
```

Re-hash both at submit time and pin in the execution record (B1 authorization precedent).

## Submission status

**NOT SUBMITTED.** Next steps per prereg §14: fresh pre-registration review → implementation
delta-check → (optional smoke) → conditional authorization → single
`sbatch scripts/slurm/enc3seed_zh_b3.sbatch`.
