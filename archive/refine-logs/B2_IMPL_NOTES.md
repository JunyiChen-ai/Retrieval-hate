# B2 Implementation Notes — Stage-D/E/T runners (C1)

**Author:** B2 prep agent · **Date:** 2026-07-14 · **Status:** C0 (Rev-1..4 applied to the
prereg) + C1 (runners authored) COMPLETE; **NO submission** — awaiting reviewer delta-check
(C2) + explicit user/main go (C3), per the conditional authorization in
`refine-logs/B2_PREREG_REVIEW.md`.

- Pre-registration: `research-wiki/experiments/exp-encoder-32b-b2.md`
  (status `DRAFT-REV1-AWAITING-DELTA-CHECK`; Rev-1/2/3/4 from `refine-logs/B2_PREREG_REVIEW.md`
  applied; free dim-inference strengthening applied).
- Runners (all NEW files; no existing script touched):
  - `scripts/slurm/b2_stage_d_download.sbatch` (Stage-D, CPU download)
  - `scripts/slurm/b2_stage_e_extract.sbatch` (Stage-E, 1×A100 extraction)
  - `scripts/slurm/b2_stage_t_train.sbatch` (Stage-T, 1×A100 training, 9 runs)
- `bash -n` syntax check: **OK for all three** (2026-07-14).

---

## (a) C0 — per-revision edit map (prereg)

| rev | where applied in `exp-encoder-32b-b2.md` |
|---|---|
| Rev-1 | frontmatter `provenance`; Design "Reference arms are REUSED" (MHC-EN split into CLIP s0/1/2 + 7B s0 = 12850 vs **7B s1/s2 = arcbase 12275/12276**); MHC-EN reference-table header; Namespace-diff gate target list (12850 / 12275-12276 / 13115); Stage-T section; ceremony step 6; connections `controls-against` |
| Rev-2 | Stage-E item 1 rewritten as a **HARD pre-submit diff-verify item** with the burn risk stated verbatim (omitted `--out_model_tag` → output written to `{split}_Qwen2.5-VL-7B-Instruct_HF.pt` = silent overwrite of the existing 7B caches; collision precedes the G-dims backstop); ceremony step 2 |
| Rev-3 | frontmatter `provenance`; new Stage-E "(Rev-3, provenance precision)" bullet (P10 proved the *scorer* on 32B bf16; the *extractor* reuses the loading path, first 32B run G-repro-gated); kill rule 1 rewording; connections `uses` |
| Rev-4 | new "G-repro execution mode (Rev-4, pinned)" gate bullet: **first-config readout of the 9-run Stage-T serial job** (config #1 = HateMM s0), NO separate smoke submit, NO mid-job intervention, gate applied at verdict processing; Stage-E order note fixed (s0 readout belongs to Stage-T); Stage-T CONFIGS table marks config #1; ceremony steps 4/6 |
| free strengthening | Asset-check bullet now cites the review §4a code read: dim inferred from loaded `.pt` shape (`run_rac.py:1102-1103` → `:1117-1120` → `classifier.py:76-77`), no hard-coded 3584/5120; G-repro = cache-integrity + first-run sanity only |
| bookkeeping | status/tags/STATUS-banner → `DRAFT-REV1-AWAITING-DELTA-CHECK`; readiness list rewritten to C-numbered gates; runner names pinned to `b2_stage_{d,e,t}_*.sbatch`; revision-history row r1 added |

## (b) Stage-D — `b2_stage_d_download.sbatch` (new file; no template — authored to review C1 spec)

- **SLURM header:** `--partition=slurmpartition`, `--cpus-per-task=8`, `--mem=32G`
  (**R-1 fix, C2 delta-check 2026-07-14**: originally authored 4/16G; corrected to match
  the prereg Stage-D spec — harmless within caps, avoids amending the prereg),
  **NO `--gres`** (CPU-only), **NO `--time`** (both stated as in-file NOTE comments).
- **Env:** `HF_HUB_OFFLINE=0` (must reach the hub), `PYTHONUNBUFFERED=1`, conda `HateVideo`.
- **`HF_HUB_ENABLE_HF_TRANSFER` intentionally OMITTED** — checked 2026-07-14 in the
  HateVideo env: `python -c "import hf_transfer"` → `ModuleNotFoundError` (hub 0.29.3
  present; `huggingface-cli` at `/data/jehc223/miniconda3/envs/HateVideo/bin/huggingface-cli`).
  Setting the flag without the package makes hub downloads FAIL, so it is omitted (recorded
  as an in-file NOTE).
- **Command:** `huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct` (the `hf` alias does
  not exist in this env — verified).
- **Blob verification step:** `df -h /data` before + after (executor duty from the
  authorization); `ls -lh` of the snapshot dir; `*.safetensors` count echoed against the
  expected **18**; `du -sh` of the whole cache; per-shard `du -Lh` sizes. Count mismatch
  prints a WARNING (verification job continues so the listing is captured; the executor
  gates Stage-E on it).
- **disk_guard intentionally NOT called** in Stage-D (no template lineage requires it;
  avoids any `data/CLIP_Embedding` reclaim racing the download stage). disk_guard could not
  touch the weights anyway (`models--*` protected, `disk_guard.sh:379-380`).

## (c) Stage-E — `b2_stage_e_extract.sbatch` vs template `gen_embed_mllm.sbatch`

Diff hunks (full `diff -u` reproduced at the bottom of this section in summary form):

1. `--job-name=mllm_embed` → `b2_ext32b`; header comment block rewritten (B2 purpose,
   Rev-2/Rev-3 notes, fail-closed statement).
2. `+ export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (32B bf16 on 80G; proven by
   the P10 scorer `p10_score_ladder.sbatch:22-24`; the base template does not set it).
3. Single-`$1`-dataset invocation → `extract_one()` called serially for
   **HateMM → MHC → MHC_zh** under the template's existing `set -euo pipefail`
   (**fail-closed between datasets**: a crash on HateMM stops MHC/MHC_zh).
4. The python argv **mirrors the 7B invocation exactly**
   (`--dataset / --num_frames ${NUM_FRAMES:-8} / --device cuda`) **plus the two Rev-2 flags**
   (see (d)).
5. `+` post-extraction **dim/rowcount echo** per dataset: CPU `torch.load` of the three
   fresh `.pt`, printing `rows / img shape / text shape / labels shape` (echo only — the
   formal G-dims gate incl. paired-id audit runs at Stage-E verification, per prereg).
6. Per-dataset `b2_push` of `data/CLIP_Embedding/<ds>` retained from the template
   (the disk-lifecycle mitigation the review endorsed).
- Unchanged from template: SBATCH GPU/CPU/mem lines, NO `--time`, conda activation,
  `disk_guard.sh || true` at startup (safe: `models--*` protected), `HF_HUB_OFFLINE=1`,
  `WANDB_MODE=disabled`, `PYTHONUNBUFFERED=1`, `nvidia-smi` echo.

## (d) Rev-2 HARD diff-verify — BOTH flags present (verified by grep, line numbers)

```
scripts/slurm/b2_stage_e_extract.sbatch
  31: MODEL_ID="Qwen/Qwen2.5-VL-32B-Instruct"
  32: OUT_TAG="Qwen2.5-VL-32B-Instruct_HF"
  44:         --model "$MODEL_ID" \
  45:         --out_model_tag "$OUT_TAG"
```

**Both `--model Qwen/Qwen2.5-VL-32B-Instruct` AND `--out_model_tag
Qwen2.5-VL-32B-Instruct_HF` are passed** on the (single) extractor invocation inside
`extract_one()`. Burn risk this prevents (review Rev-2): the base template passes neither
flag and the script defaults are the 7B values
(`generate_VideoMLLM_embedding_HF.py:81,87`); omitting the tag would silently write the 32B
features to `{split}_Qwen2.5-VL-7B-Instruct_HF.pt`, overwriting the existing 7B caches
BEFORE the G-dims backstop could see anything. **Re-verify these two lines at C2 delta-check
and again immediately before the Stage-E submit.**

## (e) Stage-T — `b2_stage_t_train.sbatch` vs parent `scripts/slurm/enc3seed.sbatch`

`diff -u` produces **exactly two hunks inside one @@ block** (verified 2026-07-14):

```diff
-QWEN=Qwen2.5-VL-7B-Instruct_HF
+QWEN=Qwen2.5-VL-32B-Instruct_HF
 ...
 CONFIGS=(
-  <10 parent rows: HateMM/MHC × CLIP/QWEN mixed>
+  "HateMM $QWEN 0"   <- config #1 = the Rev-4 G-repro sanity readout
+  "HateMM $QWEN 1"
+  "HateMM $QWEN 2"
+  "MHC $QWEN 0"
+  "MHC $QWEN 1"
+  "MHC $QWEN 2"
+  "MHC_zh $QWEN 0"
+  "MHC_zh $QWEN 1"
+  "MHC_zh $QWEN 2"
 )
```

- Everything else — SBATCH headers (job-name stays `enc3seed`, B1 precedent), env setup,
  `CLIP`/`GROUP_NAME=RAC_video_archive_seeds`/`WARMUP=5` vars, the full `run_one()` python
  command (incl. `--force False`), the val-sel/final readout parser, the loop, the b2 push —
  is **byte-identical** to the parent. The `--exp_comment "_${MODEL}"` mechanism gives the
  32B arm its own `exp_comment=_Qwen2.5-VL-32B-Instruct_HF` and output dirs automatically.
- Inherited cosmetic nit (kept for byte-minimality, same as B1 which says "10 configs" while
  running 6): the header comment still reads "Runs 10 configs serially" though B2 runs 9.
  Not functional; flagged here so delta-check does not count it as an unexplained change.
- Config order = prereg Stage-T table verbatim; **config #1 = HateMM 32B s0 = the Rev-4
  G-repro first-config readout.**

## (f) Collision checks (2026-07-14)

- **Embedding-cache collision (Rev-2 backstop): NONE.**
  `find data/CLIP_Embedding -iname "*32B*"` → **0 files**. The 9 output caches
  `data/CLIP_Embedding/{HateMM,MHC,MHC_zh}/{train,dev_seen,test_seen}_Qwen2.5-VL-32B-Instruct_HF.pt`
  are all fresh names; the existing 7B/CLIP/LoRA caches use different tags.
- **HF weight cache:** `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct`
  does NOT exist (Stage-D creates it; Stage-C deletes it).
- **Training output dirs:** `logging/Retrieval/{HateMM,MHC,MHC_zh}/RAC_video_archive_seeds/`
  contain **no** `*32B*` dirs → fresh `..._Qwen2.5-VL-32B-Instruct_HF` dirs, `FORCE=False`
  safe (no `--force` collision possible).
- **Trainlogs:** `ls slurm/logs | grep -i 32B` → only `dl_qwen25vl_32b.log` (the old P10
  download log; not an `enc3s_*` trainlog). New logs
  `enc3s_<ds>_Qwen2.5-VL-32B-Instruct_HF_seed{0,1,2}_<JID>.trainlog` cannot collide; fresh
  `$SLURM_JOB_ID` double-protects. Stage-D/E stdout use new job-names (`b2_dl32b_*.out`,
  `b2_ext32b_*.out`); Stage-T stdout `enc3seed_<newJID>.out` (existing: 12850, 13115 only).

## (g) sha256 (recorded for C2 delta-check)

```
817a951d717be56e7329ccb894c2f6ffb1edeb85e656d91286a57b34bd35284a  scripts/slurm/b2_stage_d_download.sbatch   (post R-1: 8 CPU / 32G per prereg; bash -n OK)
532a8a3458f84862919d625da17b3e7e33d437b465d9bde13e93a475c5a1ff1c  scripts/slurm/b2_stage_e_extract.sbatch
9c312da639dba0ee8061b1bb3e22b4a4a074db1812e043763732e666ef04564c  scripts/slurm/b2_stage_t_train.sbatch
56588dc1b2f492e002948e9844f5059ba4bab1a156589bc67ca75b082833eb0b  research-wiki/experiments/exp-encoder-32b-b2.md   (rev r1)
```

(Superseded pre-R-1 Stage-D hash, for lineage:
`702fd5e6c48156ad178a0412074e3d079713f889158f27b91d8d64a22703e236` — 4 CPU / 16G header,
replaced at C2 per the coordinator's R-1 ruling; the only delta is the two SBATCH lines.
Stage-E / Stage-T / prereg hashes re-verified UNCHANGED after the fix. Pre-revision prereg
hash, for lineage: `d39ea5dc…77e5` as recorded in `B2_PREREG_REVIEW.md` §0.)

## (h) What is deliberately NOT done here

- NO sbatch submitted, NO download started, NO GPU used, NO weights on disk, NO test touch.
- Next gates in order: **C2** reviewer delta-check of C0+C1 (incl. re-verifying (d) and the
  Stage-D env trio `HF_HUB_OFFLINE=0` / no `--gres` / no `--time`), then **C3** explicit
  user/main go, then the staged single-submits D → E(+G-dims) → C → T.
