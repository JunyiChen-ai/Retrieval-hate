# SAV (C2) F-G0 SMOKE — execution record

**Executor:** SAV smoke executor (separate authorized step).
**Authority:** `refine-logs/SAV_F0_CODE_REVIEW.md` §"SMOKE PRESCRIPTION" (APPROVED-to-smoke).
**Dates:** submitted 2026-07-13, completed + verified 2026-07-14 (NZST).

## VERDICT: **SMOKE_PASS** (all four criteria met; evidence below)

- **Job id:** `13058` (`sav_f0_smoke`), submitted ONCE at `2026-07-13T23:00:51` NZST.
- Held ~3.5 h as `PENDING (JobHeldUser)` (auto-released per policy; never forced, never resubmitted).
- **sacct terminal line** (`sacct -j 13058 -X --format=JobID,JobName%16,State,ExitCode,Elapsed,Start,End`):
  ```
  13058            sav_f0_smoke  COMPLETED      0:0   00:00:40 2026-07-14T02:30:32 2026-07-14T02:31:12
  ```
  (40 s wall: checkpoint shards loaded at ~6 it/s from warm page cache per the `.err` log;
  24 short forwards on an A100-SXM4-80GB.)
- Logs: `slurm/logs/sav_f0_smoke_13058.out` / `.err`.

## Criterion 1 — stdout geometry OK + 28 hooks: **PASS**

`slurm/logs/sav_f0_smoke_13058.out:9`:
```
[extract] registered 28 o_proj hooks; head geometry 28x28x128 = 784 head positions
```
The geometry cross-check is a fail-closed assert before this line (`sav_f0_extract.py:407-421`);
execution continuing past it to `[extract] ALL DONE complete=True` (`.out:17`) = geometry OK.
Note: decord failed on `MHC/All/ZZOSLjm0LqE.mp4` (`.out:13`) and the verbatim-mirrored PyAV
fallback succeeded (manifest `zero_guard=0`, payload `ok=true`) — the sampler fallback path was
exercised live and behaved exactly like the banked extractor.

## Criterion 2 — 12 .pt caches + 6 manifests: **PASS**

Verifier: scratchpad `verify_smoke.py` (CPU; loads every `.pt`, checks key-set, shapes,
norms; full JSON report archived in the executor scratchpad as `verify_smoke_report.json`).

- **12/12** `.pt` under `artifacts/sav_f0/extract/<ds>/{train,val}/` (2×2×3), each with the exact
  key set `{id,label,ok,img_pooled,text_pooled,img_hidden_final,img_head_final,img_head_spanmean,meta}`
  and shapes `img_pooled/text_pooled/img_hidden_final=[3584]`,
  `img_head_final/img_head_spanmean=[28,28,128]` → `keys_ok=true, shapes_ok=true` for all 12.
- `ok=true` for all 12 (no zero-guard payloads; the 12 smoke ids exclude `hate_video_95`).
- `img_pooled` unit-norm for all 12: fp32 norms ∈ {1.0, 1.00000012} (max |‖·‖−1| = 1.2e-7).
  (`text_pooled` also unit: min 0.99999994.)
- **6/6** `_manifest.json`: `complete=true, n=2, n_expected=2, n_zero_guard=0, limit=2` for
  HateMM/{train,val}, MHC/{train,val}, MHC_zh/{train,val}.

## Criterion 3 — PRIMARY-guard preview cosines ≥ 0.999: **PASS (12/12, not just 1/dataset)**

Fresh `img_pooled`/`text_pooled` vs the banked enc3s caches
`data/CLIP_Embedding/<ds>/{train,dev_seen}_Qwen2.5-VL-7B-Instruct_HF.pt`, loaded via
`sav_f0_common.load_cached_pooled` — the same code path `sav_f0_guard.py` uses. Cosines in
float64 (fp32 storage ⇒ |1−cos| ≲ 6.6e-8 is storage-precision-level):

| dataset | split | id | cos_img | cos_text |
|---|---|---|---|---|
| HateMM | train | hate_video_98 | 0.999999966354 | 1.000000003939 |
| HateMM | train | non_hate_video_80 | 1.000000023128 | 1.000000025382 |
| HateMM | val | non_hate_video_190 | 0.999999998093 | 1.000000017961 |
| HateMM | val | non_hate_video_58 | 0.999999934089 | 1.000000039384 |
| MHC | train | 4V0KGql_fUI | 0.999999939534 | 1.000000023838 |
| MHC | train | 5snzFreG79c | 1.000000020856 | 0.999999986208 |
| MHC | val | ZZOSLjm0LqE | 1.000000040013 | 1.000000003660 |
| MHC | val | fuUtL4mbTDU | 1.000000041670 | 1.000000043137 |
| MHC_zh | train | BV1em4y1B7bQ | 1.000000001362 | 0.999999998787 |
| MHC_zh | train | BV1f8411b7Xz | 0.999999938300 | 1.000000014036 |
| MHC_zh | val | BV1cp421o7gr | 1.000000025812 | 0.999999995797 |
| MHC_zh | val | BV1qV411w7t1 | 1.000000028584 | 1.000000026960 |

Minima: `cos_img = 0.999999934` (HateMM val non_hate_video_58), `cos_text = 0.999999986`
(MHC train 5snzFreG79c) — both ≫ 0.999. Per-dataset representative (first passing id):
HateMM `hate_video_98` (0.999999966 / 1.000000004), MHC `4V0KGql_fUI` (0.999999940 /
1.000000024), MHC_zh `BV1em4y1B7bQ` (1.000000001 / 0.999999999). **No pipeline drift** —
incl. on `ZZOSLjm0LqE`, the video that took the PyAV fallback path (cos_img 1.000000040).

## Criterion 4 — img_head_final finite + non-trivial per-head norm spread: **PASS**

One decodable video per split (6 sampled), `img_head_final` reshaped `[784,128]`, L2 norm/head:

| id | finite | norm min | max | mean | std |
|---|---|---|---|---|---|
| hate_video_98 | true | 0.222744 | 20.769737 | 2.796454 | 2.578739 |
| non_hate_video_190 | true | 0.217782 | 19.508215 | 2.871583 | 2.612372 |
| 4V0KGql_fUI | true | 0.194066 | 19.980646 | 2.780922 | 2.582417 |
| ZZOSLjm0LqE | true | 0.217109 | 20.132824 | 2.791739 | 2.604729 |
| BV1em4y1B7bQ | true | 0.234587 | 20.735081 | 2.667188 | 2.503687 |
| BV1cp421o7gr | true | 0.196373 | 21.050903 | 2.706274 | 2.476644 |

All finite (no NaN/Inf); per-head norm spread spans ~2 orders of magnitude (std ≈ 2.5 on
mean ≈ 2.8) — clearly non-trivial head differentiation.

## Submission provenance

- Command: `sbatch scripts/slurm/sav_f0_smoke.sbatch` → `Submitted batch job 13058`.
- Pre-flight (verified live before submission): all 6 banked pooled caches present
  (`data/CLIP_Embedding/<ds>/{train,dev_seen}_Qwen2.5-VL-7B-Instruct_HF.pt`; HateMM 21M/3.0M,
  MHC 16M/2.2M, MHC_zh 16M/2.2M); `artifacts/sav_f0/extract/` absent (clean start); the 12
  smoke ids (first 2 per split) exclude the undecodable `hate_video_95`.
- **Output-location note (CLI checked before writing the sbatch):** `sav_f0_extract.py`
  accepts only `--datasets/--splits/--device/--limit` — no RUN_ID / output-redirect flag —
  so the smoke wrote into the SAME cache dir the full run uses. Confirmed resume-compatible:
  per-video skip-if-exists (`sav_f0_extract.py:305-309`) + atomic same-dir writes
  (`sav_f0_common.atomic_torch_save`); the full run warm-starts over the 12 smoke caches and
  REWRITES each split `_manifest.json` at full count (`sav_f0_extract.py:362-382`), so the
  smoke's `n=2` manifests are transient and non-blocking (also independently confirmed in
  `SAV_F0_EXECUTION_AUTHORIZATION.md` §3).

## sbatch content (`scripts/slurm/sav_f0_smoke.sbatch`, new file)

```bash
#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=sav_f0_smoke
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# NOTE: intentionally NO --time (project policy). 1 GPU (frozen forward only) + 8 CPU.
#
# SAV (C2) F-G0 SMOKE — exercises the REAL GPU entry point on a tiny real subset per the
# smoke prescription in refine-logs/SAV_F0_CODE_REVIEW.md. This is EXTRACTION ONLY:
#   python scripts/analysis/sav_f0_extract.py --datasets HateMM,MHC,MHC_zh --splits train,val --limit 2
# The guard/probe are NOT smoked (they assert the full EXPECTED_COUNTS by design).
#
# Output location: the extractor CLI supports only --datasets/--splits/--device/--limit; it has
# NO RUN_ID or output-redirect flag, so the 2-per-split caches are written to the SAME dir the
# full run uses (artifacts/sav_f0/extract/<ds>/<split>/<id>.pt). This is warm-start compatible:
# the extractor is per-video resumable (skip-if-exists) with atomic same-dir writes
# (SAV_F0_IMPL_NOTES.md "per-video cache (skip-if-exists resume)"; "Fully resumable"). The full
# run will skip these pre-existing per-video caches and REWRITE the split _manifest.json with the
# complete full-count manifest, so the smoke's n=2 manifest is transient and non-blocking.
# HF offline env matches scripts/slurm/sav_f0.sbatch.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "########## [SAV F-G0 SMOKE] extraction (datasets=HateMM,MHC,MHC_zh limit=2) ##########"
python scripts/analysis/sav_f0_extract.py --datasets HateMM,MHC,MHC_zh --splits train,val --limit 2

echo "======== sav_f0 SMOKE extraction DONE (${SLURM_JOB_ID:-nojob}) ========"
```

## Consequence

Per `refine-logs/SAV_F0_EXECUTION_AUTHORIZATION.md` §4, SMOKE_PASS + a matching re-hash of the
seven frozen files makes the ONE full-chain submission (`sbatch scripts/slurm/sav_f0.sbatch`)
effective. The re-hash and submission are recorded in `refine-logs/SAV_F0_EXECUTION_RECORD.md`.
