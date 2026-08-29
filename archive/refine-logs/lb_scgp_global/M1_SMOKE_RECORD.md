# M1 Non-Lineage Smoke Record (job 13002)

Date: 2026-07-13

Author: **Claude Opus 4.8**, m1-prep role. This records the approved non-genealogy smoke
(M1_CACHE_CODE_REVIEW.md ruling ③) that empirically validates the deferred GPU path and the FIX-2 GPU
guard on a real 1-GPU allocation, before the single-submit cache runs. **Non-lineage**: HateMM
(non-contract dataset), output only to `slurm/tmp/`, zero MHC train/val/test contact, zero
`artifacts/lb_scgp_global/v1/m1/` writes, no sealed cache, no ledger, no label read.

## Job

- Job id: **13002**, name `lbscgp_global_r2_m1_smoke`, node `foscsmlprd01`.
- Terminal state: **COMPLETED**, exit `0:0`, elapsed **00:08:45**, host MaxRSS ~2.26 GB.
- Alloc: `--gres=gpu:a100:1`, 4 CPU, 32 GB, no `--time` (JobHeldUser → auto-released, not forced).
- Entities: `scripts/analysis/lb_scgp_global_r2_m1_smoke_nonlineage.py` +
  `scripts/slurm/lb_scgp_global_r2_m1_smoke_nonlineage.sbatch` (throwaway tooling, not frozen M1
  entities). The smoke imports the FROZEN `..._m1_cache_v1_common.py` (`require_slurm_cache`,
  `build_user_prompt`, `parse_certificate`, constants) and the frozen producer's `build_messages`, so
  the exact sealed code path is exercised.
- Dataset: 10 HateMM train videos (first 10 lexicographic ids: `hate_video_100, 103, 104, 106, 107,
  109, 11, 111, 112, 115`).

## Evidence collected

| check | result |
|---|---|
| **FIX-2 GPU guard under real alloc** | `require_slurm_cache()` **PASSED**; `CUDA_VISIBLE_DEVICES="0"` |
| **offline model load** | Qwen2.5-VL-7B loaded in **5.97 s** (`HF_HUB_OFFLINE=1`, 5 shards) |
| **frame decode** | `load_video_frames(…,16)` decoded **10/10** videos |
| **processor `videos=[frames]`** | accepted (PIL list, `images=None`); all `generate` calls returned |
| **R=4 greedy determinism** | **10/10 videos byte-identical** across all four replicas (`distinct_raw_outputs=1` each) |
| **strict-JSON parse rate** | **36/40 = 0.90** records parsed clean; 1 video (`hate_video_104`) had all 4 replicas fail → canonical all-unresolved (deterministically), the other 9 videos 4/4 clean |
| **GPU memory peak** | **52.71 GiB** (`56 594 189 312` bytes) — fits the A100-80GB |
| **per-video wall (R=4 calls)** | avg **50.67 s** (range 33.21–67.01 s) ≈ 12.7 s/call |

### The GPU-guard result is the load-bearing empirical finding

`CUDA_VISIBLE_DEVICES="0"` is the cgroup-remapped device index. The **old** brittle guard would have
read the global device id from `SLURM_JOB_GPUS`/`SLURM_STEP_GPUS` and string-compared it to `"1"`; on
this allocation it would have **false-failed** ("requires exactly 1 GPU, got 0") and burned the single
submit. The **FIX-2** guard (`CUDA_VISIBLE_DEVICES` non-empty) **passed**. This is the exact scenario
§4.5 warned about, now empirically settled on the real cluster.

### Determinism

Every one of the 10 videos produced four byte-identical replica outputs (greedy `do_sample=False,
num_beams=1`, `torch.manual_seed(0)`). This confirms the amendment's pinned replica semantics: the M2
`sigma_cache` gate is trivially satisfied (≈0) under true determinism; R=4 exists to bound residual FP
nondeterminism, which was **zero** here. The deterministic unresolved fallback on `hate_video_104`
(all 4 replicas identical, strict parse failed → canonical all-unresolved) demonstrates the
"invalid → canonical unresolved, no prompt/schema rescue" contract behaving correctly.

## Full-run time extrapolation

Per-video time covers all R=4 calls. Full M1 cache = 4×(549+579) = **4512** local calls over **1128**
videos (assuming no dedup, `U_D=N`), one A100:

- **Single GPU (serial):** 1128 × 50.67 s ≈ **57 157 s ≈ 15.88 h**.
- **Parallel (the plan's `m1_cache_parallel_max2`, one dataset per GPU):** MHC 549 × 50.67 s ≈ **7.73 h**;
  MHC_zh 579 × 50.67 s ≈ **8.15 h**; wall ≈ **8.15 h** (concurrent). Well within a no-`--time` job.
- vs. the amendment's **8 GPU-h/dataset** estimate: the extrapolation (7.73 / 8.15 GPU-h) **matches**
  the pinned `estimated_gpu_hours=8` per cache run almost exactly — the +16 GPU-h M1 budget is
  empirically sound. (Any real evidence-pack dedup would only reduce this.)

## Notes (benign)

- `.err` warning "`do_sample` is False. However, `temperature` is set to `1e-06`": cosmetic — the
  model's `generation_config` carries a default temperature that greedy decoding ignores; determinism
  was empirically confirmed (10/10 byte-identical). Optional producer hardening: pass `temperature=None`
  to silence it. Not a correctness issue; the frozen producer is unaffected.
- "slow image processor … `use_fast` unset": cosmetic transformers 4.49 deprecation notice.

## Isolation attestation

The smoke read only `data/gt/HateMM/train.jsonl` (id+text), `data/ASR/HateMM/…` (id+chunks), and
`data/video/HateMM/All/*.mp4`; it wrote only `slurm/tmp/m1_smoke_result.json` (captured above, then
cleaned) plus the standard `slurm/logs/…_13002.{out,err}`. No MHC/MHC_zh data, no label, no
`artifacts/lb_scgp_global/v1/m1/`, no sealed cache, no ledger. The sealed cache's
`mllm_calls_outside_train_cache` counter is unaffected (it is a per-producing-job counter emitted by
runs[4]/[5] from their own ledger; a separate job cannot make it nonzero — code-traced in
M1_CACHE_CODE_REVIEW.md §3.2).

## Verdict

All five deferred rows (GPU guard, offline load, decode, processor call, generate+parse) plus R=4
determinism are **empirically validated**. The two HIGH must-fixes are applied and the GPU guard is
confirmed correct on the real cluster. M1 is ready for the exact-hashes/no-clobber review and separate
execution authorization; this smoke record + the FIX section of `M1_CACHE_FREEZE.md` close the two
review items.
