# M1 v2 Non-Lineage Real-Path Smoke Record (job 13009)

Date: 2026-07-13

Author: **Claude Opus 4.8**, m1-prep role. Records the approved non-genealogy real-path smoke
(M1_CACHE_V1_RESULT_TO_CLAIM_REVIEW.md §4.4 / §5.1) that empirically settles the v2 symlink-tolerant
guard by calling the **actual frozen** `build_dataset_packs` on real symlinked mp4 — closing the v1-smoke
gap (which re-implemented the loop and bypassed the guard). **Non-lineage**: HateMM (non-contract),
output only to `slurm/tmp/`, zero MHC contact, zero `artifacts/lb_scgp_global/v1/m1/` writes, no label.

## Job

- Job id **13009**, name `m1_smoke2_realpath`, node `foscsmlprd01`; **COMPLETED**, exit `0:0`, elapsed
  **00:04:38**; `--gres=gpu:a100:1`, 4 CPU, 32 GB, no `--time`.
- Entities (throwaway tooling, not frozen M1 entities): `scripts/analysis/lb_scgp_global_r2_m1_smoke2_realpath.py`
  + `scripts/slurm/lb_scgp_global_r2_m1_smoke2_realpath.sbatch`. The smoke imports and calls the **frozen
  v2** `build_dataset_packs`, `canonical_video_path`, `TrainEvidenceAccessLedger`, `parse_certificate`,
  `cert_v2_object`, `require_slurm_cache`, and the producer's `build_messages` — no re-implementation of
  the guard. HateMM was registered in the in-memory `EXPECTED_TRAIN_N` so the frozen (unmodified)
  `build_dataset_packs` accepts the non-contract dataset; every line of its logic ran as frozen.

## Evidence

| check | result |
|---|---|
| **FROZEN `build_dataset_packs` on real symlinked mp4 (the v1 burn surface)** | **completed, no raise**, 4.68 s over **744/744** HateMM videos; unique_pack_count 744; missing 0 |
| **symlink audit (frozen ledger)** | **744/744** train-video reads are symlinks; **744/744** followed-targets escape the repo (e.g. `/data/jehc223/HateMM/video/hate_video_100.mp4`); **all forbidden zero-counters = 0** |
| **FIX-2 GPU guard under real alloc** | `require_slurm_cache()` PASSED; `CUDA_VISIBLE_DEVICES="0"` |
| **offline model load** | 5.94 s |
| **16-frame decode via followed symlink** | 5/5 videos decoded, **16 frames each** (`sixteen_frames_each=true`) |
| **certificate production (frozen path)** | R=4 per video; **all 20 records validate against the Run1-frozen `scgp_global_cert_v2` schema** |
| **R=4 determinism** | **5/5 videos byte-identical** across replicas (`distinct_raw_outputs=1`) |
| **strict-JSON parse rate** | **16/20 = 0.80** (4 videos 4/4 clean; `hate_video_104` 0/4 → deterministic canonical unresolved) |
| **GPU memory peak** | 52.71 GiB (A100-80GB) |

### The load-bearing finding

The v1 double-burn was `build_dataset_packs → canonical_root_path(video_rel)` raising
`path escapes repository root` on the first symlinked mp4. This smoke ran the **same frozen function**
(now calling `canonical_video_path`) on **all 744** HateMM mp4s — every one a symlink whose target
escapes the repo — and it **completed cleanly in 4.68 s**, recording each external `followed_target` for
audit while keeping every forbidden counter at 0. This is the direct, empirical proof that the v2 guard
resolves the burn on the exact code path, on a dataset whose input topology is identical to the contract
datasets (mp4 symlinks escaping to an external corpus).

Determinism is also cross-run reproducible: `hate_video_100`'s replica-0 output hash prefix
`d5b7021524e5e48c` is **identical to the v1 smoke (job 13002)** — same greedy output, different job.

## Isolation attestation

Read only `data/gt/HateMM/train.jsonl` (id+text), `data/ASR/HateMM/…` (id+chunks), and the 744
`data/video/HateMM/All/*.mp4` symlinks (bytes via the OS-followed target, for the retained
`video_sha256`); wrote only `slurm/tmp/m1_smoke2_result.json` (captured above, then cleaned) plus
`slurm/logs/…_13009.{out,err}`. No MHC/MHC_zh data, no label, no `artifacts/lb_scgp_global/v1/m1/`, no
sealed cache, no ledger artifact. The sealed cache's `mllm_calls_outside_train_cache` counter is a
per-producing-job counter and is unaffected by this separate job (M1_CACHE_CODE_REVIEW.md §3.2).

## Notes (benign, unchanged from v1 smoke)

- `.err` "`do_sample` False but `temperature`=1e-06": cosmetic; greedy ignores it; determinism was
  empirically zero-jitter (5/5 byte-identical). Optional producer hardening: pass `temperature=None`.
- "slow image processor … `use_fast` unset": cosmetic transformers 4.49 notice.

## Verdict

The v2 symlink-tolerant guard is **empirically validated on the exact frozen burn surface**: the frozen
`build_dataset_packs` processes 744 real repo-escaping symlinked mp4 without raising, forbidden counters
stay 0, 16-frame decode + R=4 cert production work end-to-end, and determinism holds. Combined with the
FIX-2 GPU guard and the mandatory readlink-topology row in `M1_CACHE_V2_FREEZE.md §5.1`, the v2 block is
ready for the exact-hashes/no-clobber review and the six-step gate's step 5–6 (one re-submit each for
MHC-v2 / MHC_zh-v2, then seal) — which are not m1-prep's role.
