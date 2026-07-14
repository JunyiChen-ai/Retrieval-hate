# Modal features-only offload (RGCL)

**Scope (binding).** ONLY derived float feature caches (`data/CLIP_Embedding/<ds>/*.pt`)
and label files (`data/gt/<ds>/*.jsonl|json|csv`) may leave the cluster. **Raw videos
never do** — `assert_uploadable` in `modal_probe_runner.py` fails loud on any media
extension or `video/` path before uploading a byte. Cloud numbers are **exploratory
triage only**; any pre-registered / paper number is re-run locally (G-repro rule).

**Commands**
1. User, once: `python3 -m modal setup` (browser auth → writes `~/.modal.toml`).
2. Validate token + billing + CPU/T4: `modal run scripts/cloud/modal_smoke.py`.
3. Push a dataset's caches + labels: `modal run scripts/cloud/modal_probe_runner.py::sync --dataset HateMM`.
4. Run a probe (CPU default; add `--gpu` for T4):
   `modal run scripts/cloud/modal_probe_runner.py::run --script src/run_rac.py --args "--path /root/data --dataset HateMM --model Qwen2.5-VL-7B-Instruct_HF --seed 0"`.

**Cost.** $30/mo free credits cover our probing load (~free). T4 ≈ $0.000164/s
(~$0.59/hr); a 25-s head-train ≈ $0.004. CPU worker is cheaper still.

**Login-node client deps (persist!).** Reaching `api.modal.com` from the cluster
needs `python-socks[asyncio]` (squid CONNECT proxy); without it modal fails with a
misleading "Could not connect". Pinned in `scripts/cloud/requirements-cloud.txt`
(there is no repo-level `environment.yml`/`requirements.txt` to hold it). Install:
`conda run -n HateVideo pip install -r scripts/cloud/requirements-cloud.txt`. Root
cause: `refine-logs/MODAL_CONNECTIVITY_DEBUG_2026-07-14.md`.

## Validated (2026-07-14)

**Smoke — PASS (no data uploaded).** `modal run scripts/cloud/modal_smoke.py`,
profile `jehc223`. Image built in 58.5 s. CPU worker: python 3.11.12, torch
2.6.0+cu124, 8192² matmul 11.29 s (97.4 GFLOPS), no GPU. T4 worker: `Tesla T4,
driver 580.95.05, 15360 MiB`, matmul 0.246 s (4463 GFLOPS). Token + billing live;
log streaming survived the squid tunnel with no mid-run disconnect. Cost ≈ cents.

**Features sync + first triage probe — PASS (end-to-end, 2026-07-15).** `::sync
--dataset HateMM` uploaded **52 files / 361.9 MB** (the full HateMM feature dir +
labels; the video-guard passed every file and refused none) to volume
`rgcl-features`. The first triage probe — HateMM · Qwen2.5-VL-7B · seed 0, the
exact `enc3seed.sbatch` args (only `--path`→`/root/data` and
`--group_name`→`RAC_modal_smoke` changed), T4, WANDB disabled — ran 30/30 epochs
in ~33 s. Cloud vs banked local
(`slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct_HF_seed0_12850.trainlog`, both
final epoch 29):

| HateMM · Qwen-7B · seed 0, final epoch (29) | Banked local (A100) | Modal triage (T4) | drift |
|---|---|---|---|
| Test_Retrieval acc     | 0.8605 | 0.8744 | +0.0139 |
| Test_Retrieval macroF1 | 0.8507 | 0.8666 | +0.0159 |
| Test_Retrieval roc     | 0.9283 | 0.9311 | +0.0028 |

Drift is ~1.4 pp acc — **not** float-eps, and that is expected: this is a 30-epoch
head-*train*, not inference, so A100-vs-T4 cuBLAS/cuDNN accumulation nudges the
optimizer onto a slightly different trajectory (≈3 test-video flips; HateMM's test
split is ~215 videos, acc quantum ≈ 1/215). The per-epoch trajectories otherwise
overlap tightly: the banked final (acc 0.8605 / mF1 0.8514) recurs at cloud epochs
13 and 20, and the cloud final (0.8744 / 0.8666) also appears at epoch 25. This is
exactly the **triage-only** regime — close enough to rank/screen, not
bit-reproducible; every paper number is re-run locally on its table's hardware
(G-repro rule).

**MHC + MHC_zh synced (2026-07-15).** `::sync --dataset MHC` uploaded **73 files /
361.0 MB** (68 feature `.pt` incl. the 3-file `v2/` archive-key subdir + 5 labels:
`train/val/test.jsonl` and `target_pred_qwen7b.json{,l}`); `::sync --dataset
MHC_zh` uploaded **83 files / 385.6 MB** (80 feature `.pt` incl. `v2/` + 3 labels:
`train/val/test.jsonl`). The video-guard passed every file and refused none (all
`.pt`/`.jsonl`/`.json`). Volume `rgcl-features` now holds **HateMM + MHC + MHC_zh**;
per-dataset volume `ls` counts match local (`CLIP_Embedding/MHC` 65 top-level +
`v2/`×3 = 68; `CLIP_Embedding/MHC_zh` 77 + `v2/`×3 = 80). ZH/EN cloud probes are now
ready.

**Two client deps are required (both now pinned in `requirements-cloud.txt`):**
`python-socks[asyncio]` (gRPC control plane) **and** `aiohttp-socks` — Modal's
volume/blob batch-upload runs over a *separate* aiohttp HTTP transport, so without
the second dep `::sync` aborts with an ImportError before moving a byte even
though `modal app list`/smoke (gRPC) work.

**Two runner bugs were fixed in `modal_probe_runner.py`** (the mounted-code `run`
path, never exercised live before): (1) `REPO_ROOT = Path(__file__).resolve().parents[2]`
IndexError-crashed the container on import — Modal mounts the entrypoint at
`/root/<name>` (depth 1), not `scripts/cloud/`; `REPO_ROOT` is only needed locally
(sync + image build), so it now falls back to the file's dir inside the container.
(2) The pinned image lacked `easydict / pandas / pillow / rank-bm25 / torchmetrics
/ wandb` (the `run_rac.py` import chain); added, pinned to the HateVideo env.

**Tunnel stability + cost.** No mid-run disconnect on the 361.9 MB sync or the
probe stream; the earlier apparent "stall" was the crash-looping container (the
local client hung on repeated container-start failures), diagnosed via `modal app
logs`, not a squid drop. All T4 attempts summed to a few minutes ≈ **$0.01–0.03**,
covered by free credits. Cloud stays **triage-only, forever.**
