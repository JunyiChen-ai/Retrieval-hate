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

**Features sync + first triage probe — BLOCKED, pending USER authorization.** The
`::sync` step (uploading `data/CLIP_Embedding/HateMM/**` derived caches to the
Modal volume) was denied by the harness data-exfiltration guard: teammate/agent
authorization does not satisfy it for pushing repo data to an external cloud. To
unblock, the **user** must approve the upload (or add a Bash permission rule for
`modal run …::sync`). The video guard itself was never the blocker — it refuses
raw media; this denial is the harness refusing any repo-data egress without user
sign-off. Once authorized, the banked comparison target is ready:

| Config (enc3seed.sbatch, seed 0, final epoch) | Banked local (A100) | Modal triage |
|---|---|---|
| HateMM · Qwen2.5-VL-7B · Test_Retrieval acc | 0.8605 | pending |
| HateMM · Qwen2.5-VL-7B · Test_Retrieval macroF1 | 0.8507 | pending |

Expected cross-hardware drift ~1e-3 (triage-only; formal numbers re-run locally).
