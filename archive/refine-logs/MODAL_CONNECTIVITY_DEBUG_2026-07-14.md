# Modal Connectivity Debug — 2026-07-14

**Status: RESOLVED.** Modal CLI is authenticated and functional from the cluster.

## Root cause (one-liner)
Not a firewall/egress problem. modal was missing the **`python-socks`** package, so its
proxy code raised `ImportError` the instant a proxy was configured — and the gRPC
fallback wrapper swallowed that into the generic "Could not connect to the Modal server."
The squid CONNECT tunnel worked the whole time.

## Working recipe (proven end-to-end)
```bash
# 1. Install the proxy transport modal needs (persists in the HateVideo conda env)
conda run -n HateVideo pip install 'python-socks[asyncio]'

# 2. Write + verify the token (proxy env vars http_proxy/https_proxy already set)
conda run -n HateVideo modal token set \
  --token-id ak-4iXK4nNQr8hHwtcppxKLLk \
  --token-secret as-iCyc…REDACTED \
  --profile=jehc223 --verify --activate
```
After this:
- `modal token set … --verify` → "Token verified successfully!"
- `modal profile list` → Profile `jehc223`, Workspace `jehc223` (was "Unknown (profile misconfigured)")
- `modal app list` → returns the (empty) apps table, i.e. a live authenticated server call.

No env changes, no NO_PROXY edits, no UPPERCASE proxy vars, and no `--no-verify`
workaround are needed once python-socks is present. The existing lowercase
`http_proxy`/`https_proxy=http://squid.auckland.ac.nz:3128` are sufficient.

## Findings per investigation step

### 1. Write token without server verification
`modal token set` has a `--verify / --no-verify` flag. `--no-verify --activate` wrote the
token to `~/.modal.toml` (`/data/jehc223/home/.modal.toml`) and activated profile `jehc223`.
`modal profile list` showed it, but as **"Unknown (profile misconfigured)"** — because the
workspace name is resolved by a server call that was still failing at that point. (After the
real fix, it resolves to `jehc223`.)

### 2. Reproduce the failure with verbose signal
- Fast-fail (no DEBUG): `modal app list` → "Could not connect to the Modal server."
- DEBUG: the "hang" the team lead saw is **not** a stalled/half-open tunnel. It is modal's
  retry loop: `create_channel_with_fallbacks` retries **18 times** with exponential backoff
  (0.1s→…→5s cap), ~1–2 min total. **Each individual attempt fails in ~8–14 ms** —
  far too fast for a TLS timeout. That instant failure was the key clue: the connect was
  aborting before touching the network, i.e. a local error, not an egress reset.

### 3. Python-level probe of modal's own proxy path (decisive)
Driving `modal._utils.proxy_support` directly in the HateVideo env:
- `get_proxy_url("api.modal.com", use_ssl=True)` → `http://squid.auckland.ac.nz:3128`
  — so modal **does** read the lowercase proxy env vars, and `api.modal.com` is **not**
  NO_PROXY-bypassed. The proxy selection logic is correct.
- `create_proxied_connection(...)` → raised:
  `ImportError("A proxy is configured (http://squid.auckland.ac.nz:3128) but the
  'python-socks' package is not installed. Install it with: pip install
  'modal[api-proxy-support]'")`
- `import python_socks` → `ModuleNotFoundError`; `pip show python-socks` → not found.

Code path: `modal/_utils/grpc_utils.py::ModalChannel._create_connection` calls
`get_proxy_url`; if a proxy URL is returned it calls `create_proxied_connection`
(`modal/_utils/proxy_support.py`), which does `from python_socks.async_.asyncio import Proxy`
and raises `ImportError` when the package is absent. That exception propagates up into
`create_channel_with_fallbacks`, which reports the generic "Could not connect" — hiding the
real cause.

### 4. Environment variants
- NO_PROXY/no_proxy: empty; `api.modal.com` is not bypassed (confirmed via `proxy_bypass`).
- UPPERCASE HTTPS_PROXY / lowercase: irrelevant — modal already resolved the proxy from the
  lowercase vars. No change needed.
- `MODAL_DISABLE_API_PROXY=1`: produces a **different** failure — modal connects directly to
  `api.modal.com:443` and the campus firewall resets the TLS handshake, so attempts hang and
  the command times out (exit 124 at 30s) instead of the instant ImportError. This confirms
  (a) the proxy path is the one that works, and (b) direct 443 is still blocked, exactly as
  the team lead observed.

### 5. Compute-node test
Not required for the fix, and would have been a red herring: the blocker was a missing local
package, identical on login and compute nodes. Note for actually *using* Modal: `modal run`
launches GPU work on Modal's cloud; the local process is a lightweight network orchestrator
(no local GPU), talking to `api.modal.com` over squid. It is not a local compute job, but if a
run streams logs for a long time it should be launched inside a SLURM CPU allocation to avoid
login-node reaping.

### 6. gRPC-over-squid transport knobs
Not needed — plain gRPC over the squid CONNECT tunnel works once python-socks is installed
(`modal app list` returned instantly). See residual risk below re: long-lived streams.

## Residual risk to watch (not currently blocking)
`modal app list` is a short request. Long-lived HTTP/2 gRPC streams (e.g. `modal run` with
live log streaming, or long function calls) traverse the same squid CONNECT tunnel; some
squid configs drop idle/long tunnels. This only surfaces under a real streaming workload, not
this smoke test. If it appears, it will look like a mid-run disconnect (not an auth failure).
Mitigations if needed: run inside a SLURM CPU allocation, and/or investigate modal keepalive
config in `modal/config.py`. No action needed now.

## Recommended next action
Modal connectivity is unblocked. Proceed to a real end-to-end smoke test — a minimal
`modal run` of a trivial CPU function — to confirm launch + log streaming survive the squid
tunnel before committing any GPU workload. Persist the dependency by adding
`python-socks[asyncio]` to the HateVideo environment spec so a rebuild doesn't reintroduce the
bug.

## End-to-end smoke — validated (2026-07-14, connectivity confirmed under a real stream)
`modal run scripts/cloud/modal_smoke.py` ran clean: image built in 58.5 s, then a
CPU worker (torch 2.6.0+cu124, matmul 97 GFLOPS) and a T4 worker (`Tesla T4, driver
580.95.05, 15360 MiB`, matmul 4463 GFLOPS) both returned. **Launch + log streaming
survived the squid CONNECT tunnel with no mid-run disconnect** — the residual risk
in the section above did not materialize on this workload. Token + billing live.

**Data-upload path is NOT yet exercised.** The next steps (`::sync` of the derived
feature caches to a Modal volume, then the first real triage probe) were **blocked
by the harness data-exfiltration guard**: it denies pushing private-repo data to an
external cloud on agent/teammate authorization alone. This is a *policy* gate, not a
connectivity or code problem — connectivity is fully proven by the smoke. Unblocking
requires explicit **user** authorization (or a Bash permission rule for the `modal
run …::sync` command). Until then, features-only offload is code-complete and
connectivity-validated but has moved zero bytes of data.

## Provenance
- modal client 1.5.2, Python 3.11.8, conda env HateVideo.
- Token file: `/data/jehc223/home/.modal.toml`, profile `jehc223`, verified against
  `https://api.modal.com`. Token secret redacted here to `as-iCyc…`.

## End-to-end validated — data upload + first triage probe (2026-07-15)
The data-upload path that the smoke section left unexercised is now proven. Three
blockers, all resolved, none of them connectivity or policy:

1. **Second proxy transport missing — `aiohttp-socks`.** `::sync` uploads via
   Modal's **volume/blob batch-upload**, which uses an **aiohttp HTTP** transport,
   not the gRPC control plane python-socks fixed. It aborted mid-run with
   `ImportError("A proxy is configured ... but the 'aiohttp-socks' package is not
   installed")`. Same failure shape as the original bug, different transport. Fix:
   `pip install aiohttp-socks` (now pinned `==0.11.0` in
   `scripts/cloud/requirements-cloud.txt`). After it, `::sync --dataset HateMM`
   uploaded 52 files / 361.9 MB over the squid tunnel with **no disconnect**.

2. **Container import crash — `parents[2]` IndexError.** The `run` (mounted-code)
   path had never run live. `modal_probe_runner.py:33`,
   `REPO_ROOT = Path(__file__).resolve().parents[2]`, is correct locally
   (`scripts/cloud/…` → repo root) but Modal mounts the entrypoint file at
   `/root/<name>` (depth 1) inside the container, so `parents[2]` → `IndexError`
   at module import, crash-looping every container. The local `modal run` client
   then **hung** on the repeated container-start failures — which looked exactly
   like the "long-stream squid drop" residual risk but was not; `modal app logs
   <app-id>` (a fresh short request) surfaced the real traceback. Fix: `REPO_ROOT`
   is only used locally (sync + image build), so guard the index and fall back to
   the file's dir in the container.

3. **Image missing run_rac.py deps.** The pinned image had
   torch/faiss/sklearn/numpy/scipy/transformers/tqdm but not
   `easydict / pandas / pillow / rank-bm25 / torchmetrics / wandb` (all top-level
   imports on the run_rac → evaluate_rac / data_loader chain; `wandb` is imported
   unconditionally but stays a no-op under `WANDB_MODE=disabled`). Added, pinned to
   the HateVideo versions.

**Result.** HateMM · Qwen2.5-VL-7B · seed 0 (exact `enc3seed.sbatch` args) ran
30/30 epochs on a T4 in ~33 s. Final-epoch Test_Retrieval: cloud **acc 0.8744 /
mF1 0.8666 / roc 0.9311** vs banked local (A100) **0.8605 / 0.8507 / 0.9283** →
drift **+0.0139 / +0.0159 / +0.0028**. That ~1.4 pp is a 3-of-~215 test-video flip
from a 30-epoch *train* diverging across A100/T4 kernels — the triage-only regime,
not bit reproduction. **Cost across all attempts ≈ $0.01–0.03** (free credits).
Cloud numbers are triage-only forever; paper numbers re-run locally (G-repro).
