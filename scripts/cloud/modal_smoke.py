"""Modal smoke test for the RGCL features-only offload.

Validates that the Modal token + billing work and that both a CPU worker and a
T4 GPU worker can spin up, import torch, and run a timed matmul. Uploads NO data
and mounts NO repo code -- it exists purely to prove the account is live once the
user has authenticated (`python3 -m modal setup`).

Run:  modal run scripts/cloud/modal_smoke.py

Scope: this is infra validation only. It never touches datasets, feature caches,
or raw videos (see scripts/cloud/README.md).
"""

import modal

app = modal.App("rgcl-smoke")

# Tiny pinned image: just torch (parity with the HateVideo env's torch==2.6.0).
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.6.0")


def _report(device: str, n: int = 8192) -> dict:
    """Import torch, print versions + nvidia-smi, and time an ~1e8-element matmul."""
    import platform
    import subprocess
    import time

    import torch

    info = {
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    # nvidia-smi (present only on the GPU worker; fail soft on CPU).
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        info["nvidia_smi"] = smi.stdout.strip() or smi.stderr.strip()
    except Exception as exc:  # noqa: BLE001 - CPU container has no nvidia-smi
        info["nvidia_smi"] = f"unavailable ({exc})"

    dev = "cuda" if (device == "T4" and torch.cuda.is_available()) else "cpu"
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)
    # warm-up
    _ = a @ b
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = a @ b
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gflops = (2.0 * n ** 3) / dt / 1e9
    info["matmul"] = {"n": n, "elements": n * n, "run_device": dev,
                      "seconds": round(dt, 4), "gflops": round(gflops, 1)}
    print(f"[smoke:{device}] {info}")
    return info


@app.function(image=image)
def cpu_probe() -> dict:
    return _report("CPU")


@app.function(image=image, gpu="T4")
def gpu_probe() -> dict:
    return _report("T4")


@app.local_entrypoint()
def main():
    print("=== CPU worker ===")
    print(cpu_probe.remote())
    print("=== T4 worker ===")
    print(gpu_probe.remote())
    print("smoke OK: token + billing + CPU/T4 workers all live.")
