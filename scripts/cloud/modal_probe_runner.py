"""Features-only Modal workhorse for RGCL head-training / probing.

BINDING SCOPE (user-ratified): ONLY derived float feature caches
(data/CLIP_Embedding/<dataset>/*.pt) and ground-truth label files
(data/gt/<dataset>/*.jsonl|json|csv) may leave the cluster. RAW VIDEOS NEVER
DO. A fail-loud guard (`assert_uploadable`) refuses anything that looks like
media (video/audio extension) or that lives under a `video/` directory, before
a single byte is uploaded. Cloud results are EXPLORATORY TRIAGE ONLY -- any
pre-registered or paper number is re-run locally on the same hardware as the
rest of its table (G-repro rule; see research-wiki/CLOUD_GPU_FEASIBILITY_2026-07-14.md).

Usage (after `python3 -m modal setup`):
  # 1. push a dataset's float caches + labels to the persistent volume
  modal run scripts/cloud/modal_probe_runner.py::sync --dataset HateMM
  # 2. run any repo script against the volume-mounted caches (CPU by default)
  modal run scripts/cloud/modal_probe_runner.py::run \
      --script src/run_rac.py \
      --args "--path /root/data --dataset HateMM --model Qwen2.5-VL-7B-Instruct_HF --seed 0"
  #    add --gpu to get a T4 instead of CPU
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "rgcl-probe"
VOLUME_NAME = "rgcl-features"
# scripts/cloud/modal_probe_runner.py -> repo root is two levels up when this
# file runs locally (sync + image build use it). Inside a Modal container the
# entrypoint is mounted at /root/<name>, so parents[2] would IndexError on
# import; the container never needs REPO_ROOT, so fall back to the file's dir.
_SELF = Path(__file__).resolve()
REPO_ROOT = _SELF.parents[2] if len(_SELF.parents) > 2 else _SELF.parent

# ---------------------------------------------------------------------------
# Image: pinned to the HateVideo conda env versions for feature/cache parity.
# ---------------------------------------------------------------------------
_IGNORE = ["**/__pycache__", "**/__pycache__/**", "*.pyc", "**/*.pyc"]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "faiss-cpu==1.13.2",
        "scikit-learn==1.5.2",
        "numpy==1.26.4",
        "scipy==1.17.1",
        "transformers==4.49.0",
        "tqdm",
        # run_rac.py import chain (evaluate_rac / data_loader) needs these too;
        # pinned to the HateVideo env so a triage probe matches the banked run.
        # wandb is imported unconditionally but stays a no-op (WANDB_MODE=disabled).
        "easydict==1.13",
        "pandas==2.3.3",
        "pillow==11.1.0",
        "rank-bm25==0.2.2",
        "torchmetrics==1.9.0",
        "wandb==0.28.0",
    )
    # Match the banked SLURM run's environment (see scripts/slurm/enc3seed.sbatch)
    # so a triage probe differs from the local number ONLY in hardware/libraries:
    # no wandb, HF offline (features are precomputed -> no downloads), live logs.
    .env({"WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    # Minimal code subset ONLY -- never the whole repo (no data/, no logs).
    .add_local_dir(str(REPO_ROOT / "src"), "/root/src", ignore=_IGNORE)
    .add_local_dir(str(REPO_ROOT / "scripts" / "analysis"),
                   "/root/scripts/analysis", ignore=_IGNORE)
)

app = modal.App(APP_NAME, image=image)
features = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Fail-loud video/media upload guard (defense-in-depth on top of the allowlist)
# ---------------------------------------------------------------------------
_MEDIA_EXTS = {
    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".flv", ".wmv",
    ".mpg", ".mpeg", ".ts", ".gif",              # video
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg",  # audio
}
_FORBIDDEN_DIR_PARTS = {"video", "videos"}
_ALLOWED_EXTS = {".pt", ".jsonl", ".json", ".csv", ".npy", ".txt"}


def guard_reason(path) -> str | None:
    """Return a rejection reason if `path` must not be uploaded, else None."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in _MEDIA_EXTS:
        return f"media (video/audio) extension {ext!r}"
    hit = {seg.lower() for seg in p.parts} & _FORBIDDEN_DIR_PARTS
    if hit:
        return f"path under forbidden media dir {sorted(hit)}"
    if ext not in _ALLOWED_EXTS:
        return (f"extension {ext!r} not in features/label allowlist "
                f"{sorted(_ALLOWED_EXTS)}")
    return None


def assert_uploadable(local_path) -> None:
    """Raise (fail-loud) unless `local_path` is an allowlisted feature/label file.

    Checks both the literal path and the symlink-resolved real path, so a
    symlink can never smuggle a video past the allowlist.
    """
    real = Path(local_path).resolve()
    for candidate in (Path(local_path), real):
        reason = guard_reason(candidate)
        if reason:
            raise RuntimeError(
                f"[VIDEO-GUARD] REFUSING to upload {local_path}: {reason} "
                f"(resolved={real}). Only derived float caches + label files "
                f"may leave the cluster; raw videos never do."
            )


# ---------------------------------------------------------------------------
# sync: upload ONLY a dataset's float caches + label files to the volume
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def sync(dataset: str = "HateMM"):
    sources = [
        (REPO_ROOT / "data" / "CLIP_Embedding" / dataset, f"/CLIP_Embedding/{dataset}"),
        (REPO_ROOT / "data" / "gt" / dataset, f"/gt/{dataset}"),
    ]
    to_upload: list[tuple[Path, str]] = []
    for base, remote_base in sources:
        if not base.exists():
            print(f"[sync] WARN: missing {base} -- skipping")
            continue
        for f in sorted(base.rglob("*")):
            if f.is_dir():
                continue
            assert_uploadable(f)  # fail-loud BEFORE any upload happens
            rel = f.relative_to(base).as_posix()
            to_upload.append((f, f"{remote_base}/{rel}"))

    if not to_upload:
        raise SystemExit(f"[sync] nothing allowlisted to upload for dataset={dataset!r}")

    total_mb = sum(p.stat().st_size for p, _ in to_upload) / 1e6
    print(f"[sync] {len(to_upload)} files ({total_mb:.1f} MB) pass the guard; "
          f"uploading to volume {VOLUME_NAME!r} ...")
    with features.batch_upload(force=True) as batch:
        for local, remote in to_upload:
            batch.put_file(str(local), remote)
    print(f"[sync] done: {dataset} caches + labels are on volume {VOLUME_NAME!r}.")


# ---------------------------------------------------------------------------
# run_probe: execute an arbitrary repo script against the volume-mounted caches
# ---------------------------------------------------------------------------
def _execute(script: str, script_args: str) -> dict:
    env = os.environ.copy()
    # run_rac.py imports are src/-relative (`from model...`, `from data_loader...`)
    env["PYTHONPATH"] = ":".join(["/root/src", "/root", env.get("PYTHONPATH", "")])
    try:
        features.reload()  # pick up the latest committed volume contents
    except Exception as exc:  # noqa: BLE001
        print(f"[run_probe] volume reload skipped: {exc}")
    cmd = [sys.executable, f"/root/{script}"] + shlex.split(script_args)
    print(f"[run_probe] cwd=/root exec: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/root", env=env)
    # Plumbing: persist any /root/data writes (e.g. a probe's --out_md/--out_json landed on the mounted
    # volume) so they survive the container and can be retrieved with `modal volume get`. Without this the
    # in-container writes are discarded on exit. Probe logic is untouched.
    try:
        features.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[run_probe] volume commit skipped: {exc}")
    return {"script": script, "returncode": result.returncode}


@app.function(volumes={"/root/data": features}, timeout=1800)
def run_probe_cpu(script: str, script_args: str = "") -> dict:
    return _execute(script, script_args)


@app.function(gpu="T4", volumes={"/root/data": features}, timeout=1800)
def run_probe_gpu(script: str, script_args: str = "") -> dict:
    return _execute(script, script_args)


@app.local_entrypoint()
def run(script: str, args: str = "", gpu: bool = False):
    # Refuse to launch a probe whose args point at a raw media file.
    for tok in shlex.split(args):
        if Path(tok).suffix.lower() in _MEDIA_EXTS:
            raise RuntimeError(
                f"[VIDEO-GUARD] refusing: script arg references a media file: {tok!r}"
            )
    fn = run_probe_gpu if gpu else run_probe_cpu
    print(f"[run] dispatching {script} on {'T4' if gpu else 'CPU'} ...")
    print(fn.remote(script, args))
