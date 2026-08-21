#!/usr/bin/env python
"""REPRO campaign Wave 2 -- CLAP (CVPR 2024) data-adaptation layer and driver.

`third_party/CLAP` (AnasEmad11/CLAP @ 3dcaadc1) is a federated-learning
codebase: `src/server/fedavg.py` reads a UCF-Crime / XD-Violence *concatenated
snippet feature matrix* plus three pickled partitions (per-video snippet index
lists, global video numbers, and a flattened index chain), runs the paper's
coarse-to-fine pseudo-label generator (C2FPL) on each client's own videos,
aggregates the per-client "normal" Gaussians into one mixture, turns that into
segment-level pseudo-labels, and trains a small MLP scorer under FedAvg.

This module builds exactly that input contract out of our frozen dense 4 fps
CLIP-L/336 caches and drives the repo's own entry point.  Nothing in the CLAP
algorithm is re-implemented here: the clustering, the Gaussian mixture, the
pseudo-label window search, the model and the FedAvg aggregation are the repo's.

Stages
  build     write third_party/CLAP/data/hate_<DS>/ (features + partitions)
  train     run src/server/fedavg.py for one (variant, seed), dumping per-round
            per-snippet scores over the ordered score set
  normality CLAP's aggregated normal-Gaussian score, no MLP (our ablation row)
  curves    assemble idea-stage/repro_clap/curves/<DS>/<vid>.npz
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
CLAP = ROOT / "third_party/CLAP"
SHIM = ROOT / "scripts/repro_campaign/shim/clap"
OUT = ROOT / "idea-stage/repro_clap"

DATASETS = ["HateMM", "MHC", "MHC_zh", "HateClipSeg"]

# --- frozen adaptation constants (see sections P.1/P.2 of the results file) ---
SNIPPET = 2          # 4 fps frames per snippet -> 0.5 s, the duration of the
                     # 16-frame I3D clip CLAP's UCF-Crime features are built on
NATIVE_RATE = 4.0 / SNIPPET          # 2.0 samples per second
NCLIENTS = {"fedavg11": 11,          # train.sh default (scene_partition_11_V3)
            "central": 1}            # the paper's "Centralized" configuration
SEEDS = [20250819, 20250820, 20250821]
DUMP_ROUNDS = [1, 2, 5, 10]          # candidate global-round counts (val knob)
LOCAL_EPOCH = 10                     # train.sh
BATCH = 128                          # train.sh
MIN_SNIPPETS_TRAIN = 4               # np.diff + a >=1 window need a few snippets
MIN_SNIPPETS_SCORE = 1


def feat_dir(ds: str) -> Path:
    return ROOT / f"data/CLIP_Embedding/{ds}/dense4fps_clipL336"


def load_split(ds: str) -> dict:
    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    return {str(v): str(s) for v, s in zip(z["video_ids"], z["split"])}


def snippets(a: np.ndarray) -> np.ndarray:
    """(T, 1024) 4 fps frames -> (T // SNIPPET, 1, 1024) 0.5 s snippets.

    Mean over the SNIPPET frames inside the snippet; the trailing remainder
    frames are dropped, exactly as a fixed-stride clip extractor drops them.
    The singleton axis is CLAP's `ncrops` axis (UCF-Crime features are 10-crop;
    ours are single-view, so ncrops = 1 and the model's `x.mean(dim=1)` is a
    no-op).
    """
    n = a.shape[0] // SNIPPET
    if n == 0:
        return np.zeros((0, 1, a.shape[1]), dtype=np.float32)
    return a[: n * SNIPPET].reshape(n, SNIPPET, a.shape[1]).mean(1)[:, None, :] \
        .astype(np.float32)


# --------------------------------------------------------------- stage build ---
def build(ds: str) -> None:
    split = load_split(ds)
    fd = feat_dir(ds)
    present = {p.stem for p in fd.glob("*.npy")}

    counts, ok_ids = {}, []
    for vid in sorted(present):
        n = int(np.load(fd / f"{vid}.npy", mmap_mode="r").shape[0]) // SNIPPET
        counts[vid] = n
        if n >= MIN_SNIPPETS_SCORE:
            ok_ids.append(vid)

    train_ids = [v for v in ok_ids
                 if split.get(v) == "train" and counts[v] >= MIN_SNIPPETS_TRAIN]
    dropped_train = [v for v in sorted(present)
                     if split.get(v) == "train" and counts[v] < MIN_SNIPPETS_TRAIN]
    missing = [v for v, s in split.items() if v not in present]
    score_ids = ok_ids

    for variant, ncl in NCLIENTS.items():
        d = CLAP / "data" / f"hate_{ds}_{variant}"
        (d / "clusters").mkdir(parents=True, exist_ok=True)

        # --- concatenated snippet matrices -------------------------------------
        tr_path, sc_path = d / "concat_train.npy", d / "concat_score.npy"
        if not tr_path.exists():
            write_concat(fd, train_ids, counts, tr_path)
        if not sc_path.exists():
            write_concat(fd, score_ids, counts, sc_path)

        # --- partitions --------------------------------------------------------
        # Deterministic, label-independent client assignment.  CLAP's own split
        # is scene-based; our corpora carry no scene metadata, so the rule is
        # crc32(video_id) mod n_clients, written down here and nowhere else.
        offs, cur = {}, 0
        for v in train_ids:
            offs[v] = cur
            cur += counts[v]
        n_score = sum(counts[v] for v in score_ids)

        per_client_vid = [[] for _ in range(ncl)]
        for i, v in enumerate(train_ids):
            per_client_vid[zlib.crc32(v.encode()) % ncl].append(i)

        data_indices, chain = [], []
        for c in range(ncl):
            vids = per_client_vid[c]
            idx_lists = [np.arange(offs[train_ids[i]],
                                   offs[train_ids[i]] + counts[train_ids[i]])
                         for i in vids]
            data_indices.append({"train": idx_lists, "test": []})
            flat = list(np.concatenate(idx_lists)) if idx_lists else []
            chain.append({"train": [int(x) for x in flat],
                          "test": list(range(n_score)) if c == 0 else []})

        partition = {"data_indices": data_indices,
                     "separation": {"train": list(range(ncl)), "test": [0],
                                    "total": ncl}}
        vnum = {"data_indices": [{"train": per_client_vid[c], "test": []}
                                 for c in range(ncl)],
                "separation": partition["separation"]}
        chain_p = {"data_indices": chain, "separation": partition["separation"]}
        for name, obj in [("partition.pkl", partition),
                          ("video_num_partition.pkl", vnum),
                          ("partition_chain.pkl", chain_p)]:
            with open(d / name, "wb") as fh:
                pickle.dump(obj, fh)

        (d / "meta.json").write_text(json.dumps({"no_of_vids": len(train_ids)}))
        (d / "args.json").write_text(json.dumps({"split": "sample"}))
        (d / "manifest.json").write_text(json.dumps({
            "dataset": ds, "variant": variant, "n_clients": ncl,
            "snippet_frames": SNIPPET, "native_rate": NATIVE_RATE,
            "train_ids": train_ids, "score_ids": score_ids,
            "counts": {v: counts[v] for v in score_ids},
            "n_train_snippets": cur, "n_score_snippets": n_score,
            "dropped_train_too_short": dropped_train,
            "missing_features": sorted(missing),
            "client_sizes": [len(x) for x in per_client_vid],
        }, indent=1))
        print(f"[build] {ds}/{variant} clients={ncl} train_vid={len(train_ids)} "
              f"train_snip={cur} score_vid={len(score_ids)} score_snip={n_score} "
              f"dropped={len(dropped_train)} missing_feat={len(missing)}",
              flush=True)


def write_concat(fd: Path, ids: list, counts: dict, path: Path) -> None:
    total = sum(counts[v] for v in ids)
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(total, 1, 1024))
    cur = 0
    for v in ids:
        s = snippets(np.load(fd / f"{v}.npy"))
        arr[cur:cur + len(s)] = s
        cur += len(s)
    arr.flush()
    del arr


# --------------------------------------------------------------- stage train ---
def train(ds: str, variant: str, seed: int) -> None:
    d = CLAP / "data" / f"hate_{ds}_{variant}"
    run_dir = OUT / "runs" / f"{ds}_{variant}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{SHIM}:{CLAP}"
    env["WANDB_MODE"] = "disabled"
    env["CUDA_VISIBLE_DEVICES"] = env.get("CLAP_GPU", "0")
    env["OMP_NUM_THREADS"] = env.get("CLAP_THREADS", "4")
    cmd = [sys.executable, "src/server/fedavg.py",
           "-d", f"hate_{ds}_{variant}", "-m", "c2fpl_XD",
           "--train_mode", "US", "--global_testset", "1",
           "-bs", str(BATCH),
           "--global_epoch", str(max(DUMP_ROUNDS)),
           "--local_epoch", str(LOCAL_EPOCH),
           "--partition", "partition.pkl",
           "--partition_chain", "partition_chain.pkl",
           "--video_num_partition", "video_num_partition.pkl",
           "--datasplit", "hate", "--test_gap", "10000",
           "--join_ratio", "1", "--gmm_pl", "1", "--eta_clustering", "0",
           "--load", "0", "--seed", str(seed),
           "--use_cuda", "1", "--gpu_memory_fraction", "0.10",
           "--save_log", "0",
           "--dump_scores", str(run_dir / "scores_r{round}.npy"),
           "--dump_rounds", ",".join(str(r) for r in DUMP_ROUNDS)]
    (run_dir / "cmd.txt").write_text(" ".join(cmd))
    t0 = time.time()
    print(f"[train] start {ds} {variant} seed={seed}", flush=True)
    with open(run_dir / "stdout.log", "w") as fh:
        subprocess.run(cmd, cwd=CLAP, env=env, check=True, stdout=fh,
                       stderr=subprocess.STDOUT)
    print(f"[train] done  {ds} {variant} seed={seed} "
          f"{time.time() - t0:.0f}s", flush=True)
    # the cluster pickles are large and deterministic; keep only the l2 ones,
    # which the normality stage re-reads
    for p in (d / "clusters").glob("*_abnormal.pkl"):
        p.unlink()


# ----------------------------------------------------------- stage normality ---
def normality(ds: str, variant: str = "fedavg11") -> np.ndarray:
    """CLAP's aggregated normal model, evaluated on every scored snippet.

    The coarse stage splits each client's videos into a normal and an abnormal
    cluster; the fine stage fits a 1-D Gaussian to the pooled squared-L2 norms
    of that client's normal videos, and `gmm_PL::sum_multivariate_normals`
    mixes the per-client Gaussians with weights proportional to the client's
    snippet count.  CLAP uses that mixture only to *search for a window* inside
    an abnormal video; the mixture density itself is a per-snippet normality
    score, and its negation is reported here as an internal ablation of the
    method's coarse-to-fine stage, with no trained MLP in the loop.
    """
    from scipy.stats import multivariate_normal
    d = CLAP / "data" / f"hate_{ds}_{variant}"
    man = json.loads((d / "manifest.json").read_text())
    ncl = man["n_clients"]
    parts = []
    for c in range(ncl):
        with open(d / "clusters" / f"hate_split_{c}_of_{ncl}_normal_l2.pkl",
                  "rb") as fh:
            normal_l2 = pickle.load(fh)
        pooled = np.concatenate([np.asarray(v) for v in normal_l2.values()]) \
            if normal_l2 else np.zeros(0)
        if pooled.size < 2:
            continue
        parts.append((float(pooled.mean()), float(np.cov(pooled)),
                      int(pooled.size)))
    tot = sum(p[2] for p in parts)
    fd = feat_dir(ds)
    out = {}
    for v in man["score_ids"]:
        l2 = np.sum(np.square(snippets(np.load(fd / f"{v}.npy"))), axis=2) \
               .mean(axis=1)
        p = np.zeros(len(l2))
        for mu, var, n in parts:
            p += multivariate_normal(mu, var).pdf(l2) * (n / tot)
        out[v] = (-p).astype(np.float64)
    return out


# -------------------------------------------------------------- stage curves ---
def split_scores(ds: str, variant: str, path: Path) -> dict:
    man = json.loads(
        (CLAP / "data" / f"hate_{ds}_{variant}" / "manifest.json").read_text())
    s = np.load(path).astype(np.float64).reshape(-1)
    assert len(s) == man["n_score_snippets"], (len(s), man["n_score_snippets"])
    out, cur = {}, 0
    for v in man["score_ids"]:
        n = man["counts"][v]
        out[v] = s[cur:cur + n]
        cur += n
    return out


def write_curves(ds: str, arrays: dict, dest: Path) -> None:
    """arrays: {variant_key: {vid: curve}} -> one npz per video."""
    d = dest / ds
    d.mkdir(parents=True, exist_ok=True)
    vids = set.intersection(*[set(a) for a in arrays.values()])
    for v in sorted(vids):
        np.savez(d / f"{v}.npz", rate=np.float64(NATIVE_RATE),
                 **{k: arrays[k][v] for k in arrays})
    print(f"[curves] {ds} -> {d} n={len(vids)} keys={sorted(arrays)}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["build", "train", "normality", "curves",
                             "knobsel_curves"])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--variants", default=",".join(NCLIENTS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--rounds", default="")
    args = ap.parse_args()
    dss = args.datasets.split(",")

    if args.stage == "build":
        for ds in dss:
            build(ds)
    elif args.stage == "train":
        for ds in dss:
            for variant in args.variants.split(","):
                for seed in [int(s) for s in args.seeds.split(",")]:
                    train(ds, variant, seed)
    elif args.stage == "knobsel_curves":
        # one key per candidate round count, seed 20250819, variant fedavg11,
        # written to a throwaway tree that is scored on val only
        for ds in dss:
            arrays = {}
            for r in DUMP_ROUNDS:
                p = OUT / "runs" / f"{ds}_fedavg11_s{SEEDS[0]}" / f"scores_r{r}.npy"
                arrays[f"r{r}"] = split_scores(ds, "fedavg11", p)
            write_curves(ds, arrays, OUT / "knobsel")
    elif args.stage == "normality":
        for ds in dss:
            np.savez(OUT / f"normality_{ds}.npz", **normality(ds))
            print(f"[normality] {ds} done", flush=True)
    elif args.stage == "curves":
        rec = json.loads((OUT / "run_record.json").read_text())
        for ds in dss:
            r = rec["frozen_rounds"][ds]
            arrays = {}
            for variant in NCLIENTS:
                for i, seed in enumerate(SEEDS):
                    p = OUT / "runs" / f"{ds}_{variant}_s{seed}" / f"scores_r{r}.npy"
                    arrays[f"{variant}_s{i}"] = split_scores(ds, variant, p)
            z = np.load(OUT / f"normality_{ds}.npz")
            arrays["normality"] = {k: z[k] for k in z.files}
            write_curves(ds, arrays, OUT / "curves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
