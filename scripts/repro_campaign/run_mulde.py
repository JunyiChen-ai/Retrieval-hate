#!/usr/bin/env python
"""REPRO campaign Wave 2 — MULDE (CVPR 2024) on the four hate corpora.

Port of `third_party/MULDE` (jakubmicorek/MULDE @ f821b965).  The model, the
denoising-score-matching loss, the noise-conditioning, the log-density
regulariser, the optimiser settings and the multiscale read-out are taken from
the upstream `main.py` / `models.py`; only the data path is ours, because the
upstream repo ships a 2-D toy `dataset.py` and no video front-end at all.

Supervision class: **one-class**.  The training pool is the frames of the
train-split videos whose video-level gold is non-hateful (`y_video == 0`).  No
test-split video and no test label ever reaches training or model selection.

Free knobs (hidden width, top noise scale, epoch budget, multiscale read-out)
are chosen on the **val** split and frozen into `run_record_<ds>_<variant>.json`
before the single test call.

Outputs `idea-stage/repro_mulde/curves/<DS>/<vid>.npz` with one key per
(variant, seed) and a scalar `rate = 4.0`, which is the interface
`scripts/repro_campaign/eval_frame.py --method curves` reads.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn import mixture
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "third_party/MULDE"))
from models import MLPs, ScoreOrLogDensityNetwork  # noqa: E402  (upstream, unmodified)

OUT = ROOT / "idea-stage/repro_mulde"
FPS = 4.0
SEEDS = [20250819, 20250820, 20250821]
SELECT_SEED = SEEDS[0]

# Upstream defaults that we do NOT touch (README "essential parameters" line).
LR = 4e-5
BATCH = 2048
BETA = 0.1
SIGMA_LOW = 1e-3
L = 16
EPOCH_CKPTS = [25, 50, 100]

# The val-selected grid.
GRID_UNITS = [[4096, 4096], [1024, 1024]]
GRID_SIGMA_HIGH = [0.5, 1.0]

FEATS = {
    "clipL336": ["dense4fps_clipL336"],
    "w2vemo": ["dense4fps_w2vemo"],
    "clip_w2vemo": ["dense4fps_clipL336", "dense4fps_w2vemo"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------- data ---
def load_gt(ds: str) -> dict:
    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    out = {}
    for i, vid in enumerate(z["video_ids"]):
        out[str(vid)] = dict(
            y4=np.asarray(z["y4"][i], dtype=np.int8),
            split=str(z["split"][i]),
            y_video=int(z["y_video"][i]),
        )
    return out


def load_features(ds: str, variant: str, gt: dict) -> dict:
    dirs = [ROOT / f"data/CLIP_Embedding/{ds}/{d}" for d in FEATS[variant]]
    feats, missing = {}, []
    for vid in gt:
        arrs = []
        ok = True
        for d in dirs:
            p = d / f"{vid}.npy"
            if not p.exists():
                ok = False
                break
            arrs.append(np.load(p).astype(np.float32))
        if not ok or any(a.size == 0 for a in arrs):
            missing.append(vid)
            continue
        T = min(a.shape[0] for a in arrs)
        if T == 0:
            missing.append(vid)
            continue
        x = np.concatenate([a[:T] for a in arrs], axis=1) if len(arrs) > 1 else arrs[0]
        if not np.isfinite(x).all():
            missing.append(vid)
            continue
        feats[vid] = x
    return feats, missing


def pool(gt: dict, feats: dict, split: str, only_negative: bool = False):
    """Concatenated frames of one split (optionally only y_video == 0 videos)."""
    xs, ys = [], []
    for vid, g in gt.items():
        if g["split"] != split or vid not in feats:
            continue
        if only_negative and g["y_video"] != 0:
            continue
        x = feats[vid]
        T = min(len(x), len(g["y4"]))
        if T <= 0:
            continue
        xs.append(x[:T])
        ys.append(g["y4"][:T])
    if not xs:
        return np.zeros((0, 0), np.float32), np.zeros(0, np.int8)
    return np.concatenate(xs), np.concatenate(ys)


# ------------------------------------------------------------------ model ---
def make_model(d: int, units, device):
    # upstream: input_dim = d + 1 for the noise conditioning, output 1, GELU MLP
    return ScoreOrLogDensityNetwork(MLPs(input_dim=d + 1, units=units),
                                    score_network=False).to(device)


def train(x_train: torch.Tensor, mean, std, units, sigma_high, seed, device,
          ckpts, tag=""):
    """Upstream `train_and_evaluate`'s training half, verbatim in its arithmetic.

    Yields (epoch, model) at each checkpoint so the epoch budget can be picked
    on val without retraining.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(x_train.shape[1], units, device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.9))
    n = x_train.shape[0]
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for epoch in range(1, max(ckpts) + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            x = x_train[idx].to(device, non_blocking=True)
            x = (x - mean) / (std + 1e-8)
            sigma = torch.as_tensor(
                np.exp(np.random.uniform(np.log(SIGMA_LOW), np.log(sigma_high),
                                         x.size(0))),
                dtype=torch.float32, device=device).unsqueeze(1)
            noise = torch.randn(x.shape, device=device) * sigma
            x = x.requires_grad_()
            x_ = x + noise
            lam = (sigma ** 2).ravel()
            score_, _ = model.score(torch.hstack([x_, sigma]), return_log_density=True)
            loss = torch.norm(score_[:, :-1] + noise / (sigma ** 2), dim=-1) ** 2
            loss = (lam * loss).mean() / 2.0
            _, ld_noise_free = model.score(torch.hstack([x, sigma]),
                                           return_log_density=True)
            loss = loss + BETA * (ld_noise_free ** 2).mean() / 2.0
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        if epoch % 10 == 0 or epoch in ckpts:
            log(f"[progress] {tag} epoch={epoch}/{max(ckpts)} "
                f"loss={tot / max(nb, 1):.4f} elapsed={time.time() - t0:.0f}s")
        if epoch in ckpts:
            yield epoch, model


@torch.enable_grad()
def multiscale_scores(model, x: torch.Tensor, mean, std, sigma_high, device):
    """Upstream `calculate_scores(..., return_scores_by_sigma=True)`.

    For each of the L noise scales linspaced over [sigma_low, sigma_high] the
    clean sample is evaluated at that conditioning value; the two read-outs the
    repo keeps are the log-density itself and the squared norm of its score
    (the sigma^2 lambda weight is applied only to the *individual* read-out the
    repo prints, not to the per-sigma arrays it aggregates, so it is not applied
    here either).  Returns (N, L) for each read-out.
    """
    sigmas = [float(f"{s:.5f}")
              for s in np.linspace(SIGMA_LOW, sigma_high, L).tolist()]
    model.eval()
    ld_out = np.empty((x.shape[0], L), np.float32)
    sn_out = np.empty((x.shape[0], L), np.float32)
    for i in range(0, x.shape[0], BATCH):
        xb = x[i:i + BATCH].to(device)
        xb = (xb - mean) / (std + 1e-8)
        for j, s in enumerate(sigmas):
            inp = torch.hstack([xb, torch.full((xb.shape[0], 1), s, device=device)])
            inp = inp.requires_grad_()
            ld = model(inp)
            grad = torch.autograd.grad(-ld.sum(), inp)[0]
            ld_out[i:i + xb.shape[0], j] = ld.detach().ravel().float().cpu().numpy()
            sn_out[i:i + xb.shape[0], j] = (
                (grad[:, :-1] ** 2).sum(1).detach().float().cpu().numpy())
    return {"log_density": ld_out, "score_norm": sn_out}, sigmas


def aggregate(ms_train: np.ndarray, ms_eval: np.ndarray, seed: int) -> dict:
    """Upstream aggregate read-outs: max / median / mean of the train-standardised
    multiscale vector, and the negative log-likelihood of a GMM fitted to the
    train multiscale vectors (upstream fits 1, 3 and 5 components)."""
    m, s = ms_train.mean(0), ms_train.std(0)
    z = (ms_eval - m) / (s + 1e-8)
    out = {"max": z.max(1), "median": np.median(z, 1), "mean": z.mean(1)}
    for k in (1, 3, 5):
        gm = mixture.GaussianMixture(n_components=k, covariance_type="full",
                                     random_state=seed).fit(ms_train)
        out[f"gmm{k}_nll"] = -gm.score_samples(ms_eval)
    return out


def aggregate_fitted(ms_train: np.ndarray, seed: int):
    """Same as `aggregate` but returns closures, so the GMMs are fitted once and
    reused for every video when the whole corpus is scored."""
    m, s = ms_train.mean(0), ms_train.std(0)
    gms = {k: mixture.GaussianMixture(n_components=k, covariance_type="full",
                                      random_state=seed).fit(ms_train)
           for k in (1, 3, 5)}

    def apply(ms_eval, name):
        z = (ms_eval - m) / (s + 1e-8)
        if name == "max":
            return z.max(1)
        if name == "median":
            return np.median(z, 1)
        if name == "mean":
            return z.mean(1)
        k = int(name[3])
        return -gms[k].score_samples(ms_eval)
    return apply


# ------------------------------------------------------------------- main ---
def run(ds: str, variant: str, device: str, force: bool = False):
    rec_path = OUT / f"run_record_{ds}_{variant}.json"
    gt = load_gt(ds)
    log(f"=== {ds} / {variant}: loading features")
    feats, missing = load_features(ds, variant, gt)
    log(f"[shape] {ds}/{variant} videos_with_features={len(feats)} missing={len(missing)}")

    x_tr_np, _ = pool(gt, feats, "train", only_negative=True)
    x_va_np, y_va = pool(gt, feats, "val", only_negative=False)
    d = x_tr_np.shape[1]
    log(f"[shape] train_pool={x_tr_np.shape} val_pool={x_va_np.shape} "
        f"val_pos_rate={float(y_va.mean()):.4f}")
    if x_tr_np.shape[0] < 1000:
        log(f"[abort] {ds}/{variant} train pool too small")
        return

    x_tr = torch.from_numpy(x_tr_np)
    x_va = torch.from_numpy(x_va_np)
    mean = x_tr.mean(0).to(device)
    std = x_tr.std(0).to(device)

    # ---------------------------------------------------------- selection ---
    if rec_path.exists() and not force:
        rec = json.loads(rec_path.read_text())
        log(f"[frozen] reusing {rec_path.name}: {rec['chosen']}")
    else:
        t0 = time.time()
        rows = []
        for units in GRID_UNITS:
            for sh in GRID_SIGMA_HIGH:
                tag = f"{ds}/{variant} sel units={units[0]} sh={sh}"
                for ep, model in train(x_tr, mean, std, units, sh, SELECT_SEED,
                                       device, EPOCH_CKPTS, tag=tag):
                    ms_tr, sigmas = multiscale_scores(model, x_tr, mean, std, sh, device)
                    ms_va, _ = multiscale_scores(model, x_va, mean, std, sh, device)
                    for st in ("log_density", "score_norm"):
                        aggs = aggregate(ms_tr[st], ms_va[st], SELECT_SEED)
                        for an, sc in aggs.items():
                            if not np.isfinite(sc).all():
                                continue
                            rows.append(dict(units=units, sigma_high=sh, epochs=ep,
                                             score_type=st, agg=an,
                                             val_roc=float(roc_auc_score(y_va, sc))))
                        # diagnostic only, never selected from
                    ind = {st: [float(roc_auc_score(y_va, ms_va[st][:, j]))
                                for j in range(L)] for st in ms_va}
                    log(f"[val] {tag} ep={ep} best_agg="
                        f"{max(r['val_roc'] for r in rows):.4f} "
                        f"ind_ld_max={max(ind['log_density']):.4f} "
                        f"ind_sn_max={max(ind['score_norm']):.4f}")
                    del model
                    torch.cuda.empty_cache()
        best = max(rows, key=lambda r: r["val_roc"])
        rec = dict(dataset=ds, variant=variant, feature_dirs=FEATS[variant],
                   feature_dim=d, native_rate=FPS,
                   supervision="one-class",
                   train_pool_videos=int(sum(1 for v, g in gt.items()
                                             if g["split"] == "train" and g["y_video"] == 0
                                             and v in feats)),
                   train_pool_frames=int(x_tr_np.shape[0]),
                   val_pool_frames=int(x_va_np.shape[0]),
                   val_pos_rate=round(float(y_va.mean()), 4),
                   fixed=dict(lr=LR, batch_size=BATCH, beta=BETA,
                              sigma_low=SIGMA_LOW, L=L,
                              optimizer="Adam(betas=(0.5,0.9))",
                              standardisation="component-wise, train pool"),
                   grid=dict(units=GRID_UNITS, sigma_high=GRID_SIGMA_HIGH,
                             epochs=EPOCH_CKPTS,
                             read_outs=["log_density", "score_norm"],
                             aggregations=["max", "median", "mean",
                                           "gmm1_nll", "gmm3_nll", "gmm5_nll"]),
                   selection=dict(split="val", criterion="frame ROC-AUC",
                                  seed=SELECT_SEED, n_cells=len(rows),
                                  select_seconds=round(time.time() - t0, 1)),
                   chosen=best, all_val_rows=rows, seeds=SEEDS,
                   missing_videos=sorted(missing))
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        rec_path.write_text(json.dumps(rec, indent=1))
        log(f"[frozen] wrote {rec_path} chosen={best}")

    # ---------------------------------------------------- final test call ---
    ch = rec["chosen"]
    units, sh, ep, st, an = (ch["units"], ch["sigma_high"], ch["epochs"],
                             ch["score_type"], ch["agg"])
    cdir = OUT / "curves" / ds
    cdir.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for si, seed in enumerate(SEEDS):
        tag = f"{ds}/{variant} final seed={seed}"
        model = None
        for _, model in train(x_tr, mean, std, units, sh, seed, device, [ep], tag=tag):
            pass
        ms_tr, _ = multiscale_scores(model, x_tr, mean, std, sh, device)
        apply = aggregate_fitted(ms_tr[st], seed)
        curves = {}
        lo, hi = np.inf, -np.inf
        for k, (vid, x) in enumerate(feats.items()):
            ms, _ = multiscale_scores(model, torch.from_numpy(x), mean, std, sh, device)
            c = apply(ms[st], an).astype(np.float64)
            curves[vid] = c
            lo, hi = min(lo, c.min()), max(hi, c.max())
            if (k + 1) % 200 == 0:
                log(f"[progress] {tag} scored={k + 1}/{len(feats)}")
        per_seed[f"{variant}_s{si}"] = curves
        log(f"[range] {tag} score min={lo:.4f} max={hi:.4f}")
        del model
        torch.cuda.empty_cache()

    for vid in feats:
        np.savez(cdir / f"{vid}.npz", rate=np.float64(FPS),
                 **{k: per_seed[k][vid] for k in per_seed})
    log(f"[written] {len(feats)} curves -> {cdir}")

    meta = OUT / f"run_meta_{ds}_{variant}.json"
    meta.write_text(json.dumps(dict(
        dataset=ds, variant=variant, chosen=ch, seeds=SEEDS,
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip(),
        upstream_commit="f821b965", torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        n_curves=len(feats), missing=sorted(missing),
        finished=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="HateMM,MHC,MHC_zh,HateClipSeg")
    ap.add_argument("--variants", default="clipL336,w2vemo,clip_w2vemo")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem-frac", type=float, default=0.10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)
    for variant in a.variants.split(","):
        for ds in a.datasets.split(","):
            try:
                run(ds, variant, a.device, a.force)
            except Exception as e:  # one dataset failing must not sink the rest
                import traceback
                log(f"[error] {ds}/{variant}: {type(e).__name__}: {e}")
                traceback.print_exc()
    log("[done]")


if __name__ == "__main__":
    main()
