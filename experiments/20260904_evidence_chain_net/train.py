"""Train / evaluate the evidence-chain network for one (corpus, seed, config, ablation).

    python experiments/20260904_evidence_chain_net/train.py --corpus hatemm --seed 234 \
        --out-dir runs/20260904_evidence_chain_net/hatemm/seed234/trial0 \
        --config <hparams.json> [--ablation full]

Protocol (RESEARCH_ITERATION_RULES.md rules 7/8): full training every time (no
shortened runs), checkpoint chosen on validation (AP+ROC)/2, test scored once
with the shared evaluator scripts/reproduction_baselines/eval_baseline_scores.py.
Writes config.json, hmm_params.json, potentials.json, run.log, run.pid, model.pth,
scores_{val,test}.jsonl, metrics{,_val}.json, summary.json.

Losses (README section 1, all weights fixed constants):
  1. video-level marginal likelihood of the chain, in the log domain:
     -[y log P(y=1) + (1-y) log P(y=0)], P(y=0) = Z0/Z
  2. verdict-block MIL on the content evidence u_t (the mechanism confirmed in
     the previous candidate): per block, top-ceil(n_j/16) mean of u_t, soft
     target = fixed evidence-model block posterior (0 for negative videos),
     weight |2p-1|, BCE-with-logits
  3. cross-modal contrast (MACIL-SD's CMAL pairing kept: visual query vs
     audio-text keys and vice versa; only the selector changes: chain posterior
     top-k / bottom-k instead of the model's own bag score), InfoNCE tau .1,
     weight ramps min(1, epoch/10)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "duplex"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, HERE)

from hate_common import data as hdata                  # noqa: E402
from macilsd import align                              # noqa: E402
import frame_eval_common as fec                        # noqa: E402
import vlm_verdict                                     # noqa: E402
import verdict_hmm                                     # noqa: E402
import dataset as ds                                   # noqa: E402
from model import EvidenceChainNet, ABLATIONS, MODEL_ABLATIONS   # noqa: E402

EVALUATOR = os.path.join(REPO_ROOT, "scripts", "reproduction_baselines",
                         "eval_baseline_scores.py")
K_FINE, J_COARSE = ds.K, ds.J

DEFAULTS = {
    "hid_dim": 128, "ffn_dim": 128, "nhead": 4, "dropout": 0.2,
    "lr": 5e-4, "batch_size": 32, "max_epoch": 50, "max_seqlen": 200,
    "sched_tmax": 60, "crop_repeat": 5,
    "block_weight": 1.0, "topk_div": 16,
    "contrast_weight": 1.0, "contrast_ramp_epochs": 10, "contrast_tau": 0.1,
    "contrast_max_normal": 256,
    # MACIL-SD AVCE fields (macilsd_encoder ablation only)
    "num_classes": 1, "a_feature_size": ds.F_A_DIM, "v_feature_size": ds.V_DIM,
}


class Args(dict):
    __getattr__ = dict.__getitem__


def git_line():
    try:
        return subprocess.check_output(["git", "log", "-1", "--format=%cd %s"],
                                       cwd=REPO_ROOT, text=True).strip()
    except Exception:            # noqa: BLE001
        return "unknown"


def usable(corpus, ids):
    return [v for v in ids if align.has_features(corpus, v)]


def fit_hmm(train_ids, labels, binary):
    pos = [binary[v] for v in train_ids if labels[v] == 1 and v in binary]
    neg = [binary[v] for v in train_ids if labels[v] == 0 and v in binary]
    return verdict_hmm.HierEvidenceHMM(K_FINE, J_COARSE).fit(pos, neg), len(pos), len(neg)


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ----------------------------------------------------------------- losses
def video_loss(out, y):
    return -(y * out["log_p_video"] + (1.0 - y) * out["log_rho"]).mean()


def block_mil_loss(out, batch, topk_div):
    """Per-block top-k MIL on the content evidence u_t; target = fixed block posterior
    (0 for negative videos), weight |2p-1|."""
    u, mask, y, j, ph = out["u"], batch["mask"], batch["label"], batch["j"], batch["ph"]
    losses, weights = [], []
    for i in range(u.shape[0]):
        m = mask[i]
        for b in torch.unique(j[i][m]):
            sel = (j[i] == b) & m
            n = int(sel.sum())
            k = max(1, int(math.ceil(n / topk_div)))
            bag = torch.topk(u[i][sel], k=k).values.mean()
            p = ph[i][sel][0] if y[i] > 0.5 else torch.zeros((), device=u.device)
            losses.append(F.binary_cross_entropy_with_logits(bag, p))
            weights.append((2 * p - 1).abs())
    if not losses:
        return u.sum() * 0.0
    L, W = torch.stack(losses), torch.stack(weights)
    return (L * W).sum() / W.sum().clamp(min=1e-6)


def _select_topk(sel, m, k):
    top = torch.topk(sel.masked_fill(~m, -1e9), k=k).indices
    bot = torch.topk(sel.masked_fill(~m, 1e9), k=k, largest=False).indices
    return top, bot


def _infonce(query, pos, neg, tau):
    q = F.normalize(query, dim=0)
    lp = (F.normalize(pos, dim=-1) @ q) / tau
    ln = (F.normalize(neg, dim=-1) @ q) / tau
    lse_neg = torch.logsumexp(ln, 0)
    return (torch.logaddexp(lp, lse_neg) - lp).mean()


def contrast_loss(out, batch, mode, tau, topk_div, max_normal):
    """CMAL pairing with the selector replaced (README section 1, objective 3)."""
    hv, hat, mask, y = out["hv"], out["hat"], batch["mask"], batch["label"]
    if mode == "posterior":
        sel = out["post"].detach()
    elif mode == "self_topk":
        sel = torch.sigmoid(out["u"].detach())
    elif mode == "vlm_thresh":
        sel = batch["bf"]
    else:
        raise ValueError(mode)
    B = hv.shape[0]
    neg_ids = [i for i in range(B) if y[i] < 0.5]
    normal_v = torch.cat([hv[i][mask[i]] for i in neg_ids], 0) if neg_ids else hv[:0, 0]
    normal_at = torch.cat([hat[i][mask[i]] for i in neg_ids], 0) if neg_ids else hat[:0, 0]
    if normal_v.shape[0] > max_normal:
        keep = torch.randperm(normal_v.shape[0], device=hv.device)[:max_normal]
        normal_v, normal_at = normal_v[keep], normal_at[keep]
    losses = []
    for i in range(B):
        if y[i] < 0.5:
            continue
        m = mask[i]
        t = int(m.sum())
        k = max(1, int(math.ceil(t / topk_div)))
        top, bot = _select_topk(sel[i], m, k)
        losses.append(_infonce(hv[i][top].mean(0), hat[i][top],
                               torch.cat([hat[i][bot], normal_at], 0), tau))
        losses.append(_infonce(hat[i][top].mean(0), hv[i][top],
                               torch.cat([hv[i][bot], normal_v], 0), tau))
    if not losses:
        return hv.sum() * 0.0
    return torch.stack(losses).mean()


# ------------------------------------------------------------- evaluation
def eval_batch(item, device):
    """EvalDataset item (batch size 1, crops stacked) -> model batch with B = crops."""
    f_v = item["f_v"][0].to(device)                       # [5,T,1024]
    n = f_v.shape[0]
    b = {"f_v": f_v}
    for k in ("f_a", "mask", "profile") + ds.ROW_KEYS + ds.IDX_KEYS:
        v = item[k][0].to(device)
        b[k] = v.unsqueeze(0).expand(n, *v.shape).contiguous()
    return b


def score_split(model, loader, device, collect=None):
    """video_id -> scores on the 1 fps grid (five-crop mean of the log-odds)."""
    model.eval()
    out = {}
    with torch.no_grad():
        for item in loader:
            vid = item["vid"][0]
            n_seconds = int(item["n_seconds"])
            index_map = item["index_map"][0].numpy()
            b = eval_batch(item, device)
            o = model(b)
            score = o["score"].mean(0).cpu().numpy()
            s = np.asarray(score, dtype=np.float64)[index_map]
            if s.shape[0] != n_seconds:
                raise RuntimeError("%s: %d rows for %d seconds" % (vid, s.shape[0], n_seconds))
            out[vid] = s
            if collect is not None:
                collect[vid] = {"d_v": float(o["d_v"][0]),
                                "gf": o["gf"][0].cpu().numpy(), "bf": b["bf"][0].cpu().numpy(),
                                "bc": b["bc"][0].cpu().numpy(), "mask": b["mask"][0].cpu().numpy()}
    model.train()
    return out


def frame_metrics(scores, gt, hate_ids):
    per_video = {v: (scores[v], np.asarray(gt[v])) for v in scores if v in gt}
    res = fec.evaluate(per_video, macro_over={v for v in per_video if v in hate_ids})
    return {"pooled_ap": res["pr_auc"], "pooled_roc": res["roc_auc"],
            "within_roc": res["per_video"]["macro_auc"], "n_videos": res["n_videos"]}


def write_scores(path, scores):
    with open(path, "w") as fh:
        for vid in sorted(scores):
            fh.write(json.dumps({"video_id": vid, "n_frames": int(len(scores[vid])),
                                 "score_chain": [round(float(x), 6) for x in scores[vid]]}) + "\n")


def run_evaluator(corpus, split, scores_path, json_out):
    subprocess.run([sys.executable, EVALUATOR, "--corpus", corpus, "--split", split,
                    "--scores", scores_path, "--json-out", json_out],
                   check=True, cwd=REPO_ROOT, stdout=subprocess.DEVNULL)
    r = json.load(open(json_out))["results"]["score_chain"]
    return {"pooled_ap": r["pr_auc"], "pooled_roc": r["roc_auc"],
            "within_roc": r["per_video"]["macro_auc"]}


def diagnostics(collect, labels, gt):
    """Gate means per verdict cell; density vs GT density; density saturation."""
    cells = {c: [] for c in ("bf0_bc0", "bf0_bc1", "bf1_bc0", "bf1_bc1")}
    dv, dens = [], []
    for vid, c in collect.items():
        m = c["mask"].astype(bool)
        for name in cells:
            sel = (c["bf"] == int(name[2])) & (c["bc"] == int(name[6])) & m
            if sel.any():
                cells[name].append(float(c["gf"][sel].mean()))
        if vid in gt:
            dv.append(c["d_v"])
            dens.append(float(np.mean(gt[vid])))
    dv, dens = np.array(dv), np.array(dens)
    return {"gate_fine_mean_per_cell": {k: (float(np.mean(v)) if v else None) for k, v in cells.items()},
            "density_corr_with_gt_density": float(np.corrcoef(dv, dens)[0, 1]) if len(dv) > 2 else None,
            "density_saturated_frac": float(np.mean((dv < 0.02) | (dv > 0.98))) if len(dv) else None,
            "density_mean_by_label": {
                "hate": float(np.mean([c["d_v"] for v, c in collect.items() if labels[v] == 1] or [0])),
                "nonhate": float(np.mean([c["d_v"] for v, c in collect.items() if labels[v] == 0] or [0]))}}


def _scalar(x):
    return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)


# ------------------------------------------------------------------ train
def train(corpus, seed, out_dir, cfg, ablation, device, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "run.log"), "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say("host %s | corpus %s | seed %d | ablation %s | code: %s"
        % (socket.gethostname(), corpus, seed, ablation, git_line()))
    with open(os.path.join(out_dir, "run.pid"), "w") as fh:
        fh.write(str(os.getpid()))
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({"corpus": corpus, "seed": seed, "ablation": ablation, "hparams": cfg,
                   "host": socket.gethostname(), "code": git_line()}, fh, indent=2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    labels = hdata.load_labels(corpus)
    val_gt = hdata.gt_arrays(corpus, "val")
    test_gt = hdata.gt_arrays(corpus, "test")
    train_ids = usable(corpus, hdata.load_split(corpus, "train"))
    val_ids = [v for v in usable(corpus, hdata.load_split(corpus, "val")) if v in val_gt]
    test_ids = [v for v in usable(corpus, hdata.load_split(corpus, "test")) if v in test_gt]
    hate_ids = {v for v, l in labels.items() if l == 1}

    V = {k: vlm_verdict.load_verdicts(corpus, k=k, tag="qwen") for k in (K_FINE, J_COARSE)}
    binary = {v: (verdict_hmm.binarize(V[K_FINE][v]), verdict_hmm.binarize(V[J_COARSE][v]))
              for v in V[K_FINE] if v in V[J_COARSE]}
    hmm, n_pos, n_neg = fit_hmm(train_ids, labels, binary)
    hmm.save(os.path.join(out_dir, "hmm_params.json"))
    pot = ds.Potentials(hmm)
    with open(os.path.join(out_dir, "potentials.json"), "w") as fh:
        json.dump(pot.as_dict(), fh, indent=2)
    say("evidence model fitted on %d positive / %d negative train videos: %s"
        % (n_pos, n_neg, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in pot.as_dict().items()})))
    caches = {name: ds.VideoCache(corpus, ids, binary, pot)
              for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))}
    missing = {n: len(c.missing_verdict) for n, c in caches.items()}
    say("train/val/test %d/%d/%d videos (%d hateful in train); missing text %d; "
        "missing verdicts %s; ablation %s"
        % (len(train_ids), len(val_ids), len(test_ids), sum(labels[v] for v in train_ids),
           sum(c.n_missing_text for c in caches.values()), json.dumps(missing), ablation))
    if any(missing.values()):
        say("ABORT: videos without verdicts (a partial verdict set would leak the label)")
        log.close()
        raise SystemExit(3)

    a = Args(cfg)
    train_set = ds.TrainDataset(corpus, train_ids, labels, caches["train"], a.max_seqlen,
                                a.crop_repeat)
    train_loader = DataLoader(train_set, batch_size=a.batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(ds.EvalDataset(corpus, val_ids, caches["val"]), batch_size=1,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(ds.EvalDataset(corpus, test_ids, caches["test"]), batch_size=1,
                             shuffle=False, num_workers=num_workers)

    model_abl = ablation if ablation in MODEL_ABLATIONS else "full"
    model = EvidenceChainNet(a, pot, model_abl).to(device)
    contrast_mode = {"no_contrast": None, "contrast_self_topk": "self_topk",
                     "contrast_vlm_thresh": "vlm_thresh"}.get(ablation, "posterior")
    block_weight = 0.0 if ablation == "no_block_mil" else float(a.block_weight)
    opt = optim.Adam(model.parameters(), lr=a.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.sched_tmax)
    say("model params %d | model ablation %s | contrast %s | block weight %.1f | "
        "window switching rate a %.4f | p0_hate %.4f"
        % (sum(p.numel() for p in model.parameters()), model_abl, contrast_mode,
           block_weight, pot.a, pot.p0_hate))

    best, best_state, best_epoch, history = -1.0, None, -1, []
    for epoch in range(a.max_epoch):
        t0 = time.time()
        model.train()
        lam_c = a.contrast_weight * min(1.0, epoch / float(a.contrast_ramp_epochs))
        tot = np.zeros(4)
        nb = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            out = model(batch)
            y = batch["label"]
            lv = video_loss(out, y)
            lb = block_mil_loss(out, batch, a.topk_div) if block_weight > 0 \
                else torch.zeros((), device=device)
            lc = contrast_loss(out, batch, contrast_mode, a.contrast_tau, a.topk_div,
                               a.contrast_max_normal) if (contrast_mode and lam_c > 0) \
                else torch.zeros((), device=device)
            total = lv + block_weight * lb + lam_c * lc
            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += [_scalar(lv), _scalar(lb), _scalar(lc), _scalar(out["d_v"].mean())]
            nb += 1
        sched.step()
        tot /= max(nb, 1)
        val_scores = score_split(model, val_loader, device)
        vm = frame_metrics(val_scores, val_gt, hate_ids)
        crit = (vm["pooled_ap"] + vm["pooled_roc"]) / 2.0
        history.append({"epoch": epoch + 1, "video": tot[0], "block": tot[1],
                        "contrast": tot[2], "d_v_mean": tot[3], "val": vm,
                        "val_criterion": crit, "seconds": round(time.time() - t0, 1)})
        say("epoch %2d | video %.4f | block %.4f | contrast %.4f | d_v %.3f | "
            "val AP %.4f ROC %.4f within %.4f | %.0fs"
            % (epoch + 1, tot[0], tot[1], tot[2], tot[3], vm["pooled_ap"], vm["pooled_roc"],
               vm["within_roc"], time.time() - t0))
        if crit > best:
            best, best_epoch = crit, epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    say("selected epoch %d (val criterion %.4f)" % (best_epoch, best))
    torch.save(best_state, os.path.join(out_dir, "model.pth"))

    val_scores = score_split(model, val_loader, device)
    write_scores(os.path.join(out_dir, "scores_val.jsonl"), val_scores)
    vm = run_evaluator(corpus, "val", os.path.join(out_dir, "scores_val.jsonl"),
                       os.path.join(out_dir, "metrics_val.json"))
    collect = {}
    test_scores = score_split(model, test_loader, device, collect)
    write_scores(os.path.join(out_dir, "scores_test.jsonl"), test_scores)
    tm = run_evaluator(corpus, "test", os.path.join(out_dir, "scores_test.jsonl"),
                       os.path.join(out_dir, "metrics.json"))
    diag = diagnostics(collect, labels, test_gt)
    say("TEST pooled AP %.4f | pooled ROC %.4f | within %.4f"
        % (tm["pooled_ap"], tm["pooled_roc"], tm["within_roc"]))
    say("test diagnostics %s" % json.dumps(diag))
    summary = {"corpus": corpus, "seed": seed, "ablation": ablation, "hparams": cfg,
               "selected_epoch": best_epoch, "val_criterion": best, "val": vm, "test": tm,
               "test_diagnostics": diag, "potentials": pot.as_dict(), "history": history,
               "host": socket.gethostname(), "code": git_line()}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    log.close()
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None, help="JSON file of hyperparameters")
    ap.add_argument("--ablation", default="full", choices=ABLATIONS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args(argv)
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as fh:
            cfg.update(json.load(fh))
    train(args.corpus, args.seed, args.out_dir, cfg, args.ablation, args.device, args.num_workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
