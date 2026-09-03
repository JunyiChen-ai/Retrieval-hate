"""Train / evaluate the evidence-chain network for one (corpus, seed, config, ablation).

    python experiments/20260904_evidence_chain_net/train.py --corpus hatemm --seed 234 \
        --out-dir runs/20260904_evidence_chain_net/hatemm/seed234/trial0 \
        --config <hparams.json> [--ablation full]

Protocol (RESEARCH_ITERATION_RULES.md rules 7/8): full training every time (no
shortened runs), checkpoint chosen on validation (AP+ROC)/2, test scored once
with the shared evaluator scripts/reproduction_baselines/eval_baseline_scores.py.
Writes config.json, hmm_params.json, potentials.json, run.log, run.pid, model.pth,
scores_{val,test}.jsonl, metrics{,_val}.json, summary.json.

Losses: BCE on the chain's video-level marginal P(y=1) = 1 - Z0/Z, plus the
posterior-guided within-video contrastive term (weight 1, temperature .1).
No CMAL, no EMA partner, no block MIL, no top-k (except the topk_head ablation).
"""

from __future__ import annotations

import argparse
import copy
import json
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
from model import EvidenceChainNet, ABLATIONS          # noqa: E402

MODEL_ABLATIONS = ABLATIONS[:11]          # the rest only change the contrastive term

EVALUATOR = os.path.join(REPO_ROOT, "scripts", "reproduction_baselines",
                         "eval_baseline_scores.py")
K_FINE, J_COARSE = ds.K, ds.J

DEFAULTS = {
    "hid_dim": 128, "ffn_dim": 128, "nhead": 4, "dropout": 0.2,
    "lr": 5e-4, "batch_size": 32, "max_epoch": 50, "max_seqlen": 200,
    "sched_tmax": 60, "crop_repeat": 5,
    "contrast_weight": 1.0, "contrast_tau": 0.1, "contrast_max_normal": 256,
    "topk_div": 16,                       # topk_head ablation only
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


def contrast_loss(out, batch, mode, tau, max_normal):
    """Posterior-guided within-video InfoNCE (README section 1, training objective 2)."""
    h, mask, y = out["h"], batch["mask"], batch["label"]
    if mode == "posterior":
        sel = out["post"].detach()
    elif mode == "self_topk":
        sel = torch.sigmoid(out["u"].detach())
    elif mode == "vlm_thresh":
        sel = batch["bf"]
    else:
        raise ValueError(mode)
    hn = F.normalize(h, dim=-1)
    normal = hn[(y < 0.5)[:, None] & mask]
    if normal.shape[0] > max_normal:
        normal = normal[torch.randperm(normal.shape[0], device=h.device)[:max_normal]]
    losses = []
    for i in range(h.shape[0]):
        if y[i] < 0.5:
            continue
        m = mask[i]
        s, hi = sel[i][m], hn[i][m]
        hate = s > 0.5
        if hate.sum() == 0:
            hate = s >= s.max()
        back = ~hate
        if hate.sum() == 0 or (back.sum() == 0 and normal.shape[0] == 0):
            continue
        anchor = F.normalize(hi[hate].mean(0), dim=0)
        pos = hi[hate] @ anchor / tau
        negs = torch.cat([hi[back], normal], 0) @ anchor / tau
        lse_neg = torch.logsumexp(negs, 0)
        losses.append((torch.logaddexp(pos, lse_neg) - pos).mean())
    if not losses:
        return h.sum() * 0.0
    return torch.stack(losses).mean()


def gate_stats(model, loader, device):
    """Mean fine gate per verdict cell and mean density by label-free profile, on a split."""
    model.eval()
    cells = {c: [] for c in ("bf0_bc0", "bf0_bc1", "bf1_bc0", "bf1_bc1")}
    dens = []
    with torch.no_grad():
        for item in loader:
            b = eval_batch(item, device)
            out = model(b)
            gf, bf, bc, m = out["gf"][0], b["bf"][0], b["bc"][0], b["mask"][0]
            for c in cells:
                sel = (bf == int(c[2])) & (bc == int(c[6])) & m
                if sel.any():
                    cells[c].append(float(gf[sel].mean()))
            dens.append((item["vid"][0], float(out["d_v"][0])))
    model.train()
    return {c: (float(np.mean(v)) if v else None) for c, v in cells.items()}, dict(dens)


def eval_batch(item, device):
    """EvalDataset item (batch size 1, crops stacked) -> model batch with B = crops."""
    f_v = item["f_v"][0].to(device)                       # [5,T,1024]
    n = f_v.shape[0]
    b = {"f_v": f_v}
    for k in ("f_a", "mask", "profile") + ds.ROW_KEYS + ds.IDX_KEYS:
        v = item[k][0].to(device)
        b[k] = v.unsqueeze(0).expand(n, *v.shape).contiguous()
    return b


def score_split(model, loader, device):
    """video_id -> scores on the 1 fps grid (five-crop mean posterior)."""
    model.eval()
    out = {}
    with torch.no_grad():
        for item in loader:
            vid = item["vid"][0]
            n_seconds = int(item["n_seconds"])
            index_map = item["index_map"][0].numpy()
            b = eval_batch(item, device)
            post = model(b)["post"].mean(0).cpu().numpy()
            s = np.asarray(post, dtype=np.float64)[index_map]
            if s.shape[0] != n_seconds:
                raise RuntimeError("%s: %d rows for %d seconds" % (vid, s.shape[0], n_seconds))
            out[vid] = s
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


def _scalar(x):
    return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)


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

    # module 3: evidence-model constants from train video labels only
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

    model = EvidenceChainNet(a, pot, ablation if ablation in MODEL_ABLATIONS else "full").to(device)
    contrast_mode = {"no_contrast": None, "contrast_self_topk": "self_topk",
                     "contrast_vlm_thresh": "vlm_thresh"}.get(ablation, "posterior")
    criterion = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=a.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.sched_tmax)
    say("model params %d | contrast %s | switching rate a %.4f | p0_hate %.4f"
        % (sum(p.numel() for p in model.parameters()), contrast_mode, pot.a, pot.p0_hate))

    best, best_state, best_epoch, history = -1.0, None, -1, []
    for epoch in range(a.max_epoch):
        t0 = time.time()
        model.train()
        tot = np.zeros(3)
        nb = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            out = model(batch)
            y = batch["label"]
            cls = criterion(out["p_video"], y)
            con = contrast_loss(out, batch, contrast_mode, a.contrast_tau, a.contrast_max_normal) \
                if contrast_mode else torch.zeros((), device=device)
            total = cls + a.contrast_weight * con
            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += [_scalar(cls), _scalar(con), _scalar(out["d_v"].mean())]
            nb += 1
        sched.step()
        tot /= max(nb, 1)
        val_scores = score_split(model, val_loader, device)
        vm = frame_metrics(val_scores, val_gt, hate_ids)
        crit = (vm["pooled_ap"] + vm["pooled_roc"]) / 2.0
        history.append({"epoch": epoch + 1, "cls": tot[0], "contrast": tot[1],
                        "d_v_mean": tot[2], "val": vm, "val_criterion": crit,
                        "seconds": round(time.time() - t0, 1)})
        say("epoch %2d | cls %.4f | contrast %.4f | d_v %.3f | val AP %.4f ROC %.4f within %.4f | %.0fs"
            % (epoch + 1, tot[0], tot[1], tot[2], vm["pooled_ap"], vm["pooled_roc"],
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
    test_scores = score_split(model, test_loader, device)
    write_scores(os.path.join(out_dir, "scores_test.jsonl"), test_scores)
    tm = run_evaluator(corpus, "test", os.path.join(out_dir, "scores_test.jsonl"),
                       os.path.join(out_dir, "metrics.json"))
    cells, dens = gate_stats(model, test_loader, device)
    d_by_label = {"hate": float(np.mean([d for v, d in dens.items() if labels[v] == 1] or [0])),
                  "nonhate": float(np.mean([d for v, d in dens.items() if labels[v] == 0] or [0]))}
    say("TEST pooled AP %.4f | pooled ROC %.4f | within %.4f" % (tm["pooled_ap"], tm["pooled_roc"], tm["within_roc"]))
    say("test fine-gate mean per verdict cell %s | density mean by video label %s"
        % (json.dumps({k: (round(v, 3) if v is not None else None) for k, v in cells.items()}),
           json.dumps({k: round(v, 3) for k, v in d_by_label.items()})))
    summary = {"corpus": corpus, "seed": seed, "ablation": ablation, "hparams": cfg,
               "selected_epoch": best_epoch, "val_criterion": best, "val": vm, "test": tm,
               "test_gate_cells": cells, "test_density_by_label": d_by_label,
               "potentials": pot.as_dict(), "history": history,
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
