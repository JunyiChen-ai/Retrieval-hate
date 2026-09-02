"""One training trial of the verdict-scaffolded boundary-contrast MIL.

Backbone, losses, optimisers, EMA self-distillation and the five-crop
inference are MACIL-SD's (scripts/reproduction_baselines/macilsd). This file
adds: the extended ``a`` stream (VGGish ⊕ BERT ⊕ scaffold), the SniCo boundary
contrast (src/snico.py), checkpoint selection on the official validation
split with the shared evaluator, and test scoring through the shared
evaluator.

    python experiments/20260902_verdict_boundary_contrast_mil/train.py \
        --corpus hatemm --seed 234 --out-dir runs/.../trial0 --config cfg.json

``--config`` is a JSON object of hyperparameters; keys missing from it take
the defaults in DEFAULTS. ``--ablation`` selects the pre-registered ablations
(full / no_snico / input_only / no_scaffold / no_scaffold_no_snico / no_k4).

Revision 4 (2026-09-02): the scaffold carries two verdict granularities,
K = 30 and K = 4 windows per video (src/vlm_verdict.GRANULARITIES); the
logit prior reads both (initialised as prior_scale times the mean level).
``no_k4`` zeroes the K = 4 channels and equals revision 3.

Design revision 2 (2026-09-02): the frozen-VLM verdict enters the frame logit
as an explicit prior, ``av_logit = MACIL_frame_logit + prior(verdict)``, with
``prior`` a learned linear map of the verdict channels (one-hot + level) that
starts as ``prior_scale * (level/3 - 1/2)``. The bag logit, the SniCo
actionness and the inference score all use this combined logit. Revision 1
only concatenated the verdict into the ``a`` input stream; on HateClipSeg
seed 234 trial 0 its output had correlation .003 with the verdict channel
and scored below the verdict alone (README section 3.1). Revision 1 is kept
as the ``input_only`` ablation.
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
from hate_common import runtime                        # noqa: E402
from macilsd import align                              # noqa: E402
from macilsd.avce_network import AVCE_Model, Single_Model  # noqa: E402
from macilsd.CMA_MIL import CMAL                       # noqa: E402
from macilsd.train import distil_step, _seq_len_of     # noqa: E402
import frame_eval_common as fec                        # noqa: E402
import vlm_verdict                                     # noqa: E402
from snico import snico_loss                           # noqa: E402
import dataset as ds                                   # noqa: E402

EVALUATOR = os.path.join(REPO_ROOT, "scripts", "reproduction_baselines",
                         "eval_baseline_scores.py")

DEFAULTS = {
    # MACIL-SD
    "hid_dim": 128, "ffn_dim": 128, "nhead": 4, "dropout": 0.2,
    "num_classes": 1, "lr": 4e-4, "batch_size": 32, "max_epoch": 50,
    "max_seqlen": 200, "sched_tmax": 60, "single_lr_scale": 0.2,
    "m": 0.91, "ema_epochs": 50, "lamda_a2b": 1.0, "lamda_a2n": 1.0,
    "lamda_cof": 0.05, "crop_repeat": 5, "fix_rep_swap": False,
    # SniCo
    "lambda_snico": 0.5, "rho": 0.5, "morph_m": 4, "tau": 0.1,
    "snico_warmup_epochs": 2,
    # verdict prior on the frame logit (revision 2)
    "prior_scale": 4.0,
    # bag = mean of the top ceil(T / topk_div) frame logits (MACIL-SD: 16);
    # SniCo mask source: "combined" (prior + content logit) or "content"
    "topk_div": 16, "snico_mask": "combined",
    # prior input: "verdict" = one-hot + level (5 dims); "scaffold" = the full
    # 7-dim scaffold including the two position channels (weights start at 0)
    "prior_dims": "verdict",
}
ABLATIONS = ("full", "no_snico", "input_only", "no_scaffold",
             "no_scaffold_no_snico", "no_k4")
SCAF_OFFSET = align.A_DIM + ds.TEXT_DIM  # first scaffold column in f_a
N_PRIOR_IN = vlm_verdict.VERDICT_DIMS    # (one-hot(4) + level/3) per K
LEVEL_COLS = [g * (vlm_verdict.N_LEVELS + 1) + vlm_verdict.N_LEVELS
              for g in range(len(vlm_verdict.GRANULARITIES))]


class Args(dict):
    __getattr__ = dict.__getitem__


class Candidate(nn.Module):
    """MACIL-SD audio-visual model plus the SniCo projection head."""

    def __init__(self, cfg, use_prior=True, n_gran=len(LEVEL_COLS)):
        super().__init__()
        self.av = AVCE_Model(cfg)
        self.proj = nn.Linear(cfg.hid_dim, cfg.hid_dim)
        self.use_prior = bool(use_prior)
        self.n_prior_in = (vlm_verdict.SCAFFOLD_DIM
                           if cfg.prior_dims == "scaffold" else N_PRIOR_IN)
        self.prior = nn.Linear(self.n_prior_in, 1)
        with torch.no_grad():
            # init: prior_scale * (mean over granularities of level/3 - 1/2)
            # (n_gran = number of active granularities; no_k4 -> 1 = rev 3)
            self.prior.weight.zero_()
            for c in LEVEL_COLS[:n_gran]:
                self.prior.weight[0, c] = float(cfg.prior_scale) / n_gran
            self.prior.bias.fill_(-0.5 * float(cfg.prior_scale))

        self.topk_div = int(cfg.topk_div)

    def bag(self, av_log, seq_len):
        """MACIL-SD's clas() with the top-k divisor as a hyperparameter."""
        if self.topk_div == 16:
            return self.av.att_mmil.clas(av_log, seq_len)
        logits = av_log.squeeze(-1)
        out = []
        for i in range(logits.shape[0]):
            if seq_len is None:
                out.append(logits[i].mean().view(1))
            else:
                t = int(seq_len[i])
                k = max(1, int(-(-t // self.topk_div)))
                out.append(torch.topk(logits[i][:t], k=k).values.mean().view(1))
        return torch.sigmoid(torch.cat(out))

    def forward(self, f_a, f_v, seq_len):
        out = self.av(f_a, f_v, seq_len)
        mmil, a_log, v_log, av_log, v_out, a_out = out
        content_log = av_log
        if self.use_prior:
            scaf = f_a[..., SCAF_OFFSET:SCAF_OFFSET + self.n_prior_in]
            av_log = av_log + self.prior(scaf)
        if self.use_prior or self.topk_div != 16:
            mmil = self.bag(av_log, seq_len)
        emb = F.normalize(self.proj(a_out + v_out), dim=-1)
        self.last_content_logit = content_log
        return mmil, a_log, v_log, av_log, v_out, a_out, emb


def _scalar(x):
    return float(x.detach()) if torch.is_tensor(x) else float(x)


def _git_describe():
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%cd %s", "--date=short"],
            cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def usable(corpus, ids):
    return [v for v in ids if align.has_features(corpus, v)]


def score_split(model, loader, device):
    """video_id -> scores on the 1 fps grid (five-crop mean of sigmoid(av))."""
    model.eval()
    out = {}
    with torch.no_grad():
        for f_v, f_a, index_map, n_seconds, vid in loader:
            vid = vid[0]
            n_seconds = int(n_seconds)
            index_map = index_map[0].numpy()
            f_v = f_v[0].to(device)
            f_a = f_a[0].to(device)
            _, _, _, av_logits, _, _, _ = model(f_a, f_v, seq_len=None)
            av = torch.sigmoid(av_logits.squeeze(-1)).mean(0).cpu().numpy()
            s = np.asarray(av, dtype=np.float64)[index_map]
            if s.shape[0] != n_seconds:
                raise RuntimeError("%s: %d rows for %d seconds"
                                   % (vid, s.shape[0], n_seconds))
            out[vid] = s
    model.train()
    return out


def frame_metrics(scores, gt, hate_ids):
    per_video = {v: (scores[v], np.asarray(gt[v])) for v in scores if v in gt}
    res = fec.evaluate(per_video, macro_over={v for v in per_video
                                              if v in hate_ids})
    return {"pooled_ap": res["pr_auc"], "pooled_roc": res["roc_auc"],
            "within_roc": res["per_video"]["macro_auc"],
            "n_videos": res["n_videos"]}


def write_scores(path, scores):
    with open(path, "w") as fh:
        for vid in sorted(scores):
            fh.write(json.dumps({"video_id": vid,
                                 "n_frames": int(len(scores[vid])),
                                 "score_av": [round(float(x), 6)
                                              for x in scores[vid]]}) + "\n")


def run_evaluator(corpus, split, scores_path, json_out):
    cmd = [sys.executable, EVALUATOR, "--corpus", corpus, "--split", split,
           "--scores", scores_path, "--json-out", json_out]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    with open(json_out) as fh:
        return json.load(fh)


def train(corpus, seed, out_dir, cfg, ablation, device, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "run.log"), "a")

    def say(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    say("host %s | corpus %s | seed %d | ablation %s | code: %s"
        % (socket.gethostname(), corpus, seed, ablation, _git_describe()))
    with open(os.path.join(out_dir, "run.pid"), "w") as fh:
        fh.write(str(os.getpid()))
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({"corpus": corpus, "seed": seed, "ablation": ablation,
                   "hparams": cfg, "device": device}, fh, indent=2)

    runtime.setup_seed(seed)
    labels = hdata.load_labels(corpus)
    train_ids = usable(corpus, hdata.load_split(corpus, "train"))
    val_gt = hdata.gt_arrays(corpus, "val")
    test_gt = hdata.gt_arrays(corpus, "test")
    val_ids = [v for v in usable(corpus, hdata.load_split(corpus, "val"))
               if v in val_gt]
    test_ids = [v for v in usable(corpus, hdata.load_split(corpus, "test"))
                if v in test_gt]
    hate_ids = {v for v, l in labels.items() if l == 1}

    use_scaffold = ablation in ("full", "no_snico", "input_only", "no_k4")
    use_prior = ablation in ("full", "no_snico", "no_k4")
    use_snico = ablation in ("full", "input_only", "no_scaffold", "no_k4")
    verdicts = [vlm_verdict.load_verdicts(corpus, k=k, tag="qwen")
                for k in vlm_verdict.GRANULARITIES]
    if not use_scaffold:
        verdicts = [{} for _ in verdicts]
    if ablation == "no_k4":
        verdicts = [verdicts[0]] + [{} for _ in verdicts[1:]]
    cache = ds.ScaffoldCache(corpus, train_ids + val_ids + test_ids, verdicts)
    n_verdict_needed = 1 if ablation == "no_k4" else len(verdicts)
    if not use_scaffold:
        # Drop the two position channels as well: the whole scaffold is off.
        for vid, (f_a, n, snip) in cache.items.items():
            f_a[:, align.A_DIM + ds.TEXT_DIM:] = 0.0
    say("train/val/test %d/%d/%d videos (%d hateful in train); missing text %d,"
        " missing verdict %d; scaffold %s, prior %s, snico %s"
        % (len(train_ids), len(val_ids), len(test_ids),
           sum(labels[v] for v in train_ids), cache.n_missing_text,
           cache.n_missing_verdict, use_scaffold, use_prior, use_snico))
    if use_scaffold:
        for g, vd in enumerate(verdicts[:n_verdict_needed]):
            cov = {name: sum(v in vd for v in ids)
                   for name, ids in (("train", train_ids), ("val", val_ids),
                                     ("test", test_ids))}
            say("verdict K%d coverage: train %d/%d, val %d/%d, test %d/%d"
                % (vlm_verdict.GRANULARITIES[g], cov["train"], len(train_ids),
                   cov["val"], len(val_ids), cov["test"], len(test_ids)))
        if sum(cache.n_missing_by_gran[:n_verdict_needed]) > 0:
            say("ABORT: missing verdicts per granularity %s; with a partial"
                " cache the scaffold channel would leak the video label on"
                " train" % cache.n_missing_by_gran)
            log.close()
            raise SystemExit(3)

    a = Args(cfg)
    a["a_feature_size"] = ds.A_EXT_DIM
    a["v_feature_size"] = align.V_DIM
    train_set = ds.TrainDataset(corpus, train_ids, labels, cache,
                                a.max_seqlen, a.crop_repeat)
    train_loader = DataLoader(train_set, batch_size=a.batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(ds.EvalDataset(corpus, val_ids, cache),
                            batch_size=1, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(ds.EvalDataset(corpus, test_ids, cache),
                             batch_size=1, shuffle=False, num_workers=num_workers)

    model = Candidate(a, use_prior=use_prior,
                      n_gran=n_verdict_needed).to(device)
    partner = Single_Model(a, n_dim=align.V_DIM).to(device)
    criterion = nn.BCELoss()
    opt_av = optim.Adam(model.parameters(), lr=a.lr)
    opt_uni = optim.Adam(partner.parameters(), lr=a.lr * a.single_lr_scale)
    sched_av = optim.lr_scheduler.CosineAnnealingLR(opt_av, T_max=a.sched_tmax)
    sched_uni = optim.lr_scheduler.CosineAnnealingLR(opt_uni, T_max=a.sched_tmax)
    say("parameters %.3fM" % ((sum(p.numel() for p in model.parameters())
                               + sum(p.numel() for p in partner.parameters()))
                              / 1e6))

    best, best_state, best_epoch, history = -1.0, None, -1, []
    for epoch in range(a.max_epoch):
        t0 = time.time()
        model.train()
        partner.train()
        lam_a2b = min(a.lamda_a2b, a.lamda_cof * epoch)
        lam_a2n = min(a.lamda_a2n, a.lamda_cof * epoch)
        lam_snico = (a.lambda_snico if (use_snico and
                                        epoch >= a.snico_warmup_epochs)
                     else 0.0)
        tot = np.zeros(5)
        nb = 0
        for f_v, f_a, label in train_loader:
            seq_len = _seq_len_of(f_v)
            keep = int(torch.max(seq_len))
            f_v = f_v[:, :keep, :].float().to(device)
            f_a = f_a[:, :keep, :].float().to(device)
            label = label.float().to(device)
            mmil, a_log, v_log, av_log, v_out, a_out, emb = model(
                f_a, f_v, seq_len)
            if a.fix_rep_swap:
                audio_rep, visual_rep = a_out, v_out
            else:
                audio_rep, visual_rep = v_out, a_out
            a_log = a_log.squeeze(-1)
            v_log = v_log.squeeze(-1)
            mmil = mmil.reshape(-1)
            clsloss = criterion(mmil, label)
            c1, c2, c3, c4 = CMAL(mmil, a_log, v_log, seq_len, audio_rep,
                                  visual_rep)
            total = (clsloss + lam_a2b * c1 + lam_a2b * c3
                     + lam_a2n * c2 + lam_a2n * c4)
            s_fg = s_bg = 0.0
            if lam_snico > 0:
                src = (model.last_content_logit if a.snico_mask == "content"
                       else av_log)
                act = torch.sigmoid(src.squeeze(-1))
                s_fg, s_bg, _ = snico_loss(act, emb, seq_len, label, a.rho,
                                           int(a.morph_m), a.tau,
                                           k_fn=lambda t: max(
                                               1, int(-(-t // a.topk_div))))
                total = total + lam_snico * (s_fg + s_bg)
            uni = partner(f_v, seq_len).reshape(-1)
            loss_uni = criterion(uni, label)
            opt_av.zero_grad()
            opt_uni.zero_grad()
            total.backward()
            opt_av.step()
            opt_av.zero_grad()
            opt_uni.zero_grad()
            loss_uni.backward()
            opt_uni.step()
            tot += [_scalar(clsloss), _scalar(c1) + _scalar(c2) + _scalar(c3)
                    + _scalar(c4), _scalar(s_fg), _scalar(s_bg),
                    _scalar(loss_uni)]
            nb += 1
        sched_av.step()
        sched_uni.step()
        m = distil_step(a, model.av, partner, epoch)
        tot /= max(nb, 1)
        val_scores = score_split(model, val_loader, device)
        vm = frame_metrics(val_scores, val_gt, hate_ids)
        crit = (vm["pooled_ap"] + vm["pooled_roc"]) / 2.0
        history.append({"epoch": epoch + 1, "cls": tot[0], "cma": tot[1],
                        "snico_fg": tot[2], "snico_bg": tot[3],
                        "uni": tot[4], "lam_snico": lam_snico, "ema_m": m,
                        "val": vm, "val_criterion": crit,
                        "seconds": round(time.time() - t0, 1)})
        say("epoch %2d | cls %.4f | cma %.4f | snico %.4f %.4f | uni %.4f | "
            "val AP %.4f ROC %.4f within %.4f | %.0fs"
            % (epoch + 1, tot[0], tot[1], tot[2], tot[3], tot[4],
               vm["pooled_ap"], vm["pooled_roc"], vm["within_roc"],
               time.time() - t0))
        if crit > best:
            best, best_epoch = crit, epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    say("selected epoch %d (val criterion %.4f)" % (best_epoch, best))
    torch.save(best_state, os.path.join(out_dir, "model.pth"))

    val_scores = score_split(model, val_loader, device)
    write_scores(os.path.join(out_dir, "scores_val.jsonl"), val_scores)
    val_eval = run_evaluator(corpus, "val",
                             os.path.join(out_dir, "scores_val.jsonl"),
                             os.path.join(out_dir, "metrics_val.json"))
    test_scores = score_split(model, test_loader, device)
    write_scores(os.path.join(out_dir, "scores_test.jsonl"), test_scores)
    test_eval = run_evaluator(corpus, "test",
                              os.path.join(out_dir, "scores_test.jsonl"),
                              os.path.join(out_dir, "metrics.json"))

    def pick(ev):
        r = ev["results"]["score_av"]
        return {"pooled_ap": r["pr_auc"], "pooled_roc": r["roc_auc"],
                "within_roc": r["per_video"]["macro_auc"],
                "n_videos": r["n_videos"]}

    summary = {"corpus": corpus, "seed": seed, "ablation": ablation,
               "selected_epoch": best_epoch, "val_criterion": best,
               "val": pick(val_eval), "test": pick(test_eval),
               "history": history, "hparams": cfg,
               "host": socket.gethostname()}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    say("TEST pooled AP %.4f | pooled ROC %.4f | within %.4f"
        % (summary["test"]["pooled_ap"], summary["test"]["pooled_roc"],
           summary["test"]["within_roc"]))
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
    train(args.corpus, args.seed, args.out_dir, cfg, args.ablation,
          args.device, args.num_workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
