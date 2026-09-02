"""One training trial of the hierarchical-evidence MIL candidate.

Backbone, cross-modal alignment loss, optimisers, EMA self-distillation and
the five-crop inference are MACIL-SD's (scripts/reproduction_baselines/macilsd).
This file adds:

  * module 3 (fusion): a hierarchical evidence HMM over the frozen-VLM
    verdicts (src/verdict_hmm.py), fitted on TRAIN video labels inside this
    script, whose per-row posterior log-odds ell_t enters the frame logit as a
    fixed-scale prior, z~_t = z_t + prior_scale * ell_t; bag, loss and
    inference use z~;
  * module 2 (backbone supervision): verdict-block MIL, one bag per coarse
    block with soft label P(h_j=1|b) from the HMM (exact 0 on negative
    videos), weight |2p-1|, scored on the CONTENT logit z (no prior) so the
    backbone has to learn within-video ordering from audio/visual/text;
  * checkpoint selection on the official validation split, test scoring
    through the shared evaluator.

    python experiments/20260903_hier_evidence_mil/train.py \
        --corpus hatemm --seed 234 --out-dir runs/.../trial0 --config cfg.json

``--ablation`` (pre-registered, README section 2):
  full            everything
  mean_prior      ell_t replaced by the revision-4 prior input (mean level - 1/2)
  indep_hmm       HMM posterior without temporal coupling (A -> stationary row)
  flat_coarse     coarse verdict emitted at every window (no block OR structure)
  no_block        lambda_block = 0
  raw_block_label block-bag label = raw coarse verdict b4_j instead of P(h_j)
  no_input        scaffold hidden from the backbone input (prior + block loss kept)
  no_prior        prior_scale = 0 (block loss + input kept)
  no_verdict      no verdict anywhere: prior 0, block loss 0, input hidden
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
import verdict_hmm                                     # noqa: E402
import dataset as ds                                   # noqa: E402

EVALUATOR = os.path.join(REPO_ROOT, "scripts", "reproduction_baselines",
                         "eval_baseline_scores.py")
K_FINE, J_COARSE = vlm_verdict.GRANULARITIES          # (30, 4)

DEFAULTS = {
    # MACIL-SD
    "hid_dim": 128, "ffn_dim": 128, "nhead": 4, "dropout": 0.2,
    "num_classes": 1, "lr": 4e-4, "batch_size": 32, "max_epoch": 50,
    "max_seqlen": 200, "sched_tmax": 60, "single_lr_scale": 0.2,
    "m": 0.91, "ema_epochs": 50, "lamda_a2b": 1.0, "lamda_a2n": 1.0,
    "lamda_cof": 0.05, "crop_repeat": 5, "fix_rep_swap": False,
    # module 3: fixed-scale prior on the HMM posterior log-odds; evidence
    # tempering of the fine (K=30) verdicts, w_fine in [0, 1]
    "prior_scale": 2.0, "w_fine": 1.0,
    # module 2: verdict-block MIL
    "lambda_block": 0.5, "topk_div": 16,
}
ABLATIONS = ("full", "mean_prior", "indep_hmm", "flat_coarse", "no_block",
             "raw_block_label", "no_input", "no_prior", "no_verdict")


class Args(dict):
    __getattr__ = dict.__getitem__


class Candidate(nn.Module):
    """MACIL-SD audio-visual model with the fixed HMM-posterior prior."""

    def __init__(self, cfg, prior_scale, hide_input=False):
        super().__init__()
        self.av = AVCE_Model(cfg)
        self.prior_scale = float(prior_scale)
        self.hide_input = bool(hide_input)
        self.topk_div = int(cfg.topk_div)

    def bag(self, logits, seq_len):
        """Mean of the top ceil(T / topk_div) logits per item (MACIL-SD clas)."""
        logits = logits.squeeze(-1)
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
        f_a_in = f_a.clone()
        f_a_in[..., ds.SCAF_OFFSET + ds.N_INPUT_SCAF:] = 0.0   # bookkeeping cols
        if self.hide_input:
            f_a_in[..., ds.SCAF_OFFSET:] = 0.0
        mmil, a_log, v_log, av_log, v_out, a_out = self.av(f_a_in, f_v, seq_len)
        content_log = av_log
        ell = f_a[..., ds.SCAF_OFFSET + ds.COL_ELL:ds.SCAF_OFFSET + ds.COL_ELL + 1]
        av_log = av_log + self.prior_scale * ell
        mmil = self.bag(av_log, seq_len)
        self.last_content_logit = content_log
        return mmil, a_log, v_log, av_log, v_out, a_out


def block_bag_loss(content_log, f_a, seq_len, labels, topk_div):
    """Verdict-block MIL: one bag per coarse block, label P(h_j=1) (column
    COL_PH, exact 0 on negative videos), weight |2p-1|, top-k mean of the
    content logit inside the block. Returns the weighted mean BCE."""
    z = content_log.squeeze(-1)
    blk = f_a[..., ds.SCAF_OFFSET + ds.COL_BLOCK]
    ph = f_a[..., ds.SCAF_OFFSET + ds.COL_PH]
    num = z.new_zeros(())
    den = z.new_zeros(())
    for i in range(z.shape[0]):
        t = int(seq_len[i])
        zi, bi, pi = z[i, :t], blk[i, :t], ph[i, :t]
        for j in torch.unique(bi):
            m = bi == j
            n_j = int(m.sum())
            if n_j == 0:
                continue
            k = max(1, int(-(-n_j // topk_div)))
            bag = torch.topk(zi[m], k=k).values.mean()
            p = pi[m][0] if labels[i] > 0.5 else zi.new_zeros(())
            w = (2.0 * p - 1.0).abs()
            num = num + w * nn.functional.binary_cross_entropy_with_logits(
                bag, p)
            den = den + w
    return num / den.clamp_min(1e-6)


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
    """video_id -> scores on the 1 fps grid (five-crop mean of sigmoid(z~))."""
    model.eval()
    out = {}
    with torch.no_grad():
        for f_v, f_a, index_map, n_seconds, vid in loader:
            vid = vid[0]
            n_seconds = int(n_seconds)
            index_map = index_map[0].numpy()
            f_v = f_v[0].to(device)
            f_a = f_a[0].to(device)
            _, _, _, av_logits, _, _ = model(f_a, f_v, seq_len=None)
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
    subprocess.run([sys.executable, EVALUATOR, "--corpus", corpus,
                    "--split", split, "--scores", scores_path,
                    "--json-out", json_out], check=True, cwd=REPO_ROOT,
                   stdout=subprocess.DEVNULL)
    with open(json_out) as fh:
        return json.load(fh)


def fit_hmm(corpus, train_ids, labels, binary):
    pos = [binary[v] for v in train_ids if labels[v] == 1 and v in binary]
    neg = [binary[v] for v in train_ids if labels[v] == 0 and v in binary]
    return verdict_hmm.HierEvidenceHMM(K_FINE, J_COARSE).fit(pos, neg), \
        len(pos), len(neg)


def make_scaffold_fn(hmm, binary, ablation, w_fine):
    """Per-video scaffold builder (README dataset.py column layout)."""
    block_of_window = hmm.block.astype(np.float32)
    kw = {}
    if ablation == "indep_hmm":
        kw["independent"] = True
    if ablation == "flat_coarse":
        kw["flat_coarse"] = True

    def fn(vid, snip, n_seconds):
        if vid not in binary:
            return None
        bf, bc = binary[vid]
        p_s, p_h = hmm.posterior(bf, bc, w_fine=w_fine, **kw)
        ell = np.log(p_s + 1e-6) - np.log(1.0 - p_s + 1e-6)
        if ablation == "mean_prior":
            # revision-4 prior input: mean binary level minus 1/2, per row via
            # the same window/block maps
            ell = (bf + bc[hmm.block]) / 2.0 - 0.5
        if ablation == "raw_block_label":
            p_h = bc.astype(np.float32)
        return ds.scaffold_rows(ell, p_s, bf, bc, p_h, block_of_window,
                                snip, n_seconds)
    return fn


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

    a = Args(cfg)
    prior_scale = 0.0 if ablation in ("no_prior", "no_verdict") else a.prior_scale
    lambda_block = 0.0 if ablation in ("no_block", "no_verdict") else a.lambda_block
    hide_input = ablation in ("no_input", "no_verdict")

    # module 3: verdict HMM fitted on train video labels only
    V = {k: vlm_verdict.load_verdicts(corpus, k=k, tag="qwen")
         for k in (K_FINE, J_COARSE)}
    binary = {v: (verdict_hmm.binarize(V[K_FINE][v]),
                  verdict_hmm.binarize(V[J_COARSE][v]))
              for v in V[K_FINE] if v in V[J_COARSE]}
    hmm, n_pos, n_neg = fit_hmm(corpus, train_ids, labels, binary)
    hmm.save(os.path.join(out_dir, "hmm_params.json"))
    say("verdict HMM fitted on %d positive / %d negative train videos: %s"
        % (n_pos, n_neg, json.dumps({k: (round(v, 4) if isinstance(v, float)
                                         else v) for k, v in hmm.params().items()
                                     if k not in ("k", "j")})))
    cache = ds.ScaffoldCache(corpus, train_ids + val_ids + test_ids,
                             make_scaffold_fn(hmm, binary, ablation, a.w_fine))
    cov = {name: sum(v in binary for v in ids)
           for name, ids in (("train", train_ids), ("val", val_ids),
                             ("test", test_ids))}
    say("train/val/test %d/%d/%d videos (%d hateful in train); missing text %d;"
        " verdict coverage train %d/%d val %d/%d test %d/%d; prior_scale %.3f,"
        " lambda_block %.3f, w_fine %.3f, hide_input %s"
        % (len(train_ids), len(val_ids), len(test_ids),
           sum(labels[v] for v in train_ids), cache.n_missing_text,
           cov["train"], len(train_ids), cov["val"], len(val_ids),
           cov["test"], len(test_ids), prior_scale, lambda_block, a.w_fine,
           hide_input))
    if cache.n_missing_verdict > 0:
        say("ABORT: %d videos without verdicts; a partial scaffold would leak"
            " the video label on train" % cache.n_missing_verdict)
        log.close()
        raise SystemExit(3)

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

    model = Candidate(a, prior_scale, hide_input).to(device)
    partner = Single_Model(a, n_dim=align.V_DIM).to(device)
    criterion = nn.BCELoss()
    opt_av = optim.Adam(model.parameters(), lr=a.lr)
    opt_uni = optim.Adam(partner.parameters(), lr=a.lr * a.single_lr_scale)
    sched_av = optim.lr_scheduler.CosineAnnealingLR(opt_av, T_max=a.sched_tmax)
    sched_uni = optim.lr_scheduler.CosineAnnealingLR(opt_uni, T_max=a.sched_tmax)

    best, best_state, best_epoch, history = -1.0, None, -1, []
    for epoch in range(a.max_epoch):
        t0 = time.time()
        model.train()
        partner.train()
        lam_a2b = min(a.lamda_a2b, a.lamda_cof * epoch)
        lam_a2n = min(a.lamda_a2n, a.lamda_cof * epoch)
        tot = np.zeros(4)
        nb = 0
        for f_v, f_a, label in train_loader:
            seq_len = _seq_len_of(f_v)
            keep = int(torch.max(seq_len))
            f_v = f_v[:, :keep, :].float().to(device)
            f_a = f_a[:, :keep, :].float().to(device)
            label = label.float().to(device)
            mmil, a_log, v_log, av_log, v_out, a_out = model(f_a, f_v, seq_len)
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
            bl = 0.0
            if lambda_block > 0:
                bl = block_bag_loss(model.last_content_logit, f_a, seq_len,
                                    label, a.topk_div)
                total = total + lambda_block * bl
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
                    + _scalar(c4), _scalar(bl), _scalar(loss_uni)]
            nb += 1
        sched_av.step()
        sched_uni.step()
        m = distil_step(a, model.av, partner, epoch)
        tot /= max(nb, 1)
        val_scores = score_split(model, val_loader, device)
        vm = frame_metrics(val_scores, val_gt, hate_ids)
        crit = (vm["pooled_ap"] + vm["pooled_roc"]) / 2.0
        history.append({"epoch": epoch + 1, "cls": tot[0], "cma": tot[1],
                        "block": tot[2], "uni": tot[3], "ema_m": m,
                        "val": vm, "val_criterion": crit,
                        "seconds": round(time.time() - t0, 1)})
        say("epoch %2d | cls %.4f | cma %.4f | block %.4f | uni %.4f | "
            "val AP %.4f ROC %.4f within %.4f | %.0fs"
            % (epoch + 1, tot[0], tot[1], tot[2], tot[3],
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
               "hmm": hmm.params(), "host": socket.gethostname()}
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
    ap.add_argument("--max-epoch", type=int, default=None,
                    help="override for smoke tests only")
    args = ap.parse_args(argv)
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as fh:
            cfg.update(json.load(fh))
    if args.max_epoch is not None:
        cfg["max_epoch"] = args.max_epoch
    train(args.corpus, args.seed, args.out_dir, cfg, args.ablation,
          args.device, args.num_workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
