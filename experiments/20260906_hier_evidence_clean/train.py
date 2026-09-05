"""Candidate 1, cleaned (2026-09-06): hierarchical-evidence MIL with nothing that
lacks three-seed evidence.

Backbone and training are MACIL-SD verbatim (scripts/reproduction_baselines/macilsd):
AVCE audio-visual network with one shared cross-modal attention layer, top-k bag
BCE, CMAL contrastive loss at the published weights, visual-partner EMA
self-distillation, cosine schedule, five-crop inference. On top of it, the two
parts of experiments/20260903_hier_evidence_mil that carry three-seed ablation
evidence:

  * fusion: the hierarchical evidence HMM over the frozen-VLM verdicts
    (src/verdict_hmm.py), fitted on TRAIN video labels only; its per-row
    posterior log-odds ell_t enters the frame logit as a fixed-scale prior,
    z~_t = z_t + prior_scale * ell_t / ELL_SCALE, and its four columns
    (ell, P(s), b_fine, b_coarse) are appended to the audio/text input;
  * backbone supervision: verdict-block MIL, one bag per coarse block with the
    HMM block posterior P(h_j=1) as soft label (exact 0 on negative videos),
    weight |2p-1|, scored on the CONTENT logit z.

Removed relative to 20260903_hier_evidence_mil (README section 1 there lists the
three-seed numbers): fine-verdict tempering w_fine (fixed 1), the search over
MACIL-SD's CMAL weights (published defaults), and every ablation-only code path
of later candidates (chain distillation, null tokens, structural arms).

    python experiments/20260906_hier_evidence_clean/train.py \
        --corpus hatemm --seed 234 --out-dir runs/.../trial0 --config cfg.json

``--ablation`` (pre-registered, README section 4):
  full         everything
  mean_prior   ell_t and P(s) replaced by the two-granularity mean level
  indep_hmm    HMM posterior without temporal coupling
  flat_coarse  coarse verdict emitted at every window (no block OR structure)
  no_block     lambda_block = 0
  no_input     verdict columns hidden from the backbone input
  no_prior     prior_scale = 0
  no_verdict   prior 0, block loss 0, input hidden
  no_ema       MACIL-SD visual-partner EMA off
  no_cmal      MACIL-SD CMAL weights 0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
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

from hate_common import data as hdata                      # noqa: E402
from hate_common import runtime                            # noqa: E402
from macilsd import align                                  # noqa: E402
from macilsd.avce_network import AVCE_Model, Single_Model  # noqa: E402
from macilsd.CMA_MIL import CMAL                           # noqa: E402
from macilsd.train import distil_step, _seq_len_of         # noqa: E402
import vlm_verdict                                         # noqa: E402
import verdict_hmm                                         # noqa: E402
import hier_evidence_common as hec                         # noqa: E402

K_FINE, J_COARSE = vlm_verdict.GRANULARITIES              # (30, 4)
ELL_SCALE = hec.ELL_SCALE

DEFAULTS = {
    # MACIL-SD, published values (not searched except lr / dropout / max_seqlen)
    "hid_dim": 128, "ffn_dim": 128, "nhead": 4, "dropout": 0.2,
    "num_classes": 1, "lr": 4e-4, "batch_size": 32, "max_epoch": 50,
    "max_seqlen": 200, "sched_tmax": 60, "single_lr_scale": 0.2,
    "m": 0.91, "ema_epochs": 50, "lamda_a2b": 1.0, "lamda_a2n": 1.0,
    "lamda_cof": 0.05, "crop_repeat": 5, "fix_rep_swap": False,
    "topk_div": 16,
    # the two method scalars (searched)
    "prior_scale": 2.0, "lambda_block": 0.5,
}
ABLATIONS = ("full", "mean_prior", "indep_hmm", "flat_coarse", "no_block",
             "no_input", "no_prior", "no_verdict", "no_ema", "no_cmal")


class Args(dict):
    __getattr__ = dict.__getitem__


class Candidate(nn.Module):
    """MACIL-SD audio-visual model with the fixed HMM-posterior prior."""

    def __init__(self, cfg, prior_scale, hide_input):
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
        f_a_in[..., hec.SCAF_OFFSET + hec.N_INPUT_SCAF:] = 0.0   # bookkeeping cols
        f_a_in[..., hec.SCAF_OFFSET + hec.COL_ELL] /= ELL_SCALE   # input in [-1, 1]
        if self.hide_input:
            f_a_in[..., hec.SCAF_OFFSET:] = 0.0
        mmil, a_log, v_log, av_log, v_out, a_out = self.av(f_a_in, f_v, seq_len)
        self.last_content_logit = av_log
        ell = f_a[..., hec.SCAF_OFFSET + hec.COL_ELL:hec.SCAF_OFFSET + hec.COL_ELL + 1]
        av_log = av_log + self.prior_scale * ell / ELL_SCALE
        mmil = self.bag(av_log, seq_len)
        return mmil, a_log, v_log, av_log, v_out, a_out


def train(corpus, seed, out_dir, cfg, ablation, device, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "run.log"), "a")

    def say(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    say("host %s | corpus %s | seed %d | ablation %s | code: %s"
        % (socket.gethostname(), corpus, seed, ablation, hec._git_describe()))
    with open(os.path.join(out_dir, "run.pid"), "w") as fh:
        fh.write(str(os.getpid()))
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({"corpus": corpus, "seed": seed, "ablation": ablation,
                   "hparams": cfg, "device": device}, fh, indent=2)

    runtime.setup_seed(seed)
    labels = hdata.load_labels(corpus)
    train_ids = hec.usable(corpus, hdata.load_split(corpus, "train"))
    val_gt = hdata.gt_arrays(corpus, "val")
    test_gt = hdata.gt_arrays(corpus, "test")
    val_ids = [v for v in hec.usable(corpus, hdata.load_split(corpus, "val")) if v in val_gt]
    test_ids = [v for v in hec.usable(corpus, hdata.load_split(corpus, "test")) if v in test_gt]
    hate_ids = {v for v, l in labels.items() if l == 1}

    a = Args(cfg)
    prior_scale = 0.0 if ablation in ("no_prior", "no_verdict") else a.prior_scale
    lambda_block = 0.0 if ablation in ("no_block", "no_verdict") else a.lambda_block
    hide_input = ablation in ("no_input", "no_verdict")
    if ablation == "no_cmal":
        a["lamda_a2b"] = 0.0
        a["lamda_a2n"] = 0.0
    use_ema = ablation != "no_ema"

    # fusion: verdict HMM fitted on train video labels only
    V = {k: vlm_verdict.load_verdicts(corpus, k=k, tag="qwen") for k in (K_FINE, J_COARSE)}
    binary = {v: (verdict_hmm.binarize(V[K_FINE][v]), verdict_hmm.binarize(V[J_COARSE][v]))
              for v in V[K_FINE] if v in V[J_COARSE]}
    hmm, n_pos, n_neg = hec.fit_hmm(corpus, train_ids, labels, binary)
    hmm.save(os.path.join(out_dir, "hmm_params.json"))
    say("verdict HMM fitted on %d positive / %d negative train videos: %s"
        % (n_pos, n_neg, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in hmm.params().items() if k not in ("k", "j")})))
    cache = hec.ScaffoldCache(corpus, train_ids + val_ids + test_ids,
                              hec.make_scaffold_fn(hmm, binary, ablation, 1.0))
    cov = {name: sum(v in binary for v in ids)
           for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))}
    say("train/val/test %d/%d/%d videos (%d hateful in train); missing text %d;"
        " verdict coverage train %d/%d val %d/%d test %d/%d; prior_scale %.3f,"
        " lambda_block %.3f, hide_input %s, ema %s, cmal %.2f/%.2f/%.3f"
        % (len(train_ids), len(val_ids), len(test_ids), sum(labels[v] for v in train_ids),
           cache.n_missing_text, cov["train"], len(train_ids), cov["val"], len(val_ids),
           cov["test"], len(test_ids), prior_scale, lambda_block, hide_input, use_ema,
           a.lamda_a2b, a.lamda_a2n, a.lamda_cof))
    if cache.n_missing_verdict > 0:
        say("ABORT: %d videos without verdicts; a partial scaffold would leak"
            " the video label on train" % cache.n_missing_verdict)
        log.close()
        raise SystemExit(3)

    a["a_feature_size"] = hec.A_EXT_DIM
    a["v_feature_size"] = align.V_DIM
    train_set = hec.TrainDataset(corpus, train_ids, labels, cache, a.max_seqlen, a.crop_repeat)
    train_loader = DataLoader(train_set, batch_size=a.batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(hec.EvalDataset(corpus, val_ids, cache), batch_size=1,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(hec.EvalDataset(corpus, test_ids, cache), batch_size=1,
                             shuffle=False, num_workers=num_workers)

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
        for f_v, f_a, _w_rows, label in train_loader:
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
            c1, c2, c3, c4 = CMAL(mmil, a_log, v_log, seq_len, audio_rep, visual_rep)
            total = clsloss + lam_a2b * c1 + lam_a2b * c3 + lam_a2n * c2 + lam_a2n * c4
            bl = 0.0
            if lambda_block > 0:
                bl = hec.block_bag_loss(model.last_content_logit, f_a, seq_len, label, a.topk_div)
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
            tot += [hec._scalar(clsloss),
                    hec._scalar(c1) + hec._scalar(c2) + hec._scalar(c3) + hec._scalar(c4),
                    hec._scalar(bl), hec._scalar(loss_uni)]
            nb += 1
        sched_av.step()
        sched_uni.step()
        m = distil_step(a, model.av, partner, epoch) if use_ema else 1.0
        tot /= max(nb, 1)
        val_scores = hec.score_split(model, val_loader, device)
        vm = hec.frame_metrics(val_scores, val_gt, hate_ids)
        crit = (vm["pooled_ap"] + vm["pooled_roc"]) / 2.0
        history.append({"epoch": epoch + 1, "cls": tot[0], "cma": tot[1], "block": tot[2],
                        "uni": tot[3], "ema_m": m, "val": vm, "val_criterion": crit,
                        "seconds": round(time.time() - t0, 1)})
        say("epoch %2d | cls %.4f | cma %.4f | block %.4f | uni %.4f | "
            "val AP %.4f ROC %.4f within %.4f | %.0fs"
            % (epoch + 1, tot[0], tot[1], tot[2], tot[3], vm["pooled_ap"],
               vm["pooled_roc"], vm["within_roc"], time.time() - t0))
        if crit > best:
            best, best_epoch = crit, epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    say("selected epoch %d (val criterion %.4f)" % (best_epoch, best))
    torch.save(best_state, os.path.join(out_dir, "model.pth"))

    val_scores = hec.score_split(model, val_loader, device)
    hec.write_scores(os.path.join(out_dir, "scores_val.jsonl"), val_scores)
    val_eval = hec.run_evaluator(corpus, "val", os.path.join(out_dir, "scores_val.jsonl"),
                                 os.path.join(out_dir, "metrics_val.json"))
    test_scores = hec.score_split(model, test_loader, device)
    hec.write_scores(os.path.join(out_dir, "scores_test.jsonl"), test_scores)
    test_eval = hec.run_evaluator(corpus, "test", os.path.join(out_dir, "scores_test.jsonl"),
                                  os.path.join(out_dir, "metrics.json"))

    def pick(ev):
        r = ev["results"]["score_av"]
        return {"pooled_ap": r["pr_auc"], "pooled_roc": r["roc_auc"],
                "within_roc": r["per_video"]["macro_auc"], "n_videos": r["n_videos"]}

    summary = {"corpus": corpus, "seed": seed, "ablation": ablation,
               "selected_epoch": best_epoch, "val_criterion": best,
               "val": pick(val_eval), "test": pick(test_eval), "history": history,
               "hparams": cfg, "hmm": hmm.params(), "host": socket.gethostname()}
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
    train(args.corpus, args.seed, args.out_dir, cfg, args.ablation, args.device, args.num_workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
