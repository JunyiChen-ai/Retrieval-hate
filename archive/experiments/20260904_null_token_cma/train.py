"""One training trial of the null-token cross-modal attention candidate
(candidate 4; model.py). Training = candidate 1 (experiments/20260903_hier_evidence_mil)
without the EMA self-distillation and its visual partner network (its
ablation no_ema showed no effect on either corpus): video-level top-k MIL on
z~ = z + prior_scale * ell / ELL_SCALE, MACIL-SD's cross-modal contrastive
loss CMAL, the verdict-block MIL on the content logit z, checkpoint on the
official validation split, test scored through the shared evaluator.

    python experiments/20260904_null_token_cma/train.py \
        --corpus hatemm --seed 234 --out-dir runs/.../trial0 --config cfg.json

--ablation (README section 3):
  structure     full | const_token | shared_token | no_token_masked | no_token_unmasked | zero_value_sink | gated_cma
  training      mean_prior | no_block | no_prior | no_cmal | no_verdict | no_input
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
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, HERE)

from hate_common import data as hdata                  # noqa: E402
from hate_common import runtime                        # noqa: E402
from macilsd import align                              # noqa: E402
from macilsd.CMA_MIL import CMAL                       # noqa: E402
from macilsd.train import _seq_len_of                  # noqa: E402
import hier_evidence_common as hc                      # noqa: E402
import vlm_verdict                                     # noqa: E402
import verdict_hmm                                     # noqa: E402
from model import NTCA, STRUCT_ARMS                    # noqa: E402

K_FINE, J_COARSE = hc.K_FINE, hc.J_COARSE

DEFAULTS = {
    # MACIL-SD backbone / optimisation (dropout and lamda_cof fixed at upstream values)
    "hid_dim": 128, "ffn_dim": 128, "nhead": 4, "dropout": 0.2,
    "num_classes": 1, "lr": 4e-4, "batch_size": 32, "max_epoch": 50,
    "max_seqlen": 200, "sched_tmax": 60, "lamda_cma": 1.0, "lamda_cof": 0.05,
    "crop_repeat": 5, "fix_rep_swap": False,
    # module 3 prior and fine-verdict tempering; module 2 block MIL
    "prior_scale": 2.0, "w_fine": 1.0, "lambda_block": 0.5, "topk_div": 16,
}
TRAIN_ARMS = ("mean_prior", "no_block", "no_prior", "no_cmal", "no_verdict", "no_input")
ABLATIONS = STRUCT_ARMS + TRAIN_ARMS


class Args(dict):
    __getattr__ = dict.__getitem__


def train(corpus, seed, out_dir, cfg, ablation, device, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "run.log"), "a")

    def say(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    say("host %s | corpus %s | seed %d | ablation %s | code: %s"
        % (socket.gethostname(), corpus, seed, ablation, hc._git_describe()))
    with open(os.path.join(out_dir, "run.pid"), "w") as fh:
        fh.write(str(os.getpid()))
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({"corpus": corpus, "seed": seed, "ablation": ablation,
                   "hparams": cfg, "device": device}, fh, indent=2)

    runtime.setup_seed(seed)
    # diagnostic only (README 8.1): advance the global RNG by `rng_burn` draws after
    # seeding, so the same arm and hparams run on a different random stream (init,
    # data order, dropout) -- measures how much of an arm difference is stream noise
    for _ in range(int(cfg.get("rng_burn", 0))):
        torch.rand(1)
    labels = hdata.load_labels(corpus)
    train_ids = hc.usable(corpus, hdata.load_split(corpus, "train"))
    val_gt = hdata.gt_arrays(corpus, "val")
    test_gt = hdata.gt_arrays(corpus, "test")
    val_ids = [v for v in hc.usable(corpus, hdata.load_split(corpus, "val"))
               if v in val_gt]
    test_ids = [v for v in hc.usable(corpus, hdata.load_split(corpus, "test"))
                if v in test_gt]
    hate_ids = {v for v, l in labels.items() if l == 1}

    a = Args(cfg)
    prior_scale = 0.0 if ablation in ("no_prior", "no_verdict") else a.prior_scale
    lambda_block = 0.0 if ablation in ("no_block", "no_verdict") else a.lambda_block
    lamda_cma = 0.0 if ablation == "no_cmal" else a.lamda_cma
    arm = ablation if ablation in STRUCT_ARMS else "full"

    # module 3: verdict HMM fitted on train video labels only
    V = {k: vlm_verdict.load_verdicts(corpus, k=k, tag="qwen")
         for k in (K_FINE, J_COARSE)}
    binary = {v: (verdict_hmm.binarize(V[K_FINE][v]),
                  verdict_hmm.binarize(V[J_COARSE][v]))
              for v in V[K_FINE] if v in V[J_COARSE]}
    hmm, n_pos, n_neg = hc.fit_hmm(corpus, train_ids, labels, binary)
    hmm.save(os.path.join(out_dir, "hmm_params.json"))
    say("verdict HMM fitted on %d positive / %d negative train videos: %s"
        % (n_pos, n_neg, json.dumps({k: (round(v, 4) if isinstance(v, float)
                                         else v) for k, v in hmm.params().items()
                                     if k not in ("k", "j")})))
    cache = hc.ScaffoldCache(corpus, train_ids + val_ids + test_ids,
                             hc.make_scaffold_fn(hmm, binary, ablation, a.w_fine))
    cov = {name: sum(v in binary for v in ids)
           for name, ids in (("train", train_ids), ("val", val_ids),
                             ("test", test_ids))}
    say("train/val/test %d/%d/%d videos (%d hateful in train); missing text %d;"
        " verdict coverage train %d/%d val %d/%d test %d/%d; prior_scale %.3f,"
        " lambda_block %.3f, lamda_cma %.3f, w_fine %.3f, arm %s, no_verdict %s"
        % (len(train_ids), len(val_ids), len(test_ids),
           sum(labels[v] for v in train_ids), cache.n_missing_text,
           cov["train"], len(train_ids), cov["val"], len(val_ids),
           cov["test"], len(test_ids), prior_scale, lambda_block, lamda_cma,
           a.w_fine, arm, ablation == "no_verdict"))
    if cache.n_missing_verdict > 0:
        say("ABORT: %d videos without verdicts; a partial scaffold would leak"
            " the video label on train" % cache.n_missing_verdict)
        log.close()
        raise SystemExit(3)

    train_set = hc.TrainDataset(corpus, train_ids, labels, cache,
                                a.max_seqlen, a.crop_repeat)
    train_loader = DataLoader(train_set, batch_size=a.batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(hc.EvalDataset(corpus, val_ids, cache),
                            batch_size=1, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(hc.EvalDataset(corpus, test_ids, cache),
                             batch_size=1, shuffle=False, num_workers=num_workers)

    model = NTCA(a, prior_scale, arm=arm, no_verdict=(ablation == "no_verdict"),
                 hide_input=(ablation == "no_input")).to(device)
    say("parameters: %d" % sum(p.numel() for p in model.parameters()))
    criterion = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=a.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.sched_tmax)

    best, best_state, best_epoch, history = -1.0, None, -1, []
    for epoch in range(a.max_epoch):
        t0 = time.time()
        model.train()
        lam = min(lamda_cma, a.lamda_cof * epoch)
        tot = np.zeros(3)
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
            total = clsloss
            cm = 0.0
            if lam > 0:
                c1, c2, c3, c4 = CMAL(mmil, a_log, v_log, seq_len, audio_rep, visual_rep)
                cm = c1 + c2 + c3 + c4
                total = total + lam * cm
            bl = 0.0
            if lambda_block > 0:
                bl = hc.block_bag_loss(model.last_content_logit, f_a, seq_len,
                                       label, a.topk_div)
                total = total + lambda_block * bl
            opt.zero_grad()
            total.backward()
            opt.step()
            tot += [hc._scalar(clsloss), hc._scalar(cm), hc._scalar(bl)]
            nb += 1
        sched.step()
        tot /= max(nb, 1)
        val_scores = hc.score_split(model, val_loader, device)
        vm = hc.frame_metrics(val_scores, val_gt, hate_ids)
        crit = (vm["pooled_ap"] + vm["pooled_roc"]) / 2.0
        history.append({"epoch": epoch + 1, "cls": tot[0], "cma": tot[1],
                        "block": tot[2], "val": vm, "val_criterion": crit,
                        "seconds": round(time.time() - t0, 1)})
        say("epoch %2d | cls %.4f | cma %.4f | block %.4f | "
            "val AP %.4f ROC %.4f within %.4f | %.0fs"
            % (epoch + 1, tot[0], tot[1], tot[2],
               vm["pooled_ap"], vm["pooled_roc"], vm["within_roc"],
               time.time() - t0))
        if crit > best:
            best, best_epoch = crit, epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    say("selected epoch %d (val criterion %.4f)" % (best_epoch, best))
    torch.save(best_state, os.path.join(out_dir, "model.pth"))

    val_scores = hc.score_split(model, val_loader, device)
    hc.write_scores(os.path.join(out_dir, "scores_val.jsonl"), val_scores)
    val_eval = hc.run_evaluator(corpus, "val",
                                os.path.join(out_dir, "scores_val.jsonl"),
                                os.path.join(out_dir, "metrics_val.json"))
    test_scores = hc.score_split(model, test_loader, device)
    hc.write_scores(os.path.join(out_dir, "scores_test.jsonl"), test_scores)
    test_eval = hc.run_evaluator(corpus, "test",
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
