"""One trial of the adaptive VLM query module (2026-09-08): backbone trained
with evidence dropout over a per-video ALLOWED set of fine windows, interval
HMM run with missing emissions, greedy backbone-driven acquisition at test.

Rounds (default 2, README section 1.3):
  round 0  allowed = 4 seed fine windows (+ the 4 coarse blocks): 8 calls per
           training video; train M0 with random subsets of the allowed set;
  policy   M0 runs the eoc policy on the training videos (b_max picks each);
           allowed |= picks (policy starts from the seed windows; train-time calls = 8 + new picks);
  round 1  HMM refitted on the observed verdicts, M1 trained with dropout
           inside the new allowed set (half random subsets, half policy-order
           prefixes).
Checkpoint: validation (AP+ROC)/2 with the deterministic uniform mask of b_max
windows (bit-reversal order), never the policy, never test.
Evaluation on test (all through the shared evaluator): fixed34, coarse4, every
policy at fine-pick budgets, eoc under (b_max cap, tau) grid with realized
calls; summary["test"] = eoc at the pre-registered operating point
(b_max fine picks, tau = 0) so the search objective is the same as before.

Arms (--ablation):
  full             everything above
  no_missing_state four-cell evidence encoder (-1 treated as 0)
  no_dropout       train on all 34 verdicts without masking, one round; test masked
  train34          train on all 34 with dropout, one round; test with the policy
  round0_only      no retraining on the policy-observed set (one round)
  coarse4_train    allowed = {} (coarse blocks only) at train, one round, test coarse-only
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
from macilsd.CMA_MIL import CMAL                       # noqa: E402
from macilsd.train import _seq_len_of                  # noqa: E402
import hier_evidence_common as hc                      # noqa: E402
import interval_evidence_hmm as ieh                    # noqa: E402
import vlm_verdict                                     # noqa: E402
import verdict_hmm                                     # noqa: E402
from model import ERCA, STRUCT_ARMS                    # noqa: E402
from acquire import (Acquirer, POLICIES, bit_reversal_order,  # noqa: E402
                     scores_at, stop_index)

K_FINE, J_COARSE = hc.K_FINE, hc.J_COARSE

DEFAULTS = {
    "hid_dim": 128, "ffn_dim": 128, "nhead": 4, "dropout": 0.2,
    "num_classes": 1, "lr": 4e-4, "batch_size": 32, "max_epoch": 50,
    "max_seqlen": 200, "sched_tmax": 60, "lamda_cma": 1.0, "lamda_cof": 0.05,
    "crop_repeat": 5, "fix_rep_swap": False,
    "prior_scale": 2.0, "w_fine": 1.0, "lambda_block": 0.5, "topk_div": 16,
    "fusion": "interval", "normalized_time": True, "positive_constraint": True,
    # backbone variant (revision 2 by default; revision 3 = gated / logit)
    "bias_mode": "key", "ctx_mode": "rep",
    # adaptive query module (method-level: b_max; the rest fixed grids / protocol)
    "b_max": 8, "seed_windows": [0, 15, 7, 22], "rounds": 2, "prefix_mix": 0.5,
    "eval_max_picks": 18, "control_max_picks": 30,
    "budgets": [0, 2, 4, 8, 12, 18, 30], "b_caps": [4, 8, 12],
    "tau_grid": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    "policies": list(POLICIES),
}
TRAIN_ARMS = ("no_dropout", "train34", "round0_only", "coarse4_train")
ABLATIONS = STRUCT_ARMS + TRAIN_ARMS


class Args(dict):
    __getattr__ = dict.__getitem__


def masked(bf, allowed):
    out = np.full(len(bf), ieh.MISSING, dtype=int)
    for w in allowed:
        out[w] = int(bf[w])
    return out


def train_one_model(a, corpus, seed, out_dir, tag, device, num_workers, say,
                    labels, hmm, cache, binary, train_ids, val_ids, val_gt, hate_ids,
                    sampler, val_masks, arm):
    train_set = hc.TrainDataset(corpus, train_ids, labels, cache, a.max_seqlen,
                                a.crop_repeat, mask_sampler=sampler, seed=seed)
    train_loader = DataLoader(train_set, batch_size=a.batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(hc.EvalDataset(corpus, val_ids, cache, masks=val_masks),
                            batch_size=1, shuffle=False, num_workers=num_workers)
    model = ERCA(a, a.prior_scale, arm=arm).to(device)
    criterion = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=a.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.sched_tmax)
    best, best_state, best_epoch, history = -1.0, None, -1, []
    for epoch in range(a.max_epoch):
        t0 = time.time()
        model.train()
        lam = min(a.lamda_cma, a.lamda_cof * epoch)
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
            if a.lambda_block > 0:
                bl = hc.block_bag_loss(model.last_content_logit, f_a, seq_len,
                                       label, a.topk_div)
                total = total + a.lambda_block * bl
            opt.zero_grad()
            total.backward()
            opt.step()
            tot += [hc._scalar(clsloss), hc._scalar(cm), hc._scalar(bl)]
            nb += 1
        sched.step()
        tot /= max(nb, 1)
        vm = hc.frame_metrics(hc.score_split(model, val_loader, device), val_gt, hate_ids)
        crit = (vm["pooled_ap"] + vm["pooled_roc"]) / 2.0
        history.append({"epoch": epoch + 1, "cls": tot[0], "cma": tot[1], "block": tot[2],
                        "val": vm, "val_criterion": crit, "seconds": round(time.time() - t0, 1)})
        say("%s epoch %2d | cls %.4f | cma %.4f | block %.4f | val AP %.4f ROC %.4f within %.4f | %.0fs"
            % (tag, epoch + 1, tot[0], tot[1], tot[2], vm["pooled_ap"], vm["pooled_roc"],
               vm["within_roc"], time.time() - t0))
        if crit > best:
            best, best_epoch = crit, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    say("%s selected epoch %d (val criterion %.4f)" % (tag, best_epoch, best))
    torch.save(best_state, os.path.join(out_dir, "model_%s.pth" % tag))
    return model, {"selected_epoch": best_epoch, "val_criterion": best, "history": history}


def evaluate_scores(corpus, split, out_dir, name, scores):
    sp = os.path.join(out_dir, "scores_%s_%s.jsonl" % (split, name))
    hc.write_scores(sp, scores)
    ev = hc.run_evaluator(corpus, split, sp, os.path.join(out_dir, "metrics_%s_%s.json" % (split, name)))
    r = ev["results"]["score_av"]
    return {"pooled_ap": r["pr_auc"], "pooled_roc": r["roc_auc"],
            "within_roc": r["per_video"]["macro_auc"], "n_videos": r["n_videos"]}


def train(corpus, seed, out_dir, cfg, ablation, device, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "run.log"), "a")

    def say(msg):
        print(msg, flush=True)
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
    a = Args(cfg)
    arm = ablation if ablation in STRUCT_ARMS else "full"
    labels = hdata.load_labels(corpus)
    train_ids = hc.usable(corpus, hdata.load_split(corpus, "train"))
    val_gt = hdata.gt_arrays(corpus, "val")
    test_gt = hdata.gt_arrays(corpus, "test")
    val_ids = [v for v in hc.usable(corpus, hdata.load_split(corpus, "val")) if v in val_gt]
    test_ids = [v for v in hc.usable(corpus, hdata.load_split(corpus, "test")) if v in test_gt]
    hate_ids = {v for v, l in labels.items() if l == 1}
    V = {k: vlm_verdict.load_verdicts(corpus, k=k, tag="qwen") for k in (K_FINE, J_COARSE)}
    binary = {v: (verdict_hmm.binarize(V[K_FINE][v]), verdict_hmm.binarize(V[J_COARSE][v]))
              for v in V[K_FINE] if v in V[J_COARSE]}
    missing = [v for v in train_ids + val_ids + test_ids if v not in binary]
    if missing:
        say("ABORT: %d videos without verdicts" % len(missing))
        raise SystemExit(3)

    # allowed fine windows per training video and the round schedule
    seed_w = [int(w) for w in a.seed_windows]
    if ablation in ("no_dropout", "train34"):
        allowed = {v: set(range(K_FINE)) for v in train_ids}
    elif ablation == "coarse4_train":
        allowed = {v: set() for v in train_ids}
    else:
        allowed = {v: set(seed_w) for v in train_ids}
    rounds = int(a.rounds) if ablation in ("full", "no_missing_state") else 1
    policy_order = {v: list(seed_w) for v in train_ids}
    uni = bit_reversal_order(K_FINE)
    b_max = int(a.b_max)
    val_masks = ({v: masked(binary[v][0], []) for v in val_ids} if ablation == "coarse4_train"
                 else {v: masked(binary[v][0], uni[:b_max]) for v in val_ids})
    calls = {"train_round0_per_video": 4 + len(seed_w) if ablation in ("full", "no_missing_state", "round0_only")
             else (34 if abled(ablation) else 4)}
    fopts = {"normalized_time": bool(a.normalized_time), "positive_constraint": bool(a.positive_constraint)}
    model, fit_info, hmm, cache = None, {}, None, None
    for r in range(rounds):
        bin_train = {v: (masked(binary[v][0], allowed[v]), binary[v][1]) for v in train_ids}
        hmm, n_pos, n_neg = hc.fit_hmm(corpus, train_ids, labels, bin_train, model="interval", **fopts)
        hmm.save(os.path.join(out_dir, "hmm_params_round%d.json" % r))
        say("round %d: HMM fitted on %d pos / %d neg train videos with %.1f fine verdicts observed per video: %s"
            % (r, n_pos, n_neg, np.mean([len(allowed[v]) for v in train_ids]),
               json.dumps({k: round(v, 4) for k, v in hmm.params().items() if isinstance(v, float)})))
        cache = hc.ScaffoldCache(corpus, train_ids + val_ids + test_ids,
                                 hc.make_scaffold_fn(hmm, binary, "full", 1.0),
                                 masked_fn=hc.make_masked_scaffold_fn(hmm, binary))
        if ablation == "no_dropout":
            sampler = lambda vid, rng: binary[vid][0]                            # noqa: E731
        else:
            mix = float(a.prefix_mix) if r > 0 else 0.0

            def sampler(vid, rng, _mix=mix):
                al = sorted(allowed[vid])
                if _mix > 0 and rng.rand() < _mix:
                    order = [w for w in policy_order[vid] if w in allowed[vid]]
                    m = rng.randint(0, len(order) + 1)
                    return masked(binary[vid][0], order[:m])
                m = rng.randint(0, len(al) + 1)
                keep = rng.choice(al, size=m, replace=False) if m > 0 else []
                return masked(binary[vid][0], keep)
        model, fit_info = train_one_model(
            a, corpus, seed, out_dir, "round%d" % r, device, num_workers, say, labels, hmm,
            cache, binary, train_ids, val_ids, val_gt, hate_ids, sampler, val_masks, arm)
        if r < rounds - 1:
            acq = Acquirer(model, hmm, cache, corpus, binary, device)
            t0 = time.time()
            # the policy starts from the seed windows already paid for in round 0,
            # so every pick is a new call (tau = 0: exactly b_max new calls per video)
            runs = acq.run_split(train_ids, "eoc", b_max, seed=seed, log=say, initial=seed_w)
            picks = [len(set(runs[v]["picks"]) - set(seed_w)) for v in train_ids]
            for v in train_ids:
                allowed[v] |= set(runs[v]["picks"])
                policy_order[v] = list(seed_w) + [w for w in runs[v]["picks"] if w not in seed_w]
            calls["train_policy_picks_mean"] = float(np.mean(picks))
            calls["train_total_per_video"] = 4 + len(seed_w) + float(np.mean(picks))
            say("policy on train: %.1f new picks per video in %.0fs; train-time calls %.1f per video"
                % (np.mean(picks), time.time() - t0, calls["train_total_per_video"]))
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))
    hmm.save(os.path.join(out_dir, "hmm_params.json"))

    # ------------------------------------------------------------ evaluation
    acq = Acquirer(model, hmm, cache, corpus, binary, device)
    results = {"curves": {}, "eoc_grid": {}, "calls": calls}
    full_masks = {v: binary[v][0] for v in test_ids}
    none_masks = {v: masked(binary[v][0], []) for v in test_ids}
    # coarse4_train never observes a fine verdict, so its HMM has no fine emission
    # parameters: only the coarse-only evaluation is defined for that arm.
    for name, masks in (("fixed34", full_masks), ("coarse4", none_masks)):
        if ablation == "coarse4_train" and name == "fixed34":
            continue
        loader = DataLoader(hc.EvalDataset(corpus, test_ids, cache, masks=masks), batch_size=1,
                            shuffle=False, num_workers=num_workers)
        results[name] = evaluate_scores(corpus, "test", out_dir, name, hc.score_split(model, loader, device))
        say("test %-8s AP %.4f ROC %.4f within %.4f" % (name, results[name]["pooled_ap"],
                                                         results[name]["pooled_roc"], results[name]["within_roc"]))
    budgets = [int(b) for b in a.budgets]
    policies = list(a.policies) if ablation != "coarse4_train" else []
    for policy in policies:
        n_steps = int(a.eval_max_picks) if policy == "eoc" else int(a.control_max_picks)
        t0 = time.time()
        runs = acq.run_split(test_ids, policy, n_steps, seed=seed, log=say)
        curve = {}
        for k in budgets:
            if k > n_steps:
                continue
            m = evaluate_scores(corpus, "test", out_dir, "%s_k%d" % (policy, k), scores_at(runs, k))
            m["calls"] = 4 + k
            curve[str(k)] = m
            say("test %-12s picks %2d calls %2d | AP %.4f ROC %.4f within %.4f"
                % (policy, k, 4 + k, m["pooled_ap"], m["pooled_roc"], m["within_roc"]))
        results["curves"][policy] = curve
        if policy == "eoc":
            with open(os.path.join(out_dir, "eoc_runs_test.json"), "w") as fh:
                json.dump({v: {"picks": r["picks"], "gains": r["gains"]} for v, r in runs.items()}, fh)
            for cap in [int(c) for c in a.b_caps]:
                for tau in [float(t) for t in a.tau_grid]:
                    stops = {v: stop_index(r, cap, tau) for v, r in runs.items()}
                    sc = {v: runs[v]["scores"][stops[v]] for v in runs}
                    key = "cap%d_tau%g" % (cap, tau)
                    m = evaluate_scores(corpus, "test", out_dir, "eoc_" + key, sc)
                    m["mean_calls"] = 4 + float(np.mean(list(stops.values())))
                    m["calls_hist"] = np.bincount(list(stops.values()), minlength=cap + 1).tolist()
                    results["eoc_grid"][key] = m
                    say("test eoc %-14s mean calls %5.2f | AP %.4f ROC %.4f within %.4f"
                        % (key, m["mean_calls"], m["pooled_ap"], m["pooled_roc"], m["within_roc"]))
        say("  %s done in %.0fs" % (policy, time.time() - t0))
    # validation: eoc (cap, tau) grid (checkpoint itself was selected under the uniform mask).
    # Pre-registered stop-rule selection (README section 2): at cap = b_max, tau_val = the
    # largest tau in the grid whose validation pooled AP and ROC are both >= the tau = 0
    # values - .005. The test number at tau_val is reported next to the tau = 0 operating point.
    op_key = "cap%d_tau0" % b_max
    if ablation == "coarse4_train":
        test_op = dict(results["coarse4"], mean_calls=4.0)
        summary = {"corpus": corpus, "seed": seed, "ablation": ablation,
                   "operating_point": {"policy": "none", "b_max": 0, "tau": 0.0, "key": "coarse4"},
                   "test": test_op, "val": None, "stop_rule": None,
                   "selected_epoch": fit_info["selected_epoch"],
                   "val_criterion": fit_info["val_criterion"], "history": fit_info["history"],
                   "results": results, "hparams": cfg, "hmm": hmm.params(), "host": socket.gethostname()}
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=float)
        say("TEST (coarse4_train; 4 calls) pooled AP %.4f | pooled ROC %.4f | within %.4f"
            % (test_op["pooled_ap"], test_op["pooled_roc"], test_op["within_roc"]))
        log.close()
        return summary
    vruns = acq.run_split(val_ids, "eoc", max(int(c) for c in a.b_caps), seed=seed)
    val_grid = {}
    for cap in [int(c) for c in a.b_caps]:
        for tau in [float(t) for t in a.tau_grid]:
            stops = {v: stop_index(r, cap, tau) for v, r in vruns.items()}
            key = "cap%d_tau%g" % (cap, tau)
            m = evaluate_scores(corpus, "val", out_dir, "eoc_" + key, {v: vruns[v]["scores"][stops[v]] for v in vruns})
            m["mean_calls"] = 4 + float(np.mean(list(stops.values())))
            val_grid[key] = m
    results["val_eoc_grid"] = val_grid
    val_op = val_grid[op_key]
    tau_val = 0.0
    for tau in sorted(float(t) for t in a.tau_grid):
        m = val_grid["cap%d_tau%g" % (b_max, tau)]
        if m["pooled_ap"] >= val_op["pooled_ap"] - 0.005 and m["pooled_roc"] >= val_op["pooled_roc"] - 0.005:
            tau_val = tau
    stop_key = "cap%d_tau%g" % (b_max, tau_val)
    results["stop_rule"] = {"tau_val": tau_val, "key": stop_key, "val": val_grid[stop_key],
                            "test": results["eoc_grid"][stop_key]}
    say("stop rule: tau_val %g (val calls %.2f AP %.4f ROC %.4f) -> test calls %.2f AP %.4f ROC %.4f within %.4f"
        % (tau_val, val_grid[stop_key]["mean_calls"], val_grid[stop_key]["pooled_ap"], val_grid[stop_key]["pooled_roc"],
           results["eoc_grid"][stop_key]["mean_calls"], results["eoc_grid"][stop_key]["pooled_ap"],
           results["eoc_grid"][stop_key]["pooled_roc"], results["eoc_grid"][stop_key]["within_roc"]))
    test_op = dict(results["eoc_grid"][op_key])
    summary = {"corpus": corpus, "seed": seed, "ablation": ablation,
               "operating_point": {"policy": "eoc", "b_max": b_max, "tau": 0.0, "key": op_key},
               "test": test_op, "val": val_op, "stop_rule": results["stop_rule"],
               "selected_epoch": fit_info["selected_epoch"],
               "val_criterion": fit_info["val_criterion"], "history": fit_info["history"],
               "results": results, "hparams": cfg, "hmm": hmm.params(), "host": socket.gethostname()}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    say("TEST (eoc, %d picks, tau 0; mean calls %.2f) pooled AP %.4f | pooled ROC %.4f | within %.4f"
        % (b_max, test_op["mean_calls"], test_op["pooled_ap"], test_op["pooled_roc"], test_op["within_roc"]))
    log.close()
    return summary


def abled(ablation):
    return ablation in ("no_dropout", "train34")


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
