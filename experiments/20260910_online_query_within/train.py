"""One trial of the online-query module (module-1 iteration 2, 2026-09-10):
ONE training run in which the backbone grows each training video's set of
asked fine windows (plan section A), a fine-window bag loss on masked known
verdicts (section C), acquisition weighted by the backbone's own window
prediction (section B), 4 coarse + at most 4 fine calls per video at train
and at test (section E).

Training (single model, max_epoch epochs):
  allowed[v]   fine windows whose cached verdict may be shown for video v;
               starts EMPTY (only the 4 coarse blocks are observed).
  dropout      every item keeps a random subset of allowed[v] (half of the
               time a prefix of the acquisition order), the rest are "not
               asked"; the interval HMM is run with missing emissions on the fly.
  window loss  windows in allowed[v] that are masked in this item have a
               known verdict that is NOT in the input: the top-k mean of the
               content logit over the window's rows is trained towards it
               (negative videos: every window towards 0). Weight lambda_block
               (shared with the coarse-block MIL).
  acquisition  after each epoch in acq_epochs, the CURRENT model runs the eoc
               policy for one step on every training video from allowed[v]
               (candidate verdicts are never read while scoring); the chosen
               window's cached verdict is revealed: allowed[v] += {w}. The HMM
               is refitted on the observed verdicts and the scaffold builder
               swapped. Train-time calls per video = 4 + len(acq_epochs).
               Iteration 1 (README section 5): acq_epochs = [1, 2, 3, 4], so
               the allowed sets are complete before the backbone peaks (this
               backbone's validation optimum is at epoch 1-10; iteration 0's
               events at 5/10/15/20 came after it and every trial's checkpoint
               had trained with <= 1 fine window per video).
Checkpoint: validation (AP + ROC + within) / 3 with the deterministic uniform
mask of b_max windows (bit-reversal order), never the policy, never test;
epochs before the last acquisition event are not eligible (ckpt_from =
"after_acq"), so the selected model has trained on the complete allowed sets.
Evaluation on test (shared evaluator): fixed34, coarse4, every policy at fine-
pick budgets, eoc under the (cap, tau) grid with realized calls;
summary["test"] = eoc at the pre-registered operating point (b_max = 4 fine
picks, tau = 0: 8 calls); stop rule tau_val from the validation grid.

Arms (--ablation; diagnostics only in this iteration, README section 3):
  full                      everything above
  no_missing_state          four-cell evidence encoder (-1 treated as 0)
  no_window_loss            section C off
  hmm_weight                eoc weights from the HMM predictive probability
  regimes3                  interval HMM with the 3-regime reliability mixture
  window_target_posterior   window-loss target = HMM posterior P(h_w | E + b_w)
  fixed_uniform_train       no online acquisition: allowed = bit-reversal first
                            len(acq_epochs) windows from the start (same 8 calls)
  window_target_verdict     window-loss target = raw cached verdict (iteration-1 default)
  no_text                   iteration-2 text evidence off (HMM families and input column)
  no_text_input             text in the HMM only, no per-second LLR input column
  evidence_hmm              iteration-1 evidence log-odds (HMM posterior with coarse emissions per second)
  no_text_term              iteration-3 decomposition without the text log-odds x_t
  no_video_term             iteration-3 decomposition without the video-level term v
  text_prior_off            x_t only as an evidence-encoder input column, not in E (rule-3 diagnostic)

Iteration 4 (README section 9, "training-time budget allocation"): each
acquisition event reveals len(train_ids) windows in total, allocated across
videos by expected output change with a two-window greedy lookahead per video
(acq_alloc = "global", acq_lookahead = 2) instead of exactly one window per
video; the average training calls stay 4 + len(acq_epochs) = 8. Arm
acq_alloc = "per_video" reproduces iterations 1-3.

Iteration 3 (README section 8, "evidence decomposition"): the per-second
evidence log-odds fed to the backbone (ell column, P(s) column, prior term
alpha * ell) is E_t = ell_fine(t) + x_t + v, where ell_fine is the HMM posterior
log-odds with the coarse emissions switched off, x_t the centred text log-odds
of the frozen text classifier (max over ASR / OCR, 0 without text; centre =
train median, no labels), and v = logit P(at least one hate segment | coarse
verdicts) from the same HMM. Coarse block verdicts act at the video level only
(they anti-localize on HCS), fine verdicts and text per second. The HMM fit,
the block MIL target P(h_j), the acquisition and the calls are unchanged.

Iteration 2 (README section 7, "text evidence"): the frozen text classifier's
per-second hate probability over the ASR chunks and the OCR windows is a FREE
observation family of the interval HMM (text=True, categorical 10-level
emissions fitted by EM on train video labels), so ell / P(s) and the window-
loss posterior target include it, the acquisition asks the VLM where the free
evidence leaves the output uncertain, and the backbone's evidence encoder reads
the per-second text LLR (hc.COL_TEXT). Calls are unchanged (8 train / 8 test).
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
    "fusion": "interval", "normalized_time": True, "positive_constraint": True, "regimes": 1,
    "bias_mode": "key", "ctx_mode": "rep",
    # online-query module: b_max (method-level protocol constant), acquisition schedule, dropout mix
    "b_max": 4, "acq_epochs": [1, 2, 3, 4], "prefix_mix": 0.5,   # iteration 1: events in the first epochs (README section 5)
    # iteration 4 (README section 9): training-time budget allocation across videos. "per_video": one window per video
    # per event (iterations 1-3); "global": each event reveals len(train_ids) windows in total, chosen by expected
    # output change across all videos with a greedy lookahead of acq_lookahead windows per video (same average of
    # 4 + len(acq_epochs) calls per training video; a video may get 0 .. acq_lookahead windows per event)
    "acq_alloc": "global", "acq_lookahead": 2,
    "ckpt_from": "after_acq",   # checkpoint eligible from epoch max(acq_epochs) + 1 on ("after_acq") or from epoch 1 ("any")
    "eoc_weight": "model", "window_loss": True, "window_target": "verdict",
    # iteration 2 (README section 7, failed): text as HMM observation families (text) + LLR input column (text_input)
    "text": False, "text_weight": 0.5, "text_input": False,
    # iteration 3 (README section 8): evidence = "decomp": per-second evidence log-odds E_t = ell_fine (coarse
    # emissions off) + centred text log-odds x_t + logit P(any hate | coarse verdicts); "hmm" = iteration-1 fusion
    "evidence": "decomp",
    "eval_max_picks": 18, "control_max_picks": 30,
    "budgets": [0, 2, 4, 8, 12, 18, 30], "b_caps": [2, 4, 8],
    "tau_grid": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    "policies": list(POLICIES),
}
TRAIN_ARMS = ("no_window_loss", "hmm_weight", "regimes3", "window_target_posterior", "window_target_verdict",
              "fixed_uniform_train", "no_text", "evidence_hmm", "no_text_term", "no_video_term", "text_prior_off")
ABLATIONS = STRUCT_ARMS + TRAIN_ARMS


class Args(dict):
    __getattr__ = dict.__getitem__


def masked(bf, allowed):
    out = np.full(len(bf), ieh.MISSING, dtype=int)
    for w in allowed:
        out[w] = int(bf[w])
    return out


def allocate_global(runs, vids, budget):
    """Iteration-4 training-time allocation: from the greedy lookahead runs (picks and
    per-step expected output change per video) choose `budget` (video, step) pairs with
    the largest gains, a video's step j being eligible only after its steps < j (the
    gains were computed along that path). Returns vid -> list of windows to reveal."""
    cands = []
    for v in vids:
        for j, g in enumerate(runs[v]["gains"]):
            cands.append((float(g), v, j))
    cands.sort(key=lambda t: -t[0])
    taken = {v: 0 for v in vids}
    total = 0
    progress = True
    while total < budget and progress:
        progress = False
        for g, v, j in cands:
            if total >= budget:
                break
            if j == taken[v] and j < len(runs[v]["picks"]):
                taken[v] += 1
                total += 1
                progress = True
    return {v: [int(w) for w in runs[v]["picks"][:taken[v]]] for v in vids}


def val_criterion(vm):
    return (vm["pooled_ap"] + vm["pooled_roc"] + vm["within_roc"]) / 3.0


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
    window_loss = bool(a.window_loss) and ablation != "no_window_loss"
    window_target = {"window_target_posterior": "posterior", "window_target_verdict": "verdict"}.get(ablation, str(a.window_target))
    use_text = bool(a.text) and ablation != "no_text"
    text_input = use_text and bool(a.text_input) and ablation != "no_text_input"
    evidence = "hmm" if ablation == "evidence_hmm" else str(a.evidence)
    assert evidence in ("hmm", "decomp"), evidence
    text_term = evidence == "decomp" and ablation not in ("no_text_term", "no_text", "text_prior_off")
    video_term = ablation != "no_video_term"
    text_column = ablation == "text_prior_off"          # x_t only as an encoder input column (rule-3 diagnostic)
    if text_column:
        text_input = True
    assert not (text_input and text_term), "x_t would enter both the encoder column and E (double counting)"
    assert window_target in ("verdict", "posterior"), window_target
    eoc_weight = "hmm" if ablation == "hmm_weight" else str(a.eoc_weight)
    regimes = 3 if ablation == "regimes3" else int(a.regimes)
    assert not (evidence == "decomp" and regimes > 1), "the decomposition's video-level term is the R = 1 formula"
    online = ablation != "fixed_uniform_train"
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
    # free text evidence (README section 7): HMM observations + per-second arrays for the LLR input column
    all_ids = train_ids + val_ids + test_ids
    grid = ieh.make_grid(K_FINE, J_COARSE)
    text_obs = hc.text_observations(corpus, all_ids, grid) if use_text else {}
    text_arrays = {v: hc.load_text_hate(corpus, v) for v in all_ids} if (text_input or text_term) else {}
    centre = hc.text_centre(corpus, train_ids) if (text_term or text_column) else 0.0
    text_x = ({v: hc.text_logit_seconds(text_arrays[v], centre) for v in all_ids if text_arrays.get(v) is not None}
              if (text_term or text_column) else {})
    say("evidence %s | text term %s (centre %.3f, %d / %d videos with text) | HMM text families %s (%d videos) | LLR input column %s"
        % (evidence, text_term, centre, len(text_x), len(all_ids), use_text, len(text_obs), text_input))

    # ------------------------------------------------ allowed sets, HMM, scaffold cache
    acq_epochs = sorted(int(e) for e in a.acq_epochs)
    b_max = int(a.b_max)
    uni = bit_reversal_order(K_FINE)
    if online:
        allowed = {v: set() for v in train_ids}
        policy_order = {v: [] for v in train_ids}
    else:
        allowed = {v: set(uni[:len(acq_epochs)]) for v in train_ids}
        policy_order = {v: list(uni[:len(acq_epochs)]) for v in train_ids}
    val_masks = {v: masked(binary[v][0], uni[:b_max]) for v in val_ids}
    calls = {"train_coarse_per_video": 4, "train_acq_events": len(acq_epochs) if online else 0,
             "train_fixed_fine_per_video": 0 if online else len(acq_epochs),
             "train_total_per_video": 4 + len(acq_epochs)}
    fopts = {"normalized_time": bool(a.normalized_time), "positive_constraint": bool(a.positive_constraint),
             "regimes": regimes, "text": use_text, "text_weight": float(a.text_weight)}
    durations = {v: hc.video_duration(corpus, v) for v in train_ids}
    state = {"hmm": None, "cache": None}

    def refit(tag):
        bin_train = {v: (masked(binary[v][0], allowed[v]), binary[v][1]) for v in train_ids}
        hmm, n_pos, n_neg = hc.fit_hmm(corpus, train_ids, labels, bin_train, model="interval", text_obs=text_obs, **fopts)
        hmm.save(os.path.join(out_dir, "hmm_params_%s.json" % tag))
        text_llr = ({v: hc.text_llr_seconds(hmm, text_arrays[v]) for v in all_ids if text_arrays.get(v) is not None}
                    if text_input else {})
        if text_term or text_column:
            text_llr = text_x                    # the per-second column carries x_t; the decomposed ell adds it (text_term)
        say("%s: HMM fitted on %d pos / %d neg train videos with %.2f fine verdicts observed per video: %s"
            % (tag, n_pos, n_neg, np.mean([len(allowed[v]) for v in train_ids]),
               json.dumps({k: round(v, 4) for k, v in hmm.params().items() if isinstance(v, float)})))
        if state["cache"] is None:
            state["cache"] = hc.ScaffoldCache(corpus, all_ids,
                                              hc.make_scaffold_fn(hmm, binary, "full", 1.0, text=text_obs, text_llr=text_llr, evidence=evidence,
                                                                  video_term=video_term, text_in_ell=text_term),
                                              masked_fn=hc.make_masked_scaffold_fn(hmm, binary, text=text_obs, text_llr=text_llr, evidence=evidence,
                                                                                   video_term=video_term, text_in_ell=text_term))
        else:
            state["cache"].masked_fn = hc.make_masked_scaffold_fn(hmm, binary, text=text_obs, text_llr=text_llr, evidence=evidence,
                                                                  video_term=video_term, text_in_ell=text_term)
        state["hmm"] = hmm

    refit("init")
    cache = state["cache"]

    # evidence dropout inside the allowed set (half random subsets, half acquisition-order prefixes)
    def sampler(vid, rng):
        al = sorted(allowed[vid])
        if not al:
            return masked(binary[vid][0], [])
        if rng.rand() < float(a.prefix_mix):
            order = [w for w in policy_order[vid] if w in allowed[vid]]
            m = rng.randint(0, len(order) + 1)
            return masked(binary[vid][0], order[:m])
        m = rng.randint(0, len(al) + 1)
        keep = rng.choice(al, size=m, replace=False) if m > 0 else []
        return masked(binary[vid][0], keep)

    # window-loss targets: known verdict of a window that is masked in this item
    # (positive videos); every window -> 0 on negative videos
    def window_targets(vid, b_input):
        tgt = np.full(K_FINE, np.nan, dtype=np.float32)
        if labels[vid] == 0:
            tgt[:] = 0.0
            return tgt
        if not window_loss:
            return tgt
        bf_true = binary[vid][0]
        hidden = [w for w in allowed[vid] if b_input[w] == ieh.MISSING]
        if not hidden:
            return tgt
        if window_target == "verdict":
            for w in hidden:
                tgt[w] = float(bf_true[w])
        else:
            hmm = state["hmm"]
            bc = binary[vid][1]
            for w in hidden:
                b2 = b_input.copy()
                b2[w] = int(bf_true[w])
                tgt[w] = float(hmm.infer(b2, bc, durations[vid], xt=text_obs.get(vid))["p_hf"][w])
        return tgt

    train_set = hc.TrainDataset(corpus, train_ids, labels, cache, a.max_seqlen, a.crop_repeat,
                                mask_sampler=sampler, seed=seed,
                                window_targets=window_targets if window_loss else None)
    train_loader = DataLoader(train_set, batch_size=a.batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(hc.EvalDataset(corpus, val_ids, cache, masks=val_masks),
                            batch_size=1, shuffle=False, num_workers=num_workers)
    a["text_input"] = bool(text_input)          # the no_text arm builds the two-input encoder
    model = ERCA(a, a.prior_scale, arm=arm).to(device)
    criterion = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=a.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.sched_tmax)
    best, best_state, best_epoch, history = -1.0, None, -1, []
    acq_log = []
    assert str(a.ckpt_from) in ("after_acq", "any"), a.ckpt_from
    ckpt_from = (max(acq_epochs) + 1 if (online and str(a.ckpt_from) == "after_acq") else 1)
    for epoch in range(a.max_epoch):
        t0 = time.time()
        model.train()
        lam = min(a.lamda_cma, a.lamda_cof * epoch)
        tot = np.zeros(4)
        nb = 0
        for batch in train_loader:
            if window_loss:
                f_v, f_a, w_rows, label, tgt = batch
            else:
                f_v, f_a, w_rows, label = batch
                tgt = None
            seq_len = _seq_len_of(f_v)
            keep = int(torch.max(seq_len))
            f_v = f_v[:, :keep, :].float().to(device)
            f_a = f_a[:, :keep, :].float().to(device)
            w_rows = w_rows[:, :keep].to(device)
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
            bl, wl = 0.0, 0.0
            if a.lambda_block > 0:
                bl = hc.block_bag_loss(model.last_content_logit, f_a, seq_len, label, a.topk_div)
                total = total + a.lambda_block * bl
                if window_loss:
                    wl = hc.window_bag_loss(model.last_content_logit, w_rows, seq_len, tgt.to(device), a.topk_div)
                    total = total + a.lambda_block * wl
            opt.zero_grad()
            total.backward()
            opt.step()
            tot += [hc._scalar(clsloss), hc._scalar(cm), hc._scalar(bl), hc._scalar(wl)]
            nb += 1
        sched.step()
        tot /= max(nb, 1)
        vm = hc.frame_metrics(hc.score_split(model, val_loader, device), val_gt, hate_ids)
        crit = val_criterion(vm)
        history.append({"epoch": epoch + 1, "cls": tot[0], "cma": tot[1], "block": tot[2], "window": tot[3],
                        "val": vm, "val_criterion": crit,
                        "fine_observed_per_video": float(np.mean([len(allowed[v]) for v in train_ids])),
                        "seconds": round(time.time() - t0, 1)})
        say("epoch %2d | cls %.4f | cma %.4f | block %.4f | window %.4f | fine/video %.2f | val AP %.4f ROC %.4f within %.4f | %.0fs"
            % (epoch + 1, tot[0], tot[1], tot[2], tot[3], history[-1]["fine_observed_per_video"],
               vm["pooled_ap"], vm["pooled_roc"], vm["within_roc"], time.time() - t0))
        if crit > best and epoch + 1 >= ckpt_from:
            best, best_epoch = crit, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        # ---------------------------------------------- acquisition event (plan section A)
        if online and (epoch + 1) in acq_epochs:
            t1 = time.time()
            acq = Acquirer(model, state["hmm"], cache, corpus, binary, device, weight=eoc_weight, topk_div=a.topk_div, text=text_obs)
            lookahead = int(a.acq_lookahead) if str(a.acq_alloc) == "global" else 1
            runs = acq.run_split(train_ids, "eoc", lookahead, seed=seed * 1000 + epoch, log=say,
                                 initial={v: sorted(allowed[v]) for v in train_ids})
            if str(a.acq_alloc) == "global":
                chosen = allocate_global(runs, train_ids, len(train_ids))
            else:
                chosen = {v: runs[v]["picks"][:1] for v in train_ids}
            n_new = 0
            for v in train_ids:
                for w in chosen[v]:
                    if w not in allowed[v]:
                        allowed[v].add(int(w))
                        policy_order[v].append(int(w))
                        n_new += 1
            alloc_hist = np.bincount([len(chosen[v]) for v in train_ids], minlength=lookahead + 1).tolist()
            model.train()
            refit("epoch%d" % (epoch + 1))
            acq_log.append({"epoch": epoch + 1, "new_windows": n_new, "alloc": str(a.acq_alloc),
                            "windows_per_video_hist": alloc_hist,
                            "fine_observed_per_video": float(np.mean([len(allowed[v]) for v in train_ids])),
                            "fine_observed_pos": float(np.mean([len(allowed[v]) for v in train_ids if labels[v] == 1])),
                            "fine_observed_neg": float(np.mean([len(allowed[v]) for v in train_ids if labels[v] == 0])),
                            "seconds": round(time.time() - t1, 1)})
            say("acquisition after epoch %d: %d windows revealed (%.2f fine/video; pos %.2f neg %.2f; per-video hist %s) in %.0fs"
                % (epoch + 1, n_new, acq_log[-1]["fine_observed_per_video"], acq_log[-1]["fine_observed_pos"],
                   acq_log[-1]["fine_observed_neg"], alloc_hist, time.time() - t1))
    model.load_state_dict(best_state)
    say("selected epoch %d (val criterion %.4f)" % (best_epoch, best))
    torch.save(best_state, os.path.join(out_dir, "model.pth"))
    hmm = state["hmm"]
    hmm.save(os.path.join(out_dir, "hmm_params.json"))
    with open(os.path.join(out_dir, "allowed_train.json"), "w") as fh:
        json.dump({v: policy_order[v] for v in train_ids}, fh)
    fit_info = {"selected_epoch": best_epoch, "val_criterion": best, "history": history, "acquisition": acq_log}
    calls["train_fine_observed_per_video"] = float(np.mean([len(allowed[v]) for v in train_ids]))
    calls["train_fine_hist"] = np.bincount([len(allowed[v]) for v in train_ids]).tolist()

    # ------------------------------------------------------------ evaluation
    acq = Acquirer(model, hmm, cache, corpus, binary, device, weight=eoc_weight, topk_div=a.topk_div, text=text_obs)
    results = {"curves": {}, "eoc_grid": {}, "calls": calls,
               "text": {"hmm": use_text, "input": text_input, "videos_with_text": len(text_obs), "videos": len(all_ids)},
               "evidence": {"mode": evidence, "text_term": text_term, "video_term": video_term, "text_column": text_column,
                            "centre": centre, "videos_with_text_term": len(text_x)}}
    full_masks = {v: binary[v][0] for v in test_ids}
    none_masks = {v: masked(binary[v][0], []) for v in test_ids}
    for name, masks in (("fixed34", full_masks), ("coarse4", none_masks)):
        loader = DataLoader(hc.EvalDataset(corpus, test_ids, cache, masks=masks), batch_size=1,
                            shuffle=False, num_workers=num_workers)
        results[name] = evaluate_scores(corpus, "test", out_dir, name, hc.score_split(model, loader, device))
        say("test %-8s AP %.4f ROC %.4f within %.4f" % (name, results[name]["pooled_ap"],
                                                         results[name]["pooled_roc"], results[name]["within_roc"]))
    budgets = [int(b) for b in a.budgets]
    for policy in list(a.policies):
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
    # validation (cap, tau) grid and the pre-registered stop rule (README section 2): at cap = b_max,
    # tau_val = largest tau whose validation AP and ROC are both >= the tau = 0 values - .005.
    op_key = "cap%d_tau0" % b_max
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
               "acquisition": fit_info["acquisition"],
               "results": results, "hparams": cfg, "hmm": hmm.params(), "host": socket.gethostname()}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    say("TEST (eoc, %d picks, tau 0; mean calls %.2f) pooled AP %.4f | pooled ROC %.4f | within %.4f"
        % (b_max, test_op["mean_calls"], test_op["pooled_ap"], test_op["pooled_roc"], test_op["within_roc"]))
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
            given = json.load(fh)
        unknown = set(given) - set(DEFAULTS)
        assert not unknown, "unknown config keys %s" % sorted(unknown)
        cfg.update(given)
    train(args.corpus, args.seed, args.out_dir, cfg, args.ablation, args.device, args.num_workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
