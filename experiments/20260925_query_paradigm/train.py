"""One trial of the query-tree method (README section 2).

Training: the prior network (model.PriorNet) and the answer model (qtree.AnswerModel) are trained jointly on the
training videos by  -log P(Y | x) - log P(all cached answers of the video | G = Y, x) / n_answers  (exact on the
query tree, qtree.TreeBatch) + lambda * CMAL (MACIL-SD, warm-up min(lamda_cma, lamda_cof * epoch)). The network
never reads an answer.
Checkpoint: the epoch with the highest validation (pooled AP + pooled ROC) / 2, where each validation video is
scored after `val_budget` questions chosen by expected information gain.
Evaluation (val and test, same checkpoint): the policy runs up to `max_calls` questions per video; reported are
fixed per-video budgets, and the adaptive threshold rule at mean budgets B (threshold c_B calibrated on the
validation videos so that their mean number of calls is <= B). summary["test"] = fixed primary_budget questions
per video (README section 2.4, pre-registered primary point). Test numbers go through the shared evaluator.

Revision 1 (README section 7, default prior "chain"): the per-second prior is a CRF over the seconds with a learned
2-state transition (ctree.py); training loss -log P(Y | x) - log P(answers | Y, x) / n_answers, both exact on the
tree; evaluation runs the policy for many videos at once (cpolicy.py) and the adaptive rule stops on the expected
squared-error risk reduction of the chosen question (stop "voi", stopping.py). prior "independent" = revision 0.

Revision 2 (README section 9, the defaults): the answer model is fitted once on the training answers whose state
the label fixes (qtree.fit_anchored: every node of a negative video is state 0, the root of a positive video is
state 1) and then fixed; two states, no length term; the chain is learned (its own learning rate lr_answer); the
network is trained by -log P(Y | x) - log P(answers | Y, x) / n_answers on the tree.

Revision 3 (README section 10, the defaults): chain_form "zero_inflated": P(G = 1 | x) = sigmoid(g) for every video
length; given G = 1 the seconds follow the closed chain with unary logits s_t conditioned on at least one harmful
second; given G = 0 all seconds are 0 (revisions 1-2: "coupled", where the chain's all-zero weight shrinks with the
length and long videos were pushed to "harmful").

Arms (config keys; README section 9 ablations): categories [0] (a, hate only), fusion "flat" (b: questions still
chosen by EIG on the tree posterior, the score is logit(prior) + mean over asked nodes covering t of
[log P(o|s=1) - log P(o|s=0)]), objective "label" (c: the network is trained by the label only, answers used at test
only), order "bfs" (d: evaluation-time question order), prior "independent" (e: seconds independent given the
video state, revision 0), answer_model "joint" (f: the answer model learned jointly with the network from the
data-driven start, revision 1). Diagnostic switches (README section 8): objective "mil" / "posterior", g_head,
chain "hazard", text_sources, n_state 3, length_term.

    python experiments/20260925_query_paradigm/train.py --corpus hatemm --seed 234 --out-dir runs/... [--config c.json]
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from hate_common import runtime                 # noqa: E402
from macilsd.CMA_MIL import CMAL                # noqa: E402
import hier_evidence_common as hc               # noqa: E402
import vlm_verdict                              # noqa: E402
import data as qdata                            # noqa: E402
import qtree                                    # noqa: E402
import policy                                   # noqa: E402
import ctree                                    # noqa: E402
import cpolicy                                  # noqa: E402
from model import PriorNet                      # noqa: E402

DEFAULTS = {
    "hid_dim": 128, "ffn_dim": 128, "nhead": 4, "dropout": 0.2,
    "lr": 4e-4, "lr_answer": 0.03, "batch_size": 32, "max_epoch": 50, "sched_tmax": 60,
    "lamda_cma": 1.0, "lamda_cof": 0.05, "crop_repeat": 5, "long_T": 512, "topk_div": 16,
    "val_budget": 8, "max_calls": 32, "fixed_budgets": [0, 1, 2, 4, 8, 16, 32], "mean_budgets": [2, 4, 8],
    "primary_budget": 8,
    "n_state": 2, "length_term": False, "categories": [0, 1, 2, 3, 4], "objective": "tree", "order": "eig",
    "fusion": "tree", "prior": "chain", "eval_chunk": 64, "answer_model": "anchored", "g_head": True,
    "text_sources": ["bert"], "chain": "learned", "boundary": "closed",
    "chain_form": "zero_inflated",
}
# Revision 2 (README section 9): anchored two-state answer model without length term, learned chain, BERT text row.
# Revision 0/1 settings: n_state 3, length_term True, answer_model "joint" (prior "independent" for revision 0).


def git_describe():
    try:
        return subprocess.run(["git", "log", "-1", "--format=%s (%ad)"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return "unknown"


def group_ap(scores, gt, ids):
    from sklearn.metrics import average_precision_score
    ids = [v for v in ids if v in scores]
    s = np.concatenate([scores[v][:len(gt[v])] for v in ids])
    g = np.concatenate([np.asarray(gt[v])[:len(scores[v])] for v in ids])
    return float(average_precision_score(g, s)) if len(ids) and g.min() != g.max() else None


class Evaluator:
    """Runs the question policy for a set of videos with the current model / answer model."""

    def __init__(self, store, answers, cfg, device):
        self.store, self.answers, self.cfg, self.device = store, answers, cfg, device
        self.trees = {}

    def vt(self, v):
        if v not in self.trees:
            self.trees[v] = qtree.VideoTree(self.store.T[v])
        return self.trees[v]

    def run(self, model, am, ids, max_calls, order=None, chain=None, record_voi=False):
        model.eval()
        out = {}
        if chain is not None:
            ids = sorted(ids, key=lambda v: self.store.T[v])
            k = int(self.cfg["eval_chunk"])
            for i in range(0, len(ids), k):
                out.update(cpolicy.run_batch(model, self.store, ids[i:i + k], am, chain, self.answers,
                                             self.cfg["categories"], max_calls, self.device,
                                             order or self.cfg["order"], self.cfg["fusion"], record_voi))
            model.train()
            return out
        for v in ids:
            s, g = policy.video_prior(model, self.store, v, self.device)
            vt = self.vt(v)
            asker = policy.Asker(vt, am, self.cfg["categories"])
            out[v] = policy.run_video(vt, s, g, asker, self.answers[v], max_calls, order or self.cfg["order"],
                                      self.cfg["fusion"])
        model.train()
        return out


def at_budget(runs, B):
    return {v: r["scores"][min(B, len(r["eig"]))] for v, r in runs.items()}


def at_threshold(runs, c, rule="eig"):
    calls = {v: policy.stop_calls(r[rule], c) for v, r in runs.items()}
    return {v: r["scores"][calls[v]] for v, r in runs.items()}, calls


def train(corpus, seed, out_dir, cfg, device, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "run.log"), "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say("host %s | code: %s | corpus %s seed %d" % (socket.gethostname(), git_describe(), corpus, seed))
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(out_dir, "run.pid"), "w") as fh:
        fh.write(str(os.getpid()))
    runtime.setup_seed(seed)
    labels, ids, gt, _ = hc.load_fixed_cohort(corpus)
    answers, T_ans = qdata.load_answers(corpus)
    all_ids = ids["train"] + ids["val"] + ids["test"]
    missing = [v for v in all_ids if v not in answers]
    assert not missing, "videos without tree answers: %d (%s)" % (len(missing), missing[:5])
    store = qdata.Store(corpus, all_ids, cfg["text_sources"])
    bad = [v for v in all_ids if store.T[v] != T_ans[v]]
    assert not bad, "tree T differs from the 1-fps grid for %s" % bad[:5]
    say("videos train/val/test %d/%d/%d; missing text rows %d" % (len(ids["train"]), len(ids["val"]),
                                                                  len(ids["test"]), store.n_missing_text))
    train_obs = {v: qdata.observed(answers[v], store.T[v]) for v in ids["train"]}
    ta = [(labels[v], [(a, b, store.T[v], o) for (a, b), o in answers[v].items()]) for v in ids["train"]]
    n_state = int(cfg["n_state"])
    loglen = np.log([b - a for v in ids["train"] for (a, b) in answers[v]])
    anchored = cfg["answer_model"] in ("anchored", "refit")
    if anchored:                  # README section 8: fitted once on the answers whose state the label fixes
        assert n_state == 2, "the anchored answer model has two states"
        theta0, omega0, n_fit = qtree.fit_anchored(ta, loglen.mean(), loglen.std(), bool(cfg["length_term"]))
        say("anchored answer model: fitted on %d state-0 and %d state-1 answers" % (n_fit[0], n_fit[1]))
        if cfg["answer_model"] == "refit":       # diagnostic: refitted with a length term after every epoch
            assert not cfg["length_term"]
            omega0 = np.zeros_like(theta0)
    else:
        theta0, omega0 = qtree.init_theta(ta, n_state), None
    am = qtree.AnswerModel(theta0, loglen.mean(), loglen.std(),
                           bool(cfg["length_term"]) or cfg["answer_model"] == "refit", n_state,
                           cfg["categories"], omega0).to(device)
    if anchored:
        am.requires_grad_(False)
    model = PriorNet(cfg).to(device)
    use_chain = cfg["prior"] == "chain"
    chain = None
    if use_chain:
        assert cfg["boundary"] in ("closed", "free") and cfg["chain"] in ("learned", "hazard")
        assert cfg["chain_form"] in ("zero_inflated", "coupled", "normalized")
        assert cfg["chain"] == "learned" or cfg["chain_form"] == "coupled", "the hazard chain is coupled only"
        closed = cfg["boundary"] == "closed"
        chain = (ctree.HazardChain(closed) if cfg["chain"] == "hazard"
                 else ctree.Chain(closed, cfg["chain_form"] == "zero_inflated",
                                  cfg["chain_form"] == "normalized")).to(device)
    # the answer model and the chain have few parameters and start from data-driven values; with the network's
    # learning rate they did not move from their start (README section 7.3), so they get their own rate
    small = ([] if anchored else list(am.parameters())) + (list(chain.parameters()) if use_chain else [])
    opt = optim.Adam([{"params": list(model.parameters()), "lr": float(cfg["lr"])}]
                     + ([{"params": small, "lr": float(cfg["lr_answer"])}] if small else []))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg["sched_tmax"]))
    ds = qdata.TrainSet(store, ids["train"], labels, int(cfg["crop_repeat"]))
    lengths = np.repeat([store.T[v] for v in ds.ids], int(cfg["crop_repeat"]))
    sampler = qdata.LengthBatches(lengths, int(cfg["batch_size"]), seed, int(cfg["long_T"]))
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=qdata.collate, num_workers=num_workers,
                        persistent_workers=num_workers > 0)
    ev = Evaluator(store, answers, cfg, device)
    if cfg["objective"] == "posterior":
        assert use_chain and anchored, "the posterior objective needs the chain prior and the anchored answer model"
    budget_rng = np.random.RandomState(seed)
    hate_val = {v for v in ids["val"] if labels[v] == 1}
    history, best = [], None
    for epoch in range(int(cfg["max_epoch"])):
        t0 = time.time()
        model.train()
        lam = min(float(cfg["lamda_cma"]), float(cfg["lamda_cof"]) * epoch)
        tot, nb = np.zeros(3), 0
        if cfg["objective"] == "posterior":      # the questions the policy would ask each training video now
            asked_train = {v: r["asked"] for v, r in
                           ev.run(model, am, ids["train"], int(cfg["primary_budget"]), chain=chain).items()}
        for f_v, f_a, seq, label, vidx in loader:
            Tm = f_v.shape[1]
            f_v, f_a, label = f_v.to(device), f_a.to(device), label.to(device)
            mask = torch.arange(Tm, device=device)[None, :] < seq.to(device)[:, None]
            s, g, a_log, v_log, v_out, a_out = model(f_a, f_v, mask)
            vids = [ds.ids[i] for i in vidx.tolist()]
            direct = cfg["objective"] in ("tree", "label", "posterior")
            s_tree = s if direct else s.detach()
            if use_chain:
                fo = ctree.Forest([store.T[v] for v in vids], Tm)
                g_tree = g if direct else g.detach()
                if cfg["objective"] in ("label", "posterior"):
                    # README section 8: the prior is trained by the label only ("label"), or by the label given the
                    # answers to the first k questions the policy asks now, k uniform in 0..primary_budget
                    # ("posterior")
                    A3 = torch.zeros(fo.N, 3, device=device)
                    if cfg["objective"] == "posterior":
                        rows, obs_, lens_ = [], [], []
                        for b, v in enumerate(vids):
                            tr = qtree.tree(store.T[v])
                            k = int(budget_rng.randint(0, int(cfg["primary_budget"]) + 1))
                            for node in asked_train[v][:k]:
                                o = answers[v].get((int(tr["a"][node]), int(tr["b"][node])))
                                if o is not None:
                                    rows.append(int(fo.offs[b]) + int(node))
                                    obs_.append(o)
                                    lens_.append(float(tr["b"][node] - tr["a"][node]))
                        if rows:
                            ll = am.loglik(torch.as_tensor(np.stack(obs_)), torch.as_tensor(np.array(lens_)))
                            A3 = A3.index_put((torch.as_tensor(rows).to(device),), ll.to(A3.dtype))
                    w1, w0 = ctree.up(fo, s_tree, g_tree, A3, chain)
                    logZ = torch.logaddexp(w1, w0)
                    l1, l0 = w1 - logZ, w0 - logZ
                    loss_ans = torch.zeros((), device=device)
                else:
                    ids_g, ans, lens = [], [], []
                    for b, v in enumerate(vids):
                        o = train_obs[v]
                        if len(o[0]):
                            ids_g.append(o[0] + fo.offs[b]); ans.append(o[1]); lens.append(o[2])
                    A3 = torch.zeros(fo.N, 3, device=device)
                    n_obs = torch.zeros(len(vids), device=device)
                    if ids_g:
                        ll = am.loglik(torch.as_tensor(np.concatenate(ans)), torch.as_tensor(np.concatenate(lens)))
                        A3 = A3.index_put((torch.as_tensor(np.concatenate(ids_g)).to(device),), ll.to(A3.dtype))
                        n_obs = torch.as_tensor([len(train_obs[v][0]) for v in vids], device=device).float()
                    l1, l0, lo1, lo0 = ctree.log_evidence(fo, s_tree, g_tree, A3, chain)
                    loss_ans = -(torch.where(label > 0.5, lo1, lo0) / n_obs.clamp(min=1)).mean()
                loss_g = -torch.where(label > 0.5, l1, l0).mean()
                p_video = torch.exp(l1)
                if cfg["objective"] == "mil":        # arm: g trained by BCE outside the tree as in revision 0
                    loss_g = F.binary_cross_entropy_with_logits(g, label)
            else:
                tb = qtree.TreeBatch([store.T[v] for v in vids], [train_obs[v] for v in vids], Tm)
                lp1, lp0 = tb.log_evidence(s_tree, mask, am)
                n_obs = tb.n_obs.clamp(min=1).to(device).float()
                loss_ans = -(torch.where(label > 0.5, lp1, lp0) / n_obs).mean()
                loss_g = F.binary_cross_entropy_with_logits(g, label)
                p_video = torch.sigmoid(g)
            total = loss_g + loss_ans
            if cfg["objective"] == "mil":            # arm: top-k MIL on the per-second logits trains the backbone
                bag = []
                for i in range(len(vids)):
                    t = int(seq[i])
                    k = max(1, -(-t // int(cfg["topk_div"])))
                    bag.append(torch.topk(s[i, :t], k=k).values.mean())
                total = total + F.binary_cross_entropy_with_logits(torch.stack(bag), label)
            cm = torch.zeros((), device=device)
            if lam > 0:
                c1, c2, c3, c4 = CMAL(p_video.detach(), torch.sigmoid(a_log), torch.sigmoid(v_log), seq,
                                      v_out, a_out)      # upstream audio/visual rep order (fix_rep_swap False)
                cm = c1 + c2 + c3 + c4
                total = total + lam * cm
            opt.zero_grad()
            total.backward()
            opt.step()
            tot += [float(loss_g), float(loss_ans), float(cm)]
            nb += 1
        sched.step()
        if cfg["answer_model"] == "refit":
            refit_answer_model(model, am, chain, store, ids["train"], train_obs, labels, device, say)
        tot /= max(nb, 1)
        runs = ev.run(model, am, ids["val"], int(cfg["val_budget"]), chain=chain)
        vm = hc.frame_metrics(at_budget(runs, int(cfg["val_budget"])), gt["val"], hate_val)
        crit = 0.5 * (vm["pooled_ap"] + vm["pooled_roc"])
        history.append({"epoch": epoch + 1, "loss_g": tot[0], "loss_answers": tot[1], "cma": tot[2],
                        "val": vm, "val_criterion": crit, "seconds": time.time() - t0})
        say("epoch %d | g %.4f ans %.4f cma %.4f | val AP %.4f ROC %.4f within %.4f | %.0fs" % (
            epoch + 1, tot[0], tot[1], tot[2], vm["pooled_ap"], vm["pooled_roc"], vm["within_roc"],
            time.time() - t0))
        if best is None or crit > best["crit"]:
            best = {"crit": crit, "epoch": epoch + 1, "model": copy.deepcopy(model.state_dict()),
                    "am": copy.deepcopy(am.state_dict()),
                    "chain": copy.deepcopy(chain.state_dict()) if use_chain else None}
    model.load_state_dict(best["model"])
    am.load_state_dict(best["am"])
    if use_chain:
        chain.load_state_dict(best["chain"])
        if cfg["chain"] != "hazard":
            say("chain: logA %s logpi %s" % (chain.logA().exp().tolist(), chain.logpi().exp().tolist()))
    torch.save({"model": best["model"], "am": best["am"], "chain": best["chain"], "epoch": best["epoch"]},
               os.path.join(out_dir, "model.pth"))
    say("checkpoint: epoch %d (val criterion %.4f)" % (best["epoch"], best["crit"]))
    summary = evaluate(corpus, out_dir, cfg, model, am, ev, ids, gt, labels, say, chain)
    summary.update({"corpus": corpus, "seed": seed, "cfg": cfg, "best_epoch": best["epoch"],
                    "val_criterion": best["crit"], "history": history, "host": socket.gethostname(),
                    "code": git_describe()})
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    return summary


def refit_answer_model(model, am, chain, store, vids, train_obs, labels, device, say):
    """Diagnostic (README section 10): refit the two-state answer model with a length term on the training answers.
    State of each answered node: negative video -> 0; positive video -> P(z_n = 1 | Y = 1, x) from the chain prior
    WITHOUT any answer (so an answer never decides its own state); weighted maximum likelihood (qtree.fit_weighted)."""
    model.eval()
    vids = sorted(vids, key=lambda v: store.T[v])
    O, U, W1 = [], [], []
    with torch.no_grad():
        pri = {v: policy.video_prior(model, store, v, device) for v in vids}
    for i in range(0, len(vids), 64):
        vs = vids[i:i + 64]
        Ts = [store.T[v] for v in vs]
        Tm = max(Ts)
        S = torch.zeros(len(vs), Tm, dtype=torch.float64)
        G = torch.zeros(len(vs), dtype=torch.float64)
        for b, v in enumerate(vs):
            S[b, :Ts[b]] = torch.from_numpy(pri[v][0])
            G[b] = pri[v][1]
        fo = ctree.Forest(Ts, Tm)
        _, m, _ = ctree.marginals(fo, S.to(device), G.to(device), torch.zeros(fo.N, 3, device=device), chain)
        m = m.cpu().numpy()
        for b, v in enumerate(vs):
            nid, oo, ln = train_obs[v]
            if not len(nid):
                continue
            O.append(oo)
            U.append((np.log(ln) - am.len_mu) / am.len_sd)
            W1.append(np.zeros(len(nid)) if labels[v] == 0 else m[fo.offs[b] + nid])
    O, U, W1 = np.concatenate(O), np.concatenate(U), np.concatenate(W1)
    t0, o0 = qtree.fit_weighted(O, U, 1.0 - W1)
    t1, o1 = qtree.fit_weighted(O, U, W1)
    with torch.no_grad():
        am.theta.copy_(torch.as_tensor(np.stack([t0, t1], 1), dtype=am.theta.dtype))
        am.omega.copy_(torch.as_tensor(np.stack([o0, o1], 1), dtype=am.omega.dtype))
    model.train()


def evaluate(corpus, out_dir, cfg, model, am, ev, ids, gt, labels, say, chain=None):
    max_calls = int(cfg["max_calls"])
    val_runs = ev.run(model, am, ids["val"], max_calls, chain=chain, record_voi=chain is not None)
    test_runs = ev.run(model, am, ids["test"], max_calls, chain=chain, record_voi=chain is not None)
    hate_val = {v for v in ids["val"] if labels[v] == 1}
    res = {"fixed": {}, "adaptive": {}, "adaptive_voi": {}}

    def test_eval(name, scores):
        sp = os.path.join(out_dir, "scores_test_%s.jsonl" % name)
        hc.write_scores(sp, scores)
        r = hc.run_evaluator(corpus, "test", sp, os.path.join(out_dir, "metrics_test_%s.json" % name))
        r = r["results"]["score_av"]
        return {"pooled_ap": r["pr_auc"], "pooled_roc": r["roc_auc"], "within_roc": r["per_video"]["macro_auc"]}

    for B in cfg["fixed_budgets"]:
        B = int(B)
        sv, st = at_budget(val_runs, B), at_budget(test_runs, B)
        res["fixed"][str(B)] = {
            "val": hc.frame_metrics(sv, gt["val"], hate_val), "test": test_eval("fixed%d" % B, st),
            "test_mean_calls": float(np.mean([min(B, len(r["eig"])) for r in test_runs.values()]))}
        say("fixed %2d calls | test AP %.4f ROC %.4f within %.4f" % (B, *[res["fixed"][str(B)]["test"][k] for k in
                                                                        ("pooled_ap", "pooled_roc", "within_roc")]))
    rules = [("adaptive", "eig")] + ([("adaptive_voi", "voi")] if chain is not None else [])
    for (key, rule), B in [(kr, B) for kr in rules for B in cfg["mean_budgets"]]:
        c, val_mean = policy.calibrate([r[rule] for r in val_runs.values()], float(B))
        sv, _ = at_threshold(val_runs, c, rule)
        st, calls = at_threshold(test_runs, c, rule)
        cv = np.array(list(calls.values()))
        res[key][str(B)] = {
            "c_bits": c, "val_mean_calls": val_mean, "val": hc.frame_metrics(sv, gt["val"], hate_val),
            "test": test_eval("%s%d" % (key, B), st), "test_mean_calls": float(cv.mean()),
            "test_calls_quantiles": [float(x) for x in np.percentile(cv, [0, 25, 50, 75, 100])],
            "test_mean_calls_pos": float(np.mean([calls[v] for v in calls if labels[v] == 1])),
            "test_mean_calls_neg": float(np.mean([calls[v] for v in calls if labels[v] == 0]))}
        r = res[key][str(B)]
        say("%s mean %d (c = %.4g) | test calls %.2f (pos %.2f, neg %.2f) | AP %.4f ROC %.4f within %.4f"
            % (key, B, c, r["test_mean_calls"], r["test_mean_calls_pos"], r["test_mean_calls_neg"],
               r["test"]["pooled_ap"], r["test"]["pooled_roc"], r["test"]["within_roc"]))
    pb = int(cfg["primary_budget"])
    primary = res["fixed"][str(pb)]
    # P3: VLM-silent group of the old fine verdicts (fine rate of level >= 2 below .1)
    old = vlm_verdict.load_verdicts(corpus, k=30, tag="qwen")
    silent = [v for v in ids["test"] if v in old and np.mean(old[v] >= 2) < 0.1]
    st = at_budget(test_runs, pb)
    res["silent_group"] = {"n": len(silent), "n_pos": int(sum(labels[v] for v in silent)),
                           "ap_primary": group_ap(st, gt["test"], silent),
                           "ap_prior_only": group_ap(at_budget(test_runs, 0), gt["test"], silent)}
    res["test_runs"] = {v: {"eig": r["eig"], "voi": r.get("voi", []), "asked": r["asked"], "p_G": r["p_G"]}
                        for v, r in test_runs.items()}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({"test": primary["test"], "test_mean_calls": primary["test_mean_calls"],
                   "primary_budget": int(cfg["primary_budget"])}, fh, indent=2)
    res["test"] = dict(primary["test"])
    res["test_mean_calls"] = primary["test_mean_calls"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    if a.config:
        given = json.load(open(a.config))
        unknown = set(given) - set(DEFAULTS)
        assert not unknown, "unknown config keys %s" % sorted(unknown)
        cfg.update(given)
    train(a.corpus, a.seed, a.out_dir, cfg, a.device, a.num_workers)


if __name__ == "__main__":
    main()
