"""Revision-5 premise check (README section 16.4; development evidence on test under rule 10: test answers and test
GT only for the numbers; nothing here trains the network or selects anything).

Does the perception layer (ptree.py) fix what revision 4 gets wrong with nested questions? Each revision-4 trial's
network (s, g, node potentials) and chain are reused unchanged; only the answer side changes:
  1. fitted once per corpus without GT, on the training answers whose state the label fixes: the answer tables
     P(o | w) (two states, five categories, the anchored parametrisation) and the per-second perception rate f0 of
     negative videos, by the exact likelihood log P(all answers of a negative training video | G = 0) (ptree.up0)
     plus log P(root answer | w = 1) of every positive training video;
  2. fitted per trial without GT: h = P(v = 1 | y = 1) and f = P(v = 1 | y = 0, G = 1) by the exact likelihood of
     all answers of the positive training videos given G = 1 under the trial's network prior (ptree.up1);
  3. test: the same fixed-budget EIG policy with the perception posterior (questions by EIG of w_n).
Pooled AP / ROC / within through the shared evaluator, Spearman of each positive video's mean score with its GT
hate fraction, share of the first 8 questions nested with an earlier one, median asked length.

    python experiments/20260925_query_paradigm/perception_check.py --corpus hatemm --trials <trial dirs>
Writes runs/20260928_query_paradigm_r5_analysis/perception/<corpus>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import concern_diagnostics as cd                # noqa: E402  (sets up the import paths)
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402
import qtree                                    # noqa: E402
import ctree                                    # noqa: E402
import cpolicy                                  # noqa: E402
import policy                                   # noqa: E402
import ptree                                    # noqa: E402
from tendency_check import nested_share         # noqa: E402

BUDGETS = (0, 1, 2, 4, 8, 16, 32)
OUT = os.path.join(ROOT, "runs", "20260928_query_paradigm_r5_analysis", "perception")
CHUNK = 64


def node_rows(answers_v, T):
    """(node ids, answers (n, C)) of the answered nodes of one video."""
    ids, ans = qdata.observed(answers_v, T)[:2]
    return torch.as_tensor(ids), torch.as_tensor(ans)


def a2_rows(theta, ans, cats):
    """(n, 2) log P(o | w) of answers (n, C) under theta (C, 2, L)."""
    lp = F.log_softmax(theta, -1)                                         # C, 2, L
    C = theta.shape[0]
    per = lp[torch.arange(C)[None, :, None], torch.arange(2)[None, None, :], ans[:, :, None]]   # n, C, 2
    return per[:, cats].sum(1)


def chunks(vids, T):
    order = sorted(vids, key=lambda v: T[v])
    return [order[i:i + CHUNK] for i in range(0, len(order), CHUNK)]


def fit_answer_side(answers, labels, T, train, cats, theta0, device):
    """Stage 1: theta (C, 2, L) and f0 from negative training videos (all answers, G = 0) and positive roots."""
    neg = [v for v in train if labels[v] == 0 and T[v] >= qtree.F_FRAMES]
    roots = [answers[v].get((0, T[v])) for v in train if labels[v] == 1 and T[v] >= qtree.F_FRAMES]
    roots = torch.as_tensor(np.stack([o for o in roots if o is not None])).to(device)
    groups = []
    for ch in chunks(neg, T):
        fo = ctree.Forest([T[v] for v in ch], max(T[v] for v in ch))
        ids, ans = [], []
        for b, v in enumerate(ch):
            i, o = node_rows(answers[v], T[v])
            ids.append(i + int(fo.offs[b]))
            ans.append(o)
        groups.append((fo, torch.cat(ids).to(device), torch.cat(ans).to(device)))
    th = torch.as_tensor(theta0, dtype=torch.float64, device=device).clone().requires_grad_(True)
    lf0 = torch.tensor(-5.0, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([th, lf0], max_iter=300, tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -a2_rows(th, roots, cats)[:, 1].sum() - F.log_softmax(th, -1).sum()
        for fo, ids, ans in groups:
            A2 = torch.zeros(fo.N, 2, dtype=torch.float64, device=device).index_put((ids,), a2_rows(th, ans, cats))
            loss = loss - ptree.up0(fo, A2, torch.sigmoid(lf0)).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return th.detach(), torch.sigmoid(lf0).detach(), len(neg), len(roots)


def prior_batch(model, store, ch, device):
    Ts = [store.T[v] for v in ch]
    fo = ctree.Forest(Ts, max(Ts))
    S = torch.zeros(len(ch), max(Ts), dtype=torch.float64)
    G = torch.zeros(len(ch), dtype=torch.float64)
    PHI = torch.zeros(fo.N, dtype=torch.float64)
    for b, v in enumerate(ch):
        s, g, phi = policy.video_prior_nodes(model, store, v, device)
        S[b, :len(s)] = torch.from_numpy(s)
        G[b] = g
        if phi is not None:
            PHI[int(fo.offs[b]):int(fo.offs[b]) + len(phi)] = torch.from_numpy(phi)
    return fo, S.to(device), G.to(device), PHI.to(device)


def fit_rates(model, store, answers, pos_train, th, logA, cats, device):
    """Stage 2: h and f from all answers of the positive training videos given G = 1 (network prior fixed)."""
    groups = []
    for ch in chunks(pos_train, store.T):
        fo, S, _, PHI = prior_batch(model, store, ch, device)
        ids, ans = [], []
        for b, v in enumerate(ch):
            i, o = node_rows(answers[v], store.T[v])
            ids.append(i + int(fo.offs[b]))
            ans.append(o)
        A2 = torch.zeros(fo.N, 2, dtype=torch.float64, device=device)
        A2 = A2.index_put((torch.cat(ids).to(device),), a2_rows(th, torch.cat(ans).to(device), cats))
        groups.append((fo, S, PHI, A2))
    lh = torch.tensor(1.0, dtype=torch.float64, device=device, requires_grad=True)
    lf = torch.tensor(-3.0, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([lh, lf], max_iter=100, tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        h, f = torch.sigmoid(lh), torch.sigmoid(lf)
        loss = 0.0
        for fo, S, PHI, A2 in groups:
            loss = loss - (ptree.up1(fo, S, A2, logA, h, f, PHI)
                           - ptree.up1(fo, S, torch.zeros_like(A2), logA, h, f, PHI)).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return torch.sigmoid(lh).detach(), torch.sigmoid(lf).detach()


def run_chunk(model, store, ch, answers, table, logA, h, f, f0, cats, max_calls, device):
    fo, S, G, PHI = prior_batch(model, store, ch, device)
    Ts = [store.T[v] for v in ch]
    vts = [cpolicy._vt(T) for T in Ts]
    po_w = np.exp(table)                                                  # 2, O
    h_w = -np.sum(po_w * table, axis=1)                                   # 2
    A2 = torch.zeros(fo.N, 2, dtype=torch.float64, device=device)
    v1p = ptree.up1(fo, S, torch.zeros_like(A2), logA, h, f, PHI)
    out = [{"scores": [], "eig": [], "asked": []} for _ in ch]
    rem = [np.ones(len(vt.q_ids), dtype=bool) for vt in vts]
    for step in range(max_calls + 1):
        pG, m1, m0, p = ptree.posterior(fo, S, G, A2, logA, h, f, f0, PHI, v1p)
        pG, m1, m0, p = pG.cpu().numpy(), m1.cpu().numpy(), m0.cpu().numpy(), p.cpu().numpy()
        for b, T in enumerate(Ts):
            if step <= len(vts[b].q_ids):
                out[b]["scores"].append(p[b, :T].copy())
        if step == max_calls:
            break
        for b, off in enumerate(fo.offs):
            cand = np.where(rem[b])[0]
            if len(cand) == 0:
                continue
            nodes = vts[b].q_ids[cand]
            pw = pG[b] * m1[off + nodes] + (1.0 - pG[b]) * m0[off + nodes]
            pw = np.clip(pw, 0.0, 1.0)
            w = np.stack([1.0 - pw, pw], axis=1)                          # n, 2
            po = w @ po_w
            e = (-np.sum(po * np.log(np.clip(po, 1e-300, None)), axis=1) - w @ h_w) / np.log(2.0)
            j = int(np.argmax(e))
            node = int(nodes[j])
            rem[b][cand[j]] = False
            tr = vts[b].tr
            o = answers[ch[b]].get((int(tr["a"][node]), int(tr["b"][node])))
            if o is not None:
                A2[int(off) + node] = torch.as_tensor(table[:, qtree.answer_index(np.asarray(o)[cats])],
                                                      dtype=torch.float64, device=device)
            out[b]["eig"].append(float(e[j]))
            out[b]["asked"].append(node)
    return {v: r for v, r in zip(ch, out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--trials", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    answers, T_ans = qdata.load_answers(a.corpus, "words")
    vids = list(ids["test"])
    frac = {v: float(np.mean(gt["test"][v])) for v in vids}
    pos = [v for v in vids if labels[v] == 1 and frac[v] > 0]
    os.makedirs(OUT, exist_ok=True)
    res = {"corpus": a.corpus, "trials": {}}
    store = None
    fitted = None
    for trial in a.trials:
        summ, cfg, model, am, chain = cd.load_trial(trial, dev, answers, ids)
        cats = list(cfg["categories"])
        assert chain.closed and chain.zero_inflated and cfg["answer_model"] == "anchored"
        if store is None:
            store = qdata.Store(a.corpus, ids["train"] + vids, cfg.get("text_sources", ["bert"]))
        if fitted is None:
            t0 = time.time()
            th, f0, n_neg, n_root = fit_answer_side(answers, labels, T_ans, ids["train"], cats,
                                                    am.theta.detach().cpu().numpy(), dev)
            lp = F.log_softmax(th, -1).cpu().numpy()[cats]                # C', 2, L
            table = qtree.outcome_table(lp[None])[0]                      # 2, O
            yes = [float(1 - np.exp(lp[0, w, 0])) for w in (0, 1)]
            fitted = (th, f0, table)
            res["answer_side"] = {"f0": float(f0), "n_negative_videos": n_neg, "n_positive_roots": n_root,
                                  "p_hate_level_ge1_given_w": yes, "seconds": time.time() - t0}
            print("%s answer side: f0 %.4f, P(hate level >= 1 | w = 0 / 1) %.3f / %.3f (%d negatives, %d roots, "
                  "%.0f s)" % (a.corpus, float(f0), yes[0], yes[1], n_neg, n_root, time.time() - t0), flush=True)
        th, f0, table = fitted
        logA = chain.logA().detach().double().to(dev)
        t0 = time.time()
        pos_train = [v for v in ids["train"] if labels[v] == 1 and store.T[v] >= qtree.F_FRAMES]
        h, f = fit_rates(model, store, answers, pos_train, th, logA, cats, dev)
        tag = "_".join(trial.rstrip("/").split("/")[-2:])
        print("== %s: h %.4f f %.4f (f0 %.4f; %.0f s)" % (tag, float(h), float(f), float(f0), time.time() - t0),
              flush=True)
        runs = {}
        for ch in chunks(vids, store.T):
            runs.update(run_chunk(model, store, ch, answers, table, logA, h, f, f0, cats, max(BUDGETS), dev))
        od = os.path.join(OUT, a.corpus, tag)
        os.makedirs(od, exist_ok=True)
        r = {"h": float(h), "f": float(f), "f0": float(f0), "method": {}, "perception": {}}
        for B in BUDGETS:
            sc = {v: runs[v]["scores"][min(B, len(runs[v]["eig"]))] for v in vids}
            sp = os.path.join(od, "scores_test_fixed%d.jsonl" % B)
            hc.write_scores(sp, sc)
            m = hc.run_evaluator(a.corpus, "test", sp, os.path.join(od, "metrics_test_fixed%d.json" % B))
            m = m["results"]["score_av"]
            r["perception"][str(B)] = {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"],
                                       "within_roc": m["per_video"]["macro_auc"],
                                       "fraction_spearman": float(spearmanr(
                                           [frac[v] for v in pos], [float(sc[v].mean()) for v in pos]).correlation)}
            r["method"][str(B)] = {k: summ["fixed"][str(B)]["test"][k] for k in ("pooled_ap", "pooled_roc",
                                                                                   "within_roc")}
        n8 = [nested_share(runs[v]["asked"][:8], qtree.tree(store.T[v])) for v in vids]
        r["nested_share_first8"] = float(np.mean([x for x in n8 if x is not None]))
        r["median_asked_len_first8"] = float(np.median([int(qtree.tree(store.T[v])["b"][n]
                                                            - qtree.tree(store.T[v])["a"][n])
                                                        for v in vids for n in runs[v]["asked"][:8]]))
        print("  nested share %.3f, median asked length %.0f s" % (r["nested_share_first8"],
                                                                    r["median_asked_len_first8"]), flush=True)
        for B in map(str, BUDGETS):
            x, y = r["perception"][B], r["method"][B]
            print("  B=%-2s perception %.4f / %.4f / %.4f (fraction Spearman %.3f) | method %.4f / %.4f / %.4f" % (
                B, x["pooled_ap"], x["pooled_roc"], x["within_roc"], x["fraction_spearman"], y["pooled_ap"],
                y["pooled_roc"], y["within_roc"]), flush=True)
        res["trials"][trial] = r
        json.dump(res, open(os.path.join(OUT, "%s.json" % a.corpus), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
