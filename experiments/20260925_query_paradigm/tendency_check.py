"""Revision-5 premise check (README section 16.4; development evidence on test under rule 10: reads test answers and
test GT for the numbers; nothing here trains the network or selects anything).

Finding (README 16.4): the VLM's answers inside one video are not independent given the node states. In positive
test videos the per-video "yes" rate on GT-clean short nodes and on fully hateful short nodes correlate across
videos (Pearson .65 HateMM, .55 HCS), and the per-video yes rates vary 5-30 times more than independent answers
would (negative videos too). Revision 4's anchored answer model treats every answer as independent evidence.

Model checked here (no GT): a per-video VLM tendency beta_v on a fixed grid (K values in [-b, b]) with learned
weights pi_k, shared by all nodes and categories of the video; it adds beta_v to the logits of the answer levels
1-3 (P(o_nc = l | s, beta) = softmax_l(theta[c, s, l] + beta 1[l >= 1])). theta (two anchored states) and pi are
fitted by maximum marginal likelihood on the training answers whose state the video label fixes (every node of a
negative video = state 0, the root of a positive video = state 1), beta integrated out per video; the same
pseudo-counts as qtree.fit_anchored.
Inference: one exact tree pass per beta value (ctree, unchanged), mixed by P(beta | answers so far); questions by
EIG over (node state, beta) jointly. The revision-4 network, chain and node potentials of each trial are reused
unchanged, so any change comes from the answer model alone. Variant "anchored" runs the same code with K = 1 and
the trial's own answer model and must reproduce the trial's summary numbers.

    python experiments/20260925_query_paradigm/tendency_check.py --corpus hatemm --trials <trial dirs>
Writes runs/20260928_query_paradigm_r5_analysis/tendency/<corpus>.json (score / metric files under
<corpus>/<variant>/<trial tag>/).
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

BUDGETS = (0, 1, 2, 4, 8, 16, 32)
OUT = os.path.join(ROOT, "runs", "20260928_query_paradigm_r5_analysis", "tendency")
LEVEL_SHIFT = torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float64)


def anchored_rows(answers, labels, vids):
    """Per training video with a label-fixed state: (state per answer, answers (n, C))."""
    out = []
    for v in vids:
        if not answers[v]:                       # shorter than F seconds: no queryable node
            continue
        T = max(b for (a, b) in answers[v])
        rows = [(0 if labels[v] == 0 else 1, o) for (a, b), o in answers[v].items()
                if o is not None and (labels[v] == 0 or (a == 0 and b == T))]
        if rows:
            out.append((np.array([r[0] for r in rows]), np.stack([r[1] for r in rows])))
    return out


def fit_tendency(rows, betas, theta0):
    """Maximum marginal likelihood of theta (C, 2, L) and log pi (K,) with beta integrated out per video."""
    K = len(betas)
    S = torch.as_tensor(np.concatenate([r[0] for r in rows]), dtype=torch.long)
    O = torch.as_tensor(np.concatenate([r[1] for r in rows]), dtype=torch.long)
    vid = torch.as_tensor(np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(rows)]))
    B = torch.as_tensor(betas, dtype=torch.float64)
    th = torch.as_tensor(theta0, dtype=torch.float64).clone().requires_grad_(True)
    lpi = torch.as_tensor(-0.5 * np.asarray(betas) ** 2, dtype=torch.float64).clone().requires_grad_(True)
    C = th.shape[0]
    opt = torch.optim.LBFGS([th, lpi], max_iter=1000, tolerance_grad=1e-10, tolerance_change=1e-13,
                            line_search_fn="strong_wolfe")

    def nll():
        logits = th[None] + B[:, None, None, None] * LEVEL_SHIFT                  # K, C, 2, L
        lp = F.log_softmax(logits, -1)
        per = lp[:, torch.arange(C)[None, :], S[:, None], O]                        # K, n, C
        ll = per.sum(-1)                                                            # K, n
        vl = torch.zeros(K, len(rows), dtype=torch.float64).index_add(1, vid, ll)   # K, V
        mix = torch.logsumexp(F.log_softmax(lpi, 0)[:, None] + vl, 0)
        return -mix.sum() - F.log_softmax(th, -1).sum(), mix

    def closure():
        opt.zero_grad()
        loss, _ = nll()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        _, mix = nll()
    return th.detach().numpy(), F.log_softmax(lpi, 0).detach().numpy(), float(mix.sum())


def tables(theta, betas, cats):
    """(K, 3, L**C') log P(o | s, beta_k) over answer vectors of the categories `cats`; the two anchored states
    mapped to the three inference states as in policy.Asker (s = 1 shares s = 0)."""
    th = torch.as_tensor(theta, dtype=torch.float64)
    out = []
    for b in betas:
        lp = F.log_softmax(th + float(b) * LEVEL_SHIFT, -1).numpy()[cats]         # C', 2, L
        lp = lp[:, [0, 0, 1]]                                                      # C', 3, L
        out.append(qtree.outcome_table(lp[None])[0])                                # 3, O
    return np.stack(out)


def nested_share(asked, tr):
    """Share of asked nodes (after the first) that lie inside, or contain, a node asked before them."""
    if len(asked) < 2:
        return None
    a, b = tr["a"], tr["b"]
    hit = 0
    for i in range(1, len(asked)):
        n = asked[i]
        hit += any((a[m] <= a[n] and b[n] <= b[m]) or (a[n] <= a[m] and b[m] <= b[n]) for m in asked[:i])
    return hit / (len(asked) - 1)


@torch.no_grad()
def run_batch_mix(model, store, vids, tab, log_pi, chain, answers, cats, max_calls, device):
    """cpolicy.run_batch (EIG, tree fusion) with the answer model mixed over the per-video tendency grid."""
    K = tab.shape[0]
    Ts = [store.T[v] for v in vids]
    Tm = max(Ts)
    S = torch.zeros(len(vids), Tm, dtype=torch.float64)
    G = torch.zeros(len(vids), dtype=torch.float64)
    fo = ctree.Forest(Ts, Tm)
    PHI = None
    for b, v in enumerate(vids):
        s, g, phi = policy.video_prior_nodes(model, store, v, device)
        S[b, :len(s)] = torch.from_numpy(s)
        G[b] = g
        if phi is not None:
            if PHI is None:
                PHI = torch.zeros(fo.N, dtype=torch.float64)
            PHI[int(fo.offs[b]):int(fo.offs[b]) + len(phi)] = torch.from_numpy(phi)
    S, G = S.to(device), G.to(device)
    PHI = None if PHI is None else PHI.to(device)
    dchain = ctree._DoubleChain(chain)
    vts = [cpolicy._vt(T) for T in Ts]
    po_s = np.exp(tab)                                                              # K, 3, O
    h_s = -np.sum(po_s * tab, axis=2)                                               # K, 3
    A3 = torch.zeros(K, fo.N, 3, dtype=torch.float64, device=device)
    out = [{"scores": [], "eig": [], "asked": [], "p_G": [], "beta_mean": []} for _ in vids]
    rem = [np.ones(len(vt.q_ids), dtype=bool) for vt in vts]
    betas_used = None
    for step in range(max_calls + 1):
        lr, pGk, mk, pk = [], [], [], []
        for k in range(K):
            pG, m, p = ctree.marginals(fo, S, G, A3[k], chain, PHI)
            w1, w0 = ctree.up(fo, S, G, A3[k], dchain, phi=PHI)
            lr.append(torch.logaddexp(w1, w0).cpu().numpy())
            pGk.append(pG.cpu().numpy())
            mk.append(m.cpu().numpy())
            pk.append(p.cpu().numpy())
        lr = np.stack(lr) + log_pi[:, None]                                         # K, B
        r = np.exp(lr - lr.max(0, keepdims=True))
        r /= r.sum(0, keepdims=True)
        pGk, mk, pk = np.stack(pGk), np.stack(mk), np.stack(pk)                    # K,B / K,N / K,B,Tm
        p = np.einsum("kb,kbt->bt", r, pk)
        pG = np.einsum("kb,kb->b", r, pGk)
        for b, T in enumerate(Ts):
            if step <= len(vts[b].q_ids):
                out[b]["scores"].append(p[b, :T].copy())
                out[b]["p_G"].append(float(pG[b]))
                out[b]["beta_mean"].append(r[:, b].tolist())
        if step == max_calls:
            break
        chosen = []
        for b, off in enumerate(fo.offs):
            cand = np.where(rem[b])[0]
            if len(cand) == 0:
                continue
            nodes = vts[b].q_ids[cand]
            mm = mk[:, off + nodes]                                                 # K, n
            w = np.stack([np.broadcast_to((1.0 - pGk[:, b])[:, None], mm.shape), pGk[:, b][:, None] * (1.0 - mm),
                          pGk[:, b][:, None] * mm], axis=2) * r[:, b][:, None, None]  # K, n, 3
            po = np.einsum("kns,kso->no", w, po_s)
            h = -np.sum(po * np.log(np.clip(po, 1e-300, None)), axis=1)
            e = (h - np.einsum("kns,ks->n", w, h_s)) / np.log(2.0)
            j = int(np.argmax(e))
            chosen.append((b, int(cand[j]), int(nodes[j]), float(e[j])))
        if not chosen:
            break
        for b, pos, node, e in chosen:
            rem[b][pos] = False
            tr = vts[b].tr
            o = answers[vids[b]].get((int(tr["a"][node]), int(tr["b"][node])))
            if o is not None:
                idx = qtree.answer_index(np.asarray(o)[cats])
                A3[:, fo.offs[b] + node] = torch.as_tensor(tab[:, :, idx], dtype=torch.float64, device=device)
            out[b]["eig"].append(e)
            out[b]["asked"].append(node)
    return {v: r_ for v, r_ in zip(vids, out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--trials", nargs="+", required=True)
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--beta-max", type=float, default=4.0)
    ap.add_argument("--variants", default="anchored,tendency")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    answers, _ = qdata.load_answers(a.corpus, "words")
    vids = list(ids["test"])
    frac = {v: float(np.mean(gt["test"][v])) for v in vids}
    pos = [v for v in vids if labels[v] == 1 and frac[v] > 0]

    rows = anchored_rows(answers, labels, ids["train"])
    th_anch, _, _ = qtree.fit_anchored([(labels[v], [(x, y, max(b for (_, b) in answers[v]), o)
                                                     for (x, y), o in answers[v].items()]) for v in ids["train"]
                                        if answers[v]],
                                       0.0, 1.0, False)
    betas = np.linspace(-a.beta_max, a.beta_max, a.K)
    th_t, log_pi, ll_t = fit_tendency(rows, betas, th_anch)
    _, _, ll_1 = fit_tendency(rows, np.zeros(1), th_anch)
    res = {"corpus": a.corpus, "betas": betas.tolist(), "pi": np.exp(log_pi).tolist(),
           "train_marginal_loglik": {"tendency": ll_t, "no_tendency": ll_1, "n_videos": len(rows)},
           "trials": {}}
    print("%s: fitted tendency on %d training videos; marginal log-lik %.1f (no tendency %.1f); pi %s" % (
        a.corpus, len(rows), ll_t, ll_1, np.round(np.exp(log_pi), 3).tolist()), flush=True)
    os.makedirs(OUT, exist_ok=True)
    store = None
    for trial in a.trials:
        summ, cfg, model, am, chain = cd.load_trial(trial, a.device, answers, ids)
        assert not cfg["length_term"] and int(cfg["n_state"]) == 2 and cfg["answer_model"] == "anchored"
        cats = list(cfg["categories"])
        if store is None:
            store = qdata.Store(a.corpus, vids, cfg.get("text_sources", ["bert"]))
        tag = "_".join(trial.rstrip("/").split("/")[-2:])
        rt = {"method": {str(B): {m: summ["fixed"][str(B)]["test"][m] for m in ("pooled_ap", "pooled_roc",
                                                                                 "within_roc")} for B in BUDGETS}}
        for variant in a.variants.split(","):
            t0 = time.time()
            if variant == "anchored":
                tab = tables(am.theta.detach().cpu().numpy(), [0.0], cats)
                lp = np.zeros(1)
            else:
                tab = tables(th_t, betas, cats)
                lp = log_pi
            runs = {}
            order = sorted(vids, key=lambda v: store.T[v])
            k = int(cfg["eval_chunk"])
            for i in range(0, len(order), k):
                runs.update(run_batch_mix(model, store, order[i:i + k], tab, lp, chain, answers, cats,
                                          max(BUDGETS), a.device))
            od = os.path.join(OUT, a.corpus, variant, tag)
            os.makedirs(od, exist_ok=True)
            r = {}
            for B in BUDGETS:
                sc = {v: runs[v]["scores"][min(B, len(runs[v]["eig"]))] for v in vids}
                sp = os.path.join(od, "scores_test_fixed%d.jsonl" % B)
                hc.write_scores(sp, sc)
                m = hc.run_evaluator(a.corpus, "test", sp, os.path.join(od, "metrics_test_fixed%d.json" % B))
                m = m["results"]["score_av"]
                r[str(B)] = {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"],
                             "within_roc": m["per_video"]["macro_auc"],
                             "fraction_spearman": float(spearmanr([frac[v] for v in pos],
                                                                  [float(sc[v].mean()) for v in pos]).correlation)}
            n8 = [nested_share(runs[v]["asked"][:8], qtree.tree(store.T[v])) for v in vids]
            r["nested_share_first8"] = float(np.mean([x for x in n8 if x is not None]))
            lens = [int(qtree.tree(store.T[v])["b"][n] - qtree.tree(store.T[v])["a"][n])
                    for v in vids for n in runs[v]["asked"][:8]]
            r["median_asked_len_first8"] = float(np.median(lens))
            r["seconds"] = time.time() - t0
            rt[variant] = r
            print("== %s %s (%.0f s): nested share %.3f, median asked length %.0f s" % (
                tag, variant, r["seconds"], r["nested_share_first8"], r["median_asked_len_first8"]), flush=True)
            for B in map(str, BUDGETS):
                x = r[B]
                print("  B=%-2s %.4f / %.4f / %.4f | fraction Spearman %.3f | method %.4f / %.4f / %.4f" % (
                    B, x["pooled_ap"], x["pooled_roc"], x["within_roc"], x["fraction_spearman"],
                    *[rt["method"][B][q] for q in ("pooled_ap", "pooled_roc", "within_roc")]), flush=True)
            res["trials"][trial] = rt
            json.dump(res, open(os.path.join(OUT, "%s.json" % a.corpus), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
