"""Diagnostic (README section 6): is the tree's greedy question order short-sighted under its own model?
Revision-6 test-selected trials; per test video and level L, two fixed question sets of the same size k_v (the E3
matched calls): the tree's first k_v asked nodes and the single level's k_v nodes (replays runs_L0.pkl /
runs_L<L>.pkl of experiments/20261108_astra_review). Under the revision-6 model (prior, chain, anchored answer
table, answers independent given node states), the joint information of each set about the video's states,
I(o_S; X) = H(o_S) - sum_n H(o_n | s_n), with H(o_S) estimated by Monte Carlo: M answer sequences sampled one node at
a time from the model's predictive P(o_i | o_<i) (exact tree inference after each sampled answer). If the level's
set carries more information than the tree's set, greedy EIG is short-sighted (it takes the best single question,
not the best set); if not, the model itself does not value covering the video.
Output runs/20261109_tree_within/joint_info/<corpus>.json: per level, mean bits of both sets (all videos and positive
videos), share of videos where the level's set has more, per trial.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import cpolicy, ctree, data as qdata, policy                   # noqa: E402
from qtl.backbones import make_model                                    # noqa: E402
from qtl.checkpoint import load_trial                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

REPLAY = ROOT / "runs/20261108_astra_review/replay"


def joint_info(model, store, vids, am, chain, cats, sets, M, rng):
    """Per video: I(o_S; X) in bits for the node list sets[v] (Monte Carlo over M answer sequences)."""
    out = {}
    vids = sorted(vids, key=lambda v: store.T[v])
    chunk_n = max(1, 512 // M)
    for i in range(0, len(vids), chunk_n):
        chunk = vids[i:i + chunk_n]
        Ts = [store.T[v] for v in chunk for _ in range(M)]
        fo = ctree.Forest(Ts, max(Ts))
        S = torch.zeros(len(Ts), max(Ts), dtype=torch.float64)
        G = torch.zeros(len(Ts), dtype=torch.float64)
        PHI = None
        askers, nodes = [], []
        for j, v in enumerate(chunk):
            s, g, phi = policy.video_prior_nodes(model, store, v, "cpu")
            vt = cpolicy._vt(store.T[v])
            ak = policy.Asker(vt, am, cats)
            for r in range(M):
                b = j * M + r
                S[b, :len(s)] = torch.from_numpy(s)
                G[b] = g
                if phi is not None:
                    if PHI is None:
                        PHI = torch.zeros(fo.N, dtype=torch.float64)
                    PHI[int(fo.offs[b]):int(fo.offs[b]) + len(phi)] = torch.from_numpy(phi)
            askers.append(ak)
            nodes.append([int(n) for n in sets[v]])
        A3 = torch.zeros(fo.N, 3, dtype=torch.float64)
        logp = np.zeros(len(Ts))
        hcond = np.zeros(len(chunk))
        kmax = max(len(n) for n in nodes)
        for step in range(kmax):
            pG, m, _ = ctree.marginals(fo, S, G, A3, chain, PHI)
            pG, m = pG.numpy(), m.numpy()
            for j, v in enumerate(chunk):
                if step >= len(nodes[j]):
                    continue
                n = nodes[j][step]
                pos = askers[j].qpos[n]
                po = askers[j].po_s[pos]                                  # 3, O
                for r in range(M):
                    b = j * M + r
                    mm = m[fo.offs[b] + n]
                    w = np.array([1.0 - pG[b], pG[b] * (1.0 - mm), pG[b] * mm])
                    pred = w @ po
                    pred = pred / pred.sum()
                    o = int(rng.choice(len(pred), p=pred))
                    logp[b] += np.log(pred[o])
                    A3[fo.offs[b] + n] = torch.as_tensor(askers[j].table[pos, :, o], dtype=torch.float64)
        # H(o_S | X) = sum_n H(o_n | s_n) under the prior marginal of s_n (no answers)
        pG0, m0, _ = ctree.marginals(fo, S, G, torch.zeros_like(A3), chain, PHI)
        pG0, m0 = pG0.numpy(), m0.numpy()
        for j, v in enumerate(chunk):
            b = j * M
            for n in nodes[j]:
                pos = askers[j].qpos[n]
                mm = m0[fo.offs[b] + n]
                w = np.array([1.0 - pG0[b], pG0[b] * (1.0 - mm), pG0[b] * mm])
                hcond[j] += float(w @ askers[j].h_s[pos])
            hs = -logp[j * M:(j + 1) * M].mean()
            out[v] = float((hs - hcond[j]) / np.log(2.0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--levels", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-root", default="runs/20261109_tree_within/joint_info")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(0)
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))
    trials = [r["trial"] for r in summ["trials"].values()]
    labels, ids, _gt, _ = hc.load_fixed_cohort(a.corpus)
    test = list(ids["test"])
    qdata.configure_source("soft_both", 8)
    answers, _ = qdata.load_answers(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, test, ["bert"])
    res = {}
    for td in trials:
        trial = Path(td).resolve()
        summary, cfg, model, am, chain = load_trial(trial, "cpu", answers, ids, model_factory=make_model)
        name = f'seed{summary["seed"]}_{trial.name}'
        full = pickle.load(open(REPLAY / a.corpus / name / "runs_L0.pkl", "rb"))["runs"]["test"]
        res[name] = {}
        for L in a.levels:
            lev = pickle.load(open(REPLAY / a.corpus / name / f"runs_L{L}.pkl", "rb"))["runs"]["test"]
            k = {v: min(8, len(lev[v]["eig"]), len(full[v]["asked"])) for v in test}
            ji_t = joint_info(model, store, test, am, chain, cfg["categories"],
                              {v: full[v]["asked"][:k[v]] for v in test}, a.samples, rng)
            ji_l = joint_info(model, store, test, am, chain, cfg["categories"],
                              {v: lev[v]["asked"][:k[v]] for v in test}, a.samples, rng)
            pos = [v for v in test if labels[v] == 1]
            d = np.array([ji_l[v] - ji_t[v] for v in test])
            dp = np.array([ji_l[v] - ji_t[v] for v in pos])
            r = {"tree_bits": float(np.mean([ji_t[v] for v in test])),
                 "level_bits": float(np.mean([ji_l[v] for v in test])),
                 "tree_bits_pos": float(np.mean([ji_t[v] for v in pos])),
                 "level_bits_pos": float(np.mean([ji_l[v] for v in pos])),
                 "share_level_more": float(np.mean(d > 0)), "share_level_more_pos": float(np.mean(dp > 0)),
                 "mean_k": float(np.mean(list(k.values())))}
            res[name][str(L)] = r
            print(a.corpus, name, L, json.dumps({kk: round(vv, 3) for kk, vv in r.items()}), flush=True)
    mean = {L: {kk: float(np.mean([res[t][L][kk] for t in res])) for kk in res[next(iter(res))][L]}
            for L in map(str, a.levels)}
    od = ROOT / a.out_root
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{a.corpus}.json").write_text(json.dumps({"trials": res, "mean": mean}, indent=1) + "\n")
    print("DONE", a.corpus, json.dumps({L: {kk: round(vv, 3) for kk, vv in mean[L].items()} for L in mean}), flush=True)


if __name__ == "__main__":
    main()
