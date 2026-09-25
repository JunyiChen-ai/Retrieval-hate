"""Offline check of the proposed fix for the loose backbone integration (README section 13; development evidence
on test, rule 10): can the answer about one short node be spread to the other seconds of the same video through
the backbone's features?

For every test video with both classes, pairs (P, N) of queryable nodes of length 4-15 s (fixed seed, at most
MAX_PAIRS per video):
    oracle  P contains harm, N contains none (the true spans; what a perfect VLM would answer)
    vlm     the cached VLM answered P with some category >= 2 and N with all five categories 0 (real answers)
The seconds outside P and N are ranked four ways, and within-video ROC is averaged over pairs, then videos:
    prior   s_t, the backbone's own per-second logit (five-crop mean), which ignores P and N
    time    distance in seconds to N minus distance to P (the propagation the tree and the chain can do)
    feat    cos(h_t, mean_P h) - cos(h_t, mean_N h), h_t = a_out_t + v_out_t, the backbone embedding the per-second
            head reads (s_t = w . h_t + const); feat_c the same with h centred on the video mean
    raw     the same on the backbone's inputs on seconds (I3D crop mean, VGGish, BERT row; z-scored per video)
feat > prior and feat > time on the oracle pairs is the precondition of the fix (a per-video correction of w).

    python experiments/20260925_query_paradigm/propagation_check.py --corpus hatemm \
        --runs runs/20260925_query_paradigm_r3/hatemm/seed234/trial3 ...
Writes <out>/<corpus>_propagation.json (default out: runs/20260925_query_paradigm/propagation_check).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import data as qdata          # noqa: E402  (sets the shared import paths)
import hier_evidence_common as hc  # noqa: E402
import qtree                  # noqa: E402
from model import PriorNet    # noqa: E402

MAX_PAIRS = 5
LEN_LO, LEN_HI = 4, 16
METHODS = ("prior", "time", "feat", "feat_c", "raw")


def embed(model, store, v, device):
    T = store.T[v]
    f_v = torch.from_numpy(np.stack([store.visual(v, c) for c in range(5)])).to(device)
    f_a = torch.from_numpy(np.repeat(store.at[v][None], 5, axis=0)).to(device)
    mask = torch.ones(5, T, dtype=torch.bool, device=device)
    with torch.no_grad():
        s, _g, _a, _vl, v_out, a_out = model(f_a, f_v, mask)
    h = (a_out + v_out).mean(0).cpu().numpy().astype(np.float64)
    raw = np.concatenate([np.mean([store.visual(v, c) for c in range(5)], axis=0), store.at[v]], axis=1)
    raw = (raw - raw.mean(0)) / (raw.std(0) + 1e-6)
    return s.mean(0).cpu().numpy().astype(np.float64), h, raw.astype(np.float64)


def cos_to(X, idx):
    c = X[idx].mean(0)
    return X @ c / (np.linalg.norm(X, axis=1) * np.linalg.norm(c) + 1e-12)


def rank_scores(s, h, raw, P, N, T):
    t = np.arange(T)
    dist = lambda a, b: np.where(t < a, a - t, np.where(t >= b, t - b + 1, 0))  # noqa: E731
    return {"prior": s,
            "time": dist(*N).astype(np.float64) - dist(*P),
            "feat": cos_to(h, np.arange(*P)) - cos_to(h, np.arange(*N)),
            "feat_c": cos_to(h - h.mean(0), np.arange(*P)) - cos_to(h - h.mean(0), np.arange(*N)),
            "raw": cos_to(raw, np.arange(*P)) - cos_to(raw, np.arange(*N))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="trained run directories (config.json + model.pth)")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "20260925_query_paradigm", "propagation_check"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    answers, _ = qdata.load_answers(a.corpus)
    vids = [v for v in ids["test"] if labels[v] == 1 and 0 < np.asarray(gt["test"][v]).mean() < 1]
    res = {"corpus": a.corpus, "n_videos_both_classes": len(vids), "runs": {}}
    stores = {}
    for run in a.runs:
        cfg = json.load(open(os.path.join(run, "config.json")))
        key = tuple(cfg.get("text_sources", ["bert"]))
        if key not in stores:
            stores[key] = qdata.Store(a.corpus, vids, text_sources=key)
        store = stores[key]
        model = PriorNet(cfg).to(a.device)
        model.load_state_dict(torch.load(os.path.join(run, "model.pth"), map_location=a.device)["model"])
        model.eval()
        rng = np.random.RandomState(0)
        per = {m: {k: [] for k in METHODS} for m in ("oracle", "vlm")}
        n_pairs = {"oracle": 0, "vlm": 0}
        for v in vids:
            y = np.asarray(gt["test"][v])[:store.T[v]]
            T = len(y)
            s, h, raw = embed(model, store, v, a.device)
            tr = qtree.tree(store.T[v])
            nodes = [(int(x), int(z)) for x, z in zip(tr["a"], tr["b"]) if LEN_LO <= z - x < LEN_HI and z <= T]
            harm = {n: bool(y[n[0]:n[1]].any()) for n in nodes}
            ans = answers.get(v, {})
            groups = {"oracle": ([n for n in nodes if harm[n]], [n for n in nodes if not harm[n]]),
                      "vlm": ([n for n in nodes if ans.get(n) is not None and max(ans[n]) >= 2],
                              [n for n in nodes if ans.get(n) is not None and max(ans[n]) == 0])}
            for m, (Ps, Ns) in groups.items():
                if not Ps or not Ns:
                    continue
                pairs = [(Ps[i], Ns[j]) for i in range(len(Ps)) for j in range(len(Ns))]
                pick = rng.choice(len(pairs), size=min(MAX_PAIRS, len(pairs)), replace=False)
                vals = {k: [] for k in METHODS}
                for i in pick:
                    P, N = pairs[i]
                    rest = np.ones(T, dtype=bool)
                    rest[P[0]:P[1]] = False
                    rest[N[0]:N[1]] = False
                    if not rest.any() or y[rest].min() == y[rest].max():
                        continue
                    sc = rank_scores(s[:T], h[:T], raw[:T], P, N, T)
                    for k in METHODS:
                        vals[k].append(roc_auc_score(y[rest], sc[k][rest]))
                if vals["prior"]:
                    n_pairs[m] += len(vals["prior"])
                    for k in METHODS:
                        per[m][k].append(float(np.mean(vals[k])))
        out = {m: {"n_videos": len(per[m]["prior"]), "n_pairs": n_pairs[m],
                   **{k: float(np.mean(per[m][k])) if per[m][k] else None for k in METHODS}} for m in per}
        res["runs"][run] = out
        print(run, json.dumps(out), flush=True)
    res["mean"] = {m: {k: float(np.mean([r[m][k] for r in res["runs"].values() if r[m][k] is not None]))
                       for k in METHODS} for m in ("oracle", "vlm")}
    print("mean", json.dumps(res["mean"]))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "%s_propagation.json" % a.corpus), "w"), indent=1)


if __name__ == "__main__":
    main()
