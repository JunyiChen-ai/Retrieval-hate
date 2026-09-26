"""Diagnostics for the reviewer concerns of 2026-09-27 (README section 14; development evidence on test, rule 10;
nothing here trains or selects).

Per trained trial (config.json, model.pth, summary.json of a revision-3 search winner):
  asked    where the questions go (summary.json test_runs, first 8 calls): node length, share of the video, depth;
           for positive videos whether the node contains GT harm and whether it is mixed (harm and no harm).
  within   per positive test video with both classes, within-video ROC after 0 and after 8 calls with the real
           answers (the trial's own score files), and whether any of its 8 asked nodes was mixed.
  oracle   the policy re-run with a perfect answerer: an asked node is revealed as harmful (state 2 only) or as
           containing no harm (states 0 / 1) from the GT; the questions are still chosen by EIG under the trial's
           answer model. Pooled AP / ROC / within at fixed budgets through the shared evaluator. This is the ceiling
           of the query framework when the answers are right, against which the real answers are compared.
Per corpus (no trained model; one backbone for the H features):
  reliab   can a node's content predict whether the VLM answer is right, beyond the node's length? Test nodes of
           4-64 s with a parsed answer. Task "miss": GT-harmful nodes, target = no category >= 2. Task "fa": nodes
           without GT harm, target = some category >= 2. Logistic regression, 5-fold cross-validation grouped by
           video, out-of-fold AUC for: L (log length, share of the video, depth); L + C (transcript words, words per
           second, and per-video z-scored node means of VGGish, BERT rows and five-crop-mean I3D, 8 PCA
           components each); L + C + H (the same for the backbone embedding h_t = a_out + v_out of the first trial).

    python experiments/20260925_query_paradigm/concern_diagnostics.py --corpus hatemm \
        --trials runs/20260925_query_paradigm_r3/hatemm/seed234/trial3 ... --out runs/20260927_query_paradigm_r4/diagnostics
Writes <out>/<corpus>.json (and the oracle score / metric files under <out>/<corpus>_oracle/<trial tag>/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import train as TR                              # noqa: E402  (sets up the import paths)
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402
import qtree                                    # noqa: E402
import ctree                                    # noqa: E402
import cpolicy                                  # noqa: E402
from model import PriorNet                      # noqa: E402
from macilsd import align                       # noqa: E402

BUDGETS = (0, 1, 2, 4, 8, 16, 32)
LEN_LO, LEN_HI = 4, 64


def load_trial(trial, device, answers, ids):
    summ = json.load(open(os.path.join(trial, "summary.json")))
    cfg = dict(TR.DEFAULTS)
    cfg.update(summ["cfg"])
    assert cfg["prior"] == "chain" and cfg["chain"] == "learned", "revision-3 trials only"
    ck = torch.load(os.path.join(trial, "model.pth"), map_location=device)
    model = PriorNet(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    loglen = np.log([b - a for v in ids["train"] for (a, b) in answers[v]])
    am = qtree.AnswerModel(ck["am"]["theta"].cpu().numpy(), loglen.mean(), loglen.std(), bool(cfg["length_term"]),
                           int(cfg["n_state"]), cfg["categories"]).to(device)
    am.load_state_dict(ck["am"], strict=False)
    am.len_mu, am.len_sd = float(loglen.mean()), float(loglen.std())
    chain = ctree.Chain(cfg["boundary"] == "closed", cfg["chain_form"] == "zero_inflated",
                        cfg["chain_form"] == "normalized").to(device)
    chain.load_state_dict(ck["chain"])
    return summ, cfg, model, am, chain


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else None


def asked_stats(summ, Tv, gt, labels, B=8):
    rows = []
    for v, r in summ["test_runs"].items():
        T = Tv[v]
        tr = qtree.tree(T)
        y = np.asarray(gt[v])[:T]
        for k, n in enumerate(r["asked"][:B]):
            a, b = int(tr["a"][n]), int(tr["b"][n])
            seg = y[a:b]
            rows.append({"k": k, "len": b - a, "rel": (b - a) / T, "depth": int(tr["depth"][n]), "pos": labels[v],
                         "harm": bool(seg.any()), "mixed": bool(seg.any() and not seg.all())})
    per_k = []
    for k in range(B):
        rk = [r for r in rows if r["k"] == k]
        if rk:
            per_k.append({"k": k, "n": len(rk), "len": float(np.mean([r["len"] for r in rk])),
                          "rel": float(np.mean([r["rel"] for r in rk])),
                          "depth": float(np.mean([r["depth"] for r in rk]))})
    pos = [r for r in rows if r["pos"] == 1]
    return {"per_call": per_k, "n_calls": len(rows),
            "share_rel_ge_half": float(np.mean([r["rel"] >= 0.5 for r in rows])),
            "share_rel_ge_quarter": float(np.mean([r["rel"] >= 0.25 for r in rows])),
            "share_root": float(np.mean([r["depth"] == 0 for r in rows])),
            "median_len": float(np.median([r["len"] for r in rows])),
            "pos_calls": len(pos),
            "pos_share_harm": float(np.mean([r["harm"] for r in pos])) if pos else None,
            "pos_share_mixed": float(np.mean([r["mixed"] for r in pos])) if pos else None,
            "pos_share_pure_harm": float(np.mean([r["harm"] and not r["mixed"] for r in pos])) if pos else None}


def read_scores(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["video_id"]] = np.asarray(r["score_av"], dtype=np.float64)
    return out


def within_change(trial, summ, Tv, gt, labels, B=8):
    s0 = read_scores(os.path.join(trial, "scores_test_fixed0.jsonl"))
    s8 = read_scores(os.path.join(trial, "scores_test_fixed%d.jsonl" % B))
    rows = []
    for v in s0:
        if labels[v] != 1:
            continue
        y = np.asarray(gt[v])[:Tv[v]]
        if not 0 < y.mean() < 1:
            continue
        tr = qtree.tree(Tv[v])
        asked = summ["test_runs"][v]["asked"][:B]
        mixed = any(y[int(tr["a"][n]):int(tr["b"][n])].any() and not y[int(tr["a"][n]):int(tr["b"][n])].all()
                    for n in asked)
        rows.append((auc(y, s0[v]), auc(y, s8[v]), mixed))
    d = np.array([b - a for a, b, _ in rows])
    mx = np.array([m for _, _, m in rows])
    return {"n_videos": len(rows), "within0": float(np.mean([a for a, _, _ in rows])),
            "within8": float(np.mean([b for _, b, _ in rows])), "mean_change": float(d.mean()),
            "share_up": float(np.mean(d > 0.01)), "share_down": float(np.mean(d < -0.01)),
            "share_any_mixed_asked": float(mx.mean()),
            "mean_change_with_mixed": float(d[mx].mean()) if mx.any() else None,
            "mean_change_without_mixed": float(d[~mx].mean()) if (~mx).any() else None}


def oracle(model, store, vids, am, chain, cfg, gt, device, out_dir, corpus):
    runs = {}
    vids = sorted(vids, key=lambda v: store.T[v])
    k = int(cfg["eval_chunk"])
    big = ctree.BIG
    for i in range(0, len(vids), k):
        chunk = vids[i:i + k]
        trs = [qtree.tree(store.T[v]) for v in chunk]
        ys = [np.asarray(gt[v])[:store.T[v]] for v in chunk]

        def ll(b, node, trs=trs, ys=ys):
            a, e = int(trs[b]["a"][node]), int(trs[b]["b"][node])
            return np.array([-big, -big, 0.0]) if ys[b][a:e].any() else np.array([0.0, 0.0, -big])

        runs.update(cpolicy.run_batch(model, store, chunk, am, chain, None, cfg["categories"], max(BUDGETS), device,
                                      "eig", "tree", False, None, ll))
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for B in BUDGETS:
        sp = os.path.join(out_dir, "scores_test_oracle%d.jsonl" % B)
        hc.write_scores(sp, TR.at_budget(runs, B))
        r = hc.run_evaluator(corpus, "test", sp, os.path.join(out_dir, "metrics_test_oracle%d.json" % B))
        r = r["results"]["score_av"]
        res[str(B)] = {"pooled_ap": r["pr_auc"], "pooled_roc": r["roc_auc"], "within_roc": r["per_video"]["macro_auc"]}
    return res


@torch.no_grad()
def embed_h(model, store, v, device):
    T = store.T[v]
    f_v = torch.from_numpy(np.stack([store.visual(v, c) for c in range(5)])).to(device)
    f_a = torch.from_numpy(np.repeat(store.at[v][None], 5, axis=0)).to(device)
    *_, v_out, a_out = model(f_a, f_v, torch.ones(5, T, dtype=torch.bool, device=device))
    return (a_out + v_out).mean(0).cpu().numpy()


def zs(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def reliability(corpus, store, vids, answers, gt, model, device):
    man = {}
    for line in open(os.path.join(ROOT, "data", "vlm_tree", qdata.CORPUS_DIR[corpus], "manifest.jsonl")):
        r = json.loads(line)
        if r["id"] in vids:
            man[r["id"]] = {(int(a), int(b)): len(text.split()) for a, b, _idx, text in r["nodes"]}
    A = align.A_DIM
    rows, blocks, groups = [], {"vgg": [], "bert": [], "i3d": [], "h": []}, []
    for gi, v in enumerate(vids):
        T = store.T[v]
        y = np.asarray(gt[v])[:T]
        at = store.at[v]
        vis = np.mean([store.visual(v, c) for c in range(5)], axis=0)
        feats = {"vgg": zs(at[:, :A]), "bert": zs(at[:, A:A + hc.TEXT_DIM]), "i3d": zs(vis)}
        if model is not None:
            feats["h"] = zs(embed_h(model, store, v, device))
        tr = qtree.tree(T)
        for n in np.where(tr["queryable"])[0]:
            a, b = int(tr["a"][n]), int(tr["b"][n])
            if not LEN_LO <= b - a <= LEN_HI:
                continue
            o = answers[v].get((a, b))
            if o is None:
                continue
            words = man.get(v, {}).get((a, b), 0)
            rows.append({"harm": bool(y[a:b].any()), "yes": bool(np.max(o) >= 2), "len": b - a, "rel": (b - a) / T,
                         "depth": int(tr["depth"][n]), "words": words, "wps": words / (b - a)})
            for k in blocks:
                if k in feats:
                    blocks[k].append(feats[k][a:b].mean(0))
            groups.append(gi)
    groups = np.asarray(groups)
    L = np.array([[np.log(r["len"]), r["rel"], r["depth"]] for r in rows])
    Cw = np.array([[np.log1p(r["words"]), r["wps"]] for r in rows])
    pcs = {k: PCA(8, random_state=0).fit_transform(np.asarray(b)) for k, b in blocks.items() if len(b)}
    C = np.concatenate([Cw, pcs["vgg"], pcs["bert"], pcs["i3d"]], axis=1)
    H = pcs.get("h")
    harm = np.array([r["harm"] for r in rows])
    yes = np.array([r["yes"] for r in rows])
    out = {"n_nodes": len(rows)}
    for task, sel, target in (("miss", harm, ~yes), ("fa", ~harm, yes)):
        res = {"n": int(sel.sum()), "rate": float(target[sel].mean())}
        sets = {"L": L, "L+C": np.concatenate([L, C], 1)}
        if H is not None:
            sets["L+C+H"] = np.concatenate([L, C, H], 1)
        for name, X in sets.items():
            Xs, ys_, gs = X[sel], target[sel].astype(int), groups[sel]
            pred = np.zeros(len(ys_))
            for trn, tst in GroupKFold(5).split(Xs, ys_, gs):
                sc = StandardScaler().fit(Xs[trn])
                clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xs[trn]), ys_[trn])
                pred[tst] = clf.predict_proba(sc.transform(Xs[tst]))[:, 1]
            res["auc_" + name] = auc(ys_, pred)
        out[task] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--trials", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "20260927_query_paradigm_r4", "diagnostics"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    answers, _ = qdata.load_answers(a.corpus)
    store = None
    res = {"corpus": a.corpus, "trials": {}}
    first_model = None
    for trial in a.trials:
        summ, cfg, model, am, chain = load_trial(trial, a.device, answers, ids)
        if store is None:
            store = qdata.Store(a.corpus, ids["test"], cfg.get("text_sources", ["bert"]))
        if first_model is None:
            first_model = model
        tag = "_".join(trial.rstrip("/").split("/")[-2:])
        r = {"asked": asked_stats(summ, store.T, gt["test"], labels),
             "within_change": within_change(trial, summ, store.T, gt["test"], labels),
             "real": {B: {k: summ["fixed"][B]["test"][k] for k in ("pooled_ap", "pooled_roc", "within_roc")}
                      for B in map(str, BUDGETS)},
             "oracle": oracle(model, store, ids["test"], am, chain, cfg, gt["test"], a.device,
                              os.path.join(a.out, "%s_oracle" % a.corpus, tag), a.corpus)}
        res["trials"][trial] = r
        print("== %s" % trial)
        print("  asked: median len %.1f, share >= half video %.2f, root %.2f; pos calls with harm %.2f, mixed %.2f"
              % (r["asked"]["median_len"], r["asked"]["share_rel_ge_half"], r["asked"]["share_root"],
                 r["asked"]["pos_share_harm"], r["asked"]["pos_share_mixed"]))
        wc = r["within_change"]
        print("  within 0 -> 8 calls (real answers): %.3f -> %.3f, up %.2f down %.2f; any mixed asked %.2f"
              % (wc["within0"], wc["within8"], wc["share_up"], wc["share_down"], wc["share_any_mixed_asked"]))
        for B in map(str, BUDGETS):
            re_, orc = r["real"][B], r["oracle"][B]
            print("  B=%-2s real %.4f/%.4f/%.4f | oracle %.4f/%.4f/%.4f" % (
                B, re_["pooled_ap"], re_["pooled_roc"], re_["within_roc"], orc["pooled_ap"], orc["pooled_roc"],
                orc["within_roc"]))
        json.dump(res, open(os.path.join(a.out, "%s.json" % a.corpus), "w"), indent=1, default=float)
    res["reliability"] = reliability(a.corpus, store, ids["test"], answers, gt["test"], first_model, a.device)
    print("  reliability:", json.dumps(res["reliability"]))
    json.dump(res, open(os.path.join(a.out, "%s.json" % a.corpus), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
