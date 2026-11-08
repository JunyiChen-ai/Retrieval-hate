"""Stage A of the extent question (README section 8): revision-6 test-selected checkpoints (backbone, chain and node
potentials unchanged, trained with the yes / no answers), the No / Some / All answers (data/vlm_tree/<C>/ext_both_p3*),
extent inference and answer table (src/qtl/extent.py), test split.
Answer table: state none on every node of the negative training videos; some / all on the validation nodes by their
per-second GT; levels = training quantiles of presence 1 - P(No) and fullness P(All) / (P(Some) + P(All)).
Runs per trial: the tree (every queryable node, 8 calls) and the single levels L4 / L8 / L16 (qtree.level_nodes, EIG
order), both asking the extent question. E3: tree read after k_v = min(8, the level's askable nodes) calls against
the level after k_v calls; fixed 8: tree after 8 calls against revision 6 after 8 calls (replay runs_L0.pkl).
--answers sim: every answer of a video with GT (and of negative videos) replaced by a draw from the fitted table given
the node's true state (one draw per video and node, shared by the tree and the levels).
Also writes answer_check.json: per state (validation and test nodes) the mean presence / fullness and the AUC of
fullness for all vs some by node length.
Output runs/20261109_tree_within/stage_a_extent[_sim]/<corpus>/ (summary.json, per trial runs_L*.pkl and metrics).
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import socket
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import cpolicy, data as qdata, extent, qtree                   # noqa: E402
from qtl.backbones import make_model                                    # noqa: E402
from qtl.checkpoint import load_trial                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

METS = ("pooled_ap", "pooled_roc", "within_roc")
REPLAY = ROOT / "runs/20261108_astra_review/replay"
E3 = ROOT / "runs/20261108_astra_review/e3"
LEVELS = (4, 8, 16)


def score(corpus, od, label, sc):
    path = od / f"scores_test_{label}.jsonl"
    hc.write_scores(str(path), sc)
    m = hc.run_evaluator(corpus, "test", str(path), str(od / f"metrics_test_{label}.json"))["results"]["score_av"]
    return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"]}


def answer_check(raw, labels, ids, gt):
    out = {}
    for sp in ("val", "test"):
        rows = []
        for v in ids[sp]:
            if v not in raw or (labels[v] == 1 and v not in gt[sp]):
                continue
            y = np.asarray(gt[sp][v]).astype(int) if labels[v] == 1 else None
            for (a, b), o in raw[v].items():
                if o is None or b - a < qtree.F_FRAMES or (y is not None and b > len(y)):
                    continue
                s = 0 if y is None else 1 + extent.node_extent(y, a, b)
                rows.append((s, b - a, o[0], o[1]))
        R = np.array(rows)
        res = {"n": {str(s): int((R[:, 0] == s).sum()) for s in range(4)},
               "mean_presence": {str(s): float(R[R[:, 0] == s, 2].mean()) for s in range(4) if (R[:, 0] == s).any()},
               "mean_fullness": {str(s): float(R[R[:, 0] == s, 3].mean()) for s in range(4) if (R[:, 0] == s).any()},
               "auc_presence_hate_vs_none": float(roc_auc_score(R[:, 0] >= 2, R[:, 2])),
               "auc_fullness_all_vs_some": {}}
        for lo, hi in ((4, 8), (8, 16), (16, 32), (32, 64), (64, 10 ** 6)):
            k = (R[:, 0] >= 2) & (R[:, 1] >= lo) & (R[:, 1] < hi)
            if k.sum() > 10 and len(np.unique(R[k, 0])) == 2:
                res["auc_fullness_all_vs_some"][f"{lo}-{hi}"] = [float(roc_auc_score(R[k, 0] == 3, R[k, 3])),
                                                                 int((R[k, 0] == 3).sum()), int((R[k, 0] == 2).sum())]
        out[sp] = res
    return out


def simulate(test, store, labels, gt, eam, seed):
    sim = {}
    for i, v in enumerate(sorted(test)):
        if labels[v] == 1 and v not in gt:
            sim[v] = None
            continue
        rng = np.random.default_rng(seed * 100003 + i)
        vt = cpolicy._vt(store.T[v])
        asker = extent.ExtentAsker(vt, eam)
        y = np.asarray(gt[v]).astype(int) if labels[v] == 1 else np.zeros(store.T[v], dtype=int)
        d = {}
        for pos, n in enumerate(asker.q):
            s = 0 if labels[v] == 0 else 1 + extent.node_extent(y, int(vt.tr["a"][n]), int(vt.tr["b"][n]))
            pr = asker.po_s[pos, s]
            o = int(rng.choice(len(pr), p=pr / pr.sum()))
            d[int(n)] = asker.table[pos, :, o].copy()
        sim[v] = d
    return sim


def replay(model, store, vids, eam, chain, ext, cfg, level, sim):
    runs = {}
    vids = sorted(vids, key=lambda v: store.T[v])
    for i in range(0, len(vids), int(cfg["eval_chunk"])):
        chunk = vids[i:i + int(cfg["eval_chunk"])]
        answer_ll = None
        if sim is not None:
            asks = [extent.ExtentAsker(cpolicy._vt(store.T[v]), eam) for v in chunk]

            def answer_ll(b, node, chunk=chunk, asks=asks):
                v = chunk[b]
                if sim[v] is not None:
                    return sim[v][int(node)]
                tr = asks[b].vt.tr
                o = ext[v].get((int(tr["a"][node]), int(tr["b"][node])))
                return None if o is None else asks[b].loglik(node, o)
        with torch.no_grad():
            runs.update(extent.run_batch(model, store, chunk, eam, chain, ext, 8, "cpu", answer_ll=answer_ll,
                                         allowed=None if not level else
                                         [set(qtree.level_nodes(store.T[v], level).tolist()) for v in chunk]))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--answers", choices=("real", "sim"), default="real")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    out_root = a.out_root or ("runs/20261109_tree_within/stage_a_extent" + ("_sim" if a.answers == "sim" else ""))
    torch.set_num_threads(a.threads)
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))
    trials = [r["trial"] for r in summ["trials"].values()]
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    test = list(ids["test"])
    qdata.configure_source("soft_both", 8)
    answers, _ = qdata.load_answers(a.corpus, "soft_both")
    raw, T_ext, _split = extent.load_extent(a.corpus)
    missing = [v for sp in ("train", "val", "test") for v in ids[sp] if v not in raw]
    assert not missing, f"{len(missing)} videos without extent answers, e.g. {missing[:3]}"
    ext, edges = extent.bin_extent(raw, {v: sp for sp in ("train", "val", "test") for v in ids[sp]}, 8)
    store = qdata.Store(a.corpus, test, ["bert"])
    od_c = (ROOT / out_root / a.corpus).resolve()
    od_c.mkdir(parents=True, exist_ok=True)
    gt_eval = {sp: {v: np.asarray(y) for v, y in gt[sp].items()} for sp in ("val", "test")}
    check = answer_check(raw, labels, ids, gt_eval)
    (od_c / "answer_check.json").write_text(json.dumps(check, indent=1) + "\n")
    print("answer check", json.dumps(check), flush=True)
    e3_r6 = json.load(open(E3 / a.corpus / "summary.json"))["mean"]
    rows, eam_info = {}, None
    for td in trials:
        trial = Path(td).resolve()
        summary, cfg, model, am, chain = load_trial(trial, "cpu", answers, ids, model_factory=make_model)
        name = f'seed{summary["seed"]}_{trial.name}'
        od = od_c / name
        od.mkdir(parents=True, exist_ok=True)
        eam, counts = extent.fit_extent(ext, labels, ids["train"], ids["val"], gt_eval["val"], am.len_mu, am.len_sd)
        eam_info = {"counts": counts, "edges": [e.tolist() for e in edges], **eam.to_dict()}
        sim = simulate(test, store, labels, gt_eval["test"], eam, int(summary["seed"])) if a.answers == "sim" else None
        runs = {}
        for L in (0,) + LEVELS:
            runs[L] = replay(model, store, test, eam, chain, ext, cfg, L, sim)
            with (od / f"runs_L{L}.pkl").open("wb") as fh:
                pickle.dump({"runs": runs[L], "trial": str(trial), "host": socket.gethostname(),
                             "created": datetime.now().astimezone().isoformat()}, fh)
        full = runs[0]
        row = {}
        for L in LEVELS:
            k = {v: min(8, len(runs[L][v]["eig"])) for v in test}
            mf = score(a.corpus, od, f"full_matchL{L}",
                       {v: full[v]["scores"][min(k[v], len(full[v]["scores"]) - 1)] for v in test})
            ml = score(a.corpus, od, f"level{L}", {v: runs[L][v]["scores"][k[v]] for v in test})
            row[str(L)] = {"full": mf, "level": ml, "full_minus_level": {m: mf[m] - ml[m] for m in METS},
                           "mean_calls": float(np.mean(list(k.values())))}
        r6 = pickle.load(open(REPLAY / a.corpus / name / "runs_L0.pkl", "rb"))["runs"]["test"]
        row["fixed8"] = score(a.corpus, od, "full_fixed8", {v: full[v]["scores"][-1] for v in test})
        row["r6_fixed8"] = score(a.corpus, od, "r6_fixed8",
                                 {v: r6[v]["scores"][min(8, len(r6[v]["scores"]) - 1)] for v in test})
        row["fixed8_minus_r6"] = {m: row["fixed8"][m] - row["r6_fixed8"][m] for m in METS}
        nested = []
        for v in test:
            tr = qtree.tree(store.T[v])
            asked = full[v]["asked"]
            nested.append(sum(any((tr["a"][m] <= tr["a"][n] and tr["b"][n] <= tr["b"][m]) or
                                  (tr["a"][n] <= tr["a"][m] and tr["b"][m] <= tr["b"][n]) for m in asked[:i])
                              for i, n in enumerate(asked)) / max(len(asked), 1))
        row["nested_share"] = float(np.mean(nested))
        rows[name] = row
        print(a.corpus, name, json.dumps({L: [round(x, 4) for x in row[L]["full_minus_level"].values()]
                                          for L in map(str, LEVELS)}),
              "fixed8 - r6", [round(x, 4) for x in row["fixed8_minus_r6"].values()],
              "nested %.2f" % row["nested_share"], flush=True)
    mean = {L: {kk: {m: float(np.mean([rows[t][L][kk][m] for t in rows])) for m in METS}
                for kk in ("full", "level", "full_minus_level")} for L in map(str, LEVELS)}
    for kk in ("fixed8", "r6_fixed8", "fixed8_minus_r6"):
        mean[kk] = {m: float(np.mean([rows[t][kk][m] for t in rows])) for m in METS}
    mean["nested_share"] = float(np.mean([rows[t]["nested_share"] for t in rows]))
    mean["r6_e3_full_minus_level"] = {L: e3_r6[L]["diff"] for L in map(str, LEVELS)}
    (od_c / "summary.json").write_text(json.dumps(
        {"corpus": a.corpus, "answers": a.answers, "host": socket.gethostname(),
         "created": datetime.now().astimezone().isoformat(), "answer_model": eam_info, "trials": rows,
         "mean": mean}, indent=1) + "\n")
    print("DONE", a.corpus, json.dumps({L: mean[L]["full_minus_level"] for L in map(str, LEVELS)}),
          "fixed8 - r6", json.dumps(mean["fixed8_minus_r6"]), "nested %.2f" % mean["nested_share"], flush=True)


if __name__ == "__main__":
    main()
