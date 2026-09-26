"""Revision-4 check (README section 14; development evidence on test, rule 10; nothing here trains or selects): do
word-timestamp transcripts make the VLM's node answers more informative inside positive videos?

The revision-3 answers (source k30) were asked with transcripts cut from Whisper chunks by the chunk midpoint; long
or broken chunks left 57-71% of the short test nodes of positive videos without any transcript. Source words asks
the same questions (same frames, prompt, model) with word-timestamp transcripts (scripts/asr_words.py). On test:
  nodes   positive test videos with GT spans, nodes of 4-16 s: P(any category >= 2 | node has GT harm / no harm) and
          the node ROC of the maximum level, per source; the same by transcript state of the k30 node
  pairs   internal nodes <= 64 s whose two halves are queryable and exactly one half has GT harm: the harmful half
          has the higher maximum level (ties count half), per source
  swap    each revision-3 search winner re-run on test with the words answers (network, answer model and chain
          unchanged; EIG questions); pooled AP / ROC / within at B = 0..32 through the shared evaluator, next to the
          trial's own k30 numbers

    python experiments/20260925_query_paradigm/pilot_words.py --corpus hatemm --trials <trial dirs>
Writes runs/20260927_query_paradigm_r4/pilot_words/<corpus>.json (score / metric files under <corpus>/<trial tag>/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import concern_diagnostics as cd                # noqa: E402  (sets up the import paths)
import train as TR                              # noqa: E402
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402
import qtree                                    # noqa: E402
import cpolicy                                  # noqa: E402

BUDGETS = (0, 1, 2, 4, 8, 16, 32)
OUT = os.path.join(ROOT, "runs", "20260927_query_paradigm_r4", "pilot_words")


def node_stats(corpus, vids, labels, gt, src):
    k30_words = {}
    for line in open(os.path.join(ROOT, "data", "vlm_tree", qdata.CORPUS_DIR[corpus], "manifest.jsonl")):
        r = json.loads(line)
        if r["id"] in vids:
            k30_words[r["id"]] = {(int(a), int(b)): len(t.split()) for a, b, _i, t in r["nodes"]}
    res = {}
    for name, ans in src.items():
        rows = []
        for v in vids:
            y = np.asarray(gt[v])
            if labels[v] == 0 or not y.any():
                continue
            for (a, b), o in ans[v].items():
                if o is None or b - a > 16 or (a, b) not in src["k30"][v] or src["k30"][v][(a, b)] is None:
                    continue
                rows.append((int(y[a:b].any()), int(max(o)), k30_words[v].get((a, b), 0) == 0))
        r = np.array(rows)
        out = {}
        for part, m in (("all", np.ones(len(r), bool)), ("k30_empty", r[:, 2] == 1), ("k30_text", r[:, 2] == 0)):
            y, s = r[m, 0], r[m, 1]
            out[part] = {"n": int(m.sum()), "yes_harm": float(np.mean(s[y == 1] >= 2)),
                         "yes_noharm": float(np.mean(s[y == 0] >= 2)), "roc": cd.auc(y, s)}
        res[name] = out
    return res


def pair_stats(vids, labels, gt, src, T):
    res = {}
    for name, ans in src.items():
        acc = []
        for v in vids:
            if labels[v] == 0:
                continue
            y = np.asarray(gt[v])
            tr = qtree.tree(T[v])
            for n in range(len(tr["a"])):
                L, R = int(tr["left"][n]), int(tr["right"][n])
                if L < 0 or tr["b"][n] - tr["a"][n] > 64:
                    continue
                kl, kr = (int(tr["a"][L]), int(tr["b"][L])), (int(tr["a"][R]), int(tr["b"][R]))
                ol, orr = ans[v].get(kl), ans[v].get(kr)
                # the same pairs for both sources: both halves answered under k30 and under words
                if any(src[s][v].get(k) is None for s in src for k in (kl, kr)):
                    continue
                hl, hr = bool(y[kl[0]:kl[1]].any()), bool(y[kr[0]:kr[1]].any())
                if hl == hr:
                    continue
                mh, mc = (max(ol), max(orr)) if hl else (max(orr), max(ol))
                acc.append(1.0 if mh > mc else 0.5 if mh == mc else 0.0)
        res[name] = {"n": len(acc), "pick_accuracy_ties_half": float(np.mean(acc))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--trials", nargs="*", default=[])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    k30, _ = qdata.load_answers(a.corpus, "k30")
    words, Tw = qdata.load_answers(a.corpus, "words")
    vids = [v for v in ids["test"] if v in words]
    assert len(vids) == len(ids["test"]), "words answers missing for %d test videos" % (len(ids["test"]) - len(vids))
    src = {"k30": k30, "words": words}
    res = {"corpus": a.corpus, "n_test": len(vids), "nodes": node_stats(a.corpus, vids, labels, gt["test"], src),
           "pairs": pair_stats(vids, labels, gt["test"], src, Tw), "trials": {}}
    print(json.dumps({k: res[k] for k in ("nodes", "pairs")}, indent=1), flush=True)
    os.makedirs(OUT, exist_ok=True)
    store = None
    for trial in a.trials:
        summ, cfg, model, am, chain = cd.load_trial(trial, a.device, k30, ids)
        if store is None:
            store = qdata.Store(a.corpus, vids, cfg.get("text_sources", ["bert"]))
        runs = {}
        order = sorted(vids, key=lambda v: store.T[v])
        k = int(cfg["eval_chunk"])
        for i in range(0, len(order), k):
            runs.update(cpolicy.run_batch(model, store, order[i:i + k], am, chain, words, cfg["categories"],
                                          max(BUDGETS), a.device, "eig", "tree"))
        tag = "_".join(trial.rstrip("/").split("/")[-2:])
        od = os.path.join(OUT, a.corpus, tag)
        os.makedirs(od, exist_ok=True)
        r = {"k30": {str(B): {m: summ["fixed"][str(B)]["test"][m] for m in ("pooled_ap", "pooled_roc", "within_roc")}
                     for B in BUDGETS}, "words": {}}
        for B in BUDGETS:
            sp = os.path.join(od, "scores_test_fixed%d.jsonl" % B)
            hc.write_scores(sp, TR.at_budget(runs, B))
            m = hc.run_evaluator(a.corpus, "test", sp, os.path.join(od, "metrics_test_fixed%d.json" % B))
            m = m["results"]["score_av"]
            r["words"][str(B)] = {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"],
                                  "within_roc": m["per_video"]["macro_auc"]}
        res["trials"][trial] = r
        print("== %s" % trial, flush=True)
        for B in map(str, BUDGETS):
            print("  B=%-2s " % B + " | ".join("%s %.4f/%.4f/%.4f" % (n, r[n][B]["pooled_ap"], r[n][B]["pooled_roc"],
                                                                     r[n][B]["within_roc"]) for n in ("k30", "words")),
                  flush=True)
        json.dump(res, open(os.path.join(OUT, "%s.json" % a.corpus), "w"), indent=1, default=float)
    json.dump(res, open(os.path.join(OUT, "%s.json" % a.corpus), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
