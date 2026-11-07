"""Review control E3 (README, round 1): tree vs one tree level with the SAME checkpoint and the SAME calls per video.

GPT 6 Astra's round-1 point 3: the lvl4 / lvl8 / lvl16 ablation arms change both the training answers and the test
calls (lvl16 asks 5.0 / 4.0 calls on HateMM / DeHate against 7.6 / 7.3 for the method). Here nothing is retrained:
for each revision-6 trial (policy_replay.py runs, test split), per video v and level L the single-level policy (only
the nodes of one depth, windows of L to 2L seconds, EIG order) asks k_v = min(8, its askable nodes at that depth)
calls, and the method's policy (every node) is read after the same k_v calls. Both use the checkpoint's own answer
table and the cached answers. Reported per trial and as 3-seed means: AP / ROC / within (unified evaluator) of the
two, their paired difference (method - single level), and the mean k_v.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import hier_evidence_common as hc                                       # noqa: E402

METS = ("pooled_ap", "pooled_roc", "within_roc")


def score(corpus, od, label, sc):
    path = od / f"scores_test_{label}.jsonl"
    hc.write_scores(str(path), sc)
    m = hc.run_evaluator(corpus, "test", str(path), str(od / f"metrics_test_{label}.json"))["results"]["score_av"]
    return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--replay-dirs", nargs="+", required=True)
    ap.add_argument("--levels", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--out-root", default="runs/20261108_astra_review/e3")
    a = ap.parse_args()
    _, ids, _, _ = hc.load_fixed_cohort(a.corpus)
    test = list(ids["test"])
    rows = {}
    for rd in a.replay_dirs:
        rd = Path(rd)
        full = pickle.load(open(rd / "runs_L0.pkl", "rb"))["runs"]["test"]
        od = (ROOT / a.out_root / a.corpus / rd.name).resolve()
        od.mkdir(parents=True, exist_ok=True)
        row = {}
        for L in a.levels:
            lev = pickle.load(open(rd / f"runs_L{L}.pkl", "rb"))["runs"]["test"]
            k = {v: min(a.cap, len(lev[v]["eig"])) for v in test}
            m_full = score(a.corpus, od, f"full_matchL{L}", {v: full[v]["scores"][min(k[v], len(full[v]["eig"]))]
                                                            for v in test})
            m_lev = score(a.corpus, od, f"level{L}", {v: lev[v]["scores"][k[v]] for v in test})
            row[str(L)] = {"method": m_full, "single_level": m_lev,
                           "diff": {m: m_full[m] - m_lev[m] for m in METS},
                           "mean_calls": float(np.mean(list(k.values()))),
                           "videos_below_cap": int(sum(x < a.cap for x in k.values()))}
            print(a.corpus, rd.name, "L%d" % L, "calls %.2f" % row[str(L)]["mean_calls"],
                  "method", [round(m_full[m], 4) for m in METS], "level", [round(m_lev[m], 4) for m in METS],
                  flush=True)
        rows[rd.name] = row
    mean = {L: {part: {m: float(np.mean([r[L][part][m] for r in rows.values()])) for m in METS}
                for part in ("method", "single_level", "diff")} for L in map(str, a.levels)}
    for L in mean:
        mean[L]["mean_calls"] = float(np.mean([r[L]["mean_calls"] for r in rows.values()]))
        mean[L]["diff_sd"] = {m: float(np.std([r[L]["diff"][m] for r in rows.values()])) for m in METS}
    out = {"corpus": a.corpus, "created": datetime.now().astimezone().isoformat(), "cap": a.cap,
           "trials": rows, "mean": mean}
    (ROOT / a.out_root / a.corpus / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print("DONE", a.corpus, json.dumps({L: mean[L]["diff"] for L in mean}), flush=True)


if __name__ == "__main__":
    main()
