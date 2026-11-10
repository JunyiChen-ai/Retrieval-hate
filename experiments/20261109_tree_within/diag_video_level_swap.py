"""Diagnostic (README section 12): is the pooled-AP tie between revision 6 and the naive-question control a matter of
the ordering of whole videos or of the seconds inside a video? Revision-6 test-selected trials at fixed 8 calls
(replay runs_L0.pkl) against the section-7 controls on the 8 equal windows: <baseline> + painted answers at alpha .5
and the painted answers alone (vlm_only). Per control and seed, the per-second scores are recombined:
  ours_video_x_control_within   our video mean, the control's shape inside the video
                                (control_t x mean(ours) / mean(control));
  control_video_x_ours_within   the control's video mean, our shape inside the video
                                (ours_t x mean(control) / mean(ours)).
Also the scores before any call of the two no-backbone runs of diag_uniform8_decompose.py: chain_only (s_t = phi_n =
0, g = 0: the learned chain's position prior alone) and g_only (s_t = phi_n = 0, the backbone's video logit).
AP / ROC / within and the pooled AP / ROC of the video-mean score, unified core evaluator (hc.frame_metrics).
Output runs/20261109_tree_within/video_level_swap/<corpus>.json.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import socket
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata, qtree                                    # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402
from diag_uniform8 import depth3                                        # noqa: E402
from naive_questions import evidence                                    # noqa: E402

REPLAY = ROOT / "runs/20261108_astra_review/replay"
DECOMP = ROOT / "runs/20261109_tree_within/uniform8_decompose"
KEYS = ("pooled_ap", "pooled_roc", "within_roc", "video_mean_ap", "video_mean_roc")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--out-root", default="runs/20261109_tree_within/video_level_swap")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    test = list(ids["test"])
    hate = {v for v in test if labels[v] == 1}
    P, _, _ = qdata.load_soft_p(a.corpus, "soft_both")
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))

    def mets(sc):
        # 6 decimals, as the evaluator reads the scores from the score files (hc.write_scores): saturated posteriors
        # tie there, which moves within by up to .01 in single trials
        sc = {v: np.round(np.asarray(s, dtype=np.float64), 6) for v, s in sc.items()}
        m = hc.frame_metrics(sc, gt["test"], hate)
        vm = hc.frame_metrics({v: np.full(len(s), round(float(np.mean(s)), 6)) for v, s in sc.items()}, gt["test"],
                              hate)
        return {"pooled_ap": float(m["pooled_ap"]), "pooled_roc": float(m["pooled_roc"]),
                "within_roc": float(m["within_roc"]), "video_mean_ap": float(vm["pooled_ap"]),
                "video_mean_roc": float(vm["pooled_roc"])}

    rows = {}
    for r in summ["trials"].values():
        seed, name = int(r["seed"]), f'seed{r["seed"]}_{Path(r["trial"]).name}'
        full = pickle.load(open(REPLAY / a.corpus / name / "runs_L0.pkl", "rb"))["runs"]["test"]
        ours = {v: np.asarray(full[v]["scores"][min(8, len(full[v]["scores"]) - 1)], dtype=np.float64) for v in test}
        ev = {}
        for v in test:
            T = len(ours[v])
            ev[v], _ = evidence(qtree.tree(T), T, sorted(depth3(T)), P[v])
        controls = {"vlm_only": ev}
        for m in hc.STRONGEST_BASELINE[a.corpus]:
            base = hc.baseline_scores(a.corpus, m, seed)
            c = {}
            for v in test:
                T = len(ours[v])
                b = base.get(v, np.full(T, 0.5))
                n = min(len(b), T)
                b = np.concatenate([b[:n], np.full(T - n, b[n - 1] if n else 0.5)])
                c[v] = 1.0 / (1.0 + np.exp(-(0.5 * hc.logit(b) + 0.5 * hc.logit(ev[v]))))
            controls[f"{m}_paint_a0.50"] = c
        row = {"ours_fixed8": mets(ours)}
        for cn, c in controls.items():
            mo = {v: max(float(np.mean(ours[v])), 1e-12) for v in test}
            mc = {v: max(float(np.mean(c[v])), 1e-12) for v in test}
            row[cn] = mets(c)
            row[f"{cn}|ours_video_x_control_within"] = mets({v: c[v] * mo[v] / mc[v] for v in test})
            row[f"{cn}|control_video_x_ours_within"] = mets({v: ours[v] * mc[v] / mo[v] for v in test})
        for tag, label in (("noprior", "chain_only_0calls"), ("flatloc", "g_only_0calls")):
            path = DECOMP / a.corpus / name / f"runs_{tag}.pkl"
            if path.exists():
                runs = pickle.load(path.open("rb"))["runs"]
                row[label] = mets({v: np.asarray(runs[v]["scores"][0], dtype=np.float64) for v in test})
        rows[name] = row
        print(a.corpus, name, json.dumps({k: [round(x, 4) for x in m.values()] for k, m in row.items()}), flush=True)
    names = list(next(iter(rows.values())))
    mean = {k: {m: float(np.mean([rows[t][k][m] for t in rows])) for m in KEYS} for k in names}
    out = (ROOT / a.out_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.corpus}.json").write_text(json.dumps(
        {"corpus": a.corpus, "host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
         "trials": rows, "mean": mean}, indent=1) + "\n")
    for k in names:
        print("MEAN", a.corpus, "%-52s" % k, " ".join("%.4f" % mean[k][m] for m in KEYS), flush=True)
    print("DONE", a.corpus, flush=True)


if __name__ == "__main__":
    main()
