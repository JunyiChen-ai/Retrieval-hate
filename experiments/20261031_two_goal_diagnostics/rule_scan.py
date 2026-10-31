"""Goal 1 diagnostic: every existing stopping rule, plus rules that add the order's own per-second selection value as a
component, on the stored policy_lab runs (runs.pkl, val and test; no new VLM inference or tree passes).

For each corpus, tag and order: per rule and target mean calls, the threshold is calibrated on validation mean calls
(no labels, policy.calibrate), applied to test, and scored with the shared core evaluator. Rows are 3-seed means and
differences to the anchored EIG fixed-8 result (tag r5, order eig) and to the order's own fixed 8.
New components (constant-free, same quantile machinery as qmixSG):
  selT   the order's own value of the next question (EIG bits, or the EER value) divided by the video length T
  sel    the same, not divided
Rule-10 developmental evidence (test labels read; the scan itself selects nothing for the method).
"""
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import policy, stop_rules
import hier_evidence_common as hc

NEW = {"qmixSGe": ("dlsum", "stab1", "hG", "selT"), "qmixSe": ("dlsum", "selT"), "qmixGe": ("hG", "selT"),
       "qmixSGE": ("dlsum", "stab1", "hG", "sel"), "qmixGE": ("hG", "sel"), "qmixTe": ("stab1", "selT"),
       "qavgSGe": ("dlsum", "stab1", "hG", "selT"), "qavgGe": ("hG", "selT"), "qmixS1": ("dlsum", "sel")}


def calibrate(runs, budget):
    """Same result as policy.calibrate (smallest threshold whose mean calls <= budget), vectorised: a video's calls at
    threshold c = number of its running-minimum values >= c, so the mean is a count over all running minima."""
    cands = np.unique(np.concatenate([np.asarray(r, dtype=np.float64) for r in runs if len(r)] + [np.zeros(1)]))
    pm = np.sort(np.concatenate([np.minimum.accumulate(np.asarray(r, dtype=np.float64)) for r in runs if len(r)]))
    mean = (len(pm) - np.searchsorted(pm, cands, side="left")) / len(runs)
    ok = np.where(mean <= budget + 1e-9)[0]
    if len(ok) == 0:
        c = float(cands[-1]) + 1e-9
        return c, float(np.mean([policy.stop_calls(r, c) for r in runs]))
    return float(cands[ok[0]]), float(mean[ok[0]])


def truncate(run, cap):
    """The run as if the policy had been stopped after at most `cap` calls (per-call lists cut; scores and p_G keep
    the prior entry, so they hold one more element)."""
    n = min(cap, len(run["eig"]))
    return {k: (list(v[:n + 1]) if k in ("scores", "p_G") else list(v[:n])) for k, v in run.items()}


def metrics(scores, gt, hate):
    m = hc.frame_metrics(scores, gt, hate)
    return [m["pooled_ap"], m["pooled_roc"], m["within_roc"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--tags", nargs="+", default=["r5"])
    ap.add_argument("--root", default="runs/20261031_two_goal_diagnostics/policy_lab")
    ap.add_argument("--targets", nargs="+", type=float, default=[4, 5])
    ap.add_argument("--rules", nargs="+", default=None, help="restrict to these rules (default: all)")
    ap.add_argument("--out-name", default="rule_scan.json")
    ap.add_argument("--orders", nargs="+", default=["eig", "eer"])
    ap.add_argument("--cap", type=int, default=None,
                    help="truncate the stored runs to this many calls before computing rule values (cap check)")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    hate = {v for v in ids["test"] if labels[v] == 1}
    stop_rules.QMIX.update(NEW)
    rules = [r for r in stop_rules.RULES if not r.startswith(("eig_f", "voi_f", "stab1_f"))] + list(NEW)
    rules += ["%s_rt%02d" % (b, d) for b in NEW for d in (5, 10)]
    if a.rules:
        rules = [r for r in rules if r in a.rules]
    report = {}
    for tag in a.tags:
        for order in a.orders:
            per_trial = []
            for d in sorted((ROOT / a.root / a.corpus / tag).glob(f"*_{order}")):
                with (d / "runs.pkl").open("rb") as fh:
                    blob = pickle.load(fh)
                runs, T = blob["runs"], blob["T"]
                if a.cap is not None:
                    runs = {s: {v: truncate(r, a.cap) for v, r in runs[s].items()} for s in runs}
                vals = {}
                for s in ("val", "test"):
                    vals[s] = {}
                    for v, r in runs[s].items():
                        x = stop_rules.rule_values(r, len(r["scores"][0]))
                        n = len(r["eig"])
                        x["sel"] = [float(e) for e in r["eig"][:n]]
                        x["selT"] = [float(e) / T[v] for e in r["eig"][:n]]
                        vals[s][v] = x
                stop_rules.add_quantile_rules(vals)
                for base in NEW:
                    for delta in (0.05, 0.10):
                        name = "%s_rt%02d" % (base, int(round(delta * 100)))
                        for s in vals:
                            for v in vals[s]:
                                vals[s][v][name] = [x * (1.0 - delta) ** k if np.isfinite(x) else x
                                                    for k, x in enumerate(vals[s][v][base])]
                stop_rules.add_relaxed_rules(vals)
                row = {"fixed8": metrics({v: runs["test"][v]["scores"][min(8, len(runs["test"][v]["eig"]))]
                                          for v in ids["test"]}, gt["test"], hate)}
                for rule in rules:
                    if rule not in vals["val"][ids["val"][0]]:
                        continue
                    for t in a.targets:
                        vr = [vals["val"][v][rule] for v in ids["val"]]
                        thr, _ = calibrate(vr, float(t))
                        if rule in ("qmixSG_rt10", "hG") and not per_trial:
                            assert thr == policy.calibrate(vr, float(t))[0], rule
                        calls = {v: policy.stop_calls(vals["test"][v][rule], thr) for v in ids["test"]}
                        c = np.array([calls[v] for v in ids["test"]])
                        pos = np.array([labels[v] == 1 for v in ids["test"]])
                        row[f"{rule}@{t}"] = metrics({v: runs["test"][v]["scores"][calls[v]] for v in ids["test"]},
                                                     gt["test"], hate) + [float(c.mean()), float(c[pos].mean()),
                                                                          float(c[~pos].mean())]
                per_trial.append(row)
                print(tag, order, d.name, flush=True)
            report[f"{tag}/{order}"] = {k: np.mean([r[k] for r in per_trial], 0).tolist() for k in per_trial[0]}
            report[f"{tag}/{order}/per_trial"] = per_trial
    out = ROOT / a.root / a.corpus / a.out_name
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(out)


if __name__ == "__main__":
    main()
