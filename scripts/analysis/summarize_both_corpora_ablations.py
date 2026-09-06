"""Three-seed, two-corpus summary for a candidate: full-model confirmation (rule 8)
and per-arm mean-only ablation judgement (rule 14(g), user ruling 2026-09-06).

Reads only evaluator outputs (metrics.json); no metric recomputation.
Full model per seed = best trial of that seed's study_summary.json.
"""
import argparse
import json
import statistics
from pathlib import Path

KEYS = ["pooled_ap", "pooled_roc", "within_roc"]
SEEDS = [234, 2025, 3407]
# rule 8: gate = strongest trained baseline three-seed mean; std used for margin
GATE = {"hatemm": {"pooled_ap": (.573, .033), "pooled_roc": (.807, .019)},
        "hateclipseg": {"pooled_ap": (.562, .0), "pooled_roc": (.528, .0)}}
WITHIN_REF = {"hatemm": .632, "hateclipseg": .524}


def read_metrics(path):
    r = json.loads(Path(path).read_text())["results"]["score_av"]
    return dict(zip(KEYS, [r["pr_auc"], r["roc_auc"], r["per_video"]["macro_auc"]]))


def mean(xs):
    return statistics.mean(xs)


def std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def corpus_report(corpus, search_root, abl_root, arms, seeds):
    full, full_src, best_trials = [], [], {}
    for s in seeds:
        summ = json.loads((search_root / f"seed{s}" / "study_summary.json").read_text())
        assert len(summ["trials"]) == summ["n_trials"] and summ["best"] is not None
        b = summ["best"]["number"]
        best_trials[s] = b
        src = search_root / f"seed{s}" / f"trial{b}" / "metrics.json"
        full.append(read_metrics(src))
        full_src.append(str(src))
    rep = {"seeds": seeds, "best_trials": best_trials, "full_sources": full_src,
           "full_by_seed": full,
           "full_mean": {k: mean([f[k] for f in full]) for k in KEYS},
           "full_std_ddof1": {k: std([f[k] for f in full]) for k in KEYS},
           "within_reference_multihateloc": WITHIN_REF[corpus], "arms": {}}
    gate = {}
    for k, (g, bstd) in GATE[corpus].items():
        m = rep["full_mean"][k]
        need = max(rep["full_std_ddof1"][k], bstd, .005)
        gate[k] = {"gate": g, "margin": m - g, "required_margin": need,
                   "pass": (m - g) >= need}
    rep["rule8_confirmation"] = {**gate, "pass": all(v["pass"] for v in gate.values())}
    for arm in arms:
        srcs = [abl_root / f"seed{s}" / arm / "metrics.json" for s in seeds]
        ms = [read_metrics(p) for p in srcs]
        drops = {k: [f[k] - m[k] for f, m in zip(full, ms)] for k in KEYS}
        md = {k: mean(v) for k, v in drops.items()}
        rep["arms"][arm] = {
            "sources": [str(p) for p in srcs],
            "mean": {k: mean([m[k] for m in ms]) for k in KEYS},
            "std_ddof1": {k: std([m[k] for m in ms]) for k in KEYS},
            "full_minus_arm_by_seed": drops, "mean_drop": md,
            "n_seeds_dropping": {k: sum(v > 0 for v in drops[k]) for k in KEYS[:2]},
            "this_corpus_pass": md["pooled_ap"] >= .01 or md["pooled_roc"] >= .01}
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hatemm-search", type=Path, required=True)
    p.add_argument("--hatemm-ablations", type=Path, required=True)
    p.add_argument("--hcs-search", type=Path, required=True)
    p.add_argument("--hcs-ablations", type=Path, required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    a = p.parse_args()
    out = {"criterion": "rule 14(g), user ruling 2026-09-06: a part counts as useful when "
                        "three-seed mean drop >= .01 in pooled AP or pooled ROC on each corpus; "
                        "no per-seed requirement. within is reported only.",
           "corpora": {"hatemm": corpus_report("hatemm", a.hatemm_search, a.hatemm_ablations, a.arms, a.seeds),
                       "hateclipseg": corpus_report("hateclipseg", a.hcs_search, a.hcs_ablations, a.arms, a.seeds)}}
    out["arm_claimable_both_corpora"] = {
        arm: all(out["corpora"][c]["arms"][arm]["this_corpus_pass"] for c in out["corpora"]) for arm in a.arms}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    for c, r in out["corpora"].items():
        fm, fs = r["full_mean"], r["full_std_ddof1"]
        print(f"{c}: full {fm['pooled_ap']:.4f}±{fs['pooled_ap']:.4f} / {fm['pooled_roc']:.4f}±{fs['pooled_roc']:.4f}"
              f" / within {fm['within_roc']:.4f}±{fs['within_roc']:.4f}  rule8 pass={r['rule8_confirmation']['pass']}"
              f"  best_trials={r['best_trials']}")
    print(f"{'arm':12s} {'HMM dAP':>8s} {'HMM dROC':>9s} {'HCS dAP':>8s} {'HCS dROC':>9s}  claim")
    for arm in a.arms:
        h, s = out["corpora"]["hatemm"]["arms"][arm]["mean_drop"], out["corpora"]["hateclipseg"]["arms"][arm]["mean_drop"]
        print(f"{arm:12s} {h['pooled_ap']:+8.3f} {h['pooled_roc']:+9.3f} {s['pooled_ap']:+8.3f} {s['pooled_roc']:+9.3f}"
              f"  {out['arm_claimable_both_corpora'][arm]}")


if __name__ == "__main__":
    main()
