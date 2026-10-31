"""Per-setting trial selection over all 20 trials of each frozen study (reads returned JSON only).

The main tables picked each seed's trial with the training-time fixed8 no-copy criterion, so the
adaptive setting never chose its own model. Here each setting (fixed0/8/32, qmixSG_rt10) picks the
trial maximising its own (AP+ROC)/2, once on validation and once on test (development protocol),
and reports that trial's test metrics. Checkpoint epochs stay as saved by the original searches.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ("fixed0", "fixed8", "qmixSG_rt10", "fixed32")
RULE = "qmixSG_rt10"
SEEDS = (234, 2025, 3407)


def load(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/20261002_complete_io_method")
    a = ap.parse_args()
    root = (ROOT / a.root).resolve()
    report = {"created": datetime.now().astimezone().isoformat(), "settings": SETTINGS, "corpora": {}}
    for corpus in ("hatemm", "hateclipseg", "dehate"):
        rows = {}
        for summary in sorted((root / "all_trials" / corpus).glob("*/results/summary.json")):
            for row in load(summary)["trials"].values():
                rows[(row["seed"], row["trial"])] = row
        assert sorted({s for s, _ in rows}) == list(SEEDS) and all(
            sum(1 for s, _ in rows if s == seed) == 20 for seed in SEEDS), f"{corpus}: incomplete all-trial results"
        # The two originally selected trials must reproduce the earlier complete-combination results. GPU
        # nondeterminism can move a rerun slightly, so record the largest deviation and allow at most .005.
        original = (root / corpus / "integrated/results/summary.json" if corpus != "dehate" else None)
        originals = ([load(original)] if original else
                     [load(root / "dehate" / f"seed{s}/results/summary.json") for s in SEEDS])
        reproduced, max_diff, calls_differ, calls_total, inexact = 0, 0.0, 0, 0, []
        for o in originals:
            for row in o["trials"].values():
                new = rows[(row["seed"], row["trial"])]
                for name in SETTINGS:
                    d = max(abs(x - y) for x, y in zip(new["results"][name]["ap_roc_within"],
                                                       row["results"][name]["ap_roc_within"]))
                    old_calls, new_calls = row["results"][name]["calls_per_video"], new["results"][name]["calls_per_video"]
                    assert set(old_calls) == set(new_calls)
                    n = sum(old_calls[v] != new_calls[v] for v in old_calls)
                    max_diff, calls_differ, calls_total = max(max_diff, d), calls_differ + n, calls_total + len(old_calls)
                    if d or n:
                        inexact.append({"seed": row["seed"], "trial": row["trial"], "setting": name,
                                        "max_abs_metric_diff": d, "videos_with_different_calls": n})
                reproduced += 1
        assert max_diff <= 0.005, f"{corpus}: rerun deviates {max_diff} from the original results"
        block = {"reproduced_original_trials": reproduced, "reproduction": {
            "max_abs_metric_diff": max_diff, "videos_with_different_calls": calls_differ,
            "video_settings_compared": calls_total, "inexact": inexact}, "by": {}}
        for basis, key in (("validation", "val_results"), ("test", "results")):
            per = {}
            for name in SETTINGS:
                chosen = []
                for seed in SEEDS:
                    cands = sorted((r for (s, _), r in rows.items() if s == seed), key=lambda r: r["trial"])
                    best = max(cands, key=lambda r: sum(r[key][name]["ap_roc_within"][:2]) / 2)
                    chosen.append({"seed": seed, "trial": best["trial"], "checkpoint_epoch": best["checkpoint_epoch"],
                                   "selection_value": sum(best[key][name]["ap_roc_within"][:2]) / 2,
                                   "test": best["results"][name]["ap_roc_within"],
                                   "test_mean_calls": best["results"][name]["mean_calls"],
                                   "metric_source": best["results"][name]["metric_source"]})
                v = np.array([c["test"] for c in chosen])
                per[name] = {"chosen": chosen, "mean": v.mean(0).tolist(), "std_population": v.std(0).tolist(),
                             "calls_mean": float(np.mean([c["test_mean_calls"] for c in chosen]))}
            d = np.array([np.subtract(a_["test"], b_["test"])
                          for a_, b_ in zip(per[RULE]["chosen"], per["fixed8"]["chosen"])])
            per["auto_minus_fixed8_each_own_trial"] = {"seed_mean": d.mean(0).tolist(),
                "seed_std_population": d.std(0).tolist(), "seed_values": d.tolist()}
            block["by"][basis] = per
        # Same-trial test comparison over all 60 trials: is adaptive better than fixed8 on a typical model?
        all_d = np.array([np.subtract(r["results"][RULE]["ap_roc_within"], r["results"]["fixed8"]["ap_roc_within"])
                          for r in rows.values()])
        block["auto_minus_fixed8_all_60_trials"] = {"mean": all_d.mean(0).tolist(),
            "std_population": all_d.std(0).tolist(), "share_positive": (all_d > 0).mean(0).tolist()}
        report["corpora"][corpus] = block
    out = root / "analysis/select_by_setting.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
