#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SOTA = {
    "hatemm": {"ap": .5938315566, "roc": .8161837922, "within": .6315317180},
    "hateclipseg": {"ap": .6193710950, "roc": .6050224699, "within": .5619078936},
}
OLD = Path("/home/jehc223/Retrieval-hate/runs/20260901_marked_temporal_splat_mil/pilot_seed234")


def metric(path, branch):
    row = json.loads(path.read_text())["results"][branch]
    return {"ap": row["pr_auc"], "roc": row["roc_auc"], "within": row["per_video"]["macro_auc"]}


parser = argparse.ArgumentParser()
parser.add_argument("--formal-dir", required=True)
args = parser.parse_args()
root = Path(args.formal_dir).resolve()
payload = {"corpora": {}}
for corpus in SOTA:
    old = metric(OLD / corpus / "splat/metrics.json", "score_final")
    core = metric(root / corpus / "metrics.json", "score_final")
    payload["corpora"][corpus] = {
        "old_splat": old,
        "mass_conserving_splat": core,
        "delta": {key: core[key] - old[key] for key in core},
        "all_sota": all(core[key] > SOTA[corpus][key] for key in SOTA[corpus]),
    }
payload["performance_gate"] = {
    "both_corpora_all_sota": all(row["all_sota"] for row in payload["corpora"].values())
}
payload["decision"] = "EXPAND" if payload["performance_gate"]["both_corpora_all_sota"] else "FAIL_RESET3_METHOD_1"
(root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
