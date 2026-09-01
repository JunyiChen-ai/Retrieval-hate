#!/usr/bin/env python3
"""Choose one completed training configuration strictly by validation video AP."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--corpus-dir", required=True)
args = parser.parse_args()
root = Path(args.corpus_dir).resolve()
rows = []
for path in sorted(root.glob("trials/*/train_log.json")):
    log = json.loads(path.read_text())
    rows.append({
        "config_name": log["config_name"],
        "selected_validation_video_ap": log["selected_validation_video_ap"],
        "selected_epoch": log["selected_epoch"],
        "trial_dir": str(path.parent),
    })
if not rows:
    raise RuntimeError(f"no completed trials under {root}")
rows.sort(key=lambda row: (-row["selected_validation_video_ap"], row["config_name"]))
payload = {
    "selection_rule": "maximum validation video AP; lexical config-name tie break",
    "test_predictions_read_during_selection": False,
    "selected": rows[0],
    "all_trials": rows,
}
(root / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
