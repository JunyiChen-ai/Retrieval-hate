#!/usr/bin/env python
"""REPRO campaign -- CLAP knob selection on the val split.

Applies the rule frozen in `idea-stage/repro_clap/knob_rule.json` (written
before any number existed): for each dataset pick the number of FedAvg global
rounds in {1, 2, 5, 10} that maximises val frame ROC-AUC of the `fedavg11`
variant at seed 20250819, ties to the smaller round count.  Writes
`idea-stage/repro_clap/run_record.json`.  Val only; no test split is read.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/jehc223/Retrieval-hate")
OUT = ROOT / "idea-stage/repro_clap"
ROUNDS = [1, 2, 5, 10]


def main() -> int:
    subprocess.run([sys.executable,
                    str(ROOT / "scripts/repro_campaign/run_clap.py"),
                    "--stage", "knobsel_curves"], cwd=ROOT, check=True)
    res_path = OUT / "eval_clap_knobsel_val.json"
    subprocess.run([sys.executable,
                    str(ROOT / "scripts/repro_campaign/eval_frame.py"),
                    "--method", "curves",
                    "--curve-dir", str(OUT / "knobsel"),
                    "--variants", ",".join(f"r{r}" for r in ROUNDS),
                    "--method-name", "CLAP-knobsel", "--wave", "2",
                    "--supervision", "unlabelled", "--split", "val",
                    "--out", str(res_path)], cwd=ROOT, check=True)

    rows = json.loads(res_path.read_text())
    chosen, table = {}, {}
    for r in rows:
        ds, var = r["dataset"], r["variant"]
        table.setdefault(ds, {})[var] = r["pooled"]["frame_ROC_AUC"]
    for ds, d in table.items():
        best = max(ROUNDS, key=lambda k: (d[f"r{k}"], -k))
        chosen[ds] = best
    rec = {
        "method": "CLAP (CVPR 2024)",
        "rule": json.loads((OUT / "knob_rule.json").read_text()),
        "val_frame_ROC_AUC_by_round": table,
        "frozen_rounds": chosen,
        "frozen_at": subprocess.run(["date", "-Is"], capture_output=True,
                                    text=True).stdout.strip(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
    }
    (OUT / "run_record.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"frozen_rounds": chosen}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
