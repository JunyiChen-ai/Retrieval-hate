"""Export this repository's HateClipSeg split for the baseline reruns in ~/Hate-follow-up (README.md, section
"HateClipSeg reruns on this repository's split").

Source: hier_evidence_common.load_fixed_cohort('hateclipseg') (train / val / test ids and the video-level labels).
Output: runs/20261115_hf_baselines/hcs_split/split.json with
    train, val, test          id lists
    train_val_labels          video id -> 0 / 1 for the train and val ids only
    test_gt_seconds           video id -> length of this repository's per-second GT array (lengths only)
No test label and no per-second label is written.
"""
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import hier_evidence_common as hc                                       # noqa: E402


def main():
    labels, ids, gt, excluded = hc.load_fixed_cohort("hateclipseg")
    out = {"source": "src/hier_evidence_common.py load_fixed_cohort('hateclipseg')", "host": socket.gethostname(),
           "train": list(ids["train"]), "val": list(ids["val"]), "test": list(ids["test"]),
           "train_val_labels": {v: int(labels[v]) for s in ("train", "val") for v in ids[s]},
           "test_gt_seconds": {v: int(len(gt["test"][v])) for v in ids["test"]},
           "excluded": excluded}
    dest = ROOT / "runs" / "20261115_hf_baselines" / "hcs_split"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "split.json").write_text(json.dumps(out, indent=1) + "\n")
    print({k: len(out[k]) for k in ("train", "val", "test")},
          {s: sum(out["train_val_labels"][v] for v in ids[s]) for s in ("train", "val")}, excluded)


if __name__ == "__main__":
    main()
