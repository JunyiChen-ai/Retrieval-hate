#!/usr/bin/env python3
"""Write an atomic completion record for one formal validation run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    meta_path = run_dir / "train_meta.json"
    meta = json.loads(meta_path.read_text())
    selected = meta.get("selected_epoch") is not None
    required = [
        run_dir / "source_snapshot.sha256",
        run_dir / "code_commit.txt",
        run_dir / "tracked_code.patch",
        run_dir / "run.pid",
        run_dir / "run.log",
        run_dir / "augmentation_manifest.jsonl",
        run_dir / "val_scores_epoch0.jsonl",
    ]
    if selected:
        required += [
            run_dir / "rank_head.pth",
            run_dir / "val_scores.jsonl",
            run_dir / "val_metrics.json",
            run_dir / "metrics.json",
        ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"incomplete run; missing {missing}")
    plan_path = HERE / "PILOT_PLAN.md"
    review_path = HERE / "PRE_RUN_REVIEW.md"
    payload = {
        "corpus": meta["corpus"],
        "arm": meta["arm"],
        "seed": meta["seed"],
        "selected_epoch": meta.get("selected_epoch"),
        "train_meta_sha256": sha256(meta_path),
        "metrics_sha256": (
            sha256(run_dir / "metrics.json") if selected else None
        ),
        "val_scores_sha256": (
            sha256(run_dir / "val_scores.jsonl") if selected else None
        ),
        "rank_head_sha256": (
            sha256(run_dir / "rank_head.pth") if selected else None
        ),
        "source_snapshot_manifest_sha256": sha256(
            run_dir / "source_snapshot.sha256"
        ),
        "pilot_plan_sha256": sha256(plan_path),
        "pre_run_review_sha256": sha256(review_path),
    }
    target = run_dir / "completion.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
