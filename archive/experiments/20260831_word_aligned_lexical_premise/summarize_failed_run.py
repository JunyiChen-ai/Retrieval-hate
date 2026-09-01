#!/usr/bin/env python
"""Record a producer-stage hard failure without reading labels or test GT."""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    rows = [json.loads(line) for line in
            (run_dir / "word_asr.jsonl").read_text().splitlines() if line.strip()]
    counts = Counter(row.get("status", "UNKNOWN") for row in rows)
    errors = [
        {"video_id": row["video_id"],
         "error_type": row.get("error_type"), "error": row.get("error")}
        for row in rows if row.get("status") == "ERROR"
    ]
    artifact = {
        "stage": "strict_word_asr_producer",
        "corpus": "hateclipseg",
        "split": "test",
        "developmental_test_evidence": True,
        "test_gt_read": False,
        "processed_before_early_stop": len(rows),
        "planned_manifest_size": 79,
        "status_counts": dict(sorted(counts.items())),
        "errors": errors,
        "frozen_gate": "zero ASR ERROR rows",
        "gate_pass": counts.get("ERROR", 0) == 0,
        "evaluation_ran": False,
        "decision": "STOP_AND_ARCHIVE",
        "reason": (
            "The frozen zero-error gate became impossible after strict native "
            "word timestamps failed the combined finite/positive/bounded/monotonic "
            "validation on multiple early videos. "
            "No sorting, chunk fallback, metric evaluation, or HMM expansion was run."
        ),
    }
    if artifact["gate_pass"]:
        raise RuntimeError("failure summarizer called although the hard gate passed")
    (run_dir / "metrics.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
