#!/usr/bin/env python3
"""Record readable provenance for completed VERA HateMM train scores."""
import json
from pathlib import Path

from relation_v9.train_timeline import hatemm_train_timeline


def main():
    root = Path("results/reproduction/official_val/final/vera/hatemm/seed_234/train_infer")
    scores = root / "scores.jsonl"
    config = Path("results/reproduction/official_val/final/vera/hatemm/seed_234/frozen_config.json")
    prompt = Path("results/reproduction/official_val/tuning/vera/hatemm/selected_prompt.json")
    records = {}
    for line in scores.read_text().splitlines():
        row = json.loads(line); records[row["video_id"]] = row
    ids, lengths, timeline = hatemm_train_timeline()
    if set(records) != set(ids):
        raise RuntimeError("VERA output is not exact frozen HateMM train cohort")
    for vid in ids:
        value = records[vid].get("score_official_postprocessed", [])
        if len(value) != lengths[vid] or not all(__import__("math").isfinite(float(x)) for x in value):
            raise RuntimeError(f"VERA train alignment/nonfinite: {vid}")
    raw = root / "raw"
    raw_files = sorted(raw.glob("*.json"))
    if {x.stem for x in raw_files} != set(ids):
        raise RuntimeError("VERA raw set is not exact frozen train cohort")
    payload = {"corpus": "hatemm", "split": "train", "expert": "vera",
               "fixed_across_seeds": True, "frozen_train_ids": sorted(ids),
               "timeline": timeline,
               "scores": str(scores.resolve()),
               "checkpoint_identity": "VERA/LLaVA-Next-Video-7B-DPO",
               "fixed_config": str(config.resolve()),
               "selected_prompt": str(prompt.resolve()),
               "raw_root": str(raw.resolve()),
               "raw_file_count": len(raw_files),
               "verification": "parsed JSON, exact train IDs, finite 1fps coverage",
               "val_or_test_scores_used_as_train": False}
    (root / "evidence_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
