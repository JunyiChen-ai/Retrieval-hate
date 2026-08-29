#!/usr/bin/env python3
"""Bind completed VERA HateMM train scores to its fixed inference identity."""
import hashlib
import json
from pathlib import Path

from relation_v4.io import sha256
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
    aggregate = hashlib.sha256("".join(
        f"{x.stem}\t{sha256(x)}\n" for x in raw_files).encode()).hexdigest()
    payload = {"corpus": "hatemm", "split": "train", "expert": "vera",
               "fixed_across_seeds": True, "frozen_train_ids": sorted(ids),
               "timeline": timeline,
               "scores": str(scores.resolve()), "scores_sha256": sha256(scores),
               "checkpoint_identity": f"vera-hatemm-{sha256(config)}",
               "fixed_config": str(config.resolve()), "fixed_config_sha256": sha256(config),
               "selected_prompt": str(prompt.resolve()), "selected_prompt_sha256": sha256(prompt),
               "raw_root": str(raw.resolve()), "raw_set_sha256": aggregate,
               "val_or_test_scores_used_as_train": False}
    (root / "evidence_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
