#!/usr/bin/env python3
"""Generate one MACIL-SD HateMM train dense score file from its frozen seed."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from relation_v4.io import sha256
from relation_v9.train_timeline import hatemm_train_timeline
from macilsd.infer_train_label_free import infer_hatemm_train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", required=True, type=int, choices=(234, 2025, 3407))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    root = Path("results/reproduction/official_val/final/macilsd/hatemm") / f"seed_{a.seed}"
    meta_path, model_path = root / "train_meta.json", root / "model.pth"
    meta = json.loads(meta_path.read_text())
    args = dict(meta.get("args", {}))
    if args.get("corpus") != "hatemm" or int(args.get("seed", -1)) != a.seed:
        raise RuntimeError("frozen MACIL train_meta corpus/seed mismatch")
    if args.get("modality") != "av":
        raise RuntimeError("expected frozen MACIL AV checkpoint")
    out = root / "train_infer"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"fresh output required: {out}")
    args.update(device=a.device, out_dir=str(out), limit_videos=0)
    score_path = Path(infer_hatemm_train(SimpleNamespace(**args), str(model_path)))
    records = {}
    for line in score_path.read_text().splitlines():
        row = json.loads(line); records[row["video_id"]] = row
    ids, lengths, timeline = hatemm_train_timeline()
    if set(records) != set(ids):
        raise RuntimeError("MACIL train output is not exact frozen HateMM train cohort")
    for vid in ids:
        for key in ("score_av", "score_visual"):
            values = records[vid].get(key)
            if values is None or len(values) != lengths[vid]:
                raise RuntimeError(f"MACIL train alignment failure {vid}/{key}")
    payload = {"corpus": "hatemm", "split": "train", "expert": "macilsd",
               "seed": a.seed, "frozen_train_ids": sorted(ids), "timeline": timeline,
               "scores": str(score_path.resolve()), "scores_sha256": sha256(score_path),
               "checkpoint": str(model_path.resolve()),
               "checkpoint_sha256": sha256(model_path),
               "train_meta": str(meta_path.resolve()),
               "train_meta_sha256": sha256(meta_path),
               "val_or_test_scores_used_as_train": False}
    (out / "evidence_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
