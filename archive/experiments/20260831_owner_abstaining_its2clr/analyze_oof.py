"""Train-only diagnostics for intervention stability and carrier coverage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from protocol import supervised_split


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    train_ids, labels = supervised_split(cache["corpus"], "train")
    if cache["train_ids"] != train_ids:
        raise RuntimeError("OOF diagnostic train cohort mismatch")
    modalities = cache["modalities"]
    totals = {name: {"n": 0, "agree": 0, "top_n": 0, "top_carrier": 0,
                     "videos": 0, "videos_with_carrier": 0,
                     "carrier_margin_sum": 0.0}
              for name in modalities}
    for video_id in train_ids:
        if labels[video_id] != 1:
            continue
        row = cache["rows"][video_id]
        first, second = row["deletion_centroid"], row["deletion_neighbor"]
        top_count = max(1, math.ceil(len(first) / 3))
        top = torch.argsort(row["fused_score"], descending=True,
                            stable=True)[:top_count]
        for index, name in enumerate(modalities):
            a, b = first[:, index] > 0, second[:, index] > 0
            carrier = a & b
            top_carrier = carrier[top]
            item = totals[name]
            item["n"] += len(a)
            item["agree"] += int((a == b).sum())
            item["top_n"] += len(top)
            item["top_carrier"] += int(top_carrier.sum())
            item["videos"] += 1
            item["videos_with_carrier"] += int(top_carrier.any())
            if top_carrier.any():
                margin = torch.minimum(first[top, index], second[top, index])
                item["carrier_margin_sum"] += float(margin[top_carrier].sum())
    result = {}
    for name, item in totals.items():
        result[name] = {
            "intervention_sign_agreement": item["agree"] / max(item["n"], 1),
            "stable_carrier_rate_in_positive_top_third":
                item["top_carrier"] / max(item["top_n"], 1),
            "positive_video_coverage":
                item["videos_with_carrier"] / max(item["videos"], 1),
            "mean_minimum_positive_deletion_margin":
                item["carrier_margin_sum"] / max(item["top_carrier"], 1),
            "n_positive_seconds": item["n"],
        }
    payload = {
        "corpus": cache["corpus"], "arm": cache["arm"],
        "split": "train", "test_used": False,
        "definition": "final iterative OOF evidence; positive videos only",
        "modalities": result,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

