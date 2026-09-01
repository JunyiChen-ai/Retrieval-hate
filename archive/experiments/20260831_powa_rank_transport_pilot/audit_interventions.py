#!/usr/bin/env python3
"""Exhaustively audit all frozen Stage-V intervention draws without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASELINES = REPO / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASELINES))

from hate_common import data as hdata  # noqa: E402
from powa_macil.dataset import usable_text_ids  # noqa: E402
from src.weak_supervision.same_corpus_insertion import (  # noqa: E402
    SameCorpusInsertionDataset,
)


def make_dataset(corpus, ids, labels, arm):
    return SameCorpusInsertionDataset(
        corpus, ids, labels, 200, "snippet", "av", 5,
        arm=arm, seed=234, min_donor_rows=12, max_donor_rows=36,
        boundary_buffer=3,
    )


def scalar(value):
    return value.item() if torch.is_tensor(value) else value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = {"split": "train", "seed": 234, "epochs": 5, "corpora": {}}
    for corpus in ("hatemm", "hateclipseg"):
        labels = hdata.load_labels(corpus)
        train_ids, _ = hdata.load_train_val(corpus, labels)
        train_ids = usable_text_ids(corpus, train_ids)
        datasets = {
            arm: make_dataset(corpus, train_ids, labels, arm)
            for arm in ("negative_donor", "shifted_mask", "positive_donor")
        }
        counts = {
            "items": 0,
            "insertions": 0,
            "ineligible_short_positive": 0,
            "stability_exceptions": 0,
            "negative_shifted_exact_match": 0,
            "positive_shared_draw_match": 0,
        }
        for epoch in range(1, 6):
            for dataset in datasets.values():
                dataset.set_epoch(epoch)
            for index in range(len(datasets["negative_donor"])):
                negative = datasets["negative_donor"][index]
                shifted = datasets["shifted_mask"][index]
                positive = datasets["positive_donor"][index]
                counts["items"] += 1
                if not bool(negative["has_insertion"]):
                    if labels[negative["recipient_id"]] == 1:
                        counts["ineligible_short_positive"] += 1
                    continue
                counts["insertions"] += 1
                if not bool(negative["has_stability_support"]):
                    counts["stability_exceptions"] += 1
                if labels[negative["donor_id"]] != 0:
                    raise AssertionError("negative donor label mismatch")
                if labels[positive["donor_id"]] != 1:
                    raise AssertionError("positive donor label mismatch")
                rows = int(negative["donor_rows"])
                if not 12 <= rows <= 36:
                    raise AssertionError("donor duration outside frozen bounds")
                for key in (
                    "recipient_id", "donor_id", "donor_crop", "donor_start",
                    "donor_rows", "insert_at", "aug_length", "recipient_map",
                    "donor_mask", "aug_v", "aug_a", "aug_t",
                ):
                    left, right = negative[key], shifted[key]
                    equal = (
                        torch.equal(left, right)
                        if torch.is_tensor(left) else left == right
                    )
                    if not equal:
                        raise AssertionError(
                            f"{corpus}/{epoch}/{index}: unmatched {key}"
                        )
                counts["negative_shifted_exact_match"] += 1
                for key in ("recipient_id", "donor_crop", "donor_rows", "insert_at"):
                    if scalar(negative[key]) != scalar(positive[key]):
                        raise AssertionError(
                            f"{corpus}/{epoch}/{index}: positive draw {key}"
                        )
                counts["positive_shared_draw_match"] += 1
                donor = shifted["donor_mask"] > 0
                control = shifted["supervision_mask"] > 0
                if bool((donor & control).any()) or int(donor.sum()) != int(control.sum()):
                    raise AssertionError("shifted control mask invalid")
        result["corpora"][corpus] = counts
    result["pass"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
