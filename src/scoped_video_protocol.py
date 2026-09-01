"""Shared split-scoped video labels and evaluator-test cohort contract."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCOPED_LABEL_ROOT = REPO / "results/reproduction/splits/scoped_labels"
EXCLUDED_WITHOUT_LOCALIZATION_GOLD = {
    "hatemm": frozenset({"hate_video_427"}),
    "mhclip_en": frozenset(
        {"FuXd3OYtlQU", "dtyQhHPShtQ", "h_wKRDyoG_c", "jEdH8-AZ3aU"}
    ),
    "mhclip_zh": frozenset(
        {"BV1Cc411k7sw", "BV1Dm4y1J7Pj", "BV1jk4y1Q7JZ", "BV1kT411t7ax"}
    ),
    "hateclipseg": frozenset(),
}


def evaluator_test_ids(corpus: str, split_ids):
    split_ids = list(split_ids)
    if len(split_ids) != len(set(split_ids)):
        raise RuntimeError(f"duplicate ids in {corpus} test split")
    excluded = EXCLUDED_WITHOUT_LOCALIZATION_GOLD[corpus]
    if not excluded.issubset(split_ids):
        raise RuntimeError(f"fixed no-localization exclusion changed for {corpus}")
    return [video_id for video_id in split_ids if video_id not in excluded]


def scoped_video_labels(corpus: str, split: str, expected_ids):
    path = SCOPED_LABEL_ROOT / f"{corpus}_{split}.json"
    payload = json.loads(path.read_text())
    if payload.get("corpus") != corpus or payload.get("split") != split:
        raise RuntimeError(f"scoped video-label provenance mismatch for {corpus}/{split}")
    labels = payload.get("labels")
    if not isinstance(labels, dict) or set(labels) != set(expected_ids):
        raise RuntimeError(f"scoped video-label coverage mismatch for {corpus}/{split}")
    if any(type(value) is not int or value not in (0, 1) for value in labels.values()):
        raise RuntimeError(f"invalid scoped video label for {corpus}/{split}")
    return labels
