"""Split-scoped protocol helpers; test prediction never loads test labels."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINES = REPO / "scripts/reproduction_baselines"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(BASELINES) not in sys.path:
    sys.path.insert(0, str(BASELINES))

from hate_common import data as hdata  # noqa: E402
from src.scoped_video_protocol import evaluator_test_ids, scoped_video_labels  # noqa: E402


def split_ids(corpus: str, split: str):
    ids = hdata.load_split(corpus, split)
    return evaluator_test_ids(corpus, ids) if split == "test" else ids


def supervised_split(corpus: str, split: str):
    if split not in ("train", "val"):
        raise ValueError("only train/val are supervised producer splits")
    ids = split_ids(corpus, split)
    return ids, scoped_video_labels(corpus, split, ids)


def blind_test_split(corpus: str):
    ids = split_ids(corpus, "test")
    return ids, {video_id: 0 for video_id in ids}
