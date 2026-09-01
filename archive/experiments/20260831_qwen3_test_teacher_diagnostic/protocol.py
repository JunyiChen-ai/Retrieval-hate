"""Pure frozen contract shared by the Qwen3 producer and evaluator."""

from __future__ import annotations

import re
from pathlib import Path

from src.vlm_windows import temporal_windows


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
CORPORA = ("hatemm", "hateclipseg")
CONTRACT_VERSION = "qwen3_pointwise_positive_test_v1"
EXCLUDED_WITHOUT_LOCALIZATION_GOLD = {
    "hatemm": frozenset({"hate_video_427"}),
    "hateclipseg": frozenset(),
}
EXPECTED_POSITIVE_TEST_COUNTS = {"hatemm": 85, "hateclipseg": 69}
EXPECTED_WITHIN_COUNTS = {"hatemm": 85, "hateclipseg": 67}
WINDOW_WIDTH = 16
WINDOW_STRIDE = 8
FRAMES_PER_WINDOW = 4
SCORE_MAX = 10
WITHIN_SOTA = {
    "hatemm": 0.631531717970362,
    "hateclipseg": 0.5619078936355938,
}
PROMPT_TEMPLATE = (
    "Judge hateful content in THIS {seconds}-second window only. Hateful "
    "means an attack, slur, dehumanization, or incitement against a "
    "protected group; do not confuse generic profanity or criticism with "
    "hate. Use frames, on-screen text, and transcript. Return one integer "
    "0 to 10 only.\n"
    'Transcript: "{transcript}"'
)
CODE_VERSION_DESCRIPTION = (
    "2026-08-31 independently reviewed Qwen3 positive-test pointwise "
    "teacher diagnostic"
)
ROW_FIELDS = {
    "contract_version",
    "corpus",
    "split",
    "video_id",
    "length",
    "model",
    "windows",
    "status",
}
WINDOW_FIELDS = {"span", "generation", "parsed_score", "status"}
REPO = Path(__file__).resolve().parents[2]
VIDEO_LABEL_SOURCES = {
    corpus: REPO / "results/reproduction/splits/scoped_labels" / f"{corpus}_test.json"
    for corpus in CORPORA
}


def prompt(transcript: str, seconds: int) -> str:
    return PROMPT_TEMPLATE.format(transcript=transcript, seconds=seconds)


def parse_score(text: str):
    values = [int(token) for token in re.findall(r"\b\d+\b", text)]
    legal = [value for value in values if 0 <= value <= SCORE_MAX]
    return legal[0] if legal else None


def positive_test_cohort(corpus, split_ids, labels):
    if corpus not in CORPORA:
        raise ValueError(f"unsupported corpus: {corpus}")
    split_ids = list(split_ids)
    if len(split_ids) != len(set(split_ids)):
        raise RuntimeError(f"duplicate ids in {corpus} test split")
    missing_labels = sorted(set(split_ids) - set(labels))
    if missing_labels:
        raise RuntimeError(f"{corpus}: test ids missing video labels: {missing_labels}")
    exclusions = EXCLUDED_WITHOUT_LOCALIZATION_GOLD[corpus]
    if not exclusions.issubset(split_ids):
        raise RuntimeError(f"{corpus}: fixed evaluator exclusion absent from test split")
    if any(labels[video_id] != 1 for video_id in exclusions):
        raise RuntimeError(f"{corpus}: fixed evaluator exclusion is not video-positive")
    result = sorted(
        video_id for video_id in split_ids
        if labels[video_id] == 1 and video_id not in exclusions
    )
    if len(result) != EXPECTED_POSITIVE_TEST_COUNTS[corpus]:
        raise RuntimeError(
            f"{corpus}: positive evaluator cohort count changed: {len(result)}"
        )
    return result


def expected_config(corpus: str, predictions) -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "corpus": corpus,
        "split": "test_positive",
        "cohort": "complete video-level-positive fixed evaluator-test cohort",
        "expected_positive_test_videos": EXPECTED_POSITIVE_TEST_COUNTS[corpus],
        "video_label_source": str(VIDEO_LABEL_SOURCES[corpus].resolve()),
        "model": MODEL_ID,
        "prompt_template": PROMPT_TEMPLATE,
        "window_width_seconds": WINDOW_WIDTH,
        "window_stride_seconds": WINDOW_STRIDE,
        "frames_per_window": FRAMES_PER_WINDOW,
        "frame_grid": "existing 1 fps frames",
        "asr": "existing timestamped ASR, overlap-selected per window",
        "decoding": {
            "do_sample": False,
            "max_new_tokens": 8,
            "score_range_inclusive": [0, SCORE_MAX],
        },
        "predictions": str(predictions.resolve()),
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
    }


def validate_prediction_row(
    row: dict,
    corpus: str,
    expected_video_id: str | None = None,
    expected_length: int | None = None,
) -> None:
    if set(row) != ROW_FIELDS:
        raise RuntimeError("prediction row schema mismatch")
    if (
        row["contract_version"] != CONTRACT_VERSION
        or row["corpus"] != corpus
        or row["split"] != "test_positive"
        or row["model"] != MODEL_ID
        or row["status"] not in ("ok", "inference_failure")
        or not isinstance(row["video_id"], str)
        or not row["video_id"]
    ):
        raise RuntimeError("prediction row provenance mismatch")
    if expected_video_id is not None and row["video_id"] != expected_video_id:
        raise RuntimeError("prediction row is not the expected cohort item")
    if type(row["length"]) is not int or row["length"] <= 0:
        raise RuntimeError("prediction row length mismatch")
    if expected_length is not None and row["length"] != expected_length:
        raise RuntimeError("prediction row does not match the current 1 fps feature length")
    if not isinstance(row["windows"], list):
        raise RuntimeError("prediction windows mismatch")

    expected_spans = [
        [start, end]
        for start, end in temporal_windows(
            row["length"], width=WINDOW_WIDTH, stride=WINDOW_STRIDE
        )
    ]
    observed_spans = []
    inference_failure = False
    for window in row["windows"]:
        if set(window) != WINDOW_FIELDS:
            raise RuntimeError("prediction window schema mismatch")
        status = window["status"]
        score = window["parsed_score"]
        generation = window["generation"]
        if status not in ("ok", "parse_failure", "inference_failure"):
            raise RuntimeError("prediction window status mismatch")
        if (
            not isinstance(window["span"], list)
            or len(window["span"]) != 2
            or any(type(value) is not int for value in window["span"])
            or type(score) is not int
            or not 0 <= score <= SCORE_MAX
            or not isinstance(generation, str)
        ):
            raise RuntimeError("prediction window value mismatch")
        parsed = parse_score(generation)
        if status == "ok" and parsed != score:
            raise RuntimeError("successful window does not match its generation")
        if status == "parse_failure" and (parsed is not None or score != 0):
            raise RuntimeError("parse failure must retain an unparsable generation and zero score")
        if status == "inference_failure" and score != 0:
            raise RuntimeError("inference failure must carry zero score")
        inference_failure |= status == "inference_failure"
        observed_spans.append(window["span"])
    if observed_spans != expected_spans:
        raise RuntimeError("prediction windows do not match the frozen window recipe")
    expected_status = "inference_failure" if inference_failure else "ok"
    if row["status"] != expected_status:
        raise RuntimeError("video status disagrees with its window statuses")
