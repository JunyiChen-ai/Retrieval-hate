"""Pure, model-free protocol constants for the OmniVTG test diagnosis."""

from __future__ import annotations

import re


MODEL_ID = "zhengmh/OmniVTG-7B"
CONTRACT_VERSION = "omnivtg_positive_test_teacher_v1"
QUERY = (
    "content that attacks, dehumanizes, threatens, or expresses hatred toward "
    "a person or group based on identity"
)

# HateMM's frozen project test split has one video-level-positive item with no
# usable localization gold.  This fixed membership exclusion is documented by
# the shared evaluation protocol; it carries no frame/span values.  Keeping it
# here lets the producer select the evaluator cohort without opening GT arrays.
EVALUATOR_TEST_EXCLUSIONS = {
    "hatemm": frozenset({"hate_video_427"}),
    "hateclipseg": frozenset(),
}
EXPECTED_POSITIVE_TEST_COUNTS = {"hatemm": 85, "hateclipseg": 69}
EXPECTED_WITHIN_COUNTS = {"hatemm": 85, "hateclipseg": 67}
FORMAL_RUNTIME_VERSIONS = {
    "vllm": "0.9.2",
    "torch": "2.7.0+cu128",
    "transformers": "4.52.4",
    "qwen-vl-utils": "0.0.14",
}
CODE_VERSION_DESCRIPTION = (
    "2026-08-31 independently reviewed OmniVTG eager-vLLM positive-test "
    "teacher premise implementation"
)
ROW_FIELDS = {
    "contract_version",
    "video_id",
    "corpus",
    "split",
    "model",
    "query",
    "source_video",
    "parse_ok",
    "interval_seconds",
    "completion",
    "error_type",
    "error_message",
    "traceback",
}


def positive_test_cohort(corpus, split_ids, labels):
    if corpus not in EVALUATOR_TEST_EXCLUSIONS:
        raise ValueError(f"unsupported corpus: {corpus}")
    split_ids = list(split_ids)
    if len(split_ids) != len(set(split_ids)):
        raise RuntimeError(f"duplicate ids in {corpus} test split")
    missing_labels = sorted(set(split_ids) - set(labels))
    if missing_labels:
        raise RuntimeError(f"{corpus}: test ids missing video labels: {missing_labels}")
    exclusions = EVALUATOR_TEST_EXCLUSIONS[corpus]
    if not exclusions.issubset(split_ids):
        raise RuntimeError(f"{corpus}: fixed evaluator exclusions absent from test split")
    if any(labels[video_id] != 1 for video_id in exclusions):
        raise RuntimeError(f"{corpus}: fixed evaluator exclusion is not video-positive")
    cohort = [
        video_id for video_id in split_ids
        if labels[video_id] == 1 and video_id not in exclusions
    ]
    if len(cohort) != EXPECTED_POSITIVE_TEST_COUNTS[corpus]:
        raise RuntimeError(
            f"{corpus}: positive evaluator cohort count changed: {len(cohort)}"
        )
    return cohort


def parse_interval(text):
    if not isinstance(text, str):
        return None
    answer_blocks = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if not answer_blocks:
        return None
    matches = re.findall(
        r"From\s+([0-9]+(?:\.[0-9]+)?)\s+seconds\s+to\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s+seconds",
        answer_blocks[-1],
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    start, end = map(float, matches[-1])
    return [start, end] if end >= start else None
