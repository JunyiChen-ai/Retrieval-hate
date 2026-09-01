"""Frozen contract for the complete single-Qwen mark/erase observation."""

from __future__ import annotations

import re

from src.vlm_windows import temporal_windows


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CORPORA = ("hatemm", "hateclipseg")
CONTRACT_VERSION = "single_qwen_temporal_mark_erase_v1"
EXPECTED_POSITIVE_COUNTS = {"hatemm": 85, "hateclipseg": 69}
EXPECTED_WITHIN_COUNTS = {"hatemm": 85, "hateclipseg": 67}
WINDOW_WIDTH = 16
WINDOW_STRIDE = 8
CONTEXT_WIDTH = 16
FRAMES_PER_SECTION = 2
SCORE_MAX = 10
SHIFT_FRACTIONS = tuple(range(1, 9))
SHIFT_DENOMINATOR = 9
CODE_VERSION_DESCRIPTION = (
    "2026-09-02 frozen complete-cohort single-Qwen temporal mark-erase observation"
)
PROMPT_TEMPLATE = (
    "The input is a 48-second local context split into BEFORE, CANDIDATE, and AFTER. "
    "Judge how strongly the full shown input supports that the CANDIDATE interval itself "
    "contains the localized evidence of hateful content. Hateful means an attack, slur, "
    "dehumanization, exclusion, threat, or incitement against a protected group; for a "
    "dataset that also labels direct harmful content, include its stated moderation categories. "
    "Use BEFORE and AFTER only to disambiguate quotation, condemnation, reporting, or context. "
    "Return one integer 0 to 10 only.\n"
    "BEFORE ASR: {before}\n"
    "{candidate_label} ASR: {candidate}\n"
    "AFTER ASR: {after}"
)


def parse_score(text: str):
    legal = [
        value for value in (int(token) for token in re.findall(r"\b\d+\b", text))
        if 0 <= value <= SCORE_MAX
    ]
    return legal[0] if legal else None


def candidate_windows(length: int):
    return temporal_windows(length, width=WINDOW_WIDTH, stride=WINDOW_STRIDE)


def context_span(length: int, start: int, end: int):
    return max(0, start - CONTEXT_WIDTH), min(length, end + CONTEXT_WIDTH)


def prompt(before: str, candidate: str, after: str, erased: bool):
    return PROMPT_TEMPLATE.format(
        before=before or "(none)",
        candidate="[ERASED]" if erased else (candidate or "(none)"),
        after=after or "(none)",
        candidate_label="[ERASED CANDIDATE]" if erased else "[MARKED CANDIDATE]",
    )
