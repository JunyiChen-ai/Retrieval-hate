"""Synthetic invariants for frozen post-training test error analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "pmil_error_analysis_test", HERE / "analyze_test_errors.py"
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class ErrorAnalysisTests(unittest.TestCase):
    def test_half_open_interval_iou(self):
        self.assertAlmostEqual(ANALYSIS.interval_iou((0, 4), (2, 6)), 2 / 6)
        self.assertEqual(ANALYSIS.interval_iou((0, 2), (2, 4)), 0.0)
        self.assertEqual(ANALYSIS.interval_iou((1, 4), (1, 4)), 1.0)

    def test_proposal_oracle_scores_and_per_gt_recall(self):
        proposals = np.asarray([[0, 2], [1, 5], [5, 8]], dtype=np.float64)
        gold = [(1, 3), (6, 8)]
        scores, best = ANALYSIS.proposal_oracle_scores(proposals, gold)
        np.testing.assert_allclose(scores, [1 / 3, 0.5, 2 / 3])
        np.testing.assert_allclose(best, [0.5, 2 / 3])

    def test_top_summary_is_tie_order_invariant(self):
        proposals = np.asarray([[0, 10], [0, 2], [2, 6]], dtype=np.float64)
        scores = np.asarray([0.7, 0.7, 0.2])
        first = ANALYSIS.top_proposal_summary(proposals, scores, 0)
        order = np.asarray([1, 0, 2])
        second = ANALYSIS.top_proposal_summary(proposals[order], scores[order], 1)
        self.assertTrue(first["whole_is_top"])
        self.assertEqual(first, second)
        self.assertEqual(first["top_tie_count"], 2)
        self.assertEqual(first["top_tied_length_median"], 6.0)


if __name__ == "__main__":
    unittest.main()
