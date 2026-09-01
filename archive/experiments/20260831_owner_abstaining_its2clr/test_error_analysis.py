"""Small invariants for the frozen developmental test analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "owner_error_analysis_test", HERE / "analyze_test_errors.py"
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class ErrorAnalysisTests(unittest.TestCase):
    def test_safe_spearman_alignment_constant_and_finite(self):
        row = ANALYSIS.safe_spearman([1, 2, 3], [3, 2, 1])
        self.assertAlmostEqual(row["rho"], -1.0)
        self.assertEqual(row["n"], 3)
        self.assertIsNone(ANALYSIS.safe_spearman([1, 1], [1, 2])["rho"])
        with self.assertRaises(RuntimeError):
            ANALYSIS.safe_spearman([1, 2], [1])
        with self.assertRaises(RuntimeError):
            ANALYSIS.safe_spearman([1, np.nan], [1, 2])

    def test_score_delta_and_occupancy_alignment(self):
        core = {"a": 0.7, "b": 0.4, "c": 0.8}
        broadcast = {"a": 0.6, "b": 0.5, "c": 0.8}
        order = sorted(core)
        delta = np.asarray([core[key] - broadcast[key] for key in order])
        occupancy = np.asarray([0.2, 0.5, 0.9])
        np.testing.assert_allclose(delta, [0.1, -0.1, 0.0])
        self.assertEqual(ANALYSIS.safe_spearman(delta, occupancy)["n"], 3)


if __name__ == "__main__":
    unittest.main()
