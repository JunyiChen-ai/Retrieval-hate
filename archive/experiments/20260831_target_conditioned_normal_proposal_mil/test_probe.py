"""Synthetic invariants for the frozen premise probe."""

from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("conditional_probe", HERE / "probe.py")
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)
DECIDE_SPEC = importlib.util.spec_from_file_location(
    "conditional_decision", HERE / "decide_premise.py"
)
DECIDE = importlib.util.module_from_spec(DECIDE_SPEC)
assert DECIDE_SPEC.loader is not None
DECIDE_SPEC.loader.exec_module(DECIDE)


class ProposalInvariantTests(unittest.TestCase):
    def test_bounds_cover_every_second_and_include_whole_video(self):
        for length in (1, 3, 17, 129):
            bounds = PROBE.proposal_bounds(length)
            self.assertTrue(np.all(bounds[:, 0] >= 0))
            self.assertTrue(np.all(bounds[:, 1] <= length))
            self.assertTrue(np.all(bounds[:, 1] > bounds[:, 0]))
            self.assertTrue(np.any(np.all(bounds == (0, length), axis=1)))
            coverage = np.zeros(length, dtype=np.int64)
            for start, end in bounds:
                coverage[start:end] += 1
            self.assertTrue(np.all(coverage > 0))

    def test_inside_surrounding_residual_exact(self):
        topic = np.arange(4, dtype=np.float64)[:, None]
        rows = np.asarray([1.0, 2.0, 4.0, 8.0])[:, None]
        bounds, proposal_topic, residual = PROBE.proposal_features(topic, rows)
        lookup = {tuple(bound): index for index, bound in enumerate(bounds)}

        middle = lookup[(1, 3)]
        self.assertAlmostEqual(proposal_topic[middle, 0], 1.5)
        self.assertAlmostEqual(residual[middle, 0], -1.5)

        left_edge = lookup[(0, 2)]
        self.assertAlmostEqual(residual[left_edge, 0], -4.5)

        whole = lookup[(0, 4)]
        self.assertAlmostEqual(residual[whole, 0], 3.75)

    def test_constant_energy_readout_is_flat(self):
        for length in (1, 7, 129):
            bounds = PROBE.proposal_bounds(length)
            score = PROBE.frame_readout(length, bounds, np.full(len(bounds), 2.75))
            np.testing.assert_allclose(score, 2.75, rtol=0.0, atol=1e-12)

    def test_producer_source_never_loads_test_labels_or_temporal_gold(self):
        source = (HERE / "probe.py").read_text()
        self.assertNotIn("gt_arrays", source)
        tree = ast.parse(source)
        scoped_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "scoped_video_labels"
        ]
        self.assertEqual(len(scoped_calls), 1)
        self.assertIsInstance(scoped_calls[0].args[1], ast.Constant)
        self.assertEqual(scoped_calls[0].args[1].value, "train")


class FrozenGateTests(unittest.TestCase):
    @staticmethod
    def _write_corpus(root, corpus, conditional, unconditional, support):
        corpus_root = Path(root) / corpus
        corpus_root.mkdir()
        (corpus_root / "metrics.json").write_text(json.dumps({
            "corpus": corpus,
            "split": "test",
            "results": {
                "score_conditional": {"per_video": {"macro_auc": conditional}},
                "score_unconditional": {"per_video": {"macro_auc": unconditional}},
            },
        }))
        (corpus_root / "support.json").write_text(json.dumps({
            "test_proposal_support_fraction": support,
            "support_pass": support >= 0.80,
        }))

    def test_gate_requires_support_positive_direction_and_one_large_gain(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_corpus(root, "hatemm", 0.62, 0.60, 0.80)
            self._write_corpus(root, "hateclipseg", 0.56, 0.559, 0.90)
            DECIDE.main(["--run-root", root])
            verdict = json.loads((Path(root) / "verdict.json").read_text())
            self.assertTrue(verdict["premise_pass_both"])

        with tempfile.TemporaryDirectory() as root:
            self._write_corpus(root, "hatemm", 0.62, 0.60, 0.80)
            self._write_corpus(root, "hateclipseg", 0.558, 0.559, 0.90)
            DECIDE.main(["--run-root", root])
            verdict = json.loads((Path(root) / "verdict.json").read_text())
            self.assertFalse(verdict["premise_pass_both"])

    def test_gate_accepts_exact_gain_boundary_and_rejects_bad_support(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_corpus(root, "hatemm", 0.58, 0.56, 0.80)
            self._write_corpus(root, "hateclipseg", 0.561, 0.56, 0.90)
            DECIDE.main(["--run-root", root])
            verdict = json.loads((Path(root) / "verdict.json").read_text())
            self.assertTrue(verdict["premise_pass_both"])

        with tempfile.TemporaryDirectory() as root:
            self._write_corpus(root, "hatemm", 0.58, 0.56, 0.80)
            self._write_corpus(root, "hateclipseg", 0.561, 0.56, 0.79)
            DECIDE.main(["--run-root", root])
            verdict = json.loads((Path(root) / "verdict.json").read_text())
            self.assertFalse(verdict["premise_pass_both"])


if __name__ == "__main__":
    unittest.main()
