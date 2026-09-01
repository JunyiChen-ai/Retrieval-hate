"""Small mathematical tests for the frozen path-size premise."""

from __future__ import annotations

import unittest
import copy
from importlib import import_module

import numpy as np

from src.proposal_mil import choice_readout, path_size


gate = import_module(
    "experiments.20260831_temporal_path_size_choice_mil.analyze_frozen_premise"
).gate


class PathSizeTests(unittest.TestCase):
    def test_duplicate_group_beta_one_is_invariant(self):
        proposals = np.asarray([[0, 3], [3, 6]], dtype=np.float32)
        utilities = np.asarray([0.4, -0.2])
        frames, _, evidence, _ = choice_readout(proposals, utilities, 6, 1)
        doubled = np.concatenate((proposals, proposals))
        doubled_u = np.concatenate((utilities, utilities))
        frames_2, _, evidence_2, _ = choice_readout(doubled, doubled_u, 6, 1)
        np.testing.assert_allclose(frames, frames_2, atol=1e-12, rtol=0)
        self.assertAlmostEqual(evidence, evidence_2, places=12)

    def test_path_size_counts_shared_seconds(self):
        proposals = np.asarray([[0, 2], [0, 2], [2, 4]], dtype=np.float32)
        np.testing.assert_allclose(path_size(proposals, 4), [0.5, 0.5, 1.0])

    def test_duplicate_all_invariance_with_overlapping_alternatives(self):
        proposals = np.asarray([[0, 4], [0, 2], [1, 4]], dtype=np.float32)
        utilities = np.asarray([0.2, 1.1, -0.7])
        frames, _, evidence, _ = choice_readout(proposals, utilities, 4, 1)
        doubled = np.concatenate((proposals, proposals))
        doubled_u = np.concatenate((utilities, utilities))
        frames_2, _, evidence_2, _ = choice_readout(doubled, doubled_u, 4, 1)
        np.testing.assert_allclose(frames, frames_2, atol=1e-12, rtol=0)
        self.assertAlmostEqual(evidence, evidence_2, places=12)

    def test_choice_readout_rejects_fractional_bounds_and_bad_utilities(self):
        with self.assertRaises(RuntimeError):
            choice_readout(np.asarray([[0.5, 2.0]]), np.asarray([0.0]), 3, 1)
        with self.assertRaises(RuntimeError):
            choice_readout(np.asarray([[0, 2]]), np.asarray([0.0, 1.0]), 3, 1)
        with self.assertRaises(RuntimeError):
            choice_readout(np.asarray([[0, 2]]), np.asarray([np.nan]), 3, 1)


def passing_corpus_result():
    perturbation = {
        "beta_0": {
            "bag_log_evidence_absolute_change_mean": 0.2,
            "bag_log_evidence_absolute_change_max": 0.3,
            "frame_spearman_mean": 0.8,
            "frame_spearman_n": 2,
        },
        "beta_1": {
            "bag_log_evidence_absolute_change_mean": 0.1,
            "bag_log_evidence_absolute_change_max": 0.2,
            "frame_spearman_mean": 0.9,
            "frame_spearman_n": 2,
        },
        "paired_frame_undefined": 0,
    }
    duplicate = {
        "beta_0": {
            "bag_log_evidence_absolute_change_mean": 0.5,
            "bag_log_evidence_absolute_change_max": 0.5,
            "frame_spearman_mean": 1.0,
            "frame_spearman_n": 2,
        },
        "beta_1": {
            "bag_log_evidence_absolute_change_mean": 0.0,
            "bag_log_evidence_absolute_change_max": 0.0,
            "frame_spearman_mean": 1.0,
            "frame_spearman_n": 2,
        },
        "paired_frame_undefined": 0,
    }
    return {
        "n_test_videos": 2,
        "metrics": {
            "beta_0": {"within_roc": 0.5},
            "beta_1": {"within_roc": 0.6},
        },
        "top_and_length": {
            "beta_0": {
                "exact_whole_top_fraction": 0.5,
                "top_duration_ratio_median": 0.8,
                "near_whole_top_fraction": 0.7,
                "long_proposal_posterior_mass_mean": 0.6,
            },
            "beta_1": {
                "exact_whole_top_fraction": 0.4,
                "top_duration_ratio_median": 0.7,
                "near_whole_top_fraction": 0.6,
                "long_proposal_posterior_mass_mean": 0.5,
            },
        },
        "correctable_wrong_top_cases": {
            "n": 1,
            "error_minus_best_log_path_size_mean": -0.2,
        },
        "candidate_set_perturbations": {
            "duplicate_all": duplicate,
            "near_duplicate": {
                key: (value.copy() if isinstance(value, dict) else value)
                for key, value in perturbation.items()
            },
            "thin_grid": {
                key: (value.copy() if isinstance(value, dict) else value)
                for key, value in perturbation.items()
            },
        },
        "duplicate_all_beta_1_frame_max_abs_error": 0.0,
    }


class GateTests(unittest.TestCase):
    def passing_both(self):
        first = passing_corpus_result()
        return {"hatemm": first, "hateclipseg": copy.deepcopy(first)}

    def test_gate_accepts_complete_finite_evidence(self):
        result = gate(self.passing_both())
        self.assertTrue(result["pass"])

    def test_gate_uses_duplicate_max_not_mean(self):
        corpora = self.passing_both()
        corpus = corpora["hatemm"]
        duplicate = corpus["candidate_set_perturbations"]["duplicate_all"]["beta_1"]
        duplicate["bag_log_evidence_absolute_change_mean"] = 0.0
        duplicate["bag_log_evidence_absolute_change_max"] = 1e-8
        result = gate(corpora)
        self.assertFalse(result["pass"])
        self.assertIn("hatemm:duplicate_exact_bag", result["failures"])

    def test_gate_fails_closed_on_undefined_or_unpaired_spearman(self):
        corpora = self.passing_both()
        corpus = corpora["hatemm"]
        near = corpus["candidate_set_perturbations"]["near_duplicate"]
        near["paired_frame_undefined"] = 1
        near["beta_0"]["frame_spearman_n"] = 1
        near["beta_1"]["frame_spearman_n"] = 1
        near["beta_1"]["frame_spearman_mean"] = None
        result = gate(corpora)
        self.assertFalse(result["pass"])
        self.assertIn("hatemm:near_duplicate_frames_more_stable", result["failures"])

    def test_gate_requires_both_frozen_corpora(self):
        result = gate({"hatemm": passing_corpus_result()})
        self.assertFalse(result["pass"])
        self.assertEqual(result["failures"], ["corpus_set_mismatch"])


if __name__ == "__main__":
    unittest.main()
