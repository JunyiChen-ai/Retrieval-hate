#!/usr/bin/env python3
"""Focused synthetic tests for the witness mechanism and leakage boundary."""

from __future__ import annotations

import ast
import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/reproduction_baselines"))
from model import NormalReferenceWitness
from protocol import evaluator_test_ids, scoped_video_labels
from hate_common import data as hdata


DIMS = {"audio": 4, "visual": 5, "text": 6}


def fixture():
    torch.manual_seed(7)
    feats = {
        "audio": torch.randn(2, 7, 4),
        "visual": torch.randn(2, 7, 5),
        "text": torch.randn(2, 7, 6),
    }
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool
    )
    return feats, mask


class WitnessTests(unittest.TestCase):
    def test_reference_bank_is_shared_not_arbitrarily_index_coupled(self):
        model = NormalReferenceWitness(DIMS, hidden=8, embed=3, atoms=2)
        self.assertEqual(model.references.shape, (2, 3))

    def test_outputs_are_finite_bounded_and_padding_is_zero(self):
        feats, mask = fixture()
        model = NormalReferenceWitness(DIMS, hidden=8, embed=3, atoms=2)
        output = model(feats, mask, include_controls=True)
        for name in (
            "score",
            "score_independent_transport_control",
            "score_nearest_normal_control",
        ):
            self.assertEqual(output[name].shape, mask.shape)
            self.assertTrue(torch.isfinite(output[name]).all())
            self.assertGreaterEqual(float(output[name].min()), 0.0)
            self.assertLessEqual(float(output[name].max()), 1.0)
            self.assertTrue(torch.equal(output[name][~mask], torch.zeros_like(output[name][~mask])))

    def test_positive_bags_cannot_move_normal_references(self):
        feats, mask = fixture()
        model = NormalReferenceWitness(DIMS, hidden=8, embed=3, atoms=2)
        labels = torch.ones(2)
        output = model(feats, mask, reference_gradient_mask=(labels == 0))
        loss, _ = model.loss(output, labels, mask, 0.1, 0.2)
        loss.backward()
        gradient = model.references.grad
        maximum = 0.0 if gradient is None else float(gradient.abs().max())
        self.assertEqual(maximum, 0.0)

    def test_negative_bags_train_normal_references(self):
        feats, mask = fixture()
        model = NormalReferenceWitness(DIMS, hidden=8, embed=3, atoms=2)
        labels = torch.zeros(2)
        output = model(feats, mask, reference_gradient_mask=(labels == 0))
        loss, _ = model.loss(output, labels, mask, 0.1, 0.2)
        loss.backward()
        self.assertIsNotNone(model.references.grad)
        self.assertGreater(float(model.references.grad.abs().sum()), 0.0)

    def test_shared_capacity_is_not_independent_transport(self):
        model = NormalReferenceWitness(DIMS, hidden=8, embed=3, atoms=2)
        cost = torch.tensor(
            [[[[0.0, 2.0], [0.0, 2.0], [2.0, 0.0]],
              [[0.0, 2.0], [0.0, 2.0], [2.0, 0.0]]]],
            dtype=torch.float32,
        )
        mask = torch.ones(1, 2, dtype=torch.bool)
        shared = model._transport(cost, mask, shared_capacity=True)
        independent = model._transport(cost, mask, shared_capacity=False)
        self.assertFalse(torch.allclose(shared, independent))

    def test_short_optimization_is_finite_and_learns(self):
        feats, mask = fixture()
        feats = {name: value.clone() for name, value in feats.items()}
        for name in feats:
            feats[name][1] += 2.0
        labels = torch.tensor([0.0, 1.0])
        model = NormalReferenceWitness(
            DIMS, hidden=8, embed=3, atoms=2, transport_steps=4
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        losses = []
        for _ in range(30):
            output = model(feats, mask, reference_gradient_mask=(labels == 0))
            loss, _ = model.loss(output, labels, mask, 0.01, 0.05)
            self.assertTrue(torch.isfinite(loss))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.assertTrue(
                all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                )
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0])


class ProtocolTests(unittest.TestCase):
    def test_train_validation_test_ids_are_isolated_and_labels_are_scoped(self):
        for corpus in hdata.CORPORA:
            train_ids, validation_ids = hdata.load_train_val(corpus)
            test_ids = hdata.load_split(corpus, "test")
            self.assertFalse(set(train_ids) & set(validation_ids), corpus)
            self.assertFalse(set(train_ids) & set(test_ids), corpus)
            self.assertFalse(set(validation_ids) & set(test_ids), corpus)
            scoped_video_labels(corpus, "train", train_ids)
            scoped_video_labels(corpus, "val", validation_ids)

    def test_frozen_evaluator_cohort_matches_gold_for_every_corpus(self):
        for corpus in hdata.CORPORA:
            expected = evaluator_test_ids(corpus, hdata.load_split(corpus, "test"))
            gold = hdata.gt_arrays(corpus, "test")
            self.assertEqual(set(expected), set(gold), corpus)
            self.assertEqual(len(expected), len(set(expected)), corpus)

    def test_training_producer_has_no_temporal_gold_api_call(self):
        source = (Path(__file__).resolve().parent / "train.py").read_text()
        tree = ast.parse(source)
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("gt_arrays", called_attributes)
        self.assertNotIn("load_labels", called_attributes)

    def test_evaluation_delegates_metrics_to_the_shared_evaluator(self):
        source = (Path(__file__).resolve().parent / "evaluate.py").read_text()
        self.assertIn("from eval_baseline_scores import evaluate_scores", source)
        self.assertNotIn("from sklearn.metrics", source)


if __name__ == "__main__":
    unittest.main()
