#!/usr/bin/env python3

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/reproduction_baselines"))

from hate_common import data as hdata  # noqa: E402
from model import TypedREBA  # noqa: E402
from src.scoped_video_protocol import evaluator_test_ids, scoped_video_labels  # noqa: E402


DIMS = {"audio": 4, "visual": 5, "text": 6}


def fixture():
    torch.manual_seed(9)
    feats = {
        "audio": torch.randn(4, 9, 4),
        "visual": torch.randn(4, 9, 5),
        "text": torch.randn(4, 9, 6),
    }
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=torch.bool)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    return feats, mask, labels


class ModelTests(unittest.TestCase):
    def test_outputs_and_control_are_valid(self):
        feats, mask, _ = fixture()
        model = TypedREBA(DIMS, width=8, dropout=0.0)
        output = model(feats, mask, include_control=True)
        for branch in ("score", "score_scale1_control"):
            self.assertEqual(output[branch].shape, mask.shape)
            self.assertTrue(torch.isfinite(output[branch]).all())
            self.assertGreaterEqual(float(output[branch].min()), 0.0)
            self.assertLessEqual(float(output[branch].max()), 1.0)
            self.assertEqual(float(output[branch][~mask].abs().max()), 0.0)
        self.assertFalse(torch.allclose(output["score"], output["score_scale1_control"]))

    def test_same_class_batch_has_no_false_negative_alignment_penalty(self):
        feats, mask, labels = fixture()
        model = TypedREBA(DIMS, width=8, dropout=0.0)
        output = model(feats, mask)
        all_same = torch.zeros_like(labels)
        loss = model.class_aware_bialign(output, all_same, mask)
        self.assertLess(abs(float(loss)), 1e-6)

    def test_alignment_has_finite_nonzero_encoder_gradient(self):
        feats, mask, labels = fixture()
        model = TypedREBA(DIMS, width=8, dropout=0.0)
        output = model(feats, mask)
        loss = model.class_aware_bialign(output, labels, mask)
        loss.backward()
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(("projectors", "temporal"))
            and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.0)

    def test_alpha_zero_is_exact_scale1_control(self):
        feats, mask, _ = fixture()
        model = TypedREBA(
            DIMS, width=8, residual_alpha=0.0, dropout=0.0
        ).eval()
        output = model(feats, mask, include_control=True)
        self.assertTrue(
            torch.allclose(output["score"], output["score_scale1_control"])
        )

    def test_pooling_uses_all_valid_frames_without_top_k_selection(self):
        source = (HERE / "model.py").read_text()
        tree = ast.parse(source)
        forbidden = {"topk", "kthvalue", "sort", "argsort", "quantile"}
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(forbidden & called)
        feats, mask, _ = fixture()
        model = TypedREBA(DIMS, width=8, dropout=0.0).eval()
        output = model(feats, mask)
        self.assertTrue(torch.all(output["occupancy"] > 0))
        self.assertTrue(torch.all(output["occupancy"] < 1))

    def test_short_optimization_is_finite(self):
        feats, mask, labels = fixture()
        model = TypedREBA(DIMS, width=8, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(10):
            output = model(feats, mask)
            loss, _ = model.loss(output, labels, mask, 0.1)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss))
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertLess(losses[-1], losses[0])


class ProtocolTests(unittest.TestCase):
    def test_split_labels_and_evaluator_cohort(self):
        for corpus in hdata.CORPORA:
            train_ids, val_ids = hdata.load_train_val(corpus, None, 0.1, 234)
            test_ids = hdata.load_split(corpus, "test")
            self.assertFalse(set(train_ids) & set(val_ids))
            self.assertFalse(set(train_ids) & set(test_ids))
            self.assertFalse(set(val_ids) & set(test_ids))
            scoped_video_labels(corpus, "train", train_ids)
            scoped_video_labels(corpus, "val", val_ids)
            expected = evaluator_test_ids(corpus, test_ids)
            self.assertEqual(set(expected), set(hdata.gt_arrays(corpus, "test")))

    def test_training_producer_does_not_call_test_gold_or_global_labels(self):
        source = (HERE / "train.py").read_text()
        tree = ast.parse(source)
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("gt_arrays", called)
        self.assertNotIn("load_labels", called)

    def test_evaluation_uses_only_the_shared_metric_entrypoint(self):
        source = (HERE / "evaluate.py").read_text()
        self.assertIn("from eval_baseline_scores import evaluate_scores", source)
        self.assertNotIn("from sklearn.metrics", source)


if __name__ == "__main__":
    unittest.main()
