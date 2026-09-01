"""Synthetic invariants for the multimodal P-MIL baseline port."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PORT = load("pmil_run_test", "run.py")
MODEL = load("pmil_model_test", "model.py")


class ProposalTests(unittest.TestCase):
    def test_generator_is_valid_deterministic_and_covers_with_whole_video(self):
        score = np.asarray([0.1, 0.2, 0.9, 0.8, 0.1, 0.7, 0.1])
        first = PORT.generate_proposals(score)
        second = PORT.generate_proposals(score)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[:, 0] >= 0))
        self.assertTrue(np.all(first[:, 1] <= len(score)))
        self.assertTrue(np.all(first[:, 1] > first[:, 0]))
        self.assertTrue(np.any(np.all(first == (0, len(score)), axis=1)))

    def test_constant_proposal_score_produces_flat_frames(self):
        proposals = PORT.generate_proposals(np.linspace(0, 1, 17))
        frames = PORT.proposal_to_frames(17, proposals, np.full(len(proposals), 0.37))
        np.testing.assert_allclose(frames, 0.37, rtol=0.0, atol=0.0)

    def test_frame_readout_rejects_count_and_bound_errors(self):
        with self.assertRaises(RuntimeError):
            PORT.proposal_to_frames(4, np.asarray([[0, 4]]), np.asarray([]))
        with self.assertRaises(RuntimeError):
            PORT.proposal_to_frames(4, np.asarray([[-1, 4]]), np.asarray([0.2]))
        with self.assertRaises(RuntimeError):
            PORT.proposal_to_frames(4, np.asarray([[0.5, 4]]), np.asarray([0.2]))

    def test_producer_source_only_requests_scoped_train_and_validation_labels(self):
        source = (HERE / "run.py").read_text()
        self.assertNotIn("gt_arrays", source)
        tree = ast.parse(source)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "scoped_video_labels"
        ]
        self.assertEqual(len(calls), 2)
        splits = set()
        for call in calls:
            self.assertIsInstance(call.args[1], ast.Constant)
            splits.add(call.args[1].value)
        self.assertEqual(splits, {"train", "val"})


class ModelTests(unittest.TestCase):
    def test_positive_base_target_is_multihot_not_normalized(self):
        cas = torch.tensor([[2.0, -1.0], [0.5, 1.0]])
        attention = torch.tensor([0.0, 0.0])
        original, _ = MODEL.MultimodalPMIL._video_classification(
            cas, attention, 1, topk_divisor=8
        )
        pooled = cas.topk(1, dim=0).values.mean(0)
        expected = -torch.log_softmax(pooled, 0).sum()
        self.assertTrue(torch.allclose(original, expected))

    def test_forward_loss_and_score_are_finite(self):
        torch.manual_seed(7)
        model = MODEL.MultimodalPMIL(
            {"visual": 8, "audio": 4, "text": 8}, hidden=16,
            roi_size=12, dropout=0.0, max_train_proposals=8,
        )
        features = {
            "visual": torch.randn(20, 8),
            "audio": torch.randn(20, 4),
            "text": torch.randn(20, 8),
        }
        proposals = torch.tensor([
            [0.0, 20.0], [0.0, 4.0], [3.0, 9.0], [8.0, 16.0], [16.0, 20.0]
        ])
        outputs, used = model(features, proposals, training_sample=True)
        terms = model.loss(outputs, used, 1, epoch=3)
        self.assertTrue(all(torch.isfinite(value) for value in terms.values()))
        terms["total"].backward()
        proposal_score, video_score = model.scores(outputs)
        self.assertTrue(torch.isfinite(proposal_score).all())
        self.assertTrue(torch.isfinite(video_score))

    def test_subsampled_proposals_are_the_same_set_used_by_every_loss(self):
        torch.manual_seed(11)
        model = MODEL.MultimodalPMIL(
            {"visual": 8, "audio": 4, "text": 8}, hidden=16,
            roi_size=12, dropout=0.0, max_train_proposals=4,
        )
        features = {
            "visual": torch.randn(20, 8),
            "audio": torch.randn(20, 4),
            "text": torch.randn(20, 8),
        }
        proposals = torch.tensor([
            [float(i), float(min(i + 4, 20))] for i in range(10)
        ])
        outputs, used = model(features, proposals, training_sample=True)
        self.assertEqual(len(used), 4)
        self.assertTrue(all(len(row["cas"]) == len(used) for row in outputs.values()))
        terms = model.loss(outputs, used, 1, epoch=3)
        self.assertTrue(all(torch.isfinite(value) for value in terms.values()))

    def test_negative_completeness_targets_zero(self):
        model = MODEL.MultimodalPMIL(
            {"visual": 8, "audio": 4, "text": 8}, hidden=16,
            roi_size=12, dropout=0.0,
        )
        predicted = torch.tensor([0.0, 1.0], requires_grad=True)
        proposals = torch.tensor([[0.0, 2.0], [2.0, 4.0]])
        loss = model._completeness_loss(
            predicted, torch.tensor([10.0, 10.0]), proposals, 0.7, False
        )
        expected = (torch.sigmoid(predicted) ** 2).mean()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.all(predicted.grad > 0))

    def test_scores_average_per_modality_products(self):
        model = MODEL.MultimodalPMIL(
            {"visual": 2, "audio": 2, "text": 2}, hidden=4, dropout=0.0
        )
        outputs = {}
        values = ((3.0, -2.0, 1.0), (-1.0, 2.0, -2.0), (0.5, 0.0, 0.0))
        expected = []
        for name, (hate_logit, attention_logit, completeness_logit) in zip(
            model.modalities, values
        ):
            cas = torch.tensor([[hate_logit, 0.0]])
            attention = torch.tensor([attention_logit])
            completeness = torch.tensor([completeness_logit])
            outputs[name] = {
                "cas": cas, "attention": attention, "completeness": completeness
            }
            expected.append(
                torch.softmax(cas, 1)[:, 0]
                * torch.sigmoid(attention)
                * torch.sigmoid(completeness)
            )
        score, _ = model.scores(outputs)
        self.assertTrue(torch.allclose(score, torch.stack(expected).mean(0)))


if __name__ == "__main__":
    unittest.main()
