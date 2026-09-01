#!/usr/bin/env python3
"""Deterministic unit tests for the actual ordinal student objective."""

from __future__ import annotations

import unittest

import numpy as np
import torch

import train
from src.powa_residual import ContextQuotientResidualHead


class OrdinalMethodTests(unittest.TestCase):
    def test_percentile_ties(self):
        np.testing.assert_allclose(train.percentile([0, 0, 1]), [.25, .25, 1])

    def test_pairs_obey_gap_and_per_anchor_bound(self):
        order = np.asarray([0.0, .1, .4, .8, 1.0])
        pairs = train.nearest_gap_pairs(order, .2)
        self.assertTrue(all(order[high] >= order[low] + .2
                            for high, low in pairs))
        incidence = {i: 0 for i in range(len(order))}
        for high, low in pairs:
            incidence[high] += 1
            incidence[low] += 1
        self.assertTrue(pairs)
        self.assertLessEqual(len(pairs), 2 * len(order))

    def test_pair_loss_rewards_correct_orientation(self):
        high = torch.tensor([[2, -1]])
        low = torch.tensor([[0, -1]])
        correct = train.pairwise_loss(
            torch.tensor([[0.0, 0.0, 2.0]]), high, low, .25
        )
        reversed_ = train.pairwise_loss(
            torch.tensor([[2.0, 0.0, 0.0]]), high, low, .25
        )
        self.assertLess(float(correct), float(reversed_))

    def test_source_ties_create_no_positional_direction(self):
        self.assertEqual(train.orient_pairs([(3, 0)], np.zeros(4)), [])

    def test_mil_ignores_padding(self):
        score = torch.tensor([[.1, .8, .2, 1.0]])
        got = train.mil_probability(score, torch.tensor([3]))
        self.assertAlmostEqual(float(got), .8, places=6)

    def test_no_pair_loss_is_differentiable_zero(self):
        logits = torch.randn(2, 3, requires_grad=True)
        missing = torch.full((2, 4), -1)
        loss = train.pairwise_loss(logits, missing, missing, .25)
        loss.backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_residual_is_padding_invariant(self):
        torch.manual_seed(7)
        head = ContextQuotientResidualHead(text_dim=8, hidden=128).eval()
        torch.nn.init.normal_(head.output.weight)
        torch.nn.init.normal_(head.output.bias)
        short = {
            "audio_rep": torch.randn(1, 5, 128),
            "visual_rep": torch.randn(1, 5, 128),
            "primitive_logits": torch.randn(1, 5, 6),
            "frame_prob": torch.sigmoid(torch.randn(1, 5)),
            "base_frame_logits": torch.randn(1, 5, 1),
        }
        text = torch.randn(1, 5, 8)
        valid = torch.ones(1, 5, dtype=torch.bool)
        expected = head(short, text, valid)
        padded = {
            key: torch.cat([value, torch.randn(
                1, 3, *value.shape[2:]
            )], dim=1)
            for key, value in short.items()
        }
        padded_text = torch.cat([text, torch.randn(1, 3, 8)], dim=1)
        padded_valid = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
        actual = head(padded, padded_text, padded_valid)
        torch.testing.assert_close(actual[:, :5], expected, atol=1e-6, rtol=1e-6)

    def test_long_video_uses_training_grid_and_lifts_back(self):
        length, width = 425, 200
        base = torch.arange(length, dtype=torch.float32)[None, :, None]
        v, a, t, index, valid_length = train.fixed_context_tensors(
            base, base, base, width
        )
        self.assertEqual(v.shape[1], width)
        self.assertEqual(valid_length, width)
        np.testing.assert_array_equal(v[0, :, 0].numpy(), index)
        residual = index.astype(float)[None]
        lifted = train.lift_residual(residual, length, index)
        self.assertAlmostEqual(float(lifted.mean()), 0.0, places=12)
        np.testing.assert_allclose(
            lifted[0, index] - lifted[0, index].mean(),
            residual[0] - residual[0].mean(),
        )

    def test_diagnostic_order_uses_fixed_grid(self):
        from test_teacher_diagnostic import fixed_grid_order

        order = np.arange(425, dtype=float)
        got = fixed_grid_order(order)
        index = np.linspace(0, 424, 200, dtype=np.uint16).astype(np.int64)
        np.testing.assert_allclose(got[index], order[index])

    def test_transport_all_ties_is_anchor_identity(self):
        from test_teacher_diagnostic import transport

        anchor = np.asarray([.8, .1, .4, .2])
        np.testing.assert_array_equal(transport(anchor, np.zeros(4)), anchor)

    def test_transport_partial_ties_preserves_anchor_tie_order(self):
        from test_teacher_diagnostic import transport

        anchor = np.asarray([.8, .1, .4, .2])
        order = np.asarray([0., 0., 1., 1.])
        output = transport(anchor, order)
        self.assertGreater(output[0], output[1])
        self.assertGreater(output[2], output[3])
        np.testing.assert_array_equal(np.sort(output), np.sort(anchor))

    def test_transport_without_ties_matches_strict_rank_assignment(self):
        from test_teacher_diagnostic import transport

        anchor = np.asarray([.8, .1, .4, .2])
        order = np.asarray([2., 0., 3., 1.])
        expected = np.empty_like(anchor)
        expected[np.argsort(order)] = np.sort(anchor)
        np.testing.assert_array_equal(transport(anchor, order), expected)


if __name__ == "__main__":
    unittest.main()
