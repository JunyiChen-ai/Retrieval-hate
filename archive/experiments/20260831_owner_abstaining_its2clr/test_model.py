from __future__ import annotations

import unittest

import torch

from model import ABSTAIN, BACKGROUND, CARRIER, CarrierItS2CLR
from oof import neighbor_replacement
from states import batch_states, pace_for_epoch, positive_states


def row():
    return {
        "fused_score": torch.tensor([0.9, 0.8, 0.7, 0.1, 0.0, 0.2]),
        "branch_score": torch.tensor([
            [0.1, 0.6, 0.2], [0.9, 0.4, 0.3], [0.8, 0.1, 0.7],
            [0.2, 0.2, 0.2], [0.1, 0.1, 0.1], [0.3, 0.3, 0.3],
        ]),
        "deletion_centroid": torch.tensor([
            [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0],
        ]),
        "deletion_neighbor": torch.tensor([
            [1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0],
        ]),
        "shuffle_key": torch.tensor([
            [0.6, 0.5, 0.4], [0.1, 0.2, 0.3], [0.9, 0.8, 0.7],
            [0.3, 0.4, 0.5], [0.2, 0.1, 0.6], [0.7, 0.9, 0.8],
        ]),
    }


class StateTests(unittest.TestCase):
    def test_pace_is_fixed_monotone(self):
        self.assertEqual(pace_for_epoch(1, 5), 0.2)
        self.assertEqual(pace_for_epoch(5, 5), 1.0)

    def test_core_requires_both_interventions(self):
        state = positive_states(row(), "core", 1.0, 3)
        expected = torch.tensor([
            [CARRIER, ABSTAIN, ABSTAIN],
            [CARRIER, ABSTAIN, ABSTAIN],
            [ABSTAIN, ABSTAIN, ABSTAIN],
            [BACKGROUND, BACKGROUND, BACKGROUND],
            [BACKGROUND, BACKGROUND, BACKGROUND],
            [ABSTAIN, ABSTAIN, ABSTAIN],
        ])
        self.assertTrue(torch.equal(state, expected))

    def test_broadcast_and_negative_semantics_differ(self):
        broadcast = positive_states(row(), "broadcast", 1.0, 3)
        forced = positive_states(row(), "abstain_negative", 1.0, 3)
        self.assertTrue((broadcast[:2] == CARRIER).all())
        self.assertEqual(int((forced == CARRIER).sum()), 2)
        self.assertEqual(int((forced == BACKGROUND).sum()), 10)

    def test_selector_and_shuffle_preserve_carrier_rate(self):
        core = positive_states(row(), "core", 1.0, 3)
        for arm in ("branch_selector", "shuffled_carrier"):
            control = positive_states(row(), arm, 1.0, 3)
            self.assertTrue(torch.equal(
                (control == CARRIER).sum(0), (core == CARRIER).sum(0)
            ))

    def test_positive_nonpositive_background_is_separate_control(self):
        core = positive_states(row(), "core", 1.0, 3)
        risky = positive_states(row(), "nonpositive_background", 1.0, 3)
        self.assertEqual(int((core == BACKGROUND).sum()), 6)
        self.assertGreater(int((risky == BACKGROUND).sum()),
                           int((core == BACKGROUND).sum()))

    def test_neighbor_replacement_excludes_endpoint_self(self):
        rows = torch.tensor([[1.0], [3.0], [9.0]])
        replaced = neighbor_replacement(rows)
        self.assertTrue(torch.equal(
            replaced, torch.tensor([[3.0], [5.0], [3.0]])
        ))
        singleton = torch.tensor([[4.0]])
        self.assertTrue(torch.equal(neighbor_replacement(singleton), singleton))

    def test_batch_negative_bag_is_background_and_padding_abstains(self):
        cache = {"modalities": ("visual", "audio", "text"),
                 "rows": {"v": row(), "n": row()}}
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0],
                             [1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
        states = batch_states(
            ["v", "n"], torch.tensor([1.0, 0.0]), torch.tensor([6, 6]),
            mask, cache, "core", 5, 5, 3,
        )
        self.assertTrue((states[1, :6] == BACKGROUND).all())
        self.assertTrue((states[:, 6] == ABSTAIN).all())


class LossTests(unittest.TestCase):
    def test_supcon_is_finite_and_updates_embeddings(self):
        model = CarrierItS2CLR(
            {"visual": 4, "audio": 3, "text": 5}, "core",
            hidden=8, embed=6, dropout=0.0, max_instances=16,
        )
        embedding = torch.randn(8, 6, requires_grad=True)
        states = torch.tensor([0, 0, 0, 1, 1, 1, -1, -1])
        loss = model._one_supcon(embedding, states)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(embedding.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
