from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import TemporalExpertChoice  # noqa: E402


def batch():
    feats = {"visual": torch.randn(2, 8, 4),
             "audio": torch.randn(2, 8, 3),
             "text": torch.randn(2, 8, 5)}
    mask = torch.tensor([[1] * 8, [1] * 5 + [0] * 3], dtype=torch.bool)
    return feats, mask


def test_expert_choice_exact_per_expert_capacity():
    feats, mask = batch()
    model = TemporalExpertChoice({"visual": 4, "audio": 3, "text": 5},
                                 "expert_choice", hidden=6, embed=2,
                                 dropout=0.0, k_proportion=4)
    out = model(feats, mask)
    expected = torch.tensor([[2, 2, 2], [2, 2, 2]])
    assert torch.equal(out["selected"].sum(-1), expected)
    assert not out["selected"][:, :, 5:][1].any()


def test_token_choice_matches_total_budget_and_at_most_one_per_time():
    feats, mask = batch()
    model = TemporalExpertChoice({"visual": 4, "audio": 3, "text": 5},
                                 "token_choice", hidden=6, embed=2,
                                 dropout=0.0, k_proportion=4)
    out = model(feats, mask)
    assert torch.equal(out["selected"].sum((1, 2)), torch.tensor([6, 6]))
    assert torch.equal(out["selected"].sum((1, 2)),
                       out["per_expert_count"] * 3)
    assert (out["selected"].sum(1) <= 2).all()
    assert not out["selected"][:, :, 5:][1].any()


def test_selected_evidence_is_load_bearing_for_final_score():
    feats, mask = batch()
    model = TemporalExpertChoice({"visual": 4, "audio": 3, "text": 5},
                                 "expert_choice", hidden=6, embed=2,
                                 dropout=0.0, k_proportion=4)
    out = model(feats, mask)
    loss = out["prob"].sum()
    loss.backward()
    assert model.experts["visual"].evidence.weight.grad.abs().sum() > 0
    assert model.experts["visual"].affinity.weight.grad.abs().sum() > 0
