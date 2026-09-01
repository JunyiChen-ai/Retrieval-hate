from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts" / "reproduction_baselines" / "multihateloc"
EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(BASE))

from model import MultiHateLoc  # noqa: E402
from method import (apply_gradient_modulation, competence,
                    modulation_coefficients)  # noqa: E402


def test_witness_competence_uses_positive_contrast_and_negative_false_alarm():
    mask = torch.ones(2, 4, dtype=torch.bool)
    lengths = torch.tensor([4, 4])
    labels = torch.tensor([1.0, 0.0])
    probs = {
        "visual": torch.tensor([[.9, .8, .1, .1], [.1, .1, .1, .1]]),
        "audio": torch.tensor([[.6, .6, .5, .5], [.7, .2, .1, .1]]),
        "text": torch.tensor([[.5, .5, .5, .5], [.5, .5, .5, .5]]),
    }
    value = competence(probs, mask, lengths, labels,
                       ("visual", "audio", "text"), "witness_dgm", 2)
    assert value[0] > value[1] > value[2]


def test_only_above_average_competence_is_attenuated():
    value = torch.tensor([.8, .2, .2])
    coeff = modulation_coefficients(value, "witness_dgm", gamma=.1)
    assert 0.0 < coeff[0] < 1.0
    assert coeff[1] == 1.0 and coeff[2] == 1.0


def test_gradient_modulation_scales_branch_and_fused_slice():
    model = MultiHateLoc({"visual": 4, "audio": 3, "text": 5},
                         hidden=6, embed=2, dropout=0.0)
    feats = {"visual": torch.randn(2, 4, 4),
             "audio": torch.randn(2, 4, 3),
             "text": torch.randn(2, 4, 5)}
    mask = torch.ones(2, 4, dtype=torch.bool)
    output = model(feats, mask)
    sum(value.sum() for value in output["probs"].values()).backward()
    branch_before = model.branches["visual"].head.weight.grad.clone()
    fuse_before = model.fuse[0].weight.grad.clone()
    apply_gradient_modulation(model, torch.tensor([.5, 1.0, 1.0]))
    assert torch.allclose(model.branches["visual"].head.weight.grad,
                          branch_before * .5)
    assert torch.allclose(model.fuse[0].weight.grad[:, :2],
                          fuse_before[:, :2] * .5)
    assert torch.allclose(model.fuse[0].weight.grad[:, 2:],
                          fuse_before[:, 2:])
