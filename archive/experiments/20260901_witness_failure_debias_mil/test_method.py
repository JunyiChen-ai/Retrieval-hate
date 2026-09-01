from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts/reproduction_baselines/multihateloc"
sys.path[:0] = [str(BASE), str(ROOT)]

from model import MultiHateLoc  # noqa: E402
from method import generalized_cross_entropy, training_mil_loss, witness_failure_loss  # noqa: E402


def main():
    torch.manual_seed(4)
    model = MultiHateLoc({"visual": 4, "audio": 3, "text": 5},
                         hidden=8, embed=6, dropout=0.0, k_proportion=2)
    feats = {"visual": torch.randn(3, 6, 4),
             "audio": torch.randn(3, 6, 3),
             "text": torch.randn(3, 6, 5)}
    mask = torch.tensor([[1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 0, 0],
                         [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    lengths = mask.sum(1)
    labels = torch.tensor([1.0, 0.0, 1.0])
    output = model(feats, mask)
    original, _ = model.mil_loss(output["probs"], mask, lengths, labels)
    anchor, _ = training_mil_loss(
        model, output, mask, lengths, labels, "anchor")
    torch.testing.assert_close(anchor, original, rtol=0, atol=0)
    zero, diag = witness_failure_loss(
        model, output, mask, lengths, labels, "anchor")
    torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)
    assert diag["positive_videos"] == 0
    uniform, uniform_diag = witness_failure_loss(
        model, output, mask, lengths, labels, "uniform")
    relative, relative_diag = witness_failure_loss(
        model, output, mask, lengths, labels, "relative")
    assert torch.isfinite(uniform) and torch.isfinite(relative)
    assert uniform_diag["positive_videos"] == 2
    assert uniform_diag["negative_videos"] == 1
    assert relative_diag["positive_videos"] == 2
    (relative + training_mil_loss(
        model, output, mask, lengths, labels, "relative")[0]).backward()
    assert model.fuse_head.weight.grad is not None
    assert torch.isfinite(model.fuse_head.weight.grad).all()
    p = torch.tensor([.2, .8])
    y = torch.tensor([0.0, 1.0])
    expected = (1.0 - torch.tensor(.8).pow(.7)) / .7
    torch.testing.assert_close(generalized_cross_entropy(p, y), expected)
    print("PASS")


if __name__ == "__main__":
    main()
