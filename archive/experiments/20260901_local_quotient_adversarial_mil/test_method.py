#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts/reproduction_baselines/multihateloc"
sys.path[:0] = [str(BASE), str(ROOT)]

from model import MultiHateLoc  # noqa: E402
from method import LocalQuotientModel, gradient_reverse  # noqa: E402


def build():
    dims = {"visual": 5, "audio": 3, "text": 4}
    global_base = MultiHateLoc(dims, hidden=8, embed=6,
                               dropout=0.0, k_proportion=3)
    local_base = MultiHateLoc(dims, hidden=8, embed=6,
                              dropout=0.0, k_proportion=3)
    return LocalQuotientModel(global_base, local_base, embed=6, n_video_ids=3,
                              n_position_bins=4, local_scale=1.0)


def main():
    torch.manual_seed(7)
    model = build()
    feats = {"visual": torch.randn(2, 5, 5),
             "audio": torch.randn(2, 5, 3),
             "text": torch.randn(2, 5, 4)}
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    model.set_grl(.2, .3)
    out = model(feats, mask)
    assert torch.isfinite(out["probs"]["fused"]).all()
    for index in range(2):
        n = int(mask[index].sum())
        assert abs(float(out["local_logit"][index, :n].mean())) < 1e-6
    assert torch.all(out["local_logit"][1, 3:] == 0)

    x = torch.tensor([2.0], requires_grad=True)
    gradient_reverse(x, .25).sum().backward()
    assert torch.allclose(x.grad, torch.tensor([-.25]))

    model.zero_grad(set_to_none=True)
    loss = out["probs"]["fused"].sum()
    loss.backward()
    assert model.local_head.weight.grad is not None
    model.zero_grad(set_to_none=True)
    out = model(feats, mask)
    indices = torch.tensor([0, 1])
    lengths = mask.sum(1)
    video_loss, position_loss = model.nuisance_loss(out, mask, lengths, indices)
    (video_loss + position_loss).backward()
    assert all(parameter.grad is None or torch.all(parameter.grad == 0)
               for parameter in model.global_base.parameters())
    assert any(parameter.grad is not None and torch.any(parameter.grad != 0)
               for parameter in model.local_base.parameters())
    print("PASS: centering, masking, GRL sign, finite forward, final local path")


if __name__ == "__main__":
    main()
