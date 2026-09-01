#!/usr/bin/env python3
"""CPU tests for coalition algebra, masking, losses, and padded-grid handling."""

from __future__ import annotations

import math

import numpy as np
import torch

from model import CoalitionModel, MODALITIES, SUBSETS


def mobius_from_worth(worth):
    atoms = {}
    for subset in SUBSETS:
        value = worth[subset].clone()
        for proper, atom in atoms.items():
            if proper & subset == proper:
                value = value - atom
        atoms[subset] = value
    return atoms


def main():
    torch.manual_seed(7)
    dims = {name: 4 for name in MODALITIES}
    model = CoalitionModel(dims, hidden=8, embed=5, dropout=0.0)
    feats = {name: torch.randn(2, 6, 4) for name in MODALITIES}
    lengths = torch.tensor([6, 4])
    mask = torch.arange(6)[None, :] < lengths[:, None]

    coalition = model.coalition_logits(feats)
    assert coalition.shape == (2, 6, 7)
    intensity = torch.exp(coalition / model.temperature)
    worth = {}
    for subset in SUBSETS:
        members = [
            index for index, atom_subset in enumerate(SUBSETS)
            if atom_subset & subset == atom_subset
        ]
        worth[subset] = intensity[:, :, members].sum(-1)
    recovered = mobius_from_worth(worth)
    for index, subset in enumerate(SUBSETS):
        assert torch.allclose(recovered[subset], intensity[:, :, index], atol=1e-5)
    full = model.reconstructed_full_logits(coalition)
    expected = model.temperature * (torch.log(worth[7]) - math.log(7))
    assert torch.allclose(full, expected, atol=1e-6)

    for arm in ("all_subset_mil", "synib", "mobius_nonminimal", "coalition_witness"):
        output = model(feats, mask, lengths, arm)
        labels = torch.tensor([1.0, 0.0])
        loss, terms = model.loss(output, labels, mask, lengths, arm)
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value) for value in terms.values())
        model.zero_grad(set_to_none=True)
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())
        assert output["frame_scores"][1, 4:].eq(0).all()

    model.eval()
    with torch.no_grad():
        base = model(feats, mask, lengths, "coalition_witness")["frame_scores"][1, :4]
        changed = {name: value.clone() for name, value in feats.items()}
        for value in changed.values():
            value[1, 4:] = 1e6
        after = model(changed, mask, lengths, "coalition_witness")["frame_scores"][1, :4]
    assert torch.allclose(base, after)
    print("coalition witness model tests: PASS")


if __name__ == "__main__":
    main()
