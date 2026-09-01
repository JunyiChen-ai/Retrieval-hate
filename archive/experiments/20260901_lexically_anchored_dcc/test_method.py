#!/usr/bin/env python
"""Non-training unit checks for region admission, memory and gradients."""
from __future__ import annotations

import torch

from method import CrossVideoRegionMemory


def main():
    torch.manual_seed(7)
    shared = torch.randn(4, 6, 8, requires_grad=True)
    evidence = torch.tensor([
        [0., 1., 4., 5., 0., 0.],
        [0., 2., 3., 6., 1., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
    ])
    speech = torch.tensor([
        [0., 1., 1., 1., 0., 0.],
        [0., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 0.],
    ])
    valid = torch.tensor([[1, 1, 1, 1, 1, 0]] * 4, dtype=torch.bool)
    labels = torch.tensor([1., 1., 0., 0.])
    video_ids = torch.arange(4)
    memory = CrossVideoRegionMemory(8, capacity=32, temperature=.2,
                                    negative_width=2)
    loss, stats = memory(shared, evidence, speech, valid, labels, video_ids, .5)
    assert torch.isfinite(loss) and float(loss) > 0
    assert stats["hate_regions"] >= 2
    assert stats["benign_regions"] == 6
    assert stats["supported_frames"] > 0
    loss.backward()
    assert shared.grad is not None and float(shared.grad.abs().sum()) > 0
    # Positive-unselected frames are abstained: benign admission comes only
    # from the two negative bags (5 valid frames each, chunked 2/2/1).
    assert stats["benign_memory"] == 6
    # Memory buffers are training-only and intentionally absent from state.
    assert not any("memory" in key or "count" in key
                   for key in memory.state_dict())
    print("PASS: asymmetric admission, cross-video loss and shared gradient")


if __name__ == "__main__":
    main()
