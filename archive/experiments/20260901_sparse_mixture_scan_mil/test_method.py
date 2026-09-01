from __future__ import annotations

import torch

from method import (scan_ranking_loss, sparse_scan_evidence,
                    update_null_scale)


def test_scan_is_invariant_to_video_broadcast_offset():
    logits = torch.tensor([[0., 0., 2., 0., 0.], [1., 1., 3., 1., 1.]])
    mask = torch.ones_like(logits, dtype=torch.bool)
    evidence, _ = sparse_scan_evidence(logits, mask, torch.tensor(1.0))
    assert torch.allclose(evidence[0], evidence[1], atol=1e-6)


def test_sparse_tail_has_more_evidence_than_flat_bag():
    logits = torch.tensor([[0., 0., 3., 0., 0.], [0., 0., 0., 0., 0.]])
    mask = torch.ones_like(logits, dtype=torch.bool)
    evidence, _ = sparse_scan_evidence(logits, mask, torch.tensor(1.0))
    assert evidence[0] > evidence[1]


def test_scan_ranking_gradient_is_finite():
    logits = torch.tensor([[0., 0., 2., 0.], [0., .1, 0., -.1]],
                          requires_grad=True)
    mask = torch.ones_like(logits, dtype=torch.bool)
    evidence, _ = sparse_scan_evidence(logits, mask, torch.tensor(1.0))
    loss = scan_ranking_loss(evidence, torch.tensor([1., 0.]), .25)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_null_scale_uses_negative_bags_only():
    centered = torch.tensor([[100., -100.], [2., -2.]])
    mask = torch.ones_like(centered, dtype=torch.bool)
    updated = update_null_scale(centered, mask, torch.tensor([1., 0.]),
                                torch.tensor(1.0), 0.0)
    assert torch.allclose(updated, torch.tensor(2.0))
