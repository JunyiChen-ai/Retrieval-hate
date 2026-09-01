#!/usr/bin/env python3
"""Mechanism tests for context quotient and span marginal."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from src.powa_residual import (  # noqa: E402
    ContextQuotientResidualHead,
    masked_center,
)
from train import normalized_span_marginal  # noqa: E402
from train import (  # noqa: E402
    centered_local_logit,
    residual_to_native,
    uniform_extract,
    uniform_model_grid,
)


def test_masked_center_is_invariant_to_video_constant():
    values = torch.randn(2, 8, 5)
    valid = torch.tensor([[1] * 8, [1] * 5 + [0] * 3], dtype=torch.bool)
    offset = torch.randn(2, 1, 5)
    left = masked_center(values, valid)
    right = masked_center(values + offset, valid)
    assert torch.allclose(left, right, atol=2e-6, rtol=0)
    assert torch.equal(right[~valid], torch.zeros_like(right[~valid]))


def test_residual_output_is_zero_mean_and_zero_initialized():
    head = ContextQuotientResidualHead()
    valid = torch.tensor([[1] * 7 + [0] * 2, [1] * 9], dtype=torch.bool)
    frozen = {
        "audio_rep": torch.randn(2, 9, 128),
        "visual_rep": torch.randn(2, 9, 128),
        "primitive_logits": torch.randn(2, 9, 6),
        "frame_prob": torch.rand(2, 9).clamp(.01, .99),
        "base_frame_logits": torch.randn(2, 9, 1),
    }
    output = head(frozen, torch.randn(2, 9, 768), valid)
    assert torch.equal(output, torch.zeros_like(output))
    for item, count in enumerate((7, 9)):
        assert abs(float(output[item, :count].mean())) <= 1e-7


def test_span_marginal_rewards_contiguous_values():
    contiguous = torch.tensor([-2., 2., 2., 2., -2., -2., -2.])
    scattered = torch.tensor([2., -2., 2., -2., 2., -2., -2.])
    core_contiguous = normalized_span_marginal(
        contiguous, "span_marginal", .5
    )
    core_scattered = normalized_span_marginal(scattered, "span_marginal", .5)
    singleton_contiguous = normalized_span_marginal(contiguous, "singleton", .5)
    singleton_scattered = normalized_span_marginal(scattered, "singleton", .5)
    assert core_contiguous > core_scattered
    assert torch.allclose(singleton_contiguous, singleton_scattered)


def test_shuffled_span_is_deterministic_and_value_preserving():
    values = torch.arange(12, dtype=torch.float32)
    one = normalized_span_marginal(
        values, "shuffled_span", .5,
        torch.Generator().manual_seed(234),
    )
    two = normalized_span_marginal(
        values, "shuffled_span", .5,
        torch.Generator().manual_seed(234),
    )
    assert torch.equal(one, two)


def test_singleton_has_nonzero_optimizer_signal_at_identity():
    head = ContextQuotientResidualHead(hidden=8)
    valid = torch.ones(1, 7, dtype=torch.bool)
    frozen = {
        "audio_rep": torch.randn(1, 7, 128),
        "visual_rep": torch.randn(1, 7, 128),
        "primitive_logits": torch.randn(1, 7, 6),
        "frame_prob": torch.tensor([[.05, .1, .2, .8, .7, .1, .05]]),
        "base_frame_logits": torch.randn(1, 7, 1),
    }
    residual = head(frozen, torch.randn(1, 7, 768), valid)[0]
    anchor = torch.logit(frozen["frame_prob"][0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        normalized_span_marginal(
            centered_local_logit(anchor, residual), "singleton", .5
        ),
        torch.ones(()),
    )
    loss.backward()
    assert head.output.weight.grad is not None
    assert float(head.output.weight.grad.abs().sum()) > 0


def test_uniform_grid_and_native_mapping_use_one_coordinate_system():
    length, width = 401, 200
    base = torch.arange(length, dtype=torch.float32)[None, :, None]
    v, a, t, count, index = uniform_model_grid(base, base, base, width)
    expected = uniform_extract(base[0].numpy(), width)[:, 0]
    assert count == width
    assert torch.equal(v[0, :, 0], torch.from_numpy(expected))
    assert torch.equal(v, a) and torch.equal(a, t)
    coarse = torch.sin(torch.linspace(0, 4, width))[None]
    native = residual_to_native(coarse, index, length)
    assert native.shape == (1, length)
    assert abs(float(native.mean())) < 1e-12


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
