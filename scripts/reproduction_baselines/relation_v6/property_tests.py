#!/usr/bin/env python3
"""Formal and adversarial property checks for the Relation-V6 mechanism.

Guaranteed properties are assertions.  Desirable but currently unsupported
properties are measured and reported as ``supported: false`` rather than
silently promoted to claims.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from relation_v6.data import calibrate, fit_calibration
from relation_v6.model import DistributionPrior, RelationV6


def max_abs(x: torch.Tensor) -> float:
    return float(x.detach().abs().max())


def main() -> None:
    torch.manual_seed(20260829)
    scores = torch.rand(2, 17, 3)
    valid = torch.tensor([[1] * 17, [1] * 11 + [0] * 6], dtype=torch.bool)
    model = RelationV6(3, hidden=16, heads=4, window=3, dropout=0).eval()
    base = model(scores * valid[..., None], valid)

    # P1: the localization branch cannot encode a video-wide intercept.
    zero_mean = (base["locator_logit"] * valid).sum(1)
    assert torch.allclose(zero_mean, torch.zeros_like(zero_mean), atol=2e-6)

    # P2: scale=0 is an exact prior-only fallback, constant on valid frames.
    fallback = model(scores * valid[..., None], valid, locator_scale=0)
    expected = fallback["prior_logit"][:, None].expand_as(fallback["frame_logit"])
    assert torch.allclose(fallback["frame_logit"][valid], expected[valid], atol=1e-7)

    # P3: per-video/per-expert constants cannot enter the locator.  They are
    # intentionally allowed to affect the video prior.
    offsets = torch.tensor([[[3.0, -2.0, .7]], [[.4, 8.0, -5.0]]])
    shifted = (scores + offsets) * valid[..., None]
    moved = model(shifted, valid)
    locator_constant_error = max_abs(base["locator_logit"] - moved["locator_logit"])
    assert locator_constant_error < 2e-6

    # P4: frozen ECDF calibration is invariant to a jointly applied strictly
    # increasing score reparameterisation.  No test-distribution statistic is
    # used here: both calibrators are fit on the synthetic train cohort.
    rng = np.random.default_rng(20260829)
    train = {str(i): rng.normal(size=(13 + i, 3)).astype(np.float32)
             for i in range(5)}
    query = rng.normal(size=(31, 3)).astype(np.float32)
    cal = fit_calibration(train)
    transformed_train = {k: np.exp(v).astype(np.float32) for k, v in train.items()}
    transformed_cal = fit_calibration(transformed_train)
    ecdf_error = float(np.max(np.abs(
        calibrate(query, cal) - calibrate(np.exp(query), transformed_cal))))
    assert ecdf_error < 1e-7

    # A1/A2: current set aggregation is not duplicate- or noise-invariant.
    # Measure this explicitly.  A future reliability-normalised aggregator may
    # turn these into assertions, but the current paper must not claim them.
    prior = DistributionPrior(hidden=16, heads=4, dropout=0).eval()
    tokens = base["distribution_tokens"][:1]
    prior_base = prior(tokens)
    duplicate_delta = max_abs(prior(torch.cat([tokens, tokens[:, :1]], 1)) - prior_base)
    noise = torch.randn_like(tokens[:, :1]) * 4
    noise_delta = max_abs(prior(torch.cat([tokens, noise], 1)) - prior_base)

    report = {
        "guaranteed": {
            "locator_exact_zero_mean": {"supported": True,
                                           "max_abs_sum": max_abs(zero_mean)},
            "prior_only_scale_zero_fallback": {"supported": True},
            "locator_per_expert_constant_invariance": {
                "supported": True, "max_abs_error": locator_constant_error},
            "train_frozen_ecdf_joint_monotone_invariance": {
                "supported": True, "max_abs_error": ecdf_error},
        },
        "unsupported": {
            "duplicate_expert_invariance": {
                "supported": duplicate_delta < 1e-7,
                "observed_abs_change": duplicate_delta,
            },
            "noise_expert_invariance": {
                "supported": noise_delta < 1e-7,
                "observed_abs_change": noise_delta,
            },
            "arbitrary_static_expert_fusion_fallback": {
                "supported": False,
                "note": "scale=0 is prior-only, not an arbitrary fixed expert mixture",
            },
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
