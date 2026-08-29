#!/usr/bin/env python3
"""V8-specific formal properties, identifiability, and negative controls."""
import json
import numpy as np
import torch

from relation_v4.io import apply_ecdf, fit_ecdf
from relation_v8.model import UnifiedRelationV8


def maximum(x): return float(x.detach().abs().max())


def main():
    torch.manual_seed(8)
    valid = torch.tensor([[1] * 13, [1] * 9 + [0] * 4], dtype=torch.bool)
    score = torch.rand(2, 13, 3) * valid[..., None]
    model = UnifiedRelationV8(3, window=3).eval()
    out = model(score, valid, beta=.7, gamma=-.2)

    zero = {}
    for key in ("static_locator", "transported_locator", "relation_residual",
                "locator_correction"):
        sums = (out[key] * valid).sum(1)
        zero[key] = maximum(sums)
        assert zero[key] < 2e-6

    fallback = model(score, valid, beta=0., gamma=0.)
    fallback_error = maximum(fallback["frame_score"] - fallback["static_prior"])
    assert fallback_error == 0
    for mode in ("prior_only", "locator_only", "uncentered", "full"):
        arm = model.forward_ablation(score, valid, beta=.7, gamma=-.2, mode=mode)
        assert arm["frame_score"].shape == valid.shape
        assert torch.isfinite(arm["frame_score"]).all()
        assert torch.equal(arm["frame_score"][~valid],
                           torch.zeros_like(arm["frame_score"][~valid]))

    offset = torch.tensor([[[.3, -.2, .8]], [[-.7, .4, .1]]])
    shifted = model((score + offset) * valid[..., None], valid, beta=.7, gamma=-.2)
    layer_errors = {k: maximum(out[k] - shifted[k]) for k in
                    ("static_locator", "relation_residual", "locator_correction")}
    assert max(layer_errors.values()) < 3e-6
    # The prior is intentionally *not* shift invariant: its exact shift is the
    # equal-expert mean of the per-video offsets.
    expected_prior_shift = offset.mean(-1).squeeze(1)
    prior_shift_error = maximum((shifted["video_prior"] - out["video_prior"])
                                - expected_prior_shift)
    assert prior_shift_error < 2e-6

    # Synthetic identifiable decomposition: zero-mean local evidence cannot
    # change the recovered video prior.
    time = torch.linspace(-1, 1, 13)
    local = torch.stack([time, -time, .5 * time], -1)
    local = local - local.mean(0, keepdim=True)
    prior = torch.tensor([.2, .75])
    local_batch = local[None].expand(2, -1, -1).clone()
    local_batch = (local_batch - (local_batch * valid[..., None]).sum(1, keepdim=True)
                   / valid.sum(1)[:, None, None]) * valid[..., None]
    synthetic = (prior[:, None, None] + local_batch) * valid[..., None]
    syn = model(synthetic, valid, beta=1., gamma=0.)
    prior_recovery = maximum(syn["video_prior"] - prior)
    assert prior_recovery < 2e-6

    # Strict monotone invariance requires applying the same transform to the
    # reference cohort and query.  A frozen reference is not invariant when
    # only the query scale changes; this is the real deployment boundary.
    rng = np.random.default_rng(8)
    ref = {"a": rng.normal(size=(31, 3)).astype("float32")}
    query = {"q": rng.normal(size=(17, 3)).astype("float32")}
    base = apply_ecdf(query, fit_ecdf(ref))["q"]
    joint = apply_ecdf({"q": np.exp(query["q"])},
                       fit_ecdf({"a": np.exp(ref["a"])}))["q"]
    joint_error = float(np.max(np.abs(base - joint)))
    assert joint_error == 0
    frozen_reference_error = float(np.max(np.abs(
        base - apply_ecdf({"q": np.exp(query["q"])}, fit_ecdf(ref))["q"])))
    assert frozen_reference_error > 0

    # Negative controls: equal consensus is not invariant to adding a duplicate
    # or arbitrary noise expert.  These are measured, never claimed.
    original_prior = score.mean(-1)
    duplicate_prior = torch.cat([score, score[..., :1]], -1).mean(-1)
    noise_prior = torch.cat([score, torch.rand_like(score[..., :1])], -1).mean(-1)
    report = {
        "guaranteed": {
            "exact_zero_mean": zero,
            "beta_gamma_zero_exact_fallback_error": fallback_error,
            "constant_shift_invariant_layers": layer_errors,
            "prior_shift_equivariance_error": prior_shift_error,
            "synthetic_prior_recovery_error": prior_recovery,
            "joint_reference_query_strict_monotone_ecdf_error": joint_error,
        },
        "boundary": {"query_only_transform_with_frozen_reference_error":
                     frozen_reference_error},
        "negative_controls": {
            "duplicate_expert_changes_equal_consensus": maximum(
                (original_prior - duplicate_prior) * valid),
            "noise_expert_changes_equal_consensus": maximum(
                (original_prior - noise_prior) * valid),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
