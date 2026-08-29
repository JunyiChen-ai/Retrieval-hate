"""Unified video-prior plus zero-mean temporal relation correction."""
from __future__ import annotations

import torch
from relation_v4.model import ExpertRelationGate


def masked_mean(value, valid):
    weight = valid.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True).clamp_min(1)


class UnifiedRelationV8(torch.nn.Module):
    """Decompose calibrated evidence into a video prior and temporal locator.

    The prior is the immutable equal-expert consensus averaged over the video.
    All temporal terms are centered per video.  Consequently transport cannot
    change cross-video prior, and beta=gamma=0 is exactly the static prior.
    """

    def __init__(self, n_experts, window=12, temperature=.2):
        super().__init__()
        if n_experts < 2:
            raise ValueError("Relation-V8 requires at least two evidence streams")
        self.n_experts = int(n_experts)
        self.window = int(window)
        self.temperature = float(temperature)
        weights = [1. / n_experts] * n_experts
        self.transport_core = ExpertRelationGate(n_experts, weights, hidden=8,
                                                  window=window, temperature=temperature)

    def forward(self, calibrated, valid, beta=0., gamma=0.):
        if calibrated.ndim != 3 or calibrated.shape[-1] != self.n_experts:
            raise ValueError("expected [batch,time,evidence] calibrated scores")
        if valid.shape != calibrated.shape[:2]:
            raise ValueError("valid-mask shape mismatch")
        valid3 = valid[..., None]
        evidence = calibrated * valid3
        consensus = evidence.mean(-1)
        video_prior = masked_mean(consensus, valid).squeeze(1)
        static_locator = (consensus - video_prior[:, None]) * valid

        centered_evidence = (evidence - masked_mean(evidence, valid)) * valid3
        aligned, lag, transport = self.transport_core.transport(centered_evidence, valid)
        # Symmetric equal consensus over target/source relations.  Centering is
        # repeated after masking so numerical drift cannot leak into the prior.
        transported = aligned.mean((-1, -2)) * valid
        transported = (transported - masked_mean(transported, valid)) * valid
        relation_residual = (transported - static_locator) * valid
        relation_residual = (relation_residual - masked_mean(relation_residual, valid)) * valid

        locator = (float(beta) * static_locator + float(gamma) * relation_residual) * valid
        locator = (locator - masked_mean(locator, valid)) * valid
        correction = locator
        score = (video_prior[:, None] + correction) * valid
        return {
            "frame_score": score,
            "video_prior": video_prior,
            "static_prior": video_prior[:, None].expand_as(score) * valid,
            "static_locator": static_locator,
            "transported_locator": transported,
            "relation_residual": relation_residual,
            "locator_correction": locator,
            "correction": correction,
            "centered_evidence": centered_evidence,
            "expected_lag": lag,
            "transport": transport,
        }

    def forward_ablation(self, calibrated, valid, beta=0., gamma=0., mode="full"):
        """Four preregistered decomposition arms used by the ablation runner.

        ``uncentered`` deliberately violates the identifiability constraint: it
        lets temporal evidence retain its video-wide mean.  It is a negative
        ablation, not an alternative proposed method.
        """
        if mode not in ("prior_only", "locator_only", "uncentered", "full"):
            raise ValueError("unknown ablation mode")
        out = self.forward(calibrated, valid, beta, gamma)
        if mode == "prior_only":
            out["frame_score"] = out["static_prior"]
            return out
        if mode == "locator_only":
            out["frame_score"] = out["locator_correction"] * valid
            return out
        if mode == "full":
            return out

        valid3 = valid[..., None]
        evidence = calibrated * valid3
        consensus = evidence.mean(-1)
        video_prior = masked_mean(consensus, valid).squeeze(1)
        aligned, lag, transport = self.transport_core.transport(evidence, valid)
        transported = aligned.mean((-1, -2)) * valid
        relation_residual = (transported - consensus) * valid
        locator = (float(beta) * consensus + float(gamma) * relation_residual) * valid
        out.update({
            "frame_score": (video_prior[:, None] + locator) * valid,
            "static_locator": consensus * valid,
            "transported_locator": transported,
            "relation_residual": relation_residual,
            "locator_correction": locator,
            "correction": locator,
            "centered_evidence": evidence,
            "expected_lag": lag,
            "transport": transport,
        })
        return out
