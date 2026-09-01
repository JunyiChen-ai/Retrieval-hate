"""Witness-conditional Dynamic Gradient Modulation utilities."""
from __future__ import annotations

import math

import torch


ARMS = ("anchor", "source_dgm", "witness_dgm")


@torch.no_grad()
def _topk_and_rest(prob, mask, counts):
    """Return per-video top-k mean and valid non-top-k mean."""
    filled = prob.masked_fill(~mask, -1.0)
    order = torch.argsort(filled, dim=1, descending=True)
    ranks = torch.arange(prob.shape[1], device=prob.device)[None, :]
    keep_rank = ranks < counts[:, None]
    keep = torch.zeros_like(mask).scatter(1, order, keep_rank) & mask
    top = (prob * keep).sum(1) / keep.sum(1).clamp(min=1).to(prob.dtype)
    rest = mask & ~keep
    rest_count = rest.sum(1)
    rest_mean = (prob * rest).sum(1) / rest_count.clamp(min=1).to(prob.dtype)
    # A one-frame/all-top-k video has no within-video contrast.
    rest_mean = torch.where(rest_count > 0, rest_mean, top)
    return top, rest_mean


@torch.no_grad()
def competence(probs, mask, lengths, labels, modalities, arm,
               k_proportion=3):
    """Compute one stop-gradient competence scalar per modality.

    source_dgm uses correct video-label confidence. witness_dgm uses positive
    top-k-vs-rest contrast and negative worst-top-k suppression.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if arm == "anchor":
        return torch.ones(len(modalities), device=labels.device)
    counts = torch.clamp(
        torch.ceil(lengths.float() / float(k_proportion)), min=1).long()
    positive = labels > 0.5
    negative = ~positive
    values = []
    for name in modalities:
        top, rest = _topk_and_rest(probs[name].detach(), mask, counts)
        if arm == "source_dgm":
            correct = torch.where(positive, top, 1.0 - top)
            value = correct.mean()
        else:
            parts = []
            if positive.any():
                parts.append((top[positive] - rest[positive]).mean())
            if negative.any():
                parts.append((1.0 - top[negative]).mean())
            value = torch.stack(parts).mean() if parts else top.new_tensor(0.0)
        values.append(value)
    return torch.stack(values).clamp(min=0.0)


@torch.no_grad()
def modulation_coefficients(values, arm, gamma=0.1, eps=1e-6):
    """Three-modality extension of source DGM Eq. 7."""
    if arm == "anchor":
        return torch.ones_like(values)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("competence must be a vector with at least two entries")
    others = (values.sum() - values) / float(len(values) - 1)
    ratio = values / others.clamp(min=eps)
    reduced = 1.0 - torch.tanh(float(gamma) * ratio)
    coeff = torch.where(ratio > 1.0, reduced, torch.ones_like(reduced))
    return coeff.clamp(min=0.0, max=1.0)


@torch.no_grad()
def apply_gradient_modulation(model, coeff):
    """Scale branch and corresponding fused-input gradients in place."""
    modalities = tuple(model.modalities)
    if tuple(coeff.shape) != (len(modalities),):
        raise ValueError("coefficient shape does not match modalities")
    for index, name in enumerate(modalities):
        scale = float(coeff[index])
        if not math.isfinite(scale):
            raise ValueError("non-finite modulation coefficient")
        for parameter in model.branches[name].parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)

    # The first fused linear layer has one contiguous input slice per modality.
    first = model.fuse[0]
    width = first.in_features // len(modalities)
    if first.in_features != width * len(modalities):
        raise ValueError("fused input is not evenly partitioned by modality")
    if first.weight.grad is not None:
        for index in range(len(modalities)):
            first.weight.grad[:, index * width:(index + 1) * width].mul_(
                float(coeff[index]))

