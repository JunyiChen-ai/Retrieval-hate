from __future__ import annotations

import math

import torch
import torch.nn.functional as F

ARMS = ("fixed_topk_control", "sparse_scan")


def _masked_center(logits, mask):
    weight = mask.to(logits.dtype)
    mean = (logits * weight).sum(1) / weight.sum(1).clamp(min=1.0)
    return (logits - mean[:, None]) * weight


def update_null_scale(centered, mask, labels, previous, momentum):
    negative = labels < .5
    valid = mask & negative[:, None]
    if valid.any():
        batch_scale = centered[valid].pow(2).mean().sqrt().clamp(min=.05)
        updated = momentum * previous + (1.0 - momentum) * batch_scale.detach()
        return updated.detach()
    return previous.detach()


def _rank_grid(length, max_fraction, n_ranks, device):
    maximum = max(1, min(length // 2, int(math.ceil(length * max_fraction))))
    if maximum == 1:
        return torch.ones(1, dtype=torch.long, device=device)
    grid = torch.exp(torch.linspace(0.0, math.log(float(maximum)),
                                    steps=n_ranks, device=device))
    return torch.unique(grid.round().long().clamp(1, maximum), sorted=True)


def sparse_scan_evidence(logits, mask, null_scale, max_fraction=.5,
                         n_ranks=8, scan_temperature=1.0):
    """Per-bag weighted average likelihood-ratio evidence."""
    centered = _masked_center(logits, mask)
    scale = null_scale.detach().clamp(min=.05)
    z = centered / scale
    pvalues = (.5 * torch.erfc(z / math.sqrt(2.0))).clamp(1e-6, 1 - 1e-6)
    evidence = []
    for index in range(logits.shape[0]):
        length = int(mask[index].sum())
        ordered = torch.sort(pvalues[index, :length]).values
        ranks = _rank_grid(length, max_fraction, n_ranks, logits.device)
        p = ordered[ranks - 1]
        q = ranks.to(logits.dtype) / float(length)
        one_sided = p < q
        log_lr = (ranks.to(logits.dtype) * torch.log(q / p) +
                  (length - ranks).to(logits.dtype) *
                  torch.log((1.0 - q).clamp(min=1e-6) /
                            (1.0 - p).clamp(min=1e-6)))
        log_lr = torch.where(one_sided, log_lr, torch.zeros_like(log_lr))
        log_weight = -torch.log(ranks.to(logits.dtype))
        log_weight = log_weight - torch.logsumexp(log_weight, dim=0)
        temperature = max(float(scan_temperature), 1e-3)
        log_alr = temperature * torch.logsumexp(
            log_lr / temperature + log_weight, dim=0)
        evidence.append(log_alr / float(length))
    return torch.stack(evidence), centered


def scan_ranking_loss(evidence, labels, margin):
    positive = evidence[labels >= .5]
    negative = evidence[labels < .5]
    if len(positive) == 0 or len(negative) == 0:
        return evidence.sum() * 0.0
    differences = positive[:, None] - negative[None, :]
    return F.softplus(margin - differences).mean()


def dense_negative_loss(prob, mask, labels):
    negative = labels < .5
    valid = mask & negative[:, None]
    if not valid.any():
        return prob.sum() * 0.0
    return F.binary_cross_entropy(prob[valid], torch.zeros_like(prob[valid]))
