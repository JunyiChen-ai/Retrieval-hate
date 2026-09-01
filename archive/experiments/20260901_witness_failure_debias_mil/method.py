from __future__ import annotations

import torch
import torch.nn.functional as F

from model import topk_counts, topk_mask, topk_mean


ARMS = ("anchor", "uniform", "relative")
EPS = 1e-7


def generalized_cross_entropy(probability, labels, q=0.7):
    """Binary GCE from LfF, evaluated on one probability per video."""
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1]")
    probability = probability.clamp(EPS, 1.0 - EPS)
    correct = torch.where(labels >= 0.5, probability, 1.0 - probability)
    return ((1.0 - correct.pow(q)) / q).mean()


def training_mil_loss(model, output, mask, lengths, labels, arm, q=0.7):
    """Exact anchor MIL or fused BCE plus GCE shortcut experts."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if arm == "anchor":
        return model.mil_loss(output["probs"], mask, lengths, labels)
    counts = topk_counts(lengths, model.k_proportion)
    per_branch = {}
    fused_video = topk_mean(output["probs"]["fused"], mask, counts).clamp(
        EPS, 1.0 - EPS)
    per_branch["fused"] = F.binary_cross_entropy(fused_video, labels)
    total = per_branch["fused"]
    for modality in model.modalities:
        video_probability = topk_mean(
            output["probs"][modality], mask, counts)
        loss = generalized_cross_entropy(video_probability, labels, q=q)
        per_branch[modality] = loss
        total = total + loss
    return total, per_branch


def _per_video_failure_loss(core_loss, bias_loss, support, relative):
    weights = torch.ones_like(core_loss)
    if relative:
        weights = (bias_loss.detach() /
                   (bias_loss.detach() + core_loss.detach()).clamp(min=EPS))
    values, weight_values = [], []
    for index in range(core_loss.shape[0]):
        selected = support[index]
        if not bool(selected.any()):
            continue
        selected_weights = weights[index, selected]
        selected_weights = selected_weights / selected_weights.mean().clamp(
            min=EPS)
        values.append((core_loss[index, selected] * selected_weights).mean())
        weight_values.append(selected_weights.detach().mean())
    if not values:
        zero = core_loss.sum() * 0.0
        return zero, zero, 0
    return torch.stack(values).mean(), torch.stack(weight_values).mean(), len(values)


def witness_failure_loss(model, output, mask, lengths, labels, arm):
    """LfF relative difficulty restricted to valid latent witness support."""
    zero = output["probs"]["fused"].sum() * 0.0
    if arm == "anchor":
        return zero, {"positive_videos": 0, "negative_videos": 0,
                      "positive_weight_mean": 0.0,
                      "negative_weight_mean": 0.0}
    relative = arm == "relative"
    counts = topk_counts(lengths, model.k_proportion)
    fused = output["probs"]["fused"].clamp(EPS, 1.0 - EPS)
    branches = torch.stack(
        [output["probs"][name].detach().clamp(EPS, 1.0 - EPS)
         for name in model.modalities], dim=1)

    positive_support = topk_mask(fused.detach(), mask, counts)
    positive_support = positive_support & (labels >= 0.5)[:, None]
    positive_core = -torch.log(fused)
    # The easiest positive shortcut is the branch with the highest p(y=1).
    positive_bias = -torch.log(branches.max(dim=1).values)
    positive, positive_weight, n_positive = _per_video_failure_loss(
        positive_core, positive_bias, positive_support, relative)

    negative_support = mask & (labels < 0.5)[:, None]
    negative_core = -torch.log1p(-fused)
    # The easiest negative shortcut is the branch with the lowest p(y=1).
    negative_bias = -torch.log1p(-branches.min(dim=1).values)
    negative, negative_weight, n_negative = _per_video_failure_loss(
        negative_core, negative_bias, negative_support, relative)

    terms = []
    if n_positive:
        terms.append(positive)
    if n_negative:
        terms.append(negative)
    loss = torch.stack(terms).mean() if terms else zero
    diagnostics = {
        "positive_videos": n_positive,
        "negative_videos": n_negative,
        "positive_weight_mean": positive_weight,
        "negative_weight_mean": negative_weight,
    }
    return loss, diagnostics
