from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModalityBranch, masked_mean, topk_counts, topk_mean


class TemporalResidualModel(nn.Module):
    """Three local modality learners with one additive final frame logit."""

    def __init__(self, dims, hidden, embed, dropout, k_proportion, temperature):
        super().__init__()
        self.modalities = tuple(dims)
        self.k_proportion = int(k_proportion)
        self.temperature = float(temperature)
        self.branches = nn.ModuleDict({
            name: ModalityBranch(dim, hidden, embed, dropout)
            for name, dim in dims.items()
        })

    def forward(self, feats, mask):
        embeds, logits, probs = {}, {}, {}
        for name in self.modalities:
            embed, logit = self.branches[name](feats[name])
            embeds[name] = embed
            logits[name] = logit
            probs[name] = torch.sigmoid(logit) * mask
        fused_logit = torch.stack(list(logits.values()), dim=0).sum(0)
        probs["fused"] = torch.sigmoid(fused_logit) * mask
        return {"embeds": embeds, "logits": logits,
                "fused_logit": fused_logit, "probs": probs}

    def bag_losses(self, probs, mask, lengths, labels):
        counts = topk_counts(lengths, self.k_proportion)
        losses = {}
        for name, prob in probs.items():
            bag = topk_mean(prob, mask, counts).clamp(1e-7, 1 - 1e-7)
            losses[name] = F.binary_cross_entropy(bag, labels)
        branch = torch.stack([losses[m] for m in self.modalities]).mean()
        return losses["fused"] + .25 * branch, losses

    def temporal_residual_loss(self, output, mask, lengths, labels, active):
        """Fit the exact negative bag-BCE functional gradient of frozen peers."""
        counts = topk_counts(lengths, self.k_proportion)
        base = torch.stack([
            value.detach() for name, value in output["logits"].items()
            if name != active
        ], dim=0).sum(0).detach().requires_grad_(True)
        base_prob = torch.sigmoid(base) * mask
        base_bag = topk_mean(base_prob, mask, counts).clamp(1e-7, 1 - 1e-7)
        base_bce = F.binary_cross_entropy(base_bag, labels)
        residual = -torch.autograd.grad(base_bce, base, create_graph=False)[0].detach()
        return -(residual * output["logits"][active] * mask).sum()

    def smoothness_loss(self, probs, mask):
        pair = mask[:, 1:] & mask[:, :-1]
        denom = pair.sum().clamp(min=1).to(probs["fused"].dtype)
        return ((((probs["fused"][:, 1:] - probs["fused"][:, :-1]) ** 2)
                 * pair).sum() / denom)

    def contrastive_loss(self, embeds, mask):
        pooled = [F.normalize(masked_mean(embeds[m], mask), dim=-1)
                  for m in self.modalities]
        batch = pooled[0].shape[0]
        if batch < 2:
            return pooled[0].sum() * 0.0
        target = torch.arange(batch, device=pooled[0].device)
        terms = []
        for left in range(len(pooled)):
            for right in range(left + 1, len(pooled)):
                similarity = pooled[left] @ pooled[right].t() / self.temperature
                terms.append(.5 * (F.cross_entropy(similarity, target) +
                                    F.cross_entropy(similarity.t(), target)))
        return torch.stack(terms).mean()

    def video_scores(self, probs, mask, lengths):
        counts = topk_counts(lengths, self.k_proportion)
        return {name: topk_mean(prob, mask, counts)
                for name, prob in probs.items()}


def configure_training(model, active):
    for name, branch in model.branches.items():
        enabled = name == active
        for parameter in branch.parameters():
            parameter.requires_grad_(enabled)
        branch.train(enabled)


ARMS = ("cyclic_control", "temporal_residual")
