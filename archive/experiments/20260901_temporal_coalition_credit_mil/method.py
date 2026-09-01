from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MultiHateLoc, topk_counts, topk_mask


ARMS = ("anchor", "aligned", "shifted")


class TemporalCoalitionCreditMIL(MultiHateLoc):
    """Anchor-compatible MultiHateLoc with local coalition-credit routing."""

    def __init__(self, dims, alpha, arm, **kwargs):
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if arm == "anchor" and alpha != 0.0:
            raise ValueError("anchor requires alpha=0")
        if arm != "anchor" and alpha <= 0.0:
            raise ValueError("result-relevant arms require alpha>0")
        super().__init__(dims, **kwargs)
        self.alpha = float(alpha)
        self.arm = arm
        self.local_router = None
        if self.alpha > 0.0:
            embed = self.branches[self.modalities[0]].head.in_features
            self.local_router = nn.Sequential(
                nn.Linear(embed, max(embed // 2, 1)), nn.Tanh(),
                nn.Linear(max(embed // 2, 1), 1))

    def _fuse_without_dropout(self, embeds, global_weights, coalition):
        """Original fused logit with absent coalition embeddings set to zero."""
        scaled = []
        n_modalities = len(self.modalities)
        for index, embed in enumerate(embeds):
            present = 1.0 if coalition & (1 << index) else 0.0
            weight = global_weights[:, index, None, None] * n_modalities
            scaled.append(embed * weight * present)
        value = torch.cat(scaled, dim=-1)
        for layer in self.fuse:
            if isinstance(layer, nn.Dropout):
                continue
            value = layer(value)
        return self.fuse_head(value).squeeze(-1)

    @torch.no_grad()
    def coalition_credit(self, embeds, global_weights):
        """Exact three-player Shapley values and a fixed responsibility map."""
        n_modalities = len(self.modalities)
        if n_modalities != 3:
            raise ValueError("formal method is fixed to three modalities")
        values = {
            coalition: self._fuse_without_dropout(
                embeds, global_weights, coalition)
            for coalition in range(1 << n_modalities)
        }
        credits = []
        normalizer = math.factorial(n_modalities)
        for modality in range(n_modalities):
            contribution = torch.zeros_like(values[0])
            for coalition in range(1 << n_modalities):
                if coalition & (1 << modality):
                    continue
                size = coalition.bit_count()
                coefficient = (math.factorial(size) *
                               math.factorial(n_modalities - size - 1) /
                               normalizer)
                contribution = contribution + coefficient * (
                    values[coalition | (1 << modality)] - values[coalition])
            credits.append(contribution)
        signed = torch.stack(credits, dim=-1)
        positive = F.relu(signed)
        denominator = positive.sum(-1, keepdim=True)
        fallback = global_weights[:, None, :].expand_as(positive)
        target = torch.where(
            denominator > 1e-12,
            positive / denominator.clamp(min=1e-12), fallback)
        return signed, target, values[0], values[(1 << n_modalities) - 1]

    def forward(self, feats, mask):
        base = super().forward(feats, mask)
        if self.alpha == 0.0:
            return base
        embeds = base["embeds"]
        global_weights = base["weights"]
        local_logits = torch.cat(
            [self.local_router(embed) for embed in embeds], dim=-1)
        local_weights = F.softmax(local_logits, dim=-1)
        time_weights = ((1.0 - self.alpha) * global_weights[:, None, :] +
                        self.alpha * local_weights)
        n_modalities = len(self.modalities)
        scaled = [embed * (time_weights[:, :, index] * n_modalities)[:, :, None]
                  for index, embed in enumerate(embeds)]
        fused_embed = self.fuse(torch.cat(scaled, dim=-1))
        fused_probability = torch.sigmoid(
            self.fuse_head(fused_embed).squeeze(-1)) * mask
        probabilities = dict(base["probs"])
        probabilities["fused"] = fused_probability
        signed, target, empty_logit, full_logit = self.coalition_credit(
            embeds, global_weights)
        base.update({
            "probs": probabilities,
            "fused_embed": fused_embed,
            "base_fused_probability": base["probs"]["fused"],
            "local_weights": local_weights,
            "time_weights": time_weights,
            "coalition_signed_credit": signed,
            "coalition_target": target,
            "coalition_empty_logit": empty_logit,
            "coalition_full_logit": full_logit,
        })
        return base

    @staticmethod
    def _shift_targets(target, mask):
        shifted = target.clone()
        for batch_index in range(target.shape[0]):
            length = int(mask[batch_index].sum().item())
            if length <= 1:
                continue
            offset = max(1, length // 2)
            if offset % length == 0:
                offset = 1
            shifted[batch_index, :length] = torch.roll(
                target[batch_index, :length], shifts=offset, dims=0)
        return shifted

    def responsibility_terms(self, output, mask, lengths, labels):
        zero = output["probs"]["fused"].sum() * 0.0
        if self.alpha == 0.0:
            return zero, zero, zero, 0
        counts = topk_counts(lengths, self.k_proportion)
        witness = topk_mask(
            output["base_fused_probability"].detach(), mask, counts)
        witness = witness & (labels >= 0.5)[:, None]
        aligned_target = output["coalition_target"].detach()
        training_target = (self._shift_targets(aligned_target, mask)
                           if self.arm == "shifted" else aligned_target)
        log_probability = torch.log(
            output["local_weights"].clamp(min=1e-8))
        per_frame = -(training_target * log_probability).sum(-1)
        denominator = witness.sum()
        loss = ((per_frame * witness).sum() /
                denominator.clamp(min=1).to(per_frame.dtype))
        aligned_agreement = (
            output["local_weights"].argmax(-1) == aligned_target.argmax(-1))
        train_agreement = (
            output["local_weights"].argmax(-1) == training_target.argmax(-1))
        aligned = ((aligned_agreement & witness).sum().to(per_frame.dtype) /
                   denominator.clamp(min=1).to(per_frame.dtype))
        trained = ((train_agreement & witness).sum().to(per_frame.dtype) /
                   denominator.clamp(min=1).to(per_frame.dtype))
        return loss, aligned, trained, int(denominator.item())
