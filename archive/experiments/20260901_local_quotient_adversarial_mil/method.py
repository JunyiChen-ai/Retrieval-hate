from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ARMS = ("local_control", "local_adversarial")


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, coefficient):
        ctx.coefficient = float(coefficient)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def gradient_reverse(value, coefficient):
    return _GradientReversal.apply(value, coefficient)


def masked_mean(value, mask):
    weight = mask.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum(1) / weight.sum(1).clamp(min=1.0)


class LocalQuotientModel(nn.Module):
    """MultiHateLoc backbone with isolated global and centered local channels."""

    def __init__(self, global_base, local_base, embed, n_video_ids, n_position_bins=8,
                 local_scale=1.0):
        super().__init__()
        self.global_base = global_base
        self.local_base = local_base
        self.modalities = global_base.modalities
        self.k_proportion = global_base.k_proportion
        self.global_head = nn.Linear(embed, 1)
        self.local_head = nn.Linear(embed, 1)
        self.video_adversary = nn.Sequential(
            nn.Linear(embed, embed), nn.ReLU(inplace=True),
            nn.Linear(embed, n_video_ids))
        self.position_adversary = nn.Sequential(
            nn.Linear(embed, embed // 2), nn.ReLU(inplace=True),
            nn.Linear(embed // 2, n_position_bins))
        self.n_position_bins = int(n_position_bins)
        self.local_scale = float(local_scale)
        self.video_grl = 0.0
        self.position_grl = 0.0

    def set_grl(self, video_coefficient, position_coefficient):
        self.video_grl = float(video_coefficient)
        self.position_grl = float(position_coefficient)

    def forward(self, feats, mask):
        global_output = self.global_base(feats, mask)
        local_output = self.local_base(feats, mask)
        global_fused = global_output["fused_embed"]
        local_fused = local_output["fused_embed"]
        global_rep = masked_mean(global_fused, mask)
        local_center = masked_mean(local_fused, mask)
        local_rep = (local_fused - local_center[:, None, :]) * mask[:, :, None]
        global_logit = self.global_head(global_rep).squeeze(-1)
        local_raw = self.local_head(local_rep).squeeze(-1)
        local_mean = ((local_raw * mask).sum(1) /
                      mask.sum(1).clamp(min=1).to(local_raw.dtype))
        local_logit = (local_raw - local_mean[:, None]) * mask
        fused_logit = global_logit[:, None] + self.local_scale * local_logit
        probs = {"fused": torch.sigmoid(fused_logit) * mask}
        video_adv = self.video_adversary(
            gradient_reverse(local_rep, self.video_grl))
        position_adv = self.position_adversary(
            gradient_reverse(local_rep, self.position_grl))
        return {
            "probs": probs, "local_embeds": local_output["embeds"],
            "global_weights": global_output["weights"],
            "local_weights": local_output["weights"],
            "global_fused_embed": global_fused,
            "local_fused_embed": local_fused,
            "global_rep": global_rep, "local_rep": local_rep,
            "global_logit": global_logit, "local_logit": local_logit,
            "fused_logit": fused_logit, "video_adv": video_adv,
            "position_adv": position_adv,
        }

    def mil_loss(self, probs, mask, lengths, labels):
        probability = probs["fused"]
        counts = torch.clamp(torch.ceil(
            lengths.float() / float(self.k_proportion)), min=1.0).long()
        filled = probability.masked_fill(~mask, -1.0)
        ordered = torch.sort(filled, dim=1, descending=True).values
        ranks = torch.arange(probability.shape[1], device=probability.device)[None]
        keep = ranks < counts[:, None]
        bag = (ordered * keep).sum(1) / counts.to(probability.dtype)
        loss = F.binary_cross_entropy(bag.clamp(1e-7, 1 - 1e-7), labels)
        return loss, {"fused": loss}

    def smoothness_loss(self, probs, mask):
        probability = probs["fused"]
        pair = mask[:, 1:] & mask[:, :-1]
        denom = pair.sum().clamp(min=1).to(probability.dtype)
        return (((probability[:, 1:] - probability[:, :-1]) ** 2) * pair).sum() / denom

    def contrastive_loss(self, embeds, mask):
        return self.local_base.contrastive_loss(embeds, mask)

    def video_scores(self, probs, mask, lengths):
        probability = probs["fused"]
        counts = torch.clamp(torch.ceil(
            lengths.float() / float(self.k_proportion)), min=1.0).long()
        filled = probability.masked_fill(~mask, -1.0)
        ordered = torch.sort(filled, dim=1, descending=True).values
        ranks = torch.arange(probability.shape[1], device=probability.device)[None]
        keep = ranks < counts[:, None]
        bag = (ordered * keep).sum(1) / counts.to(probability.dtype)
        return {"fused": bag}

    def nuisance_loss(self, output, mask, lengths, video_indices):
        video_terms, position_terms = [], []
        for index in range(mask.shape[0]):
            length = int(lengths[index])
            if length <= 0:
                continue
            video_target = video_indices[index].expand(length)
            video_terms.append(F.cross_entropy(
                output["video_adv"][index, :length], video_target))
            seconds = torch.arange(length, device=mask.device)
            bins = torch.div(seconds * self.n_position_bins, length,
                             rounding_mode="floor").clamp(
                                 max=self.n_position_bins - 1)
            position_terms.append(F.cross_entropy(
                output["position_adv"][index, :length], bins))
        zero = output["fused_logit"].sum() * 0.0
        video_loss = torch.stack(video_terms).mean() if video_terms else zero
        position_loss = (torch.stack(position_terms).mean()
                         if position_terms else zero)
        return video_loss, position_loss
