"""MultiHateLoc with modality-indexed selective supervised contrastive loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from imports import base_model


ABSTAIN = -1
BACKGROUND = 0
CARRIER = 1


class CarrierItS2CLR(nn.Module):
    ARMS = (
        "anchor",
        "broadcast",
        "core",
        "branch_selector",
        "shuffled_carrier",
        "abstain_negative",
        "nonpositive_background",
        "projection_only",
    )

    def __init__(self, dims, arm, hidden=256, embed=128, dropout=0.1,
                 k_proportion=3, temperature=0.07, max_instances=256):
        super().__init__()
        if arm not in self.ARMS:
            raise ValueError(f"unknown arm {arm}")
        self.arm = arm
        self.modalities = tuple(dims)
        self.max_instances = int(max_instances)
        self.temperature = float(temperature)
        self.backbone = base_model.MultiHateLoc(
            dims, hidden=hidden, embed=embed, dropout=dropout,
            k_proportion=k_proportion, temperature=temperature,
        )
        self.projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embed, embed), nn.ReLU(inplace=True),
                nn.Linear(embed, embed),
            )
            for name in self.modalities
        })

    def forward(self, feats, mask):
        return self.backbone(feats, mask)

    @staticmethod
    def _stratified_cap(z, labels, maximum):
        if len(z) <= maximum:
            return z, labels
        pieces = []
        per_class = maximum // 2
        for value in (BACKGROUND, CARRIER):
            index = torch.where(labels == value)[0]
            if len(index) > per_class:
                index = index[torch.randperm(len(index), device=index.device)[:per_class]]
            pieces.append(index)
        chosen = torch.cat(pieces)
        if len(chosen) < maximum:
            unused = torch.ones(len(labels), dtype=torch.bool, device=labels.device)
            unused[chosen] = False
            remainder = torch.where(unused)[0]
            take = min(maximum - len(chosen), len(remainder))
            if take:
                remainder = remainder[torch.randperm(len(remainder), device=z.device)[:take]]
                chosen = torch.cat([chosen, remainder])
        return z[chosen], labels[chosen]

    def _one_supcon(self, embeddings, states):
        valid = states != ABSTAIN
        if valid.sum() < 3:
            return embeddings.sum() * 0.0
        z = F.normalize(embeddings[valid], dim=-1)
        labels = states[valid]
        z, labels = self._stratified_cap(z, labels, self.max_instances)
        if len(z) < 3:
            return embeddings.sum() * 0.0
        logits = z @ z.t() / self.temperature
        eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(eye, -torch.inf)
        same = labels[:, None].eq(labels[None, :]) & ~eye
        usable = same.any(dim=1)
        if not usable.any():
            return embeddings.sum() * 0.0
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        mean_positive = (log_prob.masked_fill(~same, 0.0).sum(1)
                         / same.sum(1).clamp(min=1))
        return -mean_positive[usable].mean()

    def selective_contrastive_loss(self, embeds, states, mask):
        if self.arm == "anchor":
            return self.backbone.contrastive_loss(embeds, mask)
        losses = []
        for index, name in enumerate(self.modalities):
            source = embeds[index].detach() if self.arm == "projection_only" else embeds[index]
            projected = self.projectors[name](source)
            losses.append(self._one_supcon(
                projected.reshape(-1, projected.shape[-1]),
                states[:, :, index].reshape(-1),
            ))
        return torch.stack(losses).mean()

    def training_loss(self, output, states, mask, lengths, labels,
                      smooth_weight=0.1, contrast_weight=0.2):
        mil, per_branch = self.backbone.mil_loss(
            output["probs"], mask, lengths, labels
        )
        smooth = self.backbone.smoothness_loss(output["probs"], mask)
        contrast = self.selective_contrastive_loss(
            output["embeds"], states, mask
        )
        total = mil + smooth_weight * smooth + contrast_weight * contrast
        return total, {"mil": mil, "smooth": smooth, "contrast": contrast,
                       **{f"mil_{key}": value for key, value in per_branch.items()}}

    def frame_scores(self, output):
        return output["probs"]["fused"]

    def video_scores(self, output, mask, lengths):
        return self.backbone.video_scores(output["probs"], mask, lengths)["fused"]
