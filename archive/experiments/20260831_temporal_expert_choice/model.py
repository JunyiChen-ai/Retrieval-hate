"""Temporal Expert-Choice MIL model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-7
ARMS = ("token_choice", "expert_choice")


class ModalityExpert(nn.Module):
    def __init__(self, in_dim, hidden, embed, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, embed), nn.ReLU(inplace=True))
        self.evidence = nn.Linear(embed, 1)
        self.affinity = nn.Linear(embed, 1)

    def forward(self, x):
        embed = self.proj(self.norm(x))
        return embed, self.evidence(embed).squeeze(-1), \
            self.affinity(embed).squeeze(-1)


def topk_counts(lengths, k_proportion):
    return torch.clamp(torch.ceil(lengths.float() / float(k_proportion)),
                       min=1).long()


def topk_mean(prob, mask, counts):
    filled = prob.masked_fill(~mask, -1.0)
    order = torch.argsort(filled, dim=1, descending=True)
    ranks = torch.arange(prob.shape[1], device=prob.device)[None, :]
    keep = ranks < counts[:, None]
    picked = torch.gather(prob, 1, order)
    return (picked * keep).sum(1) / counts.to(prob.dtype)


def _expert_choice(affinity, mask, counts):
    """Each modality chooses exactly counts[b] valid time tokens."""
    batch, modalities, steps = affinity.shape
    selected = torch.zeros_like(affinity, dtype=torch.bool)
    for modality in range(modalities):
        filled = affinity[:, modality].masked_fill(~mask, float("-inf"))
        order = torch.argsort(filled, dim=1, descending=True)
        ranks = torch.arange(steps, device=affinity.device)[None, :]
        keep_rank = ranks < counts[:, None]
        choice = torch.zeros_like(mask).scatter(1, order, keep_rank) & mask
        selected[:, modality] = choice
    return selected


def _token_choice(affinity, mask, total_assignments):
    """Token-centric routing with exactly the matched assignment budget."""
    batch, modalities, _ = affinity.shape
    selected = torch.zeros_like(affinity, dtype=torch.bool)
    for item in range(batch):
        length = int(mask[item].sum())
        budget = int(total_assignments[item])
        base, extra = divmod(budget, length)
        if base >= modalities:
            selected[item, :, :length] = True
            continue
        modality_order = torch.argsort(
            affinity[item, :, :length], dim=0, descending=True)
        if base:
            for rank in range(base):
                selected[item].scatter_(
                    0, modality_order[rank:rank + 1],
                    torch.ones(1, length, dtype=torch.bool,
                               device=affinity.device))
        if extra:
            next_modality = modality_order[base]
            next_value = affinity[item, next_modality,
                                  torch.arange(length, device=affinity.device)]
            chosen_time = torch.topk(next_value, k=extra).indices
            selected[item, next_modality[chosen_time], chosen_time] = True
    return selected


class TemporalExpertChoice(nn.Module):
    def __init__(self, dims, arm, hidden=512, embed=64, dropout=.05,
                 k_proportion=8, temperature=.07):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        self.modalities = tuple(dims)
        self.arm = arm
        self.k_proportion = int(k_proportion)
        self.temperature = float(temperature)
        self.experts = nn.ModuleDict({
            name: ModalityExpert(dim, hidden, embed, dropout)
            for name, dim in dims.items()})
        self.background = nn.Parameter(torch.zeros(()))

    def forward(self, feats, mask):
        embeds, evidence, affinity = [], [], []
        for name in self.modalities:
            emb, ev, aff = self.experts[name](feats[name])
            embeds.append(emb)
            evidence.append(ev)
            affinity.append(aff)
        evidence = torch.stack(evidence, dim=1)
        affinity = torch.stack(affinity, dim=1)
        lengths = mask.sum(1)
        per_expert = topk_counts(lengths, self.k_proportion)
        total = per_expert * len(self.modalities)
        if self.arm == "expert_choice":
            selected = _expert_choice(affinity, mask, per_expert)
        else:
            selected = _token_choice(affinity, mask, total)
        gate = torch.sigmoid(affinity) * selected.to(affinity.dtype)
        contribution = gate * evidence
        frame_logit = self.background + contribution.sum(1)
        frame_prob = torch.sigmoid(frame_logit) * mask
        return {
            "prob": frame_prob, "logit": frame_logit,
            "selected": selected, "gate": gate,
            "contribution": contribution, "evidence": evidence,
            "affinity": affinity, "embeds": embeds,
            "per_expert_count": per_expert,
            "total_assignment_budget": total,
        }

    def mil_loss(self, prob, mask, lengths, labels):
        video = self.video_score(prob, mask, lengths)
        return F.binary_cross_entropy(video, labels), video

    def video_score(self, prob, mask, lengths):
        counts = topk_counts(lengths, self.k_proportion)
        return topk_mean(prob, mask, counts).clamp(EPS, 1.0 - EPS)

    def smoothness_loss(self, prob, mask):
        pair = mask[:, 1:] & mask[:, :-1]
        denom = pair.sum().clamp(min=1).to(prob.dtype)
        return ((((prob[:, 1:] - prob[:, :-1]) ** 2) * pair).sum() / denom)

    def contrastive_loss(self, embeds, mask):
        weights = mask.unsqueeze(-1).to(embeds[0].dtype)
        pooled = [F.normalize((x * weights).sum(1) /
                              weights.sum(1).clamp(min=1), dim=-1)
                  for x in embeds]
        batch = pooled[0].shape[0]
        if batch < 2:
            return pooled[0].sum() * 0.0
        target = torch.arange(batch, device=mask.device)
        total, pairs = 0.0, 0
        for left in range(len(pooled)):
            for right in range(left + 1, len(pooled)):
                similarity = pooled[left] @ pooled[right].t() / self.temperature
                total = total + .5 * (F.cross_entropy(similarity, target) +
                                      F.cross_entropy(similarity.t(), target))
                pairs += 1
        return total / pairs
