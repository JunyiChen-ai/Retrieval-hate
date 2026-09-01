"""Shared differentiable modality-marked temporal splat model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-7
WIDTHS = (1.0, 2.0, 4.0, 8.0)


def topk_counts(lengths, k_proportion):
    return torch.clamp(
        torch.ceil(lengths.float() / float(k_proportion)), min=1
    ).long()


def topk_mean(probability, mask, counts):
    filled = probability.masked_fill(~mask, -1.0)
    order = torch.argsort(filled, dim=1, descending=True)
    ranks = torch.arange(probability.shape[1], device=probability.device)[None]
    keep = ranks < counts[:, None]
    picked = torch.gather(probability, 1, order)
    return (picked * keep).sum(1) / counts.to(probability.dtype)


class SplatExpert(nn.Module):
    def __init__(self, in_dim, hidden, embed, dropout, scales):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed),
            nn.ReLU(inplace=True),
        )
        self.amplitude = nn.Linear(embed, 1)
        self.duration = nn.Linear(embed, scales)
        nn.init.constant_(self.amplitude.bias, -3.0)

    def forward(self, features):
        embedding = self.proj(self.norm(features))
        return (
            embedding,
            torch.sigmoid(self.amplitude(embedding).squeeze(-1)),
            F.softmax(self.duration(embedding), dim=-1),
        )


class MarkedTemporalSplatMIL(nn.Module):
    def __init__(self, dims, hidden=512, embed=64, dropout=.05,
                 k_proportion=8, temperature=.07, widths=WIDTHS):
        super().__init__()
        self.modalities = tuple(dims)
        self.widths = tuple(float(value) for value in widths)
        self.k_proportion = int(k_proportion)
        self.temperature = float(temperature)
        self.experts = nn.ModuleDict({
            name: SplatExpert(dim, hidden, embed, dropout, len(self.widths))
            for name, dim in dims.items()
        })

    def _kernel(self, sigma, device, dtype):
        radius = int(math.ceil(3.0 * sigma))
        offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        values = torch.exp(-.5 * (offsets / sigma) ** 2)
        values = values / values.sum()
        return [(int(offset), value) for offset, value in zip(offsets, values)]

    def _render(self, amplitudes, mixtures, mask):
        batch, steps = mask.shape
        log_survival = torch.zeros(
            batch, steps, device=mask.device, dtype=amplitudes[0].dtype
        )
        kernels = {}
        for scale_index, sigma in enumerate(self.widths):
            for offset, value in self._kernel(
                sigma, amplitudes[0].device, amplitudes[0].dtype
            ):
                if abs(offset) < steps:
                    if offset not in kernels:
                        kernels[offset] = torch.zeros(
                            len(self.widths),
                            device=amplitudes[0].device,
                            dtype=amplitudes[0].dtype,
                        )
                    kernels[offset][scale_index] = value
        for amplitude, mixture in zip(amplitudes, mixtures):
            amplitude = amplitude * mask
            for offset, kernel_vector in kernels.items():
                if offset >= 0:
                    source = slice(0, steps - offset)
                    target = slice(offset, steps)
                else:
                    source = slice(-offset, steps)
                    target = slice(0, steps + offset)
                mixed_kernel = (mixture[:, source] * kernel_vector).sum(-1)
                contribution = amplitude[:, source] * mixed_kernel
                valid = mask[:, source] & mask[:, target]
                contribution = (contribution * valid).clamp(min=0.0, max=1.0 - EPS)
                log_survival[:, target] += torch.log1p(-contribution)
        return (1.0 - torch.exp(log_survival)) * mask

    def forward(self, features, mask):
        embeddings, amplitudes, mixtures = [], [], []
        for name in self.modalities:
            embedding, amplitude, mixture = self.experts[name](features[name])
            embeddings.append(embedding)
            amplitudes.append(amplitude)
            mixtures.append(mixture)
        probability = self._render(amplitudes, mixtures, mask)
        return {
            "prob": probability,
            "embeds": embeddings,
            "amplitudes": torch.stack(amplitudes, dim=1),
            "mixtures": torch.stack(mixtures, dim=1),
        }

    def video_score(self, probability, mask, lengths):
        return topk_mean(
            probability, mask, topk_counts(lengths, self.k_proportion)
        ).clamp(EPS, 1.0 - EPS)

    def mil_loss(self, probability, mask, lengths, labels):
        video = self.video_score(probability, mask, lengths)
        return F.binary_cross_entropy(video, labels), video

    def smoothness_loss(self, probability, mask):
        valid = mask[:, 1:] & mask[:, :-1]
        return (
            ((probability[:, 1:] - probability[:, :-1]) ** 2 * valid).sum()
            / valid.sum().clamp(min=1).to(probability.dtype)
        )

    def contrastive_loss(self, embeddings, mask):
        weight = mask.unsqueeze(-1).to(embeddings[0].dtype)
        pooled = [
            F.normalize((value * weight).sum(1) / weight.sum(1).clamp(min=1), dim=-1)
            for value in embeddings
        ]
        batch = len(pooled[0])
        if batch < 2:
            return pooled[0].sum() * 0.0
        target = torch.arange(batch, device=mask.device)
        total, pairs = 0.0, 0
        for left in range(len(pooled)):
            for right in range(left + 1, len(pooled)):
                logits = pooled[left] @ pooled[right].t() / self.temperature
                total += .5 * (
                    F.cross_entropy(logits, target)
                    + F.cross_entropy(logits.t(), target)
                )
                pairs += 1
        return total / pairs
