"""Modality-typed REBA adaptation with class-aware bidirectional alignment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITIES = ("audio", "visual", "text")


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask[:, :, None].to(value.dtype)
    return (value * weight).sum(1) / weight.sum(1).clamp(min=1.0)


class TemporalExpert(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width, width, kernel_size=3, padding=dilation,
            dilation=dilation, groups=width,
        )
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = value
        update = self.norm(value).transpose(1, 2)
        update = self.pointwise(self.depthwise(update)).transpose(1, 2)
        update = self.dropout(F.gelu(update))
        return (residual + update) * mask[:, :, None]


class ResidualMultiScaleExperts(nn.Module):
    def __init__(self, width: int, alpha: float, dropout: float):
        super().__init__()
        self.alpha = float(alpha)
        self.scale1 = TemporalExpert(width, 1, dropout)
        self.scale2 = TemporalExpert(width, 2, dropout)
        self.scale4 = TemporalExpert(width, 4, dropout)
        self.gate = nn.Sequential(
            nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 2)
        )

    def forward(self, value: torch.Tensor, mask: torch.Tensor):
        base = self.scale1(value, mask)
        coarse2 = self.scale2(value, mask)
        coarse4 = self.scale4(value, mask)
        stop = base.detach()
        delta2 = coarse2 - stop
        delta4 = coarse4 - stop
        gate_input = torch.cat(
            [F.layer_norm(base, (base.shape[-1],)),
             F.layer_norm(delta2, (delta2.shape[-1],)),
             F.layer_norm(delta4, (delta4.shape[-1],))], dim=-1,
        )
        weights = F.softmax(self.gate(gate_input), dim=-1)
        fused = base + self.alpha * (
            weights[..., :1] * delta2 + weights[..., 1:] * delta4
        )
        return fused * mask[:, :, None], base


class TypedREBA(nn.Module):
    def __init__(
        self,
        dims: dict[str, int],
        width: int = 128,
        residual_alpha: float = 0.2,
        dropout: float = 0.1,
        pool_temperature: float = 0.25,
        alignment_temperature: float = 0.10,
    ):
        super().__init__()
        if tuple(dims) != MODALITIES:
            raise ValueError("dims must follow the frozen modality order")
        if width <= 0 or pool_temperature <= 0 or alignment_temperature <= 0:
            raise ValueError("width and temperatures must be positive")
        if residual_alpha < 0 or not 0 <= dropout < 1:
            raise ValueError("invalid residual_alpha or dropout")
        self.dims = dict(dims)
        self.pool_temperature = float(pool_temperature)
        self.alignment_temperature = float(alignment_temperature)
        self.projectors = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(dims[name]), nn.Linear(dims[name], width))
            for name in MODALITIES
        })
        self.temporal = nn.ModuleDict({
            name: ResidualMultiScaleExperts(width, residual_alpha, dropout)
            for name in MODALITIES
        })
        self.modality_gate = nn.Sequential(
            nn.Linear(width * len(MODALITIES), width), nn.GELU(),
            nn.Linear(width, len(MODALITIES)),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, 1),
        )
        self.occupancy = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def _fuse(self, typed, mask):
        gate = F.softmax(self.modality_gate(torch.cat(typed, dim=-1)), dim=-1)
        stack = torch.stack(typed, dim=2)
        fused = (stack * gate[:, :, :, None]).sum(2) * mask[:, :, None]
        return fused, gate

    def _scores(self, fused, mask):
        logits = self.head(fused).squeeze(-1).masked_fill(~mask, -30.0)
        probability = torch.sigmoid(logits) * mask
        mean = probability.sum(1) / mask.sum(1).clamp(min=1)
        attention = F.softmax(
            logits.masked_fill(~mask, -1e4) / self.pool_temperature, dim=1
        )
        softmax_pool = (attention * probability).sum(1)
        occupancy = torch.sigmoid(self.occupancy(masked_mean(fused, mask)).squeeze(-1))
        bag = (occupancy * mean + (1.0 - occupancy) * softmax_pool).clamp(
            1e-6, 1.0 - 1e-6
        )
        return probability, logits, bag, occupancy

    def forward(self, feats, mask, include_control=False):
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean [B,T]")
        if tuple(feats) != MODALITIES:
            raise ValueError("feature dictionaries must follow modality order")
        batch, length = mask.shape
        if torch.any(mask.sum(1) == 0):
            raise ValueError("every video must contain at least one valid second")
        for name in MODALITIES:
            expected = (batch, length, self.dims[name])
            if feats[name].shape != expected:
                raise ValueError(f"invalid {name} shape; expected {expected}")
        projected, residual, base = [], [], []
        for name in MODALITIES:
            value = self.projectors[name](feats[name]) * mask[:, :, None]
            projected.append(value)
            adapted, scale1 = self.temporal[name](value, mask)
            residual.append(adapted)
            base.append(scale1)
        fused, modality_gate = self._fuse(residual, mask)
        probability, logits, bag, occupancy = self._scores(fused, mask)
        output = {
            "score": bag[:, None] * probability,
            "frame_probability": probability,
            "frame_logits": logits,
            "bag_probability": bag,
            "occupancy": occupancy,
            "modality_gate": modality_gate,
            "typed_embeddings": residual,
            "fused_embedding": fused,
        }
        if include_control:
            base_fused, _ = self._fuse(base, mask)
            base_probability, _, base_bag, _ = self._scores(base_fused, mask)
            output["score_scale1_control"] = base_bag[:, None] * base_probability
        return output

    def class_aware_bialign(self, output, labels, mask):
        logits = output["frame_logits"].detach().masked_fill(~mask, -1e4)
        weights = F.softmax(logits / self.pool_temperature, dim=1)[:, :, None]
        visual_audio = 0.5 * (
            output["typed_embeddings"][0] + output["typed_embeddings"][1]
        )
        text = output["typed_embeddings"][2]
        av = F.normalize((weights * visual_audio).sum(1), dim=-1)
        tx = F.normalize((weights * text).sum(1), dim=-1)
        similarity = av @ tx.t() / self.alignment_temperature
        positive = labels[:, None].eq(labels[None, :])

        def direction(matrix, positive_mask):
            numerator = torch.logsumexp(matrix.masked_fill(~positive_mask, -1e4), 1)
            denominator = torch.logsumexp(matrix, 1)
            return (denominator - numerator).mean()

        return 0.5 * (
            direction(similarity, positive) + direction(similarity.t(), positive.t())
        )

    def loss(
        self, output, labels, mask, lambda_align,
        sample_weights=None,
    ):
        per_item = F.binary_cross_entropy(
            output["bag_probability"], labels, reduction="none"
        )
        if sample_weights is None:
            bag = per_item.mean()
        else:
            bag = (per_item * sample_weights).sum() / sample_weights.sum().clamp(min=1e-6)
        align = self.class_aware_bialign(output, labels, mask)
        total = bag + lambda_align * align
        return total, {"bag": bag.detach(), "align": align.detach()}
