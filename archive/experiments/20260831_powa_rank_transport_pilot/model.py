"""Frozen POWA values plus a trainable temporal-order residual head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_logit(probability: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    probability = probability.clamp(eps, 1.0 - eps)
    return torch.log(probability) - torch.log1p(-probability)


class TemporalOrderResidual(nn.Module):
    """A standard masked temporal head; the final layer starts at identity."""

    def __init__(self, text_dim: int = 768, hidden: int = 128):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, hidden)
        # audio/visual context: 128 each; text: hidden; typed primitives: 6;
        # anchor typed score and base AV logit: one each.
        input_dim = 128 + 128 + hidden + 6 + 1 + 1
        self.input_projection = nn.Linear(input_dim, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, frozen, text, valid_mask):
        text_context = self.text_projection(text)
        parts = [
            frozen["audio_rep"],
            frozen["visual_rep"],
            text_context,
            frozen["primitive_logits"],
            safe_logit(frozen["frame_prob"])[..., None],
            frozen["base_frame_logits"],
        ]
        hidden = self.input_norm(self.input_projection(torch.cat(parts, -1)))
        hidden = F.gelu(hidden)
        hidden = hidden * valid_mask[..., None].to(hidden.dtype)
        contextual = self.temporal(hidden.transpose(1, 2)).transpose(1, 2)
        contextual = (hidden + contextual) * valid_mask[..., None].to(hidden.dtype)
        return self.output(contextual).squeeze(-1) * valid_mask.to(hidden.dtype)


class FrozenPowaTemporalAssignment(nn.Module):
    """Keep POWA frozen and learn only an order residual over its evidence."""

    def __init__(self, powa, text_dim: int = 768, hidden: int = 128):
        super().__init__()
        self.powa = powa
        for parameter in self.powa.parameters():
            parameter.requires_grad_(False)
        self.powa.eval()
        self.order_head = TemporalOrderResidual(text_dim=text_dim, hidden=hidden)

    def train(self, mode: bool = True):
        super().train(mode)
        self.powa.eval()
        return self

    def forward(self, f_a, f_v, f_t, lengths, valid_mask, policy):
        with torch.no_grad():
            frozen = self.powa(
                f_a, f_v, f_t, lengths, valid_mask, policy=policy
            )
        frozen = {
            key: (value.detach() if torch.is_tensor(value) else value)
            for key, value in frozen.items()
        }
        anchor_logit = safe_logit(frozen["frame_prob"])
        residual = self.order_head(frozen, f_t, valid_mask)
        order_logit = (anchor_logit + residual) * valid_mask.to(anchor_logit.dtype)
        return {
            "anchor_prob": frozen["frame_prob"],
            "anchor_logit": anchor_logit,
            "order_residual": residual,
            "order_logit": order_logit,
        }
