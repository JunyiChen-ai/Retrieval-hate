"""Shared frozen-POWA checkpoint and context-quotient residual utilities."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_logit(probability: torch.Tensor, eps: float = 1e-5):
    probability = probability.clamp(eps, 1.0 - eps)
    return torch.log(probability) - torch.log1p(-probability)


def load_corpus_powa(checkpoint: Path, corpus: str, device: str):
    from powa_macil.model import POWAMACIL

    checkpoint = Path(checkpoint).resolve()
    meta = json.loads((checkpoint / "train_meta.json").read_text())
    cfg = SimpleNamespace(**meta["args"])
    if list(getattr(cfg, "corpora", [corpus])) != [corpus]:
        raise ValueError("POWA checkpoint is not corpus-specific")
    model = POWAMACIL(cfg, policy=corpus).to(device)
    state_path = checkpoint / "model.pth"
    state = torch.load(state_path, map_location=device)
    legacy = "policy_residual_gate" not in state
    model.load_state_dict(state, strict=not legacy)
    model.use_policy_residual = (
        not legacy and not getattr(cfg, "typed_only", False)
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, cfg, meta


def masked_center(values: torch.Tensor, valid_mask: torch.Tensor):
    weight = valid_mask[..., None].to(values.dtype)
    mean = (values * weight).sum(1, keepdim=True) / weight.sum(
        1, keepdim=True
    ).clamp_min(1)
    return (values - mean) * weight


class ContextQuotientResidualHead(nn.Module):
    """Temporal residual invariant to an added video-constant input vector."""

    def __init__(self, text_dim: int = 768, hidden: int = 128):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, hidden)
        input_dim = 128 + 128 + hidden + 6 + 1 + 1
        self.input_projection = nn.Linear(input_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, 3)
        self.conv2 = nn.Conv1d(hidden, hidden, 3)
        self.output = nn.Linear(hidden, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, frozen, text, valid_mask):
        parts = [
            frozen["audio_rep"], frozen["visual_rep"],
            self.text_projection(text), frozen["primitive_logits"],
            safe_logit(frozen["frame_prob"])[..., None],
            frozen["base_frame_logits"],
        ]
        centered = [masked_center(part, valid_mask) for part in parts]
        hidden = F.gelu(self.norm(self.input_projection(torch.cat(centered, -1))))
        hidden = hidden * valid_mask[..., None].to(hidden.dtype)
        # Convolve each sequence at its own valid boundary. Padding a mixed-
        # length batch to its maximum would otherwise replace replicate(last
        # valid) with zeros for short videos and make train/test disagree.
        encoded = []
        width = hidden.shape[1]
        for row, mask in zip(hidden, valid_mask):
            length = int(mask.sum().item())
            if length <= 0:
                raise ValueError("residual head received an empty sequence")
            temporal = row[:length].transpose(0, 1)[None]
            temporal = F.gelu(self.conv1(
                F.pad(temporal, (1, 1), mode="replicate")
            ))
            temporal = F.gelu(self.conv2(
                F.pad(temporal, (1, 1), mode="replicate")
            ))
            temporal = temporal[0].transpose(0, 1)
            encoded.append(F.pad(temporal, (0, 0, 0, width - length)))
        residual = self.output(torch.stack(encoded)).squeeze(-1)
        weight = valid_mask.to(residual.dtype)
        mean = (residual * weight).sum(1, keepdim=True) / weight.sum(
            1, keepdim=True
        ).clamp_min(1)
        return (residual - mean) * weight


class FrozenPowaContextResidual(nn.Module):
    def __init__(self, powa, text_dim: int = 768, hidden: int = 128,
                 max_seqlen: int = 200):
        super().__init__()
        self.powa = powa
        self.max_seqlen = int(max_seqlen)
        self.powa.eval()
        self.residual_head = ContextQuotientResidualHead(text_dim, hidden)

    def train(self, mode: bool = True):
        super().train(mode)
        self.powa.eval()
        return self

    def forward(self, f_a, f_v, f_t, lengths, valid_mask, policy):
        with torch.no_grad():
            raw = self.powa(
                f_a, f_v, f_t, lengths, valid_mask, policy=policy
            )
        frozen = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in raw.items()
        }
        residual = self.residual_head(frozen, f_t, valid_mask)
        anchor_logit = safe_logit(frozen["frame_prob"])
        return {
            "anchor_prob": frozen["frame_prob"],
            "anchor_logit": anchor_logit,
            "residual": residual,
            "candidate_prob": torch.sigmoid(anchor_logit + residual),
        }
