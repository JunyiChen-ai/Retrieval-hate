"""Typed modality-subset dynamic-MIL CRF.

Feature encoders are frame-local, so temporal effects in the core arm come
from the exact chain partition rather than an implicit temporal convolution.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.multimodal_video_data import DIMS, MODALITIES


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class FactorialWitnessCRF(nn.Module):
    ARMS = ("core", "zero_transition", "collapsed")

    def __init__(self, arm: str, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        if arm not in self.ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(DIMS[name]),
                nn.Linear(DIMS[name], hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            for name in MODALITIES
        })
        bits = torch.tensor(
            [[(state >> bit) & 1 for bit in range(3)] for state in range(8)],
            dtype=torch.float32,
        )
        self.register_buffer("state_bits", bits)
        self.raw_pair_cost = nn.Parameter(
            torch.full((3,), _inverse_softplus(0.1))
        )
        self.raw_switch_cost = nn.Parameter(
            torch.full((3,), _inverse_softplus(0.1))
        )

    def unary_logits(self, feats):
        return torch.cat(
            [self.encoders[name](feats[name]) for name in MODALITIES], dim=-1
        )

    def _emissions(self, unary):
        if self.arm == "collapsed":
            active = torch.logsumexp(unary, dim=-1) - math.log(3.0)
            return torch.stack([torch.zeros_like(active), active], dim=-1)
        emissions = torch.einsum("tm,sm->ts", unary, self.state_bits)
        pair_products = torch.stack([
            self.state_bits[:, 0] * self.state_bits[:, 1],
            self.state_bits[:, 0] * self.state_bits[:, 2],
            self.state_bits[:, 1] * self.state_bits[:, 2],
        ], dim=-1)
        return emissions - pair_products @ F.softplus(self.raw_pair_cost)

    def _transition(self, dtype, device):
        if self.arm == "collapsed":
            bits = torch.tensor([[0.0], [1.0]], dtype=dtype, device=device)
            cost = F.softplus(self.raw_switch_cost).mean()
            return -cost * torch.abs(bits[:, None, :] - bits[None, :, :]).sum(-1)
        if self.arm == "zero_transition":
            return torch.zeros(8, 8, dtype=dtype, device=device)
        bits = self.state_bits.to(dtype=dtype, device=device)
        changes = torch.abs(bits[:, None, :] - bits[None, :, :])
        return -(changes * F.softplus(self.raw_switch_cost)).sum(-1)

    @staticmethod
    def _forward(emissions, transition):
        alpha = emissions[0]
        values = [alpha]
        for time in range(1, len(emissions)):
            alpha = emissions[time] + torch.logsumexp(
                alpha[:, None] + transition, dim=0
            )
            values.append(alpha)
        return torch.stack(values), torch.logsumexp(alpha, dim=0)

    @staticmethod
    def _positive_partition(emissions, transition):
        """Partition over paths that visit a non-empty state at least once."""
        negative_inf = torch.full(
            (), -torch.inf, dtype=emissions.dtype, device=emissions.device
        )
        alpha = torch.cat([negative_inf[None], emissions[0, 1:]], dim=0)
        for time in range(1, len(emissions)):
            continued = torch.logsumexp(alpha[:, None] + transition, dim=0)
            first_active = transition[0] + emissions[time]
            first_active = torch.cat([negative_inf[None], first_active[1:]], dim=0)
            alpha = torch.logsumexp(
                torch.stack([continued + emissions[time], first_active]), dim=0
            )
        return torch.logsumexp(alpha, dim=0)

    @staticmethod
    def _backward(emissions, transition):
        beta = torch.zeros_like(emissions[-1])
        values = [beta]
        for time in range(len(emissions) - 2, -1, -1):
            beta = torch.logsumexp(
                transition + emissions[time + 1][None, :] + beta[None, :], dim=1
            )
            values.append(beta)
        return torch.stack(list(reversed(values)))

    def _one_video(self, unary):
        emissions = self._emissions(unary)
        transition = self._transition(emissions.dtype, emissions.device)
        alpha, _ = self._forward(emissions, transition)
        log_z_positive = self._positive_partition(emissions, transition)
        # Contrast against the same chain prior with zero unary evidence. This
        # is log(8^T-1) for zero-transition 8-state paths, and also removes the
        # length baseline induced by learned transition costs in the core arm.
        # Preserve every input-independent part of the chain prior.  In the
        # typed arms this includes the learned coalition cost; dropping it
        # would let that cost alone move the bag logit when unary evidence is
        # identically zero.
        null_emissions = self._emissions(torch.zeros_like(unary))
        log_z_positive_null = self._positive_partition(null_emissions, transition)
        bag_logit = log_z_positive - log_z_positive_null

        beta = self._backward(emissions, transition)
        active_log_mass = torch.logsumexp(alpha[:, 1:] + beta[:, 1:], dim=1)
        active_posterior = torch.exp(
            active_log_mass - log_z_positive
        ).clamp(0.0, 1.0)
        if emissions.shape[-1] == 8:
            state_log_mass = alpha + beta
            bit_posteriors = torch.stack([
                torch.exp(
                    torch.logsumexp(
                        state_log_mass[:, self.state_bits[:, bit].bool()], dim=1
                    ) - log_z_positive
                ).clamp(0.0, 1.0)
                for bit in range(3)
            ], dim=-1)
        else:
            bit_posteriors = active_posterior[:, None]
        frame_score = torch.sigmoid(bag_logit) * active_posterior
        return bag_logit, frame_score, active_posterior, bit_posteriors

    def forward(self, feats, lengths):
        unary = self.unary_logits(feats)
        bag_logits, frame_scores, active_posteriors, bit_posteriors = [], [], [], []
        for row, length in enumerate(lengths.tolist()):
            bag, frame, posterior, bit_posterior = self._one_video(
                unary[row, : int(length)]
            )
            bag_logits.append(bag)
            frame_scores.append(frame)
            active_posteriors.append(posterior)
            bit_posteriors.append(bit_posterior)
        return {
            "bag_logits": torch.stack(bag_logits),
            "frame_scores": frame_scores,
            "active_posteriors": active_posteriors,
            "bit_posteriors": bit_posteriors,
            "unary_logits": unary,
        }
