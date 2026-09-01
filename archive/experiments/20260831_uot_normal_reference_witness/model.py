"""Typed reject-option transport for weakly supervised hate localization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITIES = ("audio", "visual", "text")


class Projection(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(rows), dim=-1)


class NormalReferenceWitness(nn.Module):
    """Explain typed tokens with shared-capacity normal slots or reject them.

    Every time-by-modality token has fixed unit source mass. It may be assigned
    only to a reference of the same modality or to the reject state. The K
    slots share capacity across all times and modalities, so this is not three
    independent nearest-normal detectors. Reject probability is the only local
    witness used to construct both frame and bag scores.
    """

    def __init__(
        self,
        dims: dict[str, int],
        hidden: int = 128,
        embed: int = 32,
        atoms: int = 8,
        temperature: float = 0.10,
        reject_cost: float = 1.0,
        transport_steps: int = 8,
        pool_power: float = 8.0,
    ):
        super().__init__()
        if tuple(dims) != MODALITIES:
            raise ValueError("dims must follow the frozen modality order")
        if hidden <= 0 or embed <= 0 or atoms <= 0:
            raise ValueError("hidden, embed, and atoms must be positive")
        if temperature <= 0 or reject_cost < 0 or transport_steps <= 0:
            raise ValueError("invalid transport hyperparameters")
        if pool_power <= 0:
            raise ValueError("pool_power must be positive")
        self.modalities = MODALITIES
        self.atoms = int(atoms)
        self.temperature = float(temperature)
        self.reject_cost = float(reject_cost)
        self.transport_steps = int(transport_steps)
        self.pool_power = float(pool_power)
        self.projections = nn.ModuleDict(
            {name: Projection(dims[name], hidden, embed) for name in MODALITIES}
        )
        # One latent normal atom bank is shared by the modality-specific
        # projectors.  This makes atom k a genuine common capacity slot.  With
        # separate [M,K,D] banks, independently permutable atom indices would
        # be coupled only by an arbitrary array index.
        self.references = nn.Parameter(torch.randn(atoms, embed))

    def _typed_cost(
        self,
        feats: dict[str, torch.Tensor],
        reference_gradient_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        refs = F.normalize(self.references, dim=-1)
        costs = []
        for name in self.modalities:
            z = self.projections[name](feats[name])
            if reference_gradient_mask is None:
                similarity = torch.einsum("btd,kd->btk", z, refs)
            else:
                gate = reference_gradient_mask[:, None, None].to(z.dtype)
                per_item_refs = refs.detach()[None] + gate * (
                    refs[None] - refs.detach()[None]
                )
                similarity = torch.einsum("btd,bkd->btk", z, per_item_refs)
            costs.append(1.0 - similarity)
        return torch.stack(costs, dim=2)  # B,T,M,K

    def _transport(
        self, cost: torch.Tensor, mask: torch.Tensor, shared_capacity: bool = True
    ) -> torch.Tensor:
        """Return reject fractions after shared normal-slot capacity projection."""
        batch, length, modalities, atoms = cost.shape
        valid = mask[:, :, None].expand(batch, length, modalities)
        kernel = torch.exp(-cost / self.temperature) * valid[:, :, :, None]
        reject_kernel = torch.full_like(cost[..., :1], self.reject_cost)
        reject_kernel = (
            torch.exp(-reject_kernel / self.temperature) * valid[:, :, :, None]
        )

        # Alternating row normalization and shared column-capacity clipping.
        # Slot capacity sums to the number of valid typed source tokens; reject
        # remains unconstrained, so the rejected fraction is learned from cost.
        if shared_capacity:
            n_valid = valid.sum((1, 2), keepdim=True).clamp(min=1).to(cost.dtype)
            capacity = n_valid[:, :, None] / float(atoms)
            column_scale = torch.ones(
                batch, 1, 1, atoms, dtype=cost.dtype, device=cost.device
            )
            load_dims = (1, 2)
        else:
            n_valid = valid.sum(1, keepdim=True).clamp(min=1).to(cost.dtype)
            capacity = n_valid[:, :, :, None] / float(atoms)
            column_scale = torch.ones(
                batch, 1, modalities, atoms, dtype=cost.dtype, device=cost.device
            )
            load_dims = (1,)
        for _ in range(self.transport_steps):
            weighted = kernel * column_scale
            denom = weighted.sum(-1, keepdim=True) + reject_kernel
            normal = weighted / denom.clamp(min=1e-12)
            load = normal.sum(load_dims, keepdim=True)
            correction = torch.minimum(
                torch.ones_like(load), capacity / load.clamp(min=1e-12)
            )
            column_scale = column_scale * correction
        weighted = kernel * column_scale
        denom = weighted.sum(-1, keepdim=True) + reject_kernel
        reject = reject_kernel / denom.clamp(min=1e-12)
        return reject.squeeze(-1) * valid

    def _score_from_rejection(self, rejected: torch.Tensor, mask: torch.Tensor):
        frame = 1.0 - torch.prod(1.0 - rejected.clamp(0.0, 1.0), dim=2)
        frame = frame * mask
        powered = frame.clamp(min=1e-6).pow(self.pool_power) * mask
        bag = (
            powered.sum(1) / mask.sum(1).clamp(min=1)
        ).clamp(min=1e-12).pow(1.0 / self.pool_power)
        bag = bag.clamp(1e-6, 1.0 - 1e-6)
        score = bag[:, None] * frame
        return score * mask, frame, bag

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        mask: torch.Tensor,
        reference_gradient_mask: torch.Tensor | None = None,
        include_controls: bool = False,
    ):
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean [B,T] tensor")
        if tuple(feats) != self.modalities:
            raise ValueError("feature dictionaries must follow the frozen modality order")
        batch, length = mask.shape
        if torch.any(mask.sum(1) == 0):
            raise ValueError("every video must contain at least one valid second")
        for name in self.modalities:
            if feats[name].ndim != 3 or feats[name].shape[:2] != (batch, length):
                raise ValueError(f"invalid {name} feature shape")
        if reference_gradient_mask is not None:
            if reference_gradient_mask.shape != (batch,):
                raise ValueError("reference_gradient_mask must have shape [B]")
            if reference_gradient_mask.dtype != torch.bool:
                raise ValueError("reference_gradient_mask must be boolean")
        cost = self._typed_cost(feats, reference_gradient_mask)
        rejected = self._transport(cost, mask)
        score, frame, bag = self._score_from_rejection(rejected, mask)
        output = {
            "score": score,
            "frame_witness": frame,
            "bag_probability": bag,
            "rejected_by_modality": rejected,
            "typed_cost": cost,
        }
        if include_controls:
            independent = self._transport(cost, mask, shared_capacity=False)
            independent_score, _, _ = self._score_from_rejection(independent, mask)
            valid = mask[:, :, None].expand_as(cost[..., 0])
            normal_kernel = torch.exp(-cost / self.temperature)
            reject_kernel = torch.exp(
                torch.full_like(cost[..., :1], -self.reject_cost / self.temperature)
            )
            nearest_reject = reject_kernel / (
                normal_kernel.sum(-1, keepdim=True) + reject_kernel
            ).clamp(min=1e-12)
            nearest = nearest_reject.squeeze(-1) * valid
            nearest_score, _, _ = self._score_from_rejection(nearest, mask)
            output["score_independent_transport_control"] = independent_score
            output["score_nearest_normal_control"] = nearest_score
        return output

    def loss(
        self,
        output: dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: torch.Tensor,
        lambda_temporal: float,
        lambda_negative_cost: float,
    ):
        bag = F.binary_cross_entropy(output["bag_probability"], labels)
        pair = mask[:, 1:] & mask[:, :-1]
        difference = (
            output["frame_witness"][:, 1:] - output["frame_witness"][:, :-1]
        ).square()
        temporal = (difference * pair).sum() / pair.sum().clamp(min=1)

        negative = (labels == 0)[:, None, None]
        valid_negative = negative & mask[:, :, None]
        nearest = output["typed_cost"].amin(-1)
        negative_cost = (nearest * valid_negative).sum() / valid_negative.sum().clamp(min=1)
        total = bag + lambda_temporal * temporal + lambda_negative_cost * negative_cost
        return total, {
            "bag": bag.detach(),
            "temporal": temporal.detach(),
            "negative_cost": negative_cost.detach(),
        }
