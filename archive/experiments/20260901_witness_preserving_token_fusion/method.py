from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MultiHateLoc, topk_counts, topk_mask


ARMS = ("anchor", "aligned", "shifted")


class WitnessPreservingTokenFusion(MultiHateLoc):
    def __init__(self, dims, alpha_fusion, arm, **kwargs):
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        if arm == "anchor" and alpha_fusion != 0.0:
            raise ValueError("anchor requires alpha_fusion=0")
        if arm != "anchor" and not 0.0 < alpha_fusion <= 1.0:
            raise ValueError("result-relevant arms require alpha_fusion in (0,1]")
        super().__init__(dims, **kwargs)
        self.alpha_fusion = float(alpha_fusion)
        self.arm = arm
        self.retain = None
        self.projections = None
        if self.alpha_fusion > 0.0:
            embed = self.branches[self.modalities[0]].head.in_features
            self.retain = nn.ModuleDict({
                name: nn.Linear(embed, 1) for name in self.modalities})
            self.projections = nn.ModuleDict({
                f"{source}_to_{target}": nn.Linear(embed, embed, bias=False)
                for target in self.modalities for source in self.modalities
                if source != target})

    @staticmethod
    def _shift_valid(sequence, mask):
        shifted = sequence.clone()
        for index in range(sequence.shape[0]):
            length = int(mask[index].sum().item())
            if length <= 1:
                continue
            offset = max(1, length // 2)
            shifted[index, :length] = torch.roll(
                sequence[index, :length], shifts=offset, dims=0)
        return shifted

    def _substitute(self, embeds, mask):
        by_name = {name: embeds[index]
                   for index, name in enumerate(self.modalities)}
        donor_source = ({name: self._shift_valid(value, mask)
                         for name, value in by_name.items()}
                        if self.arm == "shifted" else by_name)
        gates, substituted = {}, []
        scale = math.sqrt(float(embeds[0].shape[-1]))
        for target in self.modalities:
            recipient = by_name[target]
            gate = torch.sigmoid(self.retain[target](recipient)).squeeze(-1)
            gates[target] = gate
            donors, logits = [], []
            for source in self.modalities:
                if source == target:
                    continue
                donor = self.projections[f"{source}_to_{target}"](
                    donor_source[source])
                donors.append(donor)
                logits.append((recipient * donor).sum(-1) / scale)
            donor_weights = F.softmax(torch.stack(logits, dim=-1), dim=-1)
            donor_stack = torch.stack(donors, dim=-2)
            replacement = (donor_stack * donor_weights[..., None]).sum(-2)
            amount = self.alpha_fusion * (1.0 - gate)
            value = recipient + amount[..., None] * (replacement - recipient)
            substituted.append(value)
        return substituted, torch.stack(
            [gates[name] for name in self.modalities], dim=-1)

    def forward(self, feats, mask):
        base = super().forward(feats, mask)
        if self.alpha_fusion == 0.0:
            return base
        substituted, retain_gates = self._substitute(base["embeds"], mask)
        weights = base["weights"]
        n_modalities = len(self.modalities)
        scaled = [value * (weights[:, index] * n_modalities)[:, None, None]
                  for index, value in enumerate(substituted)]
        fused_embed = self.fuse(torch.cat(scaled, dim=-1))
        fused_probability = torch.sigmoid(
            self.fuse_head(fused_embed).squeeze(-1)) * mask
        probabilities = dict(base["probs"])
        probabilities["fused"] = fused_probability
        base.update({"probs": probabilities, "fused_embed": fused_embed,
                     "retain_gates": retain_gates,
                     "substituted_embeds": substituted})
        return base

    def gate_loss(self, output, mask, lengths, labels,
                  retain_budget=0.5):
        zero = output["probs"]["fused"].sum() * 0.0
        if self.alpha_fusion == 0.0:
            return zero, {"coverage": zero, "budget": zero,
                          "witness_count": 0, "outside_count": 0}
        gates = output["retain_gates"]
        counts = topk_counts(lengths, self.k_proportion)
        witness = topk_mask(
            output["probs"]["fused"].detach(), mask, counts)
        witness = witness & (labels >= 0.5)[:, None]
        outside = mask & ~witness
        coverage_frame = F.relu(1.0 - gates.sum(-1)).pow(2)
        coverage = ((coverage_frame * witness).sum() /
                    witness.sum().clamp(min=1).to(gates.dtype))
        mean_retain = gates.mean(-1)
        budget_frame = (mean_retain - retain_budget).pow(2)
        budget = ((budget_frame * outside).sum() /
                  outside.sum().clamp(min=1).to(gates.dtype))
        return coverage + budget, {
            "coverage": coverage, "budget": budget,
            "witness_count": int(witness.sum().item()),
            "outside_count": int(outside.sum().item()),
        }
