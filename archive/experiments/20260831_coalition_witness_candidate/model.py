"""Coalition-witness models and matched controls for the minimal pilot."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITIES = ("visual", "audio", "text")
SUBSETS = tuple(range(1, 8))
FULL_SUBSET_INDEX = SUBSETS.index(7)
MISSING_ONE_INDICES = tuple(SUBSETS.index(value) for value in (6, 5, 3))


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, hidden=256, embed=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed),
            nn.ReLU(inplace=True),
        )

    def forward(self, values):
        return self.net(values)


def topk_counts(lengths, proportion=3):
    return torch.clamp(torch.ceil(lengths.float() / float(proportion)), min=1).long()


def topk_mean(values, mask, counts):
    filled = values.masked_fill(~mask, -1e9)
    order = torch.argsort(filled, dim=1, descending=True)
    picked = torch.gather(values, 1, order)
    ranks = torch.arange(values.shape[1], device=values.device)[None, :]
    keep = ranks < counts[:, None]
    return (picked * keep).sum(1) / counts.to(values.dtype).clamp(min=1)


def smoothness_loss(frame_logits, mask):
    probability = torch.sigmoid(frame_logits)
    pair = mask[:, 1:] & mask[:, :-1]
    if not bool(pair.any()):
        return probability.sum() * 0.0
    delta = (probability[:, 1:] - probability[:, :-1]).square()
    return (delta * pair).sum() / pair.sum().to(delta.dtype)


class CoalitionModel(nn.Module):
    """One shared subset scorer; strict arms read its outputs as Möbius atoms."""

    def __init__(self, dims, hidden=256, embed=128, dropout=0.1,
                 temperature=0.25, k_proportion=3, synib_margin=0.10):
        super().__init__()
        if tuple(dims) != MODALITIES:
            raise ValueError(f"modality order must be {MODALITIES}, got {tuple(dims)}")
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(dims[name], hidden, embed, dropout)
            for name in MODALITIES
        })
        self.coalition_head = nn.Sequential(
            nn.Linear(embed * len(MODALITIES) + len(MODALITIES), hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed),
            nn.ReLU(inplace=True),
            nn.Linear(embed, 1),
        )
        self.temperature = float(temperature)
        self.k_proportion = int(k_proportion)
        self.synib_margin = float(synib_margin)

    def coalition_logits(self, feats):
        embeds = {name: self.encoders[name](feats[name]) for name in MODALITIES}
        reference = embeds[MODALITIES[0]]
        batch, steps = reference.shape[:2]
        outputs = []
        for subset in SUBSETS:
            pieces = []
            availability = []
            for slot, name in enumerate(MODALITIES):
                present = bool(subset & (1 << slot))
                pieces.append(embeds[name] if present else torch.zeros_like(embeds[name]))
                availability.append(float(present))
            bits = reference.new_tensor(availability).view(1, 1, -1)
            bits = bits.expand(batch, steps, -1)
            outputs.append(self.coalition_head(torch.cat(pieces + [bits], dim=-1)).squeeze(-1))
        return torch.stack(outputs, dim=-1)

    def reconstructed_full_logits(self, atom_logits):
        # tau * log(mean_R h_t(R)), h_t(R)=exp(a_t(R)/tau). This is one
        # monotone rescaling of the exact full coalition worth sum_R h_t(R).
        tau = self.temperature
        return tau * (
            torch.logsumexp(atom_logits / tau, dim=-1) - math.log(len(SUBSETS))
        )

    def temporal_bag_probability(self, frame_logits, mask, lengths):
        counts = topk_counts(lengths, self.k_proportion)
        return topk_mean(torch.sigmoid(frame_logits), mask, counts).clamp(1e-7, 1 - 1e-7)

    def latent_bag_logits(self, atom_logits, mask, lengths):
        tau = self.temperature
        valid = mask.unsqueeze(-1).expand_as(atom_logits)
        scaled = (atom_logits / tau).masked_fill(~valid, -torch.inf)
        normalizer = torch.log(lengths.to(atom_logits.dtype) * len(SUBSETS))
        return tau * (torch.logsumexp(scaled.flatten(1), dim=1) - normalizer)

    def forward(self, feats, mask, lengths, arm):
        coalition = self.coalition_logits(feats)
        if arm in ("mobius_nonminimal", "coalition_witness"):
            frame_logits = self.reconstructed_full_logits(coalition)
        elif arm in ("all_subset_mil", "synib"):
            frame_logits = coalition[:, :, FULL_SUBSET_INDEX]
        else:
            raise ValueError(f"unknown arm {arm}")

        output = {
            "coalition_logits": coalition,
            "frame_logits": frame_logits,
            "frame_scores": torch.sigmoid(frame_logits) * mask,
        }
        if arm == "coalition_witness":
            output["video_logits"] = self.latent_bag_logits(coalition, mask, lengths)
        else:
            output["video_scores"] = self.temporal_bag_probability(frame_logits, mask, lengths)
        return output

    def loss(self, output, labels, mask, lengths, arm, lambda_smooth=0.1):
        coalition = output["coalition_logits"]
        if arm == "coalition_witness":
            primary = F.binary_cross_entropy_with_logits(output["video_logits"], labels)
        elif arm == "mobius_nonminimal":
            primary = F.binary_cross_entropy(output["video_scores"], labels)
        elif arm == "all_subset_mil":
            counts = topk_counts(lengths, self.k_proportion)
            terms = []
            for index in range(len(SUBSETS)):
                probability = topk_mean(
                    torch.sigmoid(coalition[:, :, index]), mask, counts
                ).clamp(1e-7, 1 - 1e-7)
                terms.append(F.binary_cross_entropy(probability, labels))
            primary = torch.stack(terms).mean()
        elif arm == "synib":
            full = output["video_scores"]
            primary = F.binary_cross_entropy(full, labels)
            counts = topk_counts(lengths, self.k_proportion)
            missing = []
            for index in MISSING_ONE_INDICES:
                missing.append(topk_mean(
                    torch.sigmoid(coalition[:, :, index]), mask, counts
                ))
            missing = torch.stack(missing, dim=1)
            penalty = F.relu(missing - full[:, None] + self.synib_margin)
            positive_count = labels.sum().clamp(min=1.0)
            primary = primary + (penalty.mean(1) * labels).sum() / positive_count
        else:
            raise ValueError(f"unknown arm {arm}")
        smooth = smoothness_loss(output["frame_logits"], mask)
        total = primary + float(lambda_smooth) * smooth
        return total, {"primary": primary, "smooth": smooth}

    def posterior_summary(self, atom_logits, length):
        values = atom_logits[:length]
        posterior = torch.softmax((values / self.temperature).reshape(-1), dim=0)
        posterior = posterior.reshape(length, len(SUBSETS))
        mass = posterior.sum(0)
        winner = int(torch.argmax(posterior).item())
        return {
            "coalition_posterior_mass": [float(value) for value in mass.cpu()],
            "map_second": int(winner // len(SUBSETS)),
            "map_subset": int(SUBSETS[winner % len(SUBSETS)]),
            "atom_logit_mean": [float(value) for value in values.mean(0).cpu()],
            "atom_logit_max": [float(value) for value in values.max(0).values.cpu()],
        }
