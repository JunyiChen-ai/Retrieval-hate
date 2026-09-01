"""Shared proposal generation, P-MIL inference, and interval readout utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import roi_align


def contiguous_components(mask):
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    return list(zip(np.where(changes == 1)[0], np.where(changes == -1)[0]))


def generate_proposals(score, maximum=256):
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1 or not len(score) or not np.isfinite(score).all():
        raise RuntimeError("invalid proposal-producer score sequence")
    low, high = float(score.min()), float(score.max())
    proposals = {(0, len(score))}
    for level in np.linspace(0.1, 0.9, 9):
        proposals.update(contiguous_components(score >= low + level * (high - low)))
    for peak in np.argsort(-score, kind="stable")[:min(16, len(score))]:
        for width in (1, 2, 4, 8, 16, 32, 64):
            width = min(width, len(score))
            start = min(max(0, int(peak) - width // 2), len(score) - width)
            proposals.add((start, start + width))
    proposals = [p for p in proposals if p[1] > p[0]]

    def priority(bound):
        values = score[bound[0]:bound[1]]
        return float(values.max()), float(values.mean()), -(bound[1] - bound[0]), -bound[0]

    proposals.sort(key=priority, reverse=True)
    whole = (0, len(score))
    if whole not in proposals[:maximum]:
        proposals = proposals[:maximum - 1] + [whole]
    else:
        proposals = proposals[:maximum]
    return np.asarray(sorted(set(proposals)), dtype=np.float32)


class ProposalBranch(nn.Module):
    def __init__(self, input_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fuse = nn.Sequential(
            nn.Linear(input_dim * 3, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 2)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )
        self.completeness = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, roi):
        roi = self.norm(roi)
        size = roi.shape[1]
        edge = max(1, size // 6)
        left = roi[:, :edge].amax(1)
        inside = roi[:, edge:size - edge].amax(1)
        right = roi[:, size - edge:].amax(1)
        fused = self.fuse(torch.cat((inside - left, inside, inside - right), 1))
        return {
            "cas": self.classifier(fused),
            "attention": self.attention(fused).squeeze(1),
            "completeness": self.completeness(fused).squeeze(1),
        }


class MultimodalPMIL(nn.Module):
    """Architecture-compatible inference form of the frozen P-MIL baseline."""

    def __init__(self, dims, hidden=128, roi_size=12, dropout=0.1,
                 max_train_proposals=128):
        super().__init__()
        self.modalities = tuple(dims)
        self.roi_size = int(roi_size)
        self.max_train_proposals = int(max_train_proposals)
        self.branches = nn.ModuleDict({
            name: ProposalBranch(dim, hidden, dropout) for name, dim in dims.items()
        })

    def _roi(self, frames, proposals):
        widths = proposals[:, 1] - proposals[:, 0]
        pad = int(torch.ceil(0.25 * widths.max()).item()) + 1
        padded = torch.cat((
            frames.new_zeros((pad, frames.shape[1])), frames,
            frames.new_zeros((pad, frames.shape[1])),
        ), 0)
        start = (proposals[:, 0] - 0.25 * widths + pad - 0.5).clamp(
            0, padded.shape[0] - 1
        )
        end = proposals[:, 1] + 0.25 * widths + pad - 0.5
        end = torch.maximum(end, start + 1e-3).clamp(max=padded.shape[0])
        boxes = torch.stack((torch.zeros_like(start), start, torch.ones_like(end), end), 1)
        image = padded.t().unsqueeze(0).unsqueeze(3)
        roi = roi_align(
            image, [boxes], output_size=(self.roi_size, 1), spatial_scale=1.0,
            sampling_ratio=-1, aligned=False,
        )
        return roi.squeeze(3).transpose(1, 2)

    def forward(self, features, proposals, training_sample=False):
        if training_sample and len(proposals) > self.max_train_proposals:
            keep = torch.randperm(len(proposals), device=proposals.device)[
                :self.max_train_proposals
            ]
            proposals = proposals[keep]
        return {
            name: self.branches[name](self._roi(features[name], proposals))
            for name in self.modalities
        }, proposals

    def full_scores(self, outputs):
        scores = []
        for name in self.modalities:
            out = outputs[name]
            hate = torch.softmax(out["cas"], 1)[:, 0]
            attention = torch.sigmoid(out["attention"])
            completeness = torch.sigmoid(out["completeness"])
            scores.append(hate * attention * completeness)
        return torch.stack(scores).mean(0)


def path_size(proposals, length):
    proposals = np.asarray(proposals)
    if proposals.ndim != 2 or proposals.shape[1] != 2 or len(proposals) == 0:
        raise RuntimeError("invalid proposals")
    if not np.equal(proposals, np.floor(proposals)).all():
        raise RuntimeError("proposal bounds must be integer-valued")
    bounds = proposals.astype(np.int64)
    if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] > length):
        raise RuntimeError("proposal outside video")
    widths = bounds[:, 1] - bounds[:, 0]
    if np.any(widths <= 0):
        raise RuntimeError("empty proposal")
    occupancy = np.zeros(length, dtype=np.float64)
    for start, end in bounds:
        occupancy[start:end] += 1.0
    values = np.asarray([
        np.mean(1.0 / occupancy[start:end]) for start, end in bounds
    ])
    if np.any(values <= 0) or not np.isfinite(values).all():
        raise RuntimeError("invalid path size")
    return values


def choice_readout(proposals, utilities, length, beta):
    raw_proposals = np.asarray(proposals)
    utilities = np.asarray(utilities, dtype=np.float64)
    if utilities.ndim != 1 or len(utilities) != len(raw_proposals):
        raise RuntimeError("proposal/utility count mismatch")
    if not np.isfinite(utilities).all() or not np.isfinite(float(beta)):
        raise RuntimeError("non-finite choice utility or beta")
    # Validate the original bounds before conversion; otherwise fractional
    # endpoints would be silently truncated and change both occupancy and IoU.
    ps = path_size(raw_proposals, length)
    proposals = raw_proposals.astype(np.int64)
    adjusted = utilities + float(beta) * np.log(ps)
    shifted = adjusted - adjusted.max()
    weights = np.exp(shifted)
    posterior = weights / weights.sum()
    frames = np.zeros(length, dtype=np.float64)
    for (start, end), probability in zip(proposals, posterior):
        frames[start:end] += probability
    return frames, posterior, float(np.log(np.exp(shifted).sum()) + adjusted.max()), ps
