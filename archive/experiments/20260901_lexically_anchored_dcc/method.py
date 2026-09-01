"""Training-only asymmetric cross-video region memory for POWA."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def contiguous_region_means(values: torch.Tensor, mask: torch.Tensor,
                            video_index: torch.Tensor,
                            negative_width: int = 8):
    """Return region means and source-video ids for one admission class."""
    regions, ids = [], []
    for b in range(values.shape[0]):
        valid = torch.nonzero(mask[b], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        # Split at every gap, preserving true contiguous lexical regions.
        cuts = torch.nonzero(valid[1:] != valid[:-1] + 1,
                             as_tuple=False).flatten() + 1
        groups = torch.tensor_split(valid, cuts.tolist())
        for group in groups:
            # Certified-negative support can span a full video. Fixed chunks
            # prevent long negative bags from collapsing to one region.
            for start in range(0, group.numel(), negative_width):
                part = group[start:start + negative_width]
                if part.numel():
                    regions.append(values[b, part].mean(0))
                    ids.append(video_index[b])
    if not regions:
        return (values.new_zeros((0, values.shape[-1])),
                video_index.new_zeros((0,)))
    return torch.stack(regions), torch.stack(ids)


class CrossVideoRegionMemory(nn.Module):
    """FIFO class-region memory with multi-positive InfoNCE.

    Memory entries are detached past-video shared representations. Same-video
    entries are excluded from each anchor, so the positive relation is truly
    cross-video. The module has no inference readout.
    """

    def __init__(self, dim: int, capacity: int = 512,
                 temperature: float = 0.1, negative_width: int = 8):
        super().__init__()
        self.capacity = int(capacity)
        self.temperature = float(temperature)
        self.negative_width = int(negative_width)
        self.register_buffer("hate_memory", torch.zeros(self.capacity, dim),
                             persistent=False)
        self.register_buffer("hate_ids", torch.full((self.capacity,), -1,
                                                     dtype=torch.long),
                             persistent=False)
        self.register_buffer("benign_memory", torch.zeros(self.capacity, dim),
                             persistent=False)
        self.register_buffer("benign_ids", torch.full((self.capacity,), -1,
                                                       dtype=torch.long),
                             persistent=False)
        self.register_buffer("hate_count", torch.zeros((), dtype=torch.long),
                             persistent=False)
        self.register_buffer("benign_count", torch.zeros((), dtype=torch.long),
                             persistent=False)

    @staticmethod
    def _support_mask(evidence, speech, valid, labels, quantile):
        support = torch.zeros_like(valid)
        for b in range(valid.shape[0]):
            if labels[b] < 0.5:
                continue
            eligible = valid[b] & (speech[b] > 0)
            values = evidence[b, eligible]
            if values.numel() == 0:
                continue
            threshold = torch.quantile(values.detach(), quantile)
            support[b] = eligible & (evidence[b] >= threshold)
        return support

    def _current_memory(self, kind):
        count = int(getattr(self, f"{kind}_count").item())
        return (getattr(self, f"{kind}_memory")[:count],
                getattr(self, f"{kind}_ids")[:count])

    @torch.no_grad()
    def _enqueue(self, kind, values, ids):
        if values.numel() == 0:
            return
        values = F.normalize(values.detach(), dim=-1)
        memory = getattr(self, f"{kind}_memory")
        memory_ids = getattr(self, f"{kind}_ids")
        count_buf = getattr(self, f"{kind}_count")
        count = int(count_buf.item())
        old_values = memory[:count]
        old_ids = memory_ids[:count]
        merged_values = torch.cat([old_values, values], 0)[-self.capacity:]
        merged_ids = torch.cat([old_ids, ids.detach()], 0)[-self.capacity:]
        new_count = merged_values.shape[0]
        memory.zero_()
        memory_ids.fill_(-1)
        memory[:new_count].copy_(merged_values)
        memory_ids[:new_count].copy_(merged_ids)
        count_buf.fill_(new_count)

    def _class_loss(self, anchors, anchor_ids, positive_kind,
                    current_hate, current_hate_ids,
                    current_benign, current_benign_ids):
        if anchors.numel() == 0:
            return anchors.sum() * 0.0
        old_hate, old_hate_ids = self._current_memory("hate")
        old_benign, old_benign_ids = self._current_memory("benign")
        hate = torch.cat([old_hate, current_hate.detach()], 0)
        hate_ids = torch.cat([old_hate_ids, current_hate_ids], 0)
        benign = torch.cat([old_benign, current_benign.detach()], 0)
        benign_ids = torch.cat([old_benign_ids, current_benign_ids], 0)
        candidates = torch.cat([hate, benign], 0)
        candidate_ids = torch.cat([hate_ids, benign_ids], 0)
        if candidates.numel() == 0:
            return anchors.sum() * 0.0
        positive_class = torch.cat([
            torch.ones(hate.shape[0], device=anchors.device, dtype=torch.bool),
            torch.zeros(benign.shape[0], device=anchors.device, dtype=torch.bool),
        ])
        if positive_kind == "benign":
            positive_class = ~positive_class
        logits = F.normalize(anchors, dim=-1) @ candidates.T
        logits = logits / self.temperature
        losses = []
        for i in range(anchors.shape[0]):
            cross_video = candidate_ids != anchor_ids[i]
            positive = positive_class & cross_video
            admissible = cross_video
            if positive.any() and admissible.any():
                numerator = torch.logsumexp(logits[i, positive], 0)
                denominator = torch.logsumexp(logits[i, admissible], 0)
                losses.append(denominator - numerator)
        return (torch.stack(losses).mean() if losses
                else anchors.sum() * 0.0)

    def forward(self, shared, evidence, speech, valid, labels, video_index,
                support_quantile: float):
        support = self._support_mask(
            evidence, speech, valid, labels, support_quantile)
        certified_benign = valid & (labels[:, None] < 0.5)
        hate_regions, hate_ids = contiguous_region_means(
            shared, support, video_index, negative_width=10**9)
        benign_regions, benign_ids = contiguous_region_means(
            shared, certified_benign, video_index,
            negative_width=self.negative_width)
        hate_loss = self._class_loss(
            hate_regions, hate_ids, "hate", hate_regions, hate_ids,
            benign_regions, benign_ids)
        benign_loss = self._class_loss(
            benign_regions, benign_ids, "benign", hate_regions, hate_ids,
            benign_regions, benign_ids)
        loss = 0.5 * (hate_loss + benign_loss)
        with torch.no_grad():
            self._enqueue("hate", hate_regions, hate_ids)
            self._enqueue("benign", benign_regions, benign_ids)
        stats = {
            "hate_regions": int(hate_regions.shape[0]),
            "benign_regions": int(benign_regions.shape[0]),
            "supported_frames": int(support.sum().item()),
            "hate_memory": int(self.hate_count.item()),
            "benign_memory": int(self.benign_count.item()),
        }
        return loss, stats
