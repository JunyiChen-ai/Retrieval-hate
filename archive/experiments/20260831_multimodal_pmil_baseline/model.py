"""Tri-modal binary port of the load-bearing P-MIL training structure."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class ProposalBranch(nn.Module):
    def __init__(self, input_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fuse = nn.Sequential(
            nn.Linear(input_dim * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        length = frames.shape[0]
        widths = proposals[:, 1] - proposals[:, 0]
        pad = int(torch.ceil(0.25 * widths.max()).item()) + 1
        padded = torch.cat((
            frames.new_zeros((pad, frames.shape[1])),
            frames,
            frames.new_zeros((pad, frames.shape[1])),
        ), 0)
        start = proposals[:, 0] - 0.25 * widths + pad - 0.5
        end = proposals[:, 1] + 0.25 * widths + pad - 0.5
        start = start.clamp(0, padded.shape[0] - 1)
        end = torch.maximum(end, start + 1e-3).clamp(max=padded.shape[0])
        boxes = torch.stack((
            torch.zeros_like(start), start, torch.ones_like(end), end
        ), 1)
        image = padded.t().unsqueeze(0).unsqueeze(3)
        roi = roi_align(
            image, [boxes], output_size=(self.roi_size, 1),
            spatial_scale=1.0, sampling_ratio=-1, aligned=False,
        )
        return roi.squeeze(3).transpose(1, 2)

    def forward(self, features, proposals, training_sample=False):
        if training_sample and len(proposals) > self.max_train_proposals:
            keep = torch.randperm(len(proposals), device=proposals.device)[
                :self.max_train_proposals
            ]
            proposals = proposals[keep]
        outputs = {
            name: branch(self._roi(features[name], proposals))
            for name, branch in self.branches.items()
        }
        return outputs, proposals

    @staticmethod
    def segment_iou(first, second):
        left = torch.maximum(first[:, None, 0], second[None, :, 0])
        right = torch.minimum(first[:, None, 1], second[None, :, 1])
        intersection = (right - left).clamp(min=0)
        union = (
            first[:, None, 1] - first[:, None, 0]
            + second[None, :, 1] - second[None, :, 0]
            - intersection
        )
        return intersection / union.clamp(min=1e-6)

    @staticmethod
    def _video_classification(cas, attention, label, topk_divisor=8):
        count = max(1, len(cas) // topk_divisor)
        positive = bool(float(label) > 0.5)
        # P-MIL Eq. (8): y_base = [y, 1], i.e. a positive video supervises
        # both foreground and the explicit background category.  This target
        # is intentionally multi-hot rather than a normalized distribution.
        original_target = cas.new_tensor((1.0, 1.0) if positive else (0.0, 1.0))
        suppressed_target = cas.new_tensor((1.0, 0.0) if positive else (0.0, 1.0))
        original = cas.topk(count, dim=0).values.mean(0)
        suppressed = (cas * torch.sigmoid(attention)[:, None]).topk(
            count, dim=0
        ).values.mean(0)
        original_loss = -(original_target * F.log_softmax(original, 0)).sum()
        suppressed_loss = -(suppressed_target * F.log_softmax(suppressed, 0)).sum()
        return original_loss, suppressed_loss

    def _completeness_loss(
        self, predicted, teacher_attention, proposals, gamma, positive
    ):
        if not positive:
            return F.mse_loss(torch.sigmoid(predicted), torch.zeros_like(predicted))
        attention = torch.sigmoid(teacher_attention).detach()
        overlap = self.segment_iou(proposals, proposals) > 0
        retained = attention > gamma * attention.max()
        selected = []
        while bool(retained.any()):
            candidates = torch.where(retained)[0]
            chosen = candidates[attention[candidates].argmax()]
            selected.append(chosen)
            retained[overlap[chosen]] = False
        if not selected:
            selected = [attention.argmax()]
        pseudo = proposals[torch.stack(selected)]
        pseudo_iou = self.segment_iou(proposals, pseudo).amax(1)
        return F.mse_loss(torch.sigmoid(predicted), pseudo_iou)

    def _rank_consistency(self, student, teacher, teacher_attention, proposals):
        overlap = self.segment_iou(proposals, proposals) > 0
        mask = torch.where(overlap, student.new_zeros(()), student.new_full((), -1e3))
        teacher_distribution = F.softmax(mask + teacher[None, :], 1).detach()
        student_distribution = F.log_softmax(mask + student[None, :], 1)
        per_anchor = F.kl_div(
            student_distribution, teacher_distribution, reduction="none"
        ).sum(1)
        retained = torch.sigmoid(teacher_attention) > torch.sigmoid(
            teacher_attention
        ).mean()
        if not bool(retained.any()):
            return per_anchor.mean() * 0.0
        return per_anchor[retained].mean()

    def loss(
        self, outputs, proposals, label, epoch, rampup=10, gamma=0.8,
        completeness_weight=20.0, consistency_weight=2.0,
    ):
        names = self.modalities
        positive = bool(float(label) > 0.5)
        original, suppressed, completeness = [], [], []
        for name in names:
            first, second = self._video_classification(
                outputs[name]["cas"], outputs[name]["attention"], label
            )
            original.append(first)
            suppressed.append(second)
            other_attention = torch.stack([
                outputs[other]["attention"] for other in names if other != name
            ]).mean(0)
            completeness.append(self._completeness_loss(
                outputs[name]["completeness"], other_attention, proposals,
                gamma, positive,
            ))

        consistency = []
        if positive:
            for student_name, teacher_name in itertools.permutations(names, 2):
                consistency.append(self._rank_consistency(
                    outputs[student_name]["cas"][:, 0],
                    outputs[teacher_name]["cas"][:, 0],
                    outputs[teacher_name]["attention"], proposals,
                ))
        zero = next(iter(outputs.values()))["cas"].sum() * 0.0
        consistency_loss = torch.stack(consistency).mean() if consistency else zero
        phase = 1.0 - min(float(epoch), float(rampup)) / max(float(rampup), 1.0)
        ramp = float(torch.exp(torch.tensor(-5.0 * phase * phase)))
        terms = {
            "mil_original": torch.stack(original).mean(),
            "mil_suppressed": torch.stack(suppressed).mean(),
            "rank_consistency": consistency_loss * ramp * consistency_weight,
            "completeness": (
                torch.stack(completeness).mean() * ramp * completeness_weight
            ),
        }
        terms["total"] = sum(terms.values())
        return terms

    def scores(self, outputs):
        proposal_scores, video_scores = [], []
        for name in self.modalities:
            hate = F.softmax(outputs[name]["cas"], 1)[:, 0]
            attention = torch.sigmoid(outputs[name]["attention"])
            completeness = torch.sigmoid(outputs[name]["completeness"])
            proposal_scores.append(hate * attention * completeness)
            video_scores.append(
                (hate * attention).sum() / attention.sum().clamp(min=1e-6)
            )
        return torch.stack(proposal_scores).mean(0), torch.stack(video_scores).mean()
