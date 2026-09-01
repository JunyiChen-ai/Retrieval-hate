"""Training-only policy-constrained snippet cluster transport."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# background, targeted, abuse, violence, sexual, self-harm, protected context
STATE_TARGETS = torch.tensor([
    [0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
], dtype=torch.float32)

HARMFUL = {"hatemm": (1,), "hateclipseg": (1, 2, 3, 4, 5)}


class PolicyClusterTransport(nn.Module):
    """CASE-style cluster self-labeling with policy-feasible state mass."""

    def __init__(self, dim: int, temperature: float,
                 min_harmful_mass: float = .10):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(7, dim) * (dim ** -.5))
        self.abstain_logit = nn.Parameter(torch.tensor(0.0))
        self.temperature = float(temperature)
        self.min_harmful_mass = float(min_harmful_mass)

    @staticmethod
    def _raise_harmful_mass(q, harmful, valid, minimum):
        """KL-like projection onto an aggregate harmful-mass lower bound."""
        out = q.clone()
        for i in range(len(out)):
            use = valid[i]
            n = int(use.sum())
            if n == 0:
                continue
            hidx = torch.as_tensor(harmful, device=q.device)
            current = out[i, use][:, hidx].sum() / n
            if current >= minimum:
                continue
            delta = (minimum - current).clamp_min(0)
            rows = out[i, use]
            h = rows[:, hidx]
            hshape = h / h.sum(-1, keepdim=True).clamp_min(1e-6)
            empty = h.sum(-1, keepdim=True) <= 1e-6
            uniform = torch.full_like(hshape, 1.0 / len(harmful))
            hshape = torch.where(empty, uniform, hshape)
            non_idx = torch.as_tensor(
                [j for j in range(8) if j not in harmful], device=q.device)
            non = rows[:, non_idx]
            removable = non.sum(-1, keepdim=True).clamp_min(1e-6)
            add = torch.minimum(delta.expand_as(removable), removable * .95)
            rows[:, hidx] = h + add * hshape
            rows[:, non_idx] = non * ((removable - add) / removable)
            out[i, use] = rows / rows.sum(-1, keepdim=True).clamp_min(1e-6)
        return out

    def forward(self, shared, primitive_logits, frame_prob, valid, labels,
                corpus: str, arm: str):
        if corpus not in HARMFUL:
            raise KeyError(corpus)
        if arm not in ("policy", "binary", "permuted"):
            raise ValueError(arm)
        unit_x = F.normalize(shared, dim=-1, eps=1e-6)
        unit_p = F.normalize(self.prototypes, dim=-1, eps=1e-6)
        logits7 = torch.einsum("btd,kd->btk", unit_x, unit_p)
        abstain = self.abstain_logit.expand(*logits7.shape[:-1], 1)
        logits = torch.cat([logits7, abstain], -1)
        # The constrained, temperature-sharpened distribution is the detached
        # transport target.  The prediction side deliberately retains every
        # state and uses untempered logits: using the identical masked softmax
        # on both sides makes CE(p.detach(), p) have zero data gradient and
        # prevents both snippet clustering and negative-background anchoring.
        assignment_logits = logits / self.temperature
        allowed = torch.zeros_like(assignment_logits, dtype=torch.bool)
        allowed[..., 0] = True
        allowed[..., 7] = True
        harmful = HARMFUL[corpus]
        allowed[..., 6] = True
        if arm == "binary":
            allowed[..., 1:6] = True
        else:
            allowed[..., list(harmful)] = True
        neg = labels <= .5
        allowed[neg] = False
        allowed[neg, :, 0] = True
        masked = assignment_logits.masked_fill(~allowed, -30.0)
        q = torch.softmax(masked, -1)
        positive = labels > .5
        selected = harmful if arm != "binary" else tuple(range(1, 6))
        if positive.any():
            projected = self._raise_harmful_mass(
                q[positive], selected, valid[positive], self.min_harmful_mass)
            q = q.clone()
            q[positive] = projected
        q = q.detach()
        per_frame_cluster = -(q * torch.log_softmax(logits, -1)).sum(-1)
        cluster_loss = ((per_frame_cluster * valid).sum()
                        / valid.sum().clamp_min(1))

        targets = STATE_TARGETS.to(shared.device)
        if arm == "permuted":
            targets = targets.clone()
            targets[1:7] = targets[1:7].roll(1, dims=0)
        if arm == "binary":
            target = q[..., 1:6].sum(-1)
            elem = F.binary_cross_entropy(
                frame_prob.clamp(1e-5, 1 - 1e-5), target,
                reduction="none")
            semantic_loss = (elem * valid).sum() / valid.sum().clamp_min(1)
        else:
            target = torch.einsum("bts,sk->btk", q[..., :7], targets)
            weight = (1.0 - q[..., 7]) * valid
            elem = F.binary_cross_entropy_with_logits(
                primitive_logits, target, reduction="none").mean(-1)
            semantic_loss = (elem * weight).sum() / weight.sum().clamp_min(1)
        proto_sim = unit_p @ unit_p.T
        eye = torch.eye(7, device=shared.device)
        diversity = ((proto_sim - eye) ** 2).mean()
        loss = cluster_loss + semantic_loss + .01 * diversity
        stats = {
            "cluster": float(cluster_loss.detach()),
            "semantic": float(semantic_loss.detach()),
            "harmful_mass": float(
                (q[..., list(selected)].sum(-1) * valid).sum().detach()
                / valid.sum().clamp_min(1)),
            "abstain_mass": float(
                (q[..., 7] * valid).sum().detach()
                / valid.sum().clamp_min(1)),
        }
        return loss, stats
