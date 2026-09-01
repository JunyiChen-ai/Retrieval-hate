"""Load-bearing operators shared by this experiment's checks and model."""

from __future__ import annotations

import torch


def masked_logmeanexp(logits: torch.Tensor, valid: torch.Tensor,
                      dim: int = -1) -> torch.Tensor:
    """Log-mean-exp over valid evidence channels only.

    ``valid`` must have exactly the same shape as ``logits``.  An item with no
    valid evidence is an input error rather than a learnable missingness cue.
    """
    if logits.shape != valid.shape:
        raise ValueError("logits and valid masks must have identical shape")
    counts = valid.sum(dim=dim)
    if torch.any(counts == 0):
        raise ValueError("masked_logmeanexp received an item with no evidence")
    masked = logits.masked_fill(~valid, -torch.inf)
    return torch.logsumexp(masked, dim=dim) - counts.to(logits.dtype).log()
