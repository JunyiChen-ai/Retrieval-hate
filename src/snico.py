"""Hard-boundary snippet contrast (SniCo), after CoLA (Zhang et al., CVPR 2021).

CoLA mines, inside each positive bag, the rows just outside the predicted
foreground mask (hard background: dilate(M) - M) and the rows just inside its
edge (hard foreground: M - erode(M)), then pulls hard foreground towards the
easy foreground and away from the easy background with an InfoNCE, and the
hard background the other way round. The mask here is rank based (the top
``rho`` fraction of rows by actionness) rather than threshold based, because
hateful spans are dense and a threshold mask can cover the whole video.

Everything is batched over the padded (B, T) layout MACIL-SD uses; ``seq_len``
gives the number of real rows per item.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _morph(mask, m, mode):
    """1-D erosion/dilation of a binary (T,) mask with a length-m window."""
    if m <= 1:
        return mask
    x = mask.view(1, 1, -1).float()
    pad_l = m // 2
    pad_r = m - 1 - pad_l
    if mode == "dilate":
        y = F.max_pool1d(F.pad(x, (pad_l, pad_r), value=0.0), m, stride=1)
    else:
        y = -F.max_pool1d(F.pad(-x, (pad_l, pad_r), value=-1.0), m, stride=1)
    return (y.view(-1) > 0.5)


def _info_nce(query, positive, negatives, tau):
    """query (Q, D), positive (D,) or (Q, D), negatives (N, D); unit vectors."""
    if positive.dim() == 1:
        positive = positive.unsqueeze(0).expand_as(query)
    pos_logit = (query * positive).sum(-1, keepdim=True)          # (Q, 1)
    neg_logit = query @ negatives.t()                              # (Q, N)
    logits = torch.cat([pos_logit, neg_logit], dim=1) / tau
    target = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, target)


def snico_loss(actionness, emb, seq_len, labels, rho, m, tau, k_fn=None):
    """Returns (loss_fg, loss_bg, n_pos_used) for one batch.

    actionness  (B, T)  detached probabilities used only for mining
    emb         (B, T, D) L2-normalised row embeddings (gradient flows)
    seq_len     (B,)    real rows per item
    labels      (B,)    video labels in {0, 1}
    rho         fraction of a positive video's rows taken as foreground mask
    m           erosion/dilation window in rows
    tau         InfoNCE temperature
    k_fn        rows -> k for easy sets; default MACIL-SD's T // 16 + 1
    """
    if k_fn is None:
        k_fn = lambda t: int(t // 16 + 1)
    B = emb.shape[0]
    device = emb.device
    act = actionness.detach()
    pos_items = [i for i in range(B) if float(labels[i]) > 0.5]
    neg_items = [i for i in range(B) if float(labels[i]) <= 0.5]
    if not pos_items:
        zero = emb.sum() * 0.0
        return zero, zero, 0

    # Background rows contributed by the negative videos of this batch.
    neg_pool = []
    for i in neg_items:
        t = int(seq_len[i])
        if t <= 0:
            continue
        k = min(k_fn(t), t)
        idx = torch.randperm(t, device=device)[:k]
        neg_pool.append(emb[i, idx])
    neg_pool = torch.cat(neg_pool, 0) if neg_pool else None

    fg_terms, bg_terms, used = [], [], 0
    for i in pos_items:
        t = int(seq_len[i])
        if t < 4:
            continue
        k = min(k_fn(t), t)
        a = act[i, :t]
        e = emb[i, :t]
        n_fg = int(math.ceil(rho * t))
        n_fg = min(max(n_fg, k + 1), t - 1)
        order = torch.argsort(a, descending=True)
        mask = torch.zeros(t, dtype=torch.bool, device=device)
        mask[order[:n_fg]] = True
        hard_fg = mask & ~_morph(mask, m, "erode")
        hard_bg = _morph(mask, m, "dilate") & ~mask
        easy_fg_rows = e[order[:k]]                       # (k, D)
        easy_fg = F.normalize(easy_fg_rows.mean(0), dim=0)
        easy_bg_rows = e[order[-k:]]                      # lowest-k of same video
        if neg_pool is not None:
            easy_bg_rows = torch.cat([easy_bg_rows, neg_pool], 0)
        easy_bg = F.normalize(easy_bg_rows.mean(0), dim=0)
        if hard_fg.any():
            fg_terms.append(_info_nce(e[hard_fg], easy_fg, easy_bg_rows, tau))
        if hard_bg.any():
            bg_terms.append(_info_nce(e[hard_bg], easy_bg, easy_fg_rows, tau))
        used += 1
    zero = emb.sum() * 0.0
    loss_fg = torch.stack(fg_terms).mean() if fg_terms else zero
    loss_bg = torch.stack(bg_terms).mean() if bg_terms else zero
    return loss_fg, loss_bg, used
