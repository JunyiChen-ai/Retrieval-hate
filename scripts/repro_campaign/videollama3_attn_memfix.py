#!/usr/bin/env python
"""Bound the memory of VideoLLaMA3's eager vision attention, without changing what it computes.

Why this exists
---------------
`VisionAttention.forward` in the VL3-SigLIP-NaViT remote code materialises the
full attention matrix, `[num_heads, N, N]`, and then upcasts it to fp32 for the
softmax.  N is the number of vision patches across *all* frames of the clip, so
the allocation grows quadratically with video length: the URF captioning stage
asked for 24.81 GiB on 96 separate HateMM videos and 28.22 GiB on three more,
and lost 153 of the 215 HateMM test videos (71% of that split) to
`OutOfMemoryError`.

What this patch does
--------------------
Computes exactly the same thing in query-sized chunks, so the peak allocation is
`[H, chunk, N]` instead of `[H, N, N]`.  Every chunk still attends over all N
keys, so the softmax normalisation is identical -- this is a memory change only.

What this patch deliberately does NOT do
----------------------------------------
It does not "fix" the mask.  Upstream writes

    attention_mask = torch.zeros([1, q_len, q_len], dtype=torch.bool)
    ...                                                  # True inside each block
    attn_weights = attn_weights + attention_mask

and adding a *bool* tensor to a float tensor promotes True to 1.0 and False to
0.0.  So the eager path does not mask across frames at all: it applies a +1.0
bias to in-block logits and lets attention run globally.  On a 2-block toy case
56% of a row's probability mass lands outside its own block.  That differs from
the `VisionFlashAttention2` path the authors ship as the default, which uses
`cu_seqlens` varlen and is genuinely block-diagonal -- i.e. our flash -> sdpa
adaptation (forced by the absence of a flash-attn wheel for torch 2.7.1 / cu128
/ sm_120) silently changed the model's attention semantics, not just its memory.

Choosing between "reproduce the eager path we actually ran" and "reproduce the
block-diagonal attention the authors published with" is a fidelity decision for
the campaign owner, not something to settle inside a memory patch.  This module
implements the first faithfully and leaves the second visible.

Usage
-----
    import videollama3_attn_memfix as fix
    fix.install(model)          # after from_pretrained, before generate
"""
from __future__ import annotations

import torch
import torch.nn as nn

CHUNK = 1024


def _chunked_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
    mod = type(self).__module__
    rope = __import__(mod, fromlist=["apply_rotary_pos_emb_vision"]).apply_rotary_pos_emb_vision

    q_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(q_len, self.num_heads, self.head_dim)
    k = self.k_proj(hidden_states).view(q_len, self.num_heads, self.head_dim)
    v = self.v_proj(hidden_states).view(q_len, self.num_heads, self.head_dim)
    q = rope(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = rope(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)   # [H, N, d]

    bounds = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
    scale = self.head_dim ** 0.5
    out = torch.empty_like(q)

    for s in range(0, q_len, CHUNK):
        e = min(s + CHUNK, q_len)
        # [H, chunk, N] instead of [H, N, N]
        w = torch.matmul(q[:, s:e], k.transpose(1, 2)) / scale
        # Rebuild upstream's +1.0 in-block bias for just these rows.  Same value
        # upstream adds; reproduced exactly, bug and all.
        bias = torch.zeros(e - s, q_len, device=w.device, dtype=w.dtype)
        for a, b in zip(bounds, bounds[1:]):
            lo, hi = max(a, s), min(b, e)
            if hi > lo:
                bias[lo - s:hi - s, a:b] = 1.0
        w = w + bias
        w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        w = nn.functional.dropout(w, p=self.dropout, training=self.training)
        out[:, s:e] = torch.matmul(w, v)

    return self.out_proj(out.transpose(0, 1).reshape(q_len, -1))


def install(model) -> int:
    """Rebind every eager VisionAttention on `model`. Returns how many were patched."""
    n = 0
    for m in model.modules():
        if type(m).__name__ == "VisionAttention" and hasattr(m, "q_proj"):
            m.forward = _chunked_forward.__get__(m, type(m))
            n += 1
    print(f"[patch] VideoLLaMA3 eager vision attention -> query-chunked ({n} modules)",
          flush=True)
    return n
