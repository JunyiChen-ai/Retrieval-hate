#!/usr/bin/env python
"""Memory fix for VL3-SigLIP-NaViT's `VisionSdpaAttention`, exact by construction.

**Which class we actually run.** `video_pre_caption.py` was adapted in Phase A
from `flash_attention_2` to `sdpa`, because no flash-attn wheel exists for
torch 2.7.1 / cu128 / sm_120 (`MODEL_ASSETS_STATUS §3.6`, §3.10).  The encoder's
`VISION_ATTENTION_CLASSES` maps that to **`VisionSdpaAttention`**, a subclass of
`VisionAttention` — not to `VisionAttention` itself.  The distinction is
load-bearing:

  * `VisionAttention` (eager) computes `attn_weights + attention_mask` with a
    **bool** mask, which promotes `True -> 1.0` and `False -> 0.0`.  That is not
    masking at all: it adds a +1 bias inside each frame block and lets attention
    run globally.  Measured on a 2-block case, 40% of a row's probability mass
    lands outside its own block.
  * `VisionSdpaAttention` passes the same bool mask to
    `F.scaled_dot_product_attention`, where a bool `attn_mask` means "True =
    attend, False = -inf".  That is **strict block-diagonal** attention, and it
    agrees with `VisionFlashAttention2`'s `flash_attn_varlen_func` semantics.
    Verified against a hand-written block-diagonal softmax: max |diff| 4.4e-16.

So the flash -> sdpa substitution preserves the published attention semantics,
and the reproduction does not have to choose between them.

**The problem this fixes.** `F.scaled_dot_product_attention` with an explicit
mask cannot use the flash backend and materialises `[num_heads, N, N]`, where N
is patches across *all* frames of the clip.  At the processor's default
`max_tokens = 16384` that is tens of GiB, which is why 153 of 215 HateMM test
videos died of OOM while the shorter corpora were fine.

**The fix.** The mask is exactly block-diagonal, so attention inside block
`[s, e)` only ever reads keys and values in `[s, e)`.  Running one unmasked
`scaled_dot_product_attention` per block returns the identical result -- softmax
over the same logits, no other terms exist -- while allocating
`sum_i (e_i - s_i)^2` instead of `N^2`, and letting the fast backends run because
no mask is passed.  For a 10-frame clip that is a ~10x reduction.

  import videollama3_sdpa_blockattn as fix
  n = fix.install(model)     # returns how many modules were rebound
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
    mod = type(self).__mro__[0].__module__
    rope = __import__(mod, fromlist=["apply_rotary_pos_emb_vision"]) \
        .apply_rotary_pos_emb_vision

    seq_length = hidden_states.shape[0]
    q = self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
    k = self.k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
    v = self.v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
    q = rope(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = rope(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    q = q.transpose(0, 1)          # (H, N, D)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    out = torch.empty_like(q)
    cu = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
    for i in range(1, len(cu)):
        s, e = int(cu[i - 1]), int(cu[i])
        if e <= s:
            continue
        out[:, s:e] = F.scaled_dot_product_attention(
            q[:, s:e], k[:, s:e], v[:, s:e], dropout_p=0.0)

    out = out.transpose(0, 1).reshape(seq_length, -1)
    return self.out_proj(out)


def install(model) -> int:
    """Rebind every `VisionSdpaAttention` on `model`.  Returns how many."""
    n = 0
    for m in model.modules():
        if type(m).__name__ == "VisionSdpaAttention" and hasattr(m, "q_proj"):
            m.forward = types.MethodType(_forward, m)
            n += 1
    if n:
        print(f"[patch] VisionSdpaAttention -> per-block attention on {n} modules",
              flush=True)
    return n
