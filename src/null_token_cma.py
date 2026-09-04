"""Null-token keys for MACIL-SD's shared cross-modal attention layer.

Candidate 4 (experiments/20260904_null_token_cma) established, in its own
training setting, that the cross-modal attention needs an explicit
"attend to nothing" key: MACIL-SD never masks padded rows, so at training time
the zero rows act as an accidental sink that is absent at test time; masking
them without a replacement hurts, and an explicit learnable null token whose
content is conditioned on the video-level verdict summary recovers and
improves on it (HateClipSeg +.020 AP / +.024 ROC over the same backbone
without the token; the single shared token equals the per-modality pair).

This module is the revision-1 form: the SAME MACIL-SD TransformerLayer object
(weights untouched, so every shared parameter is initialised exactly as in the
candidate-1 run of the same seed) is wrapped so that the key/value sequence of
each direction is extended by one null token n = b + W c at position 0, where c
is the masked mean over valid rows of the four verdict columns the backbone
already receives as input, and padded keys are masked. The wrapper keeps the
CrossAttentionBlock interface (forward(video, audio, valid_mask)) so it drops
into `AVCE_Model.cma`; the caller passes c and the validity mask through
`set_context` before the forward.

Arms:
  token="evidence"   n = b + W c            (revision-1 method, shared token)
  token="const"      n = b                  (no evidence conditioning)
  token="none"       no token, padding masked
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_EVID = 4


class NullTokenKeys(nn.Module):
    needs_context = True

    def __init__(self, layer, hid, token="evidence"):
        super().__init__()
        assert token in ("evidence", "const", "none"), token
        self.layer = layer
        self.size = layer.size
        self.token = token
        self.base = nn.Parameter(torch.zeros(1, hid)) if token != "none" else None
        self.cond = nn.Linear(N_EVID, hid) if token == "evidence" else None
        self.context = None
        self.mask = None

    def set_context(self, c, mask):
        """c: (B, N_EVID) video-level verdict summary; mask: (B, T) bool, True = valid row."""
        self.context = c
        self.mask = mask

    def null_token(self, B):
        if self.base is None:
            return None
        n = self.base.expand(B, -1)
        if self.cond is not None:
            n = n + self.cond(self.context)
        return n[:, None, :]

    def _one(self, q, kv, n, km):
        if n is not None:
            kv = torch.cat([n, kv], dim=1)
        return self.layer(q, kv, kv, key_padding_mask=km)

    def forward(self, video, audio, valid_mask=None):
        assert self.mask is not None, "set_context must be called before forward"
        B = video.shape[0]
        n = self.null_token(B)
        km = self.mask
        if n is not None:
            km = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=km.device), km], dim=1)
        video_cma = self._one(video, audio, n, km)
        audio_cma = self._one(audio, video, n, km)
        self.context, self.mask = None, None
        return video_cma, audio_cma
