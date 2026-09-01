"""Source-exclusive utterance/visible-speaker relation inside POWA."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from powa_macil.model import POWAMACIL


class SourceBoundPEF(nn.Module):
    """Preserve POWA PEF while augmenting its shared representation.

    The added relation can see only the utterance representation and its
    assigned visible-speaker face (or a learned null source).  Full-frame
    visual features and non-assigned faces never enter this branch.
    """

    def __init__(self, base, hidden=128, text_dim=768, face_dim=512,
                 dropout=.1, relation_weight=1.):
        super().__init__()
        self.base = base
        self.face_projection = nn.Linear(face_dim, hidden)
        self.utterance_projection = nn.Linear(text_dim, hidden)
        self.null_source = nn.Parameter(torch.zeros(hidden))
        self.state_embedding = nn.Embedding(3, hidden)
        self.relation_adapter = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.relation_weight = float(relation_weight)
        self._source_face = None
        self._source_state = None
        self._source_utterance = None

    def set_source(self, face, state, utterance):
        self._source_face = face
        self._source_state = state
        self._source_utterance = utterance

    def clear_source(self):
        self._source_face = None
        self._source_state = None
        self._source_utterance = None

    def forward(self, audio_context, visual_context, text_features, policy,
                valid_mask=None):
        if (self._source_face is None or self._source_state is None or
                self._source_utterance is None):
            raise RuntimeError("source-bound inputs were not supplied")
        state = self._source_state.long()
        if state.shape != text_features.shape[:2]:
            raise ValueError("source-state/time shape mismatch")
        text = self.base.text_temporal(
            self.base.text_projection(text_features), valid_mask=valid_mask)
        shared = self.base.fuse(
            torch.cat([audio_context, visual_context, text], -1))

        visible_face = self.face_projection(self._source_face)
        null_face = self.null_source.view(1, 1, -1).expand_as(visible_face)
        source = torch.where((state == 2)[..., None], visible_face, null_face)
        utterance = self.utterance_projection(self._source_utterance)
        physical_relation = source * utterance
        relation = self.relation_adapter(torch.cat([
            physical_relation, self.state_embedding(state.clamp(0, 2))], -1))
        speech = (state != 0)
        if valid_mask is not None:
            speech = speech & valid_mask
        relation = relation * speech[..., None].to(relation.dtype)
        shared = shared + self.relation_weight * relation

        primitive_logits = self.base.primitive_head(shared)
        semantic_logits = None
        text_present = text_features.norm(dim=-1) > 1e-6
        if self.base.has_semantic_grounding:
            prototype = (self.base.semantic_zh if policy == "mhclip_zh"
                         else self.base.semantic_en)
            if self.base.permute_semantics:
                prototype = prototype.roll(1, dims=0)
            unit_text = F.normalize(text_features, dim=-1, eps=1e-6)
            semantic_logits = torch.einsum(
                "btd,kd->btk", unit_text, prototype)
            semantic_logits = semantic_logits / self.base.semantic_temperature
            semantic_logits = semantic_logits * text_present[..., None]
            primitive_logits = (primitive_logits +
                                self.base.semantic_strength * semantic_logits)
        return (primitive_logits, self.base.hostile_query(shared),
                self.base.target_key(shared), semantic_logits, text_present,
                shared)


class ActiveSpeakerBoundPOWA(POWAMACIL):
    def __init__(self, args, policy=None):
        super().__init__(args, policy=policy)
        self.pef = SourceBoundPEF(
            self.pef, hidden=args.hid_dim,
            text_dim=getattr(args, "text_feature_size", 768),
            face_dim=getattr(args, "face_feature_size", 512),
            dropout=args.dropout,
            relation_weight=getattr(args, "relation_weight", 1.))

    def forward(self, f_a, f_v, f_t, f_face, source_state,
                source_utterance, seq_len, valid_mask=None, policy=None):
        self.pef.set_source(f_face, source_state, source_utterance)
        try:
            return super().forward(f_a, f_v, f_t, seq_len, valid_mask, policy)
        finally:
            self.pef.clear_source()
