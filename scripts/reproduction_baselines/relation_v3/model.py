"""Transport-only Relation-V3 with an exactly immutable MACIL path."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from macilsd.avce_network import AVCE_Model
from relation_v2.model import MaskedTemporalEncoder, _masked_layer


class UnbalancedTransportBank(nn.Module):
    """Directed local transport with a per-source dustbin.

    Each source row distributes at most one unit of mass.  The omitted mass is
    assigned to a learned dustbin, avoiding the forced-match saturation of a
    balanced Sinkhorn coupling.
    """

    def __init__(self, hidden, n_relations, relation_dim, window, temperature):
        super().__init__()
        self.n_relations = int(n_relations)
        self.relation_dim = int(relation_dim)
        self.window = int(window)
        self.temperature = float(temperature)
        self.source = nn.Linear(hidden, self.n_relations)
        self.target = nn.Linear(hidden, self.n_relations)
        self.query = nn.Linear(hidden, self.n_relations * self.relation_dim,
                               bias=False)
        self.key = nn.Linear(hidden, self.n_relations * self.relation_dim,
                             bias=False)
        self.dustbin = nn.Parameter(torch.zeros(self.n_relations))

    def forward(self, frame, valid):
        b, t, _ = frame.shape
        src = torch.sigmoid(self.source(frame)) * valid[..., None]
        tgt = torch.sigmoid(self.target(frame)) * valid[..., None]
        q = self.query(frame).view(b, t, self.n_relations,
                                   self.relation_dim).permute(0, 2, 1, 3)
        k = self.key(frame).view(b, t, self.n_relations,
                                 self.relation_dim).permute(0, 2, 1, 3)
        affinity = torch.einsum("brtd,brsd->brts", q, k)
        affinity = affinity / (math.sqrt(self.relation_dim) * self.temperature)
        affinity = affinity + torch.log(tgt.permute(0, 2, 1)[:, :, None, :]
                                        .clamp_min(1e-6))
        index = torch.arange(t, device=frame.device)
        legal = ((index[:, None] - index[None, :]).abs() <= self.window)
        legal = (legal[None, None] & valid[:, None, :, None]
                 & valid[:, None, None, :])
        affinity = affinity.masked_fill(~legal, -torch.inf)
        dust = self.dustbin[None, :, None, None].expand(b, -1, t, 1)
        probability = torch.softmax(torch.cat([affinity, dust], -1), -1)[..., :-1]
        probability = torch.nan_to_num(probability, nan=0.0) * legal
        transport = probability * src.permute(0, 2, 1)[..., None]
        outgoing = transport.sum(-1)
        incoming = transport.sum(-2)
        witness = torch.sqrt((outgoing * incoming).clamp_min(0.0) + 1e-8)
        endpoint = torch.stack([outgoing, incoming, witness], -1)
        endpoint = endpoint.permute(0, 2, 1, 3).reshape(b, t, -1)
        return endpoint, transport, outgoing, incoming, witness


class RelationV3(nn.Module):
    """A zero-initialized relation residual over frozen MACIL frame logits."""

    def __init__(self, args):
        super().__init__()
        hidden = args.hid_dim
        self.macil = AVCE_Model(args)
        for parameter in self.macil.parameters():
            parameter.requires_grad_(False)
        self.text_temporal = MaskedTemporalEncoder(
            args.text_feature_size, hidden, args.dropout)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(args.dropout),
            nn.LayerNorm(hidden))
        self.relations = UnbalancedTransportBank(
            hidden, args.n_relations, args.relation_dim,
            args.binding_window, args.binding_temperature)
        # The readout sees endpoint transport statistics only.  No raw frame,
        # source evidence, or target evidence is available as a shortcut.
        self.readout = nn.Sequential(
            nn.Linear(3 * args.n_relations, hidden), nn.GELU(),
            nn.Dropout(args.dropout), nn.Linear(hidden, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.topk_divisor = int(args.topk_divisor)

    def train(self, mode=True):
        super().train(mode)
        self.macil.eval()
        return self

    def forward(self, f_a, f_v, f_t, lengths, valid_mask=None):
        if valid_mask is None:
            valid_mask = (torch.arange(f_v.shape[1], device=f_v.device)[None]
                          < lengths.to(f_v.device)[:, None])
        with torch.no_grad():
            pv = self.macil.fc_v(f_v) * valid_mask[..., None]
            pa = self.macil.fc_a(f_a) * valid_mask[..., None]
            layer = self.macil.cma.layer
            v_out = _masked_layer(layer, pv, pa, pa, valid_mask, valid_mask)
            a_out = _masked_layer(layer, pa, pv, pv, valid_mask, valid_mask)
            base_bag, audio_logits, visual_logits, av_logits = self.macil.att_mmil(
                a_out, v_out, lengths)
        text = self.text_temporal(f_t, valid_mask)
        frame = self.fuse(torch.cat([a_out, v_out, text], -1))
        endpoint, transport, outgoing, incoming, witness = self.relations(
            frame, valid_mask)
        delta = self.readout(endpoint).squeeze(-1) * valid_mask
        base = av_logits.squeeze(-1)
        final = base + delta
        probability = torch.sigmoid(final) * valid_mask
        return {"frame_prob": probability, "delta_logit": delta,
                "base_frame_logits": av_logits, "base_bag_prob": base_bag,
                "audio_logits": audio_logits, "visual_logits": visual_logits,
                "transport": transport, "endpoint_outgoing": outgoing,
                "endpoint_incoming": incoming, "relation_witness": witness}
