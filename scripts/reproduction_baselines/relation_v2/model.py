"""Performance-first latent temporal-relation model over MACIL-SD AV."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from macilsd.avce_network import AVCE_Model
from powa_macil.model import AsynchronousWitnessBinder


class LatentRelationBank(nn.Module):
    """Discover several source→target temporal relations without a fixed ontology."""

    def __init__(self, hidden=128, n_relations=4, relation_dim=32, window=24,
                 temperature=.2, sinkhorn_iters=8):
        super().__init__()
        self.n_relations = int(n_relations)
        self.relation_dim = int(relation_dim)
        self.evidence = nn.Linear(hidden, 2 * self.n_relations)
        self.query = nn.Linear(hidden, self.n_relations * self.relation_dim,
                               bias=False)
        self.key = nn.Linear(hidden, self.n_relations * self.relation_dim,
                             bias=False)
        self.binders = nn.ModuleList([
            AsynchronousWitnessBinder(
                hidden=self.relation_dim, window=window,
                temperature=temperature, sinkhorn_iters=sinkhorn_iters)
            for _ in range(self.n_relations)])

    def forward(self, frame, valid_mask):
        b, t, _ = frame.shape
        evidence = torch.sigmoid(self.evidence(frame))
        source, target = evidence.chunk(2, dim=-1)
        query = self.query(frame).view(b, t, self.n_relations,
                                            self.relation_dim)
        key = self.key(frame).view(b, t, self.n_relations,
                                        self.relation_dim)
        witnesses, transports = [], []
        for r, binder in enumerate(self.binders):
            witness, transport = binder(
                source[..., r], target[..., r], query[..., r, :],
                key[..., r, :], valid_mask=valid_mask)
            witnesses.append(witness)
            transports.append(transport)
        return {
            "source": source, "target": target,
            "witness": torch.stack(witnesses, dim=-1),
            "transport": torch.stack(transports, dim=1),
        }


class MaskedTemporalEncoder(nn.Module):
    """Local text encoder whose valid outputs ignore padded extension."""

    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, valid_mask):
        mask = valid_mask[..., None].to(value.dtype)
        value = self.project(value) * mask
        value = self.dropout(F.gelu(self.conv1(value.transpose(1, 2))))
        value = value.transpose(1, 2) * mask
        value = self.dropout(F.gelu(self.conv2(value.transpose(1, 2))))
        return self.norm(value.transpose(1, 2)) * mask


def _masked_mha(module, query, key, value, key_mask):
    """MACIL MHA parameters with a padding-key mask."""
    batch = query.shape[0]
    q, k, v = [linear(x).view(batch, -1, module.h, module.d_k).transpose(1, 2)
               for linear, x in zip(module.linears, (query, key, value))]
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(module.d_k)
    score = score.masked_fill(~key_mask[:, None, None, :], -torch.inf)
    probability = torch.softmax(score, dim=-1)
    probability = torch.nan_to_num(probability, nan=0.0)
    probability = module.dropout(probability)
    out = torch.matmul(probability, v)
    out = out.transpose(1, 2).contiguous().view(
        batch, -1, module.h * module.d_k)
    return module.linears[-1](out)


def _masked_layer(layer, query, key, value, query_mask, key_mask):
    first = layer.sublayer[0]
    attended = _masked_mha(layer.self_attn, first.norm(query), key, value,
                           key_mask)
    query = (query + first.dropout(attended)) * query_mask[..., None]
    second = layer.sublayer[1]
    query = query + second.dropout(layer.feed_forward(second.norm(query)))
    return query * query_mask[..., None]


class RelationV2(nn.Module):
    """MACIL backbone whose only extension is a latent temporal relation graph."""

    def __init__(self, args):
        super().__init__()
        hidden = args.hid_dim
        self.macil = AVCE_Model(args)
        self.text_temporal = MaskedTemporalEncoder(
            args.text_feature_size, hidden, args.dropout)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(args.dropout),
            nn.LayerNorm(hidden))
        self.relations = LatentRelationBank(
            hidden=hidden, n_relations=args.n_relations,
            relation_dim=args.relation_dim, window=args.binding_window,
            temperature=args.binding_temperature,
            sinkhorn_iters=args.sinkhorn_iters)
        relation_features = 3 * args.n_relations
        self.readout = nn.Sequential(
            nn.Linear(hidden + relation_features, hidden), nn.GELU(),
            nn.Dropout(args.dropout), nn.Linear(hidden, 1))
        self.residual_scale = nn.Parameter(torch.tensor(-2.2))
        self.topk_divisor = int(args.topk_divisor)

    def bag_probability(self, frame_prob, lengths):
        bags = []
        for i in range(frame_prob.shape[0]):
            n = int(lengths[i])
            k = max(1, n // self.topk_divisor + 1)
            bags.append(frame_prob[i, :n].topk(k).values.mean())
        return torch.stack(bags)

    def forward(self, f_a, f_v, f_t, lengths, valid_mask=None):
        if valid_mask is None:
            valid_mask = (torch.arange(f_v.shape[1], device=f_v.device)[None]
                          < lengths.to(f_v.device)[:, None])
        f_v = self.macil.fc_v(f_v) * valid_mask[..., None]
        f_a = self.macil.fc_a(f_a) * valid_mask[..., None]
        layer = self.macil.cma.layer
        v_out = _masked_layer(layer, f_v, f_a, f_a, valid_mask, valid_mask)
        a_out = _masked_layer(layer, f_a, f_v, f_v, valid_mask, valid_mask)
        base_bag, audio_logits, visual_logits, av_logits = self.macil.att_mmil(
            a_out, v_out, lengths)
        text = self.text_temporal(f_t, valid_mask)
        frame = self.fuse(torch.cat([a_out, v_out, text], dim=-1))
        relation = self.relations(frame, valid_mask)
        graph_features = torch.cat([
            relation["source"], relation["target"], relation["witness"]], -1)
        graph_logit = self.readout(torch.cat([frame, graph_features], -1))
        graph_logit = graph_logit.squeeze(-1)
        scale = F.softplus(self.residual_scale)
        final_logit = av_logits.squeeze(-1) + scale * graph_logit
        frame_prob = torch.sigmoid(final_logit) * valid_mask.to(final_logit.dtype)
        return {
            "bag_prob": self.bag_probability(frame_prob, lengths),
            "frame_prob": frame_prob,
            "graph_logit": graph_logit,
            "relation_source": relation["source"],
            "relation_target": relation["target"],
            "relation_witness": relation["witness"],
            "transport": relation["transport"],
            "residual_scale": scale,
            "base_bag_prob": base_bag,
            "base_frame_logits": av_logits,
            "audio_logits": audio_logits,
            "visual_logits": visual_logits,
        }
