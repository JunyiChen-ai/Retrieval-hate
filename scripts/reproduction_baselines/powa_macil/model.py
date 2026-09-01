"""POWA modules on top of the reproduced MACIL-SD AV backbone.

Preprocessing is deliberately absent here.  The model receives already aligned
I3D, VGGish and sentence features.  Its three method modules are PEF, AWB and
PCW-MIL, as frozen in docs/duplex/PREREG_POWA_MACIL.md.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # package import: scripts.reproduction_baselines.powa_macil
    from ..macilsd.Transformer import (MultiHeadAttention,
                                       PositionwiseFeedForward,
                                       SelfAttentionBlock, TransformerLayer)
    from ..macilsd.avce_network import AVCE_Model
except ImportError:  # direct script entrypoint from reproduction_baselines/
    from macilsd.Transformer import (MultiHeadAttention,
                                     PositionwiseFeedForward,
                                     SelfAttentionBlock, TransformerLayer)
    from macilsd.avce_network import AVCE_Model


PRIMITIVES = ("hostile", "target", "violence", "sexual", "self_harm",
              "protected_context")
P_HOSTILE, P_TARGET, P_VIOLENCE, P_SEXUAL, P_SELF_HARM, P_CONTEXT = range(6)

TARGETED = ("AND", "targeted_hate", ("NOT", "context"))
ABUSE = ("AND", "untargeted_abuse", ("NOT", "context"))
POLICIES = {
    # Executable policy expressions, not dataset-specific learned heads.
    "hatemm": TARGETED,
    "mhclip_en": ("OR", TARGETED, ABUSE),
    "mhclip_zh": ("OR", TARGETED, ABUSE),
    "hateclipseg": ("OR", TARGETED, ABUSE, "violence", "sexual",
                    "self_harm"),
}
PERMUTED_POLICY = {
    "hatemm": "hateclipseg", "hateclipseg": "mhclip_en",
    "mhclip_en": "hatemm", "mhclip_zh": "hatemm",
}


def _safe_logit(p, eps=1e-5):
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def noisy_or(parts):
    """Differentiable OR over a non-empty list of probabilities."""
    if not parts:
        raise ValueError("noisy_or needs at least one witness")
    return 1.0 - torch.stack([1.0 - p.clamp(0, 1) for p in parts], 0).prod(0)


def execute_policy(expression, witnesses):
    """Recursively execute a differentiable AND/OR/NOT policy AST."""
    if isinstance(expression, str):
        if expression not in witnesses:
            raise KeyError("unknown witness %s" % expression)
        return witnesses[expression]
    op, *children = expression
    values = [execute_policy(child, witnesses) for child in children]
    if op == "AND":
        if not values:
            raise ValueError("AND needs a child")
        return torch.stack(values, 0).prod(0)
    if op == "OR":
        return noisy_or(values)
    if op == "NOT" and len(values) == 1:
        return 1.0 - values[0]
    raise ValueError("malformed policy expression %r" % (expression,))


class PolicyConditionedEvidenceFactorizer(nn.Module):
    """Typed primitive heads over AV context and aligned sentence evidence."""

    def __init__(self, hidden=128, text_dim=768, nhead=4, dropout=0.1,
                 prototype_file=None, semantic_strength=0.5,
                 semantic_temperature=0.07, permute_semantics=False):
        super().__init__()
        ff = PositionwiseFeedForward(hidden, hidden)
        self.text_projection = nn.Linear(text_dim, hidden)
        self.text_temporal = SelfAttentionBlock(
            TransformerLayer(hidden, MultiHeadAttention(nhead, hidden),
                             copy.deepcopy(ff), dropout))
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(hidden))
        self.primitive_head = nn.Linear(hidden, len(PRIMITIVES))
        self.hostile_query = nn.Linear(hidden, hidden, bias=False)
        self.target_key = nn.Linear(hidden, hidden, bias=False)
        self.semantic_strength = float(semantic_strength)
        self.semantic_temperature = float(semantic_temperature)
        self.permute_semantics = bool(permute_semantics)
        self.has_semantic_grounding = bool(prototype_file)
        if prototype_file:
            if not os.path.isfile(prototype_file):
                raise FileNotFoundError(prototype_file)
            proto = np.load(prototype_file)
            self.register_buffer("semantic_en", torch.from_numpy(proto["en"]).float())
            self.register_buffer("semantic_zh", torch.from_numpy(proto["zh"]).float())

    def forward(self, audio_context, visual_context, text_features, policy,
                valid_mask=None):
        text = self.text_temporal(self.text_projection(text_features),
                                  valid_mask=valid_mask)
        shared = self.fuse(torch.cat([audio_context, visual_context, text], -1))
        primitive_logits = self.primitive_head(shared)
        semantic_logits = None
        text_present = text_features.norm(dim=-1) > 1e-6
        if self.has_semantic_grounding:
            prototype = self.semantic_zh if policy == "mhclip_zh" else self.semantic_en
            if self.permute_semantics:
                prototype = prototype.roll(1, dims=0)
            unit_text = F.normalize(text_features, dim=-1, eps=1e-6)
            semantic_logits = torch.einsum("btd,kd->btk", unit_text, prototype)
            semantic_logits = semantic_logits / self.semantic_temperature
            semantic_logits = semantic_logits * text_present[..., None]
            primitive_logits = primitive_logits + self.semantic_strength * semantic_logits
        return (primitive_logits, self.hostile_query(shared),
                self.target_key(shared), semantic_logits, text_present, shared)


class AsynchronousWitnessBinder(nn.Module):
    """Capacity-constrained temporal transport between predicate and target.

    This is intentionally not cross-attention and not pointwise multiplication.
    Sinkhorn scaling creates a soft bipartite transport plan whose marginals are
    the normalized hostile and target evidence masses.  The smaller total mass
    caps how much relational witness can be created.
    """

    def __init__(self, hidden=128, window=12, temperature=0.2,
                 sinkhorn_iters=8):
        super().__init__()
        self.scale = hidden ** -0.5
        self.window = int(window)
        self.temperature = float(temperature)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.distance_penalty = nn.Parameter(torch.tensor(1.0))

    def forward(self, hostile_prob, target_prob, hostile_query, target_key,
                valid_mask=None):
        # Inputs: [B,T], embeddings [B,T,D].
        b, t = hostile_prob.shape
        device, dtype = hostile_prob.device, hostile_prob.dtype
        idx = torch.arange(t, device=device)
        distance = (idx[:, None] - idx[None, :]).abs().to(dtype)
        allowed = distance <= self.window
        sim = torch.einsum("btd,bsd->bts", hostile_query, target_key) * self.scale
        penalty = F.softplus(self.distance_penalty) * distance / max(1, self.window)
        # Long sequences can produce extreme learned similarities. Clamping in
        # log-kernel space preserves the ordering while preventing exp/gradient
        # overflow during Sinkhorn scaling.
        log_kernel = ((sim - penalty) / self.temperature).clamp(-30.0, 30.0)
        kernel = torch.exp(log_kernel)
        kernel = kernel * allowed.to(dtype)[None]

        if valid_mask is None:
            valid_mask = torch.ones((b, t), device=device, dtype=torch.bool)
        pair_valid = valid_mask[:, :, None] & valid_mask[:, None, :]
        support = pair_valid & allowed[None]
        kernel = kernel * support.to(dtype)

        h = hostile_prob * valid_mask.to(dtype)
        g = target_prob * valid_mask.to(dtype)
        h_mass = h.sum(-1, keepdim=True)
        g_mass = g.sum(-1, keepdim=True)
        h_marginal = h / h_mass.clamp_min(1e-6)
        g_marginal = g / g_mass.clamp_min(1e-6)

        # Sinkhorn-Knopp scaling against semantic evidence marginals.
        u = torch.ones_like(h_marginal)
        v = torch.ones_like(g_marginal)
        # Preserve structural zeros. Clamping the complete tensor would
        # silently reactivate forbidden lags and padded positions, and
        # Sinkhorn scaling can amplify even a 1e-12 forbidden edge.
        k = torch.where(support, kernel.clamp_min(1e-12),
                        torch.zeros_like(kernel))
        for _ in range(self.sinkhorn_iters):
            u = h_marginal / torch.einsum("bts,bs->bt", k, v).clamp_min(1e-6)
            v = g_marginal / torch.einsum("bts,bt->bs", k, u).clamp_min(1e-6)
        plan = u[:, :, None] * k * v[:, None, :]

        # `plan` has unit total mass. Scale it back to the amount of evidence
        # that can actually be paired. Capping this scalar at one would spread
        # at most one unit over an entire video (roughly 1/T per frame), making
        # a confident relational witness impossible for long sequences.
        capacity = torch.minimum(h_mass, g_mass)
        bound = plan * capacity[:, None]
        # Return witness mass to both participating timestamps.
        witness = (bound.sum(-1) + bound.sum(-2)).clamp(0, 1)
        no_relation = ((h_mass <= 1e-6) | (g_mass <= 1e-6)).to(dtype)
        witness = witness * (1.0 - no_relation)
        return witness, bound


class PolicyCompiledWitnessMIL(nn.Module):
    """Compile a published moderation policy into dense and bag scores."""

    def __init__(self, policy, topk_divisor=16):
        super().__init__()
        if policy not in POLICIES:
            raise KeyError("unknown policy %s" % policy)
        self.policy = policy
        self.expression = POLICIES[policy]
        self.topk_divisor = int(topk_divisor)

    def frame_probability(self, primitive_prob, targeted_hate):
        hostile = primitive_prob[..., P_HOSTILE]
        target = primitive_prob[..., P_TARGET]
        # Untargeted abuse is residual hostility not already explained by a
        # local protected target. It remains admissible only where policy says.
        witnesses = {
            "targeted_hate": targeted_hate,
            "untargeted_abuse": hostile * (1.0 - target),
            "violence": primitive_prob[..., P_VIOLENCE],
            "sexual": primitive_prob[..., P_SEXUAL],
            "self_harm": primitive_prob[..., P_SELF_HARM],
            "context": primitive_prob[..., P_CONTEXT],
        }
        return execute_policy(self.expression, witnesses)

    def bag_probability(self, frame_prob, lengths):
        bags = []
        for i in range(frame_prob.shape[0]):
            n = int(lengths[i])
            k = max(1, n // self.topk_divisor + 1)
            bags.append(frame_prob[i, :n].topk(k).values.mean())
        return torch.stack(bags)


class POWAMACIL(nn.Module):
    """MACIL-SD AV plus the three POWA method modules."""

    def __init__(self, args, policy=None):
        super().__init__()
        self.multi_backbone = getattr(args, "multi_backbone", False)
        if self.multi_backbone:
            corpora = getattr(args, "corpora", list(POLICIES))
            self.macils = nn.ModuleDict({name: AVCE_Model(args)
                                         for name in corpora})
        else:
            self.macil = AVCE_Model(args)
        self.pef = PolicyConditionedEvidenceFactorizer(
            hidden=args.hid_dim, text_dim=getattr(args, "text_feature_size", 768),
            nhead=args.nhead, dropout=args.dropout,
            prototype_file=(getattr(args, "semantic_prototype_file", None)
                            if getattr(args, "semantic_grounding", False) else None),
            semantic_strength=getattr(args, "semantic_strength", 0.5),
            semantic_temperature=getattr(args, "semantic_temperature", 0.07),
            permute_semantics=getattr(args, "ablation", "full") == "semantic_permutation")
        self.awb = AsynchronousWitnessBinder(
            hidden=args.hid_dim, window=getattr(args, "binding_window", 12),
            temperature=getattr(args, "binding_temperature", 0.2),
            sinkhorn_iters=getattr(args, "sinkhorn_iters", 8))
        self.default_policy = policy
        self.ablation = getattr(args, "ablation", "full")
        if self.ablation == "flat_fusion":
            self.flat_head = nn.Linear(len(PRIMITIVES), 1)
        if self.ablation == "anonymous_head":
            self.anonymous_head = nn.Linear(len(PRIMITIVES) + 1, 1)
        self.policy_mils = nn.ModuleDict({
            name: PolicyCompiledWitnessMIL(name) for name in POLICIES
        })
        # PCW is a correction to, rather than a replacement for, the strong
        # MACIL dense score. sigmoid(-2.2) ~= 0.10 at initialisation.
        self.policy_residual_gate = nn.Parameter(torch.tensor(-2.2))
        self.use_policy_residual = True
        self.residual_mode = getattr(args, "residual_mode", "signed")
        if self.residual_mode not in ("signed", "positive_evidence"):
            raise ValueError("unknown residual mode %s" % self.residual_mode)

    def forward(self, f_a, f_v, f_t, seq_len, valid_mask=None, policy=None):
        policy = policy or self.default_policy
        if policy is None:
            raise ValueError("policy is required for a shared POWA model")
        backbone = self.macils[policy] if self.multi_backbone else self.macil
        base = backbone(f_a, f_v, seq_len, valid_mask=valid_mask)
        base_bag, audio_logits, visual_logits, av_logits, v_out, a_out = base
        primitive_logits, hq, tk, semantic_logits, text_present, shared = self.pef(
            a_out, v_out, f_t, policy, valid_mask=valid_mask)
        primitive_prob = torch.sigmoid(primitive_logits)
        hostile = primitive_prob[..., P_HOSTILE]
        target = primitive_prob[..., P_TARGET]
        if self.ablation == "pointwise":
            targeted, transport = hostile * target, None
        elif self.ablation == "same_time":
            affinity = torch.sigmoid((hq * tk).sum(-1) * self.awb.scale)
            targeted, transport = torch.minimum(hostile, target) * affinity, None
        else:
            targeted, transport = self.awb(
                hostile, target, hq, tk, valid_mask=valid_mask)
        compiled_policy = (PERMUTED_POLICY[policy]
                           if self.ablation == "policy_permutation" else policy)
        compiler = self.policy_mils[compiled_policy]
        if self.ablation == "flat_fusion":
            witness_prob = torch.sigmoid(self.flat_head(primitive_logits).squeeze(-1))
        elif self.ablation == "anonymous_head":
            anonymous = torch.cat([primitive_logits, targeted[..., None]], -1)
            witness_prob = torch.sigmoid(self.anonymous_head(anonymous).squeeze(-1))
        else:
            witness_prob = compiler.frame_probability(primitive_prob, targeted)
        residual_scale = torch.sigmoid(self.policy_residual_gate)
        base_frame_logits = av_logits.squeeze(-1)
        if self.use_policy_residual:
            if self.residual_mode == "signed":
                residual = _safe_logit(witness_prob)
            else:
                # Primitive supervision is sparse: absence of an observed
                # witness is not evidence that a frame is benign.  The
                # cumulative-hazard transform lets positive policy evidence
                # boost MACIL while leaving zero evidence neutral.
                residual = -torch.log1p(-witness_prob.clamp(0, 1 - 1e-5))
            frame_prob = torch.sigmoid(base_frame_logits +
                                       residual_scale * residual)
        else:
            frame_prob = witness_prob
        witness_bag = compiler.bag_probability(frame_prob, seq_len)
        return {
            "bag_prob": witness_bag,
            "frame_prob": frame_prob,
            "typed_frame_prob": witness_prob,
            "policy_residual_scale": residual_scale,
            "primitive_logits": primitive_logits,
            "primitive_prob": primitive_prob,
            "semantic_logits": semantic_logits,
            "semantic_text_mask": text_present,
            "targeted_hate": targeted,
            "transport": transport,
            "base_bag_prob": base_bag,
            "base_frame_logits": av_logits,
            "audio_logits": audio_logits,
            "visual_logits": visual_logits,
            "audio_rep": a_out,
            "visual_rep": v_out,
            # Exposed for training-only representation objectives. Existing
            # POWA logits, probabilities and inference behavior are unchanged.
            "shared_rep": shared,
        }
