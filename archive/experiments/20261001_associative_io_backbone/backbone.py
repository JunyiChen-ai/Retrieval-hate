"""Inside-outside content attention with additive sufficient statistics.

Tree messages carry weighted sums and masses, without nonlinear compression.
Attention weights are pointwise; outside never uses an in-interval cache row.
"""
import torch
from torch import nn

from qtl import data as qdata
from qtl.content import ContentLayouts, ContextFusion
from macilsd import align
import hier_evidence_common as hc


EXTRA_DEFAULTS = {
    "backbone": "associative_io", "answer_source": "soft_both", "soft_levels": 8,
    "node_prior": True, "io_heads": 4, "io_outside": True, "io_attention": True,
    "io_weight_clip": 10.0,
}


class AssociativeInsideOutsidePrior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(cfg["hid_dim"])
        self.heads = int(cfg["io_heads"])
        assert self.heads >= 1 and cfg["backbone"] == "associative_io"
        self.fc_v = nn.Linear(align.V_DIM, hidden)
        self.fc_a = nn.Linear(qdata.A_IN + hc.TEXT_DIM * (len(cfg["text_sources"]) - 1), hidden)
        self.leaf_norm = nn.LayerNorm(hidden)
        # A per-head additive bias would cancel in S/Z, so do not learn one.
        self.attention = nn.Linear(hidden, self.heads, bias=False)
        self.head_mix = nn.Linear(self.heads * hidden, hidden)
        self.readout_norm = nn.LayerNorm(hidden)
        self.fuse = ContextFusion(hidden, float(cfg["dropout"]))
        self.fc = nn.Linear(hidden, 1)
        self.vid = nn.Linear(hidden, 1)
        self.node = nn.Linear(hidden, 1)
        nn.init.zeros_(self.node.weight)
        nn.init.zeros_(self.node.bias)
        self.node_prior = bool(cfg["node_prior"])
        self.g_head = bool(cfg.get("g_head", True))
        if not self.g_head:
            self.g0 = nn.Parameter(torch.zeros(()))
        self.use_outside = bool(cfg["io_outside"])
        self.use_attention = bool(cfg["io_attention"])
        self.weight_clip = float(cfg["io_weight_clip"])
        assert 0 < self.weight_clip <= 10
        self.layouts = ContentLayouts()

    def readout(self, stats):
        # stats: (nodes, modality, head, hidden+1); final coordinate is positive mass.
        sums, masses = stats[..., :-1], stats[..., -1:]
        means = sums / masses.clamp_min(1e-12)
        values = self.readout_norm(self.head_mix(means.flatten(-2)))
        # Empty root outside is exactly zero, with no unused learnable parameter.
        return values * (masses.sum(-2) > 0).to(values.dtype)

    def messages(self, leaves, levels):
        """Return unnormalized inside/outside statistics for every forest level."""
        B, T, M, H = leaves.shape
        if self.use_attention:
            weights = self.attention(leaves).clamp(-self.weight_clip, self.weight_clip).exp()
        else:
            weights = leaves.new_ones(B, T, M, self.heads)
        packed = torch.cat((leaves.unsqueeze(-2) * weights.unsqueeze(-1), weights.unsqueeze(-1)), -1)
        flat = packed.reshape(B * T, M, self.heads, H + 1)
        inside = [None] * len(levels)
        for depth in range(len(levels) - 1, -1, -1):
            lev, pieces = levels[depth], []
            if lev["n_leaf"]:
                pieces.append(flat[lev["leaf_row"]])
            if len(lev["nodes"]) > lev["n_leaf"]:
                pieces.append(inside[depth + 1][lev["L"]] + inside[depth + 1][lev["R"]])
            inside[depth] = torch.cat(pieces, 0)
        outside = [None] * len(levels)
        outside[0] = torch.zeros_like(inside[0])
        for depth, lev in enumerate(levels[:-1]):
            parent = outside[depth][lev["n_leaf"]:]
            state = torch.zeros_like(inside[depth + 1])
            if self.use_outside:
                state = (state.index_copy(0, lev["L"], parent + inside[depth + 1][lev["R"]])
                         .index_copy(0, lev["R"], parent + inside[depth + 1][lev["L"]]))
            outside[depth + 1] = state
        return inside, outside

    def forward(self, f_a, f_v, mask):
        self._last_phi = self._last_v = self._last_a = None
        B, T = mask.shape
        key, (forest, levels, roots) = self.layouts.get(mask)
        leaves = self.leaf_norm(torch.stack((self.fc_v(f_v), self.fc_a(f_a)), dim=2))
        hidden = leaves.shape[-1]
        inside, outside = self.messages(leaves, levels)
        leaf_rows, leaf_values, node_ids, node_values = [], [], [], []
        root_inside = None
        for depth, lev in enumerate(levels):
            i, o = self.readout(inside[depth]), self.readout(outside[depth])
            if depth == 0:
                root_inside = i[roots]
            rep = self.fuse(i, o)
            n_leaf = lev["n_leaf"]
            if n_leaf:
                leaf_rows.append(lev["leaf_row"])
                leaf_values.append(rep[:n_leaf])
            if len(lev["nodes"]) > n_leaf:
                node_ids.append(lev["nodes"][n_leaf:])
                node_values.append(self.node(rep[n_leaf:].sum(1)).squeeze(-1))
        dense = (leaves.new_zeros(B * T, 2, hidden)
                 .index_copy(0, torch.cat(leaf_rows), torch.cat(leaf_values)).view(B, T, 2, hidden))
        v_out, a_out = dense[:, :, 0], dense[:, :, 1]
        a_log, v_log = self.fc(a_out).squeeze(-1), self.fc(v_out).squeeze(-1)
        s = a_log + v_log
        g = self.vid(root_inside.sum(1)).squeeze(-1) if self.g_head else self.g0.expand(B)
        phi = leaves.new_zeros(forest.N)
        if node_ids:
            phi = phi.index_copy(0, torch.cat(node_ids), torch.cat(node_values))
        self._last_key, self._last_phi = key, phi
        self._last_v, self._last_a = v_out, a_out
        return s, g, a_log, v_log, v_out, a_out

    def node_logits(self, v_out, a_out, forest):
        assert v_out is self._last_v and a_out is self._last_a
        assert tuple(forest.Ts) == self._last_key[0] and forest.N == len(self._last_phi)
        return self._last_phi


def make_model(cfg):
    return AssociativeInsideOutsidePrior(cfg)
