"""Content representations on the same time tree used by QTL's label inference.

Inside and outside are neural representations, not the label posterior messages.
Outside excludes the target interval's cached rows; upstream feature receptive
fields are unchanged. No VLM answers or labels are inputs to this network.
"""
from collections import OrderedDict

import torch
from torch import nn

from qtl import ctree, data as qdata
from qtl.model import PriorNet as LegacyPriorNet
from macilsd import align
import hier_evidence_common as hc


EXTRA_DEFAULTS = {
    "backbone": "inside_outside", "answer_source": "soft_both",
    "soft_levels": 8, "node_prior": True,
    "io_outside": True, "io_merge": "gated",
}


class OrderedCompose(nn.Module):
    """Shared across tree depths and modalities, sensitive to left/right order."""

    def __init__(self, hidden):
        super().__init__()
        self.candidate = nn.Linear(2 * hidden, hidden)
        self.gate = nn.Linear(2 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, left, right):
        joined = torch.cat((left, right), dim=-1)
        gate = torch.sigmoid(self.gate(joined))
        return self.norm(gate * torch.tanh(self.candidate(joined))
                         + (1 - gate) * (left + right) * 0.5)


class ContextFusion(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6 * hidden, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, inside, outside):
        # (nodes, modality, hidden); each modality retains its own residual.
        other_i, other_o = inside.flip(1), outside.flip(1)
        x = torch.cat((inside, outside, inside - outside, inside * outside,
                       other_i, other_o), dim=-1)
        return self.norm(inside + self.drop(self.net(x)))


class InsideOutsidePrior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(cfg["hid_dim"])
        self.fc_v = nn.Linear(align.V_DIM, hidden)
        self.fc_a = nn.Linear(qdata.A_IN + hc.TEXT_DIM * (len(cfg["text_sources"]) - 1), hidden)
        self.leaf_norm = nn.LayerNorm(hidden)
        self.compose = OrderedCompose(hidden)
        self.out_left = OrderedCompose(hidden)
        self.out_right = OrderedCompose(hidden)
        self.root_outside = nn.Parameter(torch.zeros(2, hidden))
        self.fuse = ContextFusion(hidden, float(cfg["dropout"]))
        self.fc = nn.Linear(hidden, 1)
        self.vid = nn.Linear(hidden, 1)
        self.node = nn.Linear(hidden, 1)
        nn.init.zeros_(self.node.weight)
        nn.init.zeros_(self.node.bias)
        self.node_prior = bool(cfg["node_prior"])
        self.g_head = bool(cfg.get("g_head", True))
        self.g0 = nn.Parameter(torch.zeros(()))
        self.use_outside = bool(cfg["io_outside"])
        self.merge = cfg["io_merge"]
        assert self.merge in ("gated", "mean")
        self._topologies = OrderedDict()

    def _topology(self, mask):
        lengths = tuple(int(n) for n in mask.sum(1).tolist())
        key = (lengths, mask.shape[1], str(mask.device))
        if key not in self._topologies:
            forest = ctree.Forest(lengths, mask.shape[1])
            levels = [{k: v.to(mask.device) if isinstance(v, torch.Tensor) else v
                       for k, v in lev.items()} for lev in forest.levels]
            self._topologies[key] = (forest, levels, forest.pos_root.to(mask.device))
            if len(self._topologies) > 32:
                self._topologies.popitem(last=False)
        else:
            self._topologies.move_to_end(key)
        return key, self._topologies[key]

    def forward(self, f_a, f_v, mask):
        self._last_phi = self._last_v = self._last_a = None
        B, T = mask.shape
        key, (forest, levels, roots) = self._topology(mask)
        # Cross-time mixing starts only at the tree composition.
        leaves = self.leaf_norm(torch.stack((self.fc_v(f_v), self.fc_a(f_a)), dim=2))
        hidden = leaves.shape[-1]
        flat = leaves.reshape(B * T, 2, hidden)
        inside = [None] * len(levels)
        for d in range(len(levels) - 1, -1, -1):
            lev, pieces = levels[d], []
            n_leaf = lev["n_leaf"]
            if n_leaf:
                pieces.append(flat[lev["leaf_row"]])
            if len(lev["nodes"]) > n_leaf:
                left, right = inside[d + 1][lev["L"]], inside[d + 1][lev["R"]]
                pieces.append(self.compose(left, right) if self.merge == "gated"
                              else (left + right) * 0.5)
            inside[d] = torch.cat(pieces, dim=0)

        outside = [None] * len(levels)
        outside[0] = (self.root_outside[None].expand_as(inside[0]) if self.use_outside
                      else torch.zeros_like(inside[0]))
        for d, lev in enumerate(levels[:-1]):
            n_leaf = lev["n_leaf"]
            parent_out = outside[d][n_leaf:]
            left, right = inside[d + 1][lev["L"]], inside[d + 1][lev["R"]]
            if self.use_outside:
                o_left = self.out_left(parent_out, right)
                o_right = self.out_right(left, parent_out)
                outside[d + 1] = (torch.zeros_like(inside[d + 1])
                                  .index_copy(0, lev["L"], o_left)
                                  .index_copy(0, lev["R"], o_right))
            else:
                outside[d + 1] = torch.zeros_like(inside[d + 1])

        leaf_rows, leaf_values, node_ids, node_values = [], [], [], []
        for d, lev in enumerate(levels):
            rep = self.fuse(inside[d], outside[d])
            n_leaf = lev["n_leaf"]
            if n_leaf:
                leaf_rows.append(lev["leaf_row"])
                leaf_values.append(rep[:n_leaf])
            if len(lev["nodes"]) > n_leaf:
                node_ids.append(lev["nodes"][n_leaf:])
                node_values.append(self.node(rep[n_leaf:].sum(1)).squeeze(-1))
        dense = (torch.zeros_like(flat).index_copy(0, torch.cat(leaf_rows), torch.cat(leaf_values))
                 .view(B, T, 2, hidden))
        v_out, a_out = dense[:, :, 0], dense[:, :, 1]
        a_log, v_log = self.fc(a_out).squeeze(-1), self.fc(v_out).squeeze(-1)
        s = a_log + v_log
        g = self.vid(inside[0][roots].sum(1)).squeeze(-1) if self.g_head else self.g0.expand(B)
        phi = flat.new_zeros(forest.N)
        if node_ids:
            phi = phi.index_copy(0, torch.cat(node_ids), torch.cat(node_values))
        # node_logits immediately follows this forward in training and inference.
        # Keep the graph, never detach: node supervision trains both tree passes.
        self._last_key, self._last_phi = key, phi
        self._last_v, self._last_a = v_out, a_out
        return s, g, a_log, v_log, v_out, a_out

    def node_logits(self, v_out, a_out, forest):
        assert v_out is self._last_v and a_out is self._last_a, "node logits require the matching forward"
        assert tuple(forest.Ts) == self._last_key[0] and forest.N == len(self._last_phi)
        return self._last_phi


def make_model(cfg):
    if cfg["backbone"] == "inside_outside":
        return InsideOutsidePrior(cfg)
    if cfg["backbone"] == "macil":
        return LegacyPriorNet(cfg)
    raise ValueError("Unknown backbone: %s" % cfg["backbone"])
