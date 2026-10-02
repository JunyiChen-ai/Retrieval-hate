"""One cross-modal prior with pre-CMA interval/complement conditioning."""
import torch
from torch import nn
from torch.nn import functional as F

from qtl.content import ContentLayouts
from qtl.interval_stats import sum_inside_outside
from qtl.model import PriorNet


EXTRA_DEFAULTS = {
    "backbone": "residual_io", "answer_source": "soft_both", "soft_levels": 8,
    "node_prior": True, "io_outside": True, "io_residual": True,
}


class ContextResidual(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.net = nn.Sequential(nn.Linear(6 * hidden, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, hidden))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inside, outside):
        i, o = self.norm(inside), self.norm(outside)
        inputs = torch.cat((i, o, i - o, i * o, i.flip(1), o.flip(1)), -1)
        return self.net(inputs)


class ResidualInsideOutsidePrior(PriorNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg["backbone"] == "residual_io" and self.node_prior
        self.context = ContextResidual(int(cfg["hid_dim"]), float(cfg["dropout"]))
        self.use_outside = bool(cfg["io_outside"])
        self.use_residual = bool(cfg["io_residual"])
        self.layouts = ContentLayouts()

    def forward(self, f_a, f_v, mask):
        self._context_nodes = self._last_v = self._last_a = None
        if not self.use_residual:
            return super().forward(f_a, f_v, mask)
        # One set of projections, shared by CMA and content statistics.
        projected_v, projected_a = self.fc_v(f_v), self.fc_a(f_a)
        base_v, base_a = self.cma(projected_v, projected_a, mask)
        h = base_v + base_a
        w = self.att(h).squeeze(-1).masked_fill(~mask, -1e9)
        pooled = torch.einsum("bt,bth->bh", torch.softmax(w, dim=1), h)
        g = self.vid(pooled).squeeze(-1) if self.g_head else self.g0.expand(mask.shape[0])

        key, (forest, levels, _) = self.layouts.get(mask)
        leaves = torch.stack((projected_v, projected_a), 2)
        B, T, M, H = leaves.shape
        packed = torch.cat((leaves, torch.ones_like(leaves[..., :1])), -1)
        inside, outside = sum_inside_outside(packed.reshape(B * T, M, H + 1),
                                             levels, self.use_outside)
        leaf_rows, leaf_delta, nodes, node_delta = [], [], [], []
        for lev, i, o in zip(levels, inside, outside):
            di = i[..., :-1] / i[..., -1:].clamp_min(1)
            do = o[..., :-1] / o[..., -1:].clamp_min(1)
            change = self.context(di, do)
            n = lev["n_leaf"]
            if n:
                leaf_rows.append(lev["leaf_row"])
                leaf_delta.append(change[:n])
            if len(lev["nodes"]) > n:
                nodes.append(lev["nodes"][n:])
                node_delta.append(change[n:].sum(1))
        delta = (leaves.new_zeros(B * T, M, H)
                 .index_copy(0, torch.cat(leaf_rows), torch.cat(leaf_delta)).view(B, T, M, H))
        v_out, a_out = base_v + delta[:, :, 0], base_a + delta[:, :, 1]
        a_log, v_log = self.fc(a_out).squeeze(-1), self.fc(v_out).squeeze(-1)
        context_nodes = leaves.new_zeros(forest.N, H)
        if nodes:
            context_nodes = context_nodes.index_copy(0, torch.cat(nodes), torch.cat(node_delta))
        self._last_key, self._context_nodes = key, context_nodes
        self._last_v, self._last_a = v_out, a_out
        return a_log + v_log, g, a_log, v_log, v_out, a_out

    def node_logits(self, v_out, a_out, forest):
        phi = super().node_logits(v_out, a_out, forest)
        if not self.use_residual:
            return phi
        assert v_out is self._last_v and a_out is self._last_a
        assert tuple(forest.Ts) == self._last_key[0] and forest.N == len(self._context_nodes)
        # Linear(H + D) = Linear(H) + W D. Bias occurs once; leaf D is zero.
        return phi + F.linear(self._context_nodes, self.node.weight).squeeze(-1)


def make_model(cfg):
    return ResidualInsideOutsidePrior(cfg)
