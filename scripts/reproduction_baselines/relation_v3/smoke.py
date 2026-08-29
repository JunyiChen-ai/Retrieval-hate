#!/usr/bin/env python3
"""Validity and architectural invariants for Relation-V3."""

from types import SimpleNamespace
import torch
from relation_v3.model import RelationV3


def cfg():
    return SimpleNamespace(hid_dim=32, ffn_dim=32, nhead=4, dropout=0.,
        num_classes=1, a_feature_size=8, v_feature_size=16,
        text_feature_size=12, n_relations=3, relation_dim=8,
        binding_window=2, binding_temperature=.2, topk_divisor=4)


def main():
    torch.manual_seed(7)
    model = RelationV3(cfg()).eval()
    b, t = 2, 9
    lengths = torch.tensor([9, 6])
    valid = torch.arange(t)[None] < lengths[:, None]
    fa, fv, ft = torch.randn(b,t,8), torch.randn(b,t,16), torch.randn(b,t,12)
    out = model(fa, fv, ft, lengths, valid)
    # Exact identity at initialization, not merely close in probability.
    assert torch.equal(out["delta_logit"], torch.zeros_like(out["delta_logit"]))
    assert torch.equal(out["frame_prob"],
                       torch.sigmoid(out["base_frame_logits"].squeeze(-1)) * valid)
    assert all(not p.requires_grad for p in model.macil.parameters())
    assert model.readout[0].in_features == 3 * cfg().n_relations
    assert out["transport"].shape == (b, 3, t, t)
    idx = torch.arange(t)
    illegal = ((idx[:,None]-idx[None,:]).abs()>2)[None,None]
    illegal = illegal | (~valid[:,None,:,None]) | (~valid[:,None,None,:])
    assert torch.equal(out["transport"][illegal.expand_as(out["transport"])],
                       torch.zeros_like(out["transport"][illegal.expand_as(out["transport"])]))
    # Dustbin makes row marginals unbalanced rather than forced to one.
    rows = out["transport"].sum(-1)
    assert bool((rows[valid[:,None,:].expand_as(rows)] < .999).any())
    loss = out["frame_prob"].sum(); loss.backward()
    assert model.readout[-1].weight.grad is not None
    assert model.relations.query.weight.grad is not None
    # No raw-frame shortcut: before the final layer moves, upstream relation
    # features receive zero gradient through the zero-initialized readout.
    assert torch.equal(model.relations.query.weight.grad,
                       torch.zeros_like(model.relations.query.weight.grad))
    print("Relation-V3 smoke: PASS")


if __name__ == "__main__": main()
