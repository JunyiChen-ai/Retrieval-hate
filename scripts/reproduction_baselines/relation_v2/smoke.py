#!/usr/bin/env python3
"""CPU invariants for Relation-V2 before any experiment."""

from types import SimpleNamespace

import torch

from relation_v2.model import MaskedTemporalEncoder, RelationV2


def cfg():
    return SimpleNamespace(
        hid_dim=32, ffn_dim=32, nhead=4, dropout=0., num_classes=1,
        a_feature_size=8, v_feature_size=16, text_feature_size=12,
        n_relations=3, relation_dim=8, binding_window=2,
        binding_temperature=.2, sinkhorn_iters=5, topk_divisor=4)


def main():
    torch.manual_seed(7)
    model = RelationV2(cfg())
    b, t = 2, 9
    lengths = torch.tensor([9, 6])
    valid = torch.arange(t)[None] < lengths[:, None]
    out = model(torch.randn(b, t, 8), torch.randn(b, t, 16),
                torch.randn(b, t, 12), lengths, valid)
    assert out["frame_prob"].shape == (b, t)
    assert out["relation_witness"].shape == (b, t, 3)
    assert out["transport"].shape == (b, 3, t, t)
    index = torch.arange(t)
    illegal = ((index[:, None] - index[None, :]).abs() > 2)[None, None]
    illegal = illegal | (~valid[:, None, :, None]) | (~valid[:, None, None, :])
    assert torch.equal(out["transport"][illegal.expand_as(out["transport"])],
                       torch.zeros_like(out["transport"][illegal.expand_as(out["transport"])]))
    assert torch.equal(out["frame_prob"][1, 6:], torch.zeros(3))
    loss = out["bag_prob"].sum(); loss.backward()
    assert model.relations.query.weight.grad is not None
    assert model.readout[-1].weight.grad is not None

    # Whole-model valid predictions must not depend on appended padding.
    model.eval()
    sa, sv, st = torch.randn(1, 6, 8), torch.randn(1, 6, 16), torch.randn(1, 6, 12)
    short_out = model(sa, sv, st, torch.tensor([6]))["frame_prob"]
    pa = torch.cat([sa, torch.zeros(1, 3, 8)], 1)
    pv = torch.cat([sv, torch.zeros(1, 3, 16)], 1)
    pt = torch.cat([st, torch.zeros(1, 3, 12)], 1)
    padded_out = model(pa, pv, pt, torch.tensor([6]))["frame_prob"]
    assert torch.allclose(short_out, padded_out[:, :6], atol=1e-6, rtol=1e-6)

    # Masked MACIL execution must reproduce upstream algebra when all rows are valid.
    ref = model.macil(sa, sv, torch.tensor([6]))[3]
    masked = model(sa, sv, st, torch.tensor([6]))["base_frame_logits"]
    assert torch.allclose(ref, masked, atol=1e-6, rtol=1e-6)

    encoder = MaskedTemporalEncoder(12, 32, 0.).eval()
    text = torch.randn(1, 6, 12)
    short = encoder(text, torch.ones(1, 6, dtype=torch.bool))
    padded = torch.cat([text, torch.zeros(1, 3, 12)], 1)
    long_mask = torch.tensor([[True] * 6 + [False] * 3])
    extended = encoder(padded, long_mask)
    assert torch.allclose(short, extended[:, :6], atol=1e-6, rtol=1e-6)
    print("Relation-V2 smoke: PASS")


if __name__ == "__main__":
    main()
