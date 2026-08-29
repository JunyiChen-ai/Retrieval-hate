#!/usr/bin/env python3
"""CPU shape/gradient/invariant checks for POWA-MACIL; no corpus I/O."""

from types import SimpleNamespace

import torch

from powa_macil.model import (AsynchronousWitnessBinder, POWAMACIL,
                              PolicyCompiledWitnessMIL)
from powa_macil.dataset import PowaTrainDataset, usable_text_ids
from macilsd import align
from hate_common import data as hdata


def args():
    return SimpleNamespace(dropout=0.0, nhead=4, hid_dim=128, ffn_dim=128,
                           v_feature_size=1024, a_feature_size=128,
                           text_feature_size=768, num_classes=1,
                           binding_window=3, binding_temperature=0.3,
                           sinkhorn_iters=6)


def main():
    torch.manual_seed(4)
    b, t = 2, 9
    model = POWAMACIL(args())
    out = model(torch.randn(b, t, 128), torch.randn(b, t, 1024),
                torch.randn(b, t, 768), torch.tensor([9, 7]),
                policy="mhclip_en")
    assert out["frame_prob"].shape == (b, t)
    assert out["bag_prob"].shape == (b,)
    assert torch.isfinite(out["transport"]).all()
    assert ((out["frame_prob"] >= 0) & (out["frame_prob"] <= 1)).all()
    loss = out["bag_prob"].sum() + out["primitive_logits"].square().mean()
    loss.backward()
    assert model.pef.primitive_head.weight.grad is not None
    assert model.awb.distance_penalty.grad is not None

    # No target evidence means no relational hate witness.
    awb = AsynchronousWitnessBinder(hidden=4, window=2)
    h = torch.tensor([[0.8, 0.1, 0.6]])
    g = torch.zeros_like(h)
    z, _ = awb(h, g, torch.randn(1, 3, 4), torch.randn(1, 3, 4))
    assert torch.equal(z, torch.zeros_like(z))

    # Lag and padding masks are structural: Sinkhorn must never reactivate a
    # forbidden edge, even when its row/column has no admissible mass.
    awb = AsynchronousWitnessBinder(hidden=4, window=1, sinkhorn_iters=4)
    h = torch.full((1, 5), 0.7)
    g = torch.full((1, 5), 0.6)
    valid = torch.tensor([[True, True, True, False, False]])
    _, transport = awb(h, g, torch.randn(1, 5, 4),
                       torch.randn(1, 5, 4), valid_mask=valid)
    index = torch.arange(5)
    forbidden = ((index[:, None] - index[None, :]).abs() > 1)[None]
    forbidden = forbidden | (~valid[:, :, None]) | (~valid[:, None, :])
    assert torch.equal(transport[forbidden],
                       torch.zeros_like(transport[forbidden]))

    # Policies really differ: a violence-only primitive is admitted by HCS,
    # not by HateMM. This catches an accidental anonymous-head collapse.
    p = torch.zeros(1, 3, 6)
    p[..., 2] = 0.9
    targeted = torch.zeros(1, 3)
    hm = PolicyCompiledWitnessMIL("hatemm").frame_probability(p, targeted)
    hs = PolicyCompiledWitnessMIL("hateclipseg").frame_probability(p, targeted)
    assert float(hm.max()) == 0.0 and float(hs.min()) > 0.89

    # One real archived sample exercises the text/snippet alignment and the
    # four-tensor training contract without loading an entire corpus.
    labels = hdata.load_labels("mhclip_en")
    ids = usable_text_ids("mhclip_en", hdata.load_split("mhclip_en", "train"))
    ds = PowaTrainDataset("mhclip_en", ids[:1], labels, max_seqlen=200,
                          grid="snippet", modality="av", crop_repeat=1)
    rv, ra, rt, _ = ds[0]
    assert rv.shape[0] == ra.shape[0] == rt.shape[0] == 200
    assert rv.shape[1] == align.V_DIM and ra.shape[1] == align.A_DIM
    assert rt.shape[1] == 768
    print("POWA-MACIL smoke: PASS")


if __name__ == "__main__":
    main()
