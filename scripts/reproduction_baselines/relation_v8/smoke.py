#!/usr/bin/env python3
import torch
from relation_v8.model import UnifiedRelationV8
from relation_v8.run import audit_id_set, select_candidate


def main():
    torch.manual_seed(7)
    scores = torch.rand(3, 11, 3)
    valid = torch.tensor([[1] * 11, [1] * 7 + [0] * 4, [1] * 4 + [0] * 7], dtype=torch.bool)
    model = UnifiedRelationV8(3, window=2).eval()
    fallback = model(scores, valid, 0., 0.)
    assert torch.equal(fallback["frame_score"], fallback["static_prior"])
    out = model(scores, valid, 1., .5)
    for key in ("static_locator", "relation_residual", "locator_correction", "correction"):
        for b in range(len(scores)):
            assert torch.allclose(out[key][b, valid[b]].mean(), torch.tensor(0.), atol=2e-7)
        assert torch.equal(out[key][~valid], torch.zeros_like(out[key][~valid]))
    # Per-expert video offsets do not change the transport residual.
    shifted = scores + torch.tensor([.2, -.1, .4])[None, None]
    shifted_out = model(shifted, valid, 1., .5)
    assert torch.allclose(out["relation_residual"], shifted_out["relation_residual"], atol=2e-6)
    rows = [
        {"beta": 0., "gamma": 0., "frame_ap": .5, "frame_roc": .7},
        {"beta": 1., "gamma": 0., "frame_ap": .6, "frame_roc": .69},
        {"beta": 1., "gamma": 1., "frame_ap": .55, "frame_roc": .71},
    ]
    selected, fallback, eligible = select_candidate(rows)
    assert selected == rows[2] and fallback == rows[0] and rows[1] not in eligible
    audit = audit_id_set({"a", "b", "legal_extra"}, {"a", "b"})
    assert audit["extra_count"] == 1 and audit["extra_ids_sorted"] == ["legal_extra"]
    try:
        audit_id_set({"a", "legal_extra"}, {"a", "b"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing frozen-GT source ID was accepted")
    print("Relation-V8 smoke: PASS")


if __name__ == "__main__": main()
