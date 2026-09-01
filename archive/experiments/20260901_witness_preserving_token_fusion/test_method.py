from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts/reproduction_baselines/multihateloc"
sys.path[:0] = [str(BASE), str(ROOT)]

from model import MultiHateLoc  # noqa: E402
from method import WitnessPreservingTokenFusion  # noqa: E402


def main():
    dims = {"visual": 4, "audio": 3, "text": 5}
    kwargs = {"hidden": 8, "embed": 6, "dropout": 0.0,
              "k_proportion": 2}
    torch.manual_seed(7)
    base = MultiHateLoc(dims, **kwargs)
    torch.manual_seed(7)
    anchor = WitnessPreservingTokenFusion(
        dims, alpha_fusion=0.0, arm="anchor", **kwargs)
    assert list(base.state_dict()) == list(anchor.state_dict())
    for name in base.state_dict():
        torch.testing.assert_close(base.state_dict()[name],
                                   anchor.state_dict()[name], rtol=0, atol=0)
    feats = {"visual": torch.randn(3, 6, 4),
             "audio": torch.randn(3, 6, 3),
             "text": torch.randn(3, 6, 5)}
    mask = torch.tensor([[1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 0, 0],
                         [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    base.eval(); anchor.eval()
    with torch.no_grad():
        left, right = base(feats, mask), anchor(feats, mask)
    for name in left["probs"]:
        torch.testing.assert_close(left["probs"][name], right["probs"][name],
                                   rtol=0, atol=0)
    torch.manual_seed(8)
    aligned = WitnessPreservingTokenFusion(
        dims, alpha_fusion=.5, arm="aligned", **kwargs)
    torch.manual_seed(8)
    shifted = WitnessPreservingTokenFusion(
        dims, alpha_fusion=.5, arm="shifted", **kwargs)
    aligned.load_state_dict(shifted.state_dict())
    aligned.eval(); shifted.eval()
    a = aligned(feats, mask)
    s = shifted(feats, mask)
    assert not torch.equal(a["probs"]["fused"], s["probs"]["fused"])
    labels = torch.tensor([1.0, 0.0, 1.0])
    lengths = mask.sum(1)
    loss, diag = aligned.gate_loss(a, mask, lengths, labels)
    assert torch.isfinite(loss) and diag["witness_count"] > 0
    total = aligned.mil_loss(a["probs"], mask, lengths, labels)[0] + .1 * loss
    total.backward()
    assert aligned.retain["visual"].weight.grad is not None
    assert aligned.projections["audio_to_visual"].weight.grad is not None
    shifted_valid = aligned._shift_valid(torch.arange(18).reshape(3, 6, 1), mask)
    torch.testing.assert_close(shifted_valid[1, 4:], torch.tensor([[10], [11]]))
    print("PASS")


if __name__ == "__main__":
    main()
