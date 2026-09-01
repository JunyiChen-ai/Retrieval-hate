from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts/reproduction_baselines/multihateloc"
sys.path.insert(0, str(BASE))

from model import MultiHateLoc  # noqa: E402
from method import TemporalCoalitionCreditMIL  # noqa: E402


def inputs():
    torch.manual_seed(71)
    feats = {
        "visual": torch.randn(3, 7, 9),
        "audio": torch.randn(3, 7, 5),
        "text": torch.randn(3, 7, 6),
    }
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
    ], dtype=torch.bool)
    return feats, mask


def test_anchor_exact():
    dims = {"visual": 9, "audio": 5, "text": 6}
    torch.manual_seed(19)
    baseline = MultiHateLoc(dims, hidden=12, embed=8, dropout=0.0)
    torch.manual_seed(19)
    anchor = TemporalCoalitionCreditMIL(
        dims, alpha=0.0, arm="anchor", hidden=12, embed=8, dropout=0.0)
    baseline.eval()
    anchor.eval()
    assert set(baseline.state_dict()) == set(anchor.state_dict())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, anchor.state_dict()[name])
    feats, mask = inputs()
    reference = baseline(feats, mask)
    observed = anchor(feats, mask)
    for name in reference["probs"]:
        assert torch.equal(reference["probs"][name], observed["probs"][name])
    assert torch.equal(reference["weights"], observed["weights"])
    assert torch.equal(reference["fused_embed"], observed["fused_embed"])


def test_credit_efficiency_and_target():
    dims = {"visual": 9, "audio": 5, "text": 6}
    torch.manual_seed(23)
    model = TemporalCoalitionCreditMIL(
        dims, alpha=0.5, arm="aligned", hidden=12, embed=8, dropout=0.0)
    model.eval()
    feats, mask = inputs()
    output = model(feats, mask)
    credit_sum = output["coalition_signed_credit"].sum(-1)
    total_gain = (output["coalition_full_logit"] -
                  output["coalition_empty_logit"])
    assert torch.allclose(credit_sum, total_gain, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(output["coalition_target"]).all()
    assert torch.allclose(
        output["coalition_target"].sum(-1),
        torch.ones_like(total_gain), atol=1e-6)


def test_shift_and_gradient():
    dims = {"visual": 9, "audio": 5, "text": 6}
    torch.manual_seed(29)
    model = TemporalCoalitionCreditMIL(
        dims, alpha=0.5, arm="shifted", hidden=12, embed=8, dropout=0.0)
    model.train()
    feats, mask = inputs()
    output = model(feats, mask)
    synthetic = torch.arange(3 * 7 * 3, dtype=torch.float32).reshape(3, 7, 3)
    shifted = model._shift_targets(synthetic, mask)
    for index, length in enumerate((7, 5, 6)):
        assert torch.allclose(
            shifted[index, :length].sort(dim=0).values,
            synthetic[index, :length].sort(dim=0).values)
        assert not torch.equal(
            shifted[index, :length], synthetic[index, :length])
    lengths = torch.tensor([7, 5, 6])
    labels = torch.tensor([1.0, 0.0, 1.0])
    loss, aligned, trained, count = model.responsibility_terms(
        output, mask, lengths, labels)
    assert count > 0 and torch.isfinite(loss)
    assert 0 <= float(aligned) <= 1 and 0 <= float(trained) <= 1
    loss.backward()
    gradient = sum(float(parameter.grad.abs().sum())
                   for parameter in model.local_router.parameters()
                   if parameter.grad is not None)
    assert gradient > 0


if __name__ == "__main__":
    test_anchor_exact()
    test_credit_efficiency_and_target()
    test_shift_and_gradient()
    print("PASS")
