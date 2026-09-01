from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MarkedTemporalSplatMIL  # noqa: E402


def batch():
    feats = {"visual": torch.randn(2, 9, 4), "audio": torch.randn(2, 9, 3),
             "text": torch.randn(2, 9, 5)}
    mask = torch.tensor([[1] * 9, [1] * 6 + [0] * 3], dtype=torch.bool)
    return feats, mask


def main():
    torch.manual_seed(7)
    feats, mask = batch()
    for arm in ("point", "splat"):
        model = MarkedTemporalSplatMIL(
            {"visual": 4, "audio": 3, "text": 5}, arm,
            hidden=8, embed=4, dropout=0.0, k_proportion=3)
        output = model(feats, mask)
        assert output["prob"].shape == mask.shape
        assert torch.isfinite(output["prob"]).all()
        assert ((output["prob"] >= 0) & (output["prob"] <= 1)).all()
        assert not output["prob"][1, 6:].any()
        output["prob"].sum().backward()
        for name in model.modalities:
            assert model.experts[name].amplitude.weight.grad.abs().sum() > 0
            if arm == "splat":
                assert model.experts[name].duration.weight.grad.abs().sum() > 0

    valid = torch.ones(1, 4, dtype=torch.bool)
    point = MarkedTemporalSplatMIL({"visual": 4}, "point", 8, 4, 0.0, 3)
    amplitude = torch.tensor([[.2, .4, .6, .8]])
    mixture_a = torch.tensor([[[1., 0., 0., 0.], [.25, .25, .25, .25],
                               [0., 0., 1., 0.], [.1, .2, .3, .4]]])
    mixture_b = torch.softmax(torch.randn(1, 4, 4), dim=-1)
    rendered_a = point._render([amplitude], [mixture_a], valid)
    rendered_b = point._render([amplitude], [mixture_b], valid)
    assert torch.allclose(rendered_a, amplitude, atol=1e-6)
    assert torch.allclose(rendered_b, amplitude, atol=1e-6)
    second = torch.tensor([[.1, .3, .5, .7]])
    rendered_two = point._render(
        [amplitude, second], [mixture_a, mixture_b], valid)
    expected = 1.0 - (1.0 - amplitude) * (1.0 - second)
    assert torch.allclose(rendered_two, expected, atol=1e-6)
    print("marked splat model tests: PASS")


if __name__ == "__main__":
    main()
