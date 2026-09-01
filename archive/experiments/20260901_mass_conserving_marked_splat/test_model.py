import torch

from model import MassConservingMarkedSplatMIL


def test_each_center_scale_conserves_mass_at_boundaries():
    model = MassConservingMarkedSplatMIL({"visual": 2}, hidden=4, embed=3)
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    kernels = model._kernels(torch.device("cpu"), torch.float32)
    denominator = torch.zeros(2, 5, 4)
    for offset, vector in kernels.items():
        if abs(offset) >= 5:
            continue
        if offset >= 0:
            source, target = slice(0, 5 - offset), slice(offset, 5)
        else:
            source, target = slice(-offset, 5), slice(0, 5 + offset)
        valid = (mask[:, source] & mask[:, target]).unsqueeze(-1)
        denominator[:, source] += valid * vector
    for batch, length in enumerate((5, 3)):
        for center in range(length):
            assert torch.all(denominator[batch, center] > 0)
            total = torch.zeros(4)
            for offset, vector in kernels.items():
                target = center + offset
                if 0 <= target < length:
                    total += vector / denominator[batch, center]
            assert torch.allclose(total, torch.ones(4), atol=1e-6)


def test_renderer_has_duration_and_amplitude_gradients():
    model = MassConservingMarkedSplatMIL({"visual": 2}, hidden=4, embed=3)
    features = {"visual": torch.randn(2, 7, 2)}
    mask = torch.tensor([
        [True, True, True, True, True, True, True],
        [True, True, True, True, False, False, False],
    ])
    output = model(features, mask)
    output["prob"].sum().backward()
    assert model.experts["visual"].amplitude.weight.grad.abs().sum() > 0
    assert model.experts["visual"].duration.weight.grad.abs().sum() > 0
    assert torch.all(output["prob"][1, 4:] == 0)
