from __future__ import annotations

import math
import itertools

import torch

from model import FactorialWitnessCRF


def brute_force_positive(model, unary):
    emissions = model._emissions(unary)
    transition = model._transition(emissions.dtype, emissions.device)
    values = []
    states = emissions.shape[-1]
    for path_index in range(states ** len(emissions)):
        path, value = [], path_index
        for _ in range(len(emissions)):
            path.append(value % states)
            value //= states
        if not any(path):
            continue
        energy = emissions[0, path[0]]
        for time in range(1, len(path)):
            energy = energy + transition[path[time - 1], path[time]] + emissions[time, path[time]]
        values.append(energy)
    return torch.logsumexp(torch.stack(values), dim=0)


def brute_force_posteriors(model, unary):
    emissions = model._emissions(unary)
    transition = model._transition(emissions.dtype, emissions.device)
    states = emissions.shape[-1]
    paths, energies = [], []
    for path in itertools.product(range(states), repeat=len(emissions)):
        if not any(path):
            continue
        energy = emissions[0, path[0]]
        for time in range(1, len(path)):
            energy = energy + transition[path[time - 1], path[time]] + emissions[time, path[time]]
        paths.append(path)
        energies.append(energy)
    energies = torch.stack(energies)
    log_z = torch.logsumexp(energies, dim=0)
    active = torch.stack([
        torch.exp(torch.logsumexp(
            energies[torch.tensor([path[t] != 0 for path in paths])], dim=0
        ) - log_z)
        for t in range(len(emissions))
    ])
    if states == 2:
        return active, active[:, None]
    bits = torch.stack([
        torch.stack([
            torch.exp(torch.logsumexp(
                energies[torch.tensor([
                    bool(model.state_bits[path[t], bit]) for path in paths
                ])], dim=0
            ) - log_z)
            for bit in range(3)
        ])
        for t in range(len(emissions))
    ])
    return active, bits


def test_positive_partition_matches_enumeration():
    torch.manual_seed(4)
    for arm in FactorialWitnessCRF.ARMS:
        model = FactorialWitnessCRF(arm, hidden=4, dropout=0.0)
        unary = torch.randn(2, 3) * 0.2
        emissions = model._emissions(unary)
        transition = model._transition(emissions.dtype, emissions.device)
        exact = model._positive_partition(emissions, transition)
        brute = brute_force_positive(model, unary)
        assert torch.allclose(exact, brute, atol=1e-5), (arm, exact, brute)


def test_zero_evidence_length_normalization():
    for arm in FactorialWitnessCRF.ARMS:
        model = FactorialWitnessCRF(arm, hidden=4, dropout=0.0)
        for length in (1, 2, 5, 20):
            bag, _, _, _ = model._one_video(torch.zeros(length, 3))
            assert math.isclose(float(bag), 0.0, abs_tol=2e-5), (arm, length, bag)


def test_posterior_is_finite_and_bounded():
    model = FactorialWitnessCRF("core", hidden=4, dropout=0.0)
    bag, score, posterior, bit_posterior = model._one_video(torch.randn(7, 3))
    assert torch.isfinite(bag)
    assert torch.isfinite(score).all()
    assert ((posterior >= 0) & (posterior <= 1)).all()
    assert bit_posterior.shape == (7, 3)
    assert torch.isfinite(bit_posterior).all()
    assert ((bit_posterior >= 0) & (bit_posterior <= 1)).all()


def test_forward_backward_posteriors_match_enumeration():
    torch.manual_seed(7)
    for arm in FactorialWitnessCRF.ARMS:
        model = FactorialWitnessCRF(arm, hidden=4, dropout=0.0)
        unary = torch.randn(2, 3) * 0.2
        _, _, active, bits = model._one_video(unary)
        brute_active, brute_bits = brute_force_posteriors(model, unary)
        assert torch.allclose(active, brute_active, atol=1e-5), arm
        assert torch.allclose(bits, brute_bits, atol=1e-5), arm


def test_all_mechanism_parameters_receive_gradient():
    for arm in FactorialWitnessCRF.ARMS:
        model = FactorialWitnessCRF(arm, hidden=4, dropout=0.0)
        unary = torch.randn(4, 3, requires_grad=True)
        bag, score, _, _ = model._one_video(unary)
        (bag + score.sum()).backward()
        assert unary.grad is not None and torch.isfinite(unary.grad).all()
        if arm != "collapsed":
            assert model.raw_pair_cost.grad is not None
            assert torch.isfinite(model.raw_pair_cost.grad).all()
        if arm != "zero_transition":
            assert model.raw_switch_cost.grad is not None
            assert torch.isfinite(model.raw_switch_cost.grad).all()
