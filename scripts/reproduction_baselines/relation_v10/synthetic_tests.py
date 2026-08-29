#!/usr/bin/env python3
import numpy as np
from relation_v10.copula import fit


def main():
    rng = np.random.default_rng(10); base = rng.normal(size=500)
    good = np.stack([base + .15*rng.normal(size=500) for _ in range(3)], 1)
    original = fit(good, .999); w = original["cluster_robust"]
    prediction = good @ w
    duplicated = np.column_stack([good, good[:, 0]])
    duplicate_fit = fit(duplicated, .999)
    duplicate_prediction = duplicated @ duplicate_fit["cluster_robust"]
    assert np.max(np.abs(prediction - duplicate_prediction)) < 1e-12
    noise = rng.normal(scale=4, size=500); noisy = np.column_stack([good, noise])
    robust = noisy @ fit(noisy, .999)["cluster_robust"]
    equal = noisy.mean(1)
    assert np.mean((robust-prediction)**2) < np.mean((equal-prediction)**2)
    assert abs(duplicate_fit["cluster_robust"].sum()-1) < 1e-12
    print("Relation-V10 duplicate invariance/noise robustness: PASS")


if __name__ == "__main__": main()
