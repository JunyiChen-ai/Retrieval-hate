#!/usr/bin/env python3
"""Label-free synthetic contract checks for the block corruption mapping."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omnivtg_protocol import block_rotation_plan, inverse_mapped_interval_score  # noqa: E402


def main() -> None:
    plan = block_rotation_plan(32.0, 8.0)
    assert plan["order"] == [2, 3, 0, 1]
    # Content originally at seconds 0..7 moves to permuted seconds 16..23.
    recovered = inverse_mapped_interval_score([16.0, 24.0], 32, plan)
    expected = np.r_[np.ones(8), np.zeros(24)]
    assert np.array_equal(recovered, expected)
    # A fixed beginning-position shortcut maps back to original block 2.
    shortcut = inverse_mapped_interval_score([0.0, 8.0], 32, plan)
    expected_shortcut = np.r_[np.zeros(16), np.ones(8), np.zeros(8)]
    assert np.array_equal(shortcut, expected_shortcut)
    print("block rotation synthetic contract: PASS")


if __name__ == "__main__":
    main()
