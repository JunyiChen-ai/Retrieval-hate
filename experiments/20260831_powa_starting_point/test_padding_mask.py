#!/usr/bin/env python3
"""Regression checks for the optional attention padding mask repair."""

from __future__ import annotations

import os
import sys
import unittest

import torch


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))

from macilsd.Transformer import MultiHeadAttention  # noqa: E402


class PaddingMaskTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260831)
        self.attention = MultiHeadAttention(2, 8, dropout=0.0).eval()

    def test_all_valid_is_backward_compatible(self):
        x = torch.randn(2, 5, 8)
        old, _ = self.attention(x, x, x)
        valid = torch.ones(2, 5, dtype=torch.bool)
        masked, _ = self.attention(x, x, x, key_padding_mask=valid)
        torch.testing.assert_close(old, masked, rtol=0, atol=0)

    def test_padded_key_content_cannot_change_real_outputs(self):
        query = torch.randn(1, 5, 8)
        key = torch.randn(1, 5, 8)
        value = torch.randn(1, 5, 8)
        valid = torch.tensor([[True, True, True, False, False]])
        expected, _ = self.attention(
            query, key, value, key_padding_mask=valid)
        key[:, 3:] = torch.randn_like(key[:, 3:]) * 1000
        value[:, 3:] = torch.randn_like(value[:, 3:]) * 1000
        actual, _ = self.attention(
            query, key, value, key_padding_mask=valid)
        torch.testing.assert_close(expected[:, :3], actual[:, :3],
                                   rtol=1e-6, atol=1e-6)

    def test_bad_mask_shape_is_rejected(self):
        x = torch.randn(2, 5, 8)
        with self.assertRaises(ValueError):
            self.attention(x, x, x,
                           key_padding_mask=torch.ones(2, 4, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
