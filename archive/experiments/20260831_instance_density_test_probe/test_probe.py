#!/usr/bin/env python3

import unittest

import numpy as np

from src.instance_density import tie_neutral_transport


class ProbeTests(unittest.TestCase):
    def test_all_ties_are_identity(self):
        anchor = np.asarray([.8, .1, .4, .2])
        np.testing.assert_array_equal(
            tie_neutral_transport(anchor, np.zeros(4)), anchor
        )

    def test_partial_ties_keep_anchor_order(self):
        anchor = np.asarray([.8, .1, .4, .2])
        output = tie_neutral_transport(anchor, [0., 0., 1., 1.])
        self.assertGreater(output[0], output[1])
        self.assertGreater(output[2], output[3])
        np.testing.assert_array_equal(np.sort(output), np.sort(anchor))

    def test_strict_order_assigns_exact_multiset(self):
        anchor = np.asarray([.8, .1, .4, .2])
        order = np.asarray([2., 0., 3., 1.])
        expected = np.empty_like(anchor)
        expected[np.argsort(order)] = np.sort(anchor)
        np.testing.assert_array_equal(
            tie_neutral_transport(anchor, order), expected
        )


if __name__ == "__main__":
    unittest.main()
