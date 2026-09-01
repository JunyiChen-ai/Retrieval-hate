import unittest

import numpy as np

from sequence_crowd import SequenceCrowdEM


class SequenceCrowdTests(unittest.TestCase):
    def test_positive_conditioning_is_common_scale_for_positive_marginal(self):
        q = np.array([[.8, .2], [.6, .4], [.9, .1]])
        xi = np.zeros((2, 2, 2))
        conditioned, _ = SequenceCrowdEM._condition_positive(q, xi, .5)
        ratio = conditioned[:, 1] / q[:, 1]
        np.testing.assert_allclose(ratio, np.repeat(ratio[0], 3))
        np.testing.assert_array_equal(np.argsort(q[:, 1]),
                                      np.argsort(conditioned[:, 1]))

    def test_edge_emission_changes_neighbor_posterior(self):
        observations = {
            "negative": np.zeros((5, 2), dtype=np.int64),
            "positive": np.array([[0, 0], [0, 0], [4, 4], [0, 0], [0, 0]])
        }
        labels = {"negative": 0, "positive": 1}
        sequential = SequenceCrowdEM(2, sequential=True, n_iter=6)
        token = SequenceCrowdEM(2, sequential=False, n_iter=6)
        q_sequence = sequential.fit(observations, labels)["positive"]
        q_token = token.fit(observations, labels)["positive"]
        self.assertFalse(np.allclose(q_sequence, q_token))
        self.assertGreater(q_sequence[2], q_sequence[0])


if __name__ == "__main__":
    unittest.main()
