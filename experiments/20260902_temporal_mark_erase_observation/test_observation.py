import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("mark_erase_eval", HERE / "evaluate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ObservationTest(unittest.TestCase):
    def test_densify_exact_overlap(self):
        row = {
            "length": 6,
            "windows": [
                {"span": [0, 4], "marked": {"score": 6}, "erased": {"score": 2}},
                {"span": [2, 6], "marked": {"score": 8}, "erased": {"score": 3}},
            ],
        }
        np.testing.assert_allclose(module.densify(row, "marked"), [6, 6, 7, 7, 8, 8])
        np.testing.assert_allclose(module.densify(row, "contrast"), [4, 4, 4.5, 4.5, 5, 5])


if __name__ == "__main__":
    unittest.main()
