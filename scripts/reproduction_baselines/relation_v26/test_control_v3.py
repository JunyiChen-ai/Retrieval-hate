import unittest
import torch

from train_control_v3 import permutation_v3


class VariableLengthControlTest(unittest.TestCase):
    def row(self, idx, length):
        x = [torch.full((length, d), float(idx)) for d in (2, 3, 4)]
        masks = [torch.ones(length, dtype=torch.bool) for _ in range(3)]
        b = [z + 0.25 for z in x]
        return {"id": f"v{idx}", "T": length, "X": x, "masks": masks, "oof_b": b, "G": torch.tensor(float(idx)), "y": torch.tensor(float(idx % 2))}

    def test_complete_tuple_derangement_preserves_global_multisets(self):
        rows = [self.row(0, 2), self.row(1, 3), self.row(2, 5), self.row(3, 7)]
        moved, mapping = permutation_v3(rows)
        self.assertTrue(all(x["nonself"] for x in mapping))
        self.assertEqual(sorted(x["donor"] for x in mapping), sorted(x["id"] for x in rows))
        self.assertEqual(sorted(x["T"] for x in moved), sorted(x["T"] for x in rows))
        self.assertEqual(sum(x["T"] for x in moved), sum(x["T"] for x in rows))
        for recipient in moved:
            donor_value = int(recipient["X"][0][0, 0].item())
            self.assertEqual(recipient["T"], rows[donor_value]["T"])
            self.assertEqual(float(recipient["G"]), float(recipient["id"][1:]))
            self.assertTrue(all(torch.equal(x + 0.25, b) for x, b in zip(recipient["X"], recipient["oof_b"])))


if __name__ == "__main__":
    unittest.main()
