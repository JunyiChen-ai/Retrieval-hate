#!/usr/bin/env python3
"""Property tests for same-corpus benign insertion and alignment."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "reproduction_baselines"))
sys.path.insert(0, str(HERE))

from hate_common import data as hdata  # noqa: E402
from powa_macil.dataset import load_teacher_jsonl, usable_text_ids  # noqa: E402
from dataset import BenignInsertionDataset  # noqa: E402


TEACHER = (REPO / "results" / "reproduction" / "powa_macil" /
           "teacher_qwen2vl7b_train_2chunks.jsonl")


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = "hateclipseg"
        labels = hdata.load_labels(corpus)
        train_ids, _ = hdata.load_train_val(corpus, labels)
        train_ids = usable_text_ids(corpus, train_ids)
        cls.labels = labels
        cls.train_ids = train_ids
        cls.teacher = load_teacher_jsonl(str(TEACHER))

    def make(self, arm):
        return BenignInsertionDataset(
            "hateclipseg", self.train_ids, self.labels, 200, "snippet", "av",
            1, teacher_records=self.teacher, arm=arm, seed=234,
            min_donor_rows=12, max_donor_rows=36, boundary_buffer=3)

    def first_index(self, dataset, label):
        return next(i for i, video_id in enumerate(dataset.video_ids)
                    if dataset.labels[video_id] == label)

    def test_negative_donor_and_recipient_alignment(self):
        dataset = self.make("full")
        item = dataset[self.first_index(dataset, 1)]
        self.assertTrue(bool(item["has_insertion"]))
        self.assertEqual(self.labels[item["donor_id"]], 0)
        self.assertIn(item["donor_id"], self.train_ids)
        self.assertNotEqual(item["donor_id"], item["recipient_id"])
        original_length = int(item["orig_length"])
        augmented_length = int(item["aug_length"])
        donor_rows = int(item["donor_rows"])
        self.assertEqual(augmented_length, original_length + donor_rows)
        mapping = item["recipient_map"][:original_length]
        for original, augmented in (("orig_v", "aug_v"),
                                    ("orig_a", "aug_a"),
                                    ("orig_t", "aug_t")):
            torch.testing.assert_close(
                item[original][:original_length], item[augmented][mapping])
        expected_interior = donor_rows - 6
        self.assertEqual(int(item["donor_mask"].sum()), expected_interior)

    def test_negative_recipient_is_never_inserted(self):
        dataset = self.make("full")
        item = dataset[self.first_index(dataset, 0)]
        self.assertFalse(bool(item["has_insertion"]))
        self.assertEqual(item["donor_id"], "")
        self.assertEqual(int(item["donor_mask"].sum()), 0)

    def test_positive_donor_control_really_uses_positive(self):
        dataset = self.make("positive_donor")
        item = dataset[self.first_index(dataset, 1)]
        self.assertEqual(self.labels[item["donor_id"]], 1)
        self.assertNotEqual(item["donor_id"], item["recipient_id"])

    def test_epoch_index_choice_is_reproducible(self):
        dataset = self.make("full")
        index = self.first_index(dataset, 1)
        dataset.set_epoch(2)
        first = dataset[index]
        second = dataset[index]
        for key in ("donor_id", "insert_at", "donor_start", "donor_rows"):
            self.assertEqual(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
