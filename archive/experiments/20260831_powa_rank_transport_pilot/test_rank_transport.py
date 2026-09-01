#!/usr/bin/env python3
"""Mechanism-level tests for frozen-score temporal assignment."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASELINES = REPO / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASELINES))

from model import FrozenPowaTemporalAssignment, safe_logit  # noqa: E402
from train import stable_transport  # noqa: E402
from infer import claim_one_shot_test, require_test_authorization  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from powa_macil.dataset import usable_text_ids  # noqa: E402
from src.weak_supervision.same_corpus_insertion import (  # noqa: E402
    SameCorpusInsertionDataset,
    _shifted_control_mask,
)


class DummyPowa(nn.Module):
    def __init__(self):
        super().__init__()
        self.sentinel = nn.Parameter(torch.tensor(1.0))

    def forward(self, f_a, f_v, f_t, lengths, valid_mask, policy):
        batch, width = f_a.shape[:2]
        device = f_a.device
        base = torch.linspace(-2, 2, width, device=device)[None].repeat(batch, 1)
        base = base + 0.0 * self.sentinel
        return {
            "frame_prob": torch.sigmoid(base),
            "audio_rep": torch.zeros(batch, width, 128, device=device),
            "visual_rep": torch.zeros(batch, width, 128, device=device),
            "primitive_logits": torch.zeros(batch, width, 6, device=device),
            "base_frame_logits": base[..., None],
        }


def test_transport_is_exact_permutation_and_follows_order():
    values = np.asarray([0.8, 0.1, 0.4, 0.2])
    order = np.asarray([2.0, -1.0, 3.0, 0.0])
    output = stable_transport(values, order)
    assert np.array_equal(np.sort(output), np.sort(values))
    assert np.array_equal(np.argsort(output, kind="stable"),
                          np.argsort(order, kind="stable"))


def test_tie_secondary_changes_only_tied_positions():
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    order = np.asarray([0.0, 0.0, 1.0, 1.0])
    forward = stable_transport(values, order, np.arange(4))
    reverse = stable_transport(values, order, -np.arange(4))
    assert np.array_equal(np.sort(forward), np.sort(reverse))
    assert set(np.nonzero(forward != reverse)[0]) == {0, 1, 2, 3}


def test_zero_initialization_is_identity_order_and_anchor_frozen():
    powa = DummyPowa()
    model = FrozenPowaTemporalAssignment(powa)
    batch, width = 2, 9
    f_a = torch.randn(batch, width, 128)
    f_v = torch.randn(batch, width, 1024)
    f_t = torch.randn(batch, width, 768)
    lengths = torch.tensor([9, 6])
    valid = torch.arange(width)[None] < lengths[:, None]
    output = model(f_a, f_v, f_t, lengths, valid, "hatemm")
    assert torch.equal(output["order_residual"], torch.zeros_like(
        output["order_residual"]))
    assert torch.allclose(
        output["order_logit"][valid],
        safe_logit(output["anchor_prob"])[valid],
        atol=0,
        rtol=0,
    )
    assert not any(parameter.requires_grad for parameter in model.powa.parameters())
    assert all(parameter.requires_grad for parameter in model.order_head.parameters())


def test_padding_never_receives_residual():
    model = FrozenPowaTemporalAssignment(DummyPowa())
    batch, width = 1, 10
    lengths = torch.tensor([4])
    valid = torch.arange(width)[None] < lengths[:, None]
    output = model(
        torch.randn(batch, width, 128),
        torch.randn(batch, width, 1024),
        torch.randn(batch, width, 768),
        lengths,
        valid,
        "hatemm",
    )
    assert torch.equal(
        output["order_residual"][~valid],
        torch.zeros_like(output["order_residual"][~valid]),
    )


def test_shifted_control_is_same_length_and_disjoint_from_donor():
    aug_length, insert_at, donor_rows, buffer = 80, 30, 20, 3
    interior_rows = donor_rows - 2 * buffer
    shifted = _shifted_control_mask(
        aug_length, insert_at, donor_rows, interior_rows, buffer
    )
    donor = np.zeros(aug_length)
    donor[insert_at + buffer:insert_at + donor_rows - buffer] = 1
    assert int(shifted.sum()) == interior_rows
    assert not np.any((shifted > 0) & (donor > 0))
    donor_interior = np.arange(insert_at + buffer,
                               insert_at + donor_rows - buffer)
    assert not np.intersect1d(np.flatnonzero(shifted), donor_interior).size


def _small_hcs_ids():
    labels = hdata.load_labels("hateclipseg")
    train_ids, _ = hdata.load_train_val("hateclipseg", labels)
    train_ids = usable_text_ids("hateclipseg", train_ids)
    positive = [video_id for video_id in train_ids if labels[video_id] == 1][:2]
    negative = [video_id for video_id in train_ids if labels[video_id] == 0][:2]
    return labels, positive + negative


def _real_dataset(arm):
    labels, ids = _small_hcs_ids()
    return SameCorpusInsertionDataset(
        "hateclipseg", ids, labels, 200, "snippet", "av", 1,
        arm=arm, seed=234, min_donor_rows=12, max_donor_rows=36,
        boundary_buffer=3,
    ), labels


def test_real_negative_insertion_is_deterministic_and_same_corpus():
    dataset, labels = _real_dataset("negative_donor")
    first = dataset[0]
    repeated = dataset[0]
    assert first["recipient_id"] == repeated["recipient_id"]
    assert first["donor_id"] == repeated["donor_id"]
    assert labels[first["recipient_id"]] == 1
    assert labels[first["donor_id"]] == 0
    assert int(first["donor_rows"]) >= 12
    assert first["donor_id"] in dataset.video_ids
    for key in ("aug_v", "aug_a", "aug_t", "donor_mask",
                "supervision_mask", "recipient_map"):
        assert torch.equal(first[key], repeated[key])
    n = int(first["orig_length"])
    mapped = first["recipient_map"][:n]
    assert torch.equal(first["orig_v"][:n], first["aug_v"][mapped])
    assert torch.equal(first["orig_a"][:n], first["aug_a"][mapped])
    assert torch.equal(first["orig_t"][:n], first["aug_t"][mapped])
    assert torch.equal(first["donor_mask"], first["supervision_mask"])


def test_real_positive_and_shifted_controls_have_frozen_semantics():
    positive, labels = _real_dataset("positive_donor")
    positive_item = positive[0]
    assert labels[positive_item["donor_id"]] == 1
    shifted, _ = _real_dataset("shifted_mask")
    shifted_item = shifted[0]
    donor = shifted_item["donor_mask"] > 0
    supervision = shifted_item["supervision_mask"] > 0
    assert int(donor.sum()) == int(supervision.sum())
    assert not bool((donor & supervision).any())


def test_negative_and_shifted_interventions_are_exactly_matched():
    negative, _ = _real_dataset("negative_donor")
    shifted, _ = _real_dataset("shifted_mask")
    for epoch in (1, 2, 5):
        negative.set_epoch(epoch)
        shifted.set_epoch(epoch)
        for index in range(len(negative)):
            left = negative[index]
            right = shifted[index]
            for key in (
                "has_insertion", "recipient_id", "donor_id", "donor_crop",
                "donor_start", "donor_rows", "insert_at", "aug_length",
                "recipient_map", "donor_mask",
            ):
                left_value, right_value = left[key], right[key]
                if torch.is_tensor(left_value):
                    assert torch.equal(left_value, right_value), (epoch, index, key)
                else:
                    assert left_value == right_value, (epoch, index, key)
            for key in ("aug_v", "aug_a", "aug_t"):
                assert torch.equal(left[key], right[key]), (epoch, index, key)


def test_positive_control_matches_arm_independent_draws():
    negative, _ = _real_dataset("negative_donor")
    positive, _ = _real_dataset("positive_donor")
    for epoch in (1, 5):
        negative.set_epoch(epoch)
        positive.set_epoch(epoch)
        for index in range(2):
            left, right = negative[index], positive[index]
            for key in ("recipient_id", "donor_crop", "donor_rows", "insert_at"):
                left_value, right_value = left[key], right[key]
                if torch.is_tensor(left_value):
                    assert torch.equal(left_value, right_value), (epoch, index, key)
                else:
                    assert left_value == right_value, (epoch, index, key)


def test_hatemm_shifted_previously_failing_item_is_now_feasible():
    labels = hdata.load_labels("hatemm")
    train_ids, _ = hdata.load_train_val("hatemm", labels)
    train_ids = usable_text_ids("hatemm", train_ids)
    kwargs = dict(
        seed=234, min_donor_rows=12, max_donor_rows=36, boundary_buffer=3,
    )
    negative = SameCorpusInsertionDataset(
        "hatemm", train_ids, labels, 200, "snippet", "av", 5,
        arm="negative_donor", **kwargs,
    )
    shifted = SameCorpusInsertionDataset(
        "hatemm", train_ids, labels, 200, "snippet", "av", 5,
        arm="shifted_mask", **kwargs,
    )
    negative.set_epoch(1)
    shifted.set_epoch(1)
    left, right = negative[7], shifted[7]
    assert left["recipient_id"] == "hate_video_103"
    for key in ("donor_id", "donor_crop", "donor_start", "donor_rows",
                "insert_at", "aug_length", "donor_mask"):
        left_value, right_value = left[key], right[key]
        if torch.is_tensor(left_value):
            assert torch.equal(left_value, right_value), key
        else:
            assert left_value == right_value, key
    assert not bool(((right["donor_mask"] > 0)
                     & (right["supervision_mask"] > 0)).any())


def test_test_inference_fails_closed_without_stage_v_authority():
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "hatemm_negative_donor_seed234"
        checkpoint.mkdir()
        try:
            require_test_authorization(
                checkpoint,
                {"corpus": "hatemm", "arm": "negative_donor", "seed": 234},
            )
        except RuntimeError as error:
            assert "requires Stage-V" in str(error)
        else:
            raise AssertionError("test authorization unexpectedly opened")


def test_one_shot_test_claim_is_canonical_and_atomic():
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "hatemm_negative_donor_seed234"
        checkpoint.mkdir()
        canonical = checkpoint / "test_scores.jsonl"
        claim = claim_one_shot_test(checkpoint, canonical)
        assert claim.is_file()
        try:
            claim_one_shot_test(checkpoint, canonical)
        except RuntimeError as error:
            assert "already claimed" in str(error)
        else:
            raise AssertionError("second test inference claim unexpectedly opened")
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "hatemm_negative_donor_seed234"
        checkpoint.mkdir()
        try:
            claim_one_shot_test(checkpoint, checkpoint / "alternate.jsonl")
        except RuntimeError as error:
            assert "must be canonical" in str(error)
        else:
            raise AssertionError("alternate test output unexpectedly opened")
        assert not (checkpoint / "test_inference.claim.json").exists()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
