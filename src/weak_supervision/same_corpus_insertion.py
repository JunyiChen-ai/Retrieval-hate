"""Deterministic same-corpus temporal insertion for weak supervision.

The class deliberately contains no loss or evaluation logic.  It exposes the
exact intervention and alignment masks needed by experiments while enforcing
that donor and recipient ids come from one caller-supplied corpus split.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from macilsd import align
from powa_macil.dataset import PowaTrainDataset


INSERTION_ARMS = {"negative_donor", "positive_donor", "shifted_mask"}


def _pad(array: np.ndarray, width: int) -> np.ndarray:
    if len(array) > width:
        raise ValueError(f"cannot pad length {len(array)} to smaller width {width}")
    return np.pad(
        array, ((0, width - len(array)), (0, 0)),
        mode="constant", constant_values=0,
    )


def _stable_int(text: str) -> int:
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
    )


def _stable_unit(text: str) -> float:
    return _stable_int(text) / float(2 ** 64)


def _shifted_control_mask(
    aug_length: int,
    insert_at: int,
    donor_rows: int,
    interior_rows: int,
    boundary_buffer: int,
) -> np.ndarray:
    """Move supervision to a valid interval disjoint from donor interior."""
    mask = np.zeros(aug_length, dtype=np.float32)
    donor_lo = insert_at + boundary_buffer
    donor_hi = insert_at + donor_rows - boundary_buffer
    left_hi = donor_lo
    right_lo = donor_hi
    candidates = []
    if left_hi >= interior_rows:
        candidates.append((left_hi - interior_rows, left_hi))
    if aug_length - right_lo >= interior_rows:
        candidates.append((right_lo, right_lo + interior_rows))
    if not candidates:
        raise RuntimeError("no valid interval disjoint from donor interior")
    # Prefer the longer side.  The interval may include the excluded donor
    # boundary rows: that is intentional for the seam-shortcut control, while
    # it never overlaps the label-certified donor interior used by the core.
    left_space = left_hi
    right_space = aug_length - right_lo
    lo, hi = candidates[0] if left_space >= right_space else candidates[-1]
    mask[lo:hi] = 1.0
    return mask


class SameCorpusInsertionDataset(PowaTrainDataset):
    """Return original/composite tensors plus exact intervention masks."""

    def __init__(
        self,
        *args,
        arm: str,
        seed: int,
        min_donor_rows: int = 12,
        max_donor_rows: int = 36,
        boundary_buffer: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if arm not in INSERTION_ARMS:
            raise ValueError(f"unknown insertion arm {arm}")
        self.arm = arm
        self.seed = int(seed)
        self.min_donor_rows = int(min_donor_rows)
        self.max_donor_rows = int(max_donor_rows)
        self.boundary_buffer = int(boundary_buffer)
        self.aug_width = self.max_seqlen + self.max_donor_rows
        self.epoch = 0
        self.negative_ids = [v for v in self.video_ids if self.labels[v] == 0]
        self.positive_ids = [v for v in self.video_ids if self.labels[v] == 1]
        self.negative_donor_ids = [
            video_id for video_id in self.negative_ids
            if len(self._audio[video_id]) >= self.min_donor_rows
        ]
        self.positive_donor_ids = [
            video_id for video_id in self.positive_ids
            if len(self._audio[video_id]) >= self.min_donor_rows
        ]
        if not self.negative_donor_ids:
            raise ValueError(f"{self.corpus} has no negative donor video")
        if arm == "positive_donor" and len(self.positive_donor_ids) < 2:
            raise ValueError(f"{self.corpus} needs two positive donor videos")
        if not (0 <= self.boundary_buffer * 2 < self.min_donor_rows):
            raise ValueError("boundary buffer must leave a donor interior")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _draw_key(self, index: int, name: str) -> str:
        # The arm is deliberately absent.  Negative-donor and shifted-mask
        # controls must receive the exact same intervention draw.  Named keys
        # also keep later fields invariant if another draw is added.
        return f"{self.seed}:{self.epoch}:{index}:{self.corpus}:{name}"

    def _selected(self, array: np.ndarray) -> np.ndarray:
        if len(array) <= self.max_seqlen:
            return np.asarray(array)
        indices = np.linspace(
            0, len(array) - 1, self.max_seqlen, dtype=np.int64
        )
        return np.asarray(array)[indices]

    def _raw(self, video_id: str, crop: int):
        n_seconds, snippet_bounds = self._meta[video_id]
        visual = align.aligned_visual_crop(
            self.corpus, video_id, crop, self.grid, n_seconds, snippet_bounds
        )
        audio = self._audio[video_id]
        text = self._text[video_id]
        if not (len(visual) == len(audio) == len(text)):
            raise RuntimeError(f"unaligned modalities for {video_id}")
        return visual, audio, text

    def __getitem__(self, index):
        recipient = self.video_ids[index // self.crop_repeat]
        crop = index % self.crop_repeat
        raw_v, raw_a, raw_t = self._raw(recipient, crop)
        orig_v = self._selected(raw_v).astype(np.float32, copy=False)
        orig_a = self._selected(raw_a).astype(np.float32, copy=False)
        orig_t = self._selected(raw_t).astype(np.float32, copy=False)
        orig_length = len(orig_v)
        label = float(self.labels[recipient])

        aug_v, aug_a, aug_t = orig_v, orig_a, orig_t
        aug_length = orig_length
        recipient_map = np.arange(orig_length, dtype=np.int64)
        stability_mask = np.ones(orig_length, dtype=np.float32)
        donor_mask = np.zeros(orig_length, dtype=np.float32)
        supervision_mask = np.zeros(orig_length, dtype=np.float32)
        donor_id = ""
        donor_crop = donor_start = insert_at = -1
        donor_rows = 0
        # With a three-row boundary buffer, a recipient shorter than three rows
        # cannot support a same-length shifted interval for the minimum
        # 12-row donor.  Exclude it identically in every arm before any draw.
        has_insertion = label == 1 and orig_length >= self.boundary_buffer

        if has_insertion:
            maximum_rows = min(
                self.max_donor_rows,
                orig_length + 3 * self.boundary_buffer,
            )
            if maximum_rows < self.min_donor_rows:
                raise RuntimeError("recipient feasibility calculation failed")
            donor_rows = self.min_donor_rows + (
                _stable_int(self._draw_key(index, "donor_rows"))
                % (maximum_rows - self.min_donor_rows + 1)
            )
            pool = (
                self.positive_donor_ids
                if self.arm == "positive_donor"
                else self.negative_donor_ids
            )
            eligible = [
                video_id for video_id in pool
                if video_id != recipient and len(self._audio[video_id]) >= donor_rows
            ]
            if not eligible:
                raise RuntimeError("no eligible same-corpus donor")
            donor_id = eligible[
                _stable_int(self._draw_key(index, "donor_id")) % len(eligible)
            ]
            donor_crop = (
                _stable_int(self._draw_key(index, "donor_crop")) % align.N_CROPS
            )
            donor_v, donor_a, donor_t = self._raw(donor_id, donor_crop)
            donor_positions = len(donor_v) - donor_rows + 1
            donor_start = min(
                donor_positions - 1,
                int(_stable_unit(self._draw_key(index, "donor_start"))
                    * donor_positions),
            )
            interior_rows = donor_rows - 2 * self.boundary_buffer
            feasible_insertions = [
                position for position in range(orig_length + 1)
                if max(
                    position + self.boundary_buffer,
                    orig_length - position + self.boundary_buffer,
                ) >= interior_rows
            ]
            if not feasible_insertions:
                raise RuntimeError("no feasible matched insertion point")
            insert_at = feasible_insertions[
                _stable_int(self._draw_key(index, "insert_at"))
                % len(feasible_insertions)
            ]
            donor_slice = slice(donor_start, donor_start + donor_rows)
            aug_v = np.concatenate(
                [orig_v[:insert_at], donor_v[donor_slice], orig_v[insert_at:]]
            )
            aug_a = np.concatenate(
                [orig_a[:insert_at], donor_a[donor_slice], orig_a[insert_at:]]
            )
            aug_t = np.concatenate(
                [orig_t[:insert_at], donor_t[donor_slice], orig_t[insert_at:]]
            )
            aug_length = len(aug_v)
            recipient_map = np.arange(orig_length, dtype=np.int64)
            recipient_map[insert_at:] += donor_rows
            donor_mask = np.zeros(aug_length, dtype=np.float32)
            donor_lo = insert_at + self.boundary_buffer
            donor_hi = insert_at + donor_rows - self.boundary_buffer
            donor_mask[donor_lo:donor_hi] = 1.0
            supervision_mask = donor_mask.copy()
            if self.arm == "shifted_mask":
                supervision_mask = _shifted_control_mask(
                    aug_length,
                    insert_at,
                    donor_rows,
                    donor_hi - donor_lo,
                    self.boundary_buffer,
                )
            stability_mask = np.ones(orig_length, dtype=np.float32)
            seam_lo = max(0, insert_at - self.boundary_buffer)
            seam_hi = min(orig_length, insert_at + self.boundary_buffer)
            stability_mask[seam_lo:seam_hi] = 0.0

        has_stability_support = bool(has_insertion and stability_mask.any())

        if aug_length > self.aug_width:
            raise RuntimeError("augmented sequence exceeds frozen width")
        padded_map = np.zeros(self.max_seqlen, dtype=np.int64)
        padded_map[:orig_length] = recipient_map
        padded_stability = np.zeros(self.max_seqlen, dtype=np.float32)
        padded_stability[:orig_length] = stability_mask
        padded_donor = np.zeros(self.aug_width, dtype=np.float32)
        padded_donor[:aug_length] = donor_mask
        padded_supervision = np.zeros(self.aug_width, dtype=np.float32)
        padded_supervision[:aug_length] = supervision_mask

        return {
            "orig_v": torch.from_numpy(_pad(orig_v, self.max_seqlen)),
            "orig_a": torch.from_numpy(_pad(orig_a, self.max_seqlen)),
            "orig_t": torch.from_numpy(_pad(orig_t, self.max_seqlen)),
            "orig_length": torch.tensor(orig_length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
            "aug_v": torch.from_numpy(
                _pad(np.asarray(aug_v, dtype=np.float32), self.aug_width)
            ),
            "aug_a": torch.from_numpy(
                _pad(np.asarray(aug_a, dtype=np.float32), self.aug_width)
            ),
            "aug_t": torch.from_numpy(
                _pad(np.asarray(aug_t, dtype=np.float32), self.aug_width)
            ),
            "aug_length": torch.tensor(aug_length, dtype=torch.long),
            "donor_mask": torch.from_numpy(padded_donor),
            "supervision_mask": torch.from_numpy(padded_supervision),
            "recipient_map": torch.from_numpy(padded_map),
            "stability_mask": torch.from_numpy(padded_stability),
            "has_insertion": torch.tensor(has_insertion, dtype=torch.bool),
            "has_stability_support": torch.tensor(
                has_stability_support, dtype=torch.bool
            ),
            "recipient_id": recipient,
            "donor_id": donor_id,
            "donor_crop": torch.tensor(donor_crop, dtype=torch.long),
            "insert_at": torch.tensor(insert_at, dtype=torch.long),
            "donor_start": torch.tensor(donor_start, dtype=torch.long),
            "donor_rows": torch.tensor(donor_rows, dtype=torch.long),
            "boundary_buffer": torch.tensor(
                self.boundary_buffer, dtype=torch.long
            ),
            "crop": torch.tensor(crop, dtype=torch.long),
            "epoch": torch.tensor(self.epoch, dtype=torch.long),
        }
