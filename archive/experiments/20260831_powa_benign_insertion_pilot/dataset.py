"""Same-corpus negative-window insertion for the POWA pilot."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from macilsd import align
from powa_macil.dataset import PowaTrainDataset


INSERTION_ARMS = {
    "splice_only", "insertion_benign", "full", "positive_donor"
}


def _pad(array: np.ndarray, width: int) -> np.ndarray:
    if len(array) > width:
        raise ValueError(f"cannot pad length {len(array)} to smaller width {width}")
    return np.pad(array, ((0, width - len(array)), (0, 0)),
                  mode="constant", constant_values=0)


def _stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8],
                          "little")


class BenignInsertionDataset(PowaTrainDataset):
    """Return matched original/composite tensors and exact intervention masks."""

    def __init__(self, *args, arm: str, seed: int, min_donor_rows: int = 12,
                 max_donor_rows: int = 36, boundary_buffer: int = 3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.arm = arm
        self.seed = int(seed)
        self.min_donor_rows = int(min_donor_rows)
        self.max_donor_rows = int(max_donor_rows)
        self.boundary_buffer = int(boundary_buffer)
        self.aug_width = self.max_seqlen + self.max_donor_rows
        self.epoch = 0
        self.negative_ids = [v for v in self.video_ids if self.labels[v] == 0]
        self.positive_ids = [v for v in self.video_ids if self.labels[v] == 1]
        if arm in INSERTION_ARMS and not self.negative_ids:
            raise ValueError(f"{self.corpus} has no negative donor video")
        if arm == "positive_donor" and len(self.positive_ids) < 2:
            raise ValueError(f"{self.corpus} needs two positive videos for control")
        if not (0 <= self.boundary_buffer * 2 < self.min_donor_rows):
            raise ValueError("boundary buffer must leave donor interior")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        key = f"{self.seed}:{self.epoch}:{index}:{self.corpus}:{self.arm}"
        return np.random.default_rng(_stable_int(key))

    def _selected(self, array: np.ndarray) -> np.ndarray:
        if len(array) <= self.max_seqlen:
            return np.asarray(array)
        index = np.linspace(0, len(array) - 1, self.max_seqlen,
                            dtype=np.int64)
        return np.asarray(array)[index]

    def _raw(self, video_id: str, crop: int):
        n_seconds, snippet_bounds = self._meta[video_id]
        visual = align.aligned_visual_crop(
            self.corpus, video_id, crop, self.grid, n_seconds, snippet_bounds)
        audio = self._audio[video_id]
        text = self._text[video_id]
        if not (len(visual) == len(audio) == len(text)):
            raise RuntimeError(f"unaligned raw modalities for {video_id}")
        return visual, audio, text

    def _teacher_selected(self, video_id: str, real_length: int):
        if self.teacher_records is None:
            return (np.zeros((real_length, 6), dtype=np.float32),
                    np.zeros(real_length, dtype=np.float32))
        target, mask = self._teacher[video_id]
        if self.permute_teacher_channels:
            target = np.roll(target, 1, axis=-1).copy()
        return self._selected(target), self._selected(mask[:, None])[:, 0]

    def __getitem__(self, index):
        recipient = self.video_ids[index // self.crop_repeat]
        crop = index % self.crop_repeat
        rng = self._rng(index)
        raw_v, raw_a, raw_t = self._raw(recipient, crop)
        orig_v = self._selected(raw_v).astype(np.float32, copy=False)
        orig_a = self._selected(raw_a).astype(np.float32, copy=False)
        orig_t = self._selected(raw_t).astype(np.float32, copy=False)
        orig_length = len(orig_v)
        teacher_target, teacher_mask = self._teacher_selected(
            recipient, orig_length)
        label = float(self.labels[recipient])

        aug_v = orig_v
        aug_a = orig_a
        aug_t = orig_t
        aug_length = orig_length
        recipient_map = np.arange(orig_length, dtype=np.int64)
        consistency_mask = np.ones(orig_length, dtype=np.float32)
        donor_mask = np.zeros(orig_length, dtype=np.float32)
        donor_id = ""
        donor_crop = -1
        donor_start = -1
        insert_at = -1
        donor_rows = 0
        has_insertion = bool(label == 1 and self.arm in INSERTION_ARMS)

        if has_insertion:
            pool = (self.positive_ids if self.arm == "positive_donor"
                    else self.negative_ids)
            eligible = [v for v in pool if v != recipient]
            donor_id = eligible[int(rng.integers(len(eligible)))]
            donor_crop = int(rng.integers(align.N_CROPS))
            donor_v, donor_a, donor_t = self._raw(donor_id, donor_crop)
            maximum = min(self.max_donor_rows, len(donor_v))
            minimum = min(self.min_donor_rows, maximum)
            donor_rows = int(rng.integers(minimum, maximum + 1))
            donor_start = int(rng.integers(0, len(donor_v) - donor_rows + 1))
            insert_at = int(rng.integers(0, orig_length + 1))
            sl = slice(donor_start, donor_start + donor_rows)
            aug_v = np.concatenate([orig_v[:insert_at], donor_v[sl],
                                    orig_v[insert_at:]], axis=0)
            aug_a = np.concatenate([orig_a[:insert_at], donor_a[sl],
                                    orig_a[insert_at:]], axis=0)
            aug_t = np.concatenate([orig_t[:insert_at], donor_t[sl],
                                    orig_t[insert_at:]], axis=0)
            aug_length = len(aug_v)
            recipient_map = np.arange(orig_length, dtype=np.int64)
            recipient_map[insert_at:] += donor_rows
            donor_mask = np.zeros(aug_length, dtype=np.float32)
            lo = insert_at + self.boundary_buffer
            hi = insert_at + donor_rows - self.boundary_buffer
            donor_mask[lo:hi] = 1.0
            consistency_mask = np.ones(orig_length, dtype=np.float32)
            left = max(0, insert_at - self.boundary_buffer)
            right = min(orig_length, insert_at + self.boundary_buffer)
            consistency_mask[left:right] = 0.0

        if aug_length > self.aug_width:
            raise RuntimeError("augmented sequence exceeds frozen width")
        padded_map = np.zeros(self.max_seqlen, dtype=np.int64)
        padded_map[:orig_length] = recipient_map
        padded_consistency = np.zeros(self.max_seqlen, dtype=np.float32)
        padded_consistency[:orig_length] = consistency_mask
        padded_donor = np.zeros(self.aug_width, dtype=np.float32)
        padded_donor[:aug_length] = donor_mask

        return {
            "orig_v": torch.from_numpy(_pad(orig_v, self.max_seqlen)),
            "orig_a": torch.from_numpy(_pad(orig_a, self.max_seqlen)),
            "orig_t": torch.from_numpy(_pad(orig_t, self.max_seqlen)),
            "teacher_target": torch.from_numpy(
                _pad(teacher_target.astype(np.float32), self.max_seqlen)),
            "teacher_mask": torch.from_numpy(
                _pad(teacher_mask.astype(np.float32)[:, None],
                     self.max_seqlen)[:, 0]),
            "orig_length": torch.tensor(orig_length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
            "aug_v": torch.from_numpy(
                _pad(np.asarray(aug_v, dtype=np.float32), self.aug_width)),
            "aug_a": torch.from_numpy(
                _pad(np.asarray(aug_a, dtype=np.float32), self.aug_width)),
            "aug_t": torch.from_numpy(
                _pad(np.asarray(aug_t, dtype=np.float32), self.aug_width)),
            "aug_length": torch.tensor(aug_length, dtype=torch.long),
            "donor_mask": torch.from_numpy(padded_donor),
            "recipient_map": torch.from_numpy(padded_map),
            "consistency_mask": torch.from_numpy(padded_consistency),
            "has_insertion": torch.tensor(has_insertion, dtype=torch.bool),
            "recipient_id": recipient,
            "donor_id": donor_id,
            "donor_crop": torch.tensor(donor_crop, dtype=torch.long),
            "insert_at": torch.tensor(insert_at, dtype=torch.long),
            "donor_start": torch.tensor(donor_start, dtype=torch.long),
            "donor_rows": torch.tensor(donor_rows, dtype=torch.long),
            "boundary_buffer": torch.tensor(self.boundary_buffer,
                                               dtype=torch.long),
            "crop": torch.tensor(crop, dtype=torch.long),
            "epoch": torch.tensor(self.epoch, dtype=torch.long),
        }
