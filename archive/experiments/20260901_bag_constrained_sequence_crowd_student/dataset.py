"""Multimodal train data paired with frozen sequence-crowd targets."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.multimodal_video_data import (DIMS, MODALITIES,
                                       MultimodalVideoDataset)


class PosteriorTargetDataset(Dataset):
    def __init__(self, corpus, video_ids, labels, target_npz):
        self.base = MultimodalVideoDataset(corpus, video_ids, labels)
        with np.load(target_npz, allow_pickle=False) as record:
            if set(record.files) != set(video_ids):
                missing = sorted(set(video_ids) - set(record.files))[:5]
                extra = sorted(set(record.files) - set(video_ids))[:5]
                raise RuntimeError(f"target cohort mismatch missing={missing} extra={extra}")
            self.targets = {video_id: np.asarray(record[video_id], np.float32)
                            for video_id in video_ids}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        features, label, video_id = self.base[index]
        target = self.targets[video_id]
        length = len(features["audio"])
        if target.shape != (length,) or not np.isfinite(target).all():
            raise RuntimeError(f"target alignment/nonfinite {video_id}")
        return features, label, video_id, torch.from_numpy(target)


def collate_targets(items):
    lengths = torch.tensor([len(x[3]) for x in items], dtype=torch.long)
    longest = int(lengths.max())
    features = {name: torch.zeros(len(items), longest, DIMS[name])
                for name in MODALITIES}
    targets = torch.zeros(len(items), longest)
    for row, (parts, _, _, target) in enumerate(items):
        length = len(target)
        targets[row, :length] = target
        for name in MODALITIES:
            features[name][row, :length] = parts[name]
    mask = torch.arange(longest)[None] < lengths[:, None]
    labels = torch.tensor([x[1] for x in items], dtype=torch.float32)
    return features, labels, targets, lengths, mask, [x[2] for x in items]


def target_loader(corpus, ids, labels, target_npz, batch_size, workers,
                  shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(PosteriorTargetDataset(corpus, ids, labels, target_npz),
                      batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate_targets,
                      generator=generator, drop_last=False)
