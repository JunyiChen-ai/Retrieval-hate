"""Content datasets with exact integration cells and optional window observations."""
import numpy as np
import torch
import hier_evidence_common as common
from temporal_measure import integration_cells


class TrainDataset(common.TrainDataset):
    def __init__(self, *args, verdicts=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.verdicts = verdicts

    def __getitem__(self, index):
        visual, audio, window, label = super().__getitem__(index)
        vid = self.video_ids[index // self.crop_repeat]
        _, duration, snip = self.cache[vid]
        n = min(len(snip), self.max_seqlen)
        extra = np.zeros((self.max_seqlen, 36), dtype=np.float32)
        extra[:n, :2] = integration_cells(snip, duration, n)
        if self.verdicts is not None:
            extra[:n, 2:] = self.verdicts[vid]
        return visual, torch.cat([audio, torch.from_numpy(extra)], -1), window, label


class EvalDataset(common.EvalDataset):
    def __init__(self, *args, verdicts=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.verdicts = verdicts

    def __getitem__(self, index):
        visual, audio, index_map, duration, vid = super().__getitem__(index)
        snip = self.cache[vid][2]
        extra = np.zeros((len(snip), 36), dtype=np.float32)
        extra[:, :2] = integration_cells(snip, duration)
        if self.verdicts is not None:
            extra[:, 2:] = self.verdicts[vid]
        extra = torch.from_numpy(extra)[None].expand(visual.shape[0], -1, -1)
        return visual, torch.cat([audio, extra], -1), index_map, duration, vid


def content_normalization(cache, train_ids):
    """Train-only streaming statistics, visual crop0, original content row grid."""
    total = np.zeros(1920, dtype=np.float64)
    squares = total.copy()
    count = 0
    for vid in train_ids:
        audio, duration, snip = cache[vid]
        visual = common.align.aligned_visual_crop(cache.corpus, vid, 0, 'snippet', duration, snip)
        rows = np.concatenate([visual, audio[:, :common.SCAF_OFFSET]], -1).astype(np.float64)
        if rows.shape[1] != 1920 or not np.isfinite(rows).all():
            raise ValueError(f'invalid train content: {vid}')
        total += rows.sum(0)
        squares += np.square(rows).sum(0)
        count += len(rows)
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 0)).clip(1e-4)
    return mean, std
