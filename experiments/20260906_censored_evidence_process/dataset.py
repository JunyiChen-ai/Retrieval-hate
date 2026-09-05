"""Reuse the fixed content datasets; add integration cells and train-only targets."""
import numpy as np
import torch
import hier_evidence_common as common
from temporal_measure import integration_cells


class TrainDataset(common.TrainDataset):
    def __init__(self, *args, verdicts, **kwargs):
        super().__init__(*args, **kwargs)
        self.verdicts = verdicts

    def __getitem__(self, index):
        visual, audio, window, label = super().__getitem__(index)
        vid = self.video_ids[index//self.crop_repeat]
        _, duration, snip = self.cache[vid]
        n = min(len(snip), self.max_seqlen)
        extra = np.zeros((self.max_seqlen, 36), dtype=np.float32)
        extra[:n, :2] = integration_cells(snip, duration, n)
        if self.verdicts is not None:
            extra[:n, 2:] = self.verdicts[vid]
        return visual, torch.cat([audio, torch.from_numpy(extra)], -1), window, label


class EvalDataset(common.EvalDataset):
    def __getitem__(self, index):
        visual, audio, index_map, duration, vid = super().__getitem__(index)
        snip = self.cache[vid][2]
        # No validation/test VLM targets are read or supplied.
        extra = np.zeros((len(snip), 36), dtype=np.float32)
        extra[:, :2] = integration_cells(snip, duration)
        extra = torch.from_numpy(extra)[None].expand(visual.shape[0], -1, -1)
        return visual, torch.cat([audio, extra], -1), index_map, duration, vid
