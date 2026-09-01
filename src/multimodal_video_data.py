"""Shared 1 fps audio/visual/text video dataset and padded batching."""

from __future__ import annotations

import torch
import torch.utils.data as tdata

from src.hate_local_features import aligned_local_features


MODALITIES = ("audio", "visual", "text")
DIMS = {"audio": 128, "visual": 1024, "text": 768}


class MultimodalVideoDataset(tdata.Dataset):
    def __init__(self, corpus: str, video_ids, labels):
        self.corpus = corpus
        self.video_ids = list(video_ids)
        self.labels = labels
        if set(self.video_ids) - set(labels):
            raise RuntimeError("dataset video ids are missing scoped labels")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        parts = aligned_local_features(self.corpus, video_id)
        feats = {name: torch.from_numpy(parts[name]) for name in MODALITIES}
        lengths = {len(value) for value in feats.values()}
        if len(lengths) != 1 or next(iter(lengths)) <= 0:
            raise RuntimeError(f"unaligned local features for {self.corpus}/{video_id}")
        return feats, float(self.labels[video_id]), video_id


def collate_multimodal_videos(items):
    lengths = torch.tensor([len(item[0]["audio"]) for item in items], dtype=torch.long)
    longest = int(lengths.max())
    feats = {
        name: torch.zeros(len(items), longest, DIMS[name], dtype=torch.float32)
        for name in MODALITIES
    }
    for row, (parts, _, _) in enumerate(items):
        length = int(lengths[row])
        for name in MODALITIES:
            feats[name][row, :length] = parts[name]
    mask = torch.arange(longest)[None, :] < lengths[:, None]
    labels = torch.tensor([item[1] for item in items], dtype=torch.float32)
    return feats, labels, lengths, mask, [item[2] for item in items]


def multimodal_loader(
    corpus, ids, labels, batch_size, workers, shuffle, seed, generator_state=None
):
    generator = torch.Generator().manual_seed(seed)
    if generator_state is not None:
        generator.set_state(generator_state)
    return tdata.DataLoader(
        MultimodalVideoDataset(corpus, ids, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_multimodal_videos,
        generator=generator,
        drop_last=False,
    )
