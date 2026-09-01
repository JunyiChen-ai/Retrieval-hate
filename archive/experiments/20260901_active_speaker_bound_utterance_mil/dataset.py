"""POWA data plus frozen source-bound face assignments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from macilsd import align
from macilsd.utils import process_feat
from powa_macil.dataset import (PowaTestDataset, PowaTrainDataset, TEXT_DIM,
                                text_path)


ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / "data" / "active_speaker_bound"


def load_source(corpus, video_id, grid, n_seconds, snippet_bounds, arm):
    path = CACHE_ROOT / corpus / f"{video_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as record:
        face_key = "permuted_face" if arm == "permuted" else "core_face"
        face = np.asarray(record[face_key], dtype=np.float32)
        state = np.asarray(record["source_state"], dtype=np.int64)
    if face.shape != (n_seconds, 512) or state.shape != (n_seconds,):
        raise ValueError(f"malformed source cache {path}: {face.shape}/{state.shape}")
    utterance = np.load(text_path(corpus, video_id)).astype(np.float32)
    if utterance.ndim != 2 or utterance.shape[1] != TEXT_DIM:
        raise ValueError(f"malformed utterance cache for {corpus}/{video_id}")
    if len(utterance) < n_seconds:
        utterance = np.pad(utterance, ((0, n_seconds - len(utterance)), (0, 0)))
    elif len(utterance) > n_seconds:
        utterance = utterance[:n_seconds]
    destination = (snippet_bounds if grid == "snippet"
                   else align.second_bounds(n_seconds))
    # A source-bound unit is indivisible. Face, categorical state and the
    # relation-only utterance therefore use the same destination midpoint;
    # averaging them separately would create cross-second face×text terms.
    midpoint = destination.mean(1)
    index = np.clip(np.floor(midpoint).astype(np.int64), 0, n_seconds - 1)
    face = face[index]
    state = state[index]
    utterance = utterance[index]
    return face, state, utterance


class SourceTrainDataset(PowaTrainDataset):
    def __init__(self, *args, arm="core", **kwargs):
        self.arm = arm
        super().__init__(*args, **kwargs)
        self._source = {}
        for vid in self.video_ids:
            n_seconds, snippets = self._meta[vid]
            self._source[vid] = load_source(
                self.corpus, vid, self.grid, n_seconds, snippets, self.arm)

    def __getitem__(self, index):
        base = super().__getitem__(index)
        vid = self.video_ids[index // self.crop_repeat]
        face, state, utterance = self._source[vid]
        face = process_feat(face, self.max_seqlen, is_random=False)
        state = process_feat(state[:, None], self.max_seqlen,
                             is_random=False)[:, 0].astype(np.int64)
        utterance = process_feat(
            utterance, self.max_seqlen, is_random=False)
        return base + (torch.from_numpy(np.ascontiguousarray(face)),
                       torch.from_numpy(np.ascontiguousarray(state)),
                       torch.from_numpy(np.ascontiguousarray(utterance)))


class SourceTestDataset(PowaTestDataset):
    def __init__(self, *args, arm="core", **kwargs):
        self.arm = arm
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        base = super().__getitem__(index)
        f_v, _, _, _, n_seconds, vid = base
        snippets = align.snippet_bounds(self.corpus, vid)
        face, state, utterance = load_source(
            self.corpus, vid, self.grid, n_seconds, snippets, self.arm)
        face = np.repeat(face[None], f_v.shape[0], axis=0)
        state = np.repeat(state[None], f_v.shape[0], axis=0)
        utterance = np.repeat(utterance[None], f_v.shape[0], axis=0)
        return base + (torch.from_numpy(np.ascontiguousarray(face)),
                       torch.from_numpy(np.ascontiguousarray(state)),
                       torch.from_numpy(np.ascontiguousarray(utterance)))
