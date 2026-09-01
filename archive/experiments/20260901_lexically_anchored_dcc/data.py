"""POWA training data with aligned OOF lexical support."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(BASE))

from macilsd import align  # noqa: E402
from macilsd.utils import process_feat  # noqa: E402
from powa_macil.dataset import PowaTrainDataset  # noqa: E402


class LexicalPowaTrainDataset(PowaTrainDataset):
    """Append lexical evidence, speech support and stable video index.

    Evidence is first mapped from the 1 fps grid to the exact POWA grid and
    only then passed through POWA's deterministic sequence processing.  The
    shifted control rolls the valid, unpadded sequence before processing.
    """

    def __init__(self, *args, evidence_dir: str, shift_evidence: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        evidence_root = Path(evidence_dir)
        evidence_path = evidence_root / self.corpus / "train_evidence.npz"
        speech_path = evidence_root / self.corpus / "train_speech.npz"
        if not evidence_path.is_file() or not speech_path.is_file():
            raise FileNotFoundError(
                f"missing lexical evidence for {self.corpus}: {evidence_root}")
        self.shift_evidence = bool(shift_evidence)
        self._lexical: dict[str, np.ndarray] = {}
        self._speech: dict[str, np.ndarray] = {}
        with np.load(evidence_path) as evidence, np.load(speech_path) as speech:
            if set(evidence.files) != set(self.video_ids):
                raise ValueError("OOF lexical evidence does not match train cohort")
            if set(speech.files) != set(self.video_ids):
                raise ValueError("OOF speech evidence does not match train cohort")
            for vid in self.video_ids:
                lexical_1fps = np.asarray(evidence[vid], dtype=np.float32)
                speech_1fps = np.asarray(speech[vid], dtype=np.float32)
                if (lexical_1fps.ndim != 1 or speech_1fps.shape != lexical_1fps.shape
                        or not np.isfinite(lexical_1fps).all()
                        or not np.isfinite(speech_1fps).all()):
                    raise ValueError(f"invalid OOF evidence for {self.corpus}/{vid}")
                n_seconds, snippet = self._meta[vid]
                dst = (snippet if self.grid == "snippet"
                       else align.second_bounds(n_seconds))
                src = align.second_bounds(len(lexical_1fps))
                lexical = align.resample_intervals(
                    lexical_1fps[:, None], src, dst)[:, 0]
                speech_grid = align.resample_intervals(
                    speech_1fps[:, None], src, dst)[:, 0]
                if self.shift_evidence and len(lexical) > 1:
                    offset = len(lexical) // 2
                    lexical = np.roll(lexical, offset)
                    speech_grid = np.roll(speech_grid, offset)
                self._lexical[vid] = process_feat(
                    lexical[:, None], self.max_seqlen, is_random=False)[:, 0]
                self._speech[vid] = process_feat(
                    speech_grid[:, None], self.max_seqlen, is_random=False)[:, 0]
        self._video_index = {vid: i for i, vid in enumerate(self.video_ids)}

    def __getitem__(self, index):
        base = super().__getitem__(index)
        vid = self.video_ids[index // self.crop_repeat]
        return base + (
            torch.from_numpy(np.ascontiguousarray(self._lexical[vid])),
            torch.from_numpy(np.ascontiguousarray(self._speech[vid])),
            self._video_index[vid],
        )
