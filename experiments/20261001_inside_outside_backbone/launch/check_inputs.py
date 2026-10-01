"""Parse all required cache files and verify shape, coverage and split isolation."""
import argparse
import json
from pathlib import Path
import socket
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from qtl import data
from macilsd import align
import hier_evidence_common as hc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    a = ap.parse_args()
    labels, split, gt, _ = hc.load_fixed_cohort(a.corpus)
    for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(split[x]) & set(split[y]), (x, y)
    soft, times, cache_split = data.load_soft_p(a.corpus, "soft_both")
    count = 0
    for name in ("train", "val", "test"):
        for video in split[name]:
            assert video in labels and video in soft
            assert cache_split[video] == name
            visual = np.load(align.visual_path(a.corpus, video), mmap_mode="r")
            audio = np.load(align.audio_path(a.corpus, video), mmap_mode="r")
            text = np.load(hc.text_path(a.corpus, video, "bert"), mmap_mode="r")
            bounds = align.snippet_bounds(a.corpus, video, len(visual))
            assert np.isfinite(bounds).all() and (bounds[:, 1] > bounds[:, 0]).all()
            assert visual.ndim == 3 and visual.shape[1:] == (5, 1024) and len(visual)
            assert audio.ndim == 2 and audio.shape[1] == 128 and len(audio)
            assert text.ndim == 2 and text.shape[1] == 768 and len(text)
            assert len(audio) == times[video], (video, audio.shape, times[video])
            if name != "train":
                assert len(gt[name][video]) == times[video]
            assert all(0 <= start < end <= times[video] and (p is None or 0 <= p <= 1)
                       for (start, end), p in soft[video].items())
            count += 1
    print(json.dumps({"host": socket.gethostname(), "corpus": a.corpus, "videos": count,
                      "splits": {k: len(v) for k, v in split.items()}, "status": "parsed_and_covered"}))


if __name__ == "__main__":
    main()
