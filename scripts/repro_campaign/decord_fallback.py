#!/usr/bin/env python
"""A PyAV fallback for `decord.VideoReader`, installed by monkeypatch.

decord cannot open a substantial share of the released MultiHateClip containers
("cannot find video stream with wanted index: -1"): 27% of MHC-EN, 7% of MHC-ZH
and 7% of HateClipSeg on the sample the Wave 0 Qwen2.5-VL row measured.  Taking
that at face value would record a third of MHC-EN as method failures that are
really decode failures, so the Wave 0 driver already decoded those files with
PyAV.  UniTime reads video through decord in two places
(`collators/qwen_vision_process._read_video_decord` and `feature.feature`), and
both use only `len(vr)`, `vr.get_avg_fps()` and `vr.get_batch(idx).asnumpy()`.

`install()` replaces `decord.VideoReader` with a wrapper that tries the real
reader first and falls back to a PyAV-backed object exposing exactly those three
calls.  Videos decord can open are therefore untouched, byte for byte.
"""
from __future__ import annotations

import numpy as np


class _Batch:
    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr

    def numpy(self):
        return self._arr

    def __len__(self):
        return len(self._arr)


class PyAVReader:
    def __init__(self, path: str):
        import av

        self.path = str(path)
        with av.open(self.path) as c:
            s = c.streams.video[0]
            self.fps = float(s.average_rate) if s.average_rate else 25.0
            n = s.frames or 0
            if n <= 0:
                dur = float(s.duration * s.time_base) if s.duration else (
                    c.duration / av.time_base if c.duration else 0.0)
                n = int(dur * self.fps)
            self.n = max(int(n), 1)

    def __len__(self):
        return self.n

    def get_avg_fps(self):
        return self.fps

    def get_batch(self, indices):
        import av

        idx = [int(i) for i in (indices.tolist() if hasattr(indices, "tolist")
                                else list(indices))]
        want = sorted(set(max(0, min(i, self.n - 1)) for i in idx))
        got, last = {}, max(want)
        with av.open(self.path) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            k = 0
            for frame in c.decode(video=0):
                if k in got or k in want:
                    got[k] = frame.to_ndarray(format="rgb24")
                k += 1
                if k > last:
                    break
        if not got:
            raise RuntimeError(f"pyav decoded 0 frames from {self.path}")
        # a container that reports more frames than it holds: clamp to what exists
        avail = sorted(got)
        out = [got.get(i, got[min(avail, key=lambda j: abs(j - i))]) for i in idx]
        return _Batch(np.stack(out))


def install():
    import decord

    real = decord.VideoReader

    class VideoReader:
        def __new__(cls, path, *a, **kw):
            try:
                return real(path, *a, **kw)
            except Exception:
                return PyAVReader(path)

    decord.VideoReader = VideoReader
    # the two call sites import the module, not the symbol, so patching the
    # attribute is enough; patch the re-export too in case a module did
    # `from decord import VideoReader`.
    try:
        import decord.video_reader as _vr
        _vr.VideoReader = VideoReader
    except Exception:
        pass
    print("[patch] decord.VideoReader -> PyAV fallback on open failure", flush=True)
