"""Datasets for the evidence-chain network.

Rows live on MACIL-SD's I3D snippet grid (0.667 s). Per row:
    f_v   I3D RGB, one of five crops (1024)
    f_a   VGGish (128) ⊕ BERT sentence (768)          -- content only, no verdict columns
plus per-row verdict tensors built from the frozen VLM verdicts and the fixed
evidence-model constants (Potentials):
    w, j          window (K=30) and block (J=4) index of the row
    n_w, n_j      number of rows in the row's window / block
    phi_f         fine-verdict log-likelihood ratio of the window, divided by n_w
    phi_c         coarse-verdict log-likelihood ratio of the row's block
    bf, bc        the raw binary verdicts of the window / block
    bfp, bfn      fine verdict of the previous / next window (0 outside)
and a per-video verdict profile vector (PROFILE_DIM) for the density head.

Training items are (video, crop) pairs as in MACIL-SD; sequences longer than
max_seqlen are chunk-averaged (features), chunk-summed (phi_f, so the total
verdict evidence per window is preserved) or chunk-first (indices); shorter
ones are zero-padded with mask False and a padding block index J.
Validation/test items keep the full sequence with the five crops stacked.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.utils.data as data

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from macilsd import align                      # noqa: E402
import vlm_verdict                             # noqa: E402
import verdict_hmm                             # noqa: E402

TEXT_ROOT = os.path.join(REPO_ROOT, "results", "reproduction", "features",
                         "bert_sentence_1fps")
TEXT_DIM = 768
A_DIM = align.A_DIM
V_DIM = align.V_DIM
F_A_DIM = A_DIM + TEXT_DIM
K, J = vlm_verdict.GRANULARITIES          # (30, 4)
BLOCK_OF_WINDOW = verdict_hmm._block_map(K, J)
PROFILE_DIM = 6 + (J + 1)
ROW_KEYS = ("phi_f", "phi_c", "bf", "bc", "bfp", "bfn", "n_w", "n_j")   # float rows
IDX_KEYS = ("w", "j")                                                  # long rows


class Potentials:
    """Fixed evidence-model constants from the train-label EM (src/verdict_hmm)."""

    def __init__(self, hmm):
        self.q_f, self.r_f, self.q_c, self.r_c = hmm.q_f, hmm.r_f, hmm.q_c, hmm.r_c
        self.a = float(hmm.A[0, 1] + hmm.A[1, 0])        # switching rate
        self.p0_hate = float(hmm.p0[1])
        self.llr_f = (np.log((1 - self.q_f) / (1 - self.r_f)), np.log(self.q_f / self.r_f))
        self.llr_c = (np.log((1 - self.q_c) / (1 - self.r_c)), np.log(self.q_c / self.r_c))

    def as_dict(self):
        return {"q_f": self.q_f, "r_f": self.r_f, "q_c": self.q_c, "r_c": self.r_c,
                "a": self.a, "p0_hate": self.p0_hate,
                "llr_f": list(map(float, self.llr_f)), "llr_c": list(map(float, self.llr_c))}


def text_path(corpus, vid):
    return os.path.join(TEXT_ROOT, corpus, "%s.npy" % vid)


def load_text_rows(corpus, vid, snip):
    p = text_path(corpus, vid)
    if not os.path.exists(p):
        return None
    arr = np.load(p).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != TEXT_DIM or arr.shape[0] == 0:
        return None
    return align.resample_intervals(arr, align.second_bounds(arr.shape[0]), snip)


def profile_vector(bf, bc):
    blk = BLOCK_OF_WINDOW
    both = bf * bc[blk]
    fonly = bf * (1 - bc[blk])
    conly = (1 - bf) * bc[blk]
    runs = np.diff(np.r_[0, bf, 0])
    n_runs = float((runs == 1).sum())
    onehot = np.zeros(J + 1, np.float32)
    onehot[int(bc.sum())] = 1.0
    return np.concatenate([[bf.mean(), bc.mean(), both.mean(), fonly.mean(),
                            conly.mean(), n_runs / K], onehot]).astype(np.float32)


def verdict_rows(bf, bc, snip, n_seconds, pot):
    """Per-row verdict tensors (dict of (T,) arrays) from window/block verdicts."""
    w = vlm_verdict.verdict_rows(np.arange(K, dtype=np.float32), snip, n_seconds).astype(np.int64)
    j = BLOCK_OF_WINDOW[w].astype(np.int64)
    n_w = np.bincount(w, minlength=K).astype(np.float32)
    n_j = np.bincount(j, minlength=J).astype(np.float32)
    bf = np.asarray(bf, np.int64)
    bc = np.asarray(bc, np.int64)
    bfp = np.concatenate([[0], bf[:-1]])
    bfn = np.concatenate([bf[1:], [0]])
    llr_f = np.array(pot.llr_f, np.float32)[bf]           # (K,)
    llr_c = np.array(pot.llr_c, np.float32)[bc]           # (J,)
    return {
        "w": w, "j": j,
        "n_w": n_w[w], "n_j": n_j[j],
        "phi_f": (llr_f[w] / np.maximum(n_w[w], 1.0)).astype(np.float32),
        "phi_c": llr_c[j].astype(np.float32),
        "bf": bf[w].astype(np.float32), "bc": bc[j].astype(np.float32),
        "bfp": bfp[w].astype(np.float32), "bfn": bfn[w].astype(np.float32),
    }


def fit_length(f_v, f_a, vt, length):
    """Chunk-average / sum / first-of-chunk to exactly `length` rows, then pad."""
    T = f_a.shape[0]
    out = {}
    if T > length:
        bounds = np.linspace(0, T, length + 1).astype(int)
        seg = [(bounds[i], max(bounds[i + 1], bounds[i] + 1)) for i in range(length)]
        f_v = np.stack([f_v[s:e].mean(0) for s, e in seg])
        f_a = np.stack([f_a[s:e].mean(0) for s, e in seg])
        for k in ROW_KEYS:
            if k == "phi_f":
                out[k] = np.array([vt[k][s:e].sum() for s, e in seg], np.float32)
            else:
                out[k] = np.array([vt[k][s] for s, e in seg], np.float32)
        for k in IDX_KEYS:
            out[k] = np.array([vt[k][s] for s, e in seg], np.int64)
        mask = np.ones(length, bool)
    else:
        pad = length - T
        f_v = np.concatenate([f_v, np.zeros((pad, f_v.shape[1]), np.float32)])
        f_a = np.concatenate([f_a, np.zeros((pad, f_a.shape[1]), np.float32)])
        for k in ROW_KEYS:
            out[k] = np.concatenate([vt[k].astype(np.float32), np.zeros(pad, np.float32)])
        out["w"] = np.concatenate([vt["w"], np.full(pad, K, np.int64)])
        out["j"] = np.concatenate([vt["j"], np.full(pad, J, np.int64)])
        mask = np.concatenate([np.ones(T, bool), np.zeros(pad, bool)])
    return f_v.astype(np.float32), f_a.astype(np.float32), out, mask


class VideoCache:
    """Per-video content rows (audio ⊕ text), verdict rows and profile; visual is loaded per item."""

    def __init__(self, corpus, video_ids, binary, pot):
        self.corpus = corpus
        self.items = {}
        self.n_missing_text = 0
        self.missing_verdict = [v for v in video_ids if v not in binary]
        for vid in video_ids:
            if vid not in binary:
                continue
            audio, n_seconds, snip = align.aligned_audio(corpus, vid, "snippet")
            text = load_text_rows(corpus, vid, snip)
            if text is None:
                self.n_missing_text += 1
                text = np.zeros((audio.shape[0], TEXT_DIM), dtype=np.float32)
            f_a = np.ascontiguousarray(np.concatenate([audio, text], 1).astype(np.float32))
            bf, bc = binary[vid]
            vt = verdict_rows(bf, bc, snip, n_seconds, pot)
            assert vt["w"].shape[0] == f_a.shape[0]
            self.items[vid] = (f_a, n_seconds, snip, vt, profile_vector(np.asarray(bf), np.asarray(bc)))

    def __getitem__(self, vid):
        return self.items[vid]


def _to_torch(vt):
    d = {k: torch.from_numpy(np.ascontiguousarray(vt[k], dtype=np.float32)) for k in ROW_KEYS}
    d.update({k: torch.from_numpy(np.ascontiguousarray(vt[k], dtype=np.int64)) for k in IDX_KEYS})
    return d


class TrainDataset(data.Dataset):
    def __init__(self, corpus, video_ids, labels, cache, max_seqlen, crop_repeat=align.N_CROPS):
        self.corpus = corpus
        self.video_ids = list(video_ids)
        self.labels = labels
        self.cache = cache
        self.max_seqlen = int(max_seqlen)
        self.crop_repeat = int(crop_repeat)

    def __len__(self):
        return len(self.video_ids) * self.crop_repeat

    def __getitem__(self, index):
        vid = self.video_ids[index // self.crop_repeat]
        crop = index % self.crop_repeat
        f_a, n_seconds, snip, vt, prof = self.cache[vid]
        f_v = align.aligned_visual_crop(self.corpus, vid, crop, "snippet", n_seconds, snip)
        f_v, f_a, vt2, mask = fit_length(f_v, f_a, vt, self.max_seqlen)
        return {"f_v": torch.from_numpy(f_v), "f_a": torch.from_numpy(f_a),
                "mask": torch.from_numpy(mask), "profile": torch.from_numpy(prof),
                "label": torch.tensor(float(self.labels[vid])), **_to_torch(vt2)}


class EvalDataset(data.Dataset):
    """One item per video: five crops stacked, full untruncated sequence."""

    def __init__(self, corpus, video_ids, cache):
        self.corpus = corpus
        self.video_ids = list(video_ids)
        self.cache = cache

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        vid = self.video_ids[index]
        f_a, n_seconds, snip, vt, prof = self.cache[vid]
        crops = [align.aligned_visual_crop(self.corpus, vid, c, "snippet", n_seconds, snip)
                 for c in range(align.N_CROPS)]
        f_v = np.stack(crops, axis=0).astype(np.float32)
        index_map = align.snippet_index_for_seconds(snip, n_seconds)
        return {"f_v": torch.from_numpy(f_v), "f_a": torch.from_numpy(f_a),
                "mask": torch.ones(f_a.shape[0], dtype=torch.bool),
                "profile": torch.from_numpy(prof),
                "index_map": torch.from_numpy(np.asarray(index_map)),
                "n_seconds": int(n_seconds), "vid": vid, **_to_torch(vt)}
