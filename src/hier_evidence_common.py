"""Shared infrastructure of the hierarchical-evidence MIL candidates
(experiments/20260903_hier_evidence_mil and its successors): datasets on
MACIL-SD's I3D snippet grid with the verdict scaffold columns, the verdict
HMM fitting / scaffold builder, the verdict-block MIL loss, split scoring and
the call into the shared evaluator.

Promoted verbatim from experiments/20260903_hier_evidence_mil/{dataset,train}.py
on 2026-09-04 when a second experiment needed it (CLAUDE.md: shared logic
goes to src/, no third copy). The experiment's dataset.py re-exports this
module so its recorded runs are unchanged.

Rows live on MACIL-SD's I3D snippet grid (0.667 s). Per row:
    f_v   I3D RGB, one of five crops (1024)
    f_a   VGGish (128) ⊕ BERT sentence (768) ⊕ scaffold (SCAF_DIM)
The scaffold columns come from the hierarchical evidence HMM over the frozen
VLM verdicts (src/verdict_hmm.py):
    0  ell     posterior log-odds log P(s_t=1|b) - log P(s_t=0|b)  (the prior)
    1  p_s     posterior P(s_t=1)
    2  b_fine  binary K=30 verdict of the row's window
    3  b_coarse binary K=4 verdict of the row's block
    4  p_h     posterior P(h_j=1) of the row's coarse block (block-bag label)
    5  block   coarse block index j of the row (0..J-1)
Columns 0-3 are the backbone's input channels; columns 4-5 are training
bookkeeping and are always hidden from the backbone.

Training items are (video, crop) pairs exactly as in macilsd/dataset.py; the
validation/test items stack the five crops.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "duplex"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from macilsd import align                      # noqa: E402
from macilsd.utils import process_feat         # noqa: E402
import frame_eval_common as fec                # noqa: E402
import vlm_verdict                             # noqa: E402
import verdict_hmm                             # noqa: E402

EVALUATOR = os.path.join(REPO_ROOT, "scripts", "reproduction_baselines",
                         "eval_baseline_scores.py")
K_FINE, J_COARSE = vlm_verdict.GRANULARITIES          # (30, 4)
# Posterior log-odds are bounded by +-log((1-eps)/eps) with eps = 1e-6 in
# verdict_hmm.posterior_log_odds; dividing by that bound maps them
# order-preservingly onto [-1, 1], so prior_scale is the maximal logit shift.
ELL_SCALE = float(np.log((1.0 - 1e-6) / 1e-6))   # ~13.8

TEXT_ROOT = os.path.join(REPO_ROOT, "results", "reproduction", "features",
                         "bert_sentence_1fps")
TEXT_DIM = 768
SCAF_DIM = 6
COL_ELL, COL_PS, COL_BF, COL_BC, COL_PH, COL_BLOCK = range(SCAF_DIM)
N_INPUT_SCAF = 4                      # columns fed to the backbone (rev 1; the eliminated rev 2 used 2)
A_EXT_DIM = align.A_DIM + TEXT_DIM + SCAF_DIM
SCAF_OFFSET = align.A_DIM + TEXT_DIM


def text_path(corpus, vid):
    return os.path.join(TEXT_ROOT, corpus, "%s.npy" % vid)


def load_text_rows(corpus, vid, snip):
    """BERT rows resampled from the 1 s grid onto the snippet grid, or None."""
    p = text_path(corpus, vid)
    if not os.path.exists(p):
        return None
    arr = np.load(p).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != TEXT_DIM or arr.shape[0] == 0:
        return None
    return align.resample_intervals(arr, align.second_bounds(arr.shape[0]),
                                    snip)


def scaffold_rows(ell, p_s, b_fine, b_coarse, p_h, block_of_window,
                  snip, n_seconds):
    """Per-row scaffold from per-fine-window arrays (K,) and per-block (J,)."""
    k = len(ell)
    rows = lambda arr: vlm_verdict.verdict_rows(np.asarray(arr, np.float32),  # noqa: E731
                                                snip, n_seconds)
    blk = rows(block_of_window).astype(int)
    out = np.stack([rows(ell), rows(p_s), rows(b_fine),
                    np.asarray(b_coarse, np.float32)[blk],
                    np.asarray(p_h, np.float32)[blk],
                    blk.astype(np.float32)], axis=1)
    assert out.shape[1] == SCAF_DIM and k > 0
    return out.astype(np.float32)


class ScaffoldCache:
    """Per-video (f_a_ext, n_seconds, snip_bounds), computed once.

    ``scaffold_fn(vid, snip, n_seconds)`` returns the (rows, SCAF_DIM) scaffold
    or None (then zeros; counted in n_missing_verdict)."""

    def __init__(self, corpus, video_ids, scaffold_fn):
        self.corpus = corpus
        self.items = {}
        self.window_rows = {}          # per-row fine-window index (K,) grid -> rows
        self.n_missing_text = 0
        self.n_missing_verdict = 0
        k_fine = vlm_verdict.GRANULARITIES[0]
        for vid in video_ids:
            audio, n_seconds, snip = align.aligned_audio(corpus, vid, "snippet")
            self.window_rows[vid] = vlm_verdict.verdict_rows(
                np.arange(k_fine, dtype=np.float32), snip, n_seconds).astype(np.float32)
            text = load_text_rows(corpus, vid, snip)
            if text is None:
                self.n_missing_text += 1
                text = np.zeros((audio.shape[0], TEXT_DIM), dtype=np.float32)
            scaf = scaffold_fn(vid, snip, n_seconds)
            if scaf is None:
                self.n_missing_verdict += 1
                scaf = np.zeros((audio.shape[0], SCAF_DIM), dtype=np.float32)
            f_a = np.concatenate([audio, text, scaf], axis=1).astype(np.float32)
            self.items[vid] = (np.ascontiguousarray(f_a), n_seconds, snip)

    def __getitem__(self, vid):
        return self.items[vid]


class TrainDataset(data.Dataset):
    def __init__(self, corpus, video_ids, labels, cache, max_seqlen,
                 crop_repeat=align.N_CROPS):
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
        f_a, n_seconds, snip = self.cache[vid]
        f_v = align.aligned_visual_crop(self.corpus, vid, crop, "snippet",
                                        n_seconds, snip)
        w = self.cache.window_rows[vid][:, None]
        f_v = process_feat(f_v, self.max_seqlen, is_random=False)
        f_a = process_feat(f_a, self.max_seqlen, is_random=False)
        w = process_feat(w, self.max_seqlen, is_random=False)[:, 0]
        return (torch.from_numpy(np.ascontiguousarray(f_v, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(f_a, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)),
                float(self.labels[vid]))


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
        f_a, n_seconds, snip = self.cache[vid]
        crops = [align.aligned_visual_crop(self.corpus, vid, c, "snippet",
                                           n_seconds, snip)
                 for c in range(align.N_CROPS)]
        f_v = np.stack(crops, axis=0)
        f_a = np.repeat(f_a[None], align.N_CROPS, axis=0)
        index_map = align.snippet_index_for_seconds(snip, n_seconds)
        return (torch.from_numpy(np.ascontiguousarray(f_v, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(f_a, dtype=np.float32)),
                torch.from_numpy(index_map), int(n_seconds), vid)


def block_bag_loss(content_log, f_a, seq_len, labels, topk_div):
    """Verdict-block MIL: one bag per coarse block, label P(h_j=1) (column
    COL_PH, exact 0 on negative videos), weight |2p-1|, top-k mean of the
    content logit inside the block. Returns the weighted mean BCE."""
    z = content_log.squeeze(-1)
    blk = f_a[..., SCAF_OFFSET + COL_BLOCK]
    ph = f_a[..., SCAF_OFFSET + COL_PH]
    num = z.new_zeros(())
    den = z.new_zeros(())
    for i in range(z.shape[0]):
        t = int(seq_len[i])
        zi, bi, pi = z[i, :t], blk[i, :t], ph[i, :t]
        for j in torch.unique(bi):
            m = bi == j
            n_j = int(m.sum())
            if n_j == 0:
                continue
            k = max(1, int(-(-n_j // topk_div)))
            bag = torch.topk(zi[m], k=k).values.mean()
            p = pi[m][0] if labels[i] > 0.5 else zi.new_zeros(())
            w = (2.0 * p - 1.0).abs()
            num = num + w * nn.functional.binary_cross_entropy_with_logits(
                bag, p)
            den = den + w
    return num / den.clamp_min(1e-6)


def _scalar(x):
    return float(x.detach()) if torch.is_tensor(x) else float(x)


def _git_describe():
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%cd %s", "--date=short"],
            cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def usable(corpus, ids):
    return [v for v in ids if align.has_features(corpus, v)]


def score_split(model, loader, device):
    """video_id -> scores on the 1 fps grid (five-crop mean of sigmoid(z~))."""
    model.eval()
    out = {}
    with torch.no_grad():
        for f_v, f_a, index_map, n_seconds, vid in loader:
            vid = vid[0]
            n_seconds = int(n_seconds)
            index_map = index_map[0].numpy()
            f_v = f_v[0].to(device)
            f_a = f_a[0].to(device)
            _, _, _, av_logits, _, _ = model(f_a, f_v, seq_len=None)
            av = torch.sigmoid(av_logits.squeeze(-1)).mean(0).cpu().numpy()
            s = np.asarray(av, dtype=np.float64)[index_map]
            if s.shape[0] != n_seconds:
                raise RuntimeError("%s: %d rows for %d seconds"
                                   % (vid, s.shape[0], n_seconds))
            out[vid] = s
    model.train()
    return out


def frame_metrics(scores, gt, hate_ids):
    per_video = {v: (scores[v], np.asarray(gt[v])) for v in scores if v in gt}
    res = fec.evaluate(per_video, macro_over={v for v in per_video
                                              if v in hate_ids})
    return {"pooled_ap": res["pr_auc"], "pooled_roc": res["roc_auc"],
            "within_roc": res["per_video"]["macro_auc"],
            "n_videos": res["n_videos"]}


def write_scores(path, scores):
    with open(path, "w") as fh:
        for vid in sorted(scores):
            fh.write(json.dumps({"video_id": vid,
                                 "n_frames": int(len(scores[vid])),
                                 "score_av": [round(float(x), 6)
                                              for x in scores[vid]]}) + "\n")


def run_evaluator(corpus, split, scores_path, json_out):
    subprocess.run([sys.executable, EVALUATOR, "--corpus", corpus,
                    "--split", split, "--scores", scores_path,
                    "--json-out", json_out], check=True, cwd=REPO_ROOT,
                   stdout=subprocess.DEVNULL)
    with open(json_out) as fh:
        return json.load(fh)


def fit_hmm(corpus, train_ids, labels, binary):
    pos = [binary[v] for v in train_ids if labels[v] == 1 and v in binary]
    neg = [binary[v] for v in train_ids if labels[v] == 0 and v in binary]
    return verdict_hmm.HierEvidenceHMM(K_FINE, J_COARSE).fit(pos, neg), \
        len(pos), len(neg)


def make_scaffold_fn(hmm, binary, ablation, w_fine):
    """Per-video scaffold builder (README dataset.py column layout)."""
    block_of_window = hmm.block.astype(np.float32)
    kw = {}
    if ablation == "indep_hmm":
        kw["independent"] = True
    if ablation == "flat_coarse":
        kw["flat_coarse"] = True

    def fn(vid, snip, n_seconds):
        if vid not in binary:
            return None
        bf, bc = binary[vid]
        p_s, p_h = hmm.posterior(bf, bc, w_fine=w_fine, **kw)
        ell = np.log(p_s + 1e-6) - np.log(1.0 - p_s + 1e-6)
        if ablation == "mean_prior":
            # revision-4 prior input: 2*(mean binary level - 1/2) in [-1, 1],
            # stored so that ell/ELL_SCALE equals it; the p_s input column is
            # replaced by the mean level too, so no HMM quantity remains
            mean = (bf + bc[hmm.block]) / 2.0
            ell = ELL_SCALE * (2.0 * mean - 1.0)
            p_s = mean
        if ablation == "raw_block_label":
            p_h = bc.astype(np.float32)
        return scaffold_rows(ell, p_s, bf, bc, p_h, block_of_window,
                                snip, n_seconds)
    return fn

