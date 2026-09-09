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
SCAF_DIM = 7
COL_ELL, COL_PS, COL_BF, COL_BC, COL_PH, COL_BLOCK, COL_TEXT = range(SCAF_DIM)
LLR_SCALE = 5.0                       # per-second text log-likelihood ratio is fed as llr / LLR_SCALE (clipped to [-1, 1])
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


TEXT_HATE_ROOT = os.path.join(REPO_ROOT, "data", "text_hate")


def load_text_hate(corpus, vid):
    """Per-second text-classifier hate probabilities (data/text_hate, built by
    scripts/build_text_hate_scores.py): dict p_asr / w_asr / p_ocr / w_ocr, or None."""
    p = os.path.join(TEXT_HATE_ROOT, corpus, "%s.npz" % vid)
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def text_observations(corpus, video_ids, grid):
    """vid -> interval_evidence_hmm.text_observation(...) (binned segment counts)
    for the videos with any text; videos without text are absent."""
    import interval_evidence_hmm as ieh
    out = {}
    for vid in video_ids:
        arr = load_text_hate(corpus, vid)
        if arr is None:
            continue
        xt = ieh.text_observation(grid, arr)
        if xt:
            out[vid] = xt
    return out


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
                    blk.astype(np.float32),
                    np.zeros(len(blk), np.float32)], axis=1)
    assert out.shape[1] == SCAF_DIM and k > 0
    return out.astype(np.float32)


class ScaffoldCache:
    """Per-video (f_a_ext, n_seconds, snip_bounds), computed once.

    ``scaffold_fn(vid, snip, n_seconds)`` returns the (rows, SCAF_DIM) scaffold
    or None (then zeros; counted in n_missing_verdict).

    ``masked_fn(vid, b_fine_masked, snip, n_seconds)`` (optional) rebuilds the
    scaffold from a fine-verdict vector with MISSING (-1) entries; when given,
    the audio+text block is kept so ``build(vid, b_fine_masked)`` returns a
    fresh f_a_ext without touching the stored default (adaptive-query module)."""

    def __init__(self, corpus, video_ids, scaffold_fn, masked_fn=None):
        self.corpus = corpus
        self.items = {}
        self.base = {}                 # vid -> audio+text rows (only with masked_fn)
        self.window_rows = {}          # per-row fine-window index (K,) grid -> rows
        self.masked_fn = masked_fn
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
            at = np.concatenate([audio, text], axis=1).astype(np.float32)
            f_a = np.concatenate([at, scaf], axis=1).astype(np.float32)
            self.items[vid] = (np.ascontiguousarray(f_a), n_seconds, snip)
            if masked_fn is not None:
                self.base[vid] = np.ascontiguousarray(at)

    def __getitem__(self, vid):
        return self.items[vid]

    def build(self, vid, b_fine_masked):
        """f_a_ext for ``vid`` with the scaffold recomputed from ``b_fine_masked``
        (MISSING = -1 for windows not asked). Requires masked_fn."""
        f_a, n_seconds, snip = self.items[vid]
        scaf = self.masked_fn(vid, np.asarray(b_fine_masked), snip, n_seconds)
        if scaf is None:
            return f_a
        out = np.concatenate([self.base[vid], scaf], axis=1).astype(np.float32)
        return np.ascontiguousarray(out)


class TrainDataset(data.Dataset):
    """``mask_sampler(vid, rng)`` (optional) returns a fine-verdict vector with
    MISSING entries; the scaffold is then rebuilt per item (evidence dropout,
    adaptive-query module). ``seed`` fixes the sampler's random stream."""

    def __init__(self, corpus, video_ids, labels, cache, max_seqlen,
                 crop_repeat=align.N_CROPS, mask_sampler=None, seed=0,
                 window_targets=None):
        self.corpus = corpus
        self.video_ids = list(video_ids)
        self.labels = labels
        self.cache = cache
        self.max_seqlen = int(max_seqlen)
        self.crop_repeat = int(crop_repeat)
        self.mask_sampler = mask_sampler
        self.seed = int(seed)
        self._rng = None
        # window_targets(vid, b_fine_input) -> (K,) float targets for the
        # fine-window bag loss (NaN = no target for that window); when given the
        # item carries a fifth element (K,) (module-1 iteration 2, window_bag_loss)
        self.window_targets = window_targets

    def __len__(self):
        return len(self.video_ids) * self.crop_repeat

    def __getitem__(self, index):
        vid = self.video_ids[index // self.crop_repeat]
        crop = index % self.crop_repeat
        f_a, n_seconds, snip = self.cache[vid]
        b_input = None
        if self.mask_sampler is not None:
            if self._rng is None:      # one stream per worker process
                wi = data.get_worker_info()
                self._rng = np.random.RandomState(self.seed + (wi.id if wi else 0))
            b_input = self.mask_sampler(vid, self._rng)
            f_a = self.cache.build(vid, b_input)
        f_v = align.aligned_visual_crop(self.corpus, vid, crop, "snippet",
                                        n_seconds, snip)
        w = self.cache.window_rows[vid][:, None]
        f_v = process_feat(f_v, self.max_seqlen, is_random=False)
        f_a = process_feat(f_a, self.max_seqlen, is_random=False)
        w = process_feat(w, self.max_seqlen, is_random=False)[:, 0]
        item = (torch.from_numpy(np.ascontiguousarray(f_v, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(f_a, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)),
                float(self.labels[vid]))
        if self.window_targets is not None:
            tgt = np.asarray(self.window_targets(vid, b_input), dtype=np.float32)
            item = item + (torch.from_numpy(np.ascontiguousarray(tgt)),)
        return item


class EvalDataset(data.Dataset):
    """One item per video: five crops stacked, full untruncated sequence.
    ``masks`` (optional): vid -> fine-verdict vector with MISSING entries; the
    scaffold is rebuilt from it (adaptive-query module)."""

    def __init__(self, corpus, video_ids, cache, masks=None):
        self.corpus = corpus
        self.video_ids = list(video_ids)
        self.cache = cache
        self.masks = masks

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        vid = self.video_ids[index]
        f_a, n_seconds, snip = self.cache[vid]
        if self.masks is not None and vid in self.masks:
            f_a = self.cache.build(vid, self.masks[vid])
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


def window_bag_loss(content_log, w_rows, seq_len, targets, topk_div):
    """Fine-window bag loss (module-1 iteration 2, plan section C): one bag per
    fine window w with a target (targets[i, w] not NaN), top-k mean of the
    content logit over the window's rows, BCE against the target. Mean over
    the targeted windows of each item, then mean over the items that have a
    target (every video weighs the same whatever its number of targets).
    w_rows (B, T) = fine-window index of each row (TrainDataset's third
    element); targets (B, K). Returns 0 when no window has a target."""
    z = content_log.squeeze(-1)
    per_item = []
    for i in range(z.shape[0]):
        t = int(seq_len[i])
        zi, wi = z[i, :t], w_rows[i, :t]
        tg = targets[i]
        terms = []
        for w in torch.nonzero(~torch.isnan(tg)).flatten().tolist():
            m = wi == w
            n_w = int(m.sum())
            if n_w == 0:
                continue
            k = max(1, int(-(-n_w // topk_div)))
            bag = torch.topk(zi[m], k=k).values.mean()
            terms.append(nn.functional.binary_cross_entropy_with_logits(bag, tg[w]))
        if terms:
            per_item.append(torch.stack(terms).mean())
    if not per_item:
        return z.new_zeros(())
    return torch.stack(per_item).mean()


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


def load_fixed_cohort(corpus):
    """Fixed dataset/GT coverage, shared by current full-training candidates."""
    from hate_common import data as hdata
    labels = hdata.load_labels(corpus)
    ids = {s: hdata.load_split(corpus, s) for s in ['train', 'val', 'test']}
    original = sum(ids.values(), [])
    if len(original) != len(set(original)):
        raise ValueError('duplicate video or split overlap')
    gt = {s: hdata.gt_arrays(corpus, s) for s in ['val', 'test']}
    excluded = {}
    for split in ['val', 'test']:
        excluded[split] = sorted(set(ids[split])-gt[split].keys())
        allowed = ['hate_video_427'] if corpus == 'hatemm' and split == 'test' else []
        if excluded[split] != allowed:
            raise ValueError(f'unexpected GT exclusion {split}: {excluded[split]}')
        ids[split] = [v for v in ids[split] if v in gt[split]]
        if set(ids[split]) != set(gt[split]):
            raise ValueError('fixed GT coverage mismatch')
    for split, videos in ids.items():
        if set(usable(corpus, videos)) != set(videos):
            raise ValueError(f'missing baseline features in {split}')
        if any(v not in labels or labels[v] not in [0, 1] for v in videos):
            raise ValueError(f'missing/nonbinary video labels in {split}')
    return labels, ids, gt, excluded


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


def video_duration(corpus, vid):
    """Seconds of the video = rows of its VGGish array (the verdict windows are
    fractions of this duration; vlm_verdict.verdict_rows uses the same D)."""
    return int(align.load_audio(corpus, vid).shape[0])


def fit_hmm(corpus, train_ids, labels, binary, model="index", text_obs=None, **opts):
    """model = "index": src/verdict_hmm.HierEvidenceHMM (fine window t -> block
    floor(4t/30)); "interval": src/interval_evidence_hmm.IntervalEvidenceHMM
    (true verdict intervals, time-length transitions; opts =
    positive_constraint / video_effect). Train video labels only."""
    pos_ids = [v for v in train_ids if labels[v] == 1 and v in binary]
    neg_ids = [v for v in train_ids if labels[v] == 0 and v in binary]
    if model == "index":
        hmm = verdict_hmm.HierEvidenceHMM(K_FINE, J_COARSE).fit(
            [binary[v] for v in pos_ids], [binary[v] for v in neg_ids])
    elif model == "interval":
        import interval_evidence_hmm
        text_obs = text_obs or {}
        pos = [binary[v] + (video_duration(corpus, v), text_obs.get(v)) for v in pos_ids]
        neg = [binary[v] + (video_duration(corpus, v), text_obs.get(v)) for v in neg_ids]
        hmm = interval_evidence_hmm.IntervalEvidenceHMM(K_FINE, J_COARSE, **opts).fit(pos, neg)
    else:
        raise ValueError(model)
    return hmm, len(pos_ids), len(neg_ids)


def text_llr_seconds(hmm, arrays):
    """Per-second text log-likelihood ratio log p(x_t | s=1) / p(x_t | s=0) under the
    HMM's fitted categorical tables (sum over the asr / ocr families; 0 where no
    text). arrays = load_text_hate(...) dict; returns (T,) float32 or None."""
    import interval_evidence_hmm as ieh
    if arrays is None or not getattr(hmm, "text", False):
        return None
    T = len(arrays["p_asr"])
    out = np.zeros(T, np.float64)
    for fam in ieh.TEXT_FAMILIES:
        p = np.asarray(arrays["p_" + fam], np.float64)
        ok = np.isfinite(p)
        if not ok.any():
            continue
        b = np.searchsorted(ieh.TEXT_BINS, np.clip(p[ok], 0.0, 1.0), side="right")
        t = np.clip(hmm.text_emit[fam], 1e-6, 1.0)
        out[ok] += np.log(t[1, b]) - np.log(t[0, b])
    return (hmm.text_weight * out).astype(np.float32)


def text_llr_rows(llr_seconds, snip):
    """(T,) per-second LLR resampled onto the snippet rows (rows, )."""
    if llr_seconds is None:
        return None
    arr = np.asarray(llr_seconds, np.float32)[:, None]
    return align.resample_intervals(arr, align.second_bounds(arr.shape[0]), snip)[:, 0]


def scaffold_rows_interval(hmm, ell_seg, ps_seg, b_fine, b_coarse, p_h,
                           snip, n_seconds, text_llr=None):
    """Scaffold from per-segment posteriors of an IntervalEvidenceHMM: rows
    take the segment containing their midpoint; the block index is the coarse
    interval containing the row midpoint (no fine-window -> block table)."""
    import interval_evidence_hmm as ieh
    seg = lambda arr: ieh.rows_from_segments(np.asarray(arr, np.float32),  # noqa: E731
                                             hmm.grid, snip, n_seconds)
    rows = lambda arr: vlm_verdict.verdict_rows(np.asarray(arr, np.float32),  # noqa: E731
                                                snip, n_seconds)
    blk = hmm.coarse_of_rows(snip, n_seconds).astype(int)
    tl = text_llr_rows(text_llr, snip)
    if tl is None:
        tl = np.zeros(len(blk), np.float32)
    out = np.stack([seg(ell_seg), seg(ps_seg), rows(b_fine),
                    np.asarray(b_coarse, np.float32)[blk],
                    np.asarray(p_h, np.float32)[blk],
                    blk.astype(np.float32),
                    np.asarray(tl, np.float32)], axis=1)
    assert out.shape[1] == SCAF_DIM
    return out.astype(np.float32)


def make_masked_scaffold_fn(hmm, binary, text=None, text_llr=None):
    """Scaffold builder from a masked fine-verdict vector (interval HMM only):
    ``fn(vid, b_fine_masked, snip, n_seconds)``. Columns: ell / P(s) from the
    HMM posterior with MISSING emissions, b_fine column keeps -1 for windows
    not asked (the model maps it to its own "not asked" state), coarse verdict,
    block posterior, block index."""
    assert hmm.params().get("model") == "interval"

    def fn(vid, b_fine_masked, snip, n_seconds):
        if vid not in binary:
            return None
        _, bc = binary[vid]
        p_s, p_h = hmm.posterior(b_fine_masked, bc, n_seconds, xt=(text or {}).get(vid))
        ell = np.log(p_s + 1e-6) - np.log(1.0 - p_s + 1e-6)
        return scaffold_rows_interval(hmm, ell, p_s, b_fine_masked, bc, p_h, snip, n_seconds,
                                      text_llr=(text_llr or {}).get(vid))
    return fn


def make_scaffold_fn(hmm, binary, ablation, w_fine, text=None, text_llr=None):
    """Per-video scaffold builder (README dataset.py column layout).

    ablation mean_prior: prior / input columns use the plain mean verdict
    level instead of the HMM posterior (block label still P(h_j));
    mean_prior_all: additionally the block label is the raw coarse verdict,
    so no HMM quantity is used anywhere (candidate 3 revision 3);
    raw_block_label: block label only."""
    interval = hmm.params().get("model") == "interval"
    block_of_window = None if interval else hmm.block.astype(np.float32)
    kw = {}
    if ablation == "indep_hmm":
        kw["independent"] = True
    if ablation == "flat_coarse":
        kw["flat_coarse"] = True

    def fn(vid, snip, n_seconds):
        if vid not in binary:
            return None
        bf, bc = binary[vid]
        if interval:
            p_s, p_h = hmm.posterior(bf, bc, n_seconds, w_fine=w_fine, xt=(text or {}).get(vid))
        else:
            p_s, p_h = hmm.posterior(bf, bc, w_fine=w_fine, **kw)
        ell = np.log(p_s + 1e-6) - np.log(1.0 - p_s + 1e-6)
        if ablation in ("mean_prior", "mean_prior_all"):
            # prior input: 2*(mean binary level - 1/2) in [-1, 1], stored so
            # that ell/ELL_SCALE equals it; the p_s input column is replaced by
            # the mean level too, so no HMM quantity remains in the input
            if interval:
                mean = (bf[hmm.grid["fine_of"]] + bc[hmm.grid["coarse_of"]]) / 2.0
            else:
                mean = (bf + bc[hmm.block]) / 2.0
            ell = ELL_SCALE * (2.0 * mean - 1.0)
            p_s = mean
        if ablation in ("raw_block_label", "mean_prior_all"):
            p_h = bc.astype(np.float32)
        if interval:
            return scaffold_rows_interval(hmm, ell, p_s, bf, bc, p_h, snip, n_seconds,
                                          text_llr=(text_llr or {}).get(vid))
        return scaffold_rows(ell, p_s, bf, bc, p_h, block_of_window,
                                snip, n_seconds)
    return fn
