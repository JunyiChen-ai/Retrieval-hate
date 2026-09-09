"""Acquisition policies over the cached VLM verdicts, scored through the
backbone (online-query module, 2026-09-10; from
experiments/20260908_adaptive_vlm_query/acquire.py). Simulation only:
revealing a window = un-masking its cached verdict; 0 new VLM calls. The
verdict of a candidate window is never read while scoring it.

State of a video = 4 coarse verdicts (always observed) + a fine-verdict vector
with MISSING (-1) entries. Every policy reveals one fine window per step:
  eoc          expected output change of the backbone: two counterfactual
               forward passes per candidate window (verdict assumed 0 / 1),
               weighted by p(b_w = 1 | state). weight = "model": the backbone's
               own window prediction sigmoid(top-k mean of the content logit
               over the window's rows) (plan section B; the same bag the
               window loss trains); "hmm": the interval HMM's predictive
               probability (iteration-1 behaviour; regime mixtures use
               IntervalEvidenceHMM.infer).
  conflict     |mean over the window of sigmoid(content logit) - P(h_w | E)|
  entropy      HMM-only: sum over the window's segments of P(s)(1-P(s))
  localization HMM-only: expected reduction of sum_g P(s_g)(1-P(s_g))
  uniform      fixed evenly spread order (bit reversal of 0..29)
  random       random order
After every step the backbone scores the video (five-crop mean of sigmoid of
the full per-second logit); for eoc the max gain at each step is kept so any
stopping threshold tau can be read off the same run. `initial` may be one
list for all videos or a dict video -> list (training-time growth of the
per-video allowed sets, plan section A).
"""
from __future__ import annotations

import numpy as np
import torch

import hier_evidence_common as hc
import interval_evidence_hmm as ieh
from macilsd import align

POLICIES = ("eoc", "conflict", "entropy", "localization", "uniform", "random")
CHUNK = 12          # max candidate scaffolds per forward chunk (x5 crops)
SEQ_T2_BUDGET = 1e7  # max (sequences x T^2) per chunk: attention scores are B x H x T x T


def bit_reversal_order(k):
    """Evenly spread order of 0..k-1 (van der Corput on k)."""
    idx = []
    n = 1
    while n < k:
        n *= 2
    for i in range(n):
        r, x = 0, i
        for _ in range(n.bit_length() - 1):
            r = (r << 1) | (x & 1)
            x >>= 1
        w = int(r * k / n)
        if w not in idx and w < k:
            idx.append(w)
    for w in range(k):
        if w not in idx:
            idx.append(w)
    return idx


def window_bags(clog, window_rows, k, topk_div):
    """sigmoid(top-k mean of the per-row content logit) per fine window (k,)
    from one video's (T,) logit rows; windows without rows get 0.5."""
    out = np.full(k, 0.5)
    for w in range(k):
        rows = np.where(window_rows == w)[0]
        if len(rows) == 0:
            continue
        n = max(1, int(-(-len(rows) // topk_div)))
        top = np.sort(clog[rows])[::-1][:n].mean()
        out[w] = 1.0 / (1.0 + np.exp(-top))
    return out


class Acquirer:
    def __init__(self, model, hmm, cache, corpus, binary, device, weight="model", topk_div=16, text=None):
        self.model = model
        self.hmm = hmm
        self.text = text or {}                    # vid -> HMM text observation (free evidence), or absent
        self.cache = cache
        self.corpus = corpus
        self.binary = binary
        self.device = device
        self.k = hmm.k
        assert weight in ("model", "hmm"), weight
        self.weight = weight
        self.topk_div = int(topk_div)
        assert hmm.normalized_time, "Acquirer assumes normalized time (one transition matrix for all videos)"
        self.Tm = hmm._transitions(1.0)             # normalized time: one matrix for all videos
        gr = hmm.grid
        self.win_segments = {w: np.where(gr["fine_of"] == w)[0] for w in range(self.k)}

    # ------------------------------------------------------------ backbone
    def video_inputs(self, vid):
        f_a, n_seconds, snip = self.cache[vid]
        crops = [align.aligned_visual_crop(self.corpus, vid, c, "snippet", n_seconds, snip)
                 for c in range(align.N_CROPS)]
        f_v = torch.from_numpy(np.ascontiguousarray(np.stack(crops, 0), dtype=np.float32)).to(self.device)
        index_map = align.snippet_index_for_seconds(snip, n_seconds)
        return f_v, index_map, int(n_seconds)

    @torch.no_grad()
    def forward(self, f_v5, fa_list):
        """fa_list: list of (T, A_EXT) arrays. Returns (n, T) five-crop mean of
        sigmoid(full logit), of sigmoid(content logit), and the five-crop mean
        content logit itself."""
        self.model.eval()
        outs, couts, clogs = [], [], []
        T = int(f_v5.shape[1])
        rows = max(1, min(CHUNK, int(SEQ_T2_BUDGET // (align.N_CROPS * T * T))))   # long videos: fewer per chunk
        for i in range(0, len(fa_list), rows):
            chunk = fa_list[i:i + rows]
            n = len(chunk)
            f_a = torch.from_numpy(np.stack(chunk, 0)).to(self.device)
            f_a = f_a.repeat_interleave(align.N_CROPS, 0)
            f_v = f_v5.repeat(n, 1, 1)
            _, _, _, av_log, _, _ = self.model(f_a, f_v, seq_len=None)
            sig = torch.sigmoid(av_log.squeeze(-1)).view(n, align.N_CROPS, -1).mean(1)
            cl = self.model.last_content_logit.squeeze(-1).view(n, align.N_CROPS, -1)
            outs.append(sig.cpu().numpy())
            couts.append(torch.sigmoid(cl).mean(1).cpu().numpy())
            clogs.append(cl.mean(1).cpu().numpy())
        return np.concatenate(outs, 0), np.concatenate(couts, 0), np.concatenate(clogs, 0)

    # ------------------------------------------------------------ HMM side
    def state(self, bf, bc, vid=None):
        return self.hmm.infer(bf, bc, Tm=self.Tm, xt=self.text.get(vid))

    # ------------------------------------------------------------ one video
    def run_video(self, vid, policy, n_steps, rng, initial=None):
        """Greedy reveal starting from the fine windows in `initial` (already
        observed, not counted as picks; None = only the coarse blocks).
        Returns dict(picks, scores (list over k of per-second arrays),
        gains (eoc: max gain before each pick))."""
        bf_true, bc = self.binary[vid]
        bf = np.full(self.k, ieh.MISSING, dtype=int)
        for w in (initial or []):
            bf[int(w)] = int(bf_true[int(w)])
        unobserved = set(range(self.k)) - set(int(w) for w in (initial or []))
        f_v5, index_map, n_seconds = self.video_inputs(vid)
        window_rows = self.cache.window_rows[vid].astype(int)
        order = bit_reversal_order(self.k) if policy == "uniform" else (
            list(rng.permutation(self.k)) if policy == "random" else None)
        sig, csig, clog = self.forward(f_v5, [self.cache.build(vid, bf)])
        scores = [sig[0][index_map]]
        gains, picks = [], []
        for _ in range(n_steps):
            if not unobserved:
                break
            cands = sorted(unobserved)
            if policy in ("uniform", "random"):
                w = next(x for x in order if x in unobserved)
            elif policy == "entropy":
                p = self.state(bf, bc, vid)["p_s"]
                u = [float(np.sum(p[self.win_segments[c]] * (1 - p[self.win_segments[c]]))) for c in cands]
                w = cands[int(np.argmax(u))]
            elif policy == "localization":
                st = self.state(bf, bc, vid)
                p = st["p_s"]
                u_now = float(np.sum(p * (1 - p)))
                pred = st["pred_fine"]
                best, w = -np.inf, cands[0]
                for c in cands:
                    exp_u = 0.0
                    for b, pb in ((1, pred[c]), (0, 1 - pred[c])):
                        if pb < 1e-9:
                            continue
                        bf2 = bf.copy()
                        bf2[c] = b
                        p2 = self.state(bf2, bc, vid)["p_s"]
                        exp_u += pb * float(np.sum(p2 * (1 - p2)))
                    if u_now - exp_u > best:
                        best, w = u_now - exp_u, c
            elif policy == "conflict":
                p_hf = self.state(bf, bc, vid)["p_hf"]
                cur_c = csig[0]
                u = []
                for c in cands:
                    rows = np.where(window_rows == c)[0]
                    m = float(cur_c[rows].mean()) if len(rows) else 0.0
                    u.append(abs(m - float(p_hf[c])))
                w = cands[int(np.argmax(u))]
            elif policy == "eoc":
                if self.weight == "model":
                    pred = window_bags(clog[0], window_rows, self.k, self.topk_div)
                else:
                    pred = self.state(bf, bc, vid)["pred_fine"]
                fa_list = []
                for c in cands:
                    for b in (0, 1):
                        bf2 = bf.copy()
                        bf2[c] = b
                        fa_list.append(self.cache.build(vid, bf2))
                sc, _, _ = self.forward(f_v5, fa_list)
                sc = sc.reshape(len(cands), 2, -1)
                cur = sig[0]
                eoc = [float(pred[c] * np.abs(sc[i, 1] - cur).mean()
                             + (1 - pred[c]) * np.abs(sc[i, 0] - cur).mean())
                       for i, c in enumerate(cands)]
                j = int(np.argmax(eoc))
                w = cands[j]
                gains.append(float(eoc[j]))
            else:
                raise ValueError(policy)
            bf[w] = int(bf_true[w])          # the verdict is read only for the chosen window
            unobserved.discard(w)
            picks.append(int(w))
            sig, csig, clog = self.forward(f_v5, [self.cache.build(vid, bf)])
            scores.append(sig[0][index_map])
        if torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()        # release the per-video attention blocks (videos differ in T)
        return {"picks": picks, "scores": scores, "gains": gains}

    def run_split(self, vids, policy, n_steps, seed=0, log=None, initial=None):
        rng = np.random.RandomState(seed)
        out = {}
        for i, vid in enumerate(vids):
            if vid not in self.binary:
                continue
            init = initial.get(vid, []) if isinstance(initial, dict) else initial
            out[vid] = self.run_video(vid, policy, n_steps, rng, initial=init)
            if log is not None and (i + 1) % 100 == 0:
                log("  %s: %d/%d videos" % (policy, i + 1, len(vids)))
        return out


def scores_at(runs, k):
    """Per-video scores after min(k, available) picks."""
    return {vid: r["scores"][min(k, len(r["scores"]) - 1)] for vid, r in runs.items()}


def stop_index(r, b_max, tau):
    """Number of picks under cap b_max and threshold tau (stop before a pick
    whose expected output change is below tau)."""
    n = min(b_max, len(r["picks"]))
    for i in range(n):
        if r["gains"][i] < tau:
            return i
    return n
