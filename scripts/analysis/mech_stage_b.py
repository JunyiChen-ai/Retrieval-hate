#!/usr/bin/env python
"""mech_stage_b.py -- the ONE driver for the MECH-STAGE-B CPU battery.

Frozen design: refine-logs/MECH_STAGE_B_PREREG.md (every rule frozen before any
candidate number was computed).

    B1  RVS  rank-space deployed-vote surrogate loss              (additive term)
    B2  XFM  cross-fit memory training                            (sibling banks)
    B3  AQM  trained asymmetric query/memory maps                 (second head g_phi)
    B4  TRA  per-bank-item trust radius                           (no head training)

B1/B2/B3 re-mint fold heads by CALLING the frozen scripts/analysis/headspace_mint.py
main() unmodified (the c06_falsifier_mint.py / mech_probes_a.py pattern).  Nothing under
src/ is edited: every behaviour change is a monkeypatch installed by this file inside its
own process and removed on exit.

TEST CONTACT: NONE.  Three inherited layers -- headspace_mint's torch.load guard, its
patched load_feats_from_CLIP (only train_*.pt / dev_seen_*.pt reachable), and c09_guard on
PYTHONPATH.  Every query in every arm is a train-split item; K_dev is never read here.

DISK: each head writes only its out-of-fold predictions/votes plus diagnostics (a few kB).
No key matrices, no per-epoch snapshots.

STAGES (one process each, driven by scripts/slurm/mech_stage_b_cpu.sbatch):
    instrument  replay the fold arena from the banked C06 mints -> instrument.json  (HALT gate)
    head        train ONE (arm, ds, seed, fold[, lam]) head and score its fold
    b4          trust-radius fitting over the banked mints -> b4.json
    collect     fold everything into MECH_STAGE_B_RESULT.json
    selftest    synthetic end-to-end drive (planted positive + planted null)

COST: CPU only, <= 8 threads.  Zero GPU, zero cloud.
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np

_T_START = time.time()

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))

DATASETS = ("hatemm", "zh")
SEEDS = (0, 1, 2)
FOLDS = (0, 1, 2, 3, 4)
TOPK = 20                       # deployed
ARMS = ("b1", "b2", "b3")

# ---- prereg §2.2 frozen RVS constants
RVS_M = 128                     # candidate/comparison set size (top-M by cosine)
RVS_BETA_DIV = 10.0             # beta_i = IQR(s_i.) / RVS_BETA_DIV
RVS_BETA_FLOOR = 1e-8
RVS_TEMP = 0.25                 # BCE logit scale
RVS_LAMBDAS = (0.1, 0.3, 1.0)   # prereg §2.3 frozen grid

# ---- prereg §5 frozen TRA constants
TRA_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
TRA_SHRINK = 0.05
TRA_SWEEPS = 2
TRA_GROUPS = 5
TRA_N_RANDOM = 5

SCREEN_BAR = 0.020              # prereg §0.3
SELF_RECALL_BAR = 0.8           # prereg §4.2
FLOOR_TOL = 5e-5                # prereg §0.4

BANKED_MINTS = os.path.join(REPO, "artifacts/c06_falsifier/mints")

# prereg §0.2 frozen floors (scripts/analysis/headspace_arena_<ds>_s<seed>_OUT.json)
FLOOR_ACC = {"hatemm": [0.8884, 0.8858, 0.8858], "zh": [0.8929, 0.8895, 0.8946]}
FLOOR_MF1 = {"hatemm": [0.8838, 0.8811, 0.8812], "zh": [0.8747, 0.8710, 0.8765]}

PROJECTED_SECONDS = float(os.environ.get("MSB_PROJECTED_SECONDS", 7381.0))

FROZEN_SHA = {
    "scripts/analysis/headspace_mint.py":
        "cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612",
    "scripts/analysis/mechnov_pairverify.py":
        "77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d",
    "scripts/analysis/mechfix_ops.py": None,     # recorded, not pinned upstream
}


# ================================================================= infrastructure
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def assert_frozen():
    got = {}
    for rel, want in FROZEN_SHA.items():
        g = sha256_of(os.path.join(REPO, rel))
        got[rel] = g
        if want is not None and g != want:
            raise AssertionError("FROZEN MODULE CHANGED: {} {}".format(rel, g))
    return got


def heartbeat(progress_path, phase, done=None, total=None, extra=""):
    """Line-buffered, append-only, one handle per call (C09 process rule)."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    units = "{}/{}".format(done, total) if total is not None else "-"
    elapsed = time.time() - _T_START
    line = "{} | {} | {} | {:.1f}s | {:.3f}x{}".format(
        stamp, phase, units, elapsed, elapsed / PROJECTED_SECONDS,
        (" | " + extra) if extra else "")
    if progress_path:
        try:
            os.makedirs(os.path.dirname(progress_path), exist_ok=True)
            with open(progress_path, "a", buffering=1) as fh:
                fh.write(line + "\n")
        except Exception:
            pass
    print(line, flush=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=float)
    os.replace(tmp, path)
    print("[msb] wrote {}".format(path), flush=True)


def f64(X):
    """Independent float64 C-contiguous copy.  mechfix_ops._norm32 L2-normalises IN
    PLACE on whatever np.asarray(X,'float32') hands back, so every array crossing into it
    is materialised here first and never reused by the caller afterwards."""
    return np.ascontiguousarray(np.asarray(X, dtype="float64"))


def load_banked(ds, seed, fold):
    tag = "full" if fold < 0 else str(fold)
    p = os.path.join(BANKED_MINTS, "mint_{}_N_s{}_f{}.npz".format(ds, seed, tag))
    if not os.path.exists(p):
        raise AssertionError("banked mint absent: {}".format(p))
    return np.load(p, allow_pickle=True)


def head_out_path(outroot, arm, ds, seed, fold, lam=None):
    lamtag = "" if lam is None else "_lam{}".format(str(lam).replace(".", "p"))
    return os.path.join(outroot, "heads",
                        "{}_{}_s{}_f{}{}.npz".format(arm, ds, seed, fold, lamtag))


# ================================================== the deployed vote (frozen module)
def fold_score(K_bank, lab_bank, K_query, lab_query):
    """Deployed top-20 rank-weighted signed-cosine vote, via the frozen operator."""
    import mechfix_ops as MECH
    votes, pred, _, _ = MECH.deployed_vote(f64(K_bank), lab_bank, f64(K_query), topk=TOPK)
    return votes, pred, float((pred == lab_query).mean())


# =============================================================== INSTRUMENT (HALT gate)
def run_instrument(outroot, progress=None):
    """prereg §0.4: replay the fold arena straight from the banked C06 mints and require
    it to reproduce the frozen floor literals.  Zero training, zero candidate numbers."""
    import mechfix_ops as MECH
    rows, halt = [], []
    for ds in DATASETS:
        for si, seed in enumerate(SEEDS):
            lab = fold_of = None
            pred = None
            for f in FOLDS:
                z = load_banked(ds, seed, f)
                if lab is None:
                    lab = np.asarray(z["lab"]).astype(int)
                    fold_of = np.asarray(z["fold_of"]).astype(int)
                    pred = np.full(len(lab), -1, dtype=int)
                K = f64(z["K_train"])
                ho = np.flatnonzero(fold_of == f)
                fit = np.flatnonzero(fold_of != f)
                _, p, _ = fold_score(K[fit], lab[fit], K[ho], lab[ho])
                pred[ho] = p
            assert (pred >= 0).all(), "arena predictions incomplete"
            a, m = MECH.acc(lab, pred), MECH.macro_f1(lab, pred)
            da, dm = abs(a - FLOOR_ACC[ds][si]), abs(m - FLOOR_MF1[ds][si])
            ok = (da <= FLOOR_TOL) and (dm <= FLOOR_TOL)
            rows.append({"dataset": ds, "seed": seed,
                         "replay_acc": round(a, 6), "banked_acc": FLOOR_ACC[ds][si],
                         "replay_mF1": round(m, 6), "banked_mF1": FLOOR_MF1[ds][si],
                         "abs_diff_acc": round(da, 8), "abs_diff_mF1": round(dm, 8),
                         "pass": bool(ok)})
            if not ok:
                halt.append("{} s{}".format(ds, seed))
            heartbeat(progress, "INSTRUMENT", len(rows), 6,
                      "{} s{} acc={:.4f} (banked {:.4f}) mF1={:.4f} pass={}".format(
                          ds, seed, a, FLOOR_ACC[ds][si], m, ok))
    out = {"check": "fold_arena_replay_vs_banked_floors", "tol": FLOOR_TOL,
           "rows": rows, "all_pass": not halt,
           "rule": "any |replay - banked| > 5e-5 -> HALT the whole battery; no fallback floor"}
    write_json(os.path.join(outroot, "instrument.json"), out)
    if halt:
        raise AssertionError("INSTRUMENT HALT: floor replay failed for {}".format(halt))
    return out


# ============================================================== training-arm machinery
def _unwrap(dl):
    """DeviceDataLoader -> torch DataLoader; returns the object owning .dataset."""
    return getattr(dl, "dl", dl)


def raw_pool(train_dl):
    """The head's fitting-pool tensors taken DIRECTLY from the dataset object.

    prereg §1.6 / §2.4: iterating `train_dl` creates a shuffle=True sampler iterator and
    therefore DRAWS FROM THE GLOBAL TORCH RNG.  Reading .dataset draws nothing, so every
    extra bank forward this driver performs is RNG-neutral by construction."""
    ds = _unwrap(train_dl).dataset
    import torch
    return (list(ds.ids), ds.image_feats, ds.text_feats,
            torch.as_tensor(np.asarray(ds.labels)).long())


def forward_keys(model, img, txt, chunk=512):
    """Detached fused keys of (img, txt) under `model`, in eval mode, no_grad, with the
    model's train/eval mode restored exactly.  Draws no RNG (dropout inert in eval, no
    BatchNorm in this recipe)."""
    import torch
    was_training = model.training
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, img.shape[0], chunk):
            _, e = model(img[i:i + chunk], txt[i:i + chunk], return_embed=True)
            outs.append(e.detach())
    if was_training:
        model.train()
    return torch.cat(outs, dim=0)


def consume_epoch_iterator(train_dl):
    """RNG parity for arms that bypass the deployed per-epoch bank rebuild (prereg §3.3).

    The deployed rebuild's cost is a full-train forward in eval mode, which draws NO RNG;
    the RNG is drawn by creating the shuffle=True iterator.  Consuming the iterator and
    discarding the batches is therefore RNG-identical and much cheaper."""
    for _ in train_dl:
        pass


def _select_neighbours(D, I, bank_lab, query_lab, n_hn, n_pp):
    """The frozen selection loop of utils/retrieval.py:466-513, verbatim in behaviour:
    walk the top-(n*multiple) FAISS row; the first `n_hn` opposite-label rows are the hard
    negatives, the first `n_pp` same-label rows are the pseudo-gold positives.  Returns
    (hn_rows, hn_sims, pp_rows, pp_sims) with -1 padding for not-found."""
    nq = D.shape[0]
    hn_rows = np.full((nq, n_hn), -1, dtype="int64")
    pp_rows = np.full((nq, n_pp), -1, dtype="int64")
    hn_sims = np.zeros((nq, n_hn), dtype="float32")
    pp_sims = np.zeros((nq, n_pp), dtype="float32")
    for i in range(nq):
        j = k = 0
        for it in range(D.shape[1]):
            row = int(I[i, it])
            if row < 0:
                continue
            if int(bank_lab[row]) != int(query_lab[i]) and j < n_hn:
                hn_rows[i, j] = row
                hn_sims[i, j] = D[i, it]
                j += 1
            elif int(bank_lab[row]) == int(query_lab[i]) and k < n_pp:
                pp_rows[i, k] = row
                pp_sims[i, k] = D[i, it]
                k += 1
            if j == n_hn and k == n_pp:
                break
    return hn_rows, hn_sims, pp_rows, pp_sims


# ------------------------------------------------------------------------- B1 (RVS)
def rvs_term(q_grad, y, own_rows, bank_n, bank_lab_t, diag):
    """prereg §2.2.  q_grad [B,D] grad-on; bank_n [N,D] detached and L2-normalised;
    own_rows [B] the query's own bank row (self-exclusion); bank_lab_t [N] long."""
    import torch
    import torch.nn.functional as Fn
    q = Fn.normalize(q_grad, p=2, dim=1)
    s = torch.matmul(q, bank_n.t())                                  # [B,N] grad-on
    B, N = s.shape
    NEG = torch.finfo(s.dtype).min
    self_mask = torch.zeros((B, N), dtype=torch.bool, device=s.device)
    self_mask[torch.arange(B, device=s.device), own_rows] = True
    s_masked = s.masked_fill(self_mask, NEG)                         # self excluded

    with torch.no_grad():
        s_det = s_masked.detach().clone()
        s_det[self_mask] = float("nan")
        q75 = torch.nanquantile(s_det, 0.75, dim=1)
        q25 = torch.nanquantile(s_det, 0.25, dim=1)
        beta = torch.clamp((q75 - q25) / RVS_BETA_DIV, min=RVS_BETA_FLOOR)  # [B]
        m = min(RVS_M, N - 1)
        top_idx = torch.topk(s_masked.detach(), m, dim=1).indices     # [B,m] detached
    sc = torch.gather(s_masked, 1, top_idx)                           # [B,m] grad-on
    lab_c = bank_lab_t[top_idx].float()                               # [B,m]

    # soft ranks within the candidate set: r_ij = sum_k sigma((s_ik - s_ij)/beta) - 0.5
    diff = (sc.unsqueeze(1) - sc.unsqueeze(2)) / beta.view(-1, 1, 1)  # [B,m,m] (k,j)
    sig = torch.sigmoid(diff)
    r = sig.sum(dim=1) - 0.5                                          # [B,m]
    w = torch.relu(21.0 - r)
    v = ((2.0 * lab_c - 1.0) * w).sum(1) / (w.sum(1) + 1e-8)          # [B]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        v / RVS_TEMP, y.float())

    with torch.no_grad():
        diag["boundary_sigma"].append(float(sig[:, :, -1].mean()))
        diag["active_frac"].append(float(((sig > 0.01) & (sig < 0.99)).float().mean()))
        diag["nonzero_w"].append(float((w > 0).float().sum(1).mean()))
    return loss


class ArmB1(object):
    """Additive RVS term on top of the deployed hybrid loss.  Patches run_rac.compute_loss
    and the model's forward (to capture the grad-on query embedding without a second
    forward, which would draw dropout RNG and move the trajectory)."""
    name = "b1"

    def __init__(self, lam, active=True):
        self.lam = float(lam)
        self.active = bool(active)
        self.diag = {"boundary_sigma": [], "active_frac": [], "nonzero_w": []}
        self._st = {"step": 0, "bank": None, "row_of": None, "blab": None}
        self._feat = []
        self._undo = []

    def install(self, run_rac, model, args):
        import torch
        orig_loss = run_rac.compute_loss
        orig_fwd = model.forward
        st, feat, diag = self._st, self._feat, self.diag
        lam, active = self.lam, self.active

        def fwd(img, txt, return_embed=False):
            out = orig_fwd(img, txt, return_embed=return_embed)
            if return_embed and torch.is_grad_enabled() and model.training:
                feat.append(out[1])
            return out

        def wrapped(batch, train_dl, mdl, a, **kw):
            if active and (st["step"] % len(train_dl) == 0):
                ids, img, txt, lab = raw_pool(train_dl)
                K = forward_keys(mdl, img, txt)
                st["bank"] = torch.nn.functional.normalize(K, p=2, dim=1)
                st["row_of"] = {v: r for r, v in enumerate(ids)}
                st["blab"] = lab.to(K.device)
                assert len(st["row_of"]) == K.shape[0], "train ids not unique"
            st["step"] += 1
            del feat[:]
            out = orig_loss(batch, train_dl, mdl, a, **kw)
            if not active:
                return out
            assert feat, "RVS: the query forward was not captured"
            q = feat[0]
            own = torch.as_tensor([st["row_of"][v] for v in batch["ids"]],
                                  dtype=torch.long, device=q.device)
            term = rvs_term(q, batch["labels"].to(q.device), own,
                            st["bank"], st["blab"], diag)
            return (out[0] + lam * term,) + tuple(out[1:])

        run_rac.compute_loss = wrapped
        model.forward = fwd
        self._undo = [(run_rac, "compute_loss", orig_loss)]
        self._model, self._orig_fwd = model, orig_fwd

    def uninstall(self):
        for obj, attr, val in self._undo:
            setattr(obj, attr, val)
        try:
            del self._model.forward
        except AttributeError:
            self._model.forward = self._orig_fwd

    def bank_keys(self, model, img, txt, K_final):
        """Deployment unchanged: the final head's own keys."""
        return K_final

    def report(self):
        return {k: (round(float(np.mean(v)), 6) if v else None)
                for k, v in self.diag.items()}


# ------------------------------------------------------------------------- B2 (XFM)
class ArmB2(object):
    """Cross-fit memory: the retrieval terms are mined against a frozen sibling bank that
    never saw the item's fold (and never saw this head's held-out fold).  Patches
    model.loss.dense_retrieve_hard_negatives_pseudo_positive."""
    name = "b2"

    def __init__(self, ds, seed, fold, active=True):
        self.ds, self.seed, self.fold = ds, seed, fold
        self.active = bool(active)
        self.stats = {"epochs": 0, "bank_sizes": {}}
        self._undo = []

    def _build_banks(self):
        """sibling k -> (normalised float32 keys, labels, source rows) with
        fold_of not in {k, self.fold}; plus the id -> fold map over the FULL train split
        (the head's dataset holds only the fitting pool, so its row indices are NOT the
        full-train indices `fold_of` is defined over)."""
        import faiss
        import headspace_mint as HM
        import mechnov_pairverify as P
        z0 = load_banked(self.ds, self.seed, self.fold)
        fold_of = np.asarray(z0["fold_of"]).astype(int)
        lab = np.asarray(z0["lab"]).astype(int)
        cfg = P.DATASETS[self.ds]
        tr_ids = HM.load_split(cfg["cache_dir"], "train", cfg["model"])[0]
        assert len(tr_ids) == len(fold_of), "train id count != fold_of length"
        self.id_to_fold = {v: int(fold_of[i]) for i, v in enumerate(tr_ids)}
        banks = {}
        for k in FOLDS:
            if k == self.fold:
                continue
            zk = load_banked(self.ds, self.seed, k)
            rows = np.flatnonzero((fold_of != k) & (fold_of != self.fold))
            K = np.ascontiguousarray(np.asarray(zk["K_train"], dtype="float32")[rows])
            faiss.normalize_L2(K)
            ix = faiss.IndexFlatIP(K.shape[1])
            ix.add(K)                       # built ONCE, reused for every step
            banks[k] = (K, lab[rows], rows, ix)
            self.stats["bank_sizes"][str(k)] = int(len(rows))
        self.fold_of, self.lab_all = fold_of, lab
        return banks

    def install(self, run_rac, model, args):
        import faiss
        import torch
        import model.loss as LOSS
        orig = LOSS.dense_retrieve_hard_negatives_pseudo_positive
        if not self.active:
            self._undo = []
            return
        banks = self._build_banks()
        id_to_fold = self.id_to_fold
        stats = self.stats

        def wrapped(train_dl, query_feats, query_labels, mdl,
                    largest_retrieval=1, threshold=None, args=None,
                    train_feats=None, train_labels=None,
                    target_pack=None, query_ids=None):
            mdl.eval()
            B = query_feats.shape[0]
            dim = query_feats.shape[1]
            if train_feats is None or train_labels is None:
                # prereg §3.3: keep the RNG stream aligned with the floor.
                consume_epoch_iterator(train_dl)
                stats["epochs"] += 1
            q = query_feats.cpu().detach().numpy().astype("float32")
            q = np.ascontiguousarray(q)
            faiss.normalize_L2(q)
            qlab = query_labels.cpu().detach().numpy().astype(int)
            assert query_ids is not None, "XFM needs query ids to route to a sibling bank"
            qfold = np.asarray([id_to_fold[v] for v in query_ids])
            assert (qfold != self.fold).all(), \
                "XFM: a held-out-fold item appeared in the fitting pool"

            n_hn, n_pp = args.no_hard_negatives, args.no_pseudo_gold_positives
            hn_f = torch.zeros(B, n_hn, dim, device=args.device)
            pp_f = torch.zeros(B, n_pp, dim, device=args.device)
            hn_s = torch.zeros(B, largest_retrieval, device=args.device)
            pp_s = torch.zeros(B, n_pp, device=args.device)
            ktop = largest_retrieval * args.hard_negatives_multiple
            for kf in np.unique(qfold):
                Kb, lb, _, ix = banks[int(kf)]
                sel = np.flatnonzero(qfold == kf)
                D, I = ix.search(np.ascontiguousarray(q[sel]), min(ktop, Kb.shape[0]))
                hr, hs, pr, ps = _select_neighbours(D, I, lb, qlab[sel], n_hn, n_pp)
                for a, b in enumerate(sel):
                    for j in range(n_hn):
                        if hr[a, j] >= 0:
                            hn_f[b, j] = torch.from_numpy(Kb[hr[a, j]].copy()).float()
                            if j < largest_retrieval:
                                hn_s[b, j] = float(hs[a, j])
                    for j in range(n_pp):
                        if pr[a, j] >= 0:
                            pp_f[b, j] = torch.from_numpy(Kb[pr[a, j]].copy()).float()
                            pp_s[b, j] = float(ps[a, j])
            sentinel = np.zeros((1, 1), dtype="float32")
            return hn_f, hn_s, pp_f, pp_s, sentinel, np.zeros(1, dtype="int64")

        LOSS.dense_retrieve_hard_negatives_pseudo_positive = wrapped
        self._undo = [(LOSS, "dense_retrieve_hard_negatives_pseudo_positive", orig)]

    def uninstall(self):
        for obj, attr, val in self._undo:
            setattr(obj, attr, val)

    def bank_keys(self, model, img, txt, K_final):
        """Deployment unchanged: the final head's own keys."""
        return K_final

    def report(self):
        return dict(self.stats)


# ------------------------------------------------------------------------- B3 (AQM)
class ArmB3(object):
    """Trained asymmetric maps: g_phi (deepcopy of f_theta at model_pass entry) produces
    bank keys, f_theta produces query keys and the BCE.  Patches run_rac.model_pass (to
    build g_phi and inject it into the optimizer through the existing aux_pack channel)
    and model.loss.dense_retrieve_hard_negatives_pseudo_positive (to select on a detached
    g_phi bank and then RE-FORWARD the selected rows through g_phi with grad)."""
    name = "b3"

    def __init__(self, active=True):
        self.active = bool(active)
        self.g_phi = None
        self.stats = {"epochs": 0}
        self._undo = []

    def install_model_pass(self, run_rac):
        orig_mp = run_rac.model_pass
        holder = self

        def wrapped_mp(train_dl, evaluate_dl, test_seen_dl, model, **kw):
            holder.g_phi = copy.deepcopy(model)
            # prereg §4.1: the existing aux_pack channel puts g_phi's parameters in the
            # SAME AdamW.  lambda_aux == 0, so compute_aux_loss is never called.
            # This is done in the mechanism-OFF parity head TOO (prereg §4.3): the parity
            # check must prove that adding g_phi's zero-gradient parameters to the
            # optimizer does not perturb f_theta's update.
            kw["aux_pack"] = {"module": holder.g_phi}
            holder._install_mining(train_dl, kw.get("args"))
            return orig_mp(train_dl, evaluate_dl, test_seen_dl, model, **kw)

        run_rac.model_pass = wrapped_mp
        self._undo.append((run_rac, "model_pass", orig_mp))

    def _install_mining(self, train_dl, args):
        import faiss
        import torch
        import model.loss as LOSS
        orig = LOSS.dense_retrieve_hard_negatives_pseudo_positive
        self._undo.append((LOSS, "dense_retrieve_hard_negatives_pseudo_positive", orig))
        if not self.active:
            return
        holder = self
        st = {"K": None, "ids": None, "lab": None, "img": None, "txt": None}

        def wrapped(train_dl_, query_feats, query_labels, mdl,
                    largest_retrieval=1, threshold=None, args=None,
                    train_feats=None, train_labels=None,
                    target_pack=None, query_ids=None):
            mdl.eval()
            g = holder.g_phi
            if train_feats is None or train_labels is None:
                consume_epoch_iterator(train_dl_)          # prereg §4.3 RNG parity
                ids, img, txt, lab = raw_pool(train_dl_)
                K = forward_keys(g, img, txt).cpu().numpy().astype("float32")
                K = np.ascontiguousarray(K)
                faiss.normalize_L2(K)
                st.update(K=K, ids=ids, lab=lab.numpy().astype(int), img=img, txt=txt)
                holder.stats["epochs"] += 1
            B = query_feats.shape[0]
            dim = query_feats.shape[1]
            q = np.ascontiguousarray(query_feats.cpu().detach().numpy().astype("float32"))
            faiss.normalize_L2(q)
            qlab = query_labels.cpu().detach().numpy().astype(int)
            n_hn, n_pp = args.no_hard_negatives, args.no_pseudo_gold_positives
            ktop = largest_retrieval * args.hard_negatives_multiple
            ix = faiss.IndexFlatIP(st["K"].shape[1])
            ix.add(st["K"])
            D, I = ix.search(q, min(ktop, st["K"].shape[0]))
            hr, hs, pr, ps = _select_neighbours(D, I, st["lab"], qlab, n_hn, n_pp)

            # RE-FORWARD the selected rows through g_phi WITH grad (prereg §4.1): the
            # deployed numpy mining path detaches the bank, so without this g_phi would
            # receive no retrieval gradient at all.
            rows = np.unique(np.concatenate([hr.ravel(), pr.ravel()]))
            rows = rows[rows >= 0]
            g.eval()          # memory keys are deterministic: dropout off in g_phi
            _, emb = g(st["img"][rows], st["txt"][rows], return_embed=True)
            row_pos = {int(r): a for a, r in enumerate(rows)}

            hn_f = torch.zeros(B, n_hn, dim, device=args.device)
            pp_f = torch.zeros(B, n_pp, dim, device=args.device)
            hn_parts, pp_parts = [], []
            for i in range(B):
                hn_parts.append(torch.stack(
                    [emb[row_pos[int(hr[i, j])]] if hr[i, j] >= 0 else hn_f[i, j]
                     for j in range(n_hn)]))
                pp_parts.append(torch.stack(
                    [emb[row_pos[int(pr[i, j])]] if pr[i, j] >= 0 else pp_f[i, j]
                     for j in range(n_pp)]))
            hn_f = torch.stack(hn_parts)
            pp_f = torch.stack(pp_parts)
            hn_s = torch.from_numpy(hs[:, :largest_retrieval].copy()).float().to(args.device)
            pp_s = torch.from_numpy(ps.copy()).float().to(args.device)
            return (hn_f, hn_s, pp_f, pp_s,
                    np.zeros((1, 1), dtype="float32"), np.zeros(1, dtype="int64"))

        LOSS.dense_retrieve_hard_negatives_pseudo_positive = wrapped

    def install(self, run_rac, model, args):
        pass                                        # done in install_model_pass

    def uninstall(self):
        for obj, attr, val in reversed(self._undo):
            setattr(obj, attr, val)

    def bank_keys(self, model, img, txt, K_final):
        """AQM deploys asymmetrically: bank keys come from g_phi, queries from f_theta.
        The mechanism-OFF parity head keeps the symmetric deployed bank."""
        if not self.active:
            return K_final
        return forward_keys(self.g_phi, img, txt).cpu().numpy().astype("float64")

    def report(self):
        return dict(self.stats)


# ================================================================== the head stage
def run_head(arm, ds, seed, fold, lam, active, outroot, scratch, threads, progress=None):
    """Train ONE fold head under `arm` and score its fold in the arena.

    The frozen headspace_mint.main() is CALLED, never re-implemented; the only additions
    are this driver's monkeypatches, installed before and removed after.

    `active=False` is the prereg §0.4 mechanism-OFF parity head; it writes to its own
    `<arm>off_...` path so it can never collide with or clobber a treatment head."""
    assert_frozen()
    import headspace_mint as HM
    import mechnov_pairverify as P
    import run_rac
    import torch
    import mechfix_ops as MECH

    out = head_out_path(outroot, arm if active else arm + "off", ds, seed, fold, lam)
    tag = "{} {} s{} f{}{}{}".format(arm, ds, seed, fold,
                                     "" if lam is None else " lam=%s" % lam,
                                     "" if active else " [PARITY-OFF]")
    if os.path.exists(out):
        heartbeat(progress, "HEAD-SKIP", extra="{} (resume)".format(tag))
        return
    heartbeat(progress, "HEAD-START", extra=tag)

    if arm == "b1":
        armobj = ArmB1(lam if lam is not None else 0.0, active=active)
    elif arm == "b2":
        armobj = ArmB2(ds, seed, fold, active=active)
    elif arm == "b3":
        armobj = ArmB3(active=active)
    else:
        raise AssertionError("unknown arm {}".format(arm))

    hold = {}
    orig_mp = run_rac.model_pass

    def capture_mp(train_dl, evaluate_dl, test_seen_dl, model, **kw):
        hold["model"] = model
        hold["train_dl"] = train_dl
        armobj.install(run_rac, model, kw.get("args"))
        return orig_mp(train_dl, evaluate_dl, test_seen_dl, model, **kw)

    if arm == "b3":
        armobj.install_model_pass(run_rac)
        inner = run_rac.model_pass

        def capture_mp3(train_dl, evaluate_dl, test_seen_dl, model, **kw):
            hold["model"] = model
            hold["train_dl"] = train_dl
            return inner(train_dl, evaluate_dl, test_seen_dl, model, **kw)

        run_rac.model_pass = capture_mp3
    else:
        run_rac.model_pass = capture_mp

    stage_dir = os.path.join(scratch, "msb_stage",
                             os.path.basename(out).replace(".npz", ""))
    if os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)
    stage_npz = os.path.join(stage_dir, "frozen_mint.npz")

    argv_saved = sys.argv
    sys.argv = ["headspace_mint.py", "--dataset", ds, "--seed", str(seed),
                "--fold", str(fold), "--out", stage_npz, "--scratch", stage_dir,
                "--threads", str(threads)]
    t0 = time.time()
    try:
        HM.main()
    finally:
        sys.argv = argv_saved
        armobj.uninstall()
        run_rac.model_pass = orig_mp
    secs = time.time() - t0

    assert os.path.exists(stage_npz), "frozen mint produced no output"
    z = np.load(stage_npz, allow_pickle=True)
    K_final = np.asarray(z["K_train"], dtype="float64")
    lab = np.asarray(z["lab"]).astype(int)
    fold_of = np.asarray(z["fold_of"]).astype(int)

    # ---- parity against the banked mint (prereg §0.4 for the OFF heads; diagnostic
    #      otherwise -- an ON head is EXPECTED to differ, that is the treatment).
    banked_max = float(np.abs(
        K_final - np.asarray(load_banked(ds, seed, fold)["K_train"], dtype="float64")
    ).max())
    if not active:
        assert banked_max == 0.0, (
            "PARITY HALT: {} mechanism-off head differs from the banked mint "
            "(max|Δ| = {:g}) -- the harness is not a no-op".format(arm, banked_max))

    # ---- the arena, computed in-process; only predictions are persisted (disk quota)
    model = hold["model"]
    cfg = P.DATASETS[ds]
    tr = HM.load_split(cfg["cache_dir"], "train", cfg["model"])
    img_all, txt_all = tr[1], tr[2]
    K_query = K_final                                    # f_theta keys, all train rows
    K_bank_all = armobj.bank_keys(model, img_all, txt_all, K_final)
    ho = np.flatnonzero(fold_of == fold)
    fit = np.flatnonzero(fold_of != fold)
    votes, pred, a = fold_score(K_bank_all[fit], lab[fit], K_query[ho], lab[ho])

    extra = {"arm_report": armobj.report()}
    if arm == "b3":
        # prereg §4.2 self-recall over the fitting pool: bank rows are g_phi keys, query
        # space is f_theta.  Fraction of bank rows whose nearest query is their own item.
        Bk = f64(K_bank_all[fit])
        Qk = f64(K_query[fit])
        Bn = Bk / np.maximum(np.linalg.norm(Bk, axis=1, keepdims=True), 1e-12)
        Qn = Qk / np.maximum(np.linalg.norm(Qk, axis=1, keepdims=True), 1e-12)
        nn_of_bank = np.argmax(Bn @ Qn.T, axis=1)
        extra["self_recall"] = float((nn_of_bank == np.arange(len(fit))).mean())

    meta = {"driver_sha256": sha256_of(os.path.abspath(__file__)),
            "prereg": "refine-logs/MECH_STAGE_B_PREREG.md",
            "arm": arm, "dataset": ds, "seed": seed, "fold": fold,
            "lam": lam, "active": bool(active), "secs": round(secs, 1),
            "banked_parity_maxabs": banked_max,
            "n_ho": int(len(ho)), "n_fit": int(len(fit)),
            "fold_acc": round(a, 6),
            "frozen_sha256": assert_frozen(),
            "mint_meta": json.loads(str(z["meta"])),
            **extra}
    tmp = out + ".tmp.npz"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(tmp, ho_idx=ho.astype("int64"), pred=pred.astype("int8"),
             lab_ho=lab[ho].astype("int8"), votes=votes.astype("float32"),
             meta=json.dumps(meta))
    os.replace(tmp, out)
    z.close()
    shutil.rmtree(stage_dir, ignore_errors=True)
    heartbeat(progress, "HEAD-DONE",
              extra="{} {:.1f}s fold_acc={:.4f} parity_maxabs={:.3g}".format(
                  tag, secs, a, banked_max))


# ======================================================================= B4 (TRA)
def _tra_votes(S_sorted, order_lab, rho_sorted_mask, topk=TOPK):
    """Vectorised admitted-top-k deployed vote.

    S_sorted   [nq, nb]  cosines in descending order per query
    order_lab  [nq, nb]  the corresponding bank labels, already mapped to +-1
    rho_sorted_mask [nq, nb] bool admission mask in the same (sorted) order

    A row is admitted only if its radius passes; rejected rows are replaced by the next
    ranked admitted row, which is exactly what `cumsum <= topk` over the admitted mask
    does.  Weights are the deployed 20..1, normalised by the weights actually used.
    """
    csum = np.cumsum(rho_sorted_mask, axis=1)
    sel = rho_sorted_mask & (csum <= topk)
    w = np.where(sel, (topk + 1) - csum, 0).astype("float64")
    num = (order_lab * S_sorted * w).sum(1)
    den = w.sum(1)
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def _tra_eval(S_sorted, order_lab, rho_sorted_mask, topk=TOPK):
    return (_tra_votes(S_sorted, order_lab, rho_sorted_mask, topk) >= 0).astype(int)


def tra_order(bank32, query32, topk=TOPK):
    """The CANONICAL descending-similarity order for the TRA operator.

    Its first `topk` columns are faiss's own top-k rows, in faiss's own order, so an
    admit-always TRA vote is bit-identical to mechfix_ops.deployed_vote BY CONSTRUCTION.

    Why this is not just np.argsort(-S): in the deployed head space cosines are collapsed
    (0.9998+) and exact float32 ties occur.  faiss's heap and numpy's stable argsort break
    those ties differently.  When a tied pair carries opposite labels the swap moves it
    between rank 20 and rank 19, changing the vote by 2*cos/210 = 0.00952 -- which is
    exactly what the job-13998 B4 HALT measured on 7 of 149 HateMM s0 f0 queries (the
    neighbour SET was identical in every case; only the order differed).  Recorded as
    prereg §5.1 amendment A2.

    Returns (S, order): S is the full float64 similarity matrix, order [nq, nb].
    """
    import faiss
    ixb = faiss.IndexFlatIP(bank32.shape[1])
    ixb.add(bank32)
    nb = bank32.shape[0]
    Dn, In = ixb.search(query32, nb)
    Dk, Ik = ixb.search(query32, topk)
    nq = query32.shape[0]
    S = np.zeros((nq, nb), dtype="float64")
    np.put_along_axis(S, In.astype("int64"), Dn.astype("float64"), axis=1)
    # the k=topk search and the k=nb search must agree on the similarity VALUES
    assert np.array_equal(np.take_along_axis(S, Ik.astype("int64"), axis=1),
                          Dk.astype("float64")), \
        "faiss k=topk and k=n_bank disagree on similarity values"
    in_head = np.zeros((nq, nb), dtype=bool)
    np.put_along_axis(in_head, Ik.astype("int64"), True, axis=1)
    tail = In.astype("int64")[~np.take_along_axis(in_head, In.astype("int64"), axis=1)]
    order = np.concatenate([Ik.astype("int64"), tail.reshape(nq, nb - topk)], axis=1)
    return S, order


def _tra_fit(S, order, blab_signed, y_fit, rng_unused):
    """prereg §5.2 coordinate ascent.  Returns rho [nb].

    OBJECTIVE = mean signed vote MARGIN, not 0/1 accuracy.  The synthetic drive
    (selftest §3, run before any real key was read) showed the accuracy objective is
    plateau-locked: with several bad bank rows crowding a query's top-20, rejecting one
    of them changes no prediction, so every single-coordinate move scores exactly 0 and
    greedy ascent never starts.  The margin moves on every admission change, so it is the
    objective that can actually be optimised; the decision rule it optimises is the
    deployed one (`vote >= 0`).  Recorded as prereg §5.2 amendment A1.
    """
    nq, nb = S.shape
    Ss = np.take_along_axis(S, order, axis=1)
    Ls = np.take_along_axis(np.broadcast_to(blab_signed, S.shape), order, axis=1)
    sgn_y = (2.0 * y_fit - 1.0).astype("float64")
    rho = np.full(nb, -1.0, dtype="float64")

    def objective(rho_vec):
        mask_full = S >= rho_vec[None, :]
        mask_sorted = np.take_along_axis(mask_full, order, axis=1)
        v = _tra_votes(Ss, Ls, mask_sorted)
        margin = float((sgn_y * v).mean())
        rej = float(1.0 - mask_full.mean())
        return margin - TRA_SHRINK * rej

    best_obj = objective(rho)
    cands = [[-1.0] + [float(np.quantile(S[:, j], p)) for p in TRA_QUANTILES]
             for j in range(nb)]
    for _ in range(TRA_SWEEPS):
        for j in range(nb):
            cur = rho[j]
            best_v, best_o = cur, best_obj
            for v in cands[j]:
                if v == cur:
                    continue
                rho[j] = v
                o = objective(rho)
                # strict improvement only; ties keep the incumbent, which starts at -1
                # (admit always), so the shrinkage prior is never overridden by noise
                if o > best_o + 1e-12:
                    best_v, best_o = v, o
            rho[j] = best_v
            best_obj = best_o
    return rho


def run_b4(outroot, progress=None):
    import mechfix_ops as MECH
    import faiss
    per_ds = {}
    done = 0
    total = len(DATASETS) * len(SEEDS) * len(FOLDS)
    for ds in DATASETS:
        fit_acc, rnd_acc, fit_mf1 = [], [], []
        for si, seed in enumerate(SEEDS):
            lab = fold_of = None
            pred_fit = pred_rnd = None
            for f in FOLDS:
                z = load_banked(ds, seed, f)
                if lab is None:
                    lab = np.asarray(z["lab"]).astype(int)
                    fold_of = np.asarray(z["fold_of"]).astype(int)
                    pred_fit = np.full(len(lab), -1, dtype=int)
                    pred_rnd = np.full((TRA_N_RANDOM, len(lab)), -1, dtype=int)
                K = np.asarray(z["K_train"], dtype="float32")
                ho = np.flatnonzero(fold_of == f)
                fit = np.flatnonzero(fold_of != f)
                Kb = np.ascontiguousarray(K[fit]); faiss.normalize_L2(Kb)
                Kq = np.ascontiguousarray(K[ho]); faiss.normalize_L2(Kq)
                S, order = tra_order(Kb, Kq)
                blab_signed = (2 * lab[fit] - 1).astype("float64")[None, :]
                y_ho = lab[ho]

                # in-run instrument check: rho = -1 (admit always) MUST reproduce the
                # frozen deployed vote, otherwise the TRA operator is not a strict
                # generalisation of the floor and no B4 delta is interpretable.
                Ss_all = np.take_along_axis(S, order, axis=1)
                Ls_all = np.take_along_axis(np.broadcast_to(blab_signed, S.shape),
                                            order, axis=1)
                v_admit = _tra_votes(Ss_all, Ls_all, np.ones_like(Ss_all, dtype=bool))
                v_dep, p_dep, _ = fold_score(K[fit], lab[fit], K[ho], y_ho)
                assert (( v_admit >= 0).astype(int) == p_dep).all(), (
                    "B4 HALT: admit-always TRA disagrees with the deployed vote "
                    "({} s{} f{})".format(ds, seed, f))
                admit_maxdiff = float(np.abs(v_admit - v_dep).max())
                assert admit_maxdiff < 1e-9, (
                    "B4 HALT: admit-always TRA vote differs from the deployed vote by "
                    "{:g} ({} s{} f{})".format(admit_maxdiff, ds, seed, f))
                grp = np.arange(len(ho)) % TRA_GROUPS
                for g in range(TRA_GROUPS):
                    tr_m = grp != g
                    te_m = grp == g
                    rho = _tra_fit(S[tr_m], order[tr_m],
                                   blab_signed, y_ho[tr_m], None)
                    Ste = S[te_m]
                    ord_te = order[te_m]
                    Sts = np.take_along_axis(Ste, ord_te, axis=1)
                    Lts = np.take_along_axis(
                        np.broadcast_to(blab_signed, Ste.shape), ord_te, axis=1)
                    mask = np.take_along_axis(Ste >= rho[None, :], ord_te, axis=1)
                    pred_fit[ho[te_m]] = _tra_eval(Sts, Lts, mask)
                    rng = np.random.default_rng(1000 * seed + 10 * f + g)
                    for r in range(TRA_N_RANDOM):
                        rp = rho[rng.permutation(len(rho))]
                        mk = np.take_along_axis(Ste >= rp[None, :], ord_te, axis=1)
                        pred_rnd[r][ho[te_m]] = _tra_eval(Sts, Lts, mk)
                done += 1
                heartbeat(progress, "B4", done, total,
                          "{} s{} f{} rho_rejects={:.3f}".format(
                              ds, seed, f, float((rho > -1).mean())))
            assert (pred_fit >= 0).all() and (pred_rnd >= 0).all(), "B4 incomplete"
            fit_acc.append(MECH.acc(lab, pred_fit))
            fit_mf1.append(MECH.macro_f1(lab, pred_fit))
            rnd_acc.append(float(np.mean([MECH.acc(lab, pred_rnd[r])
                                          for r in range(TRA_N_RANDOM)])))
        d = [fit_acc[i] - FLOOR_ACC[ds][i] for i in range(3)]
        c = [fit_acc[i] - rnd_acc[i] for i in range(3)]
        se_c = float(np.std(c, ddof=1) / np.sqrt(3))
        per_ds[ds] = {
            "fitted_acc_by_seed": [round(x, 6) for x in fit_acc],
            "fitted_mF1_by_seed": [round(x, 6) for x in fit_mf1],
            "random_acc_by_seed": [round(x, 6) for x in rnd_acc],
            "delta_vs_floor_by_seed": [round(x, 6) for x in d],
            "delta_vs_random_by_seed": [round(x, 6) for x in c],
            "mean_delta_vs_floor": round(float(np.mean(d)), 6),
            "mean_delta_vs_random": round(float(np.mean(c)), 6),
            "se_delta_vs_random": round(se_c, 6),
            "mF1_delta_vs_floor": round(float(np.mean(fit_mf1))
                                        - float(np.mean(FLOOR_MF1[ds])), 6),
            "clears_floor_bar": bool(np.mean(d) >= SCREEN_BAR),
            "beats_random_control": bool(np.mean(c) > se_c),
        }
    killed = any((not per_ds[ds]["clears_floor_bar"])
                 or (not per_ds[ds]["beats_random_control"]) for ds in DATASETS)
    out = {"experiment": "B4_TRA_per_bank_item_trust_radius", "per_dataset": per_ds,
           "rule": "KILLED iff mean delta vs floor < +0.020 on either dataset, OR "
                   "mean(fitted - random) <= SE across the 3 seeds on either dataset",
           "verdict": "KILLED" if killed else "ALIVE"}
    write_json(os.path.join(outroot, "b4.json"), out)
    return out


# ========================================================================= collect
def _agg_arm(outroot, arm, ds, seed, lam):
    import mechfix_ops as MECH
    lab = np.full(0, 0)
    preds, labs = {}, {}
    for f in FOLDS:
        p = head_out_path(outroot, arm, ds, seed, f, lam)
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=True)
        for i, idx in enumerate(np.asarray(z["ho_idx"])):
            preds[int(idx)] = int(z["pred"][i])
            labs[int(idx)] = int(z["lab_ho"][i])
    ks = sorted(preds)
    y = np.array([labs[k] for k in ks])
    p = np.array([preds[k] for k in ks])
    return {"n": len(ks), "acc": MECH.acc(y, p), "mF1": MECH.macro_f1(y, p)}


def run_collect(outroot, progress=None):
    res = {"prereg": "refine-logs/MECH_STAGE_B_PREREG.md",
           "driver_sha256": sha256_of(os.path.abspath(__file__)),
           "frozen_sha256": assert_frozen(),
           "screening_bar": SCREEN_BAR,
           "floors": {"acc": FLOOR_ACC, "mF1": FLOOR_MF1},
           "arms": {}, "verdicts": {}}
    ip = os.path.join(outroot, "instrument.json")
    res["instrument"] = json.load(open(ip)) if os.path.exists(ip) else None

    # ---- parity heads
    parity = {}
    for arm in ARMS:
        for ds in DATASETS:
            p = head_out_path(outroot, arm + "off", ds, 0, 0, None)
            if os.path.exists(p):
                m = json.loads(str(np.load(p, allow_pickle=True)["meta"]))
                parity["{}_{}".format(arm, ds)] = {
                    "banked_parity_maxabs": m["banked_parity_maxabs"],
                    "bit_exact": m["banked_parity_maxabs"] == 0.0}
    res["parity_heads"] = parity

    # ---- B1 over the lambda grid
    b1 = {"lambdas": {}}
    for lam in RVS_LAMBDAS:
        cells, deltas = {}, {}
        for ds in DATASETS:
            per_seed = [_agg_arm(outroot, "b1", ds, s, lam) for s in SEEDS]
            if any(c is None for c in per_seed):
                cells[ds] = None
                continue
            da = [per_seed[i]["acc"] - FLOOR_ACC[ds][i] for i in range(3)]
            dm = [per_seed[i]["mF1"] - FLOOR_MF1[ds][i] for i in range(3)]
            cells[ds] = {"acc_by_seed": [round(c["acc"], 6) for c in per_seed],
                         "mF1_by_seed": [round(c["mF1"], 6) for c in per_seed],
                         "delta_acc_by_seed": [round(x, 6) for x in da],
                         "delta_mF1_by_seed": [round(x, 6) for x in dm],
                         "mean_delta_acc": round(float(np.mean(da)), 6),
                         "mean_delta_mF1": round(float(np.mean(dm)), 6)}
            deltas[ds] = float(np.mean(da))
        b1["lambdas"][str(lam)] = {"per_dataset": cells,
                                   "mean_over_datasets": (round(float(np.mean(
                                       [deltas[d] for d in DATASETS])), 6)
                                       if len(deltas) == len(DATASETS) else None)}
    ready = {k: v for k, v in b1["lambdas"].items()
             if v["mean_over_datasets"] is not None}
    if ready:
        lam_star = max(ready, key=lambda k: ready[k]["mean_over_datasets"])
        b1["lambda_star"] = lam_star
        b1["selection_rule"] = ("argmax over the frozen grid of the mean over datasets "
                                "of the 3-seed-mean out-of-fold TRAIN accuracy delta")
        d = {ds: b1["lambdas"][lam_star]["per_dataset"][ds]["mean_delta_acc"]
             for ds in DATASETS}
        b1["delta_at_lambda_star"] = d
        b1["verdict"] = ("ALIVE" if all(d[ds] >= SCREEN_BAR for ds in DATASETS)
                         else "KILLED")
    else:
        b1["verdict"] = "INCOMPLETE"
    res["arms"]["B1_RVS"] = b1

    # ---- B2, B3
    for arm, key in (("b2", "B2_XFM"), ("b3", "B3_AQM_trained_gphi")):
        cells, deltas, srec = {}, {}, {}
        for ds in DATASETS:
            per_seed = [_agg_arm(outroot, arm, ds, s, None) for s in SEEDS]
            if any(c is None for c in per_seed):
                cells[ds] = None
                continue
            da = [per_seed[i]["acc"] - FLOOR_ACC[ds][i] for i in range(3)]
            dm = [per_seed[i]["mF1"] - FLOOR_MF1[ds][i] for i in range(3)]
            cells[ds] = {"acc_by_seed": [round(c["acc"], 6) for c in per_seed],
                         "mF1_by_seed": [round(c["mF1"], 6) for c in per_seed],
                         "delta_acc_by_seed": [round(x, 6) for x in da],
                         "delta_mF1_by_seed": [round(x, 6) for x in dm],
                         "mean_delta_acc": round(float(np.mean(da)), 6),
                         "mean_delta_mF1": round(float(np.mean(dm)), 6)}
            deltas[ds] = float(np.mean(da))
            if arm == "b3":
                vals = []
                for s in SEEDS:
                    for f in FOLDS:
                        p = head_out_path(outroot, arm, ds, s, f, None)
                        if os.path.exists(p):
                            m = json.loads(str(np.load(p, allow_pickle=True)["meta"]))
                            if "self_recall" in m:
                                vals.append(m["self_recall"])
                srec[ds] = round(float(np.mean(vals)), 6) if vals else None
        blk = {"per_dataset": cells}
        if len(deltas) == len(DATASETS):
            alive = all(deltas[ds] >= SCREEN_BAR for ds in DATASETS)
            v = "ALIVE" if alive else "KILLED"
            if arm == "b3":
                blk["self_recall_by_dataset"] = srec
                if alive and any(srec.get(ds) is not None and srec[ds] < SELF_RECALL_BAR
                                 for ds in DATASETS):
                    v = "KILLED (memory-corruption artifact: self_recall < 0.8)"
            if arm == "b2" and alive:
                v = ("ALIVE-CONFOUNDED (prereg §3.2: sibling weights saw the arena's "
                     "query fold; needs {k,f}-excluding replication before it may be "
                     "quoted as positive)")
            blk["verdict"] = v
        else:
            blk["verdict"] = "INCOMPLETE"
        res["arms"][key] = blk

    bp = os.path.join(outroot, "b4.json")
    res["arms"]["B4_TRA"] = json.load(open(bp)) if os.path.exists(bp) else None

    res["verdicts"] = {
        "B1_RVS": res["arms"]["B1_RVS"].get("verdict", "MISSING"),
        "B2_XFM": res["arms"]["B2_XFM"].get("verdict", "MISSING"),
        "B3_AQM_trained_gphi": res["arms"]["B3_AQM_trained_gphi"].get("verdict", "MISSING"),
        "B4_TRA": (res["arms"]["B4_TRA"] or {}).get("verdict", "MISSING"),
    }
    res["elapsed_seconds"] = round(time.time() - _T_START, 1)
    write_json(os.path.join(outroot, "MECH_STAGE_B_RESULT.json"), res)
    heartbeat(progress, "COLLECT", extra=json.dumps(res["verdicts"]))
    return res


# ======================================================================== selftest
def _synth_keys(n, d, rng, tightness):
    base = rng.normal(size=(1, d)); base /= np.linalg.norm(base)
    j = rng.normal(size=(n, d)); j /= np.linalg.norm(j, axis=1, keepdims=True)
    X = tightness * base + (1.0 - tightness) * j
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _synth_head_files(outroot, arm, ds, seed, lam, lab, fold_of, effect, rng):
    """Write synthetic per-head prediction files with a PLANTED accuracy effect:
    `effect` = the fraction of otherwise-wrong predictions that are flipped correct."""
    for f in FOLDS:
        ho = np.flatnonzero(fold_of == f)
        y = lab[ho]
        base = y.copy()
        wrong = rng.random(len(ho)) < 0.15          # ~85% base accuracy
        base[wrong] = 1 - base[wrong]
        fix = wrong & (rng.random(len(ho)) < effect)
        base[fix] = y[fix]
        p = head_out_path(outroot, arm, ds, seed, f, lam)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez(p, ho_idx=ho.astype("int64"), pred=base.astype("int8"),
                 lab_ho=y.astype("int8"), votes=np.zeros(len(ho), "float32"),
                 meta=json.dumps({"arm": arm, "banked_parity_maxabs": 0.0,
                                  "self_recall": 0.95}))


def run_selftest(outroot):
    """Synthetic end-to-end drive: every numeric path, with a PLANTED POSITIVE and a
    PLANTED NULL, and a real training smoke test through the actual patched code paths on
    random features.  No cluster artifact and no candidate metric is read."""
    import mechfix_ops as MECH
    import torch
    rng = np.random.default_rng(0)
    ok = []
    sub = os.path.join(outroot, "selftest_out")
    if os.path.isdir(sub):
        shutil.rmtree(sub)
    os.makedirs(sub)

    # ---------- 1. RVS term: gradient flows, and it prefers the correct-label bank ------
    D, N, B = 16, 60, 8
    bank = torch.nn.functional.normalize(
        torch.as_tensor(_synth_keys(N, D, rng, 0.99), dtype=torch.float32), dim=1)
    blab = torch.as_tensor((np.arange(N) % 2), dtype=torch.long)
    q = bank[:B].clone().requires_grad_(True)
    y = blab[:B].clone()
    diag = {"boundary_sigma": [], "active_frac": [], "nonzero_w": []}
    L = rvs_term(q, y, torch.arange(B), bank, blab, diag)
    L.backward()
    assert torch.isfinite(L) and float(L) > 0, "RVS loss not finite/positive"
    assert q.grad is not None and float(q.grad.abs().sum()) > 0, "RVS gives no gradient"
    # self-exclusion: the query's own row must never receive weight
    q2 = bank[:B].clone().requires_grad_(True)
    diag2 = {"boundary_sigma": [], "active_frac": [], "nonzero_w": []}
    L_self = rvs_term(q2, 1 - y, torch.arange(B), bank, blab, diag2)
    assert float(L_self) > float(L), (
        "RVS must penalise the flipped-label target more than the true one "
        "({:.4f} vs {:.4f})".format(float(L_self), float(L)))
    ok.append("rvs_term grad_ok loss_true={:.4f} loss_flipped={:.4f} nonzero_w={:.1f}"
              .format(float(L), float(L_self), diag["nonzero_w"][0]))

    # ---------- 2. selection loop matches the frozen retrieval semantics ---------------
    Dm = np.array([[0.9, 0.8, 0.7, 0.6]], dtype="float32")
    Im = np.array([[0, 1, 2, 3]], dtype="int64")
    bl = np.array([1, 0, 0, 1])
    hr, hs, pr, ps = _select_neighbours(Dm, Im, bl, np.array([1]), 1, 1)
    assert hr[0, 0] == 1 and pr[0, 0] == 0, "selection loop picked the wrong rows"
    ok.append("selection_loop first-opposite=1 first-same=0")

    # ---------- 3. TRA path: planted positive, planted null, random control ------------
    def _tra_case(nq, nb, n_bad, seed_):
        """A similarity matrix whose kNN vote is informative, optionally poisoned with
        `n_bad` bank rows that outrank everything and all carry the wrong label."""
        r = np.random.default_rng(seed_)
        yq_ = (np.arange(nq) % 2)
        bl_ = (np.arange(nb) % 2)
        S_ = np.where(yq_[:, None] == bl_[None, :], 0.90, 0.50) \
            + 0.03 * r.normal(size=(nq, nb))
        if n_bad:
            bad_ = np.arange(n_bad)
            bl_[bad_] = 1
            S_[:, bad_] = 0.99 + 0.005 * r.normal(size=(nq, n_bad))
        return np.clip(S_, -1, 1), bl_, yq_

    def _run(S_, bl_, y_, rho_=None):
        o_ = np.argsort(-S_, axis=1, kind="stable")
        Ss_ = np.take_along_axis(S_, o_, axis=1)
        sg_ = (2 * bl_ - 1).astype("float64")[None, :]
        Ls_ = np.take_along_axis(np.broadcast_to(sg_, S_.shape), o_, axis=1)
        mask = (np.ones_like(Ss_, dtype=bool) if rho_ is None else
                np.take_along_axis(S_ >= rho_[None, :], o_, axis=1))
        return float((_tra_eval(Ss_, Ls_, mask) == y_).mean()), o_, sg_

    # planted POSITIVE: 12 poisoned rows; a per-row radius that rejects them must recover
    Sp, blp, yq = _tra_case(60, 80, 12, 11)
    t_fit0 = time.time()
    _, o_p, sg_p = _run(Sp, blp, yq)
    rho = _tra_fit(Sp, o_p, sg_p, yq, None)
    fit_secs = time.time() - t_fit0
    a0 = _run(Sp, blp, yq)[0]
    a1 = _run(Sp, blp, yq, rho)[0]
    assert a1 > a0 + 0.10, (
        "planted-positive TRA case not recovered: {:.3f} -> {:.3f}".format(a0, a1))
    # random control: permuting the fitted radii across rows must NOT recover it
    rperm = np.random.default_rng(0)
    rnd = float(np.mean([_run(Sp, blp, yq, rho[rperm.permutation(len(rho))])[0]
                         for _ in range(5)]))
    assert rnd < a1 - 0.05, (
        "rejection-rate-matched random control matched the fitted radii "
        "({:.3f} vs {:.3f}) -- the control cannot discriminate".format(rnd, a1))
    # planted NULL, evaluated OUT OF SAMPLE: clean geometry, nothing to fix; the fitter
    # must not manufacture a held-out gain from noise.
    Sn, bln, yn = _tra_case(60, 80, 0, 12)
    tr_m = np.arange(60) % 5 != 0
    te_m = ~tr_m
    rho_n = _tra_fit(Sn[tr_m], np.argsort(-Sn[tr_m], axis=1, kind="stable"),
                     (2 * bln - 1).astype("float64")[None, :], yn[tr_m], None)
    n0 = _run(Sn[te_m], bln, yn[te_m])[0]
    n1 = _run(Sn[te_m], bln, yn[te_m], rho_n)[0]
    assert n1 - n0 < 0.10, (
        "planted-null TRA case manufactured a held-out gain: {:.3f} -> {:.3f}".format(n0, n1))
    ok.append("tra positive {:.3f}->{:.3f} (random control {:.3f}) | null heldout "
              "{:.3f}->{:.3f} | reject_frac={:.3f} | fit={:.2f}s for nb=80".format(
                  a0, a1, rnd, n0, n1, float((rho > -1).mean()), fit_secs))

    # ---------- 4. collect/verdict machinery: planted positive AND planted null --------
    n = 200
    lab = (np.arange(n) % 2)
    fold_of = np.arange(n) % 5
    for arm, eff in (("b2", 0.60), ("b3", 0.00)):
        for ds in DATASETS:
            for s in SEEDS:
                _synth_head_files(sub, arm, ds, s, None, lab, fold_of, eff,
                                  np.random.default_rng(100 + s))
    for lam, eff in zip(RVS_LAMBDAS, (0.0, 0.0, 0.0)):
        for ds in DATASETS:
            for s in SEEDS:
                _synth_head_files(sub, "b1", ds, s, lam, lab, fold_of, eff,
                                  np.random.default_rng(200 + s))
    write_json(os.path.join(sub, "b4.json"), {"verdict": "KILLED", "note": "synthetic"})
    rep = run_collect(sub)
    assert rep["verdicts"]["B2_XFM"].startswith("ALIVE"), (
        "planted POSITIVE was not detected: {}".format(rep["verdicts"]["B2_XFM"]))
    assert rep["verdicts"]["B3_AQM_trained_gphi"] == "KILLED", (
        "planted NULL was not detected: {}".format(rep["verdicts"]["B3_AQM_trained_gphi"]))
    assert rep["verdicts"]["B1_RVS"] == "KILLED", "B1 planted null not detected"
    assert rep["arms"]["B1_RVS"]["lambda_star"] in [str(x) for x in RVS_LAMBDAS]
    ok.append("collect planted_positive={} planted_null={} lambda_star={}".format(
        rep["verdicts"]["B2_XFM"][:14], rep["verdicts"]["B3_AQM_trained_gphi"],
        rep["arms"]["B1_RVS"]["lambda_star"]))

    # self-recall veto must fire
    for ds in DATASETS:
        for s in SEEDS:
            for f in FOLDS:
                p = head_out_path(sub, "b3", ds, s, f, None)
                z = dict(np.load(p, allow_pickle=True))
                m = json.loads(str(z["meta"])); m["self_recall"] = 0.5
                z["meta"] = json.dumps(m)
                np.savez(p, **z)
            _synth_head_files(sub, "b3", ds, s, None, lab, fold_of, 0.60,
                              np.random.default_rng(300 + s))
            for f in FOLDS:
                p = head_out_path(sub, "b3", ds, s, f, None)
                z = dict(np.load(p, allow_pickle=True))
                m = json.loads(str(z["meta"])); m["self_recall"] = 0.5
                z["meta"] = json.dumps(m)
                np.savez(p, **z)
    rep2 = run_collect(sub)
    assert "memory-corruption" in rep2["verdicts"]["B3_AQM_trained_gphi"], (
        "self-recall veto did not fire: {}".format(rep2["verdicts"]["B3_AQM_trained_gphi"]))
    ok.append("self_recall_veto {}".format(rep2["verdicts"]["B3_AQM_trained_gphi"][:38]))

    # ---------- 5. REAL training smoke test through the patched code paths -------------
    smoke = _training_smoke(sub)
    ok.extend(smoke)

    print("\n[msb selftest] ALL PATHS OK")
    for line in ok:
        print("  - " + line)
    shutil.rmtree(sub, ignore_errors=True)
    return 0


def _training_smoke(sub):
    """Drive the ACTUAL patched training paths (B1/B2/B3) on random features through
    run_rac.model_pass for 2 epochs.  This is the check that caught every real C06 bug:
    a synthetic end-to-end drive of the code that will run unattended."""
    import torch
    import run_rac
    import model.loss as LOSS
    from data_loader.rac_dataloader import CLIP2Dataloader
    from model.classifier import classifier_hateClipper
    out = []
    n, dfeat = 48, 64
    rng = np.random.default_rng(7)
    ids = ["v%03d" % i for i in range(n)]
    img = torch.as_tensor(rng.normal(size=(n, dfeat)), dtype=torch.float32)
    txt = torch.as_tensor(rng.normal(size=(n, dfeat)), dtype=torch.float32)
    lab = torch.as_tensor((np.arange(n) % 2), dtype=torch.long)
    fold_of = np.arange(n) % 5

    class A(object):
        pass
    _orig_save = torch.save

    def _stub_save(obj, path, *ar, **kw):
        """No 34 MB state_dict dumps in a smoke test, but the file must EXIST: the
        end-of-training best-epoch bookkeeping shutil.copy's it."""
        os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
        open(str(path), "wb").close()

    torch.save = _stub_save
    for arm in ARMS:
        a = A()
        for k, v in dict(device="cpu", batch_size=16, lr=1e-4, epochs=2, topk=5,
                         proj_dim=32, map_dim=32, dropout=[0.2, 0.4, 0.1],
                         fusion_mode="align", hard_negatives_loss=True,
                         no_hard_negatives=1, no_pseudo_gold_positives=1,
                         hard_negatives_multiple=12, metric="cos", loss="triplet",
                         batch_norm=False, hybrid_loss=True, ce_weight=0.5,
                         warmup=0, triplet_margin=0.1, grad_clip=1.0,
                         lr_scheduler=False, reindex_every_step=False,
                         eval_retrieval=False, sparse_dictionary=None,
                         pos_weight_value=None, Faiss_GPU=False, dataset="HateMM",
                         similarity_threshold=-1., head_loss="triplet",
                         lambda_seg=0.0, lambda_aux=0.0, lambda_tarc=0.0,
                         sam=False, mixup=False, seed=0, num_workers=0,
                         output_path=os.path.join(sub, "smoke_" + arm),
                         save_embed=False, model="smoke", exp_comment="",
                         group_name="smoke").items():
            setattr(a, k, v)
        os.makedirs(a.output_path, exist_ok=True)
        (train_dl,) = CLIP2Dataloader((ids, img, txt, lab), batch_size=a.batch_size)
        model = classifier_hateClipper(dfeat, dfeat, 1, a.proj_dim, a.map_dim,
                                       a.fusion_mode, dropout=a.dropout,
                                       batch_norm=False, args=a)
        if arm == "b1":
            armobj = ArmB1(0.3, active=True)
        elif arm == "b2":
            armobj = ArmB2("hatemm", 0, 0, active=True)
            # synthetic sibling banks (the real ones come from the banked mints).
            # fold 0 is this head's held-out fold, so no synthetic item carries it.
            import faiss
            sfold = np.where(fold_of == 0, 1, fold_of)
            armobj.fold_of = sfold
            banks = {}
            for k in FOLDS:
                if k == 0:
                    continue
                rows = np.flatnonzero((sfold != k) & (sfold != 0))
                K = np.ascontiguousarray(
                    rng.normal(size=(len(rows), a.proj_dim)).astype("float32"))
                faiss.normalize_L2(K)
                ixk = faiss.IndexFlatIP(K.shape[1]); ixk.add(K)
                banks[k] = (K, lab.numpy()[rows], rows, ixk)

            def _fake_banks(_b=banks, _obj=armobj, _ids=ids, _sf=sfold):
                _obj.id_to_fold = {v: int(_sf[i]) for i, v in enumerate(_ids)}
                return _b
            armobj._build_banks = _fake_banks
        else:
            armobj = ArmB3(active=True)

        orig_mp = run_rac.model_pass
        # the epoch-end dev/test bookkeeping is orthogonal to the patched paths and wants
        # an output_path + wandb; stub it so the smoke exercises loss/mining/optimizer only
        orig_ee = run_rac.eval_and_save_epoch_end
        run_rac.eval_and_save_epoch_end = (
            lambda *ar, **kw: ((0., 0., 0., 0., 0., 0.), (0., 0., 0., 0., 0.)))
        t0 = time.time()
        try:
            if arm == "b3":
                armobj.install_model_pass(run_rac)
                run_rac.model_pass(train_dl, train_dl, train_dl, model,
                                   epochs=a.epochs, args=a, artifacts=None,
                                   train_set=None)
            else:
                armobj.install(run_rac, model, a)
                run_rac.model_pass(train_dl, train_dl, train_dl, model,
                                   epochs=a.epochs, args=a, artifacts=None,
                                   train_set=None)
        finally:
            armobj.uninstall()
            run_rac.model_pass = orig_mp
            run_rac.eval_and_save_epoch_end = orig_ee
        secs = time.time() - t0
        for p in model.parameters():
            assert torch.isfinite(p).all(), "{}: non-finite parameter after training".format(arm)
        if arm == "b3":
            gs = [float(p.abs().sum()) for p in armobj.g_phi.parameters()]
            m0 = [float(p.abs().sum()) for p in model.parameters()]
            assert any(g > 0 for g in gs), "g_phi parameters vanished"
            moved = any(abs(g - m) > 1e-9 for g, m in zip(gs, m0))
            assert moved, "g_phi did not move away from f_theta -> it is not being trained"
            out.append("smoke b3 ok {:.1f}s g_phi_trained=True epochs_banked={}".format(
                secs, armobj.stats["epochs"]))
        elif arm == "b1":
            r = armobj.report()
            out.append("smoke b1 ok {:.1f}s nonzero_w={} boundary_sigma={}".format(
                secs, r["nonzero_w"], r["boundary_sigma"]))
        else:
            out.append("smoke b2 ok {:.1f}s epochs_reindexed={}".format(
                secs, armobj.stats["epochs"]))
    torch.save = _orig_save
    return out


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["instrument", "head", "b4", "collect", "selftest"])
    ap.add_argument("--outroot", default=os.path.join(REPO, "artifacts/mech_stage_b"))
    ap.add_argument("--scratch",
                    default=os.path.join(REPO, "artifacts/mech_stage_b/scratch"))
    ap.add_argument("--progress", default=None)
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--parity", action="store_true",
                    help="train the arm's mechanism-OFF parity head (prereg §0.4)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    if a.stage == "selftest":
        return run_selftest(a.outroot)
    assert_frozen()
    if a.stage == "instrument":
        run_instrument(a.outroot, a.progress)
    elif a.stage == "head":
        assert a.arm and a.dataset and a.seed is not None and a.fold is not None
        run_head(a.arm, a.dataset, a.seed, a.fold, None if a.parity else a.lam,
                 not a.parity, a.outroot, a.scratch, a.threads, a.progress)
    elif a.stage == "b4":
        r = run_b4(a.outroot, a.progress)
        print("[msb] B4 {}".format(r["verdict"]), flush=True)
    elif a.stage == "collect":
        r = run_collect(a.outroot, a.progress)
        print("[msb] verdicts {}".format(json.dumps(r["verdicts"])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
