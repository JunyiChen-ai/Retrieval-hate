#!/usr/bin/env python
"""TERA Gate-0 — shared primitives (hashing, seeds, threshold rule, metrics).

Implements the frozen conventions of research-wiki/EXP_tera_gate0_impl_appendix.md
(v2): the canonical payload hash (sec 10.2), the deterministic seed derivation
(sec 3.4), the threshold selection rule (sec 3.3), macro-F1 (sec 1) and the
detached-run progress line (sec 0.3).

Repository root is taken from TERA_REPO_ROOT, else derived from this file's
location.  No absolute path is hard-coded anywhere in this package.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------- constants --
K_WINDOWS = 30
MODEL_SEED_BASE = 20260810
OUTER_FOLD_SEED = 20260807
INNER_FOLD_SEED = 20260808
BOOTSTRAP_SEED = 20260809
GATE_C_SEED = 20260807
B4_SEED = 20260807
B5_SEED = 20260807
FIXTURE_SEED_BASE = 424242

LR_GRID = (1e-2, 3e-3, 1e-3, 3e-4)
WD_GRID = (1e-4, 1e-2)
A2_K_GRID = (1, 2, 4)
A4_TAU_GRID = (0.1, 0.3, 1.0)
E_MAX = 200
PATIENCE = 40
MIN_DELTA = 1e-4
BATCH_SIZE = 64
TORCH_THREADS = 8

LR_TAG = {1e-2: "1e-2", 3e-3: "3e-3", 1e-3: "1e-3", 3e-4: "3e-4"}
WD_TAG = {1e-4: "1e-4", 1e-2: "1e-2"}
TAU_TAG = {0.1: "0.1", 0.3: "0.3", 1.0: "1.0"}


class TeraHalt(RuntimeError):
    """A registered HALT condition (prereg sec 12).  Never a performance negative."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__("%s: %s" % (code, detail) if detail else code)
        self.code = code
        self.detail = detail


# ------------------------------------------------------------------- paths --
def repo_root() -> Path:
    env = os.environ.get("TERA_REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ hashing --
def canonical_json(obj) -> str:
    """Appendix sec 10.2 canonicalization (matches scripts/analysis/edcm_a0.py)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(ids) -> str:
    return sha256_text("\n".join(sorted(ids)))


# ------------------------------------------------------------------- seeds --
def derive_seed(scope: dict) -> int:
    """Appendix sec 3.4.  scope must carry stage/arm/dataset/outer/inner/config."""
    required = {"stage", "arm", "dataset", "outer", "inner", "config"}
    missing = required - set(scope)
    if missing:
        raise TeraHalt("HALT_SEED_SCOPE", "missing scope keys %s" % sorted(missing))
    payload = canonical_json({"base": MODEL_SEED_BASE, **scope})
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31 - 1)


def setup_determinism(threads: int = TORCH_THREADS) -> dict:
    """Appendix sec 3.4 determinism switches (device cpu)."""
    import torch

    os.environ.setdefault("PYTHONHASHSEED", "0")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(threads)
    return {
        "torch_use_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
        "torch_num_threads": threads,
        "device": "cpu",
    }


# ----------------------------------------------------------------- metrics --
def macro_f1_counts(tp, pp, n_pos, n_neg):
    """Vectorized binary macro-F1 from (true positives, predicted positives)."""
    tp = np.asarray(tp, dtype=np.float64)
    pp = np.asarray(pp, dtype=np.float64)
    fp = pp - tp
    fn = n_pos - tp
    tn = n_neg - fp
    d1 = 2.0 * tp + fp + fn
    d0 = 2.0 * tn + fn + fp
    f1p = np.where(d1 > 0, 2.0 * tp / np.where(d1 > 0, d1, 1.0), 0.0)
    f1n = np.where(d0 > 0, 2.0 * tn / np.where(d0 > 0, d0, 1.0), 0.0)
    return (f1p + f1n) / 2.0


def macro_f1(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    tp = float(((y == 1) & (p == 1)).sum())
    pp = float((p == 1).sum())
    return float(macro_f1_counts(tp, pp, n_pos, n_neg))


def binary_metrics(y_true, scores, threshold) -> dict:
    """Primary + secondary + diagnostic metrics of prereg sec 8.1."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    p = (s >= threshold).astype(np.int64)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    rec_pos = tp / (tp + fn) if (tp + fn) else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) else 0.0
    prec_pos = tp / (tp + fp) if (tp + fp) else 0.0
    f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) else 0.0
    try:
        auroc = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None
    except ValueError:
        auroc = None
    return {
        "macro_f1": macro_f1(y, p),
        "balanced_accuracy": (rec_pos + rec_neg) / 2.0,
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
        "positive_class_f1": f1_pos,
        "auroc": auroc,
        "predicted_positive_rate": float(p.mean()) if len(y) else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n": int(len(y)),
        "threshold": float(threshold),
    }


# --------------------------------------------------------- threshold rule ---
def select_threshold(scores, labels):
    """Appendix sec 3.3.

    candidates = midpoints of consecutive unique scores, plus (min - 1e-6) and
    (max + 1e-6); prediction rule is `score >= theta`; argmax pooled macro-F1;
    ties -> smallest |theta - 0.5|, then smallest theta.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if s.size == 0:
        raise TeraHalt("HALT_EMPTY_SCORES", "threshold selection on an empty score set")
    u = np.unique(s)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    idx = np.searchsorted(u, s)
    per_value_n = np.bincount(idx, minlength=u.size).astype(np.float64)
    per_value_pos = np.bincount(idx, weights=(y == 1).astype(np.float64),
                                minlength=u.size)
    ge_n = per_value_n[::-1].cumsum()[::-1]     # items with score >= u[j]
    ge_tp = per_value_pos[::-1].cumsum()[::-1]

    thetas = [float(u[0] - 1e-6)]
    pp = [ge_n[0]]
    tp = [ge_tp[0]]
    for j in range(u.size - 1):
        thetas.append(float((u[j] + u[j + 1]) / 2.0))
        pp.append(ge_n[j + 1])
        tp.append(ge_tp[j + 1])
    thetas.append(float(u[-1] + 1e-6))
    pp.append(0.0)
    tp.append(0.0)

    m = macro_f1_counts(np.array(tp), np.array(pp), n_pos, n_neg)
    thetas = np.asarray(thetas, dtype=np.float64)
    best_m = float(m.max())
    tied = np.flatnonzero(m >= best_m - 0.0)          # exact float equality
    tied = tied[m[tied] == best_m]
    dist = np.abs(thetas[tied] - 0.5)
    tied = tied[dist == dist.min()]
    theta = float(thetas[tied].min())
    return theta, best_m


# ------------------------------------------------------------------- io ----
def write_json_new(path, obj, indent=1):
    """Non-overwriting writer (appendix sec 10.3: mode 'x')."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=indent, sort_keys=True)
        handle.write("\n")
    return path


def write_jsonl_new(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    return path


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ------------------------------------------------------------- progress ----
_T0 = time.time()


def progress(stage, arm, outer, cfg_i, cfg_n, epoch, t0=None):
    """Appendix sec 0.3 parseable progress line."""
    elapsed = int(time.time() - (t0 if t0 is not None else _T0))
    sys.stdout.write("[tera-gate0] stage=%s arm=%s outer=%s cfg=%s/%s epoch=%s "
                     "elapsed=%ss\n" % (stage, arm, outer, cfg_i, cfg_n, epoch, elapsed))
    sys.stdout.flush()


def note(msg):
    sys.stdout.write("[tera-gate0] %s\n" % msg)
    sys.stdout.flush()
