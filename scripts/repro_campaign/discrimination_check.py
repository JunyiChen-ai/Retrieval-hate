#!/usr/bin/env python
"""Does this component DISCRIMINATE, or does it merely RUN?

Motivation
----------
This campaign has now produced two independent silent-correctness failures.
Neither raised, neither had a wrong shape, neither left a bad exit code:

  * `openai/clip-vit-base-patch16` downloaded with the right byte count and
    wrong content (max |w| = 3.7e19).  It returned ONE identical image
    embedding for every frame of every video, so every downstream curve was
    flat.  `MODEL_ASSETS_STATUS §3.1` recorded that as a property of LaGoVAD's
    binary head for a week.
  * VideoLLaMA3's eager `VisionAttention` adds a *bool* mask to float logits,
    promoting True to 1.0.  It applies a +1.0 in-block bias and keeps attending
    globally instead of masking, and still emits plausible captions.

"It ran and the shapes are right" does not distinguish these from a working
component.  The question that does is whether the output *varies with the
input*.  A collapsed encoder cannot discriminate; a working one must.

These are cheap assertions meant for a method's smoke test, before the corpus
run, where they cost seconds and save hours.  Every check returns a
`(bool, str)` so a caller can log rather than raise if it prefers.
"""
from __future__ import annotations

import numpy as np


def curve_varies(curve, name="curve", min_std=1e-6, min_distinct=2):
    """A per-frame score curve must not be constant over time.

    Catches the corrupt-CLIP failure: a constant visual embedding makes every
    frame score identical, which is a flat curve, which is a 0.5 AUC dressed up
    as a real measurement.
    """
    a = np.asarray(curve, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return False, f"{name}: empty"
    if not np.isfinite(a).all():
        return False, f"{name}: {(~np.isfinite(a)).sum()} non-finite values"
    n_distinct = len(np.unique(a))
    if a.std() < min_std or n_distinct < min_distinct:
        return False, (f"{name}: CONSTANT over time (std={a.std():.3g}, "
                       f"{n_distinct} distinct value(s)) -- a collapsed encoder "
                       f"or a broken checkpoint looks exactly like this")
    return True, f"{name}: varies (std={a.std():.4g}, {n_distinct} distinct)"


def embeddings_discriminate(emb, name="embeddings", max_mean_cos=0.9999):
    """Distinct inputs must not map to (near-)identical vectors.

    `emb` is [N, D] for N different inputs.  If the mean off-diagonal cosine is
    ~1 the encoder has collapsed -- the exact signature of the corrupt CLIP
    checkpoint, which no shape or range check would have caught.
    """
    e = np.asarray(emb, dtype=np.float64)
    if e.ndim != 2 or e.shape[0] < 2:
        return False, f"{name}: need [N>=2, D], got {e.shape}"
    if not np.isfinite(e).all():
        return False, f"{name}: non-finite values"
    n = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    c = n @ n.T
    off = c[~np.eye(len(c), dtype=bool)]
    if off.mean() > max_mean_cos:
        return False, (f"{name}: COLLAPSED (mean off-diagonal cosine "
                       f"{off.mean():.6f}) -- distinct inputs give the same "
                       f"vector; suspect the checkpoint before the method")
    return True, f"{name}: discriminates (mean off-diag cosine {off.mean():.4f})"


def scores_separate_items(scores, name="scores", min_distinct_frac=0.05):
    """Across items, a scorer must not return the same answer every time.

    Catches a prompt/model that has stopped reading its input -- e.g. the Wave 0
    Qwen row returning one identical interval for 19% of MHC-EN.  That is a real
    finding rather than a bug, so this is a warning-grade check: report the
    modal share and let the caller decide.
    """
    a = np.asarray(scores, dtype=np.float64).reshape(-1)
    if a.size < 2:
        return False, f"{name}: need >=2 items"
    vals, counts = np.unique(a, return_counts=True)
    frac = len(vals) / len(a)
    modal = counts.max() / len(a)
    if frac < min_distinct_frac:
        return False, (f"{name}: only {len(vals)} distinct over {len(a)} items "
                       f"(modal answer {modal:.1%}) -- the scorer may not be "
                       f"reading its input")
    return True, f"{name}: {len(vals)} distinct over {len(a)} (modal {modal:.1%})"


def report(*checks, strict=True):
    """Print each (ok, msg); raise SystemExit if any failed and strict."""
    bad = []
    for ok, msg in checks:
        print(("  [ok]   " if ok else "  [FAIL] ") + msg, flush=True)
        if not ok:
            bad.append(msg)
    if bad and strict:
        raise SystemExit("discrimination check failed: " + "; ".join(bad))
    return not bad

def patch_applied(n_matched, what="monkeypatch", expect_min=1):
    """A patch that matched zero modules is a silent no-op.

    Written after making this mistake: a memory patch for VideoLLaMA3 matched
    with `type(m).__name__ == "VisionAttention"`, but the class that actually
    runs is `VisionSdpaAttention`, a *subclass*.  Exact name matching skipped it
    entirely and `install()` returned 0 while reporting success.  Match with
    `isinstance` so subclasses are caught, and assert that something was hit --
    a patch nobody applied is indistinguishable from a patch that worked.
    """
    if n_matched < expect_min:
        return False, (f"{what}: matched {n_matched} modules (expected >= "
                       f"{expect_min}) -- SILENT NO-OP. Exact class-name "
                       f"matching misses subclasses; use isinstance.")
    return True, f"{what}: applied to {n_matched} module(s)"
