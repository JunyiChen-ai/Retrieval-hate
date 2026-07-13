"""SAV (C2) F-G0/F-G1 shared constants, geometry, IO, statistics.

Pre-registration authority: research-wiki/experiments/exp-sav-f0.md (Rev-2a, APPROVED).
This module is imported by sav_f0_extract.py (F-G0 extraction), sav_f0_guard.py
(F-G0(b) two-tier reproduction guard) and sav_f0_probe.py (F-G1 statistics engine).

NOTHING here submits jobs, mutates the prereg, or touches TEST labels. All temp /
intermediate artifacts live IN-REPO under artifacts/sav_f0/ and slurm/tmp/ (realbank
$TMPDIR burn lesson). All third-party imports are top-level (no deferred/function-level
imports) so the deferred-import audit is trivial — see refine-logs/SAV_F0_IMPL_NOTES.md.

Provenance pins (verified live 2026-07-13):
  * Frozen extractor mirrored: src/utils/generate_VideoMLLM_embedding_HF.py
      - IMG/TEXT instruction constants  : lines 44-52
      - decord->PyAV 8-frame sampler    : lines 146-235
      - message build + span pooling    : lines 241-323
      - cache contract {ids:[[..]],img_feats,text_feats,labels} : lines 430-438
  * Exact extraction hyper-params that PRODUCED the banked caches
      (scripts/slurm/gen_embed_mllm.sbatch:30-32 -> script defaults):
      num_frames=8 ; max_pixels=360*420=151200 ; bf16 ; attn_implementation="sdpa".
      The reproduction guard (F-G0(b)) is only valid if these match, so they are pinned.
  * Qwen2.5-VL-7B geometry (config.json, read live): num_hidden_layers=28,
      num_attention_heads=28, hidden_size=3584 -> head_dim=128 ; num_key_value_heads=4
      (GQA does NOT change the per-QUERY-head output count: 28*28=784 head positions).
  * o_proj module path (transformers 4.49.0 modeling_qwen2_5_vl.py):
      text decoder self_attn.o_proj at :735/:798 (in_features = 28*128 = 3584);
      the vision tower uses `.proj`/`.attn.proj` (NOT self_attn.o_proj), so a
      name.endswith('self_attn.o_proj') filter selects EXACTLY the 28 LLM layers.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Repo topology (in-repo only; never $TMPDIR)                                  #
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/analysis/<file> -> repo root
assert (REPO_ROOT / "src" / "utils" / "generate_VideoMLLM_embedding_HF.py").exists(), (
    "sav_f0_common: repo-root resolution failed ({})".format(REPO_ROOT)
)

ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "sav_f0"
EXTRACT_ROOT = ARTIFACT_ROOT / "extract"
GUARD_ROOT = ARTIFACT_ROOT / "guard"
PROBE_ROOT = ARTIFACT_ROOT / "probe"
REPO_TMPDIR = REPO_ROOT / "slurm" / "tmp"  # in-repo scratch (realbank $TMPDIR lesson)

DATA_ROOT = REPO_ROOT / "data"
GT_ROOT = DATA_ROOT / "gt"
VIDEO_ROOT = DATA_ROOT / "video"
CACHE_ROOT = DATA_ROOT / "CLIP_Embedding"

# --------------------------------------------------------------------------- #
# Frozen model / extraction pins (must match the banked enc3s extraction)      #
# --------------------------------------------------------------------------- #
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CACHE_TAG = "Qwen2.5-VL-7B-Instruct_HF"           # banked pooled-feature filename tag
NUM_FRAMES = 8                                     # gen_embed_mllm.sbatch default
MAX_PIXELS = 360 * 420                             # 151200; generate_VideoMLLM_embedding_HF.py:99 default
TORCH_DTYPE = torch.bfloat16
ATTN_IMPL = "sdpa"

# Geometry (asserted against the loaded model config at runtime; fail-closed).
NUM_LAYERS = 28
NUM_HEADS = 28
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM                      # 3584
NUM_HEAD_POSITIONS = NUM_LAYERS * NUM_HEADS        # 784

# Instructions copied VERBATIM from generate_VideoMLLM_embedding_HF.py:45-52.
IMG_INSTRUCTION = (
    "Describe the people, symbols, gestures, and on-screen text in this video."
)
TEXT_INSTRUCTION = (
    "You are analysing a short video for potentially hateful or offensive content. "
    "Considering the frames together with the provided title and transcript, "
    "summarise the targets, symbols, tone, and any harmful intent conveyed."
)

# --------------------------------------------------------------------------- #
# Dataset / split pins + FROZEN expected counts (cross-check simulation table) #
#   HateMM train 744 / val 107 ; MHC 549 / 80 ; MHC_zh 579 / 78 (counted live) #
# --------------------------------------------------------------------------- #
DATASETS = ["HateMM", "MHC", "MHC_zh"]
SPLITS = ["train", "val"]                           # F-G0/F-G1 use train+val only
SPLIT_TO_OUTNAME = {"train": "train", "val": "dev_seen"}  # subset of the extractor map
EXPECTED_COUNTS = {
    "HateMM": {"train": 744, "val": 107},
    "MHC": {"train": 549, "val": 80},
    "MHC_zh": {"train": 579, "val": 78},
}
CARRYING_DATASET = "MHC"        # MHC-EN is the dilution target that carries the line
NOHARM_DATASET = "HateMM"       # banked encoder-swap win; must not regress
SECONDARY_DATASET = "MHC_zh"    # secondary / completeness (non-gating)

# --------------------------------------------------------------------------- #
# F-G0(b) reproduction-guard thresholds (Rev-2 R2, two-tier, pre-declared)     #
# --------------------------------------------------------------------------- #
GUARD_PRIMARY_MIN_COSINE = 0.999   # min per-video cosine, fresh-vs-cached, img+text
GUARD_SECONDARY_ACC_TOL = 0.010    # ±0.010 val-acc confirmatory probe (secondary)
ZERO_NORM_EPS = 1e-6               # a cached zero-vector (decode-failed) has ~0 L2 norm

# --------------------------------------------------------------------------- #
# F-G1 statistics pins (Rev-2 R1a/R1b/R1c + Rec-1)                             #
# --------------------------------------------------------------------------- #
SEEDS = [0, 1, 2, 3, 4]                 # >=5 seeds (exceeds the >=3 asked)
SELECTION_PER_CLASS = 20                # SAV few-shot scale, without replacement, from train
TOPK_SWEEP = [10, 20, 40]               # top-k heads swept
PROBE_TRAIN_FRAC = 0.80                 # stratified probe-train resample per seed
CV_FOLDS = 5                            # 5-fold CV for L2 lambda, in-train only
LAMBDAS = np.logspace(-4, 2, 7)         # L2 penalty grid {1e-4 .. 1e2} log-spaced
PROBE_MAX_ITER = 2000
BOOTSTRAP_DRAWS = 10000                 # example-level clustered bootstrap
BOOTSTRAP_SEED = 20260713               # fixed so the CI is reproducible
PROB_CLIP = 1e-6                        # holdout log-loss p-hat clip [1e-6, 1-1e-6]

# Decision bar (F-G1): projected gain must exceed +0.030 + noise band, CI excludes 0.
# REFLECTION_mllm_integration_failures.md:41 "bits->acc ... > +3 acc + 噪声带".
# NOISE_BAND_ACC pinned to 0.010 by main-loop ruling 2026-07-13 (Rev-2b): protocol
# consistency with the A-line G0-cond probe precedent
# (refine-logs/lb_scgp_global/M1_G0COND_PROBE_RECORD.md used +0.030 + 0.01 = +0.040);
# a per-gate drifting bar invites protocol-inconsistency criticism.
PROJECTED_GAIN_BASE = 0.030
NOISE_BAND_ACC = 0.010            # A-line G0-cond precedent (Rev-2b main-loop ruling)
PROJECTED_GAIN_BAR = PROJECTED_GAIN_BASE + NOISE_BAND_ACC   # 0.040
HATEMM_NOHARM_DACC = -0.010      # HateMM no-harm: Δacc CI not below -0.010

# The F-G1 probe operates on ONE stream: the IMG forward (visual+instruction "prefix"
# span). ACCEPTED by main-loop ruling 2026-07-13 (Rev-2b): the img stream pools over the
# WHOLE visual+instruction span (hundreds of tokens) — the literal "mean-pooling
# dilution" target of hypothesis H — so the C-pos position control is non-degenerate here
# (unlike the text stream, whose pooled read-out is already a near-final-token response
# tail). Per-head extraction is IMG-only (execution note 2 storage: one stream, two
# variants ~2.4 GB); the text stream is still forward-passed and pooled for the guard.
# Text-stream / concat per-head extraction is DEFERRED as an F-G2-stage option only if
# SAV wins F-G1 (pre-declared in exp-sav-f0.md F-G1, Rev-2b — not a post-hoc DoF).
PROBE_STREAM = "img"

# Probe arm identifiers (fail-closed: probe verdict requires all of these present).
PROBE_ARMS = ["pooled", "SAV", "C-pos", "C-sparse", "U-1", "U-2"]


# --------------------------------------------------------------------------- #
# Canonical JSON + atomic IO (mirrors cte_common.py:123-186 discipline)        #
# --------------------------------------------------------------------------- #
def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_obj(obj):
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _atomic_publish(tmp, path):
    """Atomic same-dir publish; overwrite the tmp target via os.replace."""
    os.replace(tmp, str(path))
    dfd = os.open(str(Path(path).parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def atomic_write_json(path, payload, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise RuntimeError("refusing to overwrite artifact: {}".format(path))
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_publish(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch_save(path, payload, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise RuntimeError("refusing to overwrite artifact: {}".format(path))
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        _atomic_publish(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------- #
# Path helpers                                                                 #
# --------------------------------------------------------------------------- #
def gt_path(dataset, split):
    return GT_ROOT / dataset / "{}.jsonl".format(split)


def video_path(dataset, vid):
    return VIDEO_ROOT / dataset / "All" / "{}.mp4".format(vid)


def cached_pooled_path(dataset, split):
    outname = SPLIT_TO_OUTNAME[split]
    return CACHE_ROOT / dataset / "{}_{}.pt".format(outname, CACHE_TAG)


def extract_split_dir(dataset, split):
    return EXTRACT_ROOT / dataset / split


def extract_video_path(dataset, split, vid):
    return extract_split_dir(dataset, split) / "{}.pt".format(vid)


def extract_manifest_path(dataset, split):
    return extract_split_dir(dataset, split) / "_manifest.json"


def guard_path(dataset):
    return GUARD_ROOT / dataset / "guard.json"


def probe_verdict_path():
    return PROBE_ROOT / "verdict.json"


# --------------------------------------------------------------------------- #
# GT reader — mirrors generate_VideoMLLM_embedding_HF.py:123-140               #
# --------------------------------------------------------------------------- #
def read_gt(dataset, split):
    items = []
    with open(gt_path(dataset, split), "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                {
                    "id": str(obj["id"]),
                    "text": "" if obj.get("text") is None else str(obj["text"]),
                    "title": "" if obj.get("title") is None else str(obj.get("title", "")),
                    "label": int(obj["label"]),
                }
            )
    return items


# --------------------------------------------------------------------------- #
# Fano / inverse-binary-entropy projection (R1c)                              #
# --------------------------------------------------------------------------- #
def binary_entropy_bits(p):
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def h2inv_lower(e):
    """Lower inverse of the binary entropy function: return p in [0, 0.5] with H2(p)=e."""
    e = float(e)
    if e <= 0.0:
        return 0.0
    if e >= 1.0:
        return 0.5
    lo, hi = 0.0, 0.5
    for _ in range(80):  # bisection to ~1e-24 precision
        mid = 0.5 * (lo + hi)
        if binary_entropy_bits(mid) < e:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fano_acc(ell_bits):
    """Fano-projected ceiling accuracy for mean per-example codelength `ell_bits`."""
    return 1.0 - h2inv_lower(min(float(ell_bits), 1.0))


# --------------------------------------------------------------------------- #
# Holdout log-loss (bits) with clipping (R1c primary MDL estimator)           #
# --------------------------------------------------------------------------- #
def per_example_bits(proba_pos, y):
    """Per-example description length (bits): -log2 p_hat(y_i). Clipped to [1e-6,1-1e-6]."""
    proba_pos = np.clip(np.asarray(proba_pos, dtype=np.float64), PROB_CLIP, 1.0 - PROB_CLIP)
    y = np.asarray(y, dtype=np.int64)
    p_true = np.where(y == 1, proba_pos, 1.0 - proba_pos)
    return -np.log2(p_true)


# --------------------------------------------------------------------------- #
# Nearest-centroid SAV head selection (§2 procedure, R1a) — cosine geometry    #
# --------------------------------------------------------------------------- #
def _l2norm_rows(x, eps=1e-12):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def head_nearest_centroid_accuracy(head_vecs, labels):
    """SAV per-head nearest-centroid cosine accuracy on the given (selection) set.

    head_vecs : [n, NUM_HEAD_POSITIONS, HEAD_DIM] final-token per-head attention vectors.
    labels    : [n] in {0,1}.
    Returns   : [NUM_HEAD_POSITIONS] resubstitution nearest-centroid cosine accuracy,
                exactly SAV's "measure the classification accuracy of each head" on the
                labelled few-shot set (arXiv 2412.00142v3, §2).
    """
    head_vecs = np.asarray(head_vecs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n, H, d = head_vecs.shape
    classes = np.array([0, 1], dtype=np.int64)
    acc = np.zeros(H, dtype=np.float64)
    for h in range(H):
        x = _l2norm_rows(head_vecs[:, h, :])          # [n, d] cosine geometry
        centroids = []
        for c in classes:
            mask = labels == c
            if not np.any(mask):
                centroids.append(np.zeros(d))
            else:
                centroids.append(x[mask].mean(axis=0))
        C = _l2norm_rows(np.stack(centroids, axis=0))  # [2, d]
        sims = x @ C.T                                 # [n, 2] cosine to each centroid
        pred = classes[np.argmax(sims, axis=1)]
        acc[h] = float((pred == labels).mean())
    return acc


def rank_heads(acc):
    """Deterministic descending rank of heads by accuracy; ties broken by head index."""
    return sorted(range(len(acc)), key=lambda h: (-acc[h], h))


# --------------------------------------------------------------------------- #
# Example-level clustered bootstrap (R1b)                                      #
#   Per-example paired deltas are averaged ACROSS SEEDS FIRST; the bootstrap    #
#   then resamples the n examples with replacement. Effective n stays n.        #
# --------------------------------------------------------------------------- #
def clustered_bootstrap_mean(per_example_seed_avg, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """CI on the mean of a per-example (already seed-averaged) paired quantity."""
    v = np.asarray(per_example_seed_avg, dtype=np.float64)
    n = v.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    boot = v[idx].mean(axis=1)
    return {
        "mean": float(v.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n_effective": int(n),
        "excludes_zero": bool(np.percentile(boot, 2.5) > 0.0 or np.percentile(boot, 97.5) < 0.0),
    }


# --------------------------------------------------------------------------- #
# Loaders that assemble per-dataset arrays from the per-video extract caches    #
#   (torch/numpy only — keeps the GPU extract import graph sklearn-free)        #
# --------------------------------------------------------------------------- #
def load_extracted_split(dataset, split, with_heads=True):
    """Assemble arrays for one split from the per-video extract .pt caches (gt order).

    Fail-closed: every gt id must have a cache file; counts must match EXPECTED_COUNTS.
    Returns a dict of numpy/torch arrays aligned to gt order.
    """
    items = read_gt(dataset, split)
    exp = EXPECTED_COUNTS[dataset][split]
    assert len(items) == exp, "gt drift {}/{}: {} != {}".format(dataset, split, len(items), exp)
    ids, labels, ok = [], [], []
    img_pooled, text_pooled, img_hidden_final = [], [], []
    head_final, head_span = [], []
    for it in items:
        vid = it["id"]
        p = extract_video_path(dataset, split, vid)
        if not p.exists():
            raise FileNotFoundError("missing extract cache: {}".format(p))
        obj = torch.load(p, map_location="cpu")
        assert obj["id"] == vid, "id mismatch in {}: {} != {}".format(p, obj["id"], vid)
        ids.append(vid)
        labels.append(int(obj["label"]))
        ok.append(bool(obj["ok"]))
        img_pooled.append(obj["img_pooled"].float())
        text_pooled.append(obj["text_pooled"].float())
        img_hidden_final.append(obj["img_hidden_final"].float())
        if with_heads:
            head_final.append(obj["img_head_final"].float())
            head_span.append(obj["img_head_spanmean"].float())
    out = {
        "ids": ids,
        "labels": np.asarray(labels, dtype=np.int64),
        "ok": np.asarray(ok, dtype=bool),
        "img_pooled": torch.stack(img_pooled).numpy(),
        "text_pooled": torch.stack(text_pooled).numpy(),
        "img_hidden_final": torch.stack(img_hidden_final).numpy(),
    }
    if with_heads:
        # [N, 784, 128]
        out["head_final"] = torch.stack(head_final).reshape(len(ids), NUM_HEAD_POSITIONS, HEAD_DIM).numpy()
        out["head_span"] = torch.stack(head_span).reshape(len(ids), NUM_HEAD_POSITIONS, HEAD_DIM).numpy()
    return out


def load_cached_pooled(dataset, split):
    """Return {id: (img_feat[3584], text_feat[3584])} from the banked enc3s cache."""
    obj = torch.load(cached_pooled_path(dataset, split), map_location="cpu")
    ids = [x for sub in obj["ids"] for x in sub]
    img = obj["img_feats"].float().numpy()
    txt = obj["text_feats"].float().numpy()
    assert len(ids) == img.shape[0] == txt.shape[0], "cached pooled shape drift {}/{}".format(dataset, split)
    return {ids[i]: (img[i], txt[i]) for i in range(len(ids))}


def clustered_bootstrap_projection(ell_pooled_seed_avg, ell_sav_seed_avg,
                                   draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """CI on the Fano projected-gain acc(ell_SAV) - acc(ell_pooled).

    ell_*_seed_avg : [n] per-example seed-averaged codelength (bits) for each arm.
    Within each bootstrap draw we resample examples, recompute the AGGREGATE mean
    codelength per arm over the drawn examples, then project each with Fano and take
    the difference (R1b: same clustered rule applied to the projected-gain bootstrap).
    """
    ep = np.asarray(ell_pooled_seed_avg, dtype=np.float64)
    es = np.asarray(ell_sav_seed_avg, dtype=np.float64)
    n = ep.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    point = fano_acc(es.mean()) - fano_acc(ep.mean())
    boot = np.empty(draws, dtype=np.float64)
    for b in range(draws):
        di = idx[b]
        boot[b] = fano_acc(es[di].mean()) - fano_acc(ep[di].mean())
    return {
        "mean": float(point),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n_effective": int(n),
        "excludes_zero": bool(np.percentile(boot, 2.5) > 0.0 or np.percentile(boot, 97.5) < 0.0),
        "acc_proj_pooled": float(fano_acc(ep.mean())),
        "acc_proj_sav": float(fano_acc(es.mean())),
    }
