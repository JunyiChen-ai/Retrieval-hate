#!/usr/bin/env python
"""Shared fail-closed utilities for SQ S0/S1.

The sole gold target in this package is the parent video's binary label.
There is no segment-level gold and no code path in this package accepts one.
Validation/test files and teacher endpoints are intentionally absent.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path("/data/jehc223/RGCL")
ALLOWED_ARCHIVE_PATHS = {
    "id", "split", "parse_ok", "archive.neutral_summary"
}
FORBIDDEN_ARCHIVE_KEYS = {
    "label", "raw_output", "schema_ok", "refusal", "error",
    "target_groups", "mechanism", "modality_cues", "explicitness",
}


def canonical_json(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj):
    return sha256_text(canonical_json(obj))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(cfg, path_or_key):
    value = cfg["paths"].get(path_or_key, path_or_key)
    p = Path(value)
    return p if p.is_absolute() else Path(cfg["paths"]["root"]) / p


def require_runtime(gpu=False):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SQ computation must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")
    if gpu:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible:
            raise RuntimeError("GPU task has no CUDA_VISIBLE_DEVICES")


def config_payload_and_hash(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    payload = dict(cfg)
    payload.pop("config_sha256", None)
    return cfg, sha256_obj(payload)


def load_config(path):
    cfg, computed = config_payload_and_hash(path)
    if cfg.get("config_sha256") != computed:
        raise RuntimeError("config not frozen: stored={} computed={}".format(
            cfg.get("config_sha256"), computed))
    cfg["computed_config_sha256"] = computed
    if cfg["supervision"] != {
            "only_gold": "parent_video_binary_label",
            "segment_gold_exists": False,
            "segment_gold_used": False,
            "validation_test_forbidden": True,
            "new_teacher_calls_forbidden": True}:
        raise RuntimeError("immutable supervision contract changed")
    return cfg


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    raise RuntimeError("{}:{} {}".format(path, lineno, exc))
    return rows


class ArchiveAllowlistReader:
    """Runtime access ledger; only neutral_summary crosses the semantic boundary."""
    def __init__(self):
        self.access = {k: 0 for k in sorted(ALLOWED_ARCHIVE_PATHS)}
        self.forbidden_access_count = 0

    def _take(self, row, key):
        if key not in {"id", "split", "parse_ok"}:
            self.forbidden_access_count += 1
            raise KeyError("forbidden archive access: {}".format(key))
        self.access[key] += 1
        return row.get(key)

    def project(self, row):
        vid = self._take(row, "id")
        split = self._take(row, "split")
        parse_ok = self._take(row, "parse_ok")
        # Accessing this nested field is the one permitted semantic projection.
        self.access["archive.neutral_summary"] += 1
        archive = row.get("archive")
        summary = archive.get("neutral_summary") if isinstance(archive, dict) else None
        return {"id": str(vid), "split": split, "parse_ok": bool(parse_ok),
                "neutral_summary": summary}

    def read(self, path):
        return [self.project(x) for x in read_jsonl(path)]


def archive_reader_poison_fixture(reader):
    poison = {
        "id": "fixture", "split": "train", "parse_ok": True,
        "label": {"__poison__": True}, "raw_output": {"__poison__": True},
        "archive": {
            "neutral_summary": "A person gives a tutorial.",
            "target_groups": {"__poison__": True},
            "mechanism": {"__poison__": True},
            "explicitness": {"__poison__": True},
        },
    }
    projected = reader.project(poison)
    if projected != {"id": "fixture", "split": "train", "parse_ok": True,
                      "neutral_summary": "A person gives a tutorial."}:
        raise RuntimeError("archive allowlist poison fixture failed")
    return sha256_obj(projected)


def acquire_namespace(path, run_id):
    """Create a formal output namespace with O_EXCL. Never overwrites."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    lock = path / ".namespace.lock"
    fd = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(str(run_id) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return path


def exclusive_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def exclusive_write_json(path, obj, pretty=True):
    if pretty:
        text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n"
    else:
        text = canonical_json(obj) + "\n"
    exclusive_write_bytes(path, text.encode("utf-8"))


def exclusive_write_jsonl(path, rows):
    data = "".join(canonical_json(x) + "\n" for x in rows).encode("utf-8")
    exclusive_write_bytes(path, data)


def git_state():
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", ":!slurm/logs/disk_guard.log"],
        cwd=ROOT)
    return head, hashlib.sha256(diff).hexdigest()


def implementation_hash():
    paths = [
        ROOT / "scripts/analysis/sq_common.py",
        ROOT / "scripts/analysis/sq_s0.py",
        ROOT / "scripts/analysis/sq_s1.py",
        ROOT / "scripts/slurm/sq_s0_cpu.sbatch",
        ROOT / "scripts/slurm/sq_s0_gpu.sbatch",
        ROOT / "scripts/slurm/sq_s1_cpu.sbatch",
        ROOT / "scripts/slurm/sq_s1_gpu.sbatch",
    ]
    return sha256_obj([{"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)}
                       for p in paths])


def base_manifest(cfg, run_id, stage, status, inputs=None, outputs=None,
                  gpu_name=None, extra=None):
    import sklearn
    import torch
    try:
        import faiss
        faiss_version = getattr(faiss, "__version__", "unknown")
    except Exception:
        faiss_version = "unavailable"
    head, diff_hash = git_state()
    obj = {
        "schema_version": 1, "run_id": run_id, "stage": stage,
        "status": status, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_head": head, "dirty_diff_sha256": diff_hash,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__, "faiss": faiss_version,
                     "sklearn": sklearn.__version__,
                     "cuda": torch.version.cuda},
        "gpu_name": gpu_name,
        "config_canonical_sha256": cfg["computed_config_sha256"],
        "implementation_sha256": implementation_hash(),
        "input_files": inputs or [], "output_files": outputs or [],
        "only_gold_supervision": "parent_video_binary_label",
        "segment_gold_exists": False, "segment_gold_used": False,
        "new_teacher_call_count": 0, "teacher_cache_read_count": 0,
        "teacher_cache_write_count": 0,
        "archive_forbidden_key_access_count": 0,
        "outer_held_q_read_count": 0, "val_content_read_count": 0,
        "test_content_read_count": 0, "val_test_teacher_artifact_count": 0,
    }
    if extra:
        obj.update(extra)
    obj["payload_sha256"] = sha256_obj(obj)
    return obj


def output_records(paths):
    return [{"path": str(Path(p).relative_to(ROOT)), "sha256": sha256_file(p)}
            for p in paths]


def input_record(path):
    p = Path(path)
    return {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)}


def kish_ess(weights):
    w = np.asarray(weights, dtype=np.float64)
    s = float(w.sum())
    s2 = float(np.square(w).sum())
    return 0.0 if s2 <= 0 else s * s / s2


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, np.float32(1e-12))


def posterior_affinity(q1, q2):
    return np.sqrt(np.maximum(q1, 0.0) * np.maximum(q2, 0.0)).sum(axis=-1)


def exact_ranking(memory_ids, memory_z, memory_labels, query_z, topk=20):
    memory_ids = [str(x) for x in memory_ids]
    mz = normalize_rows(memory_z)
    qz = normalize_rows(np.asarray(query_z, dtype=np.float32).reshape(1, -1))[0]
    sims = (mz @ qz).astype(np.float32)
    order = sorted(range(len(memory_ids)), key=lambda i: (-float(sims[i]), memory_ids[i]))
    rows = []
    for rank, i in enumerate(order[:topk], 1):
        weight = topk + 1 - rank
        signed = float(np.float32(weight) * sims[i] * np.float32(2 * int(memory_labels[i]) - 1))
        rows.append({"rank": rank, "id": memory_ids[i],
                     "label": int(memory_labels[i]), "cosine": float(sims[i]),
                     "weight": weight, "signed_contribution": signed})
    vote = float(np.float32(sum(np.float32(x["signed_contribution"]) for x in rows)))
    pred = int(vote >= 0.0)
    denom = float(np.float32(sum(np.float32(x["weight"] * abs(x["cosine"])) for x in rows)))
    return rows, vote, pred, denom


def exact_ranking_excluding_id(memory_ids, memory_z, memory_labels, query_z,
                               excluded_id, topk=20):
    """Exact ordinary top-k after excluding one train anchor.

    Exclusion occurs before ranks and the fixed k..1 weights are assigned.
    This avoids accidentally inheriting full-bank ranks/weights after self
    removal.
    """
    ids = [str(x) for x in memory_ids]
    keep = [i for i, x in enumerate(ids) if x != str(excluded_id)]
    if len(keep) != len(ids) - 1:
        raise RuntimeError("excluded anchor must occur exactly once")
    z = np.asarray(memory_z)[keep]
    y = np.asarray(memory_labels)[keep]
    kept_ids = [ids[i] for i in keep]
    return exact_ranking(kept_ids, z, y, query_z, topk=topk)


def canonical_full_order(memory_ids, memory_z, query_z):
    ids = [str(x) for x in memory_ids]
    mz = normalize_rows(memory_z)
    qz = normalize_rows(np.asarray(query_z, dtype=np.float32).reshape(1, -1))[0]
    sims = (mz @ qz).astype(np.float32)
    order = sorted(range(len(ids)), key=lambda i: (-float(sims[i]), ids[i]))
    return [{"rank": r + 1, "id": ids[i], "cosine": float(sims[i])}
            for r, i in enumerate(order)]


def add_exposure(rows, query_label):
    out = []
    for x in rows:
        t = 1 if int(x["label"]) == int(query_label) else -1
        exposure = x["weight"] * max(0.0, -t * x["cosine"])
        y = dict(x)
        y["same_label_sign"] = t
        y["exposure"] = float(exposure)
        out.append(y)
    return out


def metrics_from_predictions(labels, preds):
    from sklearn.metrics import accuracy_score, f1_score
    return {"accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro",
                                       zero_division=0))}


def stateless_seed(*parts):
    return int(sha256_text("|".join(str(x) for x in parts))[:16], 16) % (2 ** 63 - 1)


def finite_or_raise(obj, where="object"):
    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, path + "." + str(k))
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, path + "[{}]".format(i))
        elif isinstance(x, (float, np.floating)) and not math.isfinite(float(x)):
            raise RuntimeError("nonfinite {} at {}".format(where, path))
    walk(obj, "$")
    return obj


def make_shuffle_q(ids, labels, q, r, salt="sq-v1-shuffle"):
    """Complete record permutation within class x confidence quartile.

    Singleton strata are deterministically merged lower-first, then upper.  A
    cyclic shift gives a no-fixed-point permutation in every final stratum.
    """
    ids = [str(x) for x in ids]
    labels = np.asarray(labels, dtype=np.int64)
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    quart = np.zeros(len(ids), dtype=np.int64)
    for cls in (0, 1):
        idx = np.flatnonzero(labels == cls)
        order = sorted(idx.tolist(), key=lambda i: (float(r[i]), ids[i]))
        for pos, i in enumerate(order):
            quart[i] = min(3, (4 * pos) // max(1, len(order)))
    groups = {(c, k): np.flatnonzero((labels == c) & (quart == k)).tolist()
              for c in (0, 1) for k in range(4)}
    for c in (0, 1):
        for k in range(4):
            if len(groups[(c, k)]) == 1:
                target = k - 1 if k > 0 else (k + 1 if k < 3 else None)
                if target is not None:
                    groups[(c, target)].extend(groups[(c, k)])
                    groups[(c, k)] = []
    perm = np.arange(len(ids), dtype=np.int64)
    for key, idx in groups.items():
        if not idx:
            continue
        if len(idx) < 2:
            raise RuntimeError("shuffle singleton after deterministic merge: {}".format(key))
        ordered = sorted(idx, key=lambda i: (sha256_text(
            "{}|{}".format(salt, ids[i])), ids[i]))
        for pos, i in enumerate(ordered):
            perm[i] = ordered[(pos + 1) % len(ordered)]
    if np.any(perm == np.arange(len(ids))):
        raise RuntimeError("shuffle contains fixed point")
    if sorted(perm.tolist()) != list(range(len(ids))):
        raise RuntimeError("shuffle is not a permutation")
    return q[perm].copy(), r[perm].copy(), perm


def entropy_rows(q):
    q = np.asarray(q, dtype=np.float64)
    return -(q * np.log(np.maximum(q, 1e-300))).sum(axis=1)


def random_matched_q(ids, q, r, salt="sq-v1-random"):
    """Deterministic logistic-normal control matched to FULL row entropy/r.

    Per-row temperatures match the complete entropy distribution.  Repeated
    column offsets reduce the marginal-mean error without using any label.
    Confidence/missingness are copied exactly from FULL.
    """
    ids = [str(x) for x in ids]
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    logits = np.empty_like(q)
    for i, vid in enumerate(ids):
        rng = np.random.default_rng(stateless_seed(salt, vid))
        logits[i] = rng.normal(size=q.shape[1])
    target_h = entropy_rows(q)
    target_mean = q.mean(axis=0)
    offsets = np.zeros(q.shape[1], dtype=np.float64)

    def softmax(x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

    out = np.empty_like(q)
    for _ in range(20):
        for i in range(len(ids)):
            if r[i] <= 0 or target_h[i] >= math.log(q.shape[1]) - 1e-10:
                out[i] = 1.0 / q.shape[1]
                continue
            lo, hi = 1e-3, 1e3
            base = logits[i] + offsets
            for _b in range(50):
                mid = math.sqrt(lo * hi)
                cand = softmax(base / mid)
                if float(entropy_rows(cand[None, :])[0]) < target_h[i]:
                    lo = mid
                else:
                    hi = mid
            out[i] = softmax(base / math.sqrt(lo * hi))
        err = target_mean - out.mean(axis=0)
        offsets += 0.5 * err / np.maximum(target_mean, 1e-3)
        offsets -= offsets.mean()
    return out, r.copy(), {
        "max_marginal_mean_abs_error": float(np.max(np.abs(out.mean(axis=0) - target_mean))),
        "max_entropy_abs_error": float(np.max(np.abs(entropy_rows(out) - target_h))),
        "r_exact": bool(np.array_equal(r, r.copy())),
    }


def base_cluster_q(z, seed=20260711, n_clusters=6):
    from sklearn.cluster import KMeans
    x = normalize_rows(z)
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1,
                max_iter=100, random_state=seed, algorithm="lloyd").fit(x)
    centers = normalize_rows(km.cluster_centers_)
    # Canonical centroid order makes the soft record reproducible.
    order = sorted(range(n_clusters), key=lambda i: tuple(float(v) for v in centers[i]))
    centers = centers[order]
    logits = (x @ centers.T).astype(np.float64) / 0.1
    logits -= logits.max(axis=1, keepdims=True)
    q = np.exp(logits)
    q /= q.sum(axis=1, keepdims=True)
    h = entropy_rows(q)
    r = np.maximum(0.0, 1.0 - h / math.log(n_clusters))
    return q, r, centers


def sq_sampling_plan(ids, labels, bank_z, q, r, config_sha256, seed, epoch,
                     triplets=64, min_ess=8.0, topk=20, label_only=False):
    """Create the exact detached epoch-start SQ sampling plan."""
    ids = [str(x) for x in ids]
    labels = np.asarray(labels, dtype=np.int64)
    z = normalize_rows(bank_z)
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    plans = {}
    active = 0
    ess_pos, ess_neg, exposed = [], [], []
    for i, vid in enumerate(ids):
        top, _, _, _ = exact_ranking_excluding_id(
            ids, z, labels, z[i], excluded_id=vid, topk=topk)
        idx_by_id = {x: j for j, x in enumerate(ids)}
        pos_idx = np.flatnonzero((labels == labels[i]) & (np.arange(len(ids)) != i))
        if label_only:
            pos_w = np.ones(len(pos_idx), dtype=np.float64)
        else:
            aff = posterior_affinity(np.broadcast_to(q[i], (len(pos_idx), q.shape[1])), q[pos_idx])
            pos_w = r[i] * r[pos_idx] * (1.0 - aff)
        neg_idx, neg_w, neg_e = [], [], []
        for x in top:
            j = idx_by_id[x["id"]]
            if labels[j] == labels[i] or x["cosine"] <= 0:
                continue
            e = x["weight"] * x["cosine"]
            w = 1.0 if label_only else r[i] * r[j] * posterior_affinity(q[i], q[j])
            if w > 0 and e > 0:
                neg_idx.append(j); neg_w.append(float(w)); neg_e.append(float(e))
        neg_idx = np.asarray(neg_idx, dtype=np.int64)
        neg_w = np.asarray(neg_w, dtype=np.float64)
        neg_e = np.asarray(neg_e, dtype=np.float64)
        ep, en = kish_ess(pos_w), kish_ess(neg_w)
        ess_pos.append(ep); ess_neg.append(en); exposed.append(len(neg_idx))
        if ep < min_ess or en < min_ess:
            plans[vid] = {"active": False, "ess_pos": ep, "ess_neg": en,
                          "exposed_negatives": int(len(neg_idx))}
            continue
        pos_p = pos_w / pos_w.sum(); neg_p = neg_w / neg_w.sum()
        draws = []
        for d in range(triplets):
            rng = np.random.default_rng(stateless_seed(
                config_sha256, seed, epoch, vid, d))
            pp = int(rng.choice(len(pos_idx), p=pos_p))
            nn = int(rng.choice(len(neg_idx), p=neg_p))
            draws.append((int(pos_idx[pp]), int(neg_idx[nn]), float(neg_e[nn])))
        plans[vid] = {"active": True, "ess_pos": ep, "ess_neg": en,
                      "exposed_negatives": int(len(neg_idx)), "draws": draws}
        active += 1
    stats = {
        "active_anchors": active, "total_anchors": len(ids),
        "active_fraction": active / max(1, len(ids)),
        "median_positive_ess": float(np.median(ess_pos)),
        "median_negative_ess": float(np.median(ess_neg)),
        "median_exposed_negatives": float(np.median(exposed)),
        "triplet_plan_sha256": sha256_obj(plans),
    }
    return plans, stats


def sq_loss_for_batch(anchor_ids, anchor_z, bank_z, plans,
                      margin=0.1, temperature=0.1):
    """Differentiable anchor-only SQ loss against a detached bank."""
    import torch
    bank = torch.nn.functional.normalize(bank_z.detach(), p=2, dim=1)
    anchors = torch.nn.functional.normalize(anchor_z, p=2, dim=1)
    losses = []
    active = 0
    for row, vid in enumerate(anchor_ids):
        plan = plans[str(vid)]
        if not plan.get("active"):
            continue
        p = torch.as_tensor([x[0] for x in plan["draws"]], device=anchors.device)
        n = torch.as_tensor([x[1] for x in plan["draws"]], device=anchors.device)
        exposure = torch.as_tensor([x[2] for x in plan["draws"]],
                                   dtype=anchors.dtype, device=anchors.device)
        sip = (anchors[row:row + 1] * bank.index_select(0, p)).sum(dim=1)
        sin = (anchors[row:row + 1] * bank.index_select(0, n)).sum(dim=1)
        losses.append((exposure * torch.nn.functional.softplus(
            (sin - sip + margin) / temperature)).mean())
        active += 1
    if not losses:
        return anchors.sum() * 0.0, 0
    return torch.stack(losses).mean(), active
