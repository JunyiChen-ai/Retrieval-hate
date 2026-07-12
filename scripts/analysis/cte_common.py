#!/usr/bin/env python
"""Shared, fail-closed utilities for CTE C0/C1.

The only gold field read by this package is a parent video's binary label.
There is no segment/timestamp/span/localization gold.  C0/C1 make no MLLM,
OCR, or teacher-cache call and never open validation/test content.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict


ROOT = Path("/data/jehc223/RGCL")
DATASETS = ("MHC", "MHC_zh")
ARMS = ("REMOVE", "LABEL_ONLY", "MULTIVIEW", "RANDOM")
SUPERVISION = {
    "only_gold_supervision": "video_level_binary_label",
    "segment_gold_exists": False,
    "segment_gold_used": False,
    "subclip_label_status": "inherited_parent_video_label_not_segment_gold",
    "mllm_call_count": 0,
    "ocr_call_count": 0,
    "teacher_cache_read_count": 0,
    "teacher_cache_write_count": 0,
    "val_endpoint_count": 0,
    "test_endpoint_count": 0,
    "val_test_teacher_artifact_count": 0,
}


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def add_payload_hash(payload):
    out = dict(payload)
    if "payload_sha256" in out:
        raise RuntimeError("payload_sha256 already exists")
    out["payload_sha256"] = sha256_obj(out)
    return out


def verify_payload(payload):
    raw = dict(payload)
    stored = raw.pop("payload_sha256", None)
    if stored is None or stored != sha256_obj(raw):
        raise RuntimeError("payload hash mismatch")


def acquire_lock(path, run_id, stage):
    """Persistently reserve one immutable formal namespace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"formal namespace already reserved: {path}") from exc
    try:
        body = canonical_json({
            "run_id": run_id,
            "stage": stage,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "purpose": "persistent_no_clobber_namespace_lock",
        }) + "\n"
        os.write(fd, body.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive_publish(tmp, path):
    try:
        os.link(tmp, path)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite formal artifact: {path}") from exc
    os.unlink(tmp)
    directory_fd = os.open(str(Path(path).parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_json(path, payload):
    canonical_json(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite formal artifact: {path}")
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _exclusive_publish(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite formal artifact: {path}")
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _exclusive_publish(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite formal artifact: {path}")
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        _exclusive_publish(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def resolve(cfg, key):
    path = Path(cfg["paths"][key])
    return path if path.is_absolute() else Path(cfg["paths"]["root"]) / path


def load_config(path):
    cfg = read_json(path)
    if cfg.get("schema_version") != 1 or cfg.get("config_name") != "cte_v1":
        raise RuntimeError("unsupported CTE config")
    if cfg.get("stage") != "C0_C1_zero_teacher_train_only":
        raise RuntimeError("CTE stage drift")
    if cfg.get("supervision_contract") != SUPERVISION:
        raise RuntimeError("supervision contract drift")
    auth = cfg.get("authorization", {})
    if auth != {"C0": True, "C1": True, "C2": False, "C3": False,
                "C4": False, "teacher_calls_allowed": False,
                "ocr_calls_allowed": False}:
        raise RuntimeError("authorization drift")
    if set(cfg.get("datasets", {})) != set(DATASETS):
        raise RuntimeError("dataset set drift")
    cfg["computed_config_sha256"] = sha256_obj(cfg)
    return cfg


def require_slurm(gpu=None):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("all CTE computation must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")
    if gpu is True and not torch.cuda.is_available():
        raise RuntimeError("GPU phase requires CUDA")
    if gpu is False and torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("CPU phase unexpectedly has a visible GPU")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def implementation_hashes():
    paths = [
        ROOT / "scripts/analysis/cte_common.py",
        ROOT / "scripts/analysis/cte_c0.py",
        ROOT / "scripts/analysis/cte_c1.py",
        ROOT / "scripts/slurm/cte_c0_gpu.sbatch",
        ROOT / "scripts/slurm/cte_c0_cpu.sbatch",
        ROOT / "scripts/slurm/cte_c1_gpu.sbatch",
        ROOT / "scripts/slurm/cte_c1_cpu.sbatch",
    ]
    result = {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}
    return result, sha256_obj(result)


def gpu_metadata():
    if not torch.cuda.is_available():
        return {"cuda_version": str(torch.version.cuda), "gpu_name": None}
    return {"cuda_version": str(torch.version.cuda),
            "gpu_name": torch.cuda.get_device_name(0)}


def output_records(paths):
    return [{"path": str(Path(path).relative_to(ROOT)), "sha256": sha256_file(path)}
            for path in paths]


def input_records(paths):
    return output_records(paths)


def base_manifest(cfg, run_id, stage, status, input_paths, output_paths,
                  fold_ids_sha256, checkpoint_sha256=None, extra=None):
    impl, impl_sha = implementation_hashes()
    meta = gpu_metadata()
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_head": git_head(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_version": meta["cuda_version"],
        "gpu_name": meta["gpu_name"],
        "config_canonical_sha256": cfg["computed_config_sha256"],
        "implementation_sha256": impl_sha,
        "implementation_files": impl,
        "input_files": input_records(input_paths),
        "fold_ids_sha256": fold_ids_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "output_files": output_records(output_paths),
        **SUPERVISION,
    }
    if extra:
        payload.update(extra)
    return add_payload_hash(payload)


def verify_config_freeze(cfg):
    path = resolve(cfg, "artifact_root") / "CONFIG_FREEZE.json"
    freeze = read_json(path)
    verify_payload(freeze)
    if freeze.get("status") != "GO":
        raise RuntimeError("CONFIG_FREEZE is not GO")
    if freeze.get("config_canonical_sha256") != cfg["computed_config_sha256"]:
        raise RuntimeError("config changed after freeze")
    _, current = implementation_hashes()
    if freeze.get("implementation_sha256") != current:
        raise RuntimeError("implementation changed after freeze")
    for item in freeze.get("input_files", []):
        path_i = ROOT / item["path"]
        if sha256_file(path_i) != item["sha256"]:
            raise RuntimeError(f"frozen input changed: {path_i}")
    return path, freeze


def flatten_ids(value):
    if value and isinstance(value[0], (list, tuple)):
        return [str(item) for batch in value for item in batch]
    return [str(item) for item in value]


def train_cache_path(cfg, dataset):
    return resolve(cfg, "clip") / dataset / (
        "train_" + cfg["model"]["clip_model"] + ".pt")


def subclip_cache_path(cfg, dataset):
    return resolve(cfg, "clip") / dataset / (
        f"train_subclipK{cfg['model']['num_subclips']}_" +
        cfg["model"]["clip_model"] + ".pt")


def fold_path(cfg, dataset):
    return resolve(cfg, "ssr_artifacts") / "folds" / f"{dataset}.json"


def load_fold_contract(cfg, dataset):
    if dataset not in DATASETS:
        raise RuntimeError("invalid dataset")
    path = fold_path(cfg, dataset)
    fold = read_json(path)
    if fold.get("dataset") != dataset:
        raise RuntimeError("fold dataset mismatch")
    if fold.get("only_gold_supervision") != "video_level_binary_label":
        raise RuntimeError("fold gold contract mismatch")
    if fold.get("segment_gold_exists") is not False:
        raise RuntimeError("fold incorrectly declares segment gold")
    assertions = fold.get("split_assertions", {})
    if assertions.get("pairwise_disjoint") is not True:
        raise RuntimeError("frozen split disjointness proof missing")
    if any(assertions.get("overlaps", {}).get(name) != []
           for name in ("train_dev", "train_test", "dev_test")):
        raise RuntimeError("frozen split overlap is nonempty")
    records = fold.get("records", [])
    by_id = {}
    for row in records:
        vid, label, outer = str(row.get("id")), row.get("label"), row.get("fold")
        if vid in by_id or label not in (0, 1) or outer not in range(5):
            raise RuntimeError("malformed fold record")
        by_id[vid] = {"id": vid, "label": int(label), "fold": int(outer)}
    if not records:
        raise RuntimeError("empty fold records")
    return path, fold, by_id


def load_train_cache(cfg, dataset):
    path_f, fold, by_id = load_fold_contract(cfg, dataset)
    path = train_cache_path(cfg, dataset)
    frozen = fold["split_assertions"]["clip_cache"]["train"]
    if str(path.relative_to(ROOT)) != frozen["path"] or sha256_file(path) != frozen["sha256"]:
        raise RuntimeError("train cache differs from frozen train-only source")
    cache = torch.load(path, map_location="cpu")
    ids = flatten_ids(cache["ids"])
    img = torch.as_tensor(cache["img_feats"]).float()
    txt = torch.as_tensor(cache["text_feats"]).float()
    labels = torch.as_tensor(cache["labels"]).long().reshape(-1)
    if len(set(ids)) != len(ids) or not (
            len(ids) == img.shape[0] == txt.shape[0] == labels.shape[0]):
        raise RuntimeError("train cache shape or ID uniqueness failure")
    if set(ids) != set(by_id):
        raise RuntimeError("train cache IDs disagree with frozen folds")
    for index, vid in enumerate(ids):
        if int(labels[index]) != by_id[vid]["label"]:
            raise RuntimeError("train cache video label disagrees with fold")
    return {"fold_path": path_f, "fold": fold, "by_id": by_id,
            "cache_path": path, "ids": ids, "img": img, "txt": txt,
            "labels": labels}


def select_rows(bundle, selected_ids):
    selected_ids = list(selected_ids)
    row = {vid: idx for idx, vid in enumerate(bundle["ids"])}
    if len(set(selected_ids)) != len(selected_ids):
        raise RuntimeError("duplicate selected IDs")
    missing = [vid for vid in selected_ids if vid not in row]
    if missing:
        raise RuntimeError(f"selected IDs missing from train cache: {missing[:5]}")
    idx = torch.as_tensor([row[vid] for vid in selected_ids], dtype=torch.long)
    return {"ids": selected_ids,
            "img": bundle["img"].index_select(0, idx),
            "txt": bundle["txt"].index_select(0, idx),
            "labels": bundle["labels"].index_select(0, idx),
            "full_indices": idx}


def load_segment_cache(cfg, dataset, bundle, subset):
    path = subclip_cache_path(cfg, dataset)
    frozen = bundle["fold"]["split_assertions"]["subclip_cache"]
    if str(path.relative_to(ROOT)) != frozen["path"] or sha256_file(path) != frozen["sha256"]:
        raise RuntimeError("subclip cache differs from frozen source")
    raw = torch.load(path, map_location="cpu")
    parent = torch.as_tensor(raw["subclip_parent"]).long().reshape(-1)
    feats = torch.as_tensor(raw["subclip_img_feats"]).float()
    labels = torch.as_tensor(raw["labels"]).long().reshape(-1)
    if not (len(parent) == feats.shape[0] == len(labels)):
        raise RuntimeError("subclip cache shape mismatch")
    if int(parent.min()) < 0 or int(parent.max()) >= len(bundle["ids"]):
        raise RuntimeError("subclip parent index outside train cache")
    inherited = bundle["labels"].index_select(0, parent)
    if not torch.equal(labels, inherited):
        raise RuntimeError("K4 labels are not mechanical parent-video inheritance")
    old_to_new = {int(old): new for new, old in enumerate(subset["full_indices"].tolist())}
    keep = torch.as_tensor([int(item) in old_to_new for item in parent.tolist()],
                           dtype=torch.bool)
    kept_parent = parent[keep]
    new_parent = torch.as_tensor([old_to_new[int(item)] for item in kept_parent],
                                 dtype=torch.long)
    kept_labels = labels[keep]
    if not torch.equal(kept_labels, subset["labels"].index_select(0, new_parent)):
        raise RuntimeError("fold-local K4 inheritance audit failed")
    cache = {
        "subclip_img_feats": feats[keep],
        "subclip_parent": new_parent,
        "labels": kept_labels,
        "parent_id_to_row": {vid: idx for idx, vid in enumerate(subset["ids"])},
        "video_text_feats": subset["txt"],
    }
    audit = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
             "n_subclips": int(keep.sum()), "n_parent_videos": len(subset["ids"]),
             "label_source": "inherited_parent_video_label_not_segment_gold"}
    return path, cache, audit


def train_args(cfg, dataset):
    model = cfg["model"]
    data = cfg["datasets"][dataset]
    return EasyDict({
        "dataset": dataset, "device": "cuda", "batch_size": model["batch_size"],
        "lr": model["learning_rate"], "proj_dim": model["proj_dim"],
        "metric": model["metric"], "loss": model["loss"],
        "triplet_margin": model["triplet_margin"], "norm_feats_loss": False,
        "l2_sqrt": False, "hybrid_loss": model["hybrid_loss"],
        "ce_weight": model["ce_weight"], "pos_weight_value": None,
        "hard_negatives_loss": model["hard_negatives_loss"],
        "no_hard_negatives": model["no_hard_negatives"], "no_hard_positives": 0,
        "no_pseudo_gold_positives": model["no_pseudo_gold_positives"],
        "hard_negatives_multiple": 12, "sparse_dictionary": None,
        "sparse_topk": None, "Faiss_GPU": False, "grad_clip": model["grad_clip"],
        "lambda_seg": data["lambda_seg"], "seg_mode": data["seg_mode"],
        "cf_negs": False, "lambda_aux": 0.0,
    })


def build_model(cfg, dataset, image_dim, text_dim, device="cuda"):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from model.classifier import classifier_hateClipper
    spec = cfg["model"]
    model = classifier_hateClipper(
        image_dim, text_dim, spec["num_layers"], spec["proj_dim"],
        spec["map_dim"], spec["fusion_mode"], dropout=spec["dropout"],
        batch_norm=spec["batch_norm"], args=train_args(cfg, dataset))
    return model.to(device)


def state_dict_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def clone_state(state):
    return {key: value.detach().clone() for key, value in state.items()}


def normalize_checked(value, cfg, name):
    norms = torch.linalg.vector_norm(value, dim=1)
    minimum = float(norms.min().detach().cpu())
    if not bool(torch.isfinite(value).all()) or minimum < cfg["cte"]["minimum_pre_normalization_norm"]:
        raise RuntimeError(f"{name} normalization hard failure: min_norm={minimum}")
    return value / norms.clamp_min(cfg["cte"]["normalization_epsilon"]).unsqueeze(1), minimum


def encode_parts(model, img, txt, cfg, anchor_parts=None, radius=None,
                 modality=None, grad=False):
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        was_training = model.training
        model.eval()
        raw_v = model.img_proj(img)
        raw_l = model.text_proj(txt)
        p_v, min_v = normalize_checked(raw_v, cfg, "visual_projection")
        p_l, min_l = normalize_checked(raw_l, cfg, "language_projection")
        if radius is not None:
            if anchor_parts is None or modality not in ("V", "L"):
                raise RuntimeError("prototype path requires anchor parts and modality")
            if modality == "V":
                mixed = (1.0 - radius) * p_v + radius * anchor_parts["V"]
                p_v, mixed_min = normalize_checked(mixed, cfg, "visual_prototype_mix")
            else:
                mixed = (1.0 - radius) * p_l + radius * anchor_parts["L"]
                p_l, mixed_min = normalize_checked(mixed, cfg, "language_prototype_mix")
        else:
            mixed_min = None
        fused_raw = p_v * p_l
        fused_support, min_f = normalize_checked(fused_raw, cfg, "fused_support")
        endpoint = model.mlp[:-2](fused_raw)
        endpoint, min_z = normalize_checked(endpoint, cfg, "retrieval_embedding")
        pair, min_pair = normalize_checked(torch.cat((p_v, p_l), dim=1), cfg,
                                           "projected_pair")
        if was_training:
            model.train()
    return {"V": p_v, "L": p_l, "g": fused_raw,
            "fused_support": fused_support, "pair": pair, "z": endpoint,
            "minimum_norms": {"V": min_v, "L": min_l, "fused": min_f,
                              "endpoint": min_z, "pair": min_pair,
                              "mixed": mixed_min}}


def encode_dataset(model, subset, cfg, device="cuda", batch_size=256):
    pieces = {key: [] for key in ("V", "L", "g", "fused_support", "pair", "z")}
    minima = []
    for start in range(0, len(subset["ids"]), batch_size):
        out = encode_parts(model, subset["img"][start:start + batch_size].to(device),
                           subset["txt"][start:start + batch_size].to(device), cfg)
        for key in pieces:
            pieces[key].append(out[key].detach())
        minima.extend(value for value in out["minimum_norms"].values()
                      if value is not None)
    result = {key: torch.cat(value, dim=0) for key, value in pieces.items()}
    result["minimum_pre_normalization_norm"] = min(minima)
    return result


def exact_medoid_id(ids, projected):
    order = sorted(range(len(ids)), key=lambda idx: ids[idx].encode("utf-8"))
    x = projected.index_select(0, torch.as_tensor(order, device=projected.device))
    scores = (1.0 - x @ x.T).sum(dim=1)
    winner = int(torch.argmin(scores).item())
    return ids[order[winner]]


def kth_cosine_radius(query, bank, k, exclude_self=False):
    distance = 1.0 - query @ bank.T
    if exclude_self:
        if query.shape[0] != bank.shape[0]:
            raise RuntimeError("self exclusion requires aligned square bank")
        distance.fill_diagonal_(float("inf"))
    values = torch.topk(distance, k=k, dim=1, largest=False, sorted=True).values
    return values[:, k - 1]


def choose_anchors_and_support(model, subset, cfg, fixed=None, device="cuda"):
    encoded = encode_dataset(model, subset, cfg, device=device)
    id_to_row = {vid: idx for idx, vid in enumerate(subset["ids"])}
    if fixed is None:
        anchor_ids = {"V": exact_medoid_id(subset["ids"], encoded["V"]),
                      "L": exact_medoid_id(subset["ids"], encoded["L"])}
    else:
        anchor_ids = dict(fixed["anchor_ids"])
        if any(value not in id_to_row for value in anchor_ids.values()):
            raise RuntimeError("fixed anchor absent from current training subset")
    anchors = {m: encoded[m][id_to_row[anchor_ids[m]]:id_to_row[anchor_ids[m]] + 1]
               for m in ("V", "L")}
    k = int(cfg["cte"]["support_knn"])
    pair_loo = kth_cosine_radius(encoded["pair"], encoded["pair"], k, True)
    fused_loo = kth_cosine_radius(encoded["fused_support"],
                                  encoded["fused_support"], k, True)
    if fixed is None:
        thresholds = {
            "pair": float(np.quantile(pair_loo.detach().cpu().numpy(),
                                      cfg["cte"]["support_quantile"], method="linear")),
            "fused": float(np.quantile(fused_loo.detach().cpu().numpy(),
                                       cfg["cte"]["support_quantile"], method="linear")),
        }
    else:
        thresholds = dict(fixed["thresholds"])
    details, masks = [], {}
    for modality in ("V", "L"):
        for radius in cfg["cte"]["radii"]:
            anchor_parts = {"V": anchors["V"], "L": anchors["L"]}
            path = encode_parts(model, subset["img"].to(device), subset["txt"].to(device), cfg,
                                anchor_parts=anchor_parts, radius=float(radius),
                                modality=modality)
            pair_radius = kth_cosine_radius(path["pair"], encoded["pair"], k, False)
            fused_radius = kth_cosine_radius(path["fused_support"],
                                             encoded["fused_support"], k, False)
            mask = (pair_radius <= thresholds["pair"]) & \
                   (fused_radius <= thresholds["fused"])
            key = f"{modality}:{float(radius):.2f}"
            masks[key] = mask
            details.append({
                "modality": modality, "radius": float(radius),
                "mean_support": float(mask.float().mean().cpu()),
                "pair_radius_max": float(pair_radius.max().cpu()),
                "fused_radius_max": float(fused_radius.max().cpu()),
                "supported_n": int(mask.sum().cpu()), "n": len(subset["ids"]),
            })
    candidates = []
    for a1, a2 in cfg["cte"]["adjacent_radius_pairs"]:
        selected = [masks[f"{m}:{float(a):.2f}"]
                    for m in ("V", "L") for a in (a1, a2)]
        individual = [float(mask.float().mean().cpu()) for mask in selected]
        joint = torch.stack(selected).all(dim=0)
        passed = min(individual) >= cfg["cte"]["support_mean_min"] and \
            float(joint.float().mean().cpu()) >= cfg["cte"]["support_joint_video_min"]
        candidates.append({"a1": float(a1), "a2": float(a2),
                           "individual_support": individual,
                           "joint_video_support": float(joint.float().mean().cpu()),
                           "passed": bool(passed)})
    passing = [item for item in candidates if item["passed"]]
    selected_pair = max(passing, key=lambda item: item["a2"]) if passing else None
    return {"encoded": encoded, "anchor_ids": anchor_ids, "anchors": anchors,
            "thresholds": thresholds, "details": details, "masks": masks,
            "candidates": candidates, "selected_pair": selected_pair,
            "minimum_pre_normalization_norm": encoded["minimum_pre_normalization_norm"]}


def robust_mad(values, mask=None):
    x = values if mask is None else values[mask]
    if x.numel() == 0:
        raise RuntimeError("MAD requested on empty support")
    median = torch.median(x)
    return float(torch.median(torch.abs(x - median)).detach().cpu())


def vectorized_margin(query_z, query_labels, query_ids, bank_z, bank_labels,
                      bank_ids, tau):
    query_z = F.normalize(query_z, dim=1)
    bank_z = F.normalize(bank_z, dim=1)
    sims = query_z @ bank_z.T
    same = query_labels.reshape(-1, 1) == bank_labels.reshape(1, -1)
    other = query_labels.reshape(-1, 1) != bank_labels.reshape(1, -1)
    bank_row = {vid: idx for idx, vid in enumerate(bank_ids)}
    for row, vid in enumerate(query_ids):
        if vid in bank_row:
            same[row, bank_row[vid]] = False
            other[row, bank_row[vid]] = False
    if not bool(same.any(dim=1).all()) or not bool(other.any(dim=1).all()):
        raise RuntimeError("margin query lacks non-self same-label or opposite-label key")
    logits = sims / float(tau)
    same_lse = torch.logsumexp(logits.masked_fill(~same, -torch.inf), dim=1)
    other_lse = torch.logsumexp(logits.masked_fill(~other, -torch.inf), dim=1)
    out = float(tau) * (same_lse - other_lse)
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError("nonfinite full-bank margin")
    return out


def tangent_values(full_z, path_z, labels, ids, bank_z, bank_labels, bank_ids,
                   tau, radius, smin, mad=None, support=None):
    full_margin = vectorized_margin(full_z, labels, ids, bank_z, bank_labels,
                                    bank_ids, tau)
    path_margin = vectorized_margin(path_z, labels, ids, bank_z, bank_labels,
                                    bank_ids, tau)
    slopes = (path_margin - full_margin) / float(radius)
    if mad is None:
        mad = robust_mad(slopes.detach(), support)
    scale = float(radius) * max(float(mad), float(smin)) + 1e-6
    tangent = torch.tanh((path_margin - full_margin) / scale)
    return tangent, float(mad), full_margin, path_margin


def interval_cost(tangent, lower, upper):
    distance = torch.maximum(lower - tangent,
                             torch.maximum(tangent - upper,
                                           torch.zeros_like(tangent)))
    return distance.square() / 4.0


def binary_metrics(labels, predictions):
    labels = [int(value) for value in labels]
    predictions = [int(value) for value in predictions]
    if len(labels) != len(predictions) or not labels:
        raise RuntimeError("metric row mismatch or empty")
    acc = sum(a == b for a, b in zip(labels, predictions)) / len(labels)
    per_class = []
    for cls in (0, 1):
        tp = sum(y == cls and p == cls for y, p in zip(labels, predictions))
        fp = sum(y != cls and p == cls for y, p in zip(labels, predictions))
        fn = sum(y == cls and p != cls for y, p in zip(labels, predictions))
        denom = 2 * tp + fp + fn
        per_class.append((2 * tp / denom) if denom else 0.0)
    return {"accuracy": float(acc), "macro_f1": float(sum(per_class) / 2.0),
            "class_f1": {"0": float(per_class[0]), "1": float(per_class[1])}}


def ordinary_knn(memory_ids, memory_z, memory_labels, query_ids, query_z,
                 query_labels, topk=20):
    memory_z = F.normalize(memory_z, dim=1).detach().cpu()
    query_z = F.normalize(query_z, dim=1).detach().cpu()
    memory_labels = memory_labels.detach().cpu().long()
    rows, predictions = [], []
    weights = list(range(topk, 0, -1))
    for qrow, qid in enumerate(query_ids):
        sims = query_z[qrow] @ memory_z.T
        order = sorted(range(len(memory_ids)),
                       key=lambda idx: (-float(sims[idx]),
                                        memory_ids[idx].encode("utf-8")))
        top = order[:topk]
        if len(top) != topk:
            raise RuntimeError("ordinary kNN memory smaller than topk")
        neighbors = [{"rank": rank + 1, "id": memory_ids[idx],
                      "label": int(memory_labels[idx]),
                      "cosine": float(sims[idx])}
                     for rank, idx in enumerate(top)]
        score = sum(weight * item["cosine"] * (2 * item["label"] - 1)
                    for weight, item in zip(weights, neighbors)) / sum(weights)
        pred = int(score >= 0.0)
        rows.append({"query_id": qid, "query_label": int(query_labels[qrow]),
                     "neighbors": neighbors})
        predictions.append({"query_id": qid,
                            "query_label": int(query_labels[qrow]),
                            "prediction": pred, "arithmetic_cosine_score": float(score)})
    metrics = binary_metrics([row["query_label"] for row in predictions],
                             [row["prediction"] for row in predictions])
    return rows, predictions, metrics


def validate_manifest_common(manifest, expected_run_id, expected_stage=None,
                             cfg=None, expected_fold_sha256=None,
                             expected_checkpoint_sha256=None,
                             required_output_paths=None,
                             required_input_paths=None):
    verify_payload(manifest)
    if manifest.get("run_id") != expected_run_id:
        raise RuntimeError("manifest run ID mismatch")
    if expected_stage is not None and manifest.get("stage") != expected_stage:
        raise RuntimeError("manifest stage mismatch")
    if manifest.get("status") != "COMPLETED":
        raise RuntimeError("producer manifest is not COMPLETED")
    if cfg is not None:
        if manifest.get("config_canonical_sha256") != cfg["computed_config_sha256"]:
            raise RuntimeError("manifest config provenance mismatch")
        _, current_impl = implementation_hashes()
        if manifest.get("implementation_sha256") != current_impl:
            raise RuntimeError("manifest implementation provenance mismatch")
    if expected_fold_sha256 is not None and \
            manifest.get("fold_ids_sha256") != expected_fold_sha256:
        raise RuntimeError("manifest fold-ID provenance mismatch")
    if expected_checkpoint_sha256 is not None and \
            manifest.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise RuntimeError("manifest checkpoint provenance mismatch")
    for key, value in SUPERVISION.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"manifest supervision/zero-call mismatch: {key}")
    output_names = {item.get("path") for item in manifest.get("output_files", [])}
    input_names = {item.get("path") for item in manifest.get("input_files", [])}
    if len(output_names) != len(manifest.get("output_files", [])) or \
            len(input_names) != len(manifest.get("input_files", [])):
        raise RuntimeError("manifest contains duplicate/malformed file records")
    if required_output_paths is not None:
        expected = {str(Path(path).relative_to(ROOT)) for path in required_output_paths}
        if output_names != expected:
            raise RuntimeError("manifest required output membership mismatch")
    if required_input_paths is not None:
        required = {str(Path(path).relative_to(ROOT)) for path in required_input_paths}
        if not required.issubset(input_names):
            raise RuntimeError("manifest required input membership mismatch")
    for item in manifest.get("input_files", []) + manifest.get("output_files", []):
        path = ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"manifest file hash mismatch: {path}")
    return True


def deterministic_derangement(n, seed):
    if n < 2:
        raise RuntimeError("cannot derange fewer than two records")
    rng = np.random.default_rng(int(seed))
    base = np.arange(n)
    for _ in range(10000):
        candidate = rng.permutation(n)
        if bool(np.all(candidate != base)):
            return candidate.tolist()
    raise RuntimeError("deterministic derangement construction failed")


def jaccard_churn(left, right):
    a, b = set(left), set(right)
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def percentile_linear(values, q):
    return float(np.quantile(np.asarray(values, dtype=np.float64), q,
                             method="linear"))
