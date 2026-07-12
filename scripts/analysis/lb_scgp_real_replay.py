#!/usr/bin/env python
"""Independent GPU replay for LB-SCGP real fit and rollback.

This stage intentionally does not import lb_scgp_g0.  It reloads the sealed
train-only artifacts, the actual RA-HMD/RGCL checkpoint and the target emitted
by the producer, then independently replays batch order, target-fit steps,
realized bank generation, and rollback-vs-direct REMOVE.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path("/data/jehc223/RGCL")


def cjson(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def hobj(obj):
    return hashlib.sha256(cjson(obj).encode("utf-8")).hexdigest()


def hfile(path):
    path, _ = canonical_root_path(path)
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_root_path(path, must_be_under_root=True):
    root = ROOT.resolve()
    raw = Path(path)
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        if must_be_under_root:
            raise RuntimeError(
                "path escapes LB-SCGP ROOT: {} -> {}".format(path, resolved)
            ) from exc
        return resolved, Path(str(resolved))
    return resolved, rel


def payload_hash(obj):
    copy_obj = dict(obj)
    copy_obj.pop("payload_sha256", None)
    return hobj(copy_obj)


def publish_exclusive(path, obj):
    path = canonical_root_path(path)[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".publish.lock")
    lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    tmp = None
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        os.fsync(lock_fd)
        os.close(lock_fd)
        lock_fd = -1
        if path.exists():
            raise FileExistsError("refusing overwrite {}".format(path))
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(cjson(obj) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
        os.unlink(tmp)
        tmp = None
        dfd = os.open(str(path.parent), os.O_RDONLY)
        os.fsync(dfd)
        os.close(dfd)
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def resolve(cfg, key):
    return canonical_root_path(cfg["paths"][key])[0]


def read_json(path):
    fs_path, _ = canonical_root_path(path)
    with open(fs_path, encoding="utf-8") as handle:
        return json.load(handle)


def _allowed_npz(path, names):
    path = canonical_root_path(path)[0]
    out = {}
    with zipfile.ZipFile(path, "r") as archive:
        members = {Path(name).stem: name for name in archive.namelist()
                   if name.endswith(".npy")}
        missing = set(names) - set(members)
        if missing:
            raise RuntimeError("missing bank members {}".format(sorted(missing)))
        forbidden = {"query_z", "query_labels"}
        for name in names:
            if name in forbidden:
                raise RuntimeError("forbidden member requested")
            with archive.open(members[name], "r") as handle:
                payload = handle.read()
            out[name] = np.load(io.BytesIO(payload), allow_pickle=False)
    return out


def _independent_factor(gram):
    gram = 0.5 * (gram + gram.T)
    eigval, eigvec = np.linalg.eigh(gram)
    if float(eigval.min()) < -1e-7:
        raise ValueError("negative eigenvalue")
    order = np.argsort(-eigval, kind="mergesort")
    eigval = eigval[order]
    eigvec = eigvec[:, order]
    basis = np.zeros_like(eigvec)
    start = 0
    while start < len(eigval):
        end = start + 1
        scale = max(1.0, abs(float(eigval[start])))
        while end < len(eigval) and abs(float(eigval[end] - eigval[start])) <= 1e-10 * scale:
            end += 1
        projector = eigvec[:, start:end] @ eigvec[:, start:end].T
        vectors = []
        for axis in range(len(eigval)):
            v = projector[:, axis].copy()
            for q in vectors:
                v -= q * float(q @ v)
            norm = float(np.linalg.norm(v))
            if norm > 1e-10:
                v /= norm
                pivot = int(np.argmax(np.abs(v)))
                if v[pivot] < 0:
                    v = -v
                vectors.append(v)
            if len(vectors) == end - start:
                break
        basis[:, start:end] = np.column_stack(vectors)
        start = end
    return basis * np.sqrt(np.maximum(eigval, 0.0))[None, :]


def _json_state(obj):
    import torch
    if torch.is_tensor(obj):
        return {"dtype": str(obj.dtype), "shape": list(obj.shape),
                "value": obj.detach().cpu().numpy().tolist()}
    if isinstance(obj, np.ndarray):
        return {"dtype": str(obj.dtype), "shape": list(obj.shape),
                "value": obj.tolist()}
    if isinstance(obj, dict):
        return {str(k): _json_state(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple)):
        return [_json_state(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _state_digest(model, optimizer, scheduler, scaler, cursor, torch):
    rows = []
    for key, value in sorted(model.state_dict().items()):
        rows.append((key, hobj(value.detach().cpu().numpy().tolist())))
    rows += [("optimizer", hobj(_json_state(optimizer.state_dict()))),
             ("scheduler", hobj(_json_state(scheduler.state_dict()))),
             ("scaler", hobj(_json_state(scaler.state_dict()))),
             ("torch_rng", hobj(torch.get_rng_state().tolist())),
             ("cuda_rng", hobj([state.cpu().tolist() for state in torch.cuda.get_rng_state_all()])),
             ("numpy_rng", hobj(_json_state(np.random.get_state()))),
             ("python_rng", hobj(_json_state(random.getstate()))),
             ("cursor", hobj(cursor))]
    return hobj(rows)


def _repo_args():
    from easydict import EasyDict
    return EasyDict({"dataset": "MHC_zh", "device": "cuda", "batch_size": 64,
        "lr": 0.0001, "proj_dim": 1024, "metric": "cos", "loss": "triplet",
        "triplet_margin": 0.1, "norm_feats_loss": False, "l2_sqrt": False,
        "hybrid_loss": True, "ce_weight": 0.5, "pos_weight_value": None,
        "hard_negatives_loss": True, "no_hard_negatives": 1,
        "no_hard_positives": 0, "no_pseudo_gold_positives": 1,
        "hard_negatives_multiple": 12, "sparse_dictionary": None,
        "sparse_topk": None, "Faiss_GPU": False, "grad_clip": 0.1,
        "lambda_seg": 0.0, "seg_mode": "disabled", "cf_negs": False,
        "lambda_aux": 0.0})


def _assert_no_segment_objective(args, segment):
    if float(getattr(args, "lambda_seg", 0.0)) != 0.0:
        raise RuntimeError("LB-SCGP replay forbids segment loss; lambda_seg must be 0")
    if segment is not None:
        raise RuntimeError("LB-SCGP replay forbids segment cache/objective")


def _load_train_only(cfg, ids, labels, expected_feature_sha):
    import torch
    if not expected_feature_sha:
        raise RuntimeError("missing authoritative feature-cache hash for replay")
    feature_path = resolve(cfg, "outer_train_feature_cache")
    if hfile(feature_path) != expected_feature_sha:
        raise RuntimeError("feature-cache hash mismatch for replay")
    feature = torch.load(feature_path, map_location="cpu", weights_only=True)
    if set(feature) != {"ids", "img_feats", "text_feats", "labels"}:
        raise RuntimeError("feature cache schema drift")
    feature_ids = [str(x) for x in feature["ids"]]
    if feature_ids != list(ids):
        raise RuntimeError("feature ID order mismatch")
    feature_labels = torch.as_tensor(feature["labels"]).reshape(-1).long()
    if not torch.equal(feature_labels, torch.as_tensor(labels, dtype=torch.long)):
        raise RuntimeError("feature labels mismatch memory_labels")
    memory = [feature_ids, torch.as_tensor(feature["img_feats"]).float(),
              torch.as_tensor(feature["text_feats"]).float(), feature_labels]
    segment = None
    return memory, segment


def _make_model(memory, cfg):
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from model.classifier import classifier_hateClipper
    args = _repo_args()
    model = classifier_hateClipper(
        int(memory[1].shape[1]), int(memory[2].shape[1]), 3, 1024, 1024,
        "align", dropout=[0.2, 0.4, 0.1], batch_norm=False, args=args).cuda()
    state = torch.load(resolve(cfg, "checkpoint"), map_location="cuda", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model, args


def _loader(memory):
    sys.path.insert(0, str(ROOT / "src"))
    from data_loader.rac_dataloader import CLIP2Dataloader
    return CLIP2Dataloader(memory, memory, batch_size=64,
                           return_dataset=True, normalize=False)


def _project(model, memory):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(memory[0]), 128):
            _, z = model(memory[1][start:start+128].cuda(),
                         memory[2][start:start+128].cuda(),
                         return_embed=True)
            out.append(torch.nn.functional.normalize(z, dim=1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def _run_epoch(model, args, memory, segment, target_aligned=None,
               optimizer=None, scheduler=None, scaler=None, cursor=None):
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from model.loss import compute_loss
    _assert_no_segment_objective(args, segment)
    (train_dl, _), (train_set, _) = _loader(memory)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    if cursor is None:
        cursor = {"sampler_cursor": 0, "epoch_cursor": 0}
    batch_orders = []
    fit_steps = 0
    target_rows_seen = []
    train_feats = train_labels = None
    model.train()
    for step, batch in enumerate(train_dl):
        batch_orders.extend(str(v) for v in batch["ids"])
        result = compute_loss(batch, train_dl, model, args, train_set=train_set,
                              sparse_retrieval_dictionary=None,
                              train_feats=train_feats, train_labels=train_labels,
                              segment_cache=None, aux_pack=None, cf_pack=None)
        loss = result[0]
        train_feats = result[5].detach() if torch.is_tensor(result[5]) else result[5]
        train_labels = result[6].detach() if torch.is_tensor(result[6]) else result[6]
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        cursor["sampler_cursor"] += len(batch["ids"])
        if target_aligned is not None and step % 4 == 0:
            model.eval()
            optimizer.zero_grad(set_to_none=True)
            seen = 0
            for target_start in range(0, len(memory[0]), 64):
                target_stop = min(target_start + 64, len(memory[0]))
                _, pred = model(memory[1][target_start:target_stop].cuda(),
                                memory[2][target_start:target_stop].cuda(),
                                return_embed=True)
                pred = torch.nn.functional.normalize(pred, dim=1)
                target = torch.as_tensor(target_aligned[target_start:target_stop],
                                         dtype=torch.float32, device="cuda")
                target_loss = torch.square(pred - target).sum() / (len(memory[0]) * target.shape[1])
                scaler.scale(target_loss).backward()
                seen += target_stop - target_start
            scaler.step(optimizer)
            scaler.update()
            target_rows_seen.append(seen)
            fit_steps += 1
            model.train()
    scheduler.step()
    cursor["epoch_cursor"] += 1
    return {"optimizer": optimizer, "scheduler": scheduler, "scaler": scaler,
            "batch_order_sha256": hobj(batch_orders), "batch_rows": len(batch_orders),
            "fit_steps": fit_steps, "target_rows_seen_per_step": target_rows_seen,
            "epoch_cursor": cursor["epoch_cursor"],
            "sampler_cursor": cursor["sampler_cursor"]}


def _rollback(memory, segment, cfg):
    import torch
    seed = int(cfg["real_fit"]["seed"]) + 1991
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model, args = _make_model(memory, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    cursor = {"sampler_cursor": 0, "epoch_cursor": 0}
    snap = {"model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "scaler": copy.deepcopy(scaler.state_dict()),
            "torch": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
            "numpy": copy.deepcopy(np.random.get_state()),
            "python": random.getstate(),
            "cursor": copy.deepcopy(cursor)}
    optimizer.zero_grad(set_to_none=True)
    _, z = model(memory[1][:64].cuda(), memory[2][:64].cuda(), return_embed=True)
    failed_loss = torch.square(torch.nn.functional.normalize(z, dim=1) + 1.0).mean()
    scaler.scale(failed_loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
    cursor["sampler_cursor"] += 64; cursor["epoch_cursor"] += 1
    model.load_state_dict(snap["model"]); optimizer.load_state_dict(snap["optimizer"])
    scheduler.load_state_dict(snap["scheduler"]); scaler.load_state_dict(snap["scaler"])
    torch.set_rng_state(snap["torch"]); torch.cuda.set_rng_state_all(snap["cuda"])
    np.random.set_state(snap["numpy"]); random.setstate(snap["python"])
    cursor = copy.deepcopy(snap["cursor"])
    replay = _run_epoch(model, args, memory, segment, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, cursor=cursor)
    replay_cursor = {"sampler_cursor": replay["sampler_cursor"],
                     "epoch_cursor": replay["epoch_cursor"]}
    replay_hash = _state_digest(model, replay["optimizer"], replay["scheduler"],
                                replay["scaler"], replay_cursor, torch)
    direct, args2 = _make_model(memory, cfg)
    direct.load_state_dict(snap["model"])
    direct_optimizer = torch.optim.AdamW(direct.parameters(), lr=args2.lr)
    direct_scheduler = torch.optim.lr_scheduler.StepLR(direct_optimizer, step_size=1, gamma=0.9)
    direct_scaler = torch.amp.GradScaler("cuda", enabled=True)
    direct_optimizer.load_state_dict(snap["optimizer"])
    direct_scheduler.load_state_dict(snap["scheduler"])
    direct_scaler.load_state_dict(snap["scaler"])
    torch.set_rng_state(snap["torch"]); torch.cuda.set_rng_state_all(snap["cuda"])
    np.random.set_state(snap["numpy"]); random.setstate(snap["python"])
    direct_cursor = copy.deepcopy(snap["cursor"])
    direct_result = _run_epoch(direct, args2, memory, segment,
                               optimizer=direct_optimizer,
                               scheduler=direct_scheduler,
                               scaler=direct_scaler, cursor=direct_cursor)
    direct_hash = _state_digest(direct, direct_result["optimizer"],
                                direct_result["scheduler"], direct_result["scaler"],
                                {"sampler_cursor": direct_result["sampler_cursor"],
                                 "epoch_cursor": direct_result["epoch_cursor"]},
                                torch)
    return {"rollback_replay_sha256": replay_hash,
            "direct_remove_sha256": direct_hash,
            "rollback_hash_identical": replay_hash == direct_hash,
            "rollback_batch_order_sha256": replay["batch_order_sha256"],
            "direct_batch_order_sha256": direct_result["batch_order_sha256"]}


def task_replay(cfg, args):
    import torch
    expected_run_id = cfg.get("lineage", {}).get("run_ids", {}).get(
        "replay", "LBSCGP-G0-REAL-REPLAY-MHC_zh-F4-S0-v1")
    if args.run_id != expected_run_id:
        raise RuntimeError("wrong real replay run ID")
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("real replay must run under SLURM HateVideo")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("real replay requires exactly one visible CUDA device")
    base = resolve(cfg, "artifacts") / "g0/real/MHC_zh/fold4"
    out = base / "fit_replay.json"
    if out.exists() or out.with_name(out.name + ".publish.lock").exists():
        raise RuntimeError("refusing overwrite {}".format(out))
    numerics = read_json(base / "numerics.json")
    fit = read_json(base / "fit_rollback.json")
    factor = read_json(base / "factor.json")
    bank = _allowed_npz(resolve(cfg, "bank"), ["memory_ids", "memory_z", "memory_labels", "query_ids"])
    ids = [str(x) for x in bank["memory_ids"].tolist()]
    held = [str(x) for x in bank["query_ids"].tolist()]
    labels = np.asarray(bank["memory_labels"], dtype=np.int64)
    z0 = np.asarray(bank["memory_z"], dtype=np.float64)
    z0 /= np.linalg.norm(z0, axis=1, keepdims=True)
    if set(ids) & set(held):
        raise RuntimeError("memory/query sentinel overlap")
    contract = fit.get("selective_cache_contract", {})
    if contract.get("segment_cache_opened") is not False or \
            contract.get("segment_objective_allowed") is not False:
        raise RuntimeError("producer fit contract exposes segment cache/objective")
    memory, segment = _load_train_only(cfg, ids, labels, contract.get("feature_cache_sha256"))
    target = np.asarray(numerics["target_gram"], dtype=np.float64)
    independent_factor = _independent_factor(target)
    u, _, vt = np.linalg.svd(independent_factor.T @ z0, full_matrices=False)
    aligned = independent_factor @ (u @ vt)
    if hobj(aligned.tolist()) != factor.get("aligned_factor_sha256"):
        raise RuntimeError("independent aligned target hash mismatch")
    seed = int(cfg["real_fit"]["seed"])
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model, repo_args = _make_model(memory, cfg)
    result = _run_epoch(model, repo_args, memory, segment, target_aligned=aligned)
    torch.cuda.synchronize()
    realized = _project(model, memory)
    realized_sha = hobj(realized.tolist())
    rollback = _rollback(memory, segment, cfg)
    expected_fit_steps = sum(1 for step in range(math.ceil(len(ids) / cfg["real_fit"]["batch_size"]))
                             if step % cfg["real_fit"]["scheduled_divisor"] == 0)
    gates = {
        "batch_order": result["batch_order_sha256"] == fit.get("batch_order_sha256"),
        "batch_rows": int(result["batch_rows"]) == int(fit.get("full_epoch_batch_rows", -1)) == len(ids),
        "fit_steps": int(result["fit_steps"]) == int(fit.get("fit_steps", -1)) == expected_fit_steps,
        "target_rows_seen": result["target_rows_seen_per_step"] == fit.get("target_rows_seen_per_step"),
        "realized_bank": realized_sha == fit.get("realized_bank_sha256"),
        "rollback_hash": rollback["rollback_hash_identical"] is True and
            rollback["rollback_replay_sha256"] == fit.get("rollback_replay_sha256") and
            rollback["direct_remove_sha256"] == fit.get("direct_remove_sha256"),
        "rollback_batch_order": rollback["rollback_batch_order_sha256"] == fit.get("rollback_batch_order_sha256") and
            rollback["direct_batch_order_sha256"] == fit.get("direct_batch_order_sha256"),
        "no_segment_objective": segment is None and float(getattr(repo_args, "lambda_seg", 0.0)) == 0.0,
    }
    props = torch.cuda.get_device_properties(0)
    artifact = {"schema_version": 1,
                "run_id": args.run_id,
                "stage": "G0_REAL_FIT_REPLAY",
                "status": "PASS" if all(gates.values()) else "FAIL",
                "producer_fit_sha256": hfile(base / "fit_rollback.json"),
                "producer_numerics_sha256": hfile(base / "numerics.json"),
                "producer_factor_sha256": hfile(base / "factor.json"),
                "gates": gates,
                "batch_order_sha256": result["batch_order_sha256"],
                "fit_steps": result["fit_steps"],
                "target_rows_seen_per_step": result["target_rows_seen_per_step"],
                "realized_bank_sha256": realized_sha,
                "segment_cache_used": False,
                "lambda_seg": float(getattr(repo_args, "lambda_seg", 0.0)),
                "rollback": rollback,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_cuda_device_count": int(torch.cuda.device_count()),
                "gpu_name": props.name,
                "gpu_uuid": getattr(props, "uuid", None),
                "teacher_mllm_ocr_calls": 0,
                "outer_held_label_read_count": 0,
                "outer_held_content_read_count": 0}
    artifact["payload_sha256"] = payload_hash(artifact)
    publish_exclusive(out, artifact)
    print(cjson({"status": artifact["status"], "run_id": args.run_id}))
    if artifact["status"] != "PASS":
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=["replay"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)
    task_replay(cfg, args)


if __name__ == "__main__":
    main()
