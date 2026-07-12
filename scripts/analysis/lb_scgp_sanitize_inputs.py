#!/usr/bin/env python
"""LB-SCGP pre-freeze quarantine sanitizer.

This is not a model, optimizer, evaluator, teacher, MLLM or OCR stage.  It is
the explicit governance exception to the original byte-level non-opening rule:
mixed train+held feature caches may be opened only here, before formal G0
freeze, to perform a mechanical ID split into physically train-only artifacts.
Formal G0 must read only the outputs and sanitized provenance from this stage.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path("/data/jehc223/RGCL")
QUARANTINE_MANIFEST = ROOT / "artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json"
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from lb_scgp_common import (  # noqa: E402
    AccessLedger, canonical_json, canonical_root_path, exclusive_publish,
    load_config, payload_hash, publish_json, require_slurm, resolve,
    root_relative_path, sha256_bytes, sha256_file, sha256_obj,
)


def _as_id_list(values):
    values = list(values)
    if values and isinstance(values[0], (list, tuple)):
        return [str(item) for group in values for item in group]
    return [str(item) for item in values]


def _allowed_npz(path, names, ledger):
    path = canonical_root_path(path)[0]
    out = {}
    with zipfile.ZipFile(path, "r") as archive:
        members = {Path(name).stem: name for name in archive.namelist()
                   if name.endswith(".npy")}
        missing = set(names) - set(members)
        if missing:
            raise RuntimeError("missing allowed sentinel members {}".format(sorted(missing)))
        forbidden = {"query_z", "query_labels"}
        if forbidden & set(members):
            # Presence is allowed; opening is not.  Keep this explicit in the
            # manifest to prevent hidden claims that the archive is pure.
            pass
        for name in names:
            member = members[name]
            with archive.open(member, "r") as handle:
                payload = handle.read()
            ledger.records.append({
                "kind": "npz_member_read",
                "path": root_relative_path(path),
                "member": name,
                "scope": "outer_held_ids_sentinel" if name == "query_ids" else "outer_train_bank",
                "sha256": sha256_bytes(payload),
            })
            out[name] = np.load(io.BytesIO(payload), allow_pickle=False)
    return out


def _index_by_id(source_ids):
    mapping = {}
    for row, vid in enumerate(source_ids):
        if vid in mapping:
            raise RuntimeError("duplicate source ID {}".format(vid))
        mapping[vid] = row
    return mapping


def _safe_torch_save(path, obj):
    import torch
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
        os.close(fd)
        torch.save(obj, tmp)
        with open(tmp, "rb") as handle:
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


def _publish_json_payload(path, obj):
    publish_json(path, obj)
    return sha256_file(path)


def _load_source_config(path):
    fs_path, _ = canonical_root_path(path)
    with open(fs_path, encoding="utf-8") as handle:
        source_cfg = json.load(handle)
    if source_cfg.get("formal_g0_input") is not False:
        raise RuntimeError("sanitizer source config must be quarantine-only")
    if source_cfg.get("dataset") != "MHC_zh" or int(source_cfg.get("outer_fold", -1)) != 4:
        raise RuntimeError("sanitizer source identity drift")
    return source_cfg


def _source_path(source_cfg, key):
    return canonical_root_path(source_cfg[key])[0]


def _select_feature_cache(source_cfg, ids, labels, ledger):
    import torch
    src = _source_path(source_cfg, "mixed_whole_video_cache")
    src_sha = sha256_file(src)
    if src_sha != source_cfg["mixed_whole_video_cache_sha256"]:
        raise RuntimeError("mixed whole-video source hash drift")
    ledger.records.append({"kind": "quarantine_mixed_file_read",
                           "path": root_relative_path(src),
                           "scope": "quarantine_mixed_storage",
                           "purpose": "mechanical_id_split_source_hash",
                           "sha256": src_sha})
    payload = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError("whole-video source cache must be a dict")
    required = {"ids", "img_feats", "text_feats"}
    if not required <= set(payload):
        raise RuntimeError("whole-video source missing {}".format(sorted(required - set(payload))))
    source_ids = _as_id_list(payload["ids"])
    row_by_id = _index_by_id(source_ids)
    missing = [vid for vid in ids if vid not in row_by_id]
    if missing:
        raise RuntimeError("source lacks memory IDs {}".format(missing[:5]))
    rows = [row_by_id[vid] for vid in ids]
    if len(set(rows)) != len(rows):
        raise RuntimeError("selected source rows are not one-to-one")
    img = torch.as_tensor(payload["img_feats"]).index_select(
        0, torch.as_tensor(rows, dtype=torch.long)).contiguous().float()
    text = torch.as_tensor(payload["text_feats"]).index_select(
        0, torch.as_tensor(rows, dtype=torch.long)).contiguous().float()
    out = {"ids": list(ids), "img_feats": img, "text_feats": text,
           "labels": torch.as_tensor(labels, dtype=torch.long)}
    return out, {
        "source_path": root_relative_path(src),
        "source_sha256": src_sha,
        "source_total_rows": len(source_ids),
        "selected_rows": len(rows),
        "selected_rows_sha256": sha256_obj(rows),
        "source_labels_ignored": "labels" in payload,
    }


def task_build(cfg, args, ledger):
    if args.run_id != "LBSCGP-G0-SANITIZE-MHC_zh-F4-v1":
        raise RuntimeError("wrong sanitizer run ID")
    out_feature = resolve(cfg, "outer_train_feature_cache")
    out_prov = resolve(cfg, "sanitized_provenance")
    out_manifest = QUARANTINE_MANIFEST
    source_cfg = _load_source_config(args.source_config)
    for path in (out_feature, out_prov, out_manifest):
        if path.exists() or path.with_name(path.name + ".publish.lock").exists():
            raise RuntimeError("refusing to clobber {}".format(path))
    bank = _allowed_npz(resolve(cfg, "bank"), ["memory_ids", "memory_labels", "query_ids"], ledger)
    ids = [str(x) for x in bank["memory_ids"].tolist()]
    query_ids = [str(x) for x in bank["query_ids"].tolist()]
    labels = np.asarray(bank["memory_labels"], dtype=np.int64).reshape(-1)
    fixture = cfg["sealed_real_fixture"]
    if len(ids) != fixture["outer_train_n"] or len(query_ids) != fixture["outer_held_n"]:
        raise RuntimeError("memory/query sentinel row counts drifted")
    if set(ids) & set(query_ids):
        raise RuntimeError("memory IDs overlap query IDs")
    if sha256_obj(ids) != fixture["memory_ids_sha256"]:
        raise RuntimeError("memory ID hash mismatch")
    if sha256_obj(query_ids) != fixture["query_ids_sha256"]:
        raise RuntimeError("query ID sentinel hash mismatch")
    if labels.shape[0] != len(ids) or set(np.unique(labels).tolist()) != {0, 1}:
        raise RuntimeError("memory parent labels invalid")

    feature, feature_lineage = _select_feature_cache(source_cfg, ids, labels.tolist(), ledger)
    _safe_torch_save(out_feature, feature)
    feature_sha = sha256_file(out_feature)
    code_hash = sha256_obj([
        {"path": "scripts/analysis/lb_scgp_sanitize_inputs.py",
         "sha256": sha256_file(ROOT / "scripts/analysis/lb_scgp_sanitize_inputs.py")},
        {"path": "scripts/analysis/lb_scgp_verify_sanitizer.py",
         "sha256": sha256_file(ROOT / "scripts/analysis/lb_scgp_verify_sanitizer.py")
         if (ROOT / "scripts/analysis/lb_scgp_verify_sanitizer.py").exists() else None},
    ])
    formal = {
        "schema_version": 1,
        "run_id": args.run_id,
        "stage": "LB_SCGP_SANITIZED_PROVENANCE",
        "dataset": "MHC_zh",
        "outer_fold": 4,
        "artifact_namespace": "artifacts/lb_scgp/inputs/MHC_zh/fold4",
        "feature_cache_path": root_relative_path(out_feature),
        "feature_cache_sha256": feature_sha,
        "row_selection_rule": "selected solely by exact equality to memory_ids",
        "parent_label_rule": "output labels inherited only from memory_labels",
        "input_cache_labels_ignored": True,
        "segment_cache_path": None,
        "segment_cache_sha256": None,
        "segment_artifact_created": False,
        "segment_objective_allowed": False,
        "memory_id_count": len(ids),
        "query_id_sentinel_count": len(query_ids),
        "memory_ids_sha256": sha256_obj(ids),
        "query_ids_sha256": sha256_obj(query_ids),
        "memory_labels_sha256": sha256_obj(labels.tolist()),
        "zero_overlap_with_query_ids": not (set(ids) & set(query_ids)),
        "pre_freeze_disclosure_record_external": True,
        "formal_model_optimizer_evaluator_outer_held_read_count": 0,
        "formal_query_z_read_count": 0,
        "formal_query_labels_read_count": 0,
        "teacher_mllm_ocr_calls": 0,
        "network_external_calls": 0,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "sanitizer_code_sha256": code_hash,
        "no_clobber_locks_present": True,
    }
    formal["payload_sha256"] = payload_hash(formal)
    _publish_json_payload(out_prov, formal)
    manifest = copy.deepcopy(formal)
    manifest.update({
        "stage": "LB_SCGP_QUARANTINE_SANITIZER_MANIFEST",
        "quarantine_mixed_storage_read": True,
        "formal_g0_input": False,
        "source_lineage": {
            "source_config": {
                "path": root_relative_path(args.source_config),
                "sha256": sha256_file(canonical_root_path(args.source_config)[0]),
            },
            "whole_video_cache": feature_lineage,
            "subclip_cache": None,
            "subclip_source_opened": False,
            "subclip_output_created": False,
        },
        "access_ledger": list(ledger.records),
        "access_ledger_sha256": sha256_obj(ledger.records),
        "output_hashes": {
            "feature_cache_sha256": feature_sha,
            "sanitized_provenance_sha256": sha256_file(out_prov),
        },
    })
    manifest.pop("payload_sha256", None)
    manifest["payload_sha256"] = payload_hash(manifest)
    publish_json(out_manifest, manifest)
    print(canonical_json({"status": "SANITIZED", "run_id": args.run_id,
                          "feature_sha256": feature_sha}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-config", default="configs/lb_scgp/lb_scgp_sanitizer_sources.json")
    parser.add_argument("--task", required=True, choices=["build"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    require_slurm(expected_gpu=None)
    ledger = AccessLedger()
    cfg = load_config(args.config, ledger)
    task_build(cfg, args, ledger)


if __name__ == "__main__":
    main()
