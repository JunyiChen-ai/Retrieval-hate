#!/usr/bin/env python
"""Independent verifier for the LB-SCGP quarantine sanitizer.

The verifier checks the physical train-only artifacts and writes a formal
decision record that intentionally excludes mixed-cache paths and mixed-cache
hashes.  Source lineage remains confined to the quarantined sanitizer manifest.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path("/data/jehc223/RGCL")
QUARANTINE_MANIFEST = ROOT / "artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json"
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from lb_scgp_common import (  # noqa: E402
    assert_no_formal_forbidden_surface,
    AccessLedger, canonical_json, canonical_root_path, load_config, payload_hash,
    publish_json, require_slurm, resolve, root_relative_path, sha256_bytes,
    sha256_file, sha256_obj,
)


def _load_json(path):
    fs_path, _ = canonical_root_path(path)
    with open(fs_path, encoding="utf-8") as handle:
        return json.load(handle)


def _valid_payload(obj):
    return obj.get("payload_sha256") == payload_hash(obj)


def _allowed_npz(path, names):
    path = canonical_root_path(path)[0]
    out = {}
    member_hashes = {}
    with zipfile.ZipFile(path, "r") as archive:
        members = {Path(name).stem: name for name in archive.namelist()
                   if name.endswith(".npy")}
        for forbidden in ("query_z", "query_labels"):
            if forbidden in members:
                # Presence is tolerated; this verifier never opens them.
                pass
        missing = set(names) - set(members)
        if missing:
            raise RuntimeError("missing allowed members {}".format(sorted(missing)))
        for name in names:
            with archive.open(members[name], "r") as handle:
                payload = handle.read()
            member_hashes[name] = sha256_bytes(payload)
            out[name] = np.load(io.BytesIO(payload), allow_pickle=False)
    return out, member_hashes


def _exact_keys(obj, allowed, name):
    keys = set(obj)
    if keys != set(allowed):
        raise RuntimeError("{} schema keys {} != {}".format(name, sorted(keys), sorted(allowed)))


def task_verify(cfg, args, ledger):
    import torch
    if args.run_id != "LBSCGP-G0-SANITIZER-VERIFY-MHC_zh-F4-v1":
        raise RuntimeError("wrong sanitizer verifier run ID")
    out_decision = resolve(cfg, "sanitizer_decision")
    if out_decision.exists() or out_decision.with_name(out_decision.name + ".publish.lock").exists():
        raise RuntimeError("refusing to clobber {}".format(out_decision))
    manifest_path = QUARANTINE_MANIFEST
    provenance_path = resolve(cfg, "sanitized_provenance")
    feature_path = resolve(cfg, "outer_train_feature_cache")
    for path in (manifest_path, provenance_path, feature_path):
        if not path.exists():
            raise RuntimeError("missing sanitizer artifact {}".format(path))

    manifest = _load_json(manifest_path)
    provenance = _load_json(provenance_path)
    ledger.record_file(manifest_path, "quarantine_disclosure_read",
                       "sanitizer_manifest", sha256_file(manifest_path))
    ledger.record_file(provenance_path, "formal_provenance_read",
                       "sanitized_provenance", sha256_file(provenance_path))
    if not _valid_payload(manifest) or not _valid_payload(provenance):
        raise RuntimeError("invalid sanitizer payload hash")
    if manifest.get("quarantine_mixed_storage_read") is not True:
        raise RuntimeError("quarantine mixed read was not disclosed")
    if manifest.get("formal_g0_input") is not False:
        raise RuntimeError("quarantine manifest may not be a formal G0 input")
    assert_no_formal_forbidden_surface(provenance, "sanitized_provenance")
    if provenance.get("segment_cache_path") is not None or \
            provenance.get("segment_cache_sha256") is not None or \
            provenance.get("segment_artifact_created") is not False or \
            provenance.get("segment_objective_allowed") is not False:
        raise RuntimeError("sanitized provenance exposes a segment cache/objective")

    bank, member_hashes = _allowed_npz(
        resolve(cfg, "bank"), ["memory_ids", "memory_labels", "query_ids"])
    ids = [str(x) for x in bank["memory_ids"].tolist()]
    query_ids = [str(x) for x in bank["query_ids"].tolist()]
    labels = np.asarray(bank["memory_labels"], dtype=np.int64).reshape(-1)
    fixture = cfg["sealed_real_fixture"]
    if member_hashes["memory_ids"] != fixture["bank_member_sha256"]["memory_ids"] or \
            member_hashes["memory_labels"] != fixture["bank_member_sha256"]["memory_labels"] or \
            member_hashes["query_ids"] != fixture["bank_member_sha256"]["query_ids"]:
        raise RuntimeError("allowed NPZ member hash drift")
    if sha256_obj(ids) != fixture["memory_ids_sha256"] or \
            sha256_obj(query_ids) != fixture["query_ids_sha256"]:
        raise RuntimeError("ID sentinel hash drift")
    if set(ids) & set(query_ids):
        raise RuntimeError("memory/query ID overlap")

    feature = torch.load(feature_path, map_location="cpu", weights_only=True)
    _exact_keys(feature, {"ids", "img_feats", "text_feats", "labels"}, "feature cache")
    feature_ids = [str(x) for x in feature["ids"]]
    feature_labels = torch.as_tensor(feature["labels"]).reshape(-1).long()
    if feature_ids != ids or len(set(feature_ids)) != len(feature_ids):
        raise RuntimeError("feature IDs are not exact memory_ids order")
    if not torch.equal(feature_labels, torch.as_tensor(labels, dtype=torch.long)):
        raise RuntimeError("feature labels are not inherited memory_labels")
    img = torch.as_tensor(feature["img_feats"])
    text = torch.as_tensor(feature["text_feats"])
    if img.shape[0] != len(ids) or text.shape[0] != len(ids):
        raise RuntimeError("feature row count mismatch")

    gates = {
        "pre_freeze_disclosure_record_present": True,
        "formal_provenance_sanitized": True,
        "feature_schema_whitelist": True,
        "exact_id_order": feature_ids == ids,
        "zero_overlap_with_query_ids": not (set(ids) & set(query_ids)),
        "labels_inherited_from_memory_labels": True,
        "no_segment_artifact": provenance.get("segment_artifact_created") is False,
        "no_segment_objective": provenance.get("segment_objective_allowed") is False,
        "no_teacher_mllm_ocr_calls": int(provenance.get("teacher_mllm_ocr_calls", -1)) == 0,
        "no_network_external_calls": int(provenance.get("network_external_calls", -1)) == 0,
        "no_clobber_locks": all(path.with_name(path.name + ".publish.lock").exists()
                                for path in (feature_path, provenance_path, manifest_path)),
    }
    safe_contract = {
        "feature_cache_sha256": sha256_file(feature_path),
        "sanitized_provenance_sha256": sha256_file(provenance_path),
        "memory_ids_sha256": sha256_obj(ids),
        "query_ids_sha256": sha256_obj(query_ids),
        "memory_labels_sha256": sha256_obj(labels.tolist()),
        "segment_artifact_created": False,
        "segment_objective_allowed": False,
        "feature_schema": sorted(feature.keys()),
    }
    decision = {
        "schema_version": 1,
        "run_id": args.run_id,
        "stage": "LB_SCGP_SANITIZER_DECISION",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "dataset": "MHC_zh",
        "outer_fold": 4,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gates": gates,
        "feature_cache_path": root_relative_path(feature_path),
        "feature_cache_sha256": safe_contract["feature_cache_sha256"],
        "segment_cache_path": None,
        "segment_cache_sha256": None,
        "segment_artifact_created": False,
        "segment_objective_allowed": False,
        "sanitized_provenance_path": root_relative_path(provenance_path),
        "sanitized_provenance_sha256": safe_contract["sanitized_provenance_sha256"],
        "memory_id_count": len(ids),
        "query_id_sentinel_count": len(query_ids),
        "memory_ids_sha256": sha256_obj(ids),
        "query_ids_sha256": sha256_obj(query_ids),
        "memory_labels_sha256": sha256_obj(labels.tolist()),
        "safe_contract_sha256": sha256_obj(safe_contract),
        "formal_model_optimizer_evaluator_outer_held_read_count": 0,
        "formal_query_z_read_count": 0,
        "formal_query_labels_read_count": 0,
        "teacher_mllm_ocr_calls": 0,
        "network_external_calls": 0,
    }
    assert_no_formal_forbidden_surface(decision, "sanitizer_decision")
    decision["payload_sha256"] = payload_hash(decision)
    publish_json(out_decision, decision)
    print(canonical_json({"status": decision["status"], "run_id": args.run_id}))
    if decision["status"] != "PASS":
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=["verify"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    require_slurm(expected_gpu=None)
    ledger = AccessLedger()
    cfg = load_config(args.config, ledger)
    task_verify(cfg, args, ledger)


if __name__ == "__main__":
    main()
