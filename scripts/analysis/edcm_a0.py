#!/usr/bin/env python
"""Strict EDCM A0 reuse audit and video-label-only reachability screen.

This program never reads an MLLM, OCR, coalition, proxy, or teacher artifact.
The sole gold field it consumes is the parent video's binary label stored in
the frozen five-fold train OOF artifacts.  K4 subclip labels are audited only
as inherited parent-video labels and are never treated as segment gold.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np


ROOT = Path("/data/jehc223/RGCL")
DATASETS = ("MHC", "MHC_zh")
RUN_IDS = {
    "reuse-audit": "EDCM-A0-REUSE-AUDIT-v1",
    "decision": "EDCM-A0-DECISION-v1",
    "reachability:MHC": "EDCM-A0-REACH-MHC-v1",
    "reachability:MHC_zh": "EDCM-A0-REACH-MHC_zh-v1",
}
OOF_JOB_IDS = {
    "MHC": ("12691", "12692", "12693", "12694", "12695"),
    "MHC_zh": ("12696", "12697", "12698", "12699", "12700"),
}
FROZEN = {
    "topk": 20,
    "search_depth": 64,
    "max_swaps": 2,
    "support_min_each_label_relation": 4,
    "supported_fraction_min": 0.80,
    "oracle_accuracy_gain_min": 0.050,
    "oracle_macro_f1_gain_min": 0.050,
}


def canonical_json(obj) -> str:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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
                raise RuntimeError(f"{path}:{lineno}: invalid JSON: {exc}")
    return rows


def refuse_nonempty(path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        raise RuntimeError(f"refusing to overwrite nonempty output: {path}")


def acquire_persistent_lock(path, run_id):
    """Atomically reserve one immutable output namespace.

    The lock deliberately remains after success or failure.  A repeated or
    concurrent fixed-run submission therefore cannot create a mixed lineage.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(f"output namespace is already reserved: {path}") from exc
    try:
        payload = canonical_json({
            "run_id": run_id,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "purpose": "persistent_non_overwrite_namespace_lock",
        }) + "\n"
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_temp_exclusive(tmp, path):
    """Publish a same-directory temporary file only if target is absent."""
    try:
        os.link(tmp, path)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite existing output: {path}") from exc
    os.unlink(tmp)


def atomic_write_json(path, obj):
    canonical_json(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    refuse_nonempty(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                obj, handle, ensure_ascii=False, sort_keys=True, indent=2,
                allow_nan=False)
            handle.write("\n")
        publish_temp_exclusive(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    refuse_nonempty(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        publish_temp_exclusive(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def add_payload_hash(obj, field="payload_sha256"):
    out = dict(obj)
    if field in out:
        raise RuntimeError(f"payload already contains {field}")
    out[field] = sha256_obj(out)
    return out


def verify_payload(obj, field="payload_sha256"):
    raw = dict(obj)
    stored = raw.pop(field, None)
    if stored is None or stored != sha256_obj(raw):
        raise RuntimeError(f"payload hash mismatch for {field}")


def resolve(cfg, key):
    path = Path(cfg["paths"][key])
    return path if path.is_absolute() else ROOT / path


def load_config(path, allow_placeholders=False):
    cfg = read_json(path)
    raw = dict(cfg)
    stored = raw.pop("config_sha256", None)
    computed = sha256_obj(raw)
    placeholder = "__FILL_AFTER_SLURM_SANITY__"
    if not allow_placeholders and stored != computed:
        raise RuntimeError(
            f"EDCM config is not frozen: stored={stored} computed={computed}")
    if not allow_placeholders and any(
            value == placeholder
            for value in cfg["expected"]["implementation"].values()):
        raise RuntimeError("implementation hashes remain placeholders")
    cfg["computed_config_sha256"] = computed
    return cfg


def require_slurm_environment():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("EDCM A0 computation must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")


def validate_frozen_contract(cfg):
    if cfg.get("schema_version") != 1:
        raise RuntimeError("unsupported EDCM config schema")
    if cfg.get("stage") != \
            "A0_pre_mllm_frozen_geometry_reachability_cost_screen":
        raise RuntimeError("A0 stage name drift")
    for key, value in FROZEN.items():
        if cfg["reachability"].get(key) != value:
            raise RuntimeError(
                f"frozen reachability constant changed: {key}="
                f"{cfg['reachability'].get(key)} expected={value}")
    supervision = cfg["supervision_contract"]
    expected = {
        "only_gold_supervision": "video_level_binary_label",
        "segment_gold_exists": False,
        "segment_gold_used": False,
        "subclip_label_status": "inherited_parent_video_label_not_segment_gold",
        "mllm_calls_allowed": False,
        "ocr_calls_allowed": False,
        "teacher_cache_allowed": False,
        "validation_test_teacher_artifact_count": 0,
    }
    if supervision != expected:
        raise RuntimeError("supervision contract drift")
    if set(cfg["expected"]["datasets"]) != set(DATASETS):
        raise RuntimeError("dataset set drift")


def exact_vote(neighbors):
    top = list(neighbors)[:FROZEN["topk"]]
    if len(top) != FROZEN["topk"]:
        raise RuntimeError("exact vote requires 20 records")
    vote = 0.0
    denom = 0.0
    for rank, record in enumerate(top, 1):
        weight = FROZEN["topk"] + 1 - rank
        cosine = float(record["cosine"])
        label = int(record["label"])
        if label not in (0, 1) or not math.isfinite(cosine):
            raise RuntimeError("non-binary label or non-finite cosine")
        vote += weight * cosine * (2 * label - 1)
        denom += weight * abs(cosine)
    return vote, int(vote >= 0.0), denom


def source_hash(source_map, path):
    path = Path(path)
    digest = sha256_file(path)
    source_map[str(path.relative_to(ROOT))] = digest
    return digest


def verify_repository_votes(rank_rows, labels):
    sys.path.insert(0, str(ROOT / "src"))
    from utils.metrics import compute_metrics_retrieval  # noqa: E402

    logging = OrderedDict()
    for row in rank_rows:
        top = row["ranking"][:FROZEN["topk"]]
        logging[row["query_id"]] = {
            "no_retrieved": FROZEN["topk"],
            "retrieved_ids": [record["id"] for record in top],
            "retrieved_scores": [
                np.float32(record["cosine"]) for record in top],
            "retrieved_label": [int(record["label"]) for record in top],
        }
    result = compute_metrics_retrieval(
        logging, np.asarray(labels, dtype=np.int64),
        majority_voting="arithmetic", topk=FROZEN["topk"], use_sim=True)
    repo_scores = [float(value) for value in result[5]]
    repo_predictions = [int(value >= 0.0) for value in repo_scores]
    return repo_scores, repo_predictions


def verify_dataset(cfg, dataset, verify_repo=True):
    expected = cfg["expected"]["datasets"][dataset]
    ssr_root = resolve(cfg, "ssr_artifacts")
    source_map = {}
    fold_path = ssr_root / "folds" / f"{dataset}.json"
    if source_hash(source_map, fold_path) != expected["fold_artifact_sha256"]:
        raise RuntimeError(f"{dataset}: fold artifact hash mismatch")
    folds = read_json(fold_path)
    if folds.get("dataset") != dataset:
        raise RuntimeError(f"{dataset}: fold dataset mismatch")
    if folds.get("config_sha256") != \
            cfg["expected"]["ssr_canonical_config_sha256"]:
        raise RuntimeError(f"{dataset}: fold config mismatch")
    if folds.get("only_gold_supervision") != "video_level_binary_label":
        raise RuntimeError(f"{dataset}: non-video-level gold declaration")
    if folds.get("segment_gold_exists") is not False:
        raise RuntimeError(f"{dataset}: segment gold declaration is not false")

    records = folds.get("records", [])
    if len(records) != int(expected["n"]):
        raise RuntimeError(f"{dataset}: fold row count mismatch")
    by_id = {}
    for record in records:
        video_id = str(record.get("id"))
        label = record.get("label")
        fold = record.get("fold")
        if video_id in by_id or label not in (0, 1) or fold not in range(5):
            raise RuntimeError(f"{dataset}: malformed/duplicate fold record")
        by_id[video_id] = {"label": int(label), "fold": int(fold)}

    split = folds.get("split_assertions", {})
    if split.get("pairwise_disjoint") is not True:
        raise RuntimeError(f"{dataset}: frozen splits not pairwise disjoint")
    overlaps = split.get("overlaps", {})
    if any(overlaps.get(key) != [] for key in
           ("train_dev", "train_test", "dev_test")):
        raise RuntimeError(f"{dataset}: frozen split overlap is nonempty")
    data_records = [
        split["clip_cache"]["train"], split["subclip_cache"],
        split["train_gt"],
    ]
    for item in data_records:
        path = ROOT / item["path"]
        if source_hash(source_map, path) != item["sha256"]:
            raise RuntimeError(f"{dataset}: frozen data hash mismatch: {path}")

    all_rows = []
    vote_records = []
    fold_summaries = {}
    for fold in range(5):
        fold_dir = ssr_root / "oof" / dataset / f"fold{fold}"
        manifest_path = fold_dir / "manifest.json"
        manifest_digest = source_hash(source_map, manifest_path)
        if manifest_digest != expected["fold_manifest_sha256"][str(fold)]:
            raise RuntimeError(f"{dataset}/fold{fold}: manifest hash mismatch")
        manifest = read_json(manifest_path)
        expected_run = f"SSR-B0-OOF-{dataset}-F{fold}-S0"
        if manifest.get("run_id") != expected_run or \
                str(manifest.get("slurm_job_id")) != OOF_JOB_IDS[dataset][fold]:
            raise RuntimeError(f"{dataset}/fold{fold}: run/job provenance mismatch")
        if manifest.get("status") != "COMPLETED" or \
                manifest.get("outer_fold") != fold:
            raise RuntimeError(f"{dataset}/fold{fold}: incomplete/wrong fold manifest")
        if manifest.get("config_sha256") != \
                cfg["expected"]["ssr_canonical_config_sha256"] or \
                manifest.get("fold_artifact_sha256") != \
                expected["fold_artifact_sha256"]:
            raise RuntimeError(f"{dataset}/fold{fold}: config/fold provenance mismatch")
        if manifest.get("fixed_epoch_index") != expected["epoch_index"]:
            raise RuntimeError(f"{dataset}/fold{fold}: comparator epoch drift")
        if manifest.get("only_gold_supervision") != \
                "video_level_binary_label" or \
                manifest.get("segment_gold_exists") is not False:
            raise RuntimeError(f"{dataset}/fold{fold}: supervision drift")
        subclip = manifest.get("subclip_contract", {})
        if subclip.get("label_source") != \
                "inherited_parent_video_label_not_segment_gold" or \
                int(subclip.get("n_subclips", -1)) != \
                4 * int(subclip.get("n_parents", -2)):
            raise RuntimeError(
                f"{dataset}/fold{fold}: K4 labels are not verified parent inheritance")
        if manifest.get("query_memory_overlap") != [] or \
                manifest.get("dev_or_test_endpoint_count") != 0:
            raise RuntimeError(f"{dataset}/fold{fold}: endpoint leakage declaration")

        for name, digest in manifest.get("outputs", {}).items():
            path = fold_dir / name
            if source_hash(source_map, path) != digest:
                raise RuntimeError(f"{dataset}/fold{fold}: output hash mismatch: {name}")
        rank_path = fold_dir / "ranking.jsonl"
        pred_path = fold_dir / "predictions.json"
        rank_rows = read_jsonl(rank_path)
        pred_rows = read_json(pred_path)
        query_ids = sorted(
            video_id for video_id, record in by_id.items()
            if record["fold"] == fold)
        memory_ids = sorted(
            video_id for video_id, record in by_id.items()
            if record["fold"] != fold)
        if len(rank_rows) != len(query_ids) or len(pred_rows) != len(query_ids):
            raise RuntimeError(f"{dataset}/fold{fold}: query output row count mismatch")
        if [row.get("query_id") for row in rank_rows] != query_ids:
            raise RuntimeError(f"{dataset}/fold{fold}: ranking query order mismatch")
        pred_by_id = {str(row.get("query_id")): row for row in pred_rows}
        if len(pred_by_id) != len(pred_rows) or set(pred_by_id) != set(query_ids):
            raise RuntimeError(f"{dataset}/fold{fold}: prediction IDs malformed")

        labels_in_order = []
        fold_vote_records = []
        for row in rank_rows:
            query_id = str(row["query_id"])
            query_label = by_id[query_id]["label"]
            ranking = row.get("ranking", [])
            if row.get("query_label") != query_label or \
                    row.get("outer_fold") != fold or \
                    row.get("memory_n") != len(memory_ids):
                raise RuntimeError(f"{dataset}/{query_id}: query metadata mismatch")
            if len(ranking) != len(memory_ids):
                raise RuntimeError(f"{dataset}/{query_id}: ranking is not full")
            ranked_ids = [str(record.get("id")) for record in ranking]
            if len(set(ranked_ids)) != len(memory_ids) or \
                    set(ranked_ids) != set(memory_ids) or query_id in ranked_ids:
                raise RuntimeError(f"{dataset}/{query_id}: memory ranking IDs invalid")
            previous_key = None
            for index, record in enumerate(ranking, 1):
                key_id = str(record.get("id"))
                cosine = float(record.get("cosine"))
                if record.get("rank") != index or not math.isfinite(cosine) or \
                        record.get("label") != by_id[key_id]["label"]:
                    raise RuntimeError(f"{dataset}/{query_id}: ranking record invalid")
                key = (-cosine, key_id)
                if previous_key is not None and key < previous_key:
                    raise RuntimeError(f"{dataset}/{query_id}: ranking is not canonical")
                previous_key = key
            vote, prediction, _ = exact_vote(ranking)
            stored = pred_by_id[query_id]
            if stored.get("query_label") != query_label or \
                    stored.get("outer_fold") != fold or \
                    stored.get("prediction") != prediction or \
                    stored.get("baseline_error") != int(prediction != query_label) or \
                    not math.isclose(
                        float(stored.get("vote")), vote,
                        rel_tol=0.0, abs_tol=1e-10):
                raise RuntimeError(f"{dataset}/{query_id}: exact vote mismatch")
            labels_in_order.append(query_label)
            vote_record = {
                "query_id": query_id, "video_label": query_label,
                "outer_fold": fold, "vote": vote, "prediction": prediction,
                "ranking_file_sha256": source_map[str(rank_path.relative_to(ROOT))],
                "source_manifest_sha256": manifest_digest,
                "top64_records_sha256": sha256_obj(ranking[:64]),
            }
            fold_vote_records.append(vote_record)
            all_rows.append({"row": row, "prediction": stored,
                             "manifest_sha256": manifest_digest,
                             "ranking_sha256": source_map[
                                 str(rank_path.relative_to(ROOT))]})

        if verify_repo:
            repo_scores, repo_predictions = verify_repository_votes(
                rank_rows, labels_in_order)
            exact_predictions = [record["prediction"]
                                 for record in fold_vote_records]
            if repo_predictions != exact_predictions:
                raise RuntimeError(
                    f"{dataset}/fold{fold}: repository prediction mismatch")
            for repo_score, record in zip(repo_scores, fold_vote_records):
                reconstructed = repo_score * sum(range(1, 21))
                if not math.isclose(
                        reconstructed, record["vote"],
                        rel_tol=1e-7, abs_tol=1e-6):
                    raise RuntimeError(
                        f"{dataset}/{record['query_id']}: repository vote mismatch")
        vote_records.extend(fold_vote_records)
        fold_summaries[str(fold)] = {
            "query_n": len(query_ids), "memory_n": len(memory_ids),
            "source_manifest_sha256": manifest_digest,
            "ranking_sha256": source_map[str(rank_path.relative_to(ROOT))],
            "predictions_sha256": source_map[str(pred_path.relative_to(ROOT))],
            "vote_records_sha256": sha256_obj(fold_vote_records),
            "repository_vote_verified": bool(verify_repo),
        }

    query_ids = [item["row"]["query_id"] for item in all_rows]
    if len(query_ids) != int(expected["n"]) or len(set(query_ids)) != len(query_ids) \
            or set(query_ids) != set(by_id):
        raise RuntimeError(f"{dataset}: OOF query coverage is not exact")
    return {
        "dataset": dataset,
        "n": len(query_ids),
        "fold_artifact_sha256": expected["fold_artifact_sha256"],
        "folds": fold_summaries,
        "oof_query_ids_sha256": sha256_obj(sorted(query_ids)),
        "vote_records_sha256": sha256_obj(vote_records),
        "source_file_sha256": dict(sorted(source_map.items())),
        "only_gold_supervision": "video_level_binary_label",
        "segment_gold_exists": False,
        "segment_gold_used": False,
        "subclip_label_status": "inherited_parent_video_label_not_segment_gold",
        "validation_test_endpoint_count": 0,
        "validation_test_source_files_read": 0,
        "teacher_or_mllm_artifact_count": 0,
        "_rows": all_rows,
    }


def audit_payload(cfg, run_id):
    validate_frozen_contract(cfg)
    source_map = {}
    actual_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual_head != cfg["expected"]["git_head"]:
        raise RuntimeError(
            f"repository HEAD mismatch: {actual_head} != {cfg['expected']['git_head']}")
    for relative, expected_hash in {
            **cfg["expected"]["files"],
            **cfg["expected"]["implementation"],
    }.items():
        path = ROOT / relative
        actual = source_hash(source_map, path)
        if actual != expected_hash:
            raise RuntimeError(f"frozen file hash mismatch: {relative}")

    ssr_cfg = read_json(resolve(cfg, "ssr_config"))
    ssr_raw = dict(ssr_cfg)
    ssr_stored = ssr_raw.pop("config_sha256", None)
    ssr_computed = sha256_obj(ssr_raw)
    if ssr_stored != ssr_computed or ssr_computed != \
            cfg["expected"]["ssr_canonical_config_sha256"]:
        raise RuntimeError("SSR canonical config digest mismatch")
    if ssr_cfg["comparator"].get("topk") != 20 or \
            ssr_cfg["comparator"].get("use_similarity_vote") is not True or \
            ssr_cfg["comparator"].get("vote_threshold") != 0.0 or \
            ssr_cfg["comparator"].get("seed") != 0:
        raise RuntimeError("SSR retrieval/vote comparator drift")
    for dataset in DATASETS:
        frozen_dataset = cfg["expected"]["datasets"][dataset]
        actual_dataset = ssr_cfg["datasets"][dataset]
        for field in ("epoch_index", "seg_mode", "lambda_seg"):
            if actual_dataset.get(field) != frozen_dataset[field]:
                raise RuntimeError(f"{dataset}: comparator recipe drift: {field}")
    freeze = read_json(resolve(cfg, "ssr_freeze_manifest"))
    if freeze.get("status") != "GO" or \
            freeze.get("required_config_sha256") != ssr_computed or \
            freeze.get("supervision_contract", {}).get("only_gold") != \
            "video_level_binary_label" or \
            freeze.get("supervision_contract", {}).get("segment_gold_exists") \
            is not False:
        raise RuntimeError("SSR freeze manifest contract mismatch")

    datasets = {}
    for dataset in DATASETS:
        detail = verify_dataset(cfg, dataset, verify_repo=True)
        rows = detail.pop("_rows")
        if len(rows) != detail["n"]:
            raise RuntimeError(f"{dataset}: internal audit row mismatch")
        datasets[dataset] = detail
        source_map.update(detail["source_file_sha256"])
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": cfg["stage"],
        "task": "reuse-audit",
        "status": "GO",
        "all_checks_pass": True,
        "failed_checks": [],
        "config_sha256": cfg["computed_config_sha256"],
        "repository_head": actual_head,
        "ssr_canonical_config_sha256": ssr_computed,
        "datasets": datasets,
        "source_file_sha256": dict(sorted(source_map.items())),
        "vote_definition": cfg["reachability"]["vote"],
        "prediction_definition": cfg["reachability"]["prediction"],
        "only_gold_supervision": "video_level_binary_label",
        "segment_gold_exists": False,
        "segment_gold_used": False,
        "subclip_label_status": "inherited_parent_video_label_not_segment_gold",
        "dev_test_rows_consumed_as_endpoints": 0,
        "validation_test_source_files_read": 0,
        "validation_test_teacher_artifact_count": 0,
        "teacher_or_mllm_artifact_count": 0,
        "ocr_artifact_count": 0,
        "edcm_mllm_calls_before_decision": 0,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
    }
    return add_payload_hash(payload)


def task_reuse_audit(cfg, run_id):
    output = resolve(cfg, "output") / "a0" / "reuse_audit.json"
    acquire_persistent_lock(output.parent / ".reuse_audit.lock", run_id)
    refuse_nonempty(output)
    try:
        payload = audit_payload(cfg, run_id)
    except Exception as exc:
        payload = add_payload_hash({
            "schema_version": 1, "run_id": run_id,
            "stage": cfg.get("stage"), "task": "reuse-audit",
            "status": "STOP", "all_checks_pass": False,
            "failed_checks": [f"{type(exc).__name__}: {exc}"],
            "config_sha256": cfg.get("computed_config_sha256"),
            "only_gold_supervision": "video_level_binary_label",
            "segment_gold_exists": False, "segment_gold_used": False,
            "validation_test_source_files_read": 0,
            "validation_test_teacher_artifact_count": 0,
            "teacher_or_mllm_artifact_count": 0,
            "ocr_artifact_count": 0,
            "edcm_mllm_calls_before_decision": 0,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        })
    atomic_write_json(output, payload)
    print(canonical_json({
        "run_id": run_id, "status": payload["status"],
        "output": str(output.relative_to(ROOT)),
        "failed_checks": payload.get("failed_checks", []),
    }), flush=True)


def binary_metrics(labels, predictions):
    if len(labels) != len(predictions) or not labels:
        raise RuntimeError("invalid metric vectors")
    confusion = [[0, 0], [0, 0]]
    for label, prediction in zip(labels, predictions):
        if label not in (0, 1) or prediction not in (0, 1):
            raise RuntimeError("non-binary metric input")
        confusion[label][prediction] += 1
    accuracy = (confusion[0][0] + confusion[1][1]) / float(len(labels))
    class_f1 = []
    for klass in (0, 1):
        tp = confusion[klass][klass]
        fp = confusion[1 - klass][klass]
        fn = confusion[klass][1 - klass]
        denom = 2 * tp + fp + fn
        class_f1.append(0.0 if denom == 0 else 2 * tp / float(denom))
    return {
        "confusion_matrix_labels_0_1": confusion,
        "accuracy": accuracy,
        "macro_f1": sum(class_f1) / 2.0,
        "per_class_f1_labels_0_1": class_f1,
    }


def best_witness(ranking, video_label):
    top20 = list(ranking[:20])
    outside = list(ranking[20:64])
    removable = [record for record in top20
                 if int(record["label"]) != video_label]
    addable = [record for record in outside
               if int(record["label"]) == video_label]
    true_sign = 2 * video_label - 1
    for swaps in (1, 2):
        if len(removable) < swaps or len(addable) < swaps:
            continue
        candidates = []
        add_combinations = list(itertools.combinations(addable, swaps))
        for removed in itertools.combinations(removable, swaps):
            removed_ids = tuple(sorted(str(record["id"]) for record in removed))
            removed_set = set(removed_ids)
            retained = [record for record in top20
                        if str(record["id"]) not in removed_set]
            for added in add_combinations:
                edited = sorted(
                    retained + list(added),
                    key=lambda record: (-float(record["cosine"]), str(record["id"])))
                vote, prediction, _ = exact_vote(edited)
                if prediction != video_label:
                    continue
                added_ids = tuple(sorted(str(record["id"]) for record in added))
                true_margin = true_sign * vote
                candidates.append((
                    -true_margin, removed_ids, added_ids, vote, prediction,
                    sha256_obj(edited)))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            best = candidates[0]
            return {
                "minimal_swaps": swaps,
                "removed_ids": list(best[1]),
                "added_ids": list(best[2]),
                "post_swap_vote": best[3],
                "post_swap_prediction": best[4],
                "true_class_signed_margin": -best[0],
                "edited_top20_sha256": best[5],
            }
    return None


def task_reachability(cfg, dataset, run_id):
    out_dir = resolve(cfg, "output") / "a0" / dataset
    rows_path = out_dir / "reachability.jsonl"
    metrics_path = out_dir / "metrics.json"
    manifest_path = out_dir / "manifest.json"
    acquire_persistent_lock(out_dir / ".reachability.lock", run_id)
    for path in (rows_path, metrics_path, manifest_path):
        refuse_nonempty(path)
    audit_path = resolve(cfg, "output") / "a0" / "reuse_audit.json"
    try:
        audit = read_json(audit_path)
        verify_payload(audit)
        if audit.get("status") != "GO" or \
                audit.get("all_checks_pass") is not True or \
                audit.get("config_sha256") != cfg["computed_config_sha256"]:
            raise RuntimeError("verified reuse audit is not GO for this config")
        if audit.get("edcm_mllm_calls_before_decision") != 0:
            raise RuntimeError("reuse audit does not prove zero prior EDCM MLLM calls")
        detail = verify_dataset(cfg, dataset, verify_repo=True)
        source_rows = detail.pop("_rows")
        for path, digest in detail["source_file_sha256"].items():
            if audit["source_file_sha256"].get(path) != digest:
                raise RuntimeError(f"{dataset}: source changed after audit: {path}")

        output_rows = []
        labels = []
        baseline_predictions = []
        oracle_predictions = []
        supported_count = 0
        reachable_count = 0
        for index, item in enumerate(source_rows, 1):
            row = item["row"]
            prediction_row = item["prediction"]
            ranking = row["ranking"]
            video_label = int(row["query_label"])
            baseline_vote, baseline_prediction, _ = exact_vote(ranking)
            top64 = ranking[:64]
            same_count = sum(
                int(record["label"]) == video_label for record in top64)
            opposite_count = len(top64) - same_count
            supported = same_count >= 4 and opposite_count >= 4
            witness = None
            if baseline_prediction != video_label:
                witness = best_witness(ranking, video_label)
            reachable = witness is not None
            oracle_prediction = video_label if reachable else baseline_prediction
            supported_count += int(supported)
            reachable_count += int(reachable)
            labels.append(video_label)
            baseline_predictions.append(baseline_prediction)
            oracle_predictions.append(oracle_prediction)
            output = {
                "schema_version": 1,
                "run_id": run_id,
                "dataset": dataset,
                "query_id": row["query_id"],
                "outer_fold": int(row["outer_fold"]),
                "video_label": video_label,
                "baseline_vote": baseline_vote,
                "baseline_prediction": baseline_prediction,
                "baseline_correct": baseline_prediction == video_label,
                "same_video_label_keys_top64": same_count,
                "opposite_video_label_keys_top64": opposite_count,
                "candidate_supported": supported,
                "reachable": reachable,
                "minimal_swaps": None if witness is None else witness["minimal_swaps"],
                "canonical_witness": witness,
                "oracle_prediction": oracle_prediction,
                "top64_records_sha256": sha256_obj(top64),
                "source_ranking_sha256": item["ranking_sha256"],
                "source_fold_manifest_sha256": item["manifest_sha256"],
                "config_sha256": cfg["computed_config_sha256"],
                "implementation_sha256": cfg["expected"]["implementation"][
                    "scripts/analysis/edcm_a0.py"],
                "only_gold_supervision": "video_level_binary_label",
                "segment_gold_exists": False,
                "segment_gold_used": False,
                "validation_test_teacher_artifact_count": 0,
                "teacher_or_mllm_artifact_count": 0,
                "ocr_artifact_count": 0,
                "edcm_mllm_calls_before_decision": 0,
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
            }
            if not math.isclose(
                    baseline_vote, float(prediction_row["vote"]),
                    rel_tol=0.0, abs_tol=1e-10):
                raise RuntimeError(f"{dataset}/{row['query_id']}: vote changed in reach job")
            output_rows.append(add_payload_hash(output))
            if index % 50 == 0 or index == len(source_rows):
                print(canonical_json({
                    "dataset": dataset, "processed": index,
                    "total": len(source_rows), "reachable_so_far": reachable_count,
                }), flush=True)

        baseline = binary_metrics(labels, baseline_predictions)
        oracle = binary_metrics(labels, oracle_predictions)
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        for predictions, ours in (
                (baseline_predictions, baseline), (oracle_predictions, oracle)):
            if not math.isclose(
                    float(accuracy_score(labels, predictions)), ours["accuracy"],
                    rel_tol=0.0, abs_tol=1e-15) or \
                    not math.isclose(
                        float(f1_score(
                            labels, predictions, average="macro", zero_division=0)),
                        ours["macro_f1"], rel_tol=0.0, abs_tol=1e-15) or \
                    confusion_matrix(labels, predictions, labels=[0, 1]).tolist() != \
                    ours["confusion_matrix_labels_0_1"]:
                raise RuntimeError(f"{dataset}: repository metric definition mismatch")
        supported_fraction = supported_count / float(len(labels))
        delta_accuracy = oracle["accuracy"] - baseline["accuracy"]
        delta_macro_f1 = oracle["macro_f1"] - baseline["macro_f1"]
        gates = {
            "all_video_supported_fraction": supported_fraction >= 0.80,
            "unique_reachable_errors": reachable_count >= int(
                cfg["expected"]["datasets"][dataset]["reachable_errors_min"]),
            "oracle_accuracy_gain": delta_accuracy >= 0.050,
            "oracle_macro_f1_gain": delta_macro_f1 >= 0.050,
            "fold_output_vote_metric_provenance": True,
        }
        metrics = add_payload_hash({
            "schema_version": 1, "run_id": run_id, "dataset": dataset,
            "stage": cfg["stage"], "status": "COMPLETED",
            "decision": "GO" if all(gates.values()) else "STOP",
            "N": len(labels), "baseline_error_count": sum(
                prediction != label for prediction, label in
                zip(baseline_predictions, labels)),
            "supported_count": supported_count,
            "supported_fraction": supported_fraction,
            "unique_reachable_errors": reachable_count,
            "reachable_fraction_all_videos": reachable_count / float(len(labels)),
            "reachable_fraction_baseline_errors": reachable_count / float(max(
                1, sum(prediction != label for prediction, label in
                       zip(baseline_predictions, labels)))),
            "baseline": baseline, "oracle": oracle,
            "delta_accuracy": delta_accuracy,
            "delta_macro_f1": delta_macro_f1,
            "gates": gates, "all_binding_gates_pass": all(gates.values()),
            "thresholds": {
                "supported_fraction_min": 0.80,
                "reachable_errors_min": int(
                    cfg["expected"]["datasets"][dataset]["reachable_errors_min"]),
                "oracle_accuracy_gain_min": 0.050,
                "oracle_macro_f1_gain_min": 0.050,
            },
            "reachability_rows_sha256": sha256_obj(output_rows),
            "config_sha256": cfg["computed_config_sha256"],
            "implementation_sha256": cfg["expected"]["implementation"],
            "reuse_audit_sha256": sha256_file(audit_path),
            "source_vote_records_sha256": detail["vote_records_sha256"],
            "source_file_sha256": detail["source_file_sha256"],
            "only_gold_supervision": "video_level_binary_label",
            "segment_gold_exists": False, "segment_gold_used": False,
            "validation_test_teacher_artifact_count": 0,
            "teacher_or_mllm_artifact_count": 0,
            "ocr_artifact_count": 0,
            "edcm_mllm_calls_before_decision": 0,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
        })
        atomic_write_jsonl(rows_path, output_rows)
        atomic_write_json(metrics_path, metrics)
        manifest = add_payload_hash({
            "schema_version": 1, "run_id": run_id, "dataset": dataset,
            "stage": cfg["stage"], "status": "COMPLETED",
            "decision": metrics["decision"],
            "config_sha256": cfg["computed_config_sha256"],
            "implementation_sha256": cfg["expected"]["implementation"],
            "reuse_audit_sha256": sha256_file(audit_path),
            "source_fold_manifest_sha256": {
                fold: value["source_manifest_sha256"]
                for fold, value in detail["folds"].items()},
            "source_file_sha256": detail["source_file_sha256"],
            "outputs": {
                "reachability.jsonl": sha256_file(rows_path),
                "metrics.json": sha256_file(metrics_path),
            },
            "row_count": len(output_rows),
            "only_gold_supervision": "video_level_binary_label",
            "segment_gold_exists": False, "segment_gold_used": False,
            "validation_test_teacher_artifact_count": 0,
            "teacher_or_mllm_artifact_count": 0,
            "ocr_artifact_count": 0,
            "edcm_mllm_calls_before_decision": 0,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
        })
        atomic_write_json(manifest_path, manifest)
        print(canonical_json({
            "run_id": run_id, "dataset": dataset,
            "decision": metrics["decision"], "gates": gates,
            "delta_accuracy": delta_accuracy,
            "delta_macro_f1": delta_macro_f1,
        }), flush=True)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        atomic_write_jsonl(rows_path, [])
        metrics = add_payload_hash({
            "schema_version": 1, "run_id": run_id, "dataset": dataset,
            "stage": cfg.get("stage"), "status": "FAILED_CLOSED",
            "decision": "STOP", "failure": failure,
            "gates": {"fold_output_vote_metric_provenance": False},
            "all_binding_gates_pass": False,
            "config_sha256": cfg.get("computed_config_sha256"),
            "implementation_sha256": cfg.get("expected", {}).get(
                "implementation", {}),
            "only_gold_supervision": "video_level_binary_label",
            "segment_gold_exists": False, "segment_gold_used": False,
            "validation_test_teacher_artifact_count": 0,
            "teacher_or_mllm_artifact_count": 0,
            "ocr_artifact_count": 0,
            "edcm_mllm_calls_before_decision": 0,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        })
        atomic_write_json(metrics_path, metrics)
        manifest = add_payload_hash({
            "schema_version": 1, "run_id": run_id, "dataset": dataset,
            "stage": cfg.get("stage"), "status": "FAILED_CLOSED",
            "decision": "STOP", "failure": failure,
            "config_sha256": cfg.get("computed_config_sha256"),
            "implementation_sha256": cfg.get("expected", {}).get(
                "implementation", {}),
            "outputs": {
                "reachability.jsonl": sha256_file(rows_path),
                "metrics.json": sha256_file(metrics_path),
            },
            "only_gold_supervision": "video_level_binary_label",
            "segment_gold_exists": False, "segment_gold_used": False,
            "validation_test_teacher_artifact_count": 0,
            "teacher_or_mllm_artifact_count": 0,
            "ocr_artifact_count": 0,
            "edcm_mllm_calls_before_decision": 0,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        })
        atomic_write_json(manifest_path, manifest)
        print(canonical_json({
            "run_id": run_id, "dataset": dataset, "decision": "STOP",
            "failure": failure,
        }), flush=True)


def verify_current_hashes(source_map):
    for relative, expected in source_map.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"source changed or missing: {relative}")


def require_signal_free_contract(obj, context):
    expected = {
        "only_gold_supervision": "video_level_binary_label",
        "segment_gold_exists": False,
        "segment_gold_used": False,
        "validation_test_teacher_artifact_count": 0,
        "teacher_or_mllm_artifact_count": 0,
        "ocr_artifact_count": 0,
        "edcm_mllm_calls_before_decision": 0,
    }
    for key, value in expected.items():
        if obj.get(key) != value:
            raise RuntimeError(
                f"{context}: signal/supervision contract mismatch: {key}")


def require_close(actual, expected, context, tolerance=1e-15):
    if not math.isfinite(float(actual)) or not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(f"{context}: numeric mismatch: {actual} != {expected}")


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def task_decision(cfg, run_id):
    output = resolve(cfg, "output") / "A0_DECISION.json"
    acquire_persistent_lock(output.parent / ".A0_DECISION.lock", run_id)
    refuse_nonempty(output)
    failures = []
    dataset_gates = {}
    source_manifest_sha256 = {}
    audit_path = resolve(cfg, "output") / "a0" / "reuse_audit.json"
    try:
        audit = read_json(audit_path)
        verify_payload(audit)
        audit_sha256 = sha256_file(audit_path)
        if audit.get("status") != "GO" or audit.get("all_checks_pass") is not True:
            raise RuntimeError("reuse audit is not verified GO")
        if audit.get("config_sha256") != cfg["computed_config_sha256"]:
            raise RuntimeError("reuse audit/config mismatch")
        require_signal_free_contract(audit, "reuse audit")
        if audit.get("validation_test_source_files_read") != 0 or \
                audit.get("dev_test_rows_consumed_as_endpoints") != 0:
            raise RuntimeError("reuse audit consumed validation/test sources/endpoints")
        verify_current_hashes(audit["source_file_sha256"])
        for dataset in DATASETS:
            dataset_dir = resolve(cfg, "output") / "a0" / dataset
            rows_path = dataset_dir / "reachability.jsonl"
            metrics_path = dataset_dir / "metrics.json"
            manifest_path = dataset_dir / "manifest.json"
            manifest = read_json(manifest_path)
            metrics = read_json(metrics_path)
            verify_payload(manifest)
            verify_payload(metrics)
            expected_run = RUN_IDS[f"reachability:{dataset}"]
            if manifest.get("run_id") != expected_run or \
                    metrics.get("run_id") != expected_run or \
                    manifest.get("dataset") != dataset or \
                    metrics.get("dataset") != dataset:
                raise RuntimeError(f"{dataset}: reach run ID mismatch")
            if manifest.get("config_sha256") != cfg["computed_config_sha256"] or \
                    metrics.get("config_sha256") != cfg["computed_config_sha256"]:
                raise RuntimeError(f"{dataset}: reach config mismatch")
            if manifest.get("stage") != cfg["stage"] or \
                    metrics.get("stage") != cfg["stage"]:
                raise RuntimeError(f"{dataset}: reach stage mismatch")
            if manifest.get("status") != "COMPLETED" or \
                    metrics.get("status") != "COMPLETED" or \
                    manifest.get("decision") != metrics.get("decision"):
                raise RuntimeError(f"{dataset}: reach status/decision mismatch")
            if manifest.get("reuse_audit_sha256") != audit_sha256 or \
                    metrics.get("reuse_audit_sha256") != audit_sha256:
                raise RuntimeError(f"{dataset}: stale/mixed reuse-audit lineage")
            if manifest.get("implementation_sha256") != \
                    cfg["expected"]["implementation"] or \
                    metrics.get("implementation_sha256") != \
                    cfg["expected"]["implementation"]:
                raise RuntimeError(f"{dataset}: implementation lineage mismatch")
            require_signal_free_contract(manifest, f"{dataset} manifest")
            require_signal_free_contract(metrics, f"{dataset} metrics")
            if manifest.get("slurm_job_id") != metrics.get("slurm_job_id"):
                raise RuntimeError(f"{dataset}: mixed reach SLURM lineage")
            if manifest.get("outputs", {}).get("reachability.jsonl") != \
                    sha256_file(rows_path) or \
                    manifest.get("outputs", {}).get("metrics.json") != \
                    sha256_file(metrics_path):
                raise RuntimeError(f"{dataset}: reach output hash mismatch")
            rows = read_jsonl(rows_path)
            expected_n = int(cfg["expected"]["datasets"][dataset]["n"])
            if len(rows) != expected_n or \
                    len(rows) != int(manifest.get("row_count", -1)) or \
                    metrics.get("N") != expected_n:
                raise RuntimeError(f"{dataset}: reach row count mismatch")
            authoritative = verify_dataset(cfg, dataset, verify_repo=True)
            authoritative_rows = authoritative.pop("_rows")
            authoritative_by_query = {
                item["row"]["query_id"]: item for item in authoritative_rows}
            if len(authoritative_by_query) != expected_n:
                raise RuntimeError(f"{dataset}: authoritative query IDs malformed")
            if authoritative["source_file_sha256"] != \
                    manifest.get("source_file_sha256") or \
                    authoritative["vote_records_sha256"] != \
                    metrics.get("source_vote_records_sha256"):
                raise RuntimeError(f"{dataset}: authoritative source lineage mismatch")
            audit_dataset = audit.get("datasets", {}).get(dataset, {})
            if authoritative["oof_query_ids_sha256"] != \
                    audit_dataset.get("oof_query_ids_sha256") or \
                    authoritative["vote_records_sha256"] != \
                    audit_dataset.get("vote_records_sha256") or \
                    authoritative["source_file_sha256"] != \
                    audit_dataset.get("source_file_sha256"):
                raise RuntimeError(f"{dataset}: authoritative data/audit mismatch")
            query_ids = []
            labels = []
            baseline_predictions = []
            oracle_predictions = []
            supported_count = 0
            reachable_count = 0
            for row in rows:
                verify_payload(row)
                require_signal_free_contract(row, f"{dataset} reach row")
                if row.get("run_id") != expected_run or \
                        row.get("dataset") != dataset or \
                        row.get("config_sha256") != cfg["computed_config_sha256"] or \
                        row.get("implementation_sha256") != \
                        cfg["expected"]["implementation"][
                            "scripts/analysis/edcm_a0.py"] or \
                        row.get("slurm_job_id") != manifest.get("slurm_job_id"):
                    raise RuntimeError(f"{dataset}: row lineage mismatch")
                query_id = row.get("query_id")
                label = row.get("video_label")
                prediction = row.get("baseline_prediction")
                oracle_prediction = row.get("oracle_prediction")
                if not isinstance(query_id, str) or not query_id or \
                        row.get("outer_fold") not in range(5) or \
                        label not in (0, 1) or prediction not in (0, 1) or \
                        oracle_prediction not in (0, 1) or \
                        not math.isfinite(float(row.get("baseline_vote"))):
                    raise RuntimeError(f"{dataset}: malformed reach row identity/value")
                if row.get("baseline_correct") != (prediction == label):
                    raise RuntimeError(f"{dataset}/{query_id}: baseline correctness mismatch")
                same_count = row.get("same_video_label_keys_top64")
                opposite_count = row.get("opposite_video_label_keys_top64")
                if not isinstance(same_count, int) or \
                        not isinstance(opposite_count, int) or \
                        same_count + opposite_count != 64:
                    raise RuntimeError(f"{dataset}/{query_id}: support counts malformed")
                supported = same_count >= 4 and opposite_count >= 4
                if row.get("candidate_supported") is not supported:
                    raise RuntimeError(f"{dataset}/{query_id}: support boolean mismatch")
                reachable = row.get("reachable")
                witness = row.get("canonical_witness")
                if not isinstance(reachable, bool):
                    raise RuntimeError(f"{dataset}/{query_id}: reachable is not boolean")
                if query_id not in authoritative_by_query:
                    raise RuntimeError(f"{dataset}/{query_id}: query not authoritative")
                source_item = authoritative_by_query[query_id]
                source_row = source_item["row"]
                source_ranking = source_row["ranking"]
                source_label = int(source_row["query_label"])
                source_vote, source_prediction, _ = exact_vote(source_ranking)
                source_top64 = source_ranking[:64]
                source_same_count = sum(
                    int(record["label"]) == source_label
                    for record in source_top64)
                source_opposite_count = 64 - source_same_count
                source_supported = source_same_count >= 4 and \
                    source_opposite_count >= 4
                source_witness = None
                if source_prediction != source_label:
                    source_witness = best_witness(source_ranking, source_label)
                source_reachable = source_witness is not None
                source_oracle_prediction = source_label if source_reachable \
                    else source_prediction
                if row.get("outer_fold") != source_row["outer_fold"] or \
                        label != source_label or \
                        not math.isclose(
                            float(row["baseline_vote"]), source_vote,
                            rel_tol=0.0, abs_tol=1e-10) or \
                        prediction != source_prediction or \
                        same_count != source_same_count or \
                        opposite_count != source_opposite_count or \
                        supported != source_supported or \
                        reachable != source_reachable or \
                        row.get("minimal_swaps") != (
                            None if source_witness is None
                            else source_witness["minimal_swaps"]) or \
                        row.get("canonical_witness") != source_witness or \
                        oracle_prediction != source_oracle_prediction or \
                        row.get("top64_records_sha256") != \
                        sha256_obj(source_top64) or \
                        row.get("source_fold_manifest_sha256") != \
                        source_item["manifest_sha256"] or \
                        row.get("source_ranking_sha256") != \
                        source_item["ranking_sha256"]:
                    raise RuntimeError(
                        f"{dataset}/{query_id}: row differs from authoritative ranking")
                if reachable:
                    if prediction == label or not isinstance(witness, dict) or \
                            row.get("minimal_swaps") not in (1, 2) or \
                            witness.get("minimal_swaps") != row.get("minimal_swaps") or \
                            witness.get("post_swap_prediction") != label or \
                            oracle_prediction != label:
                        raise RuntimeError(f"{dataset}/{query_id}: reachable witness invalid")
                    swaps = int(row["minimal_swaps"])
                    removed = witness.get("removed_ids")
                    added = witness.get("added_ids")
                    if not isinstance(removed, list) or not isinstance(added, list) or \
                            len(removed) != swaps or len(added) != swaps or \
                            removed != sorted(removed) or added != sorted(added) or \
                            len(set(removed)) != swaps or len(set(added)) != swaps or \
                            set(removed) & set(added) or \
                            witness.get("post_swap_prediction") != label or \
                            not is_sha256(witness.get("edited_top20_sha256")):
                        raise RuntimeError(f"{dataset}/{query_id}: canonical IDs invalid")
                    post_vote = float(witness.get("post_swap_vote"))
                    true_margin = float(witness.get("true_class_signed_margin"))
                    if not math.isfinite(post_vote) or not math.isfinite(true_margin):
                        raise RuntimeError(f"{dataset}/{query_id}: non-finite witness")
                    require_close(
                        true_margin, (2 * label - 1) * post_vote,
                        f"{dataset}/{query_id}: true-class margin")
                elif witness is not None or row.get("minimal_swaps") is not None or \
                        oracle_prediction != prediction:
                    raise RuntimeError(f"{dataset}/{query_id}: unreachable row changed")
                if prediction == label and reachable:
                    raise RuntimeError(f"{dataset}/{query_id}: correct baseline was changed")
                if not is_sha256(row.get("top64_records_sha256")) or \
                        row.get("source_fold_manifest_sha256") not in \
                        set(manifest["source_fold_manifest_sha256"].values()) or \
                        row.get("source_ranking_sha256") not in \
                        set(manifest["source_file_sha256"].values()):
                    raise RuntimeError(f"{dataset}/{query_id}: row source hash mismatch")
                query_ids.append(query_id)
                labels.append(label)
                baseline_predictions.append(prediction)
                oracle_predictions.append(oracle_prediction)
                supported_count += int(supported)
                reachable_count += int(reachable)
            if len(set(query_ids)) != expected_n:
                raise RuntimeError(f"{dataset}: duplicate reach query IDs")
            if sha256_obj(sorted(query_ids)) != \
                    audit_dataset.get("oof_query_ids_sha256"):
                raise RuntimeError(f"{dataset}: reach query set/audit mismatch")
            if sha256_obj(rows) != metrics.get("reachability_rows_sha256"):
                raise RuntimeError(f"{dataset}: canonical reach rows hash mismatch")
            if manifest.get("source_file_sha256") != \
                    metrics.get("source_file_sha256"):
                raise RuntimeError(f"{dataset}: mixed source hash manifests")
            verify_current_hashes(manifest["source_file_sha256"])
            expected_fold_manifests = cfg["expected"]["datasets"][dataset][
                "fold_manifest_sha256"]
            if manifest.get("source_fold_manifest_sha256") != \
                    expected_fold_manifests:
                raise RuntimeError(f"{dataset}: fold manifest lineage mismatch")

            baseline = binary_metrics(labels, baseline_predictions)
            oracle = binary_metrics(labels, oracle_predictions)
            if metrics.get("baseline") != baseline or metrics.get("oracle") != oracle:
                raise RuntimeError(f"{dataset}: reported metrics do not recompute")
            baseline_errors = sum(
                prediction != label for prediction, label in
                zip(baseline_predictions, labels))
            supported_fraction = supported_count / float(expected_n)
            delta_accuracy = oracle["accuracy"] - baseline["accuracy"]
            delta_macro_f1 = oracle["macro_f1"] - baseline["macro_f1"]
            if metrics.get("baseline_error_count") != baseline_errors or \
                    metrics.get("supported_count") != supported_count or \
                    metrics.get("unique_reachable_errors") != reachable_count:
                raise RuntimeError(f"{dataset}: reported counts do not recompute")
            require_close(
                metrics.get("supported_fraction"), supported_fraction,
                f"{dataset}: supported fraction")
            require_close(
                metrics.get("reachable_fraction_all_videos"),
                reachable_count / float(expected_n),
                f"{dataset}: reachable/all fraction")
            require_close(
                metrics.get("reachable_fraction_baseline_errors"),
                reachable_count / float(max(1, baseline_errors)),
                f"{dataset}: reachable/error fraction")
            require_close(
                metrics.get("delta_accuracy"), delta_accuracy,
                f"{dataset}: accuracy delta")
            require_close(
                metrics.get("delta_macro_f1"), delta_macro_f1,
                f"{dataset}: macro-F1 delta")
            expected_thresholds = {
                "supported_fraction_min": 0.80,
                "reachable_errors_min": int(
                    cfg["expected"]["datasets"][dataset]["reachable_errors_min"]),
                "oracle_accuracy_gain_min": 0.050,
                "oracle_macro_f1_gain_min": 0.050,
            }
            if metrics.get("thresholds") != expected_thresholds:
                raise RuntimeError(f"{dataset}: threshold drift")
            recomputed_gates = {
                "all_video_supported_fraction": supported_fraction >= 0.80,
                "unique_reachable_errors": reachable_count >= \
                    expected_thresholds["reachable_errors_min"],
                "oracle_accuracy_gain": delta_accuracy >= 0.050,
                "oracle_macro_f1_gain": delta_macro_f1 >= 0.050,
                "fold_output_vote_metric_provenance": True,
            }
            if metrics.get("gates") != recomputed_gates or \
                    metrics.get("all_binding_gates_pass") != \
                    all(recomputed_gates.values()) or \
                    metrics.get("decision") != (
                        "GO" if all(recomputed_gates.values()) else "STOP"):
                raise RuntimeError(f"{dataset}: gates/decision do not recompute")
            dataset_gates[dataset] = recomputed_gates
            source_manifest_sha256[dataset] = sha256_file(manifest_path)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")

    complete = set(dataset_gates) == set(DATASETS)
    all_pass = complete and not failures and all(
        all(gates.values()) for gates in dataset_gates.values())
    decision = "GO" if all_pass else "STOP"
    if complete:
        for dataset, gates in dataset_gates.items():
            for gate, passed in gates.items():
                if not passed:
                    failures.append(f"{dataset}:{gate}=false")
    payload = add_payload_hash({
        "schema_version": 1, "run_id": run_id,
        "stage": cfg["stage"], "decision": decision,
        "A1_unlocked": decision == "GO",
        "edcm_mllm_calls_before_decision": 0,
        "only_gold_supervision": "video_level_binary_label",
        "segment_gold_exists": False, "segment_gold_used": False,
        "validation_test_teacher_artifact_count": 0,
        "teacher_or_mllm_artifact_count": 0,
        "ocr_artifact_count": 0,
        "config_sha256": cfg["computed_config_sha256"],
        "reuse_audit_sha256": sha256_file(audit_path)
        if audit_path.is_file() else None,
        "source_manifest_sha256": source_manifest_sha256,
        "dataset_gates": dataset_gates,
        "all_binding_gates_pass": all_pass,
        "failed_cells_or_checks": failures,
        "A2_A3_locked": True,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
    })
    atomic_write_json(output, payload)
    print(canonical_json(payload), flush=True)


def sanity(config_path):
    """SLURM-only compile/freeze helper used by the sanity job."""
    require_slurm_environment()
    cfg = load_config(config_path, allow_placeholders=True)
    script_rel = "scripts/analysis/edcm_a0.py"
    sbatch_rel = "scripts/slurm/edcm_a0_cpu.sbatch"
    actual_impl = {
        script_rel: sha256_file(ROOT / script_rel),
        sbatch_rel: sha256_file(ROOT / sbatch_rel),
    }
    proposed = read_json(config_path)
    proposed["expected"]["implementation"] = actual_impl
    payload = dict(proposed)
    payload.pop("config_sha256", None)
    suggested_config_sha256 = sha256_obj(payload)
    placeholders = any(
        value == "__FILL_AFTER_SLURM_SANITY__"
        for value in cfg["expected"]["implementation"].values()) or \
        read_json(config_path).get("config_sha256") == \
        "__FILL_AFTER_SLURM_SANITY__"
    if not placeholders:
        if cfg["expected"]["implementation"] != actual_impl:
            raise RuntimeError("frozen implementation hash mismatch")
        if cfg["computed_config_sha256"] != read_json(config_path)["config_sha256"]:
            raise RuntimeError("frozen config digest mismatch")
        validate_frozen_contract(cfg)
    result = {
        "run_id": os.environ.get("RUN_ID"),
        "task": "sanity", "status": "NEEDS_FREEZE" if placeholders else "GO",
        "implementation_sha256": actual_impl,
        "suggested_config_sha256": suggested_config_sha256,
        "config_stored_sha256": read_json(config_path).get("config_sha256"),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
    }
    print(canonical_json(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--task", required=True,
        choices=("reuse-audit", "reachability", "decision"))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    require_slurm_environment()
    cfg = load_config(args.config)
    validate_frozen_contract(cfg)
    key = args.task if args.task != "reachability" else \
        f"reachability:{args.dataset}"
    if args.task == "reachability" and args.dataset is None:
        raise RuntimeError("reachability requires --dataset")
    if args.task != "reachability" and args.dataset is not None:
        raise RuntimeError(f"{args.task} forbids --dataset")
    if args.run_id != RUN_IDS.get(key):
        raise RuntimeError(
            f"run ID mismatch for {key}: {args.run_id} != {RUN_IDS.get(key)}")
    if args.task == "reuse-audit":
        task_reuse_audit(cfg, args.run_id)
    elif args.task == "reachability":
        task_reachability(cfg, args.dataset, args.run_id)
    else:
        task_decision(cfg, args.run_id)


if __name__ == "__main__":
    main()
