#!/usr/bin/env python
"""SQ S0: provenance, proxy, blind-QC freeze, P0/power/parity and micro.

No code in this module calls an MLLM, reads teacher caches, or opens validation
or test artifacts.  Parent-video labels from the frozen train folds are the
sole gold target.  No segment-level gold exists or is accepted.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from sq_common import (  # noqa: E402
    ArchiveAllowlistReader, acquire_namespace, add_exposure,
    archive_reader_poison_fixture, base_cluster_q, base_manifest,
    canonical_full_order, canonical_json, config_payload_and_hash, entropy_rows, exact_ranking,
    exclusive_write_bytes,
    exclusive_write_json, exclusive_write_jsonl, finite_or_raise,
    implementation_hash, input_record, kish_ess, load_config, make_shuffle_q,
    metrics_from_predictions, normalize_rows, output_records,
    posterior_affinity, random_matched_q, read_json, read_jsonl, require_runtime,
    resolve, sha256_file, sha256_obj, sha256_text, sq_loss_for_batch,
    sq_sampling_plan, stateless_seed,
)

DATASETS = ("MHC", "MHC_zh")


def artifact_root(cfg):
    return resolve(cfg, "artifacts")


def fold_path(cfg, dataset):
    return resolve(cfg, "ssr_artifacts") / "folds" / (dataset + ".json")


def q_dir(cfg, dataset):
    return artifact_root(cfg) / "s0" / "qproxy" / dataset


def fold_records(cfg, dataset):
    obj = read_json(fold_path(cfg, dataset))
    return obj, obj["records"]


def ensure_run_id(actual, expected):
    if actual != expected:
        raise RuntimeError("run-id must be exactly {}, got {}".format(expected, actual))


def input_inventory(cfg):
    paths = [ROOT / "configs/sq/sq_v1.json",
             ROOT / "artifacts/ssr/v1/freeze_manifest.json"]
    for d in DATASETS:
        paths.extend([
            fold_path(cfg, d),
            ROOT / cfg["datasets"][d]["archive"],
        ])
        for f in range(5):
            p = resolve(cfg, "ssr_artifacts") / "oof" / d / ("fold{}".format(f))
            paths.extend([p / "manifest.json", p / "embeddings.npz",
                          p / "ranking.jsonl", p / "predictions.json"])
    return paths


def task_static_sanity(cfg, run_id):
    ensure_run_id(run_id, "SQ-STATIC-SANITY-v1")
    import subprocess
    py_paths = [ROOT / "scripts/analysis/sq_common.py",
                ROOT / "scripts/analysis/sq_s0.py",
                ROOT / "scripts/analysis/sq_s1.py"]
    sh_paths = [ROOT / "scripts/slurm/sq_s0_cpu.sbatch",
                ROOT / "scripts/slurm/sq_s0_gpu.sbatch",
                ROOT / "scripts/slurm/sq_s1_cpu.sbatch",
                ROOT / "scripts/slurm/sq_s1_gpu.sbatch"]
    checks = {}
    for path in py_paths:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            checks[str(path.relative_to(ROOT))] = "PASS"
        except Exception as exc:
            checks[str(path.relative_to(ROOT))] = "FAIL:{}".format(exc)
    for path in sh_paths:
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True,
                              text=True, check=False)
        checks[str(path.relative_to(ROOT))] = (
            "PASS" if proc.returncode == 0 else "FAIL:{}".format(proc.stderr.strip()))
    status = "PASS" if all(x == "PASS" for x in checks.values()) else "STOP"
    out_dir = artifact_root(cfg) / "sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (run_id + ".json")
    payload = base_manifest(
        cfg, run_id, "STATIC_SANITY", status,
        inputs=[input_record(x) for x in py_paths + sh_paths],
        extra={"checks": checks, "no_gpu_compute": True,
               "new_teacher_call_count": 0})
    exclusive_write_json(out, payload)
    print(canonical_json({"run_id": run_id, "status": status}))


def task_freeze(cfg, run_id):
    ensure_run_id(run_id, "SQ-S0-FREEZE-v1")
    sanity_path = artifact_root(cfg) / "sanity/SQ-STATIC-SANITY-v1.json"
    if not sanity_path.is_file() or read_json(sanity_path).get("status") != "PASS":
        raise RuntimeError("reviewed static sanity predecessor missing")
    root = artifact_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    out = root / "CONFIG_FREEZE.json"
    if out.exists():
        raise RuntimeError("refusing to overwrite {}".format(out))
    paths = [sanity_path] + input_inventory(cfg)
    missing = [str(x) for x in paths if not x.is_file()]
    if missing:
        raise RuntimeError("missing frozen input(s): {}".format(missing))
    archive_hash_checks = {}
    fold_checks = {}
    for d in DATASETS:
        ap = ROOT / cfg["datasets"][d]["archive"]
        got = sha256_file(ap)
        expected = cfg["datasets"][d]["archive_sha256"]
        archive_hash_checks[d] = {"expected": expected, "actual": got,
                                  "pass": got == expected}
        fo = read_json(fold_path(cfg, d))
        assertions = fo["split_assertions"]
        # We consume only the already-frozen disjointness declarations.  No
        # validation/test path is opened or hashed in this task.
        fold_checks[d] = {
            "records": len(fo["records"]),
            "folds": sorted(set(int(x["fold"]) for x in fo["records"])),
            "pairwise_disjoint_assertion": assertions["pairwise_disjoint"],
            "overlaps_assertion": assertions["overlaps"],
            "val_test_files_opened": 0,
            "only_parent_video_labels": True,
            "segment_gold_exists": False,
            "segment_gold_used": False,
        }
    ok = all(x["pass"] for x in archive_hash_checks.values()) and all(
        x["folds"] == list(range(5)) and x["pairwise_disjoint_assertion"]
        for x in fold_checks.values())
    payload = base_manifest(
        cfg, run_id, "S0_FREEZE", "PASS" if ok else "INVALID_STOP",
        inputs=[input_record(x) for x in paths],
        extra={"archive_hash_checks": archive_hash_checks,
               "fold_checks": fold_checks,
               "fold_ids_sha256": sha256_obj({d: [
                   {"id": x["id"], "fold": x["fold"]}
                   for x in fold_records(cfg, d)[1]] for d in DATASETS}),
               "initialization_sha256": None})
    finite_or_raise(payload)
    exclusive_write_json(out, payload)
    print(canonical_json({"run_id": run_id, "status": payload["status"]}))


def original_log_candidates():
    pats = ["slurm/logs/gen_archive_v2*.out", "slurm/logs/*archive*v2*.out",
            "slurm/logs/*archive*v2*.log"]
    found = []
    for pat in pats:
        found.extend(Path(x) for x in glob.glob(str(ROOT / pat)))
    return sorted(set(x for x in found if x.is_file()))


def task_provenance(cfg, dataset, run_id):
    ensure_run_id(run_id, "SQ-S0-PROVENANCE-{}-v1".format(dataset))
    if not (artifact_root(cfg) / "CONFIG_FREEZE.json").is_file():
        raise RuntimeError("freeze predecessor missing")
    out_dir = artifact_root(cfg) / "s0" / "provenance" / dataset
    acquire_namespace(out_dir, run_id)
    archive_path = ROOT / cfg["datasets"][dataset]["archive"]
    generator = ROOT / "src/utils/generate_video_archive_HF.py"
    sbatch = ROOT / "scripts/slurm/gen_archive_v2.sbatch"
    folds, records = fold_records(cfg, dataset)
    reader = ArchiveAllowlistReader()
    poison_hash = archive_reader_poison_fixture(reader)
    projected = reader.read(archive_path)
    ids = [x["id"] for x in projected]
    expected_ids = [str(x["id"]) for x in records]
    id_valid = (len(ids) == len(set(ids)) and set(ids) == set(expected_ids) and
                all(x["split"] == "train" for x in projected))
    source = generator.read_text(encoding="utf-8")
    sbatch_text = sbatch.read_text(encoding="utf-8")
    static_flow = {
        "prompt_call_uses_item_text": 'build_user_prompt(item["text"], args.prompt_version)' in source,
        "label_written_to_output_record": '"label": item["label"]' in source,
        "generate_signature_has_no_label": "def generate_archive(frames, user_prompt, processor, model, device, max_new_tokens)" in source,
        "v2_train_requested_by_sbatch_default": "SPLITS=${SPLITS:-train,val,test}" in sbatch_text,
        "label_poison_projection_sha256": poison_hash,
    }
    logs = original_log_candidates()
    evidence = [input_record(generator), input_record(sbatch), input_record(archive_path),
                input_record(fold_path(cfg, dataset))]
    evidence.extend(input_record(x) for x in logs)
    # The archive and old log format do not embed original prompt/model revision,
    # generator/input-manifest hashes in each record.  Current source hashes and
    # timestamp adjacency are explicitly insufficient under the frozen plan.
    linkage = {
        "original_prompt_sha256_embedded": False,
        "exact_model_revision_embedded": False,
        "original_generator_sha256_embedded": False,
        "original_input_manifest_sha256_embedded": False,
        "current_generator_sha256": sha256_file(generator),
        "current_sbatch_sha256": sha256_file(sbatch),
        "candidate_original_logs": [str(x.relative_to(ROOT)) for x in logs],
        "cryptographically_complete": False,
        "reason": "original-run prompt/model-revision/input/code cryptographic linkage absent",
    }
    invalid = (not id_valid or reader.forbidden_access_count != 0 or
               not all(static_flow[k] for k in (
                   "prompt_call_uses_item_text", "label_written_to_output_record",
                   "generate_signature_has_no_label")))
    result = {
        "dataset": dataset, "id_set_valid": id_valid,
        "archive_rows": len(projected), "expected_train_rows": len(records),
        "reader_access_counts": reader.access,
        "archive_forbidden_key_access_count": reader.forbidden_access_count,
        "static_data_flow": static_flow, "original_run_linkage": linkage,
        "q_signal_status": "INVALID" if invalid else "PROXY_ONLY_CHEAP_FORMAT",
        "status": "INVALID_STOP" if invalid else "PASS_PROXY_ONLY",
        "promotion_allowed": False,
        "new_teacher_call_count": 0, "teacher_cache_read_count": 0,
        "teacher_cache_write_count": 0,
        "only_gold_supervision": "parent_video_binary_label",
        "segment_gold_exists": False, "segment_gold_used": False,
    }
    result["payload_sha256"] = sha256_obj(result)
    result_path = out_dir / "provenance.json"
    exclusive_write_json(result_path, result)
    manifest = base_manifest(
        cfg, run_id, "S0_PROVENANCE", result["status"], inputs=evidence,
        outputs=output_records([result_path]),
        extra={"dataset": dataset, "q_signal_status": result["q_signal_status"],
               "archive_forbidden_key_access_count": reader.forbidden_access_count,
               "fold_ids_sha256": sha256_obj(expected_ids)})
    exclusive_write_json(out_dir / "manifest.json", manifest)
    print(canonical_json({"run_id": run_id, "status": result["status"],
                          "q_signal_status": result["q_signal_status"]}))


def task_qproxy(cfg, dataset, run_id):
    ensure_run_id(run_id, "SQ-S0-QPROXY-{}-v1".format(dataset))
    prov = artifact_root(cfg) / "s0" / "provenance" / dataset / "provenance.json"
    if not prov.is_file() or read_json(prov)["status"] == "INVALID_STOP":
        raise RuntimeError("valid provenance-reader predecessor missing")
    out_dir = q_dir(cfg, dataset)
    acquire_namespace(out_dir, run_id)
    import torch
    from transformers import CLIPModel, CLIPTokenizerFast

    archive_path = ROOT / cfg["datasets"][dataset]["archive"]
    reader = ArchiveAllowlistReader()
    rows = reader.read(archive_path)
    model_name = cfg["signal"]["model"]
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name, local_files_only=True)
    model = CLIPModel.from_pretrained(model_name, local_files_only=True).cuda().eval()
    prototypes = cfg["signal"]["prototypes"]

    def encode_texts(texts):
        outputs = []
        for start in range(0, len(texts), 128):
            batch = texts[start:start + 128]
            toks = tokenizer(batch, padding=True, truncation=True,
                             max_length=77, return_tensors="pt")
            toks = {k: v.cuda() for k, v in toks.items()}
            with torch.no_grad():
                z = model.get_text_features(**toks)
                z = torch.nn.functional.normalize(z.float(), p=2, dim=1)
            outputs.append(z.cpu())
        return torch.cat(outputs, dim=0).numpy().astype(np.float32)

    proto_z = encode_texts(prototypes)
    summaries = [str(x["neutral_summary"]).strip()
                 if x["parse_ok"] and isinstance(x["neutral_summary"], str)
                 and x["neutral_summary"].strip() else "" for x in rows]
    usable_idx = [i for i, x in enumerate(summaries) if x]
    summary_z = np.zeros((len(rows), proto_z.shape[1]), dtype=np.float32)
    if usable_idx:
        enc = encode_texts([summaries[i] for i in usable_idx])
        summary_z[np.asarray(usable_idx)] = enc
    temp = float(cfg["signal"]["temperature"])
    posterior_rows = []
    for i, row in enumerate(rows):
        if i not in set(usable_idx):
            q = np.full(6, 1.0 / 6.0, dtype=np.float64); conf = 0.0
        else:
            logits = (summary_z[i] @ proto_z.T).astype(np.float64) / temp
            logits -= logits.max()
            q = np.exp(logits); q /= q.sum()
            h = -float(np.sum(q * np.log(np.maximum(q, 1e-300))))
            conf = max(0.0, 1.0 - h / math.log(6.0))
        posterior_rows.append({
            "id": row["id"], "q": [float(x) for x in q], "r": float(conf),
            "argmax": int(np.argmax(q)), "usable_summary": bool(summaries[i]),
        })
    ppath = out_dir / "posterior.jsonl"
    exclusive_write_jsonl(ppath, posterior_rows)
    coverage = sum(x["usable_summary"] for x in posterior_rows) / len(posterior_rows)
    signal_status = read_json(prov)["q_signal_status"]
    gpu_name = torch.cuda.get_device_name(0)
    manifest = base_manifest(
        cfg, run_id, "S0_QPROXY", "PASS" if coverage >= cfg["s0_gates"]["usable_coverage_min"] else "STOP",
        inputs=[input_record(archive_path), input_record(prov)],
        outputs=output_records([ppath]), gpu_name=gpu_name,
        extra={"dataset": dataset, "q_signal_status": signal_status,
               "artifact_name": "q_proxy", "rows": len(posterior_rows),
               "usable_rows": int(sum(x["usable_summary"] for x in posterior_rows)),
               "usable_coverage": coverage, "prototype_strings": prototypes,
               "prototype_sha256": sha256_obj(prototypes),
               "prototype_embedding_sha256": sha256_obj(proto_z.tolist()),
               "temperature": temp, "reader_access_counts": reader.access,
               "archive_forbidden_key_access_count": reader.forbidden_access_count,
               "posterior_ids_sha256": sha256_obj([x["id"] for x in posterior_rows])})
    exclusive_write_json(out_dir / "manifest.json", manifest)
    print(canonical_json({"run_id": run_id, "status": manifest["status"],
                          "coverage": coverage, "q_signal_status": signal_status}))


def allocate_audit_ids(rows, n, salt, dataset):
    by = defaultdict(list)
    for row in rows:
        by[int(row["argmax"])].append(row)
    alloc = {k: min(10, len(by[k])) for k in range(6)}
    remaining = n - sum(alloc.values())
    order = sorted(range(6), key=lambda k: (-len(by[k]), k))
    while remaining > 0:
        progressed = False
        for k in order:
            if alloc[k] < len(by[k]):
                alloc[k] += 1; remaining -= 1; progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("not enough unique rows for audit")
    selected = []
    for k in range(6):
        ordered = sorted(by[k], key=lambda x: sha256_text(
            "{}|{}|{}".format(salt, dataset, x["id"])))
        selected.extend(ordered[:alloc[k]])
    return selected, alloc


def task_audit_freeze(cfg, run_id):
    ensure_run_id(run_id, "SQ-S0-AUDIT-FREEZE-v1")
    out_dir = artifact_root(cfg) / "s0" / "audit"
    acquire_namespace(out_dir, run_id)
    sample_manifest = {
        "schema_version": 1, "run_id": run_id, "status": "AWAITING_HUMAN_QC",
        "qc_not_gold_supervision": True, "whole_video_only": True,
        "segment_gold_exists": False, "segment_gold_used": False,
        "dataset_label_column_present": False, "new_teacher_call_count": 0,
        "category_definitions": cfg["signal"]["prototypes"], "datasets": {},
    }
    output_paths = []
    for d in DATASETS:
        qpath = q_dir(cfg, d) / "posterior.jsonl"
        rows = read_jsonl(qpath)
        selected, alloc = allocate_audit_ids(
            rows, int(cfg["signal"]["audit_n_per_dataset"]),
            cfg["signal"]["audit_salt"], d)
        csv_path = out_dir / (d + ".csv")
        fields = ["audit_id", "video_path"] + ["p{}".format(i) for i in range(6)] + [
            "rater1_presentation_appropriate", "rater1_semantic_contamination",
            "rater1_reason_code", "rater1_hash", "rater1_timestamp",
            "rater2_presentation_appropriate", "rater2_semantic_contamination",
            "rater2_reason_code", "rater2_hash", "rater2_timestamp",
            "adjudicated_presentation_appropriate", "adjudicated_semantic_contamination",
            "adjudication_reason_code", "adjudicator_hash", "adjudication_timestamp"]
        lines = []
        import io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        id_map = []
        for row in selected:
            audit_id = sha256_text("{}|{}|{}".format(
                cfg["signal"]["audit_salt"], d, row["id"]))[:20]
            video = ROOT / cfg["paths"]["video"] / d / "All" / (row["id"] + ".mp4")
            rec = {"audit_id": audit_id, "video_path": str(video)}
            rec.update({"p{}".format(i): "{:.9g}".format(row["q"][i]) for i in range(6)})
            writer.writerow(rec)
            id_map.append({"audit_id": audit_id, "video_id": row["id"],
                           "argmax": row["argmax"]})
        exclusive_write_bytes(csv_path, buf.getvalue().encode("utf-8"))
        map_path = out_dir / (d + "_private_id_map.json")
        exclusive_write_json(map_path, id_map)
        output_paths.extend([csv_path, map_path])
        sample_manifest["datasets"][d] = {
            "n": len(selected), "allocation": alloc,
            "audit_csv": str(csv_path.relative_to(ROOT)),
            "private_id_map": str(map_path.relative_to(ROOT)),
            "sample_ids_sha256": sha256_obj([x["id"] for x in selected]),
            "result_status": "NOT_INGESTED_PROXY_ONLY",
        }
    sample_manifest["payload_sha256"] = sha256_obj(sample_manifest)
    mpath = out_dir / "sample_manifest.json"
    exclusive_write_json(mpath, sample_manifest)
    output_paths.append(mpath)
    manifest = base_manifest(
        cfg, run_id, "S0_AUDIT_FREEZE", "PASS_ARTIFACT_PREPARED",
        inputs=[input_record(q_dir(cfg, d) / "posterior.jsonl") for d in DATASETS],
        outputs=output_records(output_paths),
        extra={"blind_qc_completed": False, "q_signal_promotion_allowed": False,
               "q_signal_status_without_ingest": "PROXY_ONLY_CHEAP_FORMAT"})
    exclusive_write_json(out_dir / "manifest.json", manifest)
    print(canonical_json({"run_id": run_id, "status": manifest["status"],
                          "promotion": False}))


def wilson_lower(successes, n, z=1.959963984540054):
    if n <= 0:
        return 0.0
    p = successes / n
    den = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n))) / den


def task_audit_ingest(cfg, dataset, audit_csv, run_id):
    ensure_run_id(run_id, "SQ-S0-AUDIT-{}-v1".format(dataset))
    out_dir = artifact_root(cfg) / "s0" / "audit"
    result_path = out_dir / (dataset + "_result.json")
    if result_path.exists():
        raise RuntimeError("refusing overwrite {}".format(result_path))
    with open(audit_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 64 or len({x["audit_id"] for x in rows}) != 64:
        raise RuntimeError("audit ingest needs 64 unique rows")
    appropriate = contaminated = 0
    for row in rows:
        a = row.get("adjudicated_presentation_appropriate") or row.get("rater1_presentation_appropriate")
        c = row.get("adjudicated_semantic_contamination") or row.get("rater1_semantic_contamination")
        if a not in {"yes", "no"} or c not in {"yes", "no"}:
            raise RuntimeError("incomplete/invalid adjudicated audit row")
        appropriate += int(a == "yes"); contaminated += int(c == "yes")
    lower = wilson_lower(appropriate, 64)
    passed = contaminated <= 3 and lower >= 0.90
    result = {"dataset": dataset, "n": 64, "appropriate": appropriate,
              "contaminated": contaminated, "appropriateness_wilson_lower": lower,
              "status": "PASS" if passed else "FAIL_PROXY_ONLY",
              "qc_not_gold_supervision": True, "segment_gold_exists": False,
              "segment_gold_used": False, "input_sha256": sha256_file(audit_csv)}
    result["payload_sha256"] = sha256_obj(result)
    exclusive_write_json(result_path, result)
    print(canonical_json({"run_id": run_id, "status": result["status"]}))


def load_q(cfg, dataset):
    rows = read_jsonl(q_dir(cfg, dataset) / "posterior.jsonl")
    by = {x["id"]: x for x in rows}
    if len(by) != len(rows):
        raise RuntimeError("duplicate q IDs")
    return rows, by


def soft_cluster_for(z, centers):
    x = normalize_rows(z)
    logits = (x @ centers.T).astype(np.float64) / 0.1
    logits -= logits.max(axis=1, keepdims=True)
    q = np.exp(logits); q /= q.sum(axis=1, keepdims=True)
    return q


def fit_predict_rotations(rows, feature_names_base, feature_names_plus,
                          target_name, fold_name="outer_fold"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    folds = sorted(set(int(x[fold_name]) for x in rows))
    out = []
    fold_delta = []
    for held in folds:
        train = [x for x in rows if int(x[fold_name]) != held]
        test = [x for x in rows if int(x[fold_name]) == held]
        ytr = np.asarray([x[target_name] for x in train], dtype=np.int64)
        yte = np.asarray([x[target_name] for x in test], dtype=np.int64)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            raise RuntimeError("single-class rotation fold {}".format(held))
        xb = np.asarray([[x[k] for k in feature_names_base] for x in train], dtype=np.float64)
        xp = np.asarray([[x[k] for k in feature_names_plus] for x in train], dtype=np.float64)
        tb = np.asarray([[x[k] for k in feature_names_base] for x in test], dtype=np.float64)
        tp = np.asarray([[x[k] for k in feature_names_plus] for x in test], dtype=np.float64)
        base = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(xb, ytr)
        plus = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(xp, ytr)
        pb = base.predict_proba(tb)[:, 1]; pp = plus.predict_proba(tp)[:, 1]
        ab = float(roc_auc_score(yte, pb)); ap = float(roc_auc_score(yte, pp))
        fold_delta.append(ap - ab)
        for row, b, p in zip(test, pb, pp):
            out.append({"anchor_id": row["anchor_id"], "outer_fold": held,
                        "video_class": int(row["video_class"]),
                        "target": int(row[target_name]),
                        "base_score": float(b), "plus_score": float(p)})
    y = [x["target"] for x in out]
    from sklearn.metrics import roc_auc_score
    overall = float(roc_auc_score(y, [x["plus_score"] for x in out]) -
                    roc_auc_score(y, [x["base_score"] for x in out]))
    return out, fold_delta, overall


def stratified_anchor_bootstrap_auc(pred_rows, replicates, seed):
    from sklearn.metrics import roc_auc_score
    by_anchor = defaultdict(list)
    meta = {}
    for row in pred_rows:
        by_anchor[row["anchor_id"]].append(row)
        meta[row["anchor_id"]] = (row["outer_fold"], row["video_class"])
    strata = defaultdict(list)
    for aid, key in meta.items():
        strata[key].append(aid)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        sampled = []
        for ids in strata.values():
            sampled.extend(rng.choice(ids, size=len(ids), replace=True).tolist())
        rows = [r for aid in sampled for r in by_anchor[aid]]
        y = np.asarray([x["target"] for x in rows])
        if len(np.unique(y)) < 2:
            continue
        deltas.append(float(roc_auc_score(y, [x["plus_score"] for x in rows]) -
                            roc_auc_score(y, [x["base_score"] for x in rows])))
    if len(deltas) != replicates:
        raise RuntimeError("undefined bootstrap AUC replicate")
    return {"replicates": replicates,
            "lower_95": float(np.percentile(deltas, 2.5)),
            "upper_95": float(np.percentile(deltas, 97.5)),
            "median": float(np.median(deltas))}


def anchor_alignment(query_z, query_label, memory_ids, memory_z, memory_labels,
                     query_q, query_r, memory_q, memory_r, cfg, anchor_id,
                     fold, label_only=False):
    zq = normalize_rows(np.asarray(query_z).reshape(1, -1))[0].astype(np.float64)
    mz = normalize_rows(memory_z).astype(np.float64)
    labels = np.asarray(memory_labels, dtype=np.int64)
    ranking, _, _, _ = exact_ranking(
        memory_ids, mz, labels, zq, topk=cfg["evaluator"]["topk"])
    byid = {x: i for i, x in enumerate(memory_ids)}
    top = ranking
    sign_y = 2 * int(query_label) - 1
    grad_margin = sign_y * sum(
        x["weight"] * (2 * int(x["label"]) - 1) * mz[byid[x["id"]]] for x in top)
    pos_idx = np.flatnonzero(labels == int(query_label))
    if label_only:
        pos_w = np.ones(len(pos_idx), dtype=np.float64)
    else:
        aff = posterior_affinity(np.broadcast_to(query_q, (len(pos_idx), 6)), memory_q[pos_idx])
        pos_w = query_r * memory_r[pos_idx] * (1.0 - aff)
    neg_idx, neg_w, neg_e = [], [], []
    for x in top:
        j = byid[x["id"]]
        if labels[j] == int(query_label) or x["cosine"] <= 0:
            continue
        w = 1.0 if label_only else query_r * memory_r[j] * posterior_affinity(query_q, memory_q[j])
        if w > 0:
            neg_idx.append(j); neg_w.append(float(w)); neg_e.append(float(x["weight"] * x["cosine"]))
    neg_idx = np.asarray(neg_idx, dtype=np.int64)
    neg_w = np.asarray(neg_w, dtype=np.float64)
    if kish_ess(pos_w) < cfg["sq"]["min_kish_ess"] or kish_ess(neg_w) < cfg["sq"]["min_kish_ess"]:
        return None, kish_ess(pos_w), kish_ess(neg_w)
    pos_p = pos_w / pos_w.sum(); neg_p = neg_w / neg_w.sum()
    grads = []
    for d in range(cfg["sq"]["triplets_per_anchor"]):
        rng = np.random.default_rng(stateless_seed(
            cfg["computed_config_sha256"], cfg["sq"]["seed"], 0,
            anchor_id, d))
        p = int(pos_idx[int(rng.choice(len(pos_idx), p=pos_p))])
        npos = int(rng.choice(len(neg_idx), p=neg_p)); n = int(neg_idx[npos])
        e = neg_e[npos]
        x = (float(zq @ mz[n]) - float(zq @ mz[p]) + cfg["sq"]["margin"]) / cfg["sq"]["temperature"]
        sig = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))
        grads.append(e * sig / cfg["sq"]["temperature"] * (mz[n] - mz[p]))
    grad_loss = np.mean(grads, axis=0)
    return int(float(grad_margin @ (-grad_loss)) > 0.0), kish_ess(pos_w), kish_ess(neg_w)


def upper_variance_and_n(values, delta, N, cfg):
    from scipy.stats import chi2, norm
    x = np.asarray(values, dtype=np.float64)
    if N < 3 or len(x) != N or not np.all(np.isfinite(x)) or len(np.unique(x)) < 2:
        return {"N": N, "s2": None, "variance_U": None, "n0": None,
                "n_FPC": None, "status": "STOP_INFEASIBLE"}
    s2 = float(np.var(x, ddof=1))
    var_u = min(1.0, (N - 1) * s2 / float(chi2.ppf(0.05, N - 1)))
    z = float(norm.ppf(1 - cfg["power"]["alpha_star"] / 2) +
              norm.ppf(cfg["power"]["power"]))
    n0 = int(math.ceil(z * z * var_u / (delta * delta)))
    nfpc = int(math.ceil(N * n0 / (N - 1 + n0)))
    return {"N": N, "s2": s2, "variance_U": var_u,
            "alpha_star": cfg["power"]["alpha_star"], "delta": delta,
            "n0": n0, "n_FPC": nfpc, "status": "PASS"}


def synthetic_parity(cfg):
    # Twenty exact ties: weights 20..15 sum to 105 and 14..1 also
    # sum to 105, giving an actual float32 zero vote.
    ids = sorted(["vid-{}-{}".format(i * 17 + 3, chr(97 + (i * 7) % 26))
                  for i in range(20)])
    z = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (20, 1))
    labels = np.asarray([1] * 6 + [0] * 14, dtype=np.int64)
    rows, vote, pred, _ = exact_ranking(ids, z, labels, np.asarray([1, 0]), topk=20)
    tie_ok = [x["id"] for x in rows] == ids
    zero_ok = np.float32(vote) == np.float32(0.0) and pred == 1
    # Independent repository metric call on exactly the same ledger.
    from easydict import EasyDict
    from utils.metrics import compute_metrics_retrieval
    logging = EasyDict({"query-noncontiguous-991": {
        "no_retrieved": 20, "retrieved_ids": ids,
        "retrieved_scores": [np.float32(1.0)] * 20,
        "retrieved_label": labels.tolist()}})
    repo = compute_metrics_retrieval(
        logging, np.asarray([1]), majority_voting="arithmetic",
        topk=20, use_sim=True)
    repo_score = float(repo[5][0]); repo_pred = int(repo_score >= 0.0)
    neg_rows, _, _, _ = exact_ranking(
        ["neg-401", "pos-2"], np.asarray([[-1, 0], [1, 0]], dtype=np.float32),
        [1, 0], np.asarray([1, 0]), topk=2)
    negative_ok = any(x["cosine"] < 0 for x in neg_rows)
    return {"tie_canonical": tie_ok, "tie_ids": ids,
            "negative_cosine_present": negative_ok,
            "vote_exact_zero_case": vote, "vote_zero_prediction": pred,
            "repository_vote": repo_score, "repository_prediction": repo_pred,
            "repository_match": repo_pred == pred and np.float32(repo_score) == np.float32(vote),
            "noncontiguous_ids": ids,
            "pass": bool(tie_ok and negative_ok and zero_ok and repo_pred == pred and
                         np.float32(repo_score) == np.float32(vote))}


def task_parity_power_p0(cfg, dataset, run_id):
    ensure_run_id(run_id, "SQ-S0-PARITY-POWER-P0-{}-v1".format(dataset))
    out_dir = artifact_root(cfg) / "s0" / "p0" / dataset
    acquire_namespace(out_dir, run_id)
    qrows, qby = load_q(cfg, dataset)
    folds_obj, records = fold_records(cfg, dataset)
    labels_by = {str(x["id"]): int(x["label"]) for x in records}
    ids_all = [str(x["id"]) for x in records]
    q_all = np.asarray([qby[x]["q"] for x in ids_all], dtype=np.float64)
    r_all = np.asarray([qby[x]["r"] for x in ids_all], dtype=np.float64)
    y_all = np.asarray([labels_by[x] for x in ids_all], dtype=np.int64)
    q_global = {x: q_all[i] for i, x in enumerate(ids_all)}
    r_global = {x: r_all[i] for i, x in enumerate(ids_all)}
    q_shuffle, r_shuffle, _ = make_shuffle_q(ids_all, y_all, q_all, r_all,
                                              cfg["sq"]["shuffle_seed"])
    q_random, r_random, random_diag = random_matched_q(
        ids_all, q_all, r_all, cfg["sq"]["random_seed"])
    qsh = {x: q_shuffle[i] for i, x in enumerate(ids_all)}
    rsh = {x: r_shuffle[i] for i, x in enumerate(ids_all)}
    qrd = {x: q_random[i] for i, x in enumerate(ids_all)}
    rrd = {x: r_random[i] for i, x in enumerate(ids_all)}

    import torch
    cache_path = ROOT / folds_obj["split_assertions"]["clip_cache"]["train"]["path"]
    if sha256_file(cache_path) != folds_obj["split_assertions"]["clip_cache"]["train"]["sha256"]:
        raise RuntimeError("frozen train cache hash changed")
    cache = torch.load(cache_path, map_location="cpu")
    cache_ids = cache["ids"]
    if cache_ids and isinstance(cache_ids[0], (list, tuple)):
        cache_ids = [str(x) for b in cache_ids for x in b]
    else:
        cache_ids = [str(x) for x in cache_ids]
    row_cache = {x: i for i, x in enumerate(cache_ids)}
    img = normalize_rows(torch.as_tensor(cache["img_feats"]).numpy())
    txt = normalize_rows(torch.as_tensor(cache["text_feats"]).numpy())

    edge_rows, anchor_rows, exposure_examples = [], [], []
    parity_queries = 0; max_cos_err = max_vote_err = 0.0
    parity_ids = parity_preds = True
    repo_logging = {}; repo_labels = []; exact_preds = []
    alignment_records = []
    anchor_top_ids = {}
    for fold in range(5):
        fdir = resolve(cfg, "ssr_artifacts") / "oof" / dataset / ("fold{}".format(fold))
        npz = np.load(fdir / "embeddings.npz")
        mem_ids = [str(x) for x in npz["memory_ids"].tolist()]
        qry_ids = [str(x) for x in npz["query_ids"].tolist()]
        mem_z = np.asarray(npz["memory_z"], dtype=np.float32)
        qry_z = np.asarray(npz["query_z"], dtype=np.float32)
        mem_y = np.asarray(npz["memory_labels"], dtype=np.int64)
        qry_y = np.asarray(npz["query_labels"], dtype=np.int64)
        if set(mem_ids) & set(qry_ids) or set(mem_ids) | set(qry_ids) != set(ids_all):
            raise RuntimeError("OOF partition invalid fold {}".format(fold))
        stored_rank = {x["query_id"]: x for x in read_jsonl(fdir / "ranking.jsonl")}
        stored_pred = {x["query_id"]: x for x in read_json(fdir / "predictions.json")}
        bc_mem, _, centers = base_cluster_q(mem_z, seed=20260711)
        bc_qry = soft_cluster_for(qry_z, centers)
        mem_q = np.asarray([q_global[x] for x in mem_ids]); mem_r = np.asarray([r_global[x] for x in mem_ids])
        msh_q = np.asarray([qsh[x] for x in mem_ids]); msh_r = np.asarray([rsh[x] for x in mem_ids])
        mrd_q = np.asarray([qrd[x] for x in mem_ids]); mrd_r = np.asarray([rrd[x] for x in mem_ids])
        for qi, aid in enumerate(qry_ids):
            full = canonical_full_order(mem_ids, mem_z, qry_z[qi])
            top_exact, vote, pred, denom = exact_ranking(
                mem_ids, mem_z, mem_y, qry_z[qi],
                topk=cfg["evaluator"]["topk"])
            stored = stored_rank[aid]["ranking"]
            if [x["id"] for x in full] != [str(x["id"]) for x in stored]:
                parity_ids = False
            for a, b in zip(full, stored):
                max_cos_err = max(max_cos_err, abs(a["cosine"] - float(b["cosine"])))
            max_vote_err = max(max_vote_err, abs(vote - float(stored_pred[aid]["vote"])))
            parity_preds &= pred == int(stored_pred[aid]["prediction"])
            parity_queries += 1
            top = add_exposure(top_exact, int(qry_y[qi]))
            repo_logging[aid] = {
                "no_retrieved": 20,
                "retrieved_ids": [x["id"] for x in top_exact],
                "retrieved_scores": [np.float32(x["cosine"]) for x in top_exact],
                "retrieved_label": [x["label"] for x in top_exact],
            }
            repo_labels.append(int(qry_y[qi])); exact_preds.append(pred)
            anchor_top_ids[aid] = [x["id"] for x in top]
            margin = (2 * int(qry_y[qi]) - 1) * vote
            q_i = q_global[aid]; r_i = r_global[aid]
            sh_i = qsh[aid]; sr_i = rsh[aid]
            rd_i = qrd[aid]; rr_i = rrd[aid]
            ewrong = []; ecorr = []; aff_wrong = []; aff_corr = []
            for x in top:
                j = mem_ids.index(x["id"])
                qaff = float(posterior_affinity(q_i, mem_q[j]))
                shaff = float(posterior_affinity(sh_i, msh_q[j]))
                rdaff = float(posterior_affinity(rd_i, mrd_q[j]))
                bcaff = float(posterior_affinity(bc_qry[qi], bc_mem[j]))
                ci = row_cache[aid]; cj = row_cache[x["id"]]
                row = {
                    "anchor_id": aid, "neighbor_id": x["id"], "outer_fold": fold,
                    "video_class": int(qry_y[qi]), "neighbor_class": int(x["label"]),
                    "wrong_class": int(x["label"] != int(qry_y[qi])),
                    "cosine": x["cosine"], "rank": x["rank"],
                    "anchor_class": int(qry_y[qi]),
                    "normalized_vote_margin": float(margin / max(denom, 1e-12)),
                    "image_cosine": float(img[ci] @ img[cj]),
                    "text_cosine": float(txt[ci] @ txt[cj]),
                    "base_cluster_affinity": bcaff, "posterior_affinity": qaff,
                    "shuffle_affinity": shaff, "random_affinity": rdaff,
                    "weight": x["weight"], "signed_contribution": x["signed_contribution"],
                    "exposure": x["exposure"],
                }
                for rnk in range(1, 21):
                    row["rank_{}".format(rnk)] = int(x["rank"] == rnk)
                edge_rows.append(row)
                if row["wrong_class"]:
                    ewrong.append(row); aff_wrong.append(qaff)
                else:
                    ecorr.append(row); aff_corr.append(qaff)
            a_full, ep, en = anchor_alignment(
                qry_z[qi], qry_y[qi], mem_ids, mem_z, mem_y,
                q_i, r_i, mem_q, mem_r, cfg, aid, fold)
            a_shuf, _, _ = anchor_alignment(
                qry_z[qi], qry_y[qi], mem_ids, mem_z, mem_y,
                sh_i, sr_i, msh_q, msh_r, cfg, aid, fold)
            a_rand, _, _ = anchor_alignment(
                qry_z[qi], qry_y[qi], mem_ids, mem_z, mem_y,
                rd_i, rr_i, mrd_q, mrd_r, cfg, aid, fold)
            pressure = sum(x["exposure"] * x["posterior_affinity"] for x in ewrong)
            anchor = {
                "anchor_id": aid, "outer_fold": fold, "video_class": int(qry_y[qi]),
                "baseline_error": int(pred != int(qry_y[qi])),
                "normalized_vote_margin": float(margin / max(denom, 1e-12)),
                "mean_wrong_cosine": float(np.mean([x["cosine"] for x in ewrong])) if ewrong else 0.0,
                "mean_wrong_rank": float(np.mean([x["rank"] for x in ewrong])) if ewrong else 21.0,
                "mean_base_cluster_affinity": float(np.mean(
                    [x["base_cluster_affinity"] for x in edge_rows[-20:]])),
                "mean_image_cosine": float(np.mean([x["image_cosine"] for x in edge_rows[-20:]])),
                "mean_text_cosine": float(np.mean([x["text_cosine"] for x in edge_rows[-20:]])),
                "quotient_pressure": float(pressure),
                "positive_ess": float(ep), "negative_ess": float(en),
                "alignment_full": a_full, "alignment_shuffle": a_shuf,
                "alignment_random": a_rand,
                "has_both_neighbor_outcomes": bool(ewrong and ecorr),
            }
            anchor_rows.append(anchor)
            if a_full is not None and a_shuf is not None and a_rand is not None:
                alignment_records.append(anchor)
            if len(exposure_examples) < 25:
                exposure_examples.append({"anchor_id": aid, "query_label": int(qry_y[qi]),
                                          "outer_fold": fold, "top20": top})

    syn = synthetic_parity(cfg)
    from utils.metrics import compute_metrics_retrieval
    repo_result = compute_metrics_retrieval(
        repo_logging, np.asarray(repo_labels), majority_voting="arithmetic",
        topk=cfg["evaluator"]["topk"], use_sim=True)
    exact_metrics = metrics_from_predictions(repo_labels, exact_preds)
    repo_metric_match = (abs(float(repo_result[0]) - exact_metrics["accuracy"]) <= 1e-12 and
                         abs(float(repo_result[7]["macro_f1"]) - exact_metrics["macro_f1"]) <= 1e-12)
    tol = cfg["evaluator"]["parity_abs_tolerance"]
    parity = {"synthetic": syn, "actual_queries": parity_queries,
              "canonical_id_ranking_match": parity_ids,
              "prediction_match": parity_preds, "max_cosine_error": max_cos_err,
              "max_vote_error": max_vote_err,
              "accuracy_macro_f1_recomputed": exact_metrics,
              "repository_accuracy": float(repo_result[0]),
              "repository_macro_f1": float(repo_result[7]["macro_f1"]),
              "repository_metric_match": repo_metric_match,
              "pass": bool(syn["pass"] and parity_ids and parity_preds and
                           repo_metric_match and max_cos_err <= tol and max_vote_err <= tol)}

    # Positivity and two-sided ESS use the complete train-OOF anchor set.
    cells = {}
    for e in range(6):
        for c in (0, 1):
            w = r_all[y_all == c] * q_all[y_all == c, e]
            cells["e{}_c{}".format(e, c)] = {"mass": float(w.sum()), "ess": kish_ess(w)}
    effective_env = {}
    for c in (0, 1):
        mass = np.asarray([cells["e{}_c{}".format(e, c)]["mass"] for e in range(6)])
        p = mass / max(mass.sum(), 1e-12)
        effective_env[str(c)] = float(1.0 / max(np.square(p).sum(), 1e-12))
    active_by_class = {}
    for c in (0, 1):
        sub = [x for x in anchor_rows if x["video_class"] == c]
        active_by_class[str(c)] = sum(
            x["positive_ess"] >= cfg["sq"]["min_kish_ess"] and
            x["negative_ess"] >= cfg["sq"]["min_kish_ess"] for x in sub) / len(sub)
    positivity_pass = (all(x["mass"] >= cfg["s0_gates"]["cell_mass_min"] and
                           x["ess"] >= cfg["s0_gates"]["cell_ess_min"] for x in cells.values()) and
                       all(x >= cfg["s0_gates"]["effective_environment_min"] for x in effective_env.values()) and
                       all(x >= cfg["s0_gates"]["active_anchor_fraction_min"] for x in active_by_class.values()))

    edge_base = ["cosine"] + ["rank_{}".format(i) for i in range(1, 21)] + [
        "anchor_class", "normalized_vote_margin", "image_cosine", "text_cosine",
        "base_cluster_affinity"]
    edge_pred, edge_fold_delta, edge_delta = fit_predict_rotations(
        edge_rows, edge_base, edge_base + ["posterior_affinity"], "wrong_class")
    edge_boot = stratified_anchor_bootstrap_auc(
        edge_pred, cfg["evaluator"]["bootstrap_replicates"],
        cfg["evaluator"]["bootstrap_seed"])
    anchor_base = ["video_class", "normalized_vote_margin", "mean_wrong_cosine",
                   "mean_wrong_rank", "mean_base_cluster_affinity",
                   "mean_image_cosine", "mean_text_cosine"]
    error_pred, error_fold_delta, error_delta = fit_predict_rotations(
        anchor_rows, anchor_base, anchor_base + ["quotient_pressure"], "baseline_error")
    error_boot = stratified_anchor_bootstrap_auc(
        error_pred, cfg["evaluator"]["bootstrap_replicates"],
        cfg["evaluator"]["bootstrap_seed"] + 1)
    align_full = np.asarray([x["alignment_full"] for x in alignment_records])
    align_shuffle = np.asarray([x["alignment_shuffle"] for x in alignment_records])
    align_random = np.asarray([x["alignment_random"] for x in alignment_records])
    align_delta_shuffle = float(np.mean(align_full - align_shuffle)) if len(align_full) else -1.0
    align_delta_random = float(np.mean(align_full - align_random)) if len(align_full) else -1.0

    # Anchor bootstrap for alignment deltas, stratified by fold x class.
    strata = defaultdict(list)
    for i, x in enumerate(alignment_records):
        strata[(x["outer_fold"], x["video_class"])].append(i)
    rng = np.random.default_rng(cfg["evaluator"]["bootstrap_seed"] + 2)
    boot_s, boot_r = [], []
    for _ in range(cfg["evaluator"]["bootstrap_replicates"]):
        idx = []
        for vals in strata.values():
            idx.extend(rng.choice(vals, size=len(vals), replace=True).tolist())
        boot_s.append(float(np.mean(align_full[idx] - align_shuffle[idx])))
        boot_r.append(float(np.mean(align_full[idx] - align_random[idx])))
    alignment = {
        "n_active": len(alignment_records),
        "full_rate": float(np.mean(align_full)) if len(align_full) else 0.0,
        "shuffle_rate": float(np.mean(align_shuffle)) if len(align_full) else 0.0,
        "random_rate": float(np.mean(align_random)) if len(align_full) else 0.0,
        "delta_shuffle": align_delta_shuffle, "delta_random": align_delta_random,
        "shuffle_lower_95": float(np.percentile(boot_s, 2.5)) if boot_s else -1.0,
        "random_lower_95": float(np.percentile(boot_r, 2.5)) if boot_r else -1.0,
    }
    p0_pass = (edge_delta >= cfg["s0_gates"]["p0_auc_delta_min"] and
               all(x > 0 for x in edge_fold_delta) and edge_boot["lower_95"] > 0 and
               error_delta >= cfg["s0_gates"]["p0_auc_delta_min"] and
               all(x > 0 for x in error_fold_delta) and error_boot["lower_95"] > 0 and
               align_delta_shuffle >= cfg["s0_gates"]["p0_alignment_delta_min"] and
               align_delta_random >= cfg["s0_gates"]["p0_alignment_delta_min"] and
               alignment["shuffle_lower_95"] > 0 and alignment["random_lower_95"] > 0)

    # Power preregistration on valid anchor estimands.
    from sklearn.metrics import roc_auc_score
    power_cells = {}
    requirements = {}
    for c in (0, 1):
        auc_vals = []; align_vals = []; valid_ids = []
        for a in [x for x in anchor_rows if x["video_class"] == c]:
            edges = [x for x in edge_rows if x["anchor_id"] == a["anchor_id"]]
            yy = np.asarray([x["wrong_class"] for x in edges])
            if len(np.unique(yy)) == 2:
                auc_vals.append(float(roc_auc_score(yy, [x["posterior_affinity"] for x in edges]) -
                                      roc_auc_score(yy, [x["shuffle_affinity"] for x in edges])))
                valid_ids.append(a["anchor_id"])
            if a["alignment_full"] is not None and a["alignment_shuffle"] is not None:
                align_vals.append(float(a["alignment_full"] - a["alignment_shuffle"]))
        acell = upper_variance_and_n(auc_vals, cfg["power"]["delta_auc"], len(auc_vals), cfg)
        lcell = upper_variance_and_n(align_vals, cfg["power"]["delta_align"], len(align_vals), cfg)
        power_cells["class{}_auc".format(c)] = acell
        power_cells["class{}_align".format(c)] = lcell
        reqs = [x["n_FPC"] for x in (acell, lcell) if x["n_FPC"] is not None]
        requirements[str(c)] = max(reqs) if len(reqs) == 2 else None
    selected_anchor_ids = []
    power_strata = {}
    achieved_power = {}
    for c in (0, 1):
        candidates = [x for x in anchor_rows if x["video_class"] == c and
                      x["has_both_neighbor_outcomes"] and
                      x["alignment_full"] is not None and
                      x["alignment_shuffle"] is not None]
        nreq = requirements[str(c)]
        # Frozen class x OOF-margin-quartile proportional allocation.
        ordered_margin = sorted(candidates, key=lambda x: (
            x["normalized_vote_margin"], x["anchor_id"]))
        strata = defaultdict(list)
        for pos, row in enumerate(ordered_margin):
            quart = min(3, (4 * pos) // max(1, len(ordered_margin)))
            strata[quart].append(row)
        selected_c = []
        alloc = {}
        if nreq is not None and nreq <= len(candidates) and all(strata[q] for q in range(4)):
            raw = {q: nreq * len(strata[q]) / len(candidates) for q in range(4)}
            alloc = {q: min(len(strata[q]), int(math.floor(raw[q]))) for q in range(4)}
            remain = nreq - sum(alloc.values())
            for q in sorted(range(4), key=lambda q: (-(raw[q] - math.floor(raw[q])), q)):
                if remain and alloc[q] < len(strata[q]):
                    alloc[q] += 1; remain -= 1
            if remain:
                for q in range(4):
                    take = min(remain, len(strata[q]) - alloc[q]); alloc[q] += take; remain -= take
                    if not remain: break
            for q in range(4):
                ranked = sorted(strata[q], key=lambda x: sha256_text(
                    "power|{}|{}|{}|{}".format(dataset, c, q, x["anchor_id"])))
                selected_c.extend(x["anchor_id"] for x in ranked[:alloc[q]])
        selected_anchor_ids.extend(selected_c)
        power_strata[str(c)] = {"available": {str(q): len(strata[q]) for q in range(4)},
                                "allocated": {str(q): alloc.get(q, 0) for q in range(4)},
                                "selected": len(selected_c), "required": nreq,
                                "sampleable": bool(nreq is not None and len(selected_c) == nreq)}
        from scipy.stats import norm
        for name in ("auc", "align"):
            cell = power_cells["class{}_{}".format(c, name)]
            if cell["variance_U"] is None or not selected_c or len(selected_c) >= cell["N"]:
                pwr = 1.0 if cell["variance_U"] is not None and len(selected_c) == cell["N"] else 0.0
            else:
                se = math.sqrt(cell["variance_U"] / len(selected_c) *
                               (cell["N"] - len(selected_c)) / (cell["N"] - 1))
                pwr = float(norm.cdf(cell["delta"] / max(se, 1e-12) -
                                     norm.ppf(1 - cfg["power"]["alpha_star"] / 2)))
            achieved_power["class{}_{}".format(c, name)] = pwr
    closure = set(selected_anchor_ids)
    for aid in selected_anchor_ids:
        closure.update(anchor_top_ids[aid])
    power_status = ("PASS" if all(x["status"] == "PASS" for x in power_cells.values()) and
                    all(x["sampleable"] for x in power_strata.values()) and
                    all(x >= cfg["power"]["power"] for x in achieved_power.values()) and
                    len(closure) <= cfg["power"]["closure_cap_per_dataset"] else "STOP_INFEASIBLE")
    power = {"cells": power_cells, "class_requirements": requirements,
             "selected_anchor_count": len(selected_anchor_ids),
             "strata": power_strata, "achieved_fpc_power": achieved_power,
             "closure_upper_bound": len(closure),
             "closure_cap": cfg["power"]["closure_cap_per_dataset"],
             "status": power_status, "selection_is_preregistration_only": True,
             "new_teacher_call_count": 0}

    epath = out_dir / "edge_ledger.jsonl"; apath = out_dir / "anchor_ledger.jsonl"
    xpath = out_dir / "exposure_examples.jsonl"; ppath = out_dir / "power.json"
    mpath = out_dir / "metrics.json"
    exclusive_write_jsonl(epath, edge_rows); exclusive_write_jsonl(apath, anchor_rows)
    exclusive_write_jsonl(xpath, exposure_examples); exclusive_write_json(ppath, power)
    metrics = {
        "dataset": dataset, "status": "PASS" if parity["pass"] and positivity_pass and p0_pass else "STOP",
        "parity": parity,
        "positivity": {"cells": cells, "effective_environment_count": effective_env,
                       "active_anchor_fraction_by_class": active_by_class,
                       "pass": positivity_pass},
        "edge_auc": {"delta": edge_delta, "fold_deltas": edge_fold_delta,
                     "bootstrap": edge_boot},
        "error_auc": {"delta": error_delta, "fold_deltas": error_fold_delta,
                      "bootstrap": error_boot},
        "alignment": alignment, "p0_pass": p0_pass,
        "random_control_calibration": random_diag,
        "power_status": power_status,
        "only_gold_supervision": "parent_video_binary_label",
        "segment_gold_exists": False, "segment_gold_used": False,
        "new_teacher_call_count": 0, "teacher_cache_read_count": 0,
        "teacher_cache_write_count": 0,
    }
    finite_or_raise(metrics)
    exclusive_write_json(mpath, metrics)
    outputs = [epath, apath, xpath, ppath, mpath]
    inputs = [q_dir(cfg, dataset) / "posterior.jsonl", fold_path(cfg, dataset), cache_path]
    for f in range(5):
        fd = resolve(cfg, "ssr_artifacts") / "oof" / dataset / ("fold{}".format(f))
        inputs.extend([fd / "embeddings.npz", fd / "ranking.jsonl", fd / "predictions.json"])
    manifest = base_manifest(
        cfg, run_id, "S0_PARITY_POWER_P0", metrics["status"],
        inputs=[input_record(x) for x in inputs], outputs=output_records(outputs),
        extra={"dataset": dataset, "fold_ids_sha256": sha256_obj(
                   [{"id": x["id"], "fold": x["fold"]} for x in records]),
               "actual_top20_rows": len(edge_rows), "p0_pass": p0_pass,
               "power_status": power_status})
    exclusive_write_json(out_dir / "manifest.json", manifest)
    print(canonical_json({"run_id": run_id, "status": metrics["status"],
                          "p0_pass": p0_pass, "edge_delta": edge_delta,
                          "error_delta": error_delta, "power": power_status}))


def tensor_grad_norm(parameters):
    import torch
    vals = [p.grad.detach().float().norm().square() for p in parameters
            if p.grad is not None]
    return float(torch.sqrt(torch.stack(vals).sum()).cpu()) if vals else 0.0


def task_micro(cfg, dataset, run_id):
    ensure_run_id(run_id, "SQ-S0-MICRO-{}-S0-v1".format(dataset))
    out_dir = artifact_root(cfg) / "s0" / "micro" / dataset
    acquire_namespace(out_dir, run_id)
    import torch
    from torch.utils.data import DataLoader
    from easydict import EasyDict
    from data_loader.rac_dataloader import RACDataset
    from model.classifier import classifier_hateClipper
    from model.loss import compute_loss
    from ssr_oof import load_train_cache, make_segment_cache, take_dataset, train_args

    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    folds, records = fold_records(cfg, dataset)
    ssr_cfg = dict(cfg); ssr_cfg["comparator"] = cfg["training"]
    cache_path, full_ids, img, txt, labels = load_train_cache(ssr_cfg, dataset)
    memory_ids = sorted(x["id"] for x in records if int(x["fold"]) != 0)
    memory, memory_idx = take_dataset(full_ids, img, txt, labels, memory_ids)
    train_set = RACDataset((memory[1], memory[2]), memory[0], memory[3])
    train_dl = DataLoader(train_set, batch_size=cfg["training"]["batch_size"],
                          shuffle=False, num_workers=0)
    seg_path, segment_cache, _ = make_segment_cache(
        ssr_cfg, dataset, full_ids, labels, memory, memory_idx)
    args = train_args(ssr_cfg, dataset)
    model = classifier_hateClipper(
        int(img.shape[1]), int(txt.shape[1]), cfg["training"]["num_layers"],
        cfg["training"]["proj_dim"], cfg["training"]["map_dim"],
        cfg["training"]["fusion_mode"], dropout=cfg["training"]["dropout"],
        batch_norm=cfg["training"]["batch_norm"], args=args).cuda()
    init_hash = sha256_obj({k: sha256_text(v.detach().cpu().numpy().tobytes().hex())
                            for k, v in model.state_dict().items()})

    @torch.no_grad()
    def project_bank():
        model.eval(); out = []
        for start in range(0, len(memory_ids), 256):
            _, z = model(memory[1][start:start + 256].cuda(),
                         memory[2][start:start + 256].cuda(), return_embed=True)
            out.append(z.detach())
        return torch.cat(out, dim=0)

    bank = project_bank()
    qrows, qby = load_q(cfg, dataset)
    q = np.asarray([qby[x]["q"] for x in memory_ids]); r = np.asarray([qby[x]["r"] for x in memory_ids])
    y = np.asarray(memory[3], dtype=np.int64)
    plans, plan_stats = sq_sampling_plan(
        memory_ids, y, bank.cpu().numpy(), q, r, cfg["computed_config_sha256"],
        cfg["sq"]["seed"], 0, cfg["sq"]["triplets_per_anchor"],
        cfg["sq"]["min_kish_ess"], cfg["evaluator"]["topk"])
    first_batch = next(iter(train_dl))
    first_ids = [str(x) for x in first_batch["ids"]]

    # Prime repository full-bank retrieval cache once, matching steady-state steps.
    model.train()
    primed = compute_loss(first_batch, train_dl, model, args, train_set=train_set,
                          train_feats=None, train_labels=None,
                          segment_cache=segment_cache, aux_pack=None, cf_pack=None)
    cached_feats = primed[-2].detach(); cached_labels = primed[-1].detach()

    def run_step(full):
        model.zero_grad(set_to_none=True)
        out = compute_loss(first_batch, train_dl, model, args, train_set=train_set,
                           train_feats=cached_feats, train_labels=cached_labels,
                           segment_cache=segment_cache, aux_pack=None, cf_pack=None)
        loss = out[0]
        if full:
            _, z = model(first_batch["image_feats"].cuda(),
                         first_batch["text_feats"].cuda(), return_embed=True)
            aux, _ = sq_loss_for_batch(first_ids, z, bank, plans,
                                       cfg["sq"]["margin"], cfg["sq"]["temperature"])
            loss = loss + 0.1 * aux
        loss.backward()
        return float(loss.detach().cpu())

    timings = {}
    torch.cuda.reset_peak_memory_stats()
    for name, full in (("REMOVE", False), ("FULL", True)):
        vals = []
        for it in range(220):
            torch.cuda.synchronize(); start = time.perf_counter()
            val = run_step(full)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            if it >= 20:
                vals.append(elapsed)
        timings[name] = {"median_step_s": float(np.median(vals)),
                         "p95_step_s": float(np.percentile(vals, 95)),
                         "iterations": 200, "warmups": 20,
                         "last_loss": val}
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    ratio = timings["FULL"]["median_step_s"] / timings["REMOVE"]["median_step_s"]
    steps = math.ceil(len(memory_ids) / cfg["training"]["batch_size"])
    epochs = cfg["datasets"][dataset]["epoch_index"] + 1
    fold_hours = timings["FULL"]["median_step_s"] * steps * epochs * 6 / 3600.0
    timings.update({"full_remove_ratio": ratio, "peak_allocated_gib": peak_gib,
                    "estimated_s1_fold_gpu_hours_six_arms": fold_hours,
                    "estimated_s1_dataset_gpu_hours_five_folds": fold_hours * 5})

    # Actual parameter first-step base/aux gradient ratio.
    model.zero_grad(set_to_none=True)
    bout = compute_loss(first_batch, train_dl, model, args, train_set=train_set,
                        train_feats=cached_feats, train_labels=cached_labels,
                        segment_cache=segment_cache, aux_pack=None, cf_pack=None)
    bout[0].backward(); base_norm = tensor_grad_norm(model.parameters())
    model.zero_grad(set_to_none=True)
    _, az = model(first_batch["image_feats"].cuda(), first_batch["text_feats"].cuda(), return_embed=True)
    aux, active = sq_loss_for_batch(first_ids, az, bank, plans,
                                    cfg["sq"]["margin"], cfg["sq"]["temperature"])
    aux.backward(); aux_norm = tensor_grad_norm(model.parameters())
    lambda_ratios = {str(lam): float(lam * aux_norm / max(base_norm, 1e-12))
                     for lam in cfg["sq"]["lambda_candidates"]}

    # Five-fold control-strength audit using the exact auxiliary gradient w.r.t.
    # normalized anchor embeddings.  No outcomes tune these scalars.
    control_folds = []
    for fold in range(5):
        fdir = resolve(cfg, "ssr_artifacts") / "oof" / dataset / ("fold{}".format(fold))
        zf = np.load(fdir / "embeddings.npz")
        mids = [str(x) for x in zf["memory_ids"].tolist()]
        mz = np.asarray(zf["memory_z"], dtype=np.float32)
        my = np.asarray(zf["memory_labels"], dtype=np.int64)
        fq = np.asarray([qby[x]["q"] for x in mids]); fr = np.asarray([qby[x]["r"] for x in mids])
        sh_q, sh_r, _ = make_shuffle_q(mids, my, fq, fr, cfg["sq"]["shuffle_seed"])
        rd_q, rd_r, _ = random_matched_q(mids, fq, fr, cfg["sq"]["random_seed"])
        bc_q, bc_r, _ = base_cluster_q(mz)
        arms = {"FULL": (fq, fr, False), "BASE_CLUSTER": (bc_q, bc_r, False),
                "LABEL_ONLY": (fq, fr, True), "SHUFFLE": (sh_q, sh_r, False),
                "RANDOM": (rd_q, rd_r, False)}
        batch_ids = mids[:64]
        norms = {}; active_counts = {}
        for arm, (aq, ar, label_only) in arms.items():
            aplans, _ = sq_sampling_plan(
                mids, my, mz, aq, ar, cfg["computed_config_sha256"], 0, 0,
                cfg["sq"]["triplets_per_anchor"], cfg["sq"]["min_kish_ess"],
                cfg["evaluator"]["topk"], label_only=label_only)
            anchor = torch.as_tensor(mz[:64], device="cuda", dtype=torch.float32).requires_grad_(True)
            bl = torch.as_tensor(mz, device="cuda", dtype=torch.float32)
            loss, ac = sq_loss_for_batch(batch_ids, anchor, bl, aplans,
                                         cfg["sq"]["margin"], cfg["sq"]["temperature"])
            grad = torch.autograd.grad(loss, anchor, allow_unused=False)[0]
            norms[arm] = float(grad.norm().detach().cpu()); active_counts[arm] = ac
        full_norm = norms["FULL"]
        scalars = {}; caps = {}
        for arm in ("BASE_CLUSTER", "LABEL_ONLY", "SHUFFLE", "RANDOM"):
            raw = full_norm / max(norms[arm], 1e-12)
            scalars[arm] = float(min(2.0, max(0.5, raw)))
            caps[arm] = bool(raw < 0.5 or raw > 2.0)
        control_folds.append({"fold": fold, "gradient_norms": norms,
                              "strength_scalars": scalars, "cap_activated": caps,
                              "active_counts": active_counts})
    cap_fraction = sum(any(x["cap_activated"].values()) for x in control_folds) / 5.0

    # Scalar float64 vs vectorized float32 loss/exposure and autograd parity.
    rng = np.random.default_rng(20260711)
    a64 = rng.normal(size=(16, 32)); p64 = rng.normal(size=(16, 32)); n64 = rng.normal(size=(16, 32))
    a64 /= np.linalg.norm(a64, axis=1, keepdims=True); p64 /= np.linalg.norm(p64, axis=1, keepdims=True); n64 /= np.linalg.norm(n64, axis=1, keepdims=True)
    e64 = rng.uniform(0.01, 20.0, size=16)
    scalar = np.asarray([e64[i] * math.log1p(math.exp(
        (float(a64[i] @ n64[i]) - float(a64[i] @ p64[i]) + 0.1) / 0.1)) for i in range(16)])
    at = torch.tensor(a64, dtype=torch.float64, requires_grad=True)
    pt = torch.tensor(p64, dtype=torch.float64); nt = torch.tensor(n64, dtype=torch.float64)
    et = torch.tensor(e64, dtype=torch.float64)
    lv64 = et * torch.nn.functional.softplus(((at * nt).sum(1) - (at * pt).sum(1) + 0.1) / 0.1)
    g64 = torch.autograd.grad(lv64.mean(), at)[0].detach().numpy()
    a32 = torch.tensor(a64, dtype=torch.float32, requires_grad=True)
    p32 = torch.tensor(p64, dtype=torch.float32); n32 = torch.tensor(n64, dtype=torch.float32)
    e32 = torch.tensor(e64, dtype=torch.float32)
    lv32 = e32 * torch.nn.functional.softplus(((a32 * n32).sum(1) - (a32 * p32).sum(1) + 0.1) / 0.1)
    g32 = torch.autograd.grad(lv32.mean(), a32)[0].detach().numpy().astype(np.float64)
    loss_err = float(np.max(np.abs(scalar - lv32.detach().numpy())))
    grad_rel = float(np.linalg.norm(g64 - g32) / max(np.linalg.norm(g64), 1e-12))
    numerics = {"max_loss_abs_error": loss_err, "gradient_relative_error": grad_rel,
                "all_finite": bool(np.all(np.isfinite(g32)) and np.all(np.isfinite(scalar))),
                "loss_tolerance": cfg["evaluator"]["parity_abs_tolerance"],
                "gradient_tolerance": cfg["evaluator"]["gradient_rel_tolerance"]}
    numerics["pass"] = bool(numerics["all_finite"] and loss_err <= numerics["loss_tolerance"] and grad_rel <= numerics["gradient_tolerance"])
    resource_pass = (peak_gib <= cfg["s0_gates"]["micro_peak_gib_max"] and
                     ratio <= cfg["s0_gates"]["micro_full_remove_ratio_max"] and
                     cap_fraction <= cfg["s0_gates"]["strength_cap_fold_fraction_max"])
    status = "PASS" if numerics["pass"] and resource_pass and aux_norm > 0 else "STOP"
    timing_path = out_dir / "timings.json"; numeric_path = out_dir / "numerics.json"
    exposure_path = out_dir / "exposure_examples.jsonl"
    exclusive_write_json(timing_path, {"status": status, "timings": timings,
                                       "base_gradient_norm": base_norm,
                                       "aux_gradient_norm": aux_norm,
                                       "lambda_gradient_ratios": lambda_ratios,
                                       "plan_stats": plan_stats,
                                       "control_folds": control_folds,
                                       "cap_activation_fold_fraction": cap_fraction,
                                       "resource_pass": resource_pass})
    exclusive_write_json(numeric_path, numerics)
    examples = []
    for vid in first_ids[:10]:
        p = plans[vid]
        examples.append({"anchor_id": vid, "active": p["active"],
                         "ess_pos": p["ess_pos"], "ess_neg": p["ess_neg"],
                         "exposed_negatives": p["exposed_negatives"],
                         "draws": p.get("draws", [])[:3]})
    exclusive_write_jsonl(exposure_path, examples)
    outputs = [timing_path, numeric_path, exposure_path]
    manifest = base_manifest(
        cfg, run_id, "S0_MICRO", status,
        inputs=[input_record(q_dir(cfg, dataset) / "posterior.jsonl"),
                input_record(cache_path), input_record(seg_path),
                input_record(fold_path(cfg, dataset))],
        outputs=output_records(outputs), gpu_name=torch.cuda.get_device_name(0),
        extra={"dataset": dataset, "initialization_sha256": init_hash,
               "numeric_pass": numerics["pass"], "resource_pass": resource_pass,
               "estimated_fold_gpu_hours": fold_hours})
    exclusive_write_json(out_dir / "manifest.json", manifest)
    print(canonical_json({"run_id": run_id, "status": status,
                          "full_remove_ratio": ratio, "peak_gib": peak_gib,
                          "cap_fraction": cap_fraction}))


def task_decide(cfg, run_id):
    ensure_run_id(run_id, "SQ-S0-DECISION-v1")
    out = artifact_root(cfg) / "S0_DECISION.json"
    if out.exists():
        raise RuntimeError("refusing overwrite {}".format(out))
    required = [artifact_root(cfg) / "CONFIG_FREEZE.json",
                artifact_root(cfg) / "s0/audit/sample_manifest.json",
                artifact_root(cfg) / "s0/audit/manifest.json"]
    for d in DATASETS:
        required.extend([
            artifact_root(cfg) / "s0/provenance" / d / "provenance.json",
            q_dir(cfg, d) / "manifest.json",
            q_dir(cfg, d) / "posterior.jsonl",
            artifact_root(cfg) / "s0/audit" / (d + ".csv"),
        ])
    missing = [str(x.relative_to(ROOT)) for x in required if not x.is_file()]
    if missing:
        raise RuntimeError("S0 decision missing inputs: {}".format(missing))
    rehash = [{"path": str(x.relative_to(ROOT)), "sha256": sha256_file(x)} for x in required]
    freeze = read_json(required[0])
    dataset_gates = {}
    signal_statuses = []
    audit_sample = read_json(artifact_root(cfg) / "s0/audit/sample_manifest.json")
    freeze_recomputed = {
        "producer_status": freeze.get("status"),
        "config_hash_matches": freeze.get("config_canonical_sha256") == cfg["computed_config_sha256"],
        "implementation_hash_matches": freeze.get("implementation_sha256") == implementation_hash(),
        "archive_hashes_match": all(
            sha256_file(ROOT / cfg["datasets"][d]["archive"]) ==
            cfg["datasets"][d]["archive_sha256"] for d in DATASETS),
    }
    freeze_recomputed["pass"] = all(v for k, v in freeze_recomputed.items()
                                        if k != "producer_status") and freeze.get("status") == "PASS"
    for d in DATASETS:
        prov = read_json(artifact_root(cfg) / "s0/provenance" / d / "provenance.json")
        qm = read_json(q_dir(cfg, d) / "manifest.json")
        qrows = read_jsonl(q_dir(cfg, d) / "posterior.jsonl")
        signal_statuses.append(prov["q_signal_status"])
        q_numeric_valid = (len(qrows) == len({x["id"] for x in qrows}) and
                           all(len(x.get("q", [])) == 6 and
                               abs(sum(float(v) for v in x["q"]) - 1.0) <= 1e-6 and
                               all(math.isfinite(float(v)) and float(v) >= 0 for v in x["q"]) and
                               math.isfinite(float(x["r"])) and 0 <= float(x["r"]) <= 1
                               for x in qrows))
        q_output_hash = next((x["sha256"] for x in qm.get("output_files", [])
                              if x["path"].endswith("posterior.jsonl")), None)
        audit_result = artifact_root(cfg) / "s0/audit" / (d + "_result.json")
        audit_completed = audit_result.is_file()
        audit_pass = False
        if audit_completed:
            ar = read_json(audit_result)
            audit_pass = (ar.get("status") == "PASS" and ar.get("n") == 64 and
                          ar.get("contaminated", 65) <= 3 and
                          ar.get("appropriateness_wilson_lower", 0) >= 0.90 and
                          ar.get("qc_not_gold_supervision") is True)
            required.append(audit_result)
        checks = {
            "freeze": freeze_recomputed["pass"],
            "reader_valid": (prov.get("id_set_valid") is True and
                             prov.get("archive_forbidden_key_access_count") == 0),
            "original_provenance_complete": bool(
                prov.get("original_run_linkage", {}).get("cryptographically_complete")),
            "q_is_numeric_and_id_unique": q_numeric_valid,
            "q_output_hash_matches": q_output_hash == sha256_file(q_dir(cfg, d) / "posterior.jsonl"),
            "blind_presentation_audit_completed": audit_completed,
            "blind_presentation_audit_pass": audit_pass,
            "zero_calls": (prov.get("new_teacher_call_count") == 0 and
                           prov.get("teacher_cache_read_count") == 0 and
                           prov.get("teacher_cache_write_count") == 0 and
                           qm.get("new_teacher_call_count") == 0 and
                           qm.get("teacher_cache_read_count") == 0 and
                           qm.get("teacher_cache_write_count") == 0),
        }
        dataset_gates[d] = {
            "checks": checks, "pass": all(checks.values()),
            "fast_fail_reason": ([] if all(checks.values()) else
                                 [k for k, v in checks.items() if not v]),
            "parity_power_p0_status": "NOT_RUN_AFTER_BINDING_PROVENANCE_AUDIT_FAST_FAIL",
            "micro_status": "NOT_RUN_AFTER_BINDING_PROVENANCE_AUDIT_FAST_FAIL",
        }
    # Promotion is impossible without original cryptographic linkage and the
    # completed blind whole-video presentation audit.  Do not trust producer
    # names/status flags to promote it.
    promotion = all(
        dataset_gates[d]["checks"]["original_provenance_complete"] and
        dataset_gates[d]["checks"]["blind_presentation_audit_pass"]
        for d in DATASETS)
    q_status = "PROMOTED_ARCHIVE_WEAK_MLLM" if promotion else "PROXY_ONLY_CHEAP_FORMAT"
    s0_go = promotion and all(x["pass"] for x in dataset_gates.values())
    chosen = None; choices = []
    decision = base_manifest(
        cfg, run_id, "S0_DECISION", "GO" if s0_go else "STOP",
        inputs=rehash,
        extra={"decision_rule": "SQ-S0-DECISION-v1",
               "dataset_gates": dataset_gates, "q_signal_status": q_status,
               "blind_audit_completed": all(
                   dataset_gates[d]["checks"]["blind_presentation_audit_completed"]
                   for d in DATASETS),
               "blind_audit_effect": "binding fail-closed STOP when absent/incomplete",
               "lambda_candidates": choices, "lambda_Q": chosen,
               "S1_unlocked": bool(s0_go), "S2_unlocked": False,
               "S2_S4_locked": True,
               "freeze_recomputed": freeze_recomputed,
               "formal_stop_reason": (None if s0_go else
                   "archive original-run cryptographic provenance and/or blind whole-video presentation QC incomplete; q remains proxy-only"),
               "segment_gold_exists": False, "segment_gold_used": False})
    exclusive_write_json(out, decision)
    print(canonical_json({"run_id": run_id, "status": decision["status"],
                          "q_signal_status": q_status, "lambda_Q": chosen,
                          "S1_unlocked": s0_go}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True, choices=[
        "hash-config", "static-sanity", "freeze", "provenance", "qproxy", "audit-freeze", "audit-ingest",
        "parity-power-p0", "micro", "decide"])
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--audit-csv")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    require_runtime(gpu=args.task in {"qproxy", "micro"})
    if args.task == "hash-config":
        _cfg, computed = config_payload_and_hash(args.config)
        print(canonical_json({"run_id": args.run_id, "config_sha256": computed,
                              "status": "HASH_ONLY_NO_OUTPUT"}))
        return
    cfg = load_config(args.config)
    if args.task in {"provenance", "qproxy", "audit-ingest", "parity-power-p0", "micro"} and not args.dataset:
        ap.error("--dataset required")
    if args.task == "audit-ingest" and not args.audit_csv:
        ap.error("--audit-csv required")
    if args.task == "static-sanity": task_static_sanity(cfg, args.run_id)
    elif args.task == "freeze": task_freeze(cfg, args.run_id)
    elif args.task == "provenance": task_provenance(cfg, args.dataset, args.run_id)
    elif args.task == "qproxy": task_qproxy(cfg, args.dataset, args.run_id)
    elif args.task == "audit-freeze": task_audit_freeze(cfg, args.run_id)
    elif args.task == "audit-ingest": task_audit_ingest(cfg, args.dataset, args.audit_csv, args.run_id)
    elif args.task == "parity-power-p0": task_parity_power_p0(cfg, args.dataset, args.run_id)
    elif args.task == "micro": task_micro(cfg, args.dataset, args.run_id)
    elif args.task == "decide": task_decide(cfg, args.run_id)


if __name__ == "__main__":
    main()
