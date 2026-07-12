#!/usr/bin/env python
"""Mine frozen SSR SC/MI directed arcs and exact video-label vote events."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from ssr_common import (  # noqa: E402
    atomic_write_json, atomic_write_jsonl, canonical_json, exact_vote, id_hash,
    load_config, read_jsonl, resolve, sha256_file, sha256_obj,
)


def canonical_pair(dataset, a, b):
    low, high = sorted((str(a), str(b)))
    pair_id = id_hash("ssr-pair-v1", dataset, low, high)
    direction = "low_to_high" if str(a) == low else "high_to_low"
    return pair_id, low, high, direction


def resort(items):
    return sorted(items, key=lambda x: (-float(x["cosine"]), str(x["id"])))


def build_candidates(dataset, row, pred, topk=20, limit=3):
    qid, qlabel = row["query_id"], int(row["query_label"])
    ranking = row["ranking"]
    if len(ranking) <= topk:
        raise RuntimeError("ranking for {} lacks rank-21 promotion item".format(qid))
    top = ranking[:topk]
    baseline_vote, baseline_pred, denom = exact_vote(top, topk)
    if baseline_pred != int(pred["prediction"]):
        raise RuntimeError("prediction/vote mismatch for {}".format(qid))
    if abs(baseline_vote - float(pred["vote"])) > 1e-6:
        raise RuntimeError("stored vote mismatch for {}".format(qid))
    error = int(baseline_pred != qlabel)
    common = {
        "dataset": dataset, "query_id": qid, "query_label": qlabel,
        "outer_fold": int(row["outer_fold"]), "memory_n": int(row["memory_n"]),
        "baseline_prediction": baseline_pred, "baseline_vote": baseline_vote,
        "baseline_error": error, "vote_abs_denom": denom,
        "normalized_abs_margin": abs(baseline_vote) / max(denom, 1e-12),
    }
    sc, mi = [], []
    for n in top:
        if int(n["label"]) == qlabel:
            continue
        without = [x for x in top if x["id"] != n["id"]]
        if len(without) != topk - 1:
            raise RuntimeError("neighbor ID duplicated in ranking")
        counter = resort(without + [ranking[topk]])
        _, changed_pred, _ = exact_vote(counter, topk)
        arc = dict(common)
        arc.update({
            "candidate_family": "SC", "neighbor_id": n["id"],
            "neighbor_label": int(n["label"]), "rank": int(n["rank"]),
            "cosine": float(n["cosine"]), "reference_id": None,
            "event": int(bool(error) and changed_pred == qlabel),
            "counterfactual_prediction": changed_pred,
            "event_definition": "remove_candidate_promote_original_rank21_exact_vote",
        })
        sc.append(arc)
        if len(sc) >= limit:
            break
    opposite_top = [n for n in top if int(n["label"]) != qlabel]
    if opposite_top:
        for n in ranking[topk:]:
            if int(n["label"]) != qlabel:
                continue
            reference = min(
                opposite_top,
                key=lambda ref: (
                    abs(float(n["cosine"]) - float(ref["cosine"])),
                    int(ref["rank"]),
                    id_hash(dataset, row["outer_fold"], qid, n["id"], ref["id"])))
            counter = resort([x for x in top if x["id"] != reference["id"]] + [n])
            if len(counter) != topk:
                raise RuntimeError("MI replacement cardinality mismatch")
            _, changed_pred, _ = exact_vote(counter, topk)
            arc = dict(common)
            arc.update({
                "candidate_family": "MI", "neighbor_id": n["id"],
                "neighbor_label": int(n["label"]), "rank": int(n["rank"]),
                "cosine": float(n["cosine"]),
                "reference_id": reference["id"],
                "reference_rank": int(reference["rank"]),
                "reference_cosine": float(reference["cosine"]),
                "event": int(bool(error) and changed_pred == qlabel),
                "counterfactual_prediction": changed_pred,
                "event_definition": "replace_fixed_opposite_reference_exact_vote",
            })
            mi.append(arc)
            if len(mi) >= limit:
                break
    return sc, mi


def allocate(dataset, all_candidates, max_pairs):
    by_q = {qid: fam for qid, fam in all_candidates.items()}
    query_order = sorted(
        by_q,
        key=lambda q: (0 if by_q[q]["baseline_error"] else 1,
                       id_hash("ssr-v1", dataset, by_q[q]["outer_fold"], q)))
    pairs, arcs, used_directions = {}, [], set()

    def add(arc):
        pair_id, low, high, direction = canonical_pair(
            dataset, arc["query_id"], arc["neighbor_id"])
        direction_key = (pair_id, direction)
        if direction_key in used_directions:
            return True
        if pair_id not in pairs and len(pairs) >= max_pairs:
            return False
        if pair_id not in pairs:
            pairs[pair_id] = {
                "canonical_pair_id": pair_id, "dataset": dataset,
                "video_a_id": low, "video_b_id": high,
                "video_a_label": int(arc["query_label"] if arc["query_id"] == low
                                     else arc["neighbor_label"]),
                "video_b_label": int(arc["query_label"] if arc["query_id"] == high
                                     else arc["neighbor_label"]),
                "direction_mask": [], "arc_ids": [],
                "only_gold": "video_level_binary_label",
            }
        arc = dict(arc)
        arc_id = id_hash("ssr-arc-v1", dataset, arc["query_id"],
                         arc["neighbor_id"], arc["candidate_family"])
        arc.update({"arc_id": arc_id, "canonical_pair_id": pair_id,
                    "canonical_direction": direction})
        pairs[pair_id]["direction_mask"].append(direction)
        pairs[pair_id]["arc_ids"].append(arc_id)
        arcs.append(arc)
        used_directions.add(direction_key)
        return True

    # Pass A: highest-ranked SC arc for every baseline error query.
    for qid in query_order:
        fam = by_q[qid]
        if fam["baseline_error"] and fam["SC"]:
            if not add(fam["SC"][0]):
                break
    # Pass B: positions 1/2/3, query order, SC then MI.
    full = len(pairs) >= max_pairs
    if not full:
        for pos in range(3):
            for qid in query_order:
                for family in ("SC", "MI"):
                    cand = by_q[qid][family]
                    if pos < len(cand) and not add(cand[pos]):
                        full = True
                        break
                if full:
                    break
            if full:
                break
    pair_rows = []
    for pair in pairs.values():
        pair["direction_mask"] = sorted(set(pair["direction_mask"]))
        pair["arc_ids"] = sorted(set(pair["arc_ids"]))
        pair_rows.append(pair)
    pair_rows.sort(key=lambda x: x["canonical_pair_id"])
    arcs.sort(key=lambda x: x["arc_id"])
    return pair_rows, arcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, choices=["MHC", "MHC_zh"])
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SSR computation must run under SLURM")
    cfg = load_config(args.config)
    oof_root = resolve(cfg, "artifacts") / "oof" / args.dataset
    fold_path = resolve(cfg, "artifacts") / "folds" / "{}.json".format(args.dataset)
    folds = json.load(open(fold_path, encoding="utf-8"))
    frozen = {x["id"]: {"fold": int(x["fold"]), "label": int(x["label"])}
              for x in folds["records"]}
    all_candidates = {}
    source_hashes = {}
    seen_queries = set()
    for fold in range(int(cfg["folds"]["n_splits"])):
        d = oof_root / "fold{}".format(fold)
        manifest = json.load(open(d / "manifest.json", encoding="utf-8"))
        if manifest["status"] != "COMPLETED" or manifest["dataset"] != args.dataset:
            raise RuntimeError("bad OOF manifest {}".format(d))
        if manifest["config_sha256"] != cfg["computed_config_sha256"]:
            raise RuntimeError("OOF config hash mismatch")
        if manifest["fold_artifact_sha256"] != sha256_file(fold_path):
            raise RuntimeError("OOF/fold artifact hash mismatch")
        for name in ("ranking.jsonl", "predictions.json"):
            if manifest["outputs"].get(name) != sha256_file(d / name):
                raise RuntimeError("OOF output hash mismatch: {}".format(d / name))
        rankings = read_jsonl(d / "ranking.jsonl")
        predictions = json.load(open(d / "predictions.json", encoding="utf-8"))
        by_pred = {x["query_id"]: x for x in predictions}
        expected_query = {vid for vid, meta in frozen.items() if meta["fold"] == fold}
        if {x["query_id"] for x in rankings} != expected_query or set(by_pred) != expected_query:
            raise RuntimeError("fold {} query set mismatch".format(fold))
        for row in rankings:
            qid = row["query_id"]
            if frozen[qid]["fold"] != fold or int(row["query_label"]) != frozen[qid]["label"]:
                raise RuntimeError("query fold/label mismatch: {}".format(qid))
            neighbor_ids = [x["id"] for x in row["ranking"]]
            if len(neighbor_ids) != len(set(neighbor_ids)):
                raise RuntimeError("duplicate neighbor in full ranking: {}".format(qid))
            expected_memory = {vid for vid, meta in frozen.items() if meta["fold"] != fold}
            if set(neighbor_ids) != expected_memory:
                raise RuntimeError("fold-local memory mismatch: {}".format(qid))
            if any(frozen[x["id"]]["fold"] == fold or
                   int(x["label"]) != frozen[x["id"]]["label"]
                   for x in row["ranking"]):
                raise RuntimeError("neighbor fold/label mismatch: {}".format(qid))
            if qid in seen_queries:
                raise RuntimeError("query appears in multiple OOF folds: {}".format(qid))
            seen_queries.add(qid)
            sc, mi = build_candidates(
                args.dataset, row, by_pred[qid],
                topk=int(cfg["comparator"]["topk"]),
                limit=int(cfg["pair_mining"]["max_sc_per_query"]))
            all_candidates[qid] = {
                "outer_fold": int(row["outer_fold"]),
                "baseline_error": int(by_pred[qid]["baseline_error"]),
                "SC": sc, "MI": mi,
            }
        source_hashes[str(fold)] = {
            "manifest": sha256_file(d / "manifest.json"),
            "ranking": sha256_file(d / "ranking.jsonl"),
            "predictions": sha256_file(d / "predictions.json"),
        }
    train_ids = {x["id"] for x in folds["records"]}
    if seen_queries != train_ids:
        raise RuntimeError("OOF queries do not cover train IDs exactly")

    pairs, arcs = allocate(args.dataset, all_candidates,
                           int(cfg["pair_mining"]["max_unique_pairs"]))
    if any(a["query_id"] not in train_ids or a["neighbor_id"] not in train_ids
           for a in arcs):
        raise RuntimeError("non-train endpoint emitted")
    out = resolve(cfg, "artifacts") / "pairs" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(out / "pairs.jsonl", pairs)
    atomic_write_jsonl(out / "arcs.jsonl", arcs)
    atomic_write_jsonl(out / "events.jsonl", [{
        "arc_id": a["arc_id"], "canonical_pair_id": a["canonical_pair_id"],
        "candidate_family": a["candidate_family"], "query_id": a["query_id"],
        "neighbor_id": a["neighbor_id"], "reference_id": a.get("reference_id"),
        "event": a["event"], "event_definition": a["event_definition"],
        "gold_source": "video_level_binary_label_only",
        "segment_gold_used": False,
    } for a in arcs])
    schema = {
        "pair_schema_version": 1,
        "canonical_pair": {
            "required": ["canonical_pair_id", "dataset", "video_a_id", "video_b_id",
                         "video_a_label", "video_b_label", "direction_mask", "arc_ids"],
            "gold_fields": ["video_a_label", "video_b_label"],
            "gold_granularity": "video",
            "segment_fields_forbidden": True,
        },
        "arc_schema": {
            "families": ["MI", "SC"],
            "event_is": "counterfactual_exact_video_vote_flip_not_segment_annotation",
        },
    }
    atomic_write_json(out / "PAIR_SCHEMA.json", schema)
    counts = defaultdict(int)
    event_counts = defaultdict(int)
    queries = defaultdict(set)
    for a in arcs:
        counts[a["candidate_family"]] += 1
        event_counts[a["candidate_family"]] += int(a["event"])
        queries[a["candidate_family"]].add(a["query_id"])
    manifest = {
        "run_id": args.run_id, "status": "COMPLETED", "dataset": args.dataset,
        "config_sha256": cfg["computed_config_sha256"],
        "n_unique_pairs": len(pairs), "n_directed_arcs": len(arcs),
        "family_arcs": dict(counts), "family_positive_events": dict(event_counts),
        "family_unique_queries": {k: len(v) for k, v in queries.items()},
        "max_unique_pairs": int(cfg["pair_mining"]["max_unique_pairs"]),
        "all_endpoints_train_only": True, "segment_gold_used": False,
        "source_oof_hashes": source_hashes,
        "outputs": {name: sha256_file(out / name) for name in
                    ("pairs.jsonl", "arcs.jsonl", "events.jsonl", "PAIR_SCHEMA.json")},
        "replay_hash": sha256_obj({"pairs": pairs, "arcs": arcs}),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_write_json(out / "manifest.json", manifest)
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
