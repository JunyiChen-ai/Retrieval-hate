#!/usr/bin/env python3
"""Steward-only V22 THVL validation evaluator; emits aggregate results only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.thvl_sealed.acquire import load_rows

GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
METRIC_KEYS = ("frame_ap", "frame_roc", "within_macro_ap", "within_macro_roc")


def mid_ecdf(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average")
    return (ranks - 0.5) / len(values)


def metrics(scores: dict, gt: dict, valid: dict) -> dict:
    ids = sorted(gt)
    yy = np.concatenate([gt[v][valid[v]] for v in ids])
    ss = np.concatenate([scores[v][valid[v]] for v in ids])
    if len(np.unique(yy)) != 2 or not np.isfinite(ss).all():
        raise RuntimeError("invalid scoped pooled metrics")
    mixed = [v for v in ids if len(np.unique(gt[v][valid[v]])) == 2]
    return {
        "frame_ap": float(average_precision_score(yy, ss)),
        "frame_roc": float(roc_auc_score(yy, ss)),
        "within_macro_ap": float(np.mean([average_precision_score(gt[v][valid[v]], scores[v][valid[v]]) for v in mixed])),
        "within_macro_roc": float(np.mean([roc_auc_score(gt[v][valid[v]], scores[v][valid[v]]) for v in mixed])),
        "mixed_videos": len(mixed),
    }


def project(rows: list, durations: dict, score_getter, chunk_ecdf: bool = False) -> tuple[dict, dict]:
    if chunk_ecdf:
        raw = np.asarray([score_getter(x) for x in rows], dtype=float)
        transformed = mid_ecdf(raw)
        values = {id(row): value for row, value in zip(rows, transformed)}
        score_getter = lambda row: values[id(row)]
    by = defaultdict(list)
    for row in rows:
        by[row["video_id"]].append(row)
    output, covered = {}, {}
    for video_id, duration in durations.items():
        length = int(math.ceil(duration))
        numerator = np.zeros(length, dtype=float)
        denominator = np.zeros(length, dtype=float)
        for row in by[video_id]:
            start = max(0.0, float(row["start"]))
            end = min(duration, float(row["end"]))
            if end <= start:
                continue
            lo = max(0, int(math.floor(start)))
            hi = min(length, int(math.ceil(end)))
            for frame in range(lo, hi):
                overlap = max(0.0, min(end, frame + 1.0) - max(start, frame))
                numerator[frame] += overlap * float(score_getter(row))
                denominator[frame] += overlap
        mask = denominator > 0
        value = np.zeros(length, dtype=float)
        value[mask] = numerator[mask] / denominator[mask]
        if mask.any():
            value[mask] -= value[mask].mean()
        output[video_id], covered[video_id] = value, mask
    return output, covered


def build_gt(csv_path: Path, private_map: Path, qc_path: Path) -> tuple[dict, dict, dict, dict]:
    mapping = {x["canonical_id"]: x for x in json.loads(private_map.read_text())["records"]}
    qc = json.loads(qc_path.read_text())
    durations = {x["hashed_id"]: max(float(p["duration_seconds"]) for p in x["paths"]) for x in qc["rows"]}
    gt, valid, clusters = {}, {}, {}
    for row in load_rows(csv_path):
        item = mapping[row["canonical_id"]]
        if item["split"] != "validation":
            continue
        hid = item["hashed_id"]
        length = int(math.ceil(durations[hid]))
        positive = np.zeros(length, dtype=bool)
        other = np.zeros(length, dtype=bool)
        timeline = np.arange(length, dtype=float)
        for segment in row["segments"]:
            overlap = (timeline < segment["end"]) & ((timeline + 1.0) > segment["start"])
            if segment["relevant"]:
                positive |= overlap
            else:
                other |= overlap
        gt[hid] = positive.astype(np.uint8)
        valid[hid] = ~(other & ~positive)
        clusters[hid] = item["source_group"]
    if not (set(gt) == set(valid) == set(clusters) == set(durations)):
        raise RuntimeError("full32 steward cohort mismatch")
    return gt, valid, durations, clusters


def global_component(rows: list, durations: dict) -> dict:
    by = defaultdict(list)
    for row in rows:
        by[row["video_id"]].append(row)
    raw = []
    ids = sorted(durations)
    for video_id in ids:
        q = by[video_id]
        weights = np.asarray([min(durations[video_id], x["end"]) - max(0.0, x["start"]) for x in q], dtype=float)
        vals = np.asarray([x["scores"]["causal_continuous"] for x in q], dtype=float)
        if not len(q) or not np.isfinite(vals).all() or np.any(weights <= 0):
            raise RuntimeError("invalid global chunks")
        raw.append(float(np.average(vals, weights=weights)))
    scaled = mid_ecdf(np.asarray(raw)) - 0.5
    return {video_id: np.full(int(math.ceil(durations[video_id])), value) for video_id, value in zip(ids, scaled)}


def combine(global_score: dict, local_score: dict, alpha: float, beta: float) -> dict:
    return {v: alpha * global_score[v] + beta * local_score[v] for v in global_score}


def shuffle_local(local: dict, covered: dict, draw: int) -> dict:
    result = {}
    for video_id, values in local.items():
        seed = int.from_bytes(hashlib.sha256(f"V22:{draw}:{video_id}".encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        copy = values.copy()
        copy[covered[video_id]] = rng.permutation(copy[covered[video_id]])
        result[video_id] = copy
    return result


def cluster_bootstrap(global_scores, selected_scores, gt, valid, clusters, B=2000, seed=22022):
    members = defaultdict(list)
    for video_id, cluster in clusters.items():
        members[cluster].append(video_id)
    keys = sorted(members)
    rng = np.random.default_rng(seed)
    values = {key: [] for key in METRIC_KEYS}
    for _ in range(B):
        g, s, y, mask = {}, {}, {}, {}
        for draw_index, cluster in enumerate(rng.choice(keys, len(keys), replace=True)):
            for video_id in members[cluster]:
                key = f"{draw_index}:{video_id}"
                g[key], s[key], y[key], mask[key] = global_scores[video_id], selected_scores[video_id], gt[video_id], valid[video_id]
        try:
            mg, ms = metrics(g, y, mask), metrics(s, y, mask)
        except (RuntimeError, ValueError):
            continue
        for key in METRIC_KEYS:
            values[key].append(ms[key] - mg[key])
    return {key: {"n_valid": len(vals), "point_delta": metrics(selected_scores, gt, valid)[key] - metrics(global_scores, gt, valid)[key], "lower95": float(np.quantile(vals, 0.025)), "upper95": float(np.quantile(vals, 0.975))} for key, vals in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.raw.read_text().splitlines()]
    gt, valid, durations, clusters = build_gt(args.csv, args.private_map, args.qc)
    if {x["video_id"] for x in rows} != set(gt):
        raise RuntimeError("raw/GT cohort mismatch")

    global_score = global_component(rows, durations)
    local_raw, covered = project(rows, durations, lambda x: x["scores"]["masked_branch_reset"])
    covered_values = np.concatenate([local_raw[v][covered[v]] for v in sorted(local_raw)])
    rms = float(np.sqrt(np.mean(covered_values ** 2)))
    if not np.isfinite(rms) or rms <= 1e-12:
        raise RuntimeError("zero local RMS")
    local_score = {v: x / rms for v, x in local_raw.items()}

    audit_rows = [x for x in rows if x.get("sequential_reference") is not None]
    packed_audit, packed_support = project(audit_rows, {v: durations[v] for v in {x["video_id"] for x in audit_rows}}, lambda x: x["scores"]["masked_branch_reset"], True)
    sequential_audit, sequential_support = project(audit_rows, {v: durations[v] for v in {x["video_id"] for x in audit_rows}}, lambda x: x["sequential_reference"], True)
    if any(not np.array_equal(packed_support[v], sequential_support[v]) for v in packed_support):
        raise RuntimeError("fidelity support mismatch")
    audit_ids = set(packed_audit)
    audit_gt = {v: gt[v] for v in audit_ids}
    audit_valid = {v: valid[v] for v in audit_ids}
    packed_metrics = metrics(packed_audit, audit_gt, audit_valid)
    sequential_metrics = metrics(sequential_audit, audit_gt, audit_valid)
    metric_delta = {key: packed_metrics[key] - sequential_metrics[key] for key in METRIC_KEYS}
    packed_chunks = np.asarray([x["scores"]["masked_branch_reset"] for x in audit_rows])
    sequential_chunks = np.asarray([x["sequential_reference"] for x in audit_rows])
    fidelity = {
        "n_rows": len(audit_rows),
        "n_videos": len(audit_ids),
        "spearman": float(spearmanr(packed_chunks, sequential_chunks).statistic),
        "max_abs_error_report_only": float(np.max(np.abs(packed_chunks - sequential_chunks))),
        "packed_metrics": packed_metrics,
        "sequential_metrics": sequential_metrics,
        "metric_delta_packed_minus_sequential": metric_delta,
        "max_abs_metric_delta": max(abs(x) for x in metric_delta.values()),
    }

    surface = []
    for alpha in GRID:
        for beta in GRID:
            if alpha == beta == 0:
                continue
            score = combine(global_score, local_score, alpha, beta)
            surface.append({"alpha": alpha, "beta": beta, "metrics": metrics(score, gt, valid)})
    selected = max(surface, key=lambda x: (x["metrics"]["frame_ap"], x["metrics"]["frame_roc"], x["metrics"]["within_macro_ap"], -(x["alpha"] + x["beta"]), -x["alpha"], -x["beta"]))
    global_metrics = metrics(global_score, gt, valid)
    local_metrics = metrics(local_score, gt, valid)
    equal_metrics = metrics(combine(global_score, local_score, 1, 1), gt, valid)
    selected_scores = combine(global_score, local_score, selected["alpha"], selected["beta"])

    shuffled = [metrics(combine(global_score, shuffle_local(local_score, covered, draw), selected["alpha"], selected["beta"]), gt, valid) for draw in range(200)]
    shuffle_summary = {key: {"mean": float(np.mean([x[key] for x in shuffled])), "q975": float(np.quantile([x[key] for x in shuffled], 0.975))} for key in METRIC_KEYS}
    bootstrap = cluster_bootstrap(global_score, selected_scores, gt, valid, clusters)
    selected_metrics = selected["metrics"]
    gates = {
        "fidelity_spearman": fidelity["spearman"] >= 0.99,
        "fidelity_metric_delta": fidelity["max_abs_metric_delta"] <= 0.01,
        "frame_ap_gain": selected_metrics["frame_ap"] - global_metrics["frame_ap"] >= 0.01,
        "frame_roc_nonregression": selected_metrics["frame_roc"] - global_metrics["frame_roc"] >= -0.005,
        "bootstrap_ap_lower": bootstrap["frame_ap"]["lower95"] >= 0,
        "within_macro_roc_absolute": selected_metrics["within_macro_roc"] >= 0.55,
        "within_macro_roc_shuffle": selected_metrics["within_macro_roc"] > shuffle_summary["within_macro_roc"]["q975"],
        "within_macro_ap_noninferior": selected_metrics["within_macro_ap"] >= global_metrics["within_macro_ap"],
    }
    payload = {
        "status": "V22_VAL_INFORMED_SELECTION_FROZEN",
        "test_labels_opened": False,
        "test_open_authorized": all(gates.values()),
        "amendment_sha256": hashlib.sha256(args.amendment.read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "cohort": {"videos": len(gt), "mixed_videos": selected_metrics["mixed_videos"], "speech_frame_coverage": float(sum(x.sum() for x in covered.values()) / sum(len(x) for x in covered.values())), "local_rms": rms},
        "fidelity": fidelity,
        "mandatory_arms": {"global_only": global_metrics, "local_only": local_metrics, "equal": equal_metrics},
        "selection_surface": surface,
        "selected": selected,
        "shuffle_B200": shuffle_summary,
        "paired_source_group_bootstrap_B2000": bootstrap,
        "activation_gates": gates,
        "activation_pass": all(gates.values()),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "test_open_authorized", "cohort", "fidelity", "mandatory_arms", "selected", "shuffle_B200", "paired_source_group_bootstrap_B2000", "activation_gates", "activation_pass")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

