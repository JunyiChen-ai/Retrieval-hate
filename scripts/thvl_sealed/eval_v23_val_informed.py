#!/usr/bin/env python3
"""Steward-only aggregate V23 THVL validation evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.thvl_sealed.eval_v22_val_informed import GRID, METRIC_KEYS, build_gt, metrics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expand_windows(rows: list[dict], durations: dict[str, float], value_key: str) -> tuple[dict, dict]:
    by = defaultdict(list)
    for row in rows:
        by[row["video_id"]].append(row)
    scores, window_values = {}, {}
    for video_id in sorted(durations):
        timeline = np.zeros(int(math.ceil(durations[video_id])), dtype=float)
        windows = sorted(by[video_id], key=lambda x: x["window_index"])
        if not windows or abs(windows[0]["start"]) > 1e-9 or abs(windows[-1]["end"] - durations[video_id]) > 1e-6:
            raise RuntimeError("window endpoint coverage failure")
        previous = 0.0
        values = []
        for expected_index, row in enumerate(windows):
            if row["window_index"] != expected_index or abs(row["start"] - previous) > 1e-6 or row["end"] <= row["start"]:
                raise RuntimeError("window ordering/gap/overlap failure")
            value = float(row[value_key])
            if not np.isfinite(value):
                raise RuntimeError("nonfinite V23 score")
            frame_centers = np.arange(len(timeline), dtype=float) + 0.5
            mask = (frame_centers >= row["start"]) & (frame_centers < row["end"])
            timeline[mask] = value
            values.append(value)
            previous = row["end"]
        scores[video_id] = timeline
        window_values[video_id] = np.asarray(values, dtype=float)
    return scores, window_values


def combine(global_score, local_score, alpha, beta):
    return {v: alpha * global_score[v] + beta * local_score[v] for v in global_score}


def shuffled_local(rows, durations, draw):
    by = defaultdict(list)
    for row in rows:
        by[row["video_id"]].append(dict(row))
    shuffled = []
    for video_id in sorted(by):
        windows = sorted(by[video_id], key=lambda x: x["window_index"])
        seed = int.from_bytes(hashlib.sha256(f"V23-window:{draw}:{video_id}".encode()).digest()[:8], "big")
        values = np.asarray([x["score_centered_rms_scaled"] for x in windows], dtype=float)
        permutation = np.random.default_rng(seed).permutation(len(values))
        for row, value in zip(windows, values[permutation]):
            row["score_centered_rms_scaled"] = float(value)
            shuffled.append(row)
    return expand_windows(shuffled, durations, "score_centered_rms_scaled")[0]


def cluster_bootstrap(global_scores, selected_scores, gt, valid, clusters, B=2000, seed=23023):
    members = defaultdict(list)
    for video_id, group in clusters.items():
        members[group].append(video_id)
    groups = sorted(members)
    rng = np.random.default_rng(seed)
    deltas = {key: [] for key in METRIC_KEYS}
    for _ in range(B):
        bg, bs, y, mask = {}, {}, {}, {}
        for sample_index, group in enumerate(rng.choice(groups, len(groups), replace=True)):
            for video_id in members[group]:
                key = f"{sample_index}:{video_id}"
                bg[key], bs[key], y[key], mask[key] = global_scores[video_id], selected_scores[video_id], gt[video_id], valid[video_id]
        try:
            gm, sm = metrics(bg, y, mask), metrics(bs, y, mask)
        except (RuntimeError, ValueError):
            continue
        for key in METRIC_KEYS:
            deltas[key].append(sm[key] - gm[key])
    point_g, point_s = metrics(global_scores, gt, valid), metrics(selected_scores, gt, valid)
    return {key: {"n_valid": len(values), "point_delta": point_s[key] - point_g[key], "lower95": float(np.quantile(values, 0.025)), "upper95": float(np.quantile(values, 0.975))} for key, values in deltas.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--derived-manifest", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v22-components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    raw_manifest = json.loads(args.raw_manifest.read_text())
    derived_manifest = json.loads(args.derived_manifest.read_text())
    rows = load_jsonl(args.derived)
    if sha256(args.derived) != derived_manifest["derived_sha256"] or len(rows) != derived_manifest["n_rows"]:
        raise RuntimeError("derived artifact hash/count mismatch")
    if derived_manifest["source_raw_sha256"] != raw_manifest["raw_sha256"]:
        raise RuntimeError("raw/derived provenance mismatch")
    if config["n_windows"] != len(rows) or config["n_videos"] != 32:
        raise RuntimeError("config/raw cohort mismatch")

    gt, valid, durations, clusters = build_gt(args.csv, args.private_map, args.qc)
    if {x["video_id"] for x in rows} != set(gt):
        raise RuntimeError("derived/GT exact cohort mismatch")
    local_score, window_values = expand_windows(rows, durations, "score_centered_rms_scaled")

    components = {x["video_id"]: x for x in load_jsonl(args.v22_components)}
    if set(components) != set(gt):
        raise RuntimeError("V22 global component cohort mismatch")
    global_score = {}
    for video_id in sorted(gt):
        value = np.asarray(components[video_id]["global_calibrated"], dtype=float)
        if len(value) != len(gt[video_id]) or not np.all(value == value[0]):
            raise RuntimeError("invalid frozen V22 global")
        global_score[video_id] = value

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

    shuffle_metrics = []
    for draw in range(200):
        local_shuffle = shuffled_local(rows, durations, draw)
        shuffle_metrics.append(metrics(combine(global_score, local_shuffle, selected["alpha"], selected["beta"]), gt, valid))
    shuffle_summary = {key: {"mean": float(np.mean([x[key] for x in shuffle_metrics])), "q975": float(np.quantile([x[key] for x in shuffle_metrics], 0.975))} for key in METRIC_KEYS}
    bootstrap = cluster_bootstrap(global_score, selected_scores, gt, valid, clusters)
    sm = selected["metrics"]
    gates = {
        "full_window_coverage": len(rows) == 480 and all(np.isfinite(np.concatenate(list(window_values.values())))),
        "selected_beta_positive": selected["beta"] > 0,
        "within_macro_roc_absolute": sm["within_macro_roc"] >= 0.55,
        "pooled_ap_over_global": sm["frame_ap"] > global_metrics["frame_ap"],
        "pooled_roc_over_global": sm["frame_roc"] > global_metrics["frame_roc"],
        "within_ap_noninferior": sm["within_macro_ap"] >= global_metrics["within_macro_ap"],
        "within_roc_noninferior": sm["within_macro_roc"] >= global_metrics["within_macro_roc"],
        "within_roc_over_shuffle_q975": sm["within_macro_roc"] > shuffle_summary["within_macro_roc"]["q975"],
        "bootstrap_ap_lower_nonnegative": bootstrap["frame_ap"]["lower95"] >= 0,
        "bootstrap_roc_lower_nonnegative": bootstrap["frame_roc"]["lower95"] >= 0,
    }
    payload = {
        "status": "V23_VAL_INFORMED_SELECTION_FROZEN",
        "test_labels_opened": False,
        "test_open_authorized": all(gates.values()),
        "provenance": {"config_sha256": sha256(args.config), "raw_manifest_sha256": sha256(args.raw_manifest), "derived_manifest_sha256": sha256(args.derived_manifest), "derived_sha256": sha256(args.derived)},
        "cohort": {"videos": len(gt), "windows": len(rows), "mixed_videos": sm["mixed_videos"], "all_center_decode_offset_zero": all(x["frame_fallback_offset"] == 0 for x in rows)},
        "mandatory_arms": {"global_only": global_metrics, "local_only": local_metrics, "equal": equal_metrics},
        "selection_surface": surface,
        "selected": selected,
        "shuffle_B200": shuffle_summary,
        "paired_source_group_bootstrap_B2000": bootstrap,
        "activation_gates": gates,
        "activation_pass": all(gates.values()),
        "protocol_note": "VAL-INFORMED; producer used midpoint ASR assignment frozen before forward; no test access",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "test_open_authorized", "cohort", "mandatory_arms", "selected", "shuffle_B200", "paired_source_group_bootstrap_B2000", "activation_gates", "activation_pass")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

