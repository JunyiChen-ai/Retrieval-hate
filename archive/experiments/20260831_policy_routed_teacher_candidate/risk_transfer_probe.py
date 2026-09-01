#!/usr/bin/env python3
"""Gate A: do certified negative-video teacher errors transfer locally?"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASE))

from hate_common import data as hdata  # noqa: E402
from src.hate_local_features import aligned_local_features  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")
VERA = {
    corpus: REPO / f"results/reproduction/official_val/final/vera/{corpus}/seed_234/val_infer/scores.jsonl"
    for corpus in CORPORA
}
CHANNELS = ("audio", "visual", "text", "concat")
COVERAGES = (.25, .50, .75, 1.0)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scores(path):
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["video_id"]] = row
    return rows


def prototypes(rows, seed, maximum=64):
    rows = np.asarray(rows, dtype=np.float32)
    count = min(maximum, len(rows))
    if count == 0:
        raise RuntimeError("cannot fit an empty prototype bank")
    if count == len(rows):
        centers = rows.copy()
    else:
        model = MiniBatchKMeans(
            n_clusters=count, random_state=seed, batch_size=2048,
            n_init=3, max_iter=100,
        ).fit(rows)
        centers = model.cluster_centers_.astype(np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-6)
    return centers


def proximity(rows, centers):
    return np.max(np.asarray(rows, dtype=np.float32) @ centers.T, axis=1)


def accepted_curve(risk, correct):
    order = np.argsort(risk, kind="stable")
    output = {}
    for coverage in COVERAGES:
        count = max(1, int(np.floor(len(order) * coverage)))
        output[str(coverage)] = {
            "n": count, "precision": float(np.mean(correct[order[:count]])),
        }
    return output


def risk_report(risk, error, video_ids, seed):
    rng = np.random.default_rng(seed)
    shuffled = risk.copy()
    for video_id in sorted(set(video_ids)):
        index = np.flatnonzero(video_ids == video_id)
        shuffled[index] = risk[index][rng.permutation(len(index))]
    per_video_auc = []
    shuffled_per_video_auc = []
    per_video_accepted = {str(value): [] for value in COVERAGES}
    for video_id in sorted(set(video_ids)):
        index = np.flatnonzero(video_ids == video_id)
        local_error = error[index]
        if len(np.unique(local_error)) == 2:
            per_video_auc.append(roc_auc_score(local_error, risk[index]))
            shuffled_per_video_auc.append(
                roc_auc_score(local_error, shuffled[index])
            )
        local_curve = accepted_curve(risk[index], 1 - local_error)
        for coverage in COVERAGES:
            per_video_accepted[str(coverage)].append(
                local_curve[str(coverage)]["precision"]
            )
    return {
        "error_roc": float(roc_auc_score(error, risk)),
        "error_ap": float(average_precision_score(error, risk)),
        "accepted": accepted_curve(risk, 1 - error),
        "within_video_macro_error_roc": float(np.mean(per_video_auc)),
        "within_video_n": len(per_video_auc),
        "accepted_within_video_macro_precision": {
            key: float(np.mean(values))
            for key, values in per_video_accepted.items()
        },
        "within_video_shuffle_macro_error_roc": float(
            np.mean(shuffled_per_video_auc)
        ),
    }


def analyze(corpus, seed=234):
    labels = hdata.load_labels(corpus)
    gt = hdata.gt_arrays(corpus, "val")
    teacher = load_scores(VERA[corpus])
    if set(gt) != set(teacher):
        raise RuntimeError(f"VERA/GT coverage mismatch: {corpus}")
    negative = [v for v in gt if labels[v] == 0]
    positive = [v for v in gt if labels[v] == 1]
    fp_rows = {name: [] for name in CHANNELS}
    normal_rows = {name: [] for name in CHANNELS}
    negative_teacher_positive = 0
    for video_id in negative:
        parts = aligned_local_features(corpus, video_id)
        raw = np.asarray(teacher[video_id]["score_raw"]) > 0
        negative_teacher_positive += int(raw.sum())
        for name in CHANNELS:
            normal_rows[name].append(parts[name])
            fp_rows[name].append(parts[name][raw])
    fp_centers, normal_centers = {}, {}
    for name in CHANNELS:
        false_positive = np.concatenate(fp_rows[name])
        normal = np.concatenate(normal_rows[name])
        fp_centers[name] = prototypes(false_positive, seed)
        normal_centers[name] = prototypes(normal, seed)
    candidate_parts = {name: [] for name in CHANNELS}
    confidence, error, candidate_video = [], [], []
    for video_id in positive:
        raw = np.asarray(teacher[video_id]["score_raw"]) > 0
        if not raw.any():
            continue
        parts = aligned_local_features(corpus, video_id)
        for name in CHANNELS:
            candidate_parts[name].append(parts[name][raw])
        confidence.append(
            np.asarray(teacher[video_id]["score_neighbor"], dtype=float)[raw]
        )
        error.append((np.asarray(gt[video_id])[raw] == 0).astype(np.int64))
        candidate_video.extend([video_id] * int(raw.sum()))
    confidence = np.concatenate(confidence)
    error = np.concatenate(error)
    candidate_video = np.asarray(candidate_video, dtype=object)
    risks = {"teacher_confidence": -confidence}
    for name in CHANNELS:
        rows = np.concatenate(candidate_parts[name])
        risks[f"fp_proximity_{name}"] = proximity(rows, fp_centers[name])
        risks[f"normal_density_{name}"] = proximity(rows, normal_centers[name])
    reports = {
        name: risk_report(risk, error, candidate_video, seed)
        for name, risk in risks.items()
    }
    core = reports["fp_proximity_concat"]
    confidence_report = reports["teacher_confidence"]
    normal = reports["normal_density_concat"]
    gates = {
        "core_within_roc_beats_confidence":
        core["within_video_macro_error_roc"]
        > confidence_report["within_video_macro_error_roc"],
        "core_within_roc_beats_normal_density":
        core["within_video_macro_error_roc"]
        > normal["within_video_macro_error_roc"],
        "core_within_roc_beats_feature_shuffle":
        core["within_video_macro_error_roc"]
        > core["within_video_shuffle_macro_error_roc"],
    }
    for coverage in (.25, .50, .75):
        key = str(coverage)
        gates[f"precision_at_{key}_beats_controls"] = (
            core["accepted_within_video_macro_precision"][key]
            >= confidence_report["accepted_within_video_macro_precision"][key]
            and core["accepted_within_video_macro_precision"][key]
            >= normal["accepted_within_video_macro_precision"][key]
        )
    return {
        "vera_path": str(VERA[corpus]), "vera_sha256": sha256(VERA[corpus]),
        "negative_videos": len(negative), "positive_videos": len(positive),
        "negative_teacher_positive_seconds": negative_teacher_positive,
        "positive_teacher_positive_seconds": int(len(error)),
        "positive_teacher_positive_error_rate": float(error.mean()),
        "prototype_count": {name: len(fp_centers[name]) for name in CHANNELS},
        "reports": reports, "gates": gates, "pass": all(gates.values()),
    }


def main():
    corpora = {corpus: analyze(corpus) for corpus in CORPORA}
    payload = {
        "date": "2026-08-31", "split": "val",
        "status": "risk_transfer_gate_a", "test_used": False,
        "frame_gt_role": "evaluation_only_after_frozen risk construction",
        "corpora": corpora, "pass": all(row["pass"] for row in corpora.values()),
        "verdict": ("GO_TO_STUDENT_PILOT" if all(row["pass"] for row in corpora.values())
                    else "STOP_BEFORE_STUDENT"),
    }
    out = REPO / "runs/20260831_policy_routed_teacher_candidate/gate_a.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
