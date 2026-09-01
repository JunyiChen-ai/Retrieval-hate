"""Developmental test error analysis after the frozen two-corpus pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from imports import BASELINE_ROOT, base_data
from model import CarrierItS2CLR
from oof import negative_centroids, pseudo_for_video
from protocol import blind_test_split, supervised_split

if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))
from hate_common import data as hdata  # noqa: E402


def compact_formal_metrics(path, corpus):
    payload = json.loads(path.read_text())
    if payload.get("corpus") != corpus or payload.get("split") != "test":
        raise RuntimeError(f"unexpected formal metric scope in {path}")
    result = payload["results"]["score_core"]
    if result["n_videos_missing_from_scores"] or result["n_videos_not_in_gold"]:
        raise RuntimeError(f"formal evaluator coverage failure in {path}")
    return {
        "pooled_ap": result["pr_auc"], "pooled_roc": result["roc_auc"],
        "within_roc": result["per_video"]["macro_auc"],
        "per_video_auc": result["per_video"]["per_video_auc"],
    }


def load_scores(path):
    records = hdata.load_scores_jsonl(str(path))
    return {video_id: np.asarray(row["score_core"], dtype=np.float64)
            for video_id, row in records.items()}


def safe_spearman(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise RuntimeError("Spearman inputs must be aligned one-dimensional arrays")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise RuntimeError("non-finite Spearman input")
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return {"rho": None, "n": int(len(first))}
    value = spearmanr(first, second).statistic
    if not np.isfinite(value):
        return {"rho": None, "n": int(len(first))}
    return {"rho": float(value), "n": int(len(first))}


def model_carrier_rates(corpus, checkpoint_path, expected_scores, device):
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=True)
    if checkpoint.get("corpus") != corpus or checkpoint["model_args"].get("arm") != "core":
        raise RuntimeError(f"unexpected frozen core checkpoint in {checkpoint_path}")
    model = CarrierItS2CLR(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    train_ids, train_labels = supervised_split(corpus, "train")
    centroids = negative_centroids(corpus, train_ids, train_labels)
    test_ids, placeholders = blind_test_split(corpus)
    if set(test_ids) != set(expected_scores):
        raise RuntimeError(f"frozen core prediction cohort mismatch for {corpus}")
    dataset = base_data.MultiModalDataset(corpus, test_ids, placeholders)
    rates = {}
    maximum_score_error = 0.0
    for index, video_id in enumerate(test_ids):
        parts, _, length, observed_id = dataset[index]
        if observed_id != video_id:
            raise RuntimeError("test dataset order mismatch")
        feats = {name: value[None] for name, value in parts.items()}
        row = pseudo_for_video(model.backbone, feats, length, centroids, device)
        frozen_score = np.asarray(expected_scores[video_id], dtype=np.float64)
        recomputed_score = row["fused_score"].numpy().astype(np.float64)
        if frozen_score.shape != (length,) or recomputed_score.shape != (length,):
            raise RuntimeError(f"frozen core score length mismatch for {corpus}/{video_id}")
        score_error = float(np.max(np.abs(frozen_score - recomputed_score)))
        maximum_score_error = max(maximum_score_error, score_error)
        if score_error > 1e-6:
            raise RuntimeError(
                f"carrier diagnostic changed frozen core prediction for "
                f"{corpus}/{video_id}: {score_error}"
            )
        divisor = int(checkpoint["model_args"]["k_proportion"])
        top_count = max(1, math.ceil(length / divisor))
        top = torch.argsort(row["fused_score"], descending=True,
                            stable=True)[:top_count]
        stable = ((row["deletion_centroid"] > 0)
                  & (row["deletion_neighbor"] > 0))
        rates[video_id] = {
            name: float(stable[top, modality].float().mean())
            for modality, name in enumerate(checkpoint["model_args"]["dims"])
        }
    return rates, maximum_score_error


def analyze_corpus(root, corpus, device):
    core_checkpoint_path = root / corpus / "core" / "model.pt"
    core_train_log_path = root / corpus / "core" / "train_log.json"
    core_checkpoint_meta = torch.load(
        core_checkpoint_path, map_location="cpu", weights_only=True
    )
    core_train_log = json.loads(core_train_log_path.read_text())
    if (
        core_checkpoint_meta.get("corpus") != corpus
        or core_checkpoint_meta.get("selected_epoch")
        != core_train_log.get("selected_epoch")
        or core_train_log.get("test_used_for_gradient_or_checkpoint_selection") is not False
    ):
        raise RuntimeError(f"frozen selected-checkpoint provenance mismatch for {corpus}")
    arm_metrics = {
        arm: compact_formal_metrics(root / corpus / arm / "metrics.json", corpus)
        for arm in ("anchor", "broadcast", "core")
    }
    arm_scores = {
        arm: load_scores(root / corpus / arm / "scores.jsonl")
        for arm in ("anchor", "broadcast", "core")
    }
    gt = hdata.gt_arrays(corpus, "test")
    for arm, scores in arm_scores.items():
        if set(scores) != set(gt):
            raise RuntimeError(f"{arm} score cohort mismatch for {corpus}")
        for video_id, score in scores.items():
            if score.shape != np.asarray(gt[video_id]).shape:
                raise RuntimeError(f"{arm} score length mismatch for {corpus}/{video_id}")
            if not np.isfinite(score).all():
                raise RuntimeError(f"non-finite {arm} score for {corpus}/{video_id}")
    carrier, frozen_score_max_error = model_carrier_rates(
        corpus, core_checkpoint_path, arm_scores["core"], device
    )
    core_auc = arm_metrics["core"]["per_video_auc"]
    broadcast_auc = arm_metrics["broadcast"]["per_video_auc"]
    expected_eligible = {
        video_id for video_id, rows in gt.items()
        if len(np.unique(np.asarray(rows))) == 2
    }
    if set(core_auc) != expected_eligible or set(broadcast_auc) != expected_eligible:
        raise RuntimeError(f"formal per-video AUC cohort mismatch for {corpus}")
    eligible = sorted(expected_eligible)
    if set(carrier) != set(gt):
        raise RuntimeError(f"carrier diagnostic cohort mismatch for {corpus}")
    deltas, occupancy, rank_similarity = [], [], []
    modality_rates = {name: [] for name in next(iter(carrier.values()))}
    strata = {"le_one_third": [], "one_to_two_thirds": [],
              "gt_two_thirds": []}
    for video_id in eligible:
        delta = core_auc[video_id] - broadcast_auc[video_id]
        fraction = float(np.asarray(gt[video_id]).mean())
        deltas.append(delta)
        occupancy.append(fraction)
        if fraction <= 1 / 3:
            strata["le_one_third"].append(delta)
        elif fraction <= 2 / 3:
            strata["one_to_two_thirds"].append(delta)
        else:
            strata["gt_two_thirds"].append(delta)
        rank_similarity.append(safe_spearman(
            arm_scores["core"][video_id], arm_scores["broadcast"][video_id]
        ))
        for name in modality_rates:
            modality_rates[name].append(carrier[video_id][name])
    absolute_difference = np.concatenate([
        np.abs(arm_scores["core"][video_id]
               - arm_scores["broadcast"][video_id])
        for video_id in sorted(gt)
    ])
    finite_rank = [row["rho"] for row in rank_similarity if row["rho"] is not None]
    return {
        "formal_test_artifacts": {
            arm: {
                "scores": str((root / corpus / arm / "scores.jsonl").resolve()),
                "metrics": str((root / corpus / arm / "metrics.json").resolve()),
            }
            for arm in ("anchor", "broadcast", "core")
        },
        "frozen_core_checkpoint": {
            "model": str(core_checkpoint_path.resolve()),
            "train_log": str(core_train_log_path.resolve()),
            "selected_epoch": int(core_checkpoint_meta["selected_epoch"]),
        },
        "formal_metrics": {
            arm: {key: value for key, value in values.items()
                  if key != "per_video_auc"}
            for arm, values in arm_metrics.items()
        },
        "core_minus_broadcast_within_per_video": {
            "mean": float(np.mean(deltas)), "median": float(np.median(deltas)),
            "fraction_improved": float(np.mean(np.asarray(deltas) > 0)),
            "fraction_worsened": float(np.mean(np.asarray(deltas) < 0)),
            "n": len(deltas),
        },
        "delta_by_gt_positive_fraction": {
            name: {"n": len(values),
                   "mean": float(np.mean(values)) if values else None}
            for name, values in strata.items()
        },
        "delta_vs_gt_positive_fraction": safe_spearman(deltas, occupancy),
        "core_vs_broadcast_per_video_score_spearman": {
            "mean": float(np.mean(finite_rank)) if finite_rank else None,
            "median": float(np.median(finite_rank)) if finite_rank else None,
            "n_finite": len(finite_rank),
            "n_undefined": len(rank_similarity) - len(finite_rank),
            "definition": (
                "per-video Spearman between frozen core and broadcast frame scores "
                "on the exact within-video AUC eligible cohort"
            ),
        },
        "pooled_absolute_score_difference": {
            "mean": float(np.mean(absolute_difference)),
            "median": float(np.median(absolute_difference)),
        },
        "core_test_carrier_rate_top_third_on_within_eligible_videos": {
            name: {"mean": float(np.mean(values)),
                   "delta_auc_spearman": safe_spearman(values, deltas)}
            for name, values in modality_rates.items()
        },
        "carrier_diagnostic_policy": {
            "frozen_core_prediction_max_absolute_error": frozen_score_max_error,
            "prediction_changed": False,
            "centroid_source": (
                "per-modality mean over this corpus's scoped negative train frames only"
            ),
            "top_fraction_divisor_from_frozen_checkpoint": int(
                torch.load(root / corpus / "core" / "model.pt", map_location="cpu",
                           weights_only=True)["model_args"]["k_proportion"]
            ),
            "diagnostic_is_not_an_inference_arm_or_routing_rule": True,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    root = Path(args.root)
    output = Path(args.out)
    if output.resolve().parent != root.resolve():
        raise RuntimeError("error-analysis output must be written directly in the frozen run root")
    payload = {
        "split": "test", "developmental_error_analysis": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "corpora": {
            corpus: analyze_corpus(root, corpus, torch.device(args.device))
            for corpus in ("hatemm", "hateclipseg")
        },
        "design_use": (
            "Determine whether the tiny core-vs-broadcast gains reflect "
            "meaningful carrier-dependent reranking or nearly unchanged "
            "predictions; do not use any diagnostic as test-time routing."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
