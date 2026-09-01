"""Frozen developmental test premise for temporal path-size choice MIL."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASELINES = REPO / "scripts/reproduction_baselines"
for item in (REPO, BASELINES):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from hate_common import data as hdata  # noqa: E402
from eval_baseline_scores import evaluate_scores  # noqa: E402
from scripts.reproduction_baselines.multihateloc import data as mdata  # noqa: E402
from scripts.reproduction_baselines.multihateloc.model import MultiHateLoc  # noqa: E402
from src.proposal_mil import (  # noqa: E402
    MultimodalPMIL, choice_readout, generate_proposals,
)
from src.scoped_video_protocol import evaluator_test_ids  # noqa: E402


FROZEN_ROOT = REPO / "runs/20260831_multimodal_pmil_baseline/pilot_seed234"
CORPORA = ("hatemm", "hateclipseg")
MODALITIES = tuple(mdata.MODALITIES)
EPS = 1e-6


def load_features(corpus, video_id, device):
    result = {}
    length = None
    for modality in MODALITIES:
        rows = np.load(mdata.feature_path(modality, corpus, video_id)).astype(np.float32)
        if rows.ndim != 2 or rows.shape[1] != mdata.FEATURE_DIMS[modality]:
            raise RuntimeError(f"invalid feature shape {corpus}/{video_id}/{modality}")
        if not np.isfinite(rows).all() or len(rows) == 0:
            raise RuntimeError(f"invalid feature values {corpus}/{video_id}/{modality}")
        if length is None:
            length = len(rows)
        elif len(rows) != length:
            raise RuntimeError(f"unaligned features {corpus}/{video_id}")
        result[modality] = torch.from_numpy(rows).to(device)
    return result, int(length)


@torch.no_grad()
def source_scores(model, features):
    length = len(next(iter(features.values())))
    batch = {name: rows[None] for name, rows in features.items()}
    mask = torch.ones((1, length), dtype=torch.bool, device=next(model.parameters()).device)
    return model(batch, mask)["probs"]["fused"][0].cpu().numpy()


@torch.no_grad()
def proposal_utility(model, features, proposals, device):
    bounds = torch.from_numpy(np.asarray(proposals, dtype=np.float32)).to(device)
    outputs, used = model(features, bounds, training_sample=False)
    if not torch.equal(bounds, used):
        raise RuntimeError("frozen inference unexpectedly changed proposals")
    score = model.full_scores(outputs).cpu().numpy().astype(np.float64)
    if score.shape != (len(proposals),) or not np.isfinite(score).all():
        raise RuntimeError("invalid frozen proposal score")
    score = np.clip(score, EPS, 1.0 - EPS)
    return np.log(score) - np.log1p(-score), score


def max_readout(proposals, scores, length):
    frames = np.full(length, -np.inf, dtype=np.float64)
    for (start, end), score in zip(np.asarray(proposals, dtype=np.int64), scores):
        frames[start:end] = np.maximum(frames[start:end], float(score))
    if not np.isfinite(frames).all():
        raise RuntimeError("proposal set does not cover every frame")
    return frames


def intervals(binary):
    padded = np.pad(np.asarray(binary, dtype=np.int8) > 0, (1, 1))
    changes = np.diff(padded.astype(np.int8))
    return list(zip(np.where(changes == 1)[0], np.where(changes == -1)[0]))


def interval_iou(first, second):
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = first[1] - first[0] + second[1] - second[0] - intersection
    return intersection / max(union, 1)


def oracle_iou(proposals, gold):
    if not gold:
        return np.zeros(len(proposals), dtype=np.float64)
    return np.asarray([max(interval_iou(p, g) for g in gold) for p in proposals])


def near_duplicate(proposals, length):
    base = [tuple(map(int, p)) for p in proposals]
    added = []
    for start, end in base:
        shifted = (start + 1, end + 1)
        if shifted[1] <= length and shifted not in base and shifted not in added:
            added.append(shifted)
    return np.asarray(base + added, dtype=np.float32)


def thin_grid(proposals, length):
    ordered = sorted(tuple(map(int, p)) for p in proposals)
    retained = ordered[::2]
    whole = (0, length)
    if whole not in retained:
        retained.append(whole)
    return np.asarray(sorted(set(retained)), dtype=np.float32)


def finite_spearman(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if len(first) < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return None
    value = float(spearmanr(first, second).statistic)
    return value if np.isfinite(value) else None


def compact_metrics(raw):
    return {
        "pooled_ap": float(raw["pr_auc"]),
        "pooled_roc": float(raw["roc_auc"]),
        "within_roc": float(raw["per_video"]["macro_auc"]),
        "within_n": int(raw["per_video"]["n_videos_both_classes"]),
    }


def mean(values):
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite summary input")
    return float(np.mean(values))


def median(values):
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite summary input")
    return float(np.median(values))


def maximum(values):
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite summary input")
    return float(np.max(values))


def analyze_corpus(corpus, device):
    run_dir = FROZEN_ROOT / corpus
    config = json.loads((run_dir / "config.json").read_text())
    frozen = hdata.load_scores_jsonl(run_dir / "scores.jsonl")
    gt = hdata.gt_arrays(corpus, "test")
    test_ids = evaluator_test_ids(corpus, hdata.load_split(corpus, "test"))
    if set(test_ids) != set(gt) or set(test_ids) != set(frozen):
        raise RuntimeError(f"frozen cohort mismatch for {corpus}")

    source_state = torch.load(
        Path(config["source_checkpoint_path"]), map_location="cpu", weights_only=True
    )
    source = MultiHateLoc(
        {name: mdata.FEATURE_DIMS[name] for name in MODALITIES},
        hidden=int(config["source_model_hidden"]),
        embed=int(config["source_model_embed"]), dropout=0.05,
        k_proportion=3, temperature=0.07,
    ).to(device)
    source.load_state_dict(source_state)
    source.eval()
    pmil = MultimodalPMIL(
        {name: mdata.FEATURE_DIMS[name] for name in MODALITIES},
        hidden=int(config["hidden"]), roi_size=int(config["roi_size"]),
        dropout=0.1,
        max_train_proposals=int(config["maximum_sampled_train_proposals"]),
    ).to(device)
    pmil.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))
    pmil.eval()

    frame_scores = {0: {}, 1: {}}
    top_stats = {0: [], 1: []}
    correction_cases = []
    perturb = {
        name: {
            "paired_frame_undefined": 0,
            **{beta: {"bag_delta": [], "frame_rho": []} for beta in (0, 1)},
        }
        for name in ("duplicate_all", "near_duplicate", "thin_grid")
    }
    max_reconstruction_error = 0.0
    exact_duplicate_frame_error = []

    for index, video_id in enumerate(test_ids, 1):
        features, length = load_features(corpus, video_id, device)
        proposals = generate_proposals(source_scores(source, features))
        utilities, raw_scores = proposal_utility(pmil, features, proposals, device)
        reconstructed = max_readout(proposals, raw_scores, length)
        formal = np.asarray(frozen[video_id]["score_pmil"], dtype=np.float64)
        if (
            formal.ndim != 1 or reconstructed.shape != formal.shape
            or not np.isfinite(formal).all()
        ):
            raise RuntimeError(f"formal score length mismatch {corpus}/{video_id}")
        error = float(np.max(np.abs(reconstructed - formal)))
        max_reconstruction_error = max(max_reconstruction_error, error)
        if error > 1e-6:
            raise RuntimeError(f"frozen P-MIL reconstruction failed {corpus}/{video_id}: {error}")

        per_beta = {}
        for beta in (0, 1):
            frames, posterior, evidence, ps = choice_readout(
                proposals, utilities, length, beta
            )
            frame_scores[beta][video_id] = frames
            top = int(np.argmax(posterior))
            widths = proposals[:, 1] - proposals[:, 0]
            long = widths >= (2.0 * length / 3.0)
            top_stats[beta].append({
                "whole_top": bool(proposals[top, 0] == 0 and proposals[top, 1] == length),
                "top_duration_ratio": float(widths[top] / length),
                "near_whole_top": bool(long[top]),
                "long_posterior_mass": float(posterior[long].sum()),
            })
            per_beta[beta] = (frames, posterior, evidence, ps)

        gold = intervals(gt[video_id])
        if gold:
            ious = oracle_iou(proposals.astype(np.int64), gold)
            best = int(np.argmax(ious))
            raw_top = int(np.argmax(raw_scores))
            if ious[best] >= 0.5 and ious[raw_top] < 0.3:
                correction_cases.append({
                    "error_log_ps_minus_best_log_ps": float(
                        np.log(per_beta[1][3][raw_top]) - np.log(per_beta[1][3][best])
                    )
                })

        variants = {
            "duplicate_all": np.concatenate((proposals, proposals), axis=0),
            "near_duplicate": near_duplicate(proposals, length),
            "thin_grid": thin_grid(proposals, length),
        }
        for name, variant in variants.items():
            if name == "duplicate_all":
                variant_utilities = np.concatenate((utilities, utilities))
            else:
                variant_utilities, _ = proposal_utility(pmil, features, variant, device)
            paired_rho = {}
            for beta in (0, 1):
                altered_frames, _, altered_evidence, _ = choice_readout(
                    variant, variant_utilities, length, beta
                )
                base_frames, _, base_evidence, _ = per_beta[beta]
                perturb[name][beta]["bag_delta"].append(
                    abs(altered_evidence - base_evidence)
                )
                paired_rho[beta] = finite_spearman(base_frames, altered_frames)
                if name == "duplicate_all" and beta == 1:
                    exact_duplicate_frame_error.append(
                        float(np.max(np.abs(base_frames - altered_frames)))
                    )
            # Compare beta=0 and beta=1 on exactly the same videos.  If either
            # ranking is undefined, the perturbation evidence is insufficient
            # and the fail-closed gate below rejects the premise.
            if paired_rho[0] is None or paired_rho[1] is None:
                perturb[name]["paired_frame_undefined"] += 1
            else:
                for beta in (0, 1):
                    perturb[name][beta]["frame_rho"].append(paired_rho[beta])
        if index % 50 == 0:
            print(f"{corpus} frozen premise {index}/{len(test_ids)}", flush=True)

    hate_ids = {video_id for video_id, value in gt.items() if np.asarray(value).max() > 0}
    metrics = {
        f"beta_{beta}": compact_metrics(evaluate_scores(scores, gt, hate_ids))
        for beta, scores in frame_scores.items()
    }
    top_summary = {}
    for beta in (0, 1):
        rows = top_stats[beta]
        top_summary[f"beta_{beta}"] = {
            "exact_whole_top_fraction": mean([r["whole_top"] for r in rows]),
            "top_duration_ratio_median": median([r["top_duration_ratio"] for r in rows]),
            "near_whole_top_fraction": mean([r["near_whole_top"] for r in rows]),
            "long_proposal_posterior_mass_mean": mean([r["long_posterior_mass"] for r in rows]),
        }
    perturb_summary = {}
    for name, by_beta in perturb.items():
        perturb_summary[name] = {
            f"beta_{beta}": {
                "bag_log_evidence_absolute_change_mean": mean(values["bag_delta"]),
                "bag_log_evidence_absolute_change_max": maximum(values["bag_delta"]),
                "frame_spearman_mean": mean(values["frame_rho"]),
                "frame_spearman_n": len(values["frame_rho"]),
            }
            for beta, values in by_beta.items() if isinstance(beta, int)
        }
        perturb_summary[name]["paired_frame_undefined"] = int(
            by_beta["paired_frame_undefined"]
        )
    return {
        "corpus": corpus,
        "n_test_videos": len(test_ids),
        "formal_frozen_score_reconstruction_max_abs_error": max_reconstruction_error,
        "metrics": metrics,
        "top_and_length": top_summary,
        "correctable_wrong_top_cases": {
            "n": len(correction_cases),
            "error_minus_best_log_path_size_mean": mean([
                row["error_log_ps_minus_best_log_ps"] for row in correction_cases
            ]),
            "required_direction": "negative",
        },
        "candidate_set_perturbations": perturb_summary,
        "duplicate_all_beta_1_frame_max_abs_error": max(exact_duplicate_frame_error),
    }


def finite_number(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def gate(corpora):
    if set(corpora) != set(CORPORA):
        return {
            "pass": False,
            "decision": "STOP_BEFORE_FORMAL_METHOD",
            "failures": ["corpus_set_mismatch"],
        }
    failures = []
    for corpus, result in corpora.items():
        m0, m1 = result["metrics"]["beta_0"], result["metrics"]["beta_1"]
        t0 = result["top_and_length"]["beta_0"]
        t1 = result["top_and_length"]["beta_1"]
        checks = {
            "within_non_decrease": (
                finite_number(m0["within_roc"])
                and finite_number(m1["within_roc"])
                and m1["within_roc"] >= m0["within_roc"]
            ),
            "whole_top_decrease": (
                finite_number(t0["exact_whole_top_fraction"])
                and finite_number(t1["exact_whole_top_fraction"])
                and t1["exact_whole_top_fraction"] < t0["exact_whole_top_fraction"]
            ),
            "median_duration_nonincrease": (
                finite_number(t0["top_duration_ratio_median"])
                and finite_number(t1["top_duration_ratio_median"])
                and t1["top_duration_ratio_median"] <= t0["top_duration_ratio_median"]
            ),
            "near_whole_top_decrease": (
                finite_number(t0["near_whole_top_fraction"])
                and finite_number(t1["near_whole_top_fraction"])
                and t1["near_whole_top_fraction"] < t0["near_whole_top_fraction"]
            ),
            "long_mass_nonincrease": (
                finite_number(t0["long_proposal_posterior_mass_mean"])
                and finite_number(t1["long_proposal_posterior_mass_mean"])
                and t1["long_proposal_posterior_mass_mean"] <= t0["long_proposal_posterior_mass_mean"]
            ),
            "wrong_top_has_lower_path_size": (
                result["correctable_wrong_top_cases"]["n"] > 0
                and finite_number(result["correctable_wrong_top_cases"]["error_minus_best_log_path_size_mean"])
                and result["correctable_wrong_top_cases"]["error_minus_best_log_path_size_mean"] < 0
            ),
            "duplicate_exact_bag": (
                finite_number(result["candidate_set_perturbations"]["duplicate_all"]["beta_1"]["bag_log_evidence_absolute_change_max"])
                and result["candidate_set_perturbations"]["duplicate_all"]["beta_1"]["bag_log_evidence_absolute_change_max"] <= 1e-10
            ),
            "duplicate_exact_frames": (
                finite_number(result["duplicate_all_beta_1_frame_max_abs_error"])
                and result["duplicate_all_beta_1_frame_max_abs_error"] <= 1e-10
            ),
        }
        for name in ("near_duplicate", "thin_grid"):
            p = result["candidate_set_perturbations"][name]
            checks[f"{name}_bag_more_stable"] = (
                finite_number(p["beta_0"]["bag_log_evidence_absolute_change_mean"])
                and finite_number(p["beta_1"]["bag_log_evidence_absolute_change_mean"])
                and
                p["beta_1"]["bag_log_evidence_absolute_change_mean"]
                < p["beta_0"]["bag_log_evidence_absolute_change_mean"]
            )
            checks[f"{name}_frames_more_stable"] = (
                p["paired_frame_undefined"] == 0
                and p["beta_0"]["frame_spearman_n"] == result["n_test_videos"]
                and p["beta_1"]["frame_spearman_n"] == result["n_test_videos"]
                and finite_number(p["beta_0"]["frame_spearman_mean"])
                and finite_number(p["beta_1"]["frame_spearman_mean"])
                and
                p["beta_1"]["frame_spearman_mean"]
                > p["beta_0"]["frame_spearman_mean"]
            )
        for name, passed in checks.items():
            if not passed:
                failures.append(f"{corpus}:{name}")
        result["premise_checks"] = checks
    return {
        "pass": not failures,
        "decision": "ALLOW_FORMAL_METHOD" if not failures else "STOP_BEFORE_FORMAL_METHOD",
        "failures": failures,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    expected_root = (REPO / "runs/20260831_temporal_path_size_choice_mil").resolve()
    if expected_root not in out.parents:
        raise RuntimeError(f"output must be below {expected_root}")
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    result = {
        "split": "test",
        "developmental_test_evidence": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "frozen_source_run": str(FROZEN_ROOT),
        "fixed_policy": {
            "utility": "logit(clamp(mean modality hate*attention*completeness,1e-6,1-1e-6))",
            "path_size": "mean over proposal seconds of inverse candidate occupancy",
            "core_beta": 1,
            "frame_readout": "sum proposal-conditional posterior over proposals covering the second",
            "outside_option": "not fit in frozen premise",
            "parameter_or_formula_search": False,
        },
        "corpora": {corpus: analyze_corpus(corpus, device) for corpus in CORPORA},
    }
    result["verdict"] = gate(result["corpora"])
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["verdict"], indent=2))


if __name__ == "__main__":
    main()
