#!/usr/bin/env python
"""Attribute fixed-pair gains to aligned lexical timing on developmental test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reproduction_baselines"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402

OLD = Path("/home/jehc223/Hate-follow-up/results/reproduction")
SPECS = {
    "hatemm": {
        "base": (OLD / "powa_macil/final_maskfix_finetune_hatemm_seed234_e5/hatemm/scores.jsonl",
                 "score_powa", "powa"),
        "lexical": (ROOT / "runs/20260831_video_label_lexical_locality/premise/hatemm/scores.jsonl",
                    "score_lexical"),
        "speech_branch": "score_speech",
        "lexical_weight": 0.35,
        "sota": {"pr_auc": 0.5938315566, "roc_auc": 0.8161837922,
                 "within": 0.6315317180},
    },
    "hateclipseg": {
        "base": (OLD / "official_val/final/vera/hateclipseg/seed_234/scores.jsonl",
                 "score_official_postprocessed", "vera"),
        "lexical": (ROOT / "runs/20260831_video_label_lexical_locality/premise/hateclipseg/scores.jsonl",
                    "score_lexical"),
        "speech_branch": "score_speech",
        "lexical_weight": 0.05,
        "sota": {"pr_auc": 0.6193710950, "roc_auc": 0.6050224699,
                 "within": 0.5619078936},
    },
}
N_PHASES = 16


def load_branch(path: Path, branch: str) -> dict[str, np.ndarray]:
    rows = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            vid = str(row["video_id"])
            if vid in rows:
                raise ValueError(f"duplicate score ID in {path}: {vid}")
            values = np.asarray(row[branch], dtype=np.float64)
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError(f"invalid score branch in {path}: {vid}")
            rows[vid] = values
    return rows


def global_rank(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ids = sorted(scores)
    lengths = [len(scores[v]) for v in ids]
    values = np.concatenate([scores[v] for v in ids])
    ranked = (rankdata(values, method="average") - 1.0) / max(len(values) - 1, 1)
    out, start = {}, 0
    for vid, length in zip(ids, lengths):
        out[vid] = ranked[start:start + length]
        start += length
    return out


def compact(result: dict) -> dict:
    return {
        "pr_auc": float(result["pr_auc"]),
        "roc_auc": float(result["roc_auc"]),
        "within": float(result["per_video"]["macro_auc"]),
        "n_videos": int(result["n_videos"]),
        "n_frames": int(result["n_frames"]),
    }


def all_sota(row: dict, threshold: dict) -> bool:
    return bool(row["pr_auc"] >= threshold["pr_auc"] and
                row["roc_auc"] >= threshold["roc_auc"] and
                row["within"] >= threshold["within"])


def one_auc(score: np.ndarray, y: np.ndarray) -> float:
    result = evaluate_scores({"v": score}, {"v": y}, {"v"})
    return float(result["per_video"]["macro_auc"])


def phase_offset(length: int, phase: int) -> int:
    if length <= 1:
        return 0
    raw = int(round(phase * length / float(N_PHASES + 1)))
    return min(max(raw, 1), length - 1)


def summarize_stratum(rows: list[dict], key: str) -> dict:
    values = np.asarray([r[key] for r in rows], dtype=float)
    cut = float(np.median(values))
    out = {"median_cut": cut, "groups": {}}
    for name, keep in (("low_or_equal", values <= cut), ("high", values > cut)):
        selected = [r for r, flag in zip(rows, keep) if flag]
        out["groups"][name] = {
            "n_videos": len(selected),
            "aligned_minus_base_auc_mean": (
                float(np.mean([r["aligned_minus_base_auc"] for r in selected]))
                if selected else None),
            "aligned_minus_shift_mean_auc_mean": (
                float(np.mean([r["aligned_minus_shift_mean_auc"] for r in selected]))
                if selected else None),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    config = {
        "developmental_test_evidence": True,
        "split": "test",
        "fixed_seed": 234,
        "n_phase_shift_controls": N_PHASES,
        "normalization": "global test-frame empirical CDF, average ties",
        "phase_rule": "per-video circular offset round(j*T/17), clipped to [1,T-1]",
        "interpretation_gate": {
            "aligned_all_sota_both_corpora": True,
            "aligned_minus_shift_mean_within_min": 0.020,
            "aligned_strictly_beats_at_least_n_shifts": 14,
        },
        "inputs": {
            corpus: {
                "base": {"path": str(spec["base"][0]), "branch": spec["base"][1],
                         "name": spec["base"][2], "seed": 234},
                "lexical": {"path": str(spec["lexical"][0]),
                            "branch": spec["lexical"][1], "seed": 234},
                "speech_presence_branch": spec["speech_branch"],
                "lexical_weight": spec["lexical_weight"],
                "sota": spec["sota"],
            }
            for corpus, spec in SPECS.items()
        },
    }
    output = {**config, "corpora": {}}
    for corpus, spec in SPECS.items():
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        hate_ids = {v for v in gt if labels.get(v) == 1}
        base_raw = load_branch(spec["base"][0], spec["base"][1])
        lex_raw = load_branch(spec["lexical"][0], spec["lexical"][1])
        speech = load_branch(spec["lexical"][0], spec["speech_branch"])
        if (set(base_raw) != set(gt) or set(lex_raw) != set(gt) or
                set(speech) != set(gt)):
            raise RuntimeError(f"{corpus}: non-exact test cohort")
        for vid in gt:
            if (base_raw[vid].shape != gt[vid].shape or
                    lex_raw[vid].shape != gt[vid].shape or
                    speech[vid].shape != gt[vid].shape):
                raise RuntimeError(f"{corpus}/{vid}: score/GT shape mismatch")
        base = global_rank(base_raw)
        lexical = global_rank(lex_raw)
        weight = float(spec["lexical_weight"])
        aligned_scores = {v: weight * lexical[v] + (1.0 - weight) * base[v]
                          for v in gt}
        base_metrics = compact(evaluate_scores(base, gt, hate_ids))
        aligned_metrics = compact(evaluate_scores(aligned_scores, gt, hate_ids))
        aligned_metrics["all_sota"] = all_sota(aligned_metrics, spec["sota"])
        shifted_metrics, shifted_scores = [], []
        for phase in range(1, N_PHASES + 1):
            arm = {
                v: weight * np.roll(lexical[v], phase_offset(len(lexical[v]), phase)) +
                (1.0 - weight) * base[v]
                for v in gt
            }
            shifted_scores.append(arm)
            row = compact(evaluate_scores(arm, gt, hate_ids))
            row.update({"phase": phase,
                        "all_sota": all_sota(row, spec["sota"])})
            shifted_metrics.append(row)
        eligible_rows = []
        for vid in sorted(hate_ids):
            y = gt[vid]
            if len(np.unique(y)) != 2:
                continue
            base_auc = one_auc(base[vid], y)
            aligned_auc = one_auc(aligned_scores[vid], y)
            shift_aucs = [one_auc(arm[vid], y) for arm in shifted_scores]
            _, counts = np.unique(base[vid], return_counts=True)
            tied_frames = int(counts[counts > 1].sum())
            tau = kendalltau(base[vid], aligned_scores[vid], variant="b").statistic
            eligible_rows.append({
                "video_id": vid,
                "n_frames": len(y),
                "gt_positive_fraction": float(np.mean(y)),
                "speech_coverage": float(np.mean(speech[vid] > 0.0)),
                "lexical_std": float(np.std(lexical[vid])),
                "base_tied_frame_fraction": tied_frames / float(len(y)),
                "base_aligned_kendall_tau_b": float(tau) if np.isfinite(tau) else None,
                "base_auc": base_auc,
                "aligned_auc": aligned_auc,
                "shift_auc_mean": float(np.mean(shift_aucs)),
                "aligned_minus_base_auc": aligned_auc - base_auc,
                "aligned_minus_shift_mean_auc": aligned_auc - float(np.mean(shift_aucs)),
            })
        shifted_within = np.asarray([r["within"] for r in shifted_metrics])
        attribution = {
            "aligned_minus_shift_mean_within": (
                aligned_metrics["within"] - float(np.mean(shifted_within))),
            "aligned_minus_best_shift_within": (
                aligned_metrics["within"] - float(np.max(shifted_within))),
            "n_shifts_strictly_below_aligned_within": int(
                np.sum(shifted_within < aligned_metrics["within"])),
            "n_shift_controls_all_sota": int(sum(r["all_sota"] for r in shifted_metrics)),
            "eligible_video_mean_aligned_minus_base_auc": float(np.mean(
                [r["aligned_minus_base_auc"] for r in eligible_rows])),
            "eligible_video_mean_aligned_minus_shift_mean_auc": float(np.mean(
                [r["aligned_minus_shift_mean_auc"] for r in eligible_rows])),
        }
        gate = {
            "aligned_all_sota": aligned_metrics["all_sota"],
            "aligned_minus_shift_mean_within_at_least_020":
                attribution["aligned_minus_shift_mean_within"] >= 0.020,
            "aligned_beats_at_least_14_of_16_shifts":
                attribution["n_shifts_strictly_below_aligned_within"] >= 14,
        }
        gate["pass"] = bool(all(gate.values()))
        output["corpora"][corpus] = {
            "exact_cohort": {"n_videos": len(gt),
                             "n_frames": int(sum(len(y) for y in gt.values()))},
            "base": base_metrics,
            "aligned": aligned_metrics,
            "shift_controls": shifted_metrics,
            "attribution": attribution,
            "strata": {
                key: summarize_stratum(eligible_rows, key)
                for key in ("speech_coverage", "lexical_std",
                            "base_tied_frame_fraction", "gt_positive_fraction")
            },
            "eligible_video_rows": eligible_rows,
            "gate": gate,
        }
        print(corpus, "aligned", aligned_metrics, "attribution", attribution,
              "gate", gate, flush=True)
    output["joint_gate_pass"] = bool(all(
        row["gate"]["pass"] for row in output["corpora"].values()))
    output["decision"] = (
        "LEXICAL_TIMING_LOAD_BEARING" if output["joint_gate_pass"]
        else "CLOSE_LEXICAL_COMPLEMENTARITY_MECHANISM")
    target = Path(args.out).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    (target.parent / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    target.write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
