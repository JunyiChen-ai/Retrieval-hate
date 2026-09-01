#!/usr/bin/env python
"""Developmental test complementarity diagnostic using the shared evaluator."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reproduction_baselines"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402

OLD = Path("/home/jehc223/Hate-follow-up/results/reproduction")
SIGNALS = {
    "hatemm": {
        "powa": (OLD / "powa_macil/final_maskfix_finetune_hatemm_seed234_e5/hatemm/scores.jsonl", "score_powa"),
        "multihateloc": (OLD / "official_val/final/multihateloc/hatemm/seed_234/hatemm/scores.jsonl", "score_fused"),
        "vera": (OLD / "official_val/final/vera/hatemm/seed_234/scores.jsonl", "score_official_postprocessed"),
        "macilsd": (OLD / "official_val/final/macilsd/hatemm/seed_234/scores.jsonl", "score_av"),
        "lexical": (ROOT / "runs/20260831_video_label_lexical_locality/premise/hatemm/scores.jsonl", "score_lexical"),
    },
    "hateclipseg": {
        "powa": (ROOT / "runs/20260831_powa_starting_point/hcs_maskfix_seed234/scores.jsonl", "score_powa"),
        "multihateloc": (OLD / "official_val/final/multihateloc/hateclipseg/seed_234/hateclipseg/scores.jsonl", "score_fused"),
        "vera": (OLD / "official_val/final/vera/hateclipseg/seed_234/scores.jsonl", "score_official_postprocessed"),
        "dsanet": (OLD / "official_val/final/dsanet/hateclipseg/seed_234/scores.jsonl", "score_mlp"),
        "lexical": (ROOT / "runs/20260831_video_label_lexical_locality/premise/hateclipseg/scores.jsonl", "score_lexical"),
    },
}
SOTA = {
    "hatemm": {"pr_auc": 0.5938315566, "roc_auc": 0.8161837922,
               "within": 0.6315317180},
    "hateclipseg": {"pr_auc": 0.6193710950, "roc_auc": 0.6050224699,
                    "within": 0.5619078936},
}
WEIGHTS = [i / 20.0 for i in range(21)]
NORMALIZATION = (
    "Per signal, empirical CDF over the complete test-frame pool; "
    "scipy rankdata(method='average'), scaled to [0,1]."
)


def load_branch(path: Path, branch: str) -> dict[str, np.ndarray]:
    rows = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            vid = str(row["video_id"])
            if vid in rows:
                raise ValueError(f"duplicate score ID in {path}: {vid}")
            rows[vid] = np.asarray(row[branch], dtype=np.float64)
    return rows


def global_rank(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ids = sorted(scores)
    lengths = [len(scores[v]) for v in ids]
    values = np.concatenate([scores[v] for v in ids])
    ranked = (rankdata(values, method="average") - 1.0) / max(len(values) - 1, 1)
    out = {}
    start = 0
    for vid, length in zip(ids, lengths):
        out[vid] = ranked[start:start + length]
        start += length
    return out


def compact(result: dict) -> dict:
    return {"pr_auc": float(result["pr_auc"]),
            "roc_auc": float(result["roc_auc"]),
            "within": float(result["per_video"]["macro_auc"]),
            "n_videos": int(result["n_videos"]),
            "n_frames": int(result["n_frames"])}


def all_sota(row: dict, threshold: dict) -> bool:
    return bool(row["pr_auc"] >= threshold["pr_auc"] and
                row["roc_auc"] >= threshold["roc_auc"] and
                row["within"] >= threshold["within"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    provenance = {
        corpus: {
            name: {"path": str(path), "branch": branch, "seed": 234}
            for name, (path, branch) in specs.items()
        }
        for corpus, specs in SIGNALS.items()
    }
    config = {
        "analysis_scope": {
            "pairwise": "fixed global-CDF convex pair grid only",
            "selector": "whole-video single-signal within-AUC oracle readout only",
            "negative_result_limit": (
                "Does not rule out nonlinear/per-frame fusion, distillation, "
                "or selectors using another objective."
            ),
        },
        "split": "test",
        "developmental_test_evidence": True,
        "fixed_seed": 234,
        "normalization": NORMALIZATION,
        "pairwise_left_weights": WEIGHTS,
        "input_signals": provenance,
        "sota_thresholds": SOTA,
    }
    output = {**config, "corpora": {}}
    for corpus, specs in SIGNALS.items():
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        hate_ids = {v for v in gt if labels.get(v) == 1}
        signals = {name: load_branch(path, branch)
                   for name, (path, branch) in specs.items()}
        for name, score in signals.items():
            if set(score) != set(gt):
                raise RuntimeError(f"{corpus}/{name}: non-exact test cohort")
            for vid in gt:
                if score[vid].shape != gt[vid].shape or not np.isfinite(score[vid]).all():
                    raise RuntimeError(f"{corpus}/{name}/{vid}: invalid score contract")
        ranked = {name: global_rank(score) for name, score in signals.items()}
        singles = {name: compact(evaluate_scores(score, gt, hate_ids))
                   for name, score in signals.items()}
        pairs = []
        for left, right in itertools.combinations(sorted(ranked), 2):
            for weight in WEIGHTS:
                blend = {v: weight * ranked[left][v] +
                         (1.0 - weight) * ranked[right][v] for v in gt}
                row = compact(evaluate_scores(blend, gt, hate_ids))
                pairs.append({"left": left, "right": right,
                              "left_weight": weight, **row,
                              "all_sota": all_sota(row, SOTA[corpus])})
        default = max(singles, key=lambda name: singles[name]["roc_auc"])
        selector_scores = {}
        choices = Counter()
        for vid in sorted(gt):
            y = gt[vid]
            if vid in hate_ids and len(np.unique(y)) == 2:
                aucs = {}
                for name in ranked:
                    one = evaluate_scores({vid: ranked[name][vid]}, {vid: y}, {vid})
                    aucs[name] = float(one["per_video"]["macro_auc"])
                chosen = max(sorted(aucs), key=lambda name: aucs[name])
            else:
                chosen = default
            choices[chosen] += 1
            selector_scores[vid] = ranked[chosen][vid]
        selector_readout = compact(evaluate_scores(selector_scores, gt, hate_ids))
        selector_readout["all_sota"] = all_sota(selector_readout, SOTA[corpus])
        selector_readout["within_above_sota_gate"] = bool(
            selector_readout["within"] >= SOTA[corpus]["within"])
        output["corpora"][corpus] = {
            "sota_thresholds": SOTA[corpus], "single_signals": singles,
            "exact_cohort": {"n_videos": len(gt),
                             "n_frames": int(sum(len(y) for y in gt.values()))},
            "pairwise_grid": pairs,
            "n_pairwise_all_sota": sum(row["all_sota"] for row in pairs),
            "best_pair_by_within": max(pairs, key=lambda row: row["within"]),
            "best_pair_by_min_sota_ratio": max(
                pairs, key=lambda row: min(
                    row["pr_auc"] / SOTA[corpus]["pr_auc"],
                    row["roc_auc"] / SOTA[corpus]["roc_auc"],
                    row["within"] / SOTA[corpus]["within"])),
            "whole_video_signal_within_oracle_readout": selector_readout,
            "within_oracle_choice_counts": dict(choices),
            "ineligible_video_default_signal_selected_by_test_pooled_roc": default,
        }
        print(corpus, "pairwise all-SOTA", output["corpora"][corpus]["n_pairwise_all_sota"],
              "whole-video selector readout", selector_readout,
              "choices", dict(choices), flush=True)
    output["joint_fixed_pairwise_grid_has_all_sota"] = bool(all(
        row["n_pairwise_all_sota"] > 0 for row in output["corpora"].values()))
    output["joint_whole_video_selector_readout_is_all_sota"] = bool(all(
        row["whole_video_signal_within_oracle_readout"]["all_sota"]
        for row in output["corpora"].values()))
    target = Path(args.out).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    (target.parent / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    target.write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
