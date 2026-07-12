#!/usr/bin/env python
"""Frozen SSR B1 audit, conditional-information, and oracle gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from ssr_common import (  # noqa: E402
    atomic_write_json, canonical_json, id_hash, load_config, rank_bin,
    read_audit_csv, read_jsonl, resolve, sha256_file, wilson,
)


def cell_path(cfg, dataset, family, gate):
    return resolve(cfg, "artifacts") / "b1" / "cells" / \
        "{}_{}_{}.json".format(dataset, family, gate)


def load_authoritative(cfg, dataset):
    root = resolve(cfg, "artifacts") / "aggregate" / dataset
    manifest = json.load(open(root / "manifest.json", encoding="utf-8"))
    if manifest["status"] != "COMPLETED" or \
            manifest["config_sha256"] != cfg["computed_config_sha256"]:
        raise RuntimeError("aggregate upstream invalid")
    for name, digest in manifest["outputs"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError("aggregate artifact changed: {}".format(name))
    return (manifest, read_jsonl(root / "graph.jsonl"),
            read_jsonl(root / "directed_arcs.jsonl"))


def kappa(a, b):
    values = ("valid", "invalid", "unclear")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / float(max(n, 1))
    pa = {v: sum(x == v for x in a) / float(max(n, 1)) for v in values}
    pb = {v: sum(x == v for x in b) / float(max(n, 1)) for v in values}
    pe = sum(pa[v] * pb[v] for v in values)
    return (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0


def run_audit(cfg, dataset, family, run_id):
    audit_n = int(cfg["b1"]["audit_n"])
    root = resolve(cfg, "artifacts") / "audit" / dataset / family
    pack = read_jsonl(root / "audit_pack.jsonl")
    expected = [x["canonical_pair_id"] for x in pack]
    if len(expected) != audit_n or len(set(expected)) != audit_n:
        raise RuntimeError("audit pack must contain exactly {} unique records".format(audit_n))
    tables = {name: read_audit_csv(root / "{}.csv".format(name))
              for name in ("A1", "A2", "ADJ")}
    maps = {}
    allowed = {"valid", "invalid", "unclear"}
    for name, rows in tables.items():
        if [x.get("canonical_pair_id") for x in rows] != expected:
            raise RuntimeError("{} audit IDs/order mismatch".format(name))
        maps[name] = {x["canonical_pair_id"]: x.get("validity", "").lower()
                      for x in rows}
    if any(maps[name][x] not in allowed for name in ("A1", "A2") for x in expected):
        raise RuntimeError("A1/A2 judgments incomplete or invalid")
    final = []
    disagreements = 0
    for pair_id in expected:
        a, b = maps["A1"][pair_id], maps["A2"][pair_id]
        if a == b:
            final.append(a)
        else:
            disagreements += 1
            adj = maps["ADJ"][pair_id]
            if adj not in allowed:
                raise RuntimeError("ADJ missing for disagreement {}".format(pair_id))
            final.append(adj)
    valid = sum(x == "valid" for x in final)
    interval = wilson(valid, audit_n, float(cfg["b1"]["wilson_z"]))
    passed = float(interval["lower"]) >= float(cfg["b1"]["wilson_lower_min"])
    result = {
        "run_id": run_id, "gate": "audit", "dataset": dataset, "family": family,
        "status": "GO" if passed else "FAIL", "valid": valid, "n": audit_n,
        "point_precision": interval["point"], "wilson_lower": interval["lower"],
        "wilson_upper": interval["upper"],
        "annotator_exact_agreement": 1.0 - disagreements / float(audit_n),
        "cohen_kappa": kappa([maps["A1"][x] for x in expected],
                             [maps["A2"][x] for x in expected]),
        "disagreements": disagreements,
        "unclear_counts_invalid": True,
        "audit_used_for_training_or_graph": False,
        "segment_annotation_performed": False,
        "source_hashes": {name: sha256_file(root / name) for name in
                          ("audit_pack.jsonl", "A1.csv", "A2.csv", "ADJ.csv")},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_write_json(cell_path(cfg, dataset, family, "audit"), result)
    return result


def load_oof_predictions(cfg, dataset):
    preds = []
    for fold in range(int(cfg["folds"]["n_splits"])):
        path = resolve(cfg, "artifacts") / "oof" / dataset / \
               "fold{}".format(fold) / "predictions.json"
        preds.extend(json.load(open(path, encoding="utf-8")))
    by_id = {x["query_id"]: x for x in preds}
    folds = json.load(open(resolve(cfg, "artifacts") / "folds" /
                           "{}.json".format(dataset), encoding="utf-8"))
    if len(by_id) != len(preds) or set(by_id) != {x["id"] for x in folds["records"]}:
        raise RuntimeError("OOF prediction coverage mismatch")
    return by_id


def run_oracle(cfg, dataset, family, run_id):
    _manifest, _graph, arcs = load_authoritative(cfg, dataset)
    preds = load_oof_predictions(cfg, dataset)
    touched = {x["query_id"] for x in arcs
               if x["candidate_family"] == family and int(x["accepted"]) == 1
               and int(x["event"]) == 1 and int(x["baseline_error"]) == 1}
    ordered = sorted(preds)
    y = np.asarray([int(preds[x]["query_label"]) for x in ordered])
    base = np.asarray([int(preds[x]["prediction"]) for x in ordered])
    oracle = base.copy()
    for i, qid in enumerate(ordered):
        if qid in touched:
            oracle[i] = y[i]
    bacc, oacc = accuracy_score(y, base), accuracy_score(y, oracle)
    bf1 = f1_score(y, base, average="macro", zero_division=0)
    of1 = f1_score(y, oracle, average="macro", zero_division=0)
    gain_acc, gain_f1 = oacc - bacc, of1 - bf1
    threshold = float(cfg["b1"]["oracle_gain_min"])
    passed = gain_acc >= threshold and gain_f1 >= threshold
    result = {
        "run_id": run_id, "gate": "oracle", "dataset": dataset, "family": family,
        "status": "GO" if passed else "FAIL",
        "baseline_accuracy": bacc, "oracle_accuracy": oacc,
        "accuracy_gain": gain_acc, "baseline_macro_f1": bf1,
        "oracle_macro_f1": of1, "macro_f1_gain": gain_f1,
        "touched_errors": len(touched), "unique_query_coverage": len(touched),
        "n_oof_videos": len(ordered),
        "required_realized_fraction_accuracy": 0.03 / gain_acc if gain_acc > 0 else None,
        "required_realized_fraction_macro_f1": 0.03 / gain_f1 if gain_f1 > 0 else None,
        "feasibility_ceiling_not_model_result": True,
        "only_gold": "video_level_binary_label", "segment_gold_used": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_write_json(cell_path(cfg, dataset, family, "oracle"), result)
    return result


def grouped_oof(X_reduced, flag, y, groups, cfg):
    pr, pf = np.zeros(len(y)), np.zeros(len(y))
    for held in range(int(cfg["b1"]["conditional_groups"])):
        train, test = groups != held, groups == held
        if not bool(test.any()) or len(np.unique(y[train])) != 2:
            raise RuntimeError("empty held group or one-class training outcome")
        xr_tr, xr_te = X_reduced[train].copy(), X_reduced[test].copy()
        mean, std = xr_tr[:, :3].mean(axis=0), xr_tr[:, :3].std(axis=0)
        std[std == 0] = 1.0
        xr_tr[:, :3] = (xr_tr[:, :3] - mean) / std
        xr_te[:, :3] = (xr_te[:, :3] - mean) / std
        xf_tr = np.concatenate([xr_tr, flag[train, None]], axis=1)
        xf_te = np.concatenate([xr_te, flag[test, None]], axis=1)
        kwargs = dict(C=float(cfg["b1"]["logistic_C"]),
                      solver=cfg["b1"]["logistic_solver"],
                      max_iter=int(cfg["b1"]["logistic_max_iter"]),
                      tol=float(cfg["b1"]["logistic_tol"]), fit_intercept=True,
                      class_weight=None, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            reduced, full = LogisticRegression(**kwargs), LogisticRegression(**kwargs)
            reduced.fit(xr_tr, y[train])
            full.fit(xf_tr, y[train])
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError("logistic convergence failure")
        pr[test] = reduced.predict_proba(xr_te)[:, 1]
        pf[test] = full.predict_proba(xf_te)[:, 1]
    pr, pf = np.clip(pr, 1e-6, 1 - 1e-6), np.clip(pf, 1e-6, 1 - 1e-6)
    return pr, pf


def metric_delta(y, pr, pf):
    return (float(log_loss(y, pr, labels=[0, 1]) - log_loss(y, pf, labels=[0, 1])),
            float(roc_auc_score(y, pf) - roc_auc_score(y, pr)))


def bin_index(values, q):
    edges = np.quantile(values, np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(edges, values, side="right"), [float(x) for x in edges]


def cond_group(query_id):
    digest = hashlib.sha256(("cond-v1|" + str(query_id)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 5


def run_conditional(cfg, dataset, family, run_id):
    audit = json.load(open(cell_path(cfg, dataset, family, "audit"), encoding="utf-8"))
    if audit["status"] != "GO":
        raise RuntimeError("conditional gate requires audit GO")
    _manifest, graph, all_arcs = load_authoritative(cfg, dataset)
    arcs = [x for x in all_arcs if x["candidate_family"] == family]
    if not arcs:
        raise RuntimeError("no candidate arcs")
    y = np.asarray([int(x["event"]) for x in arcs], dtype=int)
    if len(np.unique(y)) != 2:
        result = {"run_id": run_id, "gate": "conditional", "dataset": dataset,
                  "family": family, "status": "FAIL", "reason": "one_class_outcome",
                  "p_raw": 1.0, "delta_nll": None, "delta_auc": None,
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        atomic_write_json(cell_path(cfg, dataset, family, "conditional"), result)
        return result
    X = []
    for x in arcs:
        fold_onehot = [int(int(x["outer_fold"]) == k) for k in range(5)]
        X.append([float(x["cosine"]), float(x["rank"]) / float(x["memory_n"]),
                  float(x["normalized_abs_margin"]), int(x["query_label"])] +
                 fold_onehot + [int(x["baseline_error"])])
    X = np.asarray(X, dtype=float)
    flag = np.asarray([int(x["accepted"]) for x in arcs], dtype=float)
    groups = np.asarray([cond_group(x["query_id"]) for x in arcs], dtype=int)
    pr, pf = grouped_oof(X, flag, y, groups, cfg)
    delta_nll, delta_auc = metric_delta(y, pr, pf)

    # Freeze bins from every selected pre-MLLM candidate arc in this dataset.
    cos_bins, cos_edges = bin_index(np.asarray([float(x["cosine"]) for x in all_arcs]), 10)
    mar_bins, mar_edges = bin_index(
        np.asarray([float(x["normalized_abs_margin"]) for x in all_arcs]), 4)
    arc_bin = {}
    for x, cb, mb in zip(all_arcs, cos_bins, mar_bins):
        arc_bin[x["arc_id"]] = (int(cb), rank_bin(x["rank"]), int(mb),
                                int(x["baseline_error"]), int(x["outer_fold"]))
    by_pair = defaultdict(list)
    for x in all_arcs:
        by_pair[x["canonical_pair_id"]].append(x)
    graph_by = {x["canonical_pair_id"]: x for x in graph}
    signature_groups = defaultdict(list)
    for pair_id, directions in by_pair.items():
        g = graph_by[pair_id]
        signature = (
            "equal" if g["video_a_label"] == g["video_b_label"] else "different",
            tuple(sorted(g["direction_mask"])),
            tuple(sorted(arc_bin[x["arc_id"]] for x in directions)),
        )
        signature_groups[repr(signature)].append(pair_id)
    source_flag = {p: int(graph_by[p]["status"] == "accepted" and
                          graph_by[p]["family"] == family) for p in by_pair}
    arc_pair = [x["canonical_pair_id"] for x in arcs]
    nperm = int(cfg["b1"]["permutations"])
    perm_deltas = []
    for pidx in range(nperm):
        rng = np.random.default_rng(int(cfg["b1"]["permutation_seed_base"]) + pidx)
        target_flag = {}
        for members in signature_groups.values():
            sources = list(members)
            rng.shuffle(sources)
            for target, source in zip(members, sources):
                target_flag[target] = source_flag[source]
        perm_flag = np.asarray([target_flag[p] for p in arc_pair], dtype=float)
        ppr, ppf = grouped_oof(X, perm_flag, y, groups, cfg)
        pdnll, _ = metric_delta(y, ppr, ppf)
        perm_deltas.append(pdnll)
        if (pidx + 1) % 100 == 0:
            print(canonical_json({"permutations_done": pidx + 1,
                                  "dataset": dataset, "family": family}), flush=True)
    count = sum(x >= delta_nll for x in perm_deltas)
    p_raw = (1 + count) / float(nperm + 1)
    result = {
        "run_id": run_id, "gate": "conditional", "dataset": dataset,
        "family": family,
        "status": "PRE_HOLM_GO" if delta_nll > 0 and delta_auc > 0 else "FAIL",
        "n_arcs": len(arcs), "n_positive_events": int(y.sum()),
        "n_accepted_arcs": int(flag.sum()),
        "nll_reduced": float(log_loss(y, pr, labels=[0, 1])),
        "nll_full": float(log_loss(y, pf, labels=[0, 1])),
        "delta_nll": delta_nll, "delta_auc": delta_auc, "p_raw": p_raw,
        "permutations": nperm, "permutation_ge_observed": count,
        "cosine_decile_edges": cos_edges, "margin_quartile_edges": mar_edges,
        "cv_predictions": [{"arc_id": x["arc_id"], "y": int(y[i]),
                            "p_reduced": float(pr[i]), "p_full": float(pf[i]),
                            "group": int(groups[i])} for i, x in enumerate(arcs)],
        "permutation_delta_nll": perm_deltas,
        "only_gold_outcome": "exact_video_label_vote_event",
        "segment_gold_used": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_write_json(cell_path(cfg, dataset, family, "conditional"), result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gate", required=True, choices=["audit", "conditional", "oracle"])
    ap.add_argument("--dataset", required=True, choices=["MHC", "MHC_zh"])
    ap.add_argument("--family", required=True, choices=["MI", "SC"])
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SSR computation must run under SLURM")
    cfg = load_config(args.config)
    if args.gate == "audit":
        result = run_audit(cfg, args.dataset, args.family, args.run_id)
    elif args.gate == "oracle":
        result = run_oracle(cfg, args.dataset, args.family, args.run_id)
    else:
        result = run_conditional(cfg, args.dataset, args.family, args.run_id)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
