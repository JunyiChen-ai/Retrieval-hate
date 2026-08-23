#!/usr/bin/env python
"""TERA Gate-0 — harness entry point (prereg EXP_tera_gate0_prereg.md +
implementation appendix EXP_tera_gate0_impl_appendix.md v2).

Runs the registered stages on a data root that follows the canonical repository
layout, and writes the sec 11.1 artifact namespace.  Nothing here may select an
arm, endpoint, threshold, split or decision rule: every such quantity comes from
the frozen documents.

Usage (detached, per appendix sec 0.3):

    scripts/tera_gate0/run_detached.sh tera_gate0_stageA \\
        python -m scripts.tera_gate0.run_gate0 --stages A --run-root artifacts/tera_gate0

Development and fixture use passes `--data-root` at a synthetic corpus.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):                      # allow `python run_gate0.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.tera_gate0"

from .arms import (A_ARMS, B_ARMS, K_WINDOWS, config_list, head_capacity_check,
                   params_b2, params_b3, relative_time_encoding, select_pair,
                   solve_h3)
from .common import (BOOTSTRAP_SEED, B4_SEED, B5_SEED, TeraHalt, canonical_json,
                     macro_f1, note, read_jsonl, repo_root, select_threshold,
                     setup_determinism, sha256_file, sha256_ids, sha256_obj,
                     write_json_new, write_jsonl_new)
from .data import (Corpus, failure_report, load_corpus, read_hatemm_ids,
                   read_p11_split, split_manifest)
from .guards import Authorization, SealGuard
from .nested import ArmData, outer_folds, run_arm_fold, score_rows, select_on_partition
from .oracles import assert_oracle_ordering, o1_video_logit, o2_video_logit, sigmoid
from .stats import mean_ci, paired_delta_ci, stratified_indices
from .temporal import eligible_videos, temporal_metrics
from .verdict import gate_a_decision, gate_b_decision, overall_verdict

ROW_TEMPLATE = {
    "video_id": None, "dataset": None, "split": "train", "confirmation": False,
    "outer_fold": None, "gold_label": None, "arm": None, "score": None,
    "prediction": None, "threshold": None, "threshold_source": None,
    "config_id": None, "epoch": None, "seed": None,
    "selected_segment_ids": [], "selected_second_intervals": [],
    "d_score_source": None, "d_config_id": None, "d_epoch": None,
    "b5_donor_id": None, "b5_fallback": None, "b4_swapped": None, "o1_fallback": None,
    "oracle_or_eval_only": False, "label_leaking": False,
    "gold_overlap_windows": None, "gold_span_ratio": None,
}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_row(**kwargs):
    row = dict(ROW_TEMPLATE)
    row.update(kwargs)
    return row


# ------------------------------------------------------------ frozen config --
def load_frozen_config(path):
    with open(path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    payload_hash = sha256_obj(cfg["payload"])
    if cfg.get("payload_sha256") and cfg["payload_sha256"] != payload_hash:
        raise TeraHalt("HALT_CONFIG_HASH_MISMATCH",
                       "recorded %s, computed %s" % (cfg["payload_sha256"], payload_hash))
    return cfg, payload_hash


# ------------------------------------------------------------------ arm data --
def arm_data_a(arm, corpus):
    y = torch.as_tensor(corpus.labels, dtype=torch.float32)
    if arm == "A0":
        return ArmData(ids=corpus.ids, y=y, inputs=(corpus.X_whole,), d=corpus.d)
    return ArmData(ids=corpus.ids, y=y, inputs=(corpus.X_seg,), d=corpus.d,
                   seg_input=corpus.X_seg)


def metrics_from_preds(y, scores, preds, threshold_note):
    """Primary/secondary/diagnostic metrics at the frozen per-fold thresholds."""
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.int64)
    tp = int(((y == 1) & (preds == 1)).sum())
    fp = int(((y == 0) & (preds == 1)).sum())
    fn = int(((y == 1) & (preds == 0)).sum())
    tn = int(((y == 0) & (preds == 0)).sum())
    rec_pos = tp / (tp + fn) if (tp + fn) else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) else 0.0
    prec_pos = tp / (tp + fp) if (tp + fp) else 0.0
    f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) else 0.0
    auroc = None
    if 0 < int(y.sum()) < len(y):
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(y, scores))
    return {
        "macro_f1": macro_f1(y, preds),
        "balanced_accuracy": (rec_pos + rec_neg) / 2.0,
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
        "positive_class_f1": f1_pos,
        "auroc": auroc,
        "predicted_positive_rate": float(preds.mean()) if len(y) else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n": int(len(y)),
        "threshold_source": threshold_note,
    }


# ------------------------------------------------------------------- run ------
class Gate0Run(object):
    def __init__(self, args):
        self.args = args
        self.t0 = time.time()
        self.root = repo_root()
        self.data_root = Path(args.data_root).resolve()
        self.stages = [s.strip().upper() for s in args.stages.split(",") if s.strip()]
        self.hooks = {}
        if args.hooks:
            with open(args.hooks, encoding="utf-8") as handle:
                self.hooks = json.load(handle)
        self.bootstrap_n = int(args.bootstrap_n)
        self.metrics = {}
        self.manifest = {}
        self.status = "COMPLETE"
        self.halt = None
        self.guard = None
        self.auth = None
        self.gate_a = None
        self.gate_b = None
        self.gate_c = None

    # -- setup ------------------------------------------------------------
    def prepare(self):
        cfg, payload_hash = load_frozen_config(self.args.config)
        self.frozen_config = cfg
        self.payload_hash = payload_hash
        run_id = self.args.run_id or ("tera-gate0-%s-%s" % (
            dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"), payload_hash[:8]))
        self.run_id = run_id
        self.run_dir = Path(self.args.run_root).resolve() / run_id
        os.makedirs(self.run_dir, exist_ok=False)
        (self.run_dir / "folds").mkdir()
        self.determinism = setup_determinism()
        self.guard = SealGuard(self.data_root).install()
        note("run_id=%s run_dir=%s" % (run_id, self.run_dir))

    def build_authorization(self):
        development, confirmation_parts = {}, {}
        hatemm_train = sorted(read_hatemm_ids(self.data_root, "train"))
        development["HateMM"] = hatemm_train
        p11 = None
        if (self.data_root / "gt/HateClipSeg/p11_split.json").exists():
            p11, p11_sha = read_p11_split(self.data_root)
            development["HateClipSeg"] = sorted(p11["train"])
            confirmation_parts["HateClipSeg"] = sorted(set(p11["train"]) | set(p11["val"]))
            self.p11 = p11
            self.p11_sha = p11_sha
        else:
            self.p11 = None
            self.p11_sha = None

        data_root = self.data_root

        def confirmation_factory():
            out = dict(confirmation_parts)
            out["HateMM"] = sorted(set(hatemm_train) |
                                   set(read_hatemm_ids(data_root, "val")))
            return out

        self.auth = Authorization(development, confirmation_factory)
        if p11 is not None:
            self.auth.hateclipseg_test_ids = set(p11["test"])

    # -- stage A ----------------------------------------------------------
    def load_primary(self):
        self.corpus = load_corpus(self.data_root, "HateMM", "train", self.auth)
        self.label_map = {v: int(l) for v, l in zip(self.corpus.ids, self.corpus.labels)}
        # accounting first, HALT after the provenance artifacts are on disk
        self.failures = failure_report(self.corpus, halt_on_rate=False)
        write_json_new(self.run_dir / "split_manifest.json", split_manifest(self.corpus))
        write_json_new(self.run_dir / "feature_manifest.json", {
            "encoder_id": self.frozen_config["payload"]["features"]["encoder_id"],
            "dims": self.corpus.dims,
            "caches": self.corpus.cache_info,
            "failure_accounting": self.failures,
            "boundary_rule": self.frozen_config["payload"]["temporal_grid"]["window_rule"],
            "frame_time_convention":
                self.frozen_config["payload"]["temporal_grid"]["frame_time_convention"],
        })
        if self.failures["failure_rate"] > 0.01:
            raise TeraHalt("HALT_DECODE_FAILURE_RATE",
                           "%d/%d = %.4f > 0.01" % (self.failures["union"],
                                                    self.corpus.n,
                                                    self.failures["failure_rate"]))
        if self.hooks.get("probe_forbidden_path"):
            # fixture F14: deliberately touch a registered forbidden path
            with open(self.data_root / "gt/HateMM/test.jsonl", encoding="utf-8"):
                pass

    def build_folds(self):
        self.folds = outer_folds(self.corpus.ids, self.label_map)
        seen = {}
        overlap = {"outer_disjoint": True, "segment_disjoint": True,
                   "inner_nested": True, "one_query_fold_per_video": True}
        for k, (train_ids, query_ids) in enumerate(self.folds):
            if set(train_ids) & set(query_ids):
                overlap["outer_disjoint"] = False
            train_seg = {"%s#w%d" % (v, w) for v in train_ids for w in range(K_WINDOWS)}
            query_seg = {"%s#w%d" % (v, w) for v in query_ids for w in range(K_WINDOWS)}
            if train_seg & query_seg:
                overlap["segment_disjoint"] = False
            for vid in query_ids:
                seen[vid] = seen.get(vid, 0) + 1
            fold_dir = self.run_dir / "folds" / ("fold_%d" % k)
            fold_dir.mkdir(parents=True, exist_ok=False)
            write_json_new(fold_dir / "train_ids.json", sorted(train_ids))
            write_json_new(fold_dir / "query_ids.json", sorted(query_ids))
        if any(c != 1 for c in seen.values()) or len(seen) != self.corpus.n:
            overlap["one_query_fold_per_video"] = False
        from .nested import inner_folds as _inner
        for k, (train_ids, _) in enumerate(self.folds):
            for itr, iva in _inner(train_ids, self.label_map):
                if not (set(itr) | set(iva)) <= set(train_ids) or (set(itr) & set(iva)):
                    overlap["inner_nested"] = False
        if not all(overlap.values()):
            raise TeraHalt("HALT_FOLD_OVERLAP", canonical_json(overlap))
        self.overlap_assertions = overlap

    def run_stage_a(self):
        note("stage A: %d videos, %d folds" % (self.corpus.n, len(self.folds)))
        self.a_results = {}
        for arm in A_ARMS:
            data = arm_data_a(arm, self.corpus)
            configs = config_list(arm)
            per_fold = []
            for k, (train_ids, query_ids) in enumerate(self.folds):
                res = run_arm_fold("A", arm, "HateMM", data, k, train_ids, query_ids,
                                   configs, self.label_map, t0=self.t0)
                per_fold.append(res)
            self.a_results[arm] = per_fold
            note("stage A arm=%s done (%d configs)" % (arm, len(configs)))
        self.assemble_oracles()

    def assemble_oracles(self):
        """O1/O2 from A1's fold-trained segment logits (appendix sec 5)."""
        spans, durations = self.corpus.spans, self.corpus.durations
        self.oracle = {"O1": {"scores": {}, "thresholds": {}, "rows": []},
                       "O2": {"scores": {}, "thresholds": {}, "rows": []}}
        ordering_records = []
        for k, res in enumerate(self.a_results["A1"]):
            inner_o1, inner_o2, inner_y = [], [], []
            for vid, logits in sorted(res.inner_seg_logit.items()):
                l1, _, _ = o1_video_logit(logits, spans.get(vid, []), durations.get(vid))
                l2, _ = o2_video_logit(logits, self.label_map[vid])
                inner_o1.append(sigmoid(l1))
                inner_o2.append(sigmoid(l2))
                inner_y.append(self.label_map[vid])
            th1, _ = select_threshold(inner_o1, inner_y)
            th2, _ = select_threshold(inner_o2, inner_y)
            self.oracle["O1"]["thresholds"][k] = th1
            self.oracle["O2"]["thresholds"][k] = th2
            for vid, logits in sorted(res.query_seg_logit.items()):
                l1, sel1, fallback = o1_video_logit(logits, spans.get(vid, []),
                                                   durations.get(vid))
                l2, sel2 = o2_video_logit(logits, self.label_map[vid])
                s1, s2 = sigmoid(l1), sigmoid(l2)
                self.oracle["O1"]["scores"][vid] = s1
                self.oracle["O2"]["scores"][vid] = s2
                ordering_records.append({"video_id": vid, "gold_label": self.label_map[vid],
                                         "o1_logit": l1, "o2_logit": l2})
                dur = durations.get(vid)
                ratio = None
                if dur and dur > 0 and spans.get(vid):
                    ratio = sum(max(0.0, b - a) for a, b in spans[vid]) / dur
                self.oracle["O1"]["rows"].append(make_row(
                    video_id=vid, dataset="HateMM", outer_fold=k,
                    gold_label=self.label_map[vid], arm="O1", score=s1,
                    prediction=int(s1 >= th1), threshold=th1,
                    threshold_source="inner_oof:fold%d:O1" % k,
                    config_id=res.config["config_id"], epoch=res.epoch, seed=res.seed,
                    selected_segment_ids=sel1, o1_fallback=fallback,
                    oracle_or_eval_only=True, gold_overlap_windows=sel1 if not fallback else [],
                    gold_span_ratio=ratio))
                self.oracle["O2"]["rows"].append(make_row(
                    video_id=vid, dataset="HateMM", outer_fold=k,
                    gold_label=self.label_map[vid], arm="O2", score=s2,
                    prediction=int(s2 >= th2), threshold=th2,
                    threshold_source="inner_oof:fold%d:O2" % k,
                    config_id=res.config["config_id"], epoch=res.epoch, seed=res.seed,
                    selected_segment_ids=sel2, oracle_or_eval_only=True,
                    label_leaking=True, gold_span_ratio=ratio))
        assert_oracle_ordering(ordering_records)
        self.oracle_ordering_ok = True

    # -- assembly and metrics ---------------------------------------------
    def assemble_oof(self):
        self.oof = {}
        rows = []
        for arm in A_ARMS:
            scores, preds, meta = {}, {}, {}
            for k, res in enumerate(self.a_results[arm]):
                for vid, score in res.query_scores.items():
                    scores[vid] = score
                    preds[vid] = int(score >= res.theta)
                    meta[vid] = {"outer_fold": k, "theta": res.theta,
                                 "config_id": res.config["config_id"],
                                 "epoch": res.epoch, "seed": res.seed}
            self.oof[arm] = {"scores": scores, "preds": preds, "meta": meta}
            for vid in sorted(scores):
                info = meta[vid]
                rows.append(make_row(
                    video_id=vid, dataset="HateMM", outer_fold=info["outer_fold"],
                    gold_label=self.label_map[vid], arm=arm, score=scores[vid],
                    prediction=preds[vid], threshold=info["theta"],
                    threshold_source="inner_oof:fold%d:%s:%s:epoch%d" % (
                        info["outer_fold"], arm, info["config_id"], info["epoch"]),
                    config_id=info["config_id"], epoch=info["epoch"], seed=info["seed"]))
        for arm in ("O1", "O2"):
            scores = self.oracle[arm]["scores"]
            preds = {r["video_id"]: r["prediction"] for r in self.oracle[arm]["rows"]}
            self.oof[arm] = {"scores": scores, "preds": preds, "meta": {}}
        write_jsonl_new(self.run_dir / "oof_predictions.jsonl", rows)
        oracle_rows = sorted(self.oracle["O1"]["rows"] + self.oracle["O2"]["rows"],
                             key=lambda r: (r["arm"], r["video_id"]))
        write_jsonl_new(self.run_dir / "oracle_predictions.jsonl", oracle_rows)
        write_json_new(self.run_dir / "oracle_predictions.jsonl.marker.json",
                       {"oracle_or_eval_only": True})

    def write_fold_artifacts(self):
        for k in range(len(self.folds)):
            fold_dir = self.run_dir / "folds" / ("fold_%d" % k)
            hp, seg_rows, pred_rows = {}, [], []
            for arm in A_ARMS:
                res = self.a_results[arm][k]
                hp[arm] = {"config": {kk: vv for kk, vv in res.config.items()},
                           "epoch": res.epoch, "theta": res.theta, "seed": res.seed,
                           "inner_oof_macro_f1": res.inner_macro_f1,
                           "n_trainings": res.n_trainings,
                           "candidates": res.candidates}
                for vid in sorted(res.query_seg):
                    dur = self.corpus.durations.get(vid)
                    bounds = [[float(w * dur / K_WINDOWS), float((w + 1) * dur / K_WINDOWS)]
                              for w in range(K_WINDOWS)] if dur else [[0.0, 0.0]] * K_WINDOWS
                    seg_rows.append({"video_id": vid, "arm": arm, "outer_fold": k,
                                     "scores": res.query_seg[vid],
                                     "attention_weights": res.query_att.get(vid),
                                     "second_boundaries": bounds})
                for vid in sorted(res.query_scores):
                    pred_rows.append(make_row(
                        video_id=vid, dataset="HateMM", outer_fold=k,
                        gold_label=self.label_map[vid], arm=arm,
                        score=res.query_scores[vid],
                        prediction=int(res.query_scores[vid] >= res.theta),
                        threshold=res.theta, threshold_source="inner_oof:fold%d:%s" % (k, arm),
                        config_id=res.config["config_id"], epoch=res.epoch, seed=res.seed))
            if getattr(self, "b_results", None):
                for arm in B_ARMS:
                    res = self.b_results[arm][k]
                    hp[arm] = {"config": {kk: vv for kk, vv in res.config.items()},
                               "epoch": res.epoch, "theta": res.theta, "seed": res.seed,
                               "inner_oof_macro_f1": res.inner_macro_f1,
                               "n_trainings": res.n_trainings}
                    for vid in sorted(res.query_scores):
                        ev = self.b_evidence[k][vid]
                        pred_rows.append(make_row(
                            video_id=vid, dataset="HateMM", outer_fold=k,
                            gold_label=self.label_map[vid], arm=arm,
                            score=res.query_scores[vid],
                            prediction=int(res.query_scores[vid] >= res.theta),
                            threshold=res.theta,
                            threshold_source="inner_oof:fold%d:%s" % (k, arm),
                            config_id=res.config["config_id"], epoch=res.epoch,
                            seed=res.seed, selected_segment_ids=list(ev["pair"]),
                            d_score_source=ev["d_score_source"],
                            d_config_id=ev["d_config_id"], d_epoch=ev["d_epoch"],
                            b4_swapped=ev["b4_swapped"] if arm == "B4" else None,
                            b5_donor_id=ev["b5_donor_id"] if arm == "B5" else None,
                            b5_fallback=ev["b5_fallback"] if arm == "B5" else None))
                write_jsonl_new(fold_dir / "selected_evidence.jsonl",
                                [self.b_evidence[k][v] | {"video_id": v}
                                 for v in sorted(self.b_evidence[k])])
            write_json_new(fold_dir / "selected_hparams.json", hp)
            write_jsonl_new(fold_dir / "segment_scores.jsonl", seg_rows)
            write_jsonl_new(fold_dir / "video_predictions.jsonl",
                            sorted(pred_rows, key=lambda r: (r["arm"], r["video_id"])))

    def pooled_inner_oof_macro_f1(self, arm):
        """POOLED-INNER-OOF-MACRO-F1 (appendix sec 5.3): one macro-F1 on the
        concatenation of all 5 outer folds' inner-held-out predictions."""
        scores, labels = [], []
        for res in self.a_results[arm]:
            for vid, score in sorted(res.inner_scores.items()):
                scores.append(int(score >= res.theta))
                labels.append(self.label_map[vid])
        return macro_f1(labels, scores)

    def compute_metrics(self):
        ids = sorted(self.corpus.ids)
        y = np.array([self.label_map[v] for v in ids], dtype=np.int64)
        arm_metrics = {}
        for arm in list(A_ARMS) + ["O1", "O2"]:
            scores = np.array([self.oof[arm]["scores"][v] for v in ids], dtype=np.float64)
            preds = np.array([self.oof[arm]["preds"][v] for v in ids], dtype=np.int64)
            m = metrics_from_preds(y, scores, preds, "per-fold inner-OOF")
            m["oracle_or_eval_only"] = arm in ("O1", "O2")
            m["label_leaking"] = arm == "O2"
            arm_metrics[arm] = m
        self.arm_metrics = arm_metrics

        pooled = {arm: self.pooled_inner_oof_macro_f1(arm) for arm in ("A2", "A3", "A4")}
        best = max(pooled.values())
        d_arm = [a for a in ("A2", "A3", "A4") if pooled[a] == best][0]
        self.d_arm = d_arm
        self.metrics["arm_D"] = {
            "identity": d_arm,
            "pooled_inner_oof_macro_f1": pooled,
            "selection_rule": "argmax POOLED-INNER-OOF-MACRO-F1 over {A2,A3,A4}; "
                              "ties -> arm-id order A2 < A3 < A4",
            "per_fold": [{"outer_fold": k, "config_id": r.config["config_id"],
                          "epoch": r.epoch, "theta": r.theta}
                         for k, r in enumerate(self.a_results[d_arm])],
            "note": "selection-optimistic; used only to rank arms, never reported "
                    "as performance (appendix sec 5.3)",
        }

        # temporal metrics -- eligible set frozen once, shared across arms
        elig = eligible_videos(ids, lambda v: self.label_map[v], self.corpus.durations,
                               self.corpus.spans)
        self.eligible = elig
        write_json_new(self.run_dir / "eligible_videos.json",
                       {"n": len(elig), "video_ids": elig,
                        "rule": "label==1 and >=1 span and valid duration and both "
                                "classes present among evaluated seconds"})
        temporal = {}
        for arm in ("A1", "A2", "A3", "A4"):
            seg = {}
            for res in self.a_results[arm]:
                seg.update(res.query_seg)
            temporal[arm] = temporal_metrics(arm, elig, seg, self.corpus.durations,
                                             self.corpus.spans)
        att = {}
        for res in self.a_results["A3"]:
            att.update(res.query_att)
        if att:
            temporal["A3_alpha_diagnostic"] = temporal_metrics(
                "A3_alpha", elig, att, self.corpus.durations, self.corpus.spans)
            temporal["A3_alpha_diagnostic"]["binding"] = False
        temporal["A0_broadcast"] = {"arm": "A0", "n_eligible": len(elig),
                                    "mean_within_video_auroc": 0.5,
                                    "note": "video-score broadcast is exactly 0.5"}
        self.metrics["temporal"] = temporal

        # bootstrap
        boot = stratified_indices(y, self.bootstrap_n, BOOTSTRAP_SEED)
        self.boot = boot
        base_arm = "A0" if self.arm_metrics["A0"]["macro_f1"] >= self.arm_metrics["A1"]["macro_f1"] else "A1"
        pred = {a: np.array([self.oof[a]["preds"][v] for v in ids], dtype=np.int64)
                for a in list(A_ARMS) + ["O1", "O2"]}
        d_ci = paired_delta_ci(y, pred[d_arm], pred[base_arm], boot)
        self.metrics["bootstrap"] = {
            "n_resamples": int(self.bootstrap_n), "seed": BOOTSTRAP_SEED,
            "base_arm_for_d_delta": base_arm,
            "d_minus_base_ci": d_ci,
            "o1_minus_base_ci": paired_delta_ci(y, pred["O1"], pred[base_arm], boot),
            "o2_minus_base_ci": paired_delta_ci(y, pred["O2"], pred[base_arm], boot),
        }
        if elig:
            temporal_idx = stratified_indices(np.ones(len(elig), dtype=np.int64),
                                              self.bootstrap_n, BOOTSTRAP_SEED)
            per_video = temporal[d_arm]["per_video_auroc"]
            values = np.array([per_video[v] for v in elig if v in per_video])
            if values.size == len(elig):
                self.metrics["temporal"]["d_mean_auroc_ci"] = mean_ci(values, temporal_idx)
            np.savez(str(self.run_dir / "bootstrap_indices.npz"), master=boot,
                     temporal=temporal_idx)
        else:
            np.savez(str(self.run_dir / "bootstrap_indices.npz"), master=boot)

        macro = {a: self.arm_metrics[a]["macro_f1"] for a in ("A0", "A1", "O1", "O2")}
        macro["D"] = self.arm_metrics[d_arm]["macro_f1"]
        self.gate_a = gate_a_decision(
            macro, d_ci, temporal[d_arm]["mean_within_video_auroc"],
            self.confirmation_summary("A"))
        self.metrics["arms"] = arm_metrics
        self.metrics["budget_report"] = {
            arm: {"n_trainings": sum(r.n_trainings for r in self.a_results[arm]),
                  "n_configs": len(config_list(arm))} for arm in A_ARMS}

    # -- stage B ----------------------------------------------------------
    def build_pair_inputs(self, fold_idx):
        """D's segment scores, pair selection, B4 swap and B5 donors (sec 6.1-6.6)."""
        d_arm = self.d_arm
        res = self.a_results[d_arm][fold_idx]
        train_ids, query_ids = self.folds[fold_idx]
        seg_scores, source = {}, {}
        for vid, seg in res.inner_seg.items():
            seg_scores[vid] = seg
            source[vid] = "inner_oof"
        for vid, seg in res.query_seg.items():
            seg_scores[vid] = seg
            source[vid] = "outer_fold_model"
        override = self.hooks.get("d_segment_scores_file")
        if override:
            # FIXTURE-ONLY: replace D's segment scores with a supplied ranking so a
            # Gate-B fixture can exercise pair selection and the lesions
            # independently of whether the weak selector converged on synthetic
            # data.  Refused outside --fixture-mode.
            if not self.args.fixture_mode:
                raise TeraHalt("HALT_D_SCORE_OVERRIDE", "override outside fixture mode")
            with open(override, encoding="utf-8") as handle:
                supplied = json.load(handle)
            for vid in list(seg_scores):
                if vid in supplied:
                    seg_scores[vid] = supplied[vid]
                    source[vid] = "fixture_hook"
            self.metrics["d_score_override"] = {"file": override, "fixture_only": True}
        dpred = {}
        for vid in train_ids:
            dpred[vid] = int(res.inner_scores[vid] >= res.theta)
        for vid in query_ids:
            dpred[vid] = int(res.query_scores[vid] >= res.theta)
        hook = self.hooks.get("force_dpred")
        if hook and int(hook.get("outer_fold", -1)) == fold_idx:
            value = int(hook.get("value", 1))
            exceptions = sorted(query_ids)[:int(hook.get("n_query_exceptions", 2))]
            for vid in list(dpred):
                dpred[vid] = value
            for vid in exceptions:
                dpred[vid] = 1 - value

        if not hasattr(self, "b4_swap"):
            rng4 = np.random.default_rng(B4_SEED)
            self.b4_swap = {vid: bool(rng4.integers(0, 2))
                            for vid in sorted(self.corpus.ids)}
            frac = sum(self.b4_swap.values()) / len(self.b4_swap)
            if not (0.40 <= frac <= 0.60):
                raise TeraHalt("HALT_B4_SWAP_FRACTION", "realized fraction %.4f" % frac)
            self.b4_swap_fraction = frac
            self.b4_swap_hash = sha256_obj({k: int(v) for k, v in sorted(self.b4_swap.items())})

        index = self.corpus.index
        n = self.corpus.n
        d = self.corpus.d
        e_first = torch.zeros(n, d)
        e_second = torch.zeros(n, d)
        e_top = torch.zeros(n, d)
        phi = torch.zeros(n, 6)
        e_a = torch.zeros(n, d)
        e_b = torch.zeros(n, d)
        phi4 = torch.zeros(n, 6)
        evidence = {}
        for vid in sorted(self.corpus.ids):
            i_top, (a, b) = select_pair(seg_scores[vid])
            row = index[vid]
            e_first[row] = self.corpus.X_seg[row, a]
            e_second[row] = self.corpus.X_seg[row, b]
            e_top[row] = self.corpus.X_seg[row, i_top]
            phi_v = relative_time_encoding(a, b)
            phi[row] = torch.tensor(phi_v, dtype=torch.float32)
            swapped = self.b4_swap[vid]
            ia, ib = (b, a) if swapped else (a, b)
            e_a[row] = self.corpus.X_seg[row, ia]
            e_b[row] = self.corpus.X_seg[row, ib]
            phi4_v = relative_time_encoding(ia, ib)
            phi4[row] = torch.tensor(phi4_v, dtype=torch.float32)
            evidence[vid] = {"arm": "D:%s" % d_arm, "i_top": int(i_top),
                             "pair": [int(a), int(b)],
                             "presented_slots": [int(ia), int(ib)],
                             "phi": [float(x) for x in phi4_v],
                             "b4_swapped": bool(swapped), "b5_donor_id": None,
                             "b5_fallback": None,
                             "d_score_source": source[vid],
                             "d_config_id": res.config["config_id"], "d_epoch": res.epoch}
            if swapped:
                if not (ia > ib):
                    raise TeraHalt("HALT_B4_ORDER", vid)
                if phi4_v[2] * phi_v[2] >= 0:
                    raise TeraHalt("HALT_B4_DELTA_SIGN", vid)

        # B5 donors -- rng5 is a SEPARATE generator, consumed in sorted id order
        rng5 = np.random.default_rng(B5_SEED)
        train_set = sorted(train_ids)
        e_donor = e_second.clone()
        fallback_count = 0
        for vid in sorted(self.corpus.ids):
            pool = [u for u in train_set if u != vid]
            if not pool:
                raise TeraHalt("HALT_B5_EMPTY_POOL", vid)
            stratum = [u for u in pool if dpred[u] == dpred[vid]]
            if stratum:
                donor = str(rng5.choice(np.array(stratum, dtype=object)))
                fb = False
            else:
                donor = str(rng5.choice(np.array(pool, dtype=object)))
                fb = True
                fallback_count += 1
            e_donor[index[vid]] = e_second[index[donor]]
            evidence[vid]["b5_donor_id"] = donor
            evidence[vid]["b5_fallback"] = fb
        self.b5_fallback_count = getattr(self, "b5_fallback_count", 0) + fallback_count
        return {"e_first": e_first, "e_second": e_second, "e_top": e_top, "phi": phi,
                "e_a": e_a, "e_b": e_b, "phi4": phi4, "e_donor": e_donor,
                "evidence": evidence, "dpred": dpred}

    def arm_data_b(self, arm, tensors, corpus=None):
        corpus = corpus if corpus is not None else self.corpus
        y = torch.as_tensor(corpus.labels, dtype=torch.float32)
        ids, d = corpus.ids, corpus.d
        if arm in ("B0", "B3"):
            inputs = (tensors["e_top"],)
        elif arm == "B1":
            inputs = (tensors["e_first"], tensors["e_second"])
        elif arm == "B2":
            inputs = (tensors["e_first"], tensors["e_second"], tensors["phi"])
        elif arm == "B4":
            inputs = (tensors["e_a"], tensors["e_b"], tensors["phi4"])
        elif arm == "B5":
            inputs = (tensors["e_first"], tensors["e_donor"], tensors["phi"])
        else:
            raise TeraHalt("HALT_UNKNOWN_ARM", arm)
        return ArmData(ids=ids, y=y, inputs=inputs, d=d)

    def run_stage_b(self):
        self.h3, rel = solve_h3(self.corpus.d)
        self.metrics["capacity_match"] = dict(head_capacity_check(self.corpus.d),
                                              relative_param_difference=rel)
        self.b_evidence = {}
        self.b_tensors = {}
        for k in range(len(self.folds)):
            tensors = self.build_pair_inputs(k)
            self.b_evidence[k] = tensors.pop("evidence")
            self.b_tensors[k] = tensors
        self.b_results = {arm: [] for arm in B_ARMS}
        for arm in B_ARMS:
            configs = config_list(arm)
            for k, (train_ids, query_ids) in enumerate(self.folds):
                data = self.arm_data_b(arm, self.b_tensors[k])
                res = run_arm_fold("B", arm, "HateMM", data, k, train_ids, query_ids,
                                   configs, self.label_map, h3=self.h3, t0=self.t0)
                self.b_results[arm].append(res)
            note("stage B arm=%s done" % arm)
        self.assemble_stage_b()

    def assemble_stage_b(self):
        ids = sorted(self.corpus.ids)
        y = np.array([self.label_map[v] for v in ids], dtype=np.int64)
        preds, scores = {}, {}
        for arm in B_ARMS:
            s, p = {}, {}
            for res in self.b_results[arm]:
                for vid, score in res.query_scores.items():
                    s[vid] = score
                    p[vid] = int(score >= res.theta)
            scores[arm] = s
            preds[arm] = p
            self.arm_metrics[arm] = metrics_from_preds(
                y, [s[v] for v in ids], [p[v] for v in ids], "per-fold inner-OOF")
        rows = []
        for arm in B_ARMS:
            for k, res in enumerate(self.b_results[arm]):
                for vid in sorted(res.query_scores):
                    ev = self.b_evidence[k][vid]
                    rows.append(make_row(
                        video_id=vid, dataset="HateMM", outer_fold=k,
                        gold_label=self.label_map[vid], arm=arm,
                        score=res.query_scores[vid],
                        prediction=int(res.query_scores[vid] >= res.theta),
                        threshold=res.theta,
                        threshold_source="inner_oof:fold%d:%s:%s:epoch%d" % (
                            k, arm, res.config["config_id"], res.epoch),
                        config_id=res.config["config_id"], epoch=res.epoch,
                        seed=res.seed, selected_segment_ids=list(ev["pair"]),
                        d_score_source=ev["d_score_source"], d_config_id=ev["d_config_id"],
                        d_epoch=ev["d_epoch"],
                        b4_swapped=ev["b4_swapped"] if arm == "B4" else None,
                        b5_donor_id=ev["b5_donor_id"] if arm == "B5" else None,
                        b5_fallback=ev["b5_fallback"] if arm == "B5" else None))
        write_jsonl_new(self.run_dir / "stage_b_predictions.jsonl",
                        sorted(rows, key=lambda r: (r["arm"], r["video_id"])))
        base_arm = max(("B0", "B1", "B3"), key=lambda a: self.arm_metrics[a]["macro_f1"])
        pred = {a: np.array([preds[a][v] for v in ids], dtype=np.int64) for a in B_ARMS}
        ci = paired_delta_ci(y, pred["B2"], pred[base_arm], self.boot)
        macro = {a: self.arm_metrics[a]["macro_f1"] for a in B_ARMS}
        rescue = {"state": "not_evaluated",
                  "reason": "Gate-C msc subset requires the adjudicated human audit"}
        msc_ids = self.hooks.get("msc_subset") or getattr(self, "msc_ids", None)
        if msc_ids:
            from .gate_c import rescue_metrics
            rescue = rescue_metrics(msc_ids, self.label_map, preds["B0"], preds["B2"])
            rescue["state"] = "evaluated"
        self._gate_b_inputs = (macro, ci, rescue)
        self.gate_b = gate_b_decision(macro, ci, rescue, self.confirmation_summary("B"))
        self.metrics["stage_b"] = {
            "base_arm_for_b2_delta": base_arm,
            "b2_delta_ci": ci,
            "b4": {"swap_fraction": self.b4_swap_fraction,
                   "swap_set_sha256": self.b4_swap_hash,
                   "swap_set_sha256_eval": sha256_obj(
                       {k: int(v) for k, v in sorted(self.b4_swap.items())}),
                   "assertions": "sign(delta) flipped and iA > iB on every swapped video"},
            "b5_fallback_count": self.b5_fallback_count,
            "h3": self.h3,
            "params": {"B2": params_b2(self.corpus.d),
                       "B3": params_b3(self.corpus.d, self.h3)},
        }
        self.metrics["stage_b"]["b4"]["train_eval_swap_sets_identical"] = (
            self.metrics["stage_b"]["b4"]["swap_set_sha256"] ==
            self.metrics["stage_b"]["b4"]["swap_set_sha256_eval"])

    # -- stage C -----------------------------------------------------------
    def run_stage_c(self):
        """Gate-C sampling, blinded item list and (if an audit exists) the decision.

        Prediction source is the A0 whole-video OOF run on HateMM-train only
        (appendix sec 11.1).  Sampling, frozen population weights and the
        coverage bootstrap follow sec 11.3; the audit itself is human work and is
        supplied through --gate-c-audit.
        """
        from . import gate_c as gc

        rows = [r for r in self.rows_for_gate_c()]
        sample = gc.select_audit_sample(rows)
        protocol = {
            "taxonomy": list(gc.TAXONOMY),
            "form_fields": ["video_id", "coder_id", "primary_cause", "secondary_causes",
                            "minimal_sufficient_intervals", "required_modalities",
                            "single_interval_sufficient", "span_video_duration_ratio",
                            "confidence", "notes", "protocol_sha256", "form_version"],
            "required_modalities_enum": ["visual", "speech", "on_screen_text",
                                         "audio_nonspeech", "transcript"],
            "confidence_enum": ["high", "medium", "low"],
            "blinding": {"shown": ["video", "transcript", "official span overlay"],
                         "hidden": ["model score", "correctness category",
                                    "retrieval output", "TERA output"]},
            "double_coding_fraction": 0.20,
            "form_version": "tera-gate0-gatec/1",
            "prediction_source": "A0 whole-video OOF on HateMM-train",
        }
        protocol_path = write_json_new(self.run_dir / "annotation_protocol.json", protocol)
        protocol_sha = sha256_file(protocol_path)
        audited = sorted(set(sample["audit_fn"]) |
                         set(sample["controls"]["true_positives"]) |
                         set(sample["controls"]["false_positives"]))
        rng = np.random.default_rng(20260807)
        order = rng.permutation(len(audited))
        blinded = [{"video_id": audited[int(i)], "protocol_sha256": protocol_sha}
                   for i in order]
        write_jsonl_new(self.run_dir / "gate_c_items_blinded.jsonl", blinded)
        n_double = int(np.ceil(0.20 * len(audited)))
        double_coded = [audited[int(i)] for i in
                        np.random.default_rng(20260807).permutation(len(audited))[:n_double]]
        sample_out = dict(sample)
        sample_out["weights"] = {k: float(v) for k, v in sample["weights"].items()}
        sample_out["double_coded"] = sorted(double_coded)
        sample_out["protocol_sha256"] = protocol_sha
        write_json_new(self.run_dir / "gate_c_sample.json", sample_out)
        self.metrics["gate_c_sampling"] = {
            "n_fn_population": sample["n_fn_population"],
            "audited_all": sample["audited_all"],
            "population_sizes": sample["population_sizes"],
            "sampled_sizes": sample["sampled_sizes"],
            "tercile_cuts": sample["tercile_cuts"],
            "n_controls": {k: len(v) for k, v in sample["controls"].items()},
            "n_double_coded": n_double,
        }
        if not self.args.gate_c_audit:
            self.gate_c = None
            self.metrics["gate_c_sampling"]["audit"] = "not_supplied"
            return
        audit_rows = [r for r in read_jsonl(self.args.gate_c_audit)
                      if not r.get("superseded")]
        by_video, resolved = gc.resolve_audit_rows(audit_rows)
        adjudicated = {}
        pairs = []
        for vid, rws in by_video.items():
            adjudicated[vid] = gc.mechanisms_of(resolved[vid])
            if len(rws) >= 2:
                pairs.append((rws[0]["primary_cause"], rws[1]["primary_cause"]))
        audit_fn = [v for v in sample["audit_fn"] if v in adjudicated]
        weights = {v: sample["weights"][v] for v in audit_fn}
        tercile_of = {v: sample["tercile_of"][v] for v in audit_fn}
        union = gc.weighted_coverage(audit_fn, adjudicated, weights, gc.UNION_SET)
        ci = gc.coverage_bootstrap(audit_fn, tercile_of, adjudicated, weights,
                                   gc.UNION_SET, n_resamples=self.bootstrap_n)
        msc = gc.weighted_coverage(audit_fn, adjudicated, weights,
                                   ["multi_segment_complementary"])
        noise = gc.weighted_coverage(audit_fn, adjudicated, weights,
                                     ["annotation_ambiguity_or_noise"])
        agreement, kappa = gc.cohen_kappa(pairs)
        self.gate_c = gc.gate_c_decision(union, ci["ci_lower"], msc, noise, kappa)
        self.gate_c["values"] = {"union_coverage": union, "union_ci": ci,
                                 "msc_coverage": msc, "noise_coverage": noise,
                                 "raw_agreement": agreement, "kappa": kappa,
                                 "n_audited_fn": len(audit_fn),
                                 "unweighted_union": gc.unweighted_coverage(
                                     audit_fn, adjudicated, gc.UNION_SET)}
        msc_ids = gc.msc_subset(audit_rows)
        path = write_json_new(self.run_dir / "msc_subset.json",
                              {"video_ids": msc_ids, "n": len(msc_ids)})
        self.msc_subset_sha256 = sha256_file(path)
        self.msc_ids = msc_ids
        self.metrics["gate_c"] = self.gate_c

    def rows_for_gate_c(self):
        return [{"video_id": v, "score": self.oof["A0"]["scores"][v],
                 "gold_label": self.label_map[v], "prediction": self.oof["A0"]["preds"][v]}
                for v in sorted(self.corpus.ids)]

    # -- confirmation ------------------------------------------------------
    def confirmation_summary(self, stage):
        conf = getattr(self, "confirmation_results", None)
        if not conf:
            return {"all_positive": "not_evaluated", "state": "not_run"}
        deltas = conf.get("stage_%s" % stage.lower(), {})
        if not deltas:
            return {"all_positive": "not_evaluated", "state": "not_run"}
        vals = [v for v in deltas.values() if isinstance(v, (int, float))]
        return {"all_positive": bool(vals) and all(v > 0 for v in vals),
                "deltas": deltas, "state": "evaluated"}

    def pair_tensors(self, corpus, seg_scores_by_vid):
        """Pair inputs (e_first/e_second/e_top/phi) from a segment-score source.

        Used by the confirmation refits, where no outer fold exists.  B4/B5 are
        not needed on a confirmation set (appendix sec 7.10.1).
        """
        n, d = corpus.n, corpus.d
        index = corpus.index
        e_first = torch.zeros(n, d)
        e_second = torch.zeros(n, d)
        e_top = torch.zeros(n, d)
        phi = torch.zeros(n, 6)
        evidence = {}
        for vid in sorted(corpus.ids):
            i_top, (a, b) = select_pair(seg_scores_by_vid[vid])
            row = index[vid]
            e_first[row] = corpus.X_seg[row, a]
            e_second[row] = corpus.X_seg[row, b]
            e_top[row] = corpus.X_seg[row, i_top]
            phi_v = relative_time_encoding(a, b)
            phi[row] = torch.tensor(phi_v, dtype=torch.float32)
            evidence[vid] = {"i_top": int(i_top), "pair": [int(a), int(b)],
                             "presented_slots": [int(a), int(b)],
                             "phi": [float(x) for x in phi_v]}
        # B4/B5 are not evaluated on a confirmation set, so their slots mirror B2's.
        return {"e_first": e_first, "e_second": e_second, "e_top": e_top, "phi": phi,
                "e_a": e_first, "e_b": e_second, "phi4": phi, "e_donor": e_second,
                "evidence": evidence}

    def d_outer_oof_segment_scores(self):
        seg = {}
        for res in self.a_results[self.d_arm]:
            seg.update(res.query_seg)
        return seg

    def run_confirmation(self):
        """Appendix sec 7.10 -- one-time, after every choice is frozen."""
        self.auth.unlock_confirmation()
        results = {"stage_a": {}, "stage_b": {},
                   "passes": {"hatemm_val": 0, "hateclipseg_val": 0},
                   "hatemm_val": {}, "hateclipseg_val": {}}
        results["hatemm_val"] = self.confirm_hatemm_val()
        results["passes"]["hatemm_val"] = 1
        results["stage_a"]["hatemm_val_d_delta"] = results["hatemm_val"]["d_delta"]
        if results["hatemm_val"].get("b2_delta") is not None:
            results["stage_b"]["hatemm_val_b2_delta"] = results["hatemm_val"]["b2_delta"]
        if self.args.confirmation == "all":
            results["hateclipseg_val"] = self.confirm_hateclipseg_val()
            results["passes"]["hateclipseg_val"] = 1
            if not results["hateclipseg_val"].get("underpowered"):
                results["stage_a"]["hateclipseg_val_d_delta"] = \
                    results["hateclipseg_val"]["d_delta"]
                if results["hateclipseg_val"].get("b2_delta") is not None:
                    results["stage_b"]["hateclipseg_val_b2_delta"] = \
                        results["hateclipseg_val"]["b2_delta"]
        self.confirmation_results = results
        write_json_new(self.run_dir / "confirmation_summary.json", results)
        return results

    def confirm_hatemm_val(self):
        """One refit on the whole of HateMM-train per arm; score val once (sec 7.10.1)."""
        from .nested import refit_full

        val = load_corpus(self.data_root, "HateMM", "val", self.auth)
        val_labels = {v: int(l) for v, l in zip(val.ids, val.labels)}
        ids_val = sorted(val.ids)
        y_val = np.array([val_labels[v] for v in ids_val], dtype=np.int64)
        train_ids = sorted(self.corpus.ids)
        out = {"n_val": len(ids_val), "macro_f1": {}, "hparams": {}}
        rows = []

        d_val_seg = None
        for arm in ("A0", "A1", self.d_arm):
            cfg, epoch, theta = self.transfer_hparams(self.a_results[arm], arm)
            model, seed = refit_full("A", arm, "HateMM", arm_data_a(arm, self.corpus),
                                     train_ids, cfg, epoch)
            data_val = arm_data_a(arm, val)
            scores, seg, _, _ = score_rows(model, data_val, data_val.rows(ids_val))
            preds = (scores >= theta).astype(np.int64)
            out["macro_f1"][arm] = macro_f1(y_val, preds)
            out["hparams"][arm] = {"config_id": cfg["config_id"], "epoch": epoch,
                                   "theta": theta, "seed": seed}
            rows.extend(self._conf_rows(arm, ids_val, val_labels, scores, preds, theta,
                                        cfg, epoch, seed, "HateMM"))
            if arm == self.d_arm and seg is not None:
                d_val_seg = {v: [float(x) for x in seg[i]] for i, v in enumerate(ids_val)}
        base = max(out["macro_f1"]["A0"], out["macro_f1"]["A1"])
        out["d_delta"] = out["macro_f1"][self.d_arm] - base

        if getattr(self, "b_results", None) and d_val_seg is not None:
            train_tensors = self.pair_tensors(self.corpus, self.d_outer_oof_segment_scores())
            val_tensors = self.pair_tensors(val, d_val_seg)
            for arm in ("B0", "B1", "B2", "B3"):
                cfg, epoch, theta = self.transfer_hparams(self.b_results[arm], arm)
                model, seed = refit_full("B", arm, "HateMM",
                                         self.arm_data_b(arm, train_tensors, self.corpus),
                                         train_ids, cfg, epoch, h3=self.h3)
                data_val = self.arm_data_b(arm, val_tensors, val)
                scores, _, _, _ = score_rows(model, data_val, data_val.rows(ids_val))
                preds = (scores >= theta).astype(np.int64)
                out["macro_f1"][arm] = macro_f1(y_val, preds)
                out["hparams"][arm] = {"config_id": cfg["config_id"], "epoch": epoch,
                                       "theta": theta, "seed": seed}
                rows.extend(self._conf_rows(arm, ids_val, val_labels, scores, preds,
                                            theta, cfg, epoch, seed, "HateMM"))
            b_base = max(out["macro_f1"]["B0"], out["macro_f1"]["B1"], out["macro_f1"]["B3"])
            out["b2_delta"] = out["macro_f1"]["B2"] - b_base
        out["train_pair_source"] = ("D outer-OOF segment scores on HateMM-train; "
                                    "D val-refit segment scores on HateMM-val")
        write_jsonl_new(self.run_dir / "confirmation_predictions.jsonl", rows)
        return out

    def confirm_hateclipseg_val(self):
        """Fit every arm on HateClipSeg-train, score val once (sec 7.10.2)."""
        corpus = load_corpus(self.data_root, "HateClipSeg", "all", self.auth)
        labels = {v: int(l) for v, l in zip(corpus.ids, corpus.labels)}
        train_ids = sorted(set(self.p11["train"]) & set(corpus.ids))
        val_ids = sorted(set(self.p11["val"]) & set(corpus.ids))
        counts = {"val_positive": sum(labels[v] for v in val_ids),
                  "val_negative": len(val_ids) - sum(labels[v] for v in val_ids),
                  "train_positive": sum(labels[v] for v in train_ids),
                  "train_negative": len(train_ids) - sum(labels[v] for v in train_ids)}
        out = {"binding_endpoint_counts": counts}
        if min(counts["val_positive"], counts["val_negative"]) < 10:
            out["underpowered"] = True
            out["note"] = ("prereg sec 2.2: fewer than 10 videos in a validation class; "
                           "the confirmation cannot satisfy any criterion")
            return out
        out["underpowered"] = False
        y_val = np.array([labels[v] for v in val_ids], dtype=np.int64)
        out["macro_f1"], out["hparams"] = {}, {}
        d_val_seg = None
        for arm in ("A0", "A1", self.d_arm):
            data = arm_data_a(arm, corpus)
            res = select_on_partition("A", arm, "HateClipSeg", data, train_ids,
                                      config_list(arm), labels, t0=self.t0)
            scores, seg, _, _ = score_rows(res.model, data, data.rows(val_ids))
            preds = (scores >= res.theta).astype(np.int64)
            out["macro_f1"][arm] = macro_f1(y_val, preds)
            out["hparams"][arm] = {"config_id": res.config["config_id"],
                                   "epoch": res.epoch, "theta": res.theta}
            if arm == self.d_arm and seg is not None:
                d_val_seg = {v: [float(x) for x in seg[i]] for i, v in enumerate(val_ids)}
                d_train_seg = dict(res.inner_seg)
        base = max(out["macro_f1"]["A0"], out["macro_f1"]["A1"])
        out["d_delta"] = out["macro_f1"][self.d_arm] - base
        if getattr(self, "b_results", None) and d_val_seg is not None:
            seg_all = dict(d_train_seg)
            seg_all.update(d_val_seg)
            for vid in corpus.ids:
                seg_all.setdefault(vid, [0.5] * K_WINDOWS)
            tensors = self.pair_tensors(corpus, seg_all)
            for arm in ("B0", "B1", "B2", "B3"):
                data = self.arm_data_b(arm, tensors, corpus)
                res = select_on_partition("B", arm, "HateClipSeg", data, train_ids,
                                          config_list(arm), labels, h3=self.h3, t0=self.t0)
                scores, _, _, _ = score_rows(res.model, data, data.rows(val_ids))
                preds = (scores >= res.theta).astype(np.int64)
                out["macro_f1"][arm] = macro_f1(y_val, preds)
            b_base = max(out["macro_f1"]["B0"], out["macro_f1"]["B1"], out["macro_f1"]["B3"])
            out["b2_delta"] = out["macro_f1"]["B2"] - b_base
        return out

    def _conf_rows(self, arm, ids_val, labels, scores, preds, theta, cfg, epoch, seed,
                   dataset):
        rows = []
        for i, vid in enumerate(ids_val):
            rows.append(make_row(video_id=vid, dataset=dataset, split="val",
                                 confirmation=True, outer_fold=None,
                                 gold_label=labels[vid], arm=arm,
                                 score=float(scores[i]), prediction=int(preds[i]),
                                 threshold=theta,
                                 threshold_source="transferred:median_of_outer_folds",
                                 config_id=cfg["config_id"], epoch=epoch, seed=seed))
        return rows

    def transfer_hparams(self, per_fold, arm):
        """cfg_val = modal cfg* (ties -> smaller registered index); epoch/theta medians."""
        configs = config_list(arm)
        order = {c["config_id"]: i for i, c in enumerate(configs)}
        counts = {}
        for res in per_fold:
            counts[res.config["config_id"]] = counts.get(res.config["config_id"], 0) + 1
        best = sorted(counts.items(), key=lambda kv: (-kv[1], order[kv[0]]))[0][0]
        cfg_val = configs[order[best]]
        epoch_val = int(np.median([r.epoch for r in per_fold]))
        theta_val = float(np.median([r.theta for r in per_fold]))
        return cfg_val, epoch_val, theta_val

    # -- output ------------------------------------------------------------
    def write_outputs(self):
        self.metrics["run_id"] = self.run_id
        self.metrics["stages_run"] = self.stages
        self.metrics["failure_accounting"] = getattr(self, "failures", None)
        self.metrics["overlap_assertions"] = getattr(self, "overlap_assertions", None)
        if getattr(self, "confirmation_results", None):
            self.metrics["confirmation"] = self.confirmation_results
        if getattr(self, "a_results", None):
            try:
                self.write_fold_artifacts()
            except Exception as exc:                       # pragma: no cover
                note("fold artifacts not written: %r" % exc)
        metrics_path = write_json_new(self.run_dir / "metrics.json", self.metrics)

        required = []
        if self.gate_a and self.gate_a["pass"] and self.d_arm == "A3":
            required.append("the A3 advantage over A0/A1 is not capacity-controlled "
                            "at Gate-A")
        verdict = {
            "status": self.status,
            "halt": self.halt,
            "run_id": self.run_id,
            "stages_run": self.stages,
            "gate_c": self.gate_c,
            "gate_a": self.gate_a,
            "gate_b": self.gate_b,
            "arm_D": self.metrics.get("arm_D", {}).get("identity"),
            "verdict": overall_verdict(self.gate_c, self.gate_a, self.gate_b,
                                       forced_stage_b=getattr(self, "forced_stage_b",
                                                              False)),
            "forced_stage_b": getattr(self, "forced_stage_b", False),
            "required_records": required,
            "fixture_mode": bool(self.args.fixture_mode),
        }
        verdict_path = write_json_new(self.run_dir / "verdict.json", verdict)
        self.verdict = verdict
        try:
            self.write_manifest(metrics_path, verdict_path)
        except Exception as exc:                           # pragma: no cover
            note("manifest not written: %r" % exc)
            write_json_new(self.run_dir / "manifest_incomplete.json",
                           {"error": repr(exc), "status": self.status,
                            "halt": self.halt, "run_id": self.run_id})

    def write_manifest(self, metrics_path, verdict_path):
        guard_report = self.guard.report() if self.guard else {}
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             cwd=str(self.root)).decode().strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                                 cwd=str(self.root)).decode().strip())
        except Exception:
            commit, dirty = None, None
        manifest = {
            "run_id": self.run_id,
            "git_commit": commit, "git_dirty": dirty, "git_diff_sha256": None,
            "host": socket.gethostname(), "platform": platform.platform(),
            "cpu_model": platform.processor(),
            "nvidia_smi_device": _nvidia_smi(),
            "gpu_used": False,
            "python_version": sys.version.split()[0],
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "package_versions": _package_versions(),
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_flags": self.determinism,
            "command_line": " ".join(sys.argv),
            "scheduler": "none", "job_id": None,
            "log_path": os.environ.get("TERA_LOG_PATH"),
            "pid_file": os.environ.get("TERA_PID_FILE"),
            "start_utc": self.start_utc, "end_utc": utc_now(),
            "wall_clock_s": round(time.time() - self.t0, 3),
            "seeds": {
                "outer_folds": 20260807, "inner_folds": 20260808,
                "video_bootstrap": BOOTSTRAP_SEED, "temporal_bootstrap": BOOTSTRAP_SEED,
                "gate_c_sampling": 20260807, "b4_order_permutation": B4_SEED,
                "b5_donor_draw": B5_SEED, "model_init_base": 20260810,
            },
            "inputs": self.input_hashes(),
            "outputs": [{"path": str(p.relative_to(self.run_dir)),
                         "sha256": sha256_file(p), "bytes": p.stat().st_size}
                        for p in sorted(self.run_dir.rglob("*")) if p.is_file()],
            "split_source": self.corpus.split_source,
            "split_id_hash": sha256_ids(self.corpus.ids),
            "split_manifest_sha256": sha256_file(self.run_dir / "split_manifest.json"),
            "hateclipseg_surviving_id_hash": (sha256_ids(self.p11["train"] + self.p11["val"] +
                                                         self.p11["test"])
                                              if self.p11 else None),
            "authorized_id_hash": self.auth.hash_history,
            "sealed_ids_dropped": self.auth.dropped_totals(),
            "confirmation_unlock_utc": self.auth.confirmation_unlock_utc,
            "confirmation_passes": (self.confirmation_results["passes"]
                                    if getattr(self, "confirmation_results", None)
                                    else {"hatemm_val": 0, "hateclipseg_val": 0}),
            "encoder_id": self.frozen_config["payload"]["features"]["encoder_id"],
            "encoder_revision": None, "encoder_config_sha256": None,
            "duration_rule": self.frozen_config["payload"]["temporal_grid"][
                "duration_source_hatemm"],
            "boundary_rule": self.frozen_config["payload"]["temporal_grid"]["window_rule"],
            "frame_time_convention": self.frozen_config["payload"]["temporal_grid"][
                "frame_time_convention"],
            "sampling_alignment_proof": self.frozen_config["payload"]["temporal_grid"][
                "sampling_alignment_proof"],
            "zero_vector_videos": self.failures["zero_vector_videos"],
            "missing_duration_videos": self.failures["missing_duration_videos"],
            "failure_rate": self.failures["failure_rate"],
            "test_contact_count": guard_report.get("test_contact_count"),
            "opened_test_paths": guard_report.get("opened_test_paths"),
            "forbidden_paths": guard_report.get("forbidden_paths"),
            "overlap_assertions": getattr(self, "overlap_assertions", None),
            "msc_subset_sha256": getattr(self, "msc_subset_sha256", None),
            "eligible_videos_sha256": (sha256_file(self.run_dir / "eligible_videos.json")
                                       if (self.run_dir / "eligible_videos.json").exists()
                                       else None),
            "frozen_config_sha256": sha256_file(self.args.config),
            "frozen_config_payload_sha256": self.payload_hash,
            "appendix_sha256": self.frozen_config["payload"]["study"]["appendix_sha256"],
            "fixtures_report_sha256": os.environ.get("TERA_FIXTURES_REPORT_SHA256"),
            "fixture_code_sha256": os.environ.get("TERA_FIXTURE_CODE_SHA256"),
            "verdict_sha256": sha256_file(verdict_path),
            "metrics_sha256": sha256_file(metrics_path),
            "fixture_mode": bool(self.args.fixture_mode),
            "data_root": str(self.data_root),
        }
        write_json_new(self.run_dir / "manifest.json", manifest)
        self.manifest = manifest

    def input_hashes(self):
        out = []
        for entry in (self.corpus.cache_info.get("segment_cache"),
                      self.corpus.cache_info.get("wholevideo_cache")):
            if entry:
                out.append({"path": entry["path"], "sha256": entry["sha256"],
                            "bytes": entry["bytes"]})
        for rel in ("gt/HateMM/hate_spans.json", "gt/HateMM/train.jsonl"):
            path = self.data_root / rel
            if not path.exists():
                continue
            # provenance hashing of a corpus-spanning artifact reads bytes only and
            # never deserializes an id or a label; it runs inside the reader scope
            # so the sec 2.8 unrestricted-handle guard is satisfied explicitly.
            scope = self.guard.reader_scope() if self.guard else None
            if scope is not None:
                with scope:
                    digest = sha256_file(path)
            else:
                digest = sha256_file(path)
            out.append({"path": str(path), "sha256": digest,
                        "bytes": path.stat().st_size, "read_mode": "hash_only"})
        return out

    # -- driver -----------------------------------------------------------
    def run(self):
        self.start_utc = utc_now()
        self.prepare()
        try:
            self.build_authorization()
            self.load_primary()
            self.build_folds()
            if "A" in self.stages:
                self.run_stage_a()
                self.assemble_oof()
                self.compute_metrics()
            if "C" in self.stages:
                if self.gate_a is None:
                    raise TeraHalt("HALT_STAGE_ORDER", "C needs the A0 OOF run (stage A)")
                self.run_stage_c()
            if "B" in self.stages:
                if self.gate_a is None:
                    raise TeraHalt("HALT_STAGE_ORDER", "B requires stage A")
                if not self.gate_a["pass"] and not self.args.fixture_mode:
                    note("stage B skipped: Gate-A did not pass")
                else:
                    self.forced_stage_b = not self.gate_a["pass"]
                    self.run_stage_b()
            if self.args.confirmation != "none":
                self.run_confirmation()
                if "A" in self.stages:
                    self.gate_a = gate_a_decision(
                        {a: self.arm_metrics[a]["macro_f1"] for a in ("A0", "A1", "O1", "O2")}
                        | {"D": self.arm_metrics[self.d_arm]["macro_f1"]},
                        self.metrics["bootstrap"]["d_minus_base_ci"],
                        self.metrics["temporal"][self.d_arm]["mean_within_video_auroc"],
                        self.confirmation_summary("A"))
                if getattr(self, "_gate_b_inputs", None):
                    macro_b, ci_b, rescue_b = self._gate_b_inputs
                    self.gate_b = gate_b_decision(macro_b, ci_b, rescue_b,
                                                  self.confirmation_summary("B"))
        except TeraHalt as exc:
            self.status = exc.code
            self.halt = {"code": exc.code, "detail": exc.detail}
            note("HALT %s: %s" % (exc.code, exc.detail))
        finally:
            try:
                self.write_outputs()
            except Exception as exc:                       # pragma: no cover
                note("failed to write outputs: %r" % exc)
            if self.guard:
                self.guard.uninstall()
        return 0 if self.status == "COMPLETE" else 3


def _nvidia_smi():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name",
                                       "--format=csv,noheader"], stderr=subprocess.DEVNULL)
        return out.decode().strip().splitlines()[0]
    except Exception:
        return None


def _package_versions():
    import sklearn
    versions = {"torch": torch.__version__, "numpy": np.__version__,
                "sklearn": sklearn.__version__}
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except Exception:
        versions["transformers"] = None
    return versions


def build_parser():
    root = repo_root()
    ap = argparse.ArgumentParser(description="TERA Gate-0 harness")
    ap.add_argument("--data-root", default=str(root / "data"))
    ap.add_argument("--run-root", default=str(root / "artifacts/tera_gate0"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config",
                    default=str(root / "research-wiki/tera_gate0_frozen_config.draft.json"))
    ap.add_argument("--stages", default="A")
    ap.add_argument("--bootstrap-n", type=int, default=10000)
    ap.add_argument("--confirmation", default="none",
                    choices=("none", "hatemm_val", "all"))
    ap.add_argument("--gate-c-audit", default=None,
                    help="path to the adjudicated gate_c_audit.jsonl for stage C")
    ap.add_argument("--hooks", default=None,
                    help="fixture-only JSON of registered debug hooks")
    ap.add_argument("--fixture-mode", action="store_true",
                    help="synthetic-data mode: stage B may run without a Gate-A pass")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return Gate0Run(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
