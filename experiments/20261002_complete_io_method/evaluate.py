"""Frozen initial backbone + copy likelihood + qmixSG_rt10, on full val/test cohorts.

This evaluates the preselected test-objective and validation-objective trial from
each completed study. It does not reselect trials using the integrated results.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import pickle
import socket
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import cpolicy, data as qdata, policy
from qtl.checkpoint import load_trial
from qtl.copy_noise import BUCKETS, estimate_pi_neg, bucket_of
from qtl.inside_outside import make_model
from qtl.stop_rules import rule_values, add_quantile_rules, add_relaxed_rules, QMIX
import hier_evidence_common as hc

RULE = "qmixSG_rt10"
FIXED = (0, 8, 32)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def selected_trials(studies):
    selected = {}
    for study in studies:
        study = Path(study).resolve()
        report = json.loads((study / "study_summary.json").read_text())
        budget = json.loads((study / "budget.json").read_text())["n_trials"]
        assert budget == report["n_trials"] and budget in (5, 20)
        assert len(report["trials"]) == budget
        assert all(t["state"] == "COMPLETE" for t in report["trials"])
        # These are full training searches, not the locked-configuration diagnostics.
        for trial in report["trials"]:
            s = json.loads((study / f'trial{trial["number"]}' / "summary.json").read_text())
            assert s["cfg"]["max_epoch"] == 50
            assert [r["epoch"] for r in s["history"]] == list(range(1, 51))
        for selection, key in (("test_selected", "best"), ("validation_selected", "validation_selected")):
            trial = study / f'trial{report[key]["number"]}'
            row = selected.setdefault(str(trial), {
                "seed": report["seed"], "study": str(study), "selections": [],
                "trial": report[key]["number"], "source_budget": budget})
            row["selections"].append(selection)
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--studies", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "summary.json").exists(), "Completed evaluation exists; inspect instead of overwriting"
    manifest = selected_trials(a.studies)
    write_json(out / "selection.json", manifest)
    print(f"host {socket.gethostname()} | {datetime.now().astimezone().isoformat()} | {a.corpus}", flush=True)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    assert not (set(ids["train"]) & set(ids["val"]) or set(ids["train"]) & set(ids["test"])
                or set(ids["val"]) & set(ids["test"]))
    qdata.configure_source("soft_both", 8)
    answers, answer_T = qdata.load_answers(a.corpus, "soft_both")
    assert all(v in answers for split in ids.values() for v in split)
    store = qdata.Store(a.corpus, ids["val"] + ids["test"], ["bert"])
    assert all(store.T[v] == answer_T[v] for v in store.T)
    report = {"host": socket.gethostname(), "corpus": a.corpus, "method": "initial_inside_outside+copy_neg_both+qmixSG_rt10",
              "rule": RULE, "threshold_target_validation_mean_calls": 8, "max_calls": 32,
              "copy_fit_split": "negative videos from train", "quantile_reference_split": "val",
              "selection_protocol": "Frozen winners of complete fixed8 no-copy training studies; no integrated-result reselection",
              "trials": {}, "started_at": datetime.now().astimezone().isoformat()}
    for trial, selection in manifest.items():
        started = time.time()
        dest = out / f'seed{selection["seed"]}_trial{selection["trial"]}'
        dest.mkdir(parents=True, exist_ok=True)
        summary, cfg, model, am, chain = load_trial(trial, a.device, answers, ids, model_factory=make_model)
        assert summary["corpus"] == a.corpus and summary["seed"] == selection["seed"]
        assert cfg["backbone"] == "inside_outside" and cfg["io_outside"] and cfg["io_merge"] == "gated"
        assert cfg["answer_source"] == "soft_both" and cfg["soft_levels"] == 8 and cfg["node_prior"]
        assert cfg["order"] == "eig" and cfg["fusion"] == "tree" and cfg["answer_model"] == "anchored"
        assert cfg["text_sources"] == ["bert"]
        write_json(dest / "config.json", {"source_trial": trial, "source_config": cfg, "copy_pi": "neg",
                    "copy_mode": "both", "stop_rule": RULE, "validation_target_calls": 8, "max_calls": 32,
                    "selected_checkpoint_epoch": summary["best_epoch"], **selection})
        pi, same_rate, independent_rate, pair_count = estimate_pi_neg(
            answers, labels, ids["train"], answer_T, am, cfg["categories"])
        copy_fn = lambda length, p=pi: float(p[bucket_of(length)])
        runs = {}
        for split in ("val", "test"):
            order = sorted(ids[split], key=lambda v: store.T[v])
            runs[split] = {}
            batch = int(cfg["eval_chunk"])
            for i in range(0, len(order), batch):
                with torch.no_grad():
                    runs[split].update(cpolicy.run_batch(model, store, order[i:i + batch], am, chain, answers,
                        cfg["categories"], 32, a.device, record_voi=True, copy_pi=copy_fn, copy_mode="both"))
                print(f'{Path(trial).parent.name}/{Path(trial).name} {split} {min(i+batch,len(order))}/{len(order)}', flush=True)
            assert set(runs[split]) == set(ids[split])
            for v, r in runs[split].items():
                assert len(r["scores"]) == len(r["eig"]) + 1 == len(r["p_G"])
                assert len(r["asked"]) == len(r["eig"])
                assert all(len(score) == store.T[v] and np.isfinite(score).all() for score in r["scores"])
        with (dest / "runs.pkl").open("wb") as f:
            pickle.dump({"runs": runs, "T": store.T}, f)
        vals = {split: {v: rule_values(r, store.T[v]) for v, r in runs[split].items()} for split in runs}
        # Exact existing r5 transformations; references use validation only, never test values or labels.
        refs = {}
        for component in QMIX["qmixSG"]:
            values = np.concatenate([np.asarray(vals["val"][v][component], dtype=float) for v in vals["val"]])
            refs[component] = np.sort(values[np.isfinite(values)]).tolist()
        add_quantile_rules(vals)
        add_relaxed_rules(vals)
        threshold, val_mean = policy.calibrate([vals["val"][v][RULE] for v in ids["val"]], 8.0)
        write_json(dest / "calibration.json", {
            "threshold": float(threshold), "validation_mean_calls": float(val_mean), "rule": RULE,
            "relaxation": 0.10, "quantile_reference": refs, "reference_video_ids": ids["val"],
            "copy": {"buckets": BUCKETS, "pi": pi.tolist(), "same_rate": same_rate.tolist(),
                     "independent_rate": independent_rate.tolist(), "pair_count": pair_count.tolist(),
                     "source_ids": [v for v in ids["train"] if labels[v] == 0]}})
        row = {**selection, "source_trial": trial, "checkpoint_epoch": summary["best_epoch"],
               "output": str(dest), "validation_mean_calls": float(val_mean), "results": {}}
        for name in [f"fixed{b}" for b in FIXED] + [RULE]:
            calls = {v: (min(int(name[5:]), len(r["eig"])) if name.startswith("fixed")
                         else policy.stop_calls(vals["test"][v][RULE], threshold)) for v, r in runs["test"].items()}
            scores = {v: runs["test"][v]["scores"][calls[v]] for v in ids["test"]}
            sp, mp = dest / f"scores_test_{name}.jsonl", dest / f"metrics_test_{name}.json"
            hc.write_scores(str(sp), scores)
            metric = hc.run_evaluator(a.corpus, "test", str(sp), str(mp))
            r = metric["results"]["score_av"]
            values = [r["pr_auc"], r["roc_auc"], r["per_video"]["macro_auc"]]
            assert r["n_videos_missing_from_scores"] == 0 and r["n_videos_not_in_gold"] == 0
            assert r["n_videos"] == len(ids["test"])
            assert r["n_frames"] == sum(len(gt["test"][v]) for v in ids["test"])
            assert all(math.isfinite(x) for x in values)
            counts = np.asarray(list(calls.values()))
            row["results"][name] = {"metric_source": str(mp), "ap_roc_within": values,
                "mean_calls": float(counts.mean()), "calls_per_video": calls,
                "calls_quantiles": np.percentile(counts, [0, 25, 50, 75, 100]).tolist(),
                "mean_calls_positive": float(np.mean([calls[v] for v in calls if labels[v] == 1])),
                "mean_calls_negative": float(np.mean([calls[v] for v in calls if labels[v] == 0]))}
        row["seconds"] = time.time() - started
        write_json(dest / "summary.json", row)
        report["trials"][trial] = row
        write_json(out / "progress.json", report)
        print(json.dumps({"trial": trial, "seconds": row["seconds"], "complete": True}), flush=True)
        del model, am, chain, runs, vals
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    report["ended_at"] = datetime.now().astimezone().isoformat()
    write_json(out / "summary.json", report)


if __name__ == "__main__":
    main()
