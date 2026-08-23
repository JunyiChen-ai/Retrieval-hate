"""Fail-closed entry point for the trained 45-run synthetic closed loop.

This executor is intentionally separate from ``formal_runner.py`` and the signed
two-fold fixture.  It currently performs only provenance validation; training is
not allowed to start until the generalized implementation satisfies every item
below.  This prevents the hash-score dry run or the random-classifier synthetic
pipeline from being mistaken for trained evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .artifacts import (assert_run_contract, validate_full_prediction_rows, validate_run_deep,
                        validate_targets, validate_trace_join, write_jsonl)
from .bootstrap import (bootstrap_complete_runs, interval_gate, replicate,
                        resample_complete_run_ids)
from .closed_loop_fixture import fit, prob, registered_seed
from .common import (ContactLedger, atomic_json, atomic_write, canonical_bytes,
                     sha256_file)
from .nested import nested_fit_predict, predict_refit
from .protocol import macro_f1_binary, strongest_admissible, utility
from .selectors import b2_executions, execute_random_arm, execute_stateful_arm
from .selector_trainer import fit_router, nested_selector_fit, predict_selector
from .fresh5_confirmation import run as run_fresh5_confirmation


EXPECTED_SCHEDULE_SHA256 = "dcae6f71b1b9c2347354cf357abbe48bc66e394fbcc5549bb6515b34c885f687"
EXPECTED_TEMPLATE_SHA256 = "17b1d809b30a7123c0b1a07a32371bfff891e8c601421ca326a8ba2a5da5ac79"


def validate_inputs(schedule_path: Path, template_path: Path) -> dict:
    ledger = ContactLedger()
    ledger.register(schedule_path, "signed_45_run_schedule")
    ledger.register(template_path, "signed_trained_two_fold_template")
    if sha256_file(schedule_path) != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError("HALT_SCHEDULE_HASH")
    if sha256_file(template_path) != EXPECTED_TEMPLATE_SHA256:
        raise RuntimeError("HALT_TEMPLATE_HASH")
    schedule = json.loads(schedule_path.read_text())
    template = json.loads(template_path.read_text())
    runs = schedule.get("runs", [])
    keys = {(r["split_seed"], r["outer_fold"], r["refit_id"]) for r in runs}
    if len(runs) != 45 or len(keys) != 45:
        raise RuntimeError("HALT_NOT_EXACT_45_RUNS")
    if any(r.get("inner_count") != 4 for r in runs):
        raise RuntimeError("HALT_NOT_FOUR_INNER_FOLDS")
    if template.get("folds") != 2 or template.get("test_contact_count") != 0:
        raise RuntimeError("HALT_TEMPLATE_CONTRACT")
    return {"schedule": schedule, "template": template, "contact": ledger.snapshot()}


def _one_outer(root: Path, inputs: dict, registered: dict | None = None) -> None:
    """Run one real-shaped synthetic outer/refit through the final artifact path."""
    import torch
    if root.exists():
        raise FileExistsError(root)
    for d in ("groups", "actions", "selection", "targets", "predictions",
              "traces", "metrics", "fixtures"):
        (root / d).mkdir(parents=True)
    schedule = inputs["schedule"]
    registered = schedule["runs"][0] if registered is None else registered
    split_seed, outer_fold, refit = (registered[k] for k in
                                     ("split_seed", "outer_fold", "refit_id"))
    run_id = registered["run_id"]
    vids = [f"v{i}" for i in range(20)]
    groups = [f"sg{i // 2}" for i in range(20)]
    labels = np.asarray(([0, 1, 1, 0, 0, 1, 1, 0, 0, 1] * 2))
    syn_outer = inputs["template"]["synthetic_fold_artifact"][str(split_seed)]["folds"][outer_fold]
    syn_inner = inputs["template"]["synthetic_inner_artifact"][str(split_seed)][str(outer_fold)]["folds"]
    query_groups = set(syn_outer["group_ids"])
    query_ix = [i for i, g in enumerate(groups) if g in query_groups]
    train_ix = [i for i, g in enumerate(groups) if g not in query_groups]
    base = np.tile(np.eye(4), (5, 1))
    action_ids = [f"v:{kind}:{window:02d}" for window in (14, 15, 16)
                  for kind in ("ocr", "dense4")]
    action_cost = {a: float(1 + (j % 2)) for j, a in enumerate(action_ids)}
    outcome = np.stack([np.asarray([(i + a) % 2, (2 * i + a) % 3]) / 3
                        for i in range(20) for a in range(6)]).reshape(20, 6, 2)

    def phi(i, chosen):
        return np.r_[base[i], outcome[i, list(chosen)].sum(0) if chosen else [0, 0]]

    # Four-way group cross-fit target generation.  The imported fit/prob/seed
    # primitives are exactly those exercised and signed by two_fold_v10.
    train_groups = sorted(set(groups[i] for i in train_ix))
    inner_groups = [q["group_ids"] for q in syn_inner]
    targets = []
    classifier_records = []
    for inner_fold, eval_groups in enumerate(inner_groups):
        fit_groups = sorted(set(train_groups) - set(eval_groups))
        fit_ix = [i for i in train_ix if groups[i] in set(fit_groups)]
        x, y = [], []
        for i in fit_ix:
            for state in ((), (0,), (3,), (0, 3)):
                x.append(phi(i, state)); y.append(labels[i])
        seed = registered_seed(split_seed, outer_fold, inner_fold, refit,
                               "classifier", "trained45")
        torch.manual_seed(seed)
        model = fit(torch.nn.Linear(6, 1), x, y)
        model_hash = hashlib.sha256(canonical_bytes(
            {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()})).hexdigest()
        classifier_records.append({"inner_fold": inner_fold,
                                   "fit_groups": fit_groups,
                                   "eval_groups": eval_groups,
                                   "rng_seed": seed,
                                   "model_sha256": model_hash})
        for i in train_ix:
            if groups[i] not in set(eval_groups):
                continue
            local = []
            for state in ((), (0,), (3,)):
                for action in range(6):
                    if action in state:
                        continue
                    before = prob(model, phi(i, state))
                    after = prob(model, phi(i, tuple(state) + (action,)))
                    local.append({"video_id": vids[i], "eval_group": groups[i],
                                  "fit_groups": fit_groups, "state": list(state),
                                  "action_id": action_ids[action],
                                  "action_features": np.r_[base[i], action / 5,
                                                             len(state) / 3].tolist(),
                                  "cheap_features": base[i].tolist(),
                                  "state_features": (outcome[i,list(state)].sum(0) if state else np.zeros(2)).tolist(),
                                  "before_probability": before,
                                  "after_probability": after,
                                  "utility": utility(labels[i], before, after),
                                  "estimated_cost": action_cost[action_ids[action]],
                                  # Stateful executor charges the next decision
                                  # overhead before invoking its callback.
                                  "remaining_budget": 4.0-sum(action_cost[action_ids[q]] for q in state)-.25*(len(state)+1),
                                  "generator_provenance": [model_hash, str(inner_fold)],
                                  "terminal": False})
            weight = 1.0 / len(local)
            for row in local:
                row["weight"] = weight
            targets.extend(local)
    validate_targets(targets)

    arms = tuple(f"B{i}" for i in range(2, 13))
    inner_records=[{"inner_fold":k,"eval_groups":list(gs),
                    "fit_groups":sorted(set(train_groups)-set(gs))} for k,gs in enumerate(inner_groups)]
    selector_fit={arm:nested_selector_fit(targets,inner_records,arm,split_seed,outer_fold,refit,
                  configs=((.01,20),(.03,30))) for arm in ("B7","B8","B9","B10","B11","B12")}
    router=fit_router([base[i] for i in train_ix],[labels[i] for i in train_ix],
                      (split_seed,outer_fold,refit))
    rw=np.asarray(router["state"]["weight"])[0];rb=float(router["state"]["bias"][0])

    def learned_signals(i,state=()):
        remaining=4-sum(action_cost[action_ids[q]] for q in state)-.25*(len(state)+1)
        if abs(remaining-(4-sum(action_cost[action_ids[q]] for q in state)-.25*(len(state)+1)))>1e-9:raise RuntimeError("HALT_REMAINING_BUDGET_SEMANTICS")
        rows=[]
        for j,a in enumerate(action_ids):
            rows.append({"cheap_features":base[i].tolist(),
                         "state_features":(outcome[i,list(state)].sum(0) if state else np.zeros(2)).tolist(),
                         "action_features":np.r_[base[i],j/5,len(state)/3].tolist(),
                         "remaining_budget":remaining,
                         "estimated_cost":action_cost[a]})
        out={"salience":{a:float(outcome[i,j].sum()) for j,a in enumerate(action_ids)},
             "uncertainty":{a:float(1-abs(outcome[i,j].mean()-.5)) for j,a in enumerate(action_ids)},
             "remaining_budget":remaining}
        for arm,key in (("B7","B7"),("B8","B8"),("B9","singleton"),("B10","set_utility"),("B11","singleton_ridge")):
            sc=predict_selector(selector_fit[arm]["refit_model"],rows,arm);out[key]={a:float(sc[j]) for j,a in enumerate(action_ids)}
        # B12 is a separately selected/refit model, never an alias for B10.
        b12=predict_selector(selector_fit["B12"]["refit_model"],rows,"B12")
        out["B12_set_utility"]={a:float(b12[j]) for j,a in enumerate(action_ids)}
        return out

    def random_choice(rng, draw_id):
        return execute_random_arm(action_ids,action_cost,4,rng,decision_overhead=.25)

    def choices(i, arm, draw=None):
        if arm == "B2":
            return b2_executions(random_choice, split_seed, refit, vids[i], .5)
        router_positive=bool(1/(1+np.exp(-(base[i]@rw+rb)))>=.5)
        def callback(state):
            out=learned_signals(i,tuple(action_ids.index(a) for a in state))
            if arm=="B12":out["set_utility"]=out["B12_set_utility"]
            return out
        ex=execute_stateful_arm(arm,action_ids,action_cost,4,callback,
             router_positive=router_positive,decision_overhead=.25)
        return [{"draw_id":draw,**ex}]

    def features(i, chosen):
        ids = [action_ids.index(a) for a in chosen]
        return np.r_[phi(i, ids), len(ids) / 3,
                     sum(action_cost[a] for a in chosen) / 4]

    selection, fitted = [], {}
    for arm in arms:
        representative = [choices(i, arm)[0]["actions"] for i in range(20)]
        xx = np.stack([features(i, c) for i, c in enumerate(representative)])
        result = nested_fit_predict(xx, labels, groups, query_groups,
                                    split_seed, refit,
                                    configs=(("lr1e-2", .01), ("lr3e-2", .03)), epochs=30)
        fitted[arm] = result
        selection.append({"schema": "cvoi-inner-selection/2",
                          "procedure_run_id": run_id, "split_seed": split_seed,
                          "outer_fold": outer_fold, "refit_seed": refit,
                          "arm_id": arm, **result["selection"],
                          "threshold": result["threshold"],
                          "threshold_source": "four_fold_inner_oof",
                          "model_sha256": result["refit_model_sha256"]})

    predictions, traces = [], []
    for i in query_ix:
        for arm in arms:
            for ex in choices(i, arm):
                chosen = tuple(ex["actions"]); draw_id = ex.get("draw_id")
                result = fitted[arm]
                score = float(predict_refit(result["refit_state"],
                                             features(i, chosen)[None])[0])
                decisions=ex["decisions"]
                trace_hash = hashlib.sha256(canonical_bytes(decisions)).hexdigest()
                common = {"run_id": run_id, "outer_split_seed": split_seed,
                          "refit_seed": refit, "video_id": vids[i], "arm_id": arm,
                          "budget_fraction": .5, "draw_id": draw_id}
                traces.append({"schema": "cvoi-acquisition-trace/1", **common,
                               "ordered_actions": chosen, "estimated_cost_ms": ex["estimated_cost"],
                               "realized_cost_ms": ex["realized_cost"],
                               "decision_overhead_ms":ex["overhead_cost"],
                               "decision_records": decisions, "trace_sha256": trace_hash})
                predictions.append({"schema": "cvoi-prediction/1", **common,
                                    "dataset": "SYNTHETIC", "split_role": "outer_oof",
                                    "group_id": groups[i], "outer_fold": outer_fold,
                                    "score": score,
                                    "prediction": int(score >= result["threshold"]),
                                    "threshold": result["threshold"],
                                    "threshold_source": "four_fold_inner_oof",
                                    "estimated_budget_ms": 4.0,
                                    "realized_cost_ms":ex["realized_cost"],
                                    "action_trace_sha256": trace_hash,
                                    "config_id": result["selection"]["selected_config"],
                                    "epoch": result["selection"]["outer_refit_epoch"],
                                    "model_sha256": result["refit_model_sha256"],
                                    "payload_sha256": "PENDING_MANIFEST"})
    validate_full_prediction_rows(predictions, 120)
    validate_trace_join(predictions, traces)

    # One-run structural bootstrap; the 45-run expansion will use
    # bootstrap_complete_runs with the same complete comparator envelope.
    def emitted(arm,video):
        q=[p for p in predictions if p["arm_id"]==arm and p["video_id"]==video]
        return float(np.mean([x["score"] for x in q])),float(np.mean([x["realized_cost_ms"] for x in q])),q[0]["threshold"]
    scores={a:[emitted(a,vids[i])[0] for i in query_ix] for a in arms}
    costs={a:[emitted(a,vids[i])[1] for i in query_ix] for a in arms}
    thresholds={a:emitted(a,vids[query_ix[0]])[2] for a in arms}
    boot = replicate(labels[query_ix], scores, costs,
                     [groups[i] for i in query_ix], thresholds, "B10",
                     [a for a in arms if a != "B10"], n_boot=10000)

    val_rows, val_traces, confirmation_lineage = run_fresh5_confirmation(
        inputs["template"], split_seed, refit, arms, vids, groups, labels,
        base, outcome, action_ids, action_cost, phi, features,
        inputs["contact"]["test_contact_count"])
    for row in val_rows:
        row.update({"run_id":run_id+"-fresh5-confirmation",
                    "dataset":"SYNTHETIC_CONFIRMATION",
                    "split_role":"confirmation_inference_only",
                    "outer_split_seed":split_seed,"outer_fold":-1,
                    "refit_seed":refit,"budget_fraction":.5,
                    "estimated_budget_ms":4.,"config_id":"fresh5-selected",
                    "epoch":30,"model_sha256":confirmation_lineage["state_classifier_sha256"]})
    for row in val_traces:
        row.update({"run_id":run_id+"-fresh5-confirmation","outer_split_seed":split_seed,
                    "refit_seed":refit,"budget_fraction":.5})

    # Persist learned-arm selection/refit lineage, including the B12 selector
    # that shares B10's stateful execution semantics.
    for row in selection:
        if row["arm_id"] in selector_fit:
            sf=selector_fit[row["arm_id"]]
            row["selector_selected_config"]=sf["selected"]["config_id"]
            row["selector_refit_sha256"]=sf["refit_model"]["sha256"]
            row["selector_config_records"]=sf["configs"]

    sources = [Path("scripts/cvoi_acq/trained45_executor.py"),
               Path("scripts/cvoi_acq/closed_loop_fixture.py"),
               Path("scripts/cvoi_acq/nested.py"),
               Path("scripts/cvoi_acq/selectors.py"),
               Path("scripts/cvoi_acq/selector_trainer.py"),
               Path("scripts/cvoi_acq/fresh5_confirmation.py")]
    source_sha = {str(p): sha256_file(p) for p in sources}
    payload_sha = hashlib.sha256(canonical_bytes(
        {"run_root": root.name, "source_sha256": source_sha})).hexdigest()
    for row in predictions:
        row["payload_sha256"] = payload_sha
    for row in selection:
        row["payload_sha256"] = payload_sha
    for row in val_rows:
        row["payload_sha256"] = payload_sha
    validate_full_prediction_rows(val_rows, 300)
    atomic_json(root / "manifest.json", {"schema": "cvoi-trained45-manifest/1",
                "synthetic": True, "run_root": root.name,
                "source_sha256": source_sha, "payload_sha256": payload_sha,
                "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
                "template_sha256": EXPECTED_TEMPLATE_SHA256})
    atomic_json(root / "asset_registry.json", {"synthetic": True, "assets": []})
    write_jsonl(root / "groups/sources.jsonl", [{"video_id": v, "group_id": g,
                "label": int(y)} for v, g, y in zip(vids, groups, labels)])
    write_jsonl(root / "groups/edges.jsonl", [])
    atomic_json(root / "groups/components.json", {g: [v for v, q in zip(vids, groups)
                if q == g] for g in sorted(set(groups))})
    atomic_json(root / "groups/folds.json", {"outer": 1, "inner": 4})
    write_jsonl(root / "actions/ocr_actions.jsonl", [{"action_id": a} for a in action_ids])
    atomic_write(root / "actions/dense_frames.f32", np.zeros((1, 4, 1024), "<f4").tobytes())
    write_jsonl(root / "actions/dense_sidecar.jsonl", [{"synthetic": True}])
    write_jsonl(root / "actions/cost_actions.jsonl", [{"action_id": a, "cost": c}
                for a, c in action_cost.items()])
    atomic_json(root / "actions/cost_summary.json", {"synthetic": True})
    write_jsonl(root / "selection/inner_selection.jsonl", selection)
    import pyarrow as pa
    import pyarrow.parquet as pq
    sink = pa.BufferOutputStream(); pq.write_table(pa.Table.from_pylist(targets), sink)
    atomic_write(root / "targets/utility_targets.parquet", sink.getvalue().to_pybytes())
    write_jsonl(root / "predictions/train_oof.jsonl", predictions)
    write_jsonl(root / "predictions/val_confirmation.jsonl", val_rows)
    all_traces=traces+val_traces
    for trace in all_traces:
        decisions=trace["decision_records"]
        if not decisions or any(float(d.get("decision_overhead_ms",0))<=0 for d in decisions):
            raise RuntimeError("HALT_UNCHARGED_DECISION_TRACE")
        stops=[d for d in decisions if d["status"]=="STOP"]
        if not stops or any(float(d["decision_overhead_ms"])<=0 for d in stops):
            raise RuntimeError("HALT_UNCHARGED_STOP_TRACE")
        action_total=sum(float(d.get("realized_cost_ms",0)) for d in decisions if d["status"]=="ACQUIRE")
        overhead_total=sum(float(d["decision_overhead_ms"]) for d in decisions)
        if abs(trace["realized_cost_ms"]-(action_total+overhead_total))>1e-9 or trace["realized_cost_ms"]>4+1e-9:
            raise RuntimeError("HALT_TRACE_COST_CONSERVATION")
    validate_trace_join(predictions+val_rows,all_traces)
    write_jsonl(root / "traces/acquisitions.jsonl", all_traces)
    atomic_json(root / "metrics/metrics.json", {"synthetic": True,
                "candidate_metric_firewall": True, "comparator_envelope": list(arms),
                "bootstrap": interval_gate([x["delta"] for x in boot])})
    bio = io.BytesIO(); np.savez_compressed(bio, delta=np.asarray([x["delta"] for x in boot]))
    atomic_write(root / "metrics/bootstrap.npz", bio.getvalue())
    atomic_json(root / "fixtures/report.json", {"passed": True,
                "scope": "one_outer_one_refit",
                "fresh5_confirmation_lineage": confirmation_lineage})
    write_jsonl(root / "resources.jsonl", [{"event": "complete", "synthetic": True}])
    atomic_write(root / "RESULTS.md", b"# Trained one-outer structural run\n")
    assert_run_contract(root)
    atomic_json(root / "trained45_audit.json", {"scope": "one_outer_one_refit",
                "trained_target_generator": True, "trained_arm_classifiers": True,
                "prediction_rows": len(predictions), "trace_rows": len(traces)+len(val_traces),
                "target_rows": len(targets), "selection_rows": len(selection),
                "bootstrap_replicates": len(boot), "full_comparator_envelope": True,
                "classifier_records": classifier_records,
                "outer_selector_refits": {arm:{
                    "selected_config":selector_fit[arm]["selected"]["config_id"],
                    "refit_model_sha256":selector_fit[arm]["refit_model"]["sha256"]}
                    for arm in ("B7","B8","B9","B10","B11","B12")},
                "fresh5_confirmation": confirmation_lineage,
                "contact": inputs["contact"]})
    validate_run_deep(root, expected_train_rows=120)


def _validate_smoke_child(root: Path) -> dict:
    """Independent post-write checks required before indexing a smoke child."""
    deep=validate_run_deep(root,expected_train_rows=120)
    trace_rows=[json.loads(x) for x in (root/"traces/acquisitions.jsonl").open() if x.strip()]
    for trace in trace_rows:
        ds=trace["decision_records"]
        if not ds or any(float(d.get("decision_overhead_ms",0))<=0 for d in ds):
            raise RuntimeError("HALT_SMOKE_UNCHARGED_DECISION")
        if not any(d["status"]=="STOP" and float(d["decision_overhead_ms"])>0 for d in ds):
            raise RuntimeError("HALT_SMOKE_UNCHARGED_STOP")
        actions=sum(float(d.get("realized_cost_ms",0)) for d in ds if d["status"]=="ACQUIRE")
        overhead=sum(float(d["decision_overhead_ms"]) for d in ds)
        if abs(float(trace["realized_cost_ms"])-actions-overhead)>1e-9 or float(trace["realized_cost_ms"])>4+1e-9:
            raise RuntimeError("HALT_SMOKE_COST_CONSERVATION")
    return {"deep":deep,"trace_count":len(trace_rows),"cost_invariants":True}


def _aggregate_complete_runs(root: Path, completed: list[dict]) -> dict:
    """Bind 45 outer children into nine 5-fold-complete statistical runs."""
    arms=[f"B{i}" for i in range(2,13)];runs=[]
    by={(int(r["split_seed"]),int(r["refit_id"])):[] for r in completed}
    for r in completed:by[(int(r["split_seed"]),int(r["refit_id"]))].append(r)
    if len(by)!=9 or any(len(v)!=5 for v in by.values()):raise RuntimeError("HALT_COMPLETE_RUN_BINDING")
    original=[]
    for (split,refit),children in sorted(by.items()):
        if sorted(int(x["outer_fold"]) for x in children)!=list(range(5)):raise RuntimeError("HALT_COMPLETE_OUTER_COVERAGE")
        src=[json.loads(x) for x in (root/"runs"/children[0]["run_id"]/"groups/sources.jsonl").open() if x.strip()]
        vids=[x["video_id"] for x in src];labels=[int(x["label"]) for x in src];groups=[x["group_id"] for x in src]
        rows=[]
        for child in children:rows.extend(json.loads(x) for x in (root/"runs"/child["run_id"]/"predictions/train_oof.jsonl").open() if x.strip())
        scores={};costs={};thresholds={}
        for arm in arms:
            scores[arm]=[];costs[arm]=[];thresholds[arm]=[]
            for vid in vids:
                q=[x for x in rows if x["arm_id"]==arm and x["video_id"]==vid]
                if len(q)!=(20 if arm=="B2" else 1):raise RuntimeError("HALT_COMPLETE_EMITTED_ROWS")
                scores[arm].append(float(np.mean([x["score"] for x in q])))
                costs[arm].append(float(np.mean([x["realized_cost_ms"] for x in q])))
                thresholds[arm].append(float(q[0]["threshold"]))
        profiles={g:"mixed" for g in set(groups)}
        run={"split_seed":split,"refit":refit,"y":labels,"groups":groups,"profiles":profiles,
             "scores":scores,"costs":costs,"thresholds":thresholds}
        runs.append(run)
        f={a:macro_f1_binary(np.asarray(labels),np.asarray(scores[a])>=np.asarray(thresholds[a])) for a in arms}
        c={a:float(np.mean(costs[a])) for a in arms}
        best=strongest_admissible(f["B10"],c["B10"],[(a,f[a],c[a]) for a in arms if a!="B10"])
        original.append({"split_seed":split,"refit":refit,"candidate_f1":f["B10"],"candidate_cost":c["B10"],"selected_baseline":best[0],"delta":f["B10"]-best[1]})
    boot=bootstrap_complete_runs(runs,"B10",[a for a in arms if a!="B10"],n_boot=10000,seed=20260819)
    sensitivity=resample_complete_run_ids([x["delta"] for x in original],n_boot=10000,seed=20260820)
    payload={"schema":"cvoi-synthetic-complete-run-aggregation/1","synthetic":True,
      "statistical_unit":"3 split seeds x 3 refits; each run concatenates 5 outer folds",
      "not_independent_runs":"45 children are outer-fold executions, not 45 independent statistical runs",
      "complete_run_count":9,"outer_child_count":45,"binding_bootstrap_replicates":10000,
      "full_comparator_reselection_each_replicate":True,"candidate":"B10","comparators":[a for a in arms if a!="B10"],
      "binding_delta":interval_gate([x["delta"] for x in boot]),
      "complete_run_id_sensitivity":interval_gate(sensitivity.tolist()),
      "pareto":{"joint_delta_positive_cost_saving_nonnegative_fraction":float(np.mean([x["delta"]>0 and x["cost_saving"]>=0 for x in boot])),
                "cost_saving":interval_gate([x["cost_saving"] for x in boot])},
      "original_complete_runs":original}
    atomic_json(root/"complete_run_aggregation.json",payload)
    bio=io.BytesIO();np.savez_compressed(bio,delta=np.asarray([x["delta"] for x in boot]),cost_saving=np.asarray([x["cost_saving"] for x in boot]),sensitivity=sensitivity)
    atomic_write(root/"complete_run_bootstrap.npz",bio.getvalue())
    return {"aggregation_sha256":sha256_file(root/"complete_run_aggregation.json"),"bootstrap_sha256":sha256_file(root/"complete_run_bootstrap.npz"),"complete_run_count":9,"bootstrap_replicates":10000}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-count", type=int, default=1,
                        help="locked to 1 until the one-outer audit passes")
    args = parser.parse_args()
    inputs = validate_inputs(args.schedule, args.template)
    if args.run_count not in (1,2,4,16,45):
        raise RuntimeError("HALT_EXPANSION_OUTSIDE_SIGNED_EXACT_45")
    if args.run_count == 1:
        _one_outer(args.out_dir, inputs)
        return
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    (args.out_dir / "runs").mkdir(parents=True)
    completed = []
    registered_runs=inputs["schedule"]["runs"][:args.run_count]
    schedule_keys=[(r["run_id"],r["split_seed"],r["outer_fold"],r["refit_id"]) for r in registered_runs]
    if len(schedule_keys)!=args.run_count or len(set(schedule_keys))!=args.run_count:
        raise RuntimeError("HALT_SMOKE_SCHEDULE_UNIQUENESS")
    for registered in registered_runs:
        child = args.out_dir / "runs" / registered["run_id"]
        _one_outer(child, inputs, registered)
        smoke_validation=_validate_smoke_child(child)
        child_manifest=json.loads((child/"manifest.json").read_text())
        child_audit=json.loads((child/"trained45_audit.json").read_text())
        state_fingerprint=hashlib.sha256(canonical_bytes({
            "classifier_records":child_audit["classifier_records"],
            "outer_selector_refits":child_audit["outer_selector_refits"],
            "fresh5_state_classifier":child_audit["fresh5_confirmation"]["state_classifier_sha256"],
            "fresh5_selectors":child_audit["fresh5_confirmation"]["selector_sha256"],
            "fresh5_router":child_audit["fresh5_confirmation"]["router_sha256"]})).hexdigest()
        completed.append({"run_id": registered["run_id"],
                          "outer_fold": registered["outer_fold"],
                          "split_seed": registered["split_seed"],
                          "refit_id": registered["refit_id"],
                          "manifest_sha256": sha256_file(child / "manifest.json"),
                          "audit_sha256": sha256_file(child / "trained45_audit.json"),
                          "payload_sha256":child_manifest["payload_sha256"],
                          "state_fingerprint":state_fingerprint,
                          "smoke_validation":smoke_validation})
    if len(completed)!=args.run_count or len({x["run_id"] for x in completed})!=args.run_count:
        raise RuntimeError("HALT_SMOKE_COMPLETION")
    if len({x["payload_sha256"] for x in completed})!=args.run_count or len({x["state_fingerprint"] for x in completed})!=args.run_count:
        raise RuntimeError("HALT_SMOKE_STATE_REUSE")
    aggregation=_aggregate_complete_runs(args.out_dir,completed) if args.run_count==45 else None
    atomic_json(args.out_dir / "run_index.json", {
        "schema": "cvoi-trained45-index/1", "synthetic": True,
        "requested_runs": args.run_count, "completed_runs": len(completed),
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "template_sha256": EXPECTED_TEMPLATE_SHA256, "runs": completed,
        "candidate_metric_firewall": True,
        "complete_run_aggregation":aggregation,
        "schedule_keys":schedule_keys,
        "test_contact_count": inputs["contact"]["test_contact_count"]})


if __name__ == "__main__":
    main()
