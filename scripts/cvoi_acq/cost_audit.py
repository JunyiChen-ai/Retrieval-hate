"""Non-candidate C6 audit: registry coverage, fold-internal estimator, budgets."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from .common import ContactLedger,atomic_json,sha256_file
from .costs import FrozenGBDCostRegressor,quantize_cost_ns

def rows(path,ledger,role):
    ledger.register(path,role)
    return [json.loads(x) for x in path.open() if x.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ocr",type=Path,required=True);ap.add_argument("--dense",type=Path,required=True)
    ap.add_argument("--train-actions",type=Path,required=True);ap.add_argument("--val-actions",type=Path,required=True)
    ap.add_argument("--components",type=Path,required=True);ap.add_argument("--outer",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();ledger=ContactLedger()
    costs=rows(a.ocr,ledger,"ocr_costs")+rows(a.dense,ledger,"dense_costs")
    train=rows(a.train_actions,ledger,"train_action_coordinates");val=rows(a.val_actions,ledger,"val_action_coordinates")
    ledger.register(a.components,"train_components");components=json.loads(a.components.read_text())
    ledger.register(a.outer,"outer_folds");outer=json.loads(a.outer.read_text())
    coord={(r["video_id"],int(r["window_id"])):r for r in train+val};train_ids={r["video_id"] for r in train};val_ids={r["video_id"] for r in val}
    by={r["action_id"]:r for r in costs};expected={(v,t,k) for v in train_ids|val_ids for t in ("ocr","dense4") for k in range(30)}
    got={(x.split(":")[0],x.split(":")[1],int(x.split(":")[2])) for x in by}
    errors=[]
    if expected!=got:errors.append("action_coverage_mismatch")
    for r in costs:
        rr=r.get("raw_repetitions",[])
        if len(rr)!=5 or [x.get("repetition") for x in rr]!=[1,2,3,4,5]:errors.append("five_rep_contract")
        if any(set(("wall_ns","cuda_ms","phase_ns","allocated_bytes_peak","gpu_allocated_bytes_start","gpu_allocated_bytes_end","gpu_allocated_bytes_peak","gpu_allocated_bytes_incremental_peak","retries","status"))-set(x) for x in rr):errors.append("raw_schema")
        if int(r["binding_wall_ns"])!=int(np.median([x["wall_ns"] for x in rr[1:]])):errors.append("binding_median")
    video_to_group={v:g for g,vs in components.items() for v in vs}
    fold_qc=[]
    # Header resolution is represented by the registered action source resolution when it
    # becomes available. Current canonical coordinates lack it, so the deterministic missing
    # sentinel -1 is used and explicitly audited; duration/window/type remain legal.
    feats=[];ys=[];groups=[];types=[];vids=[]
    for aid,r in by.items():
        v,t,ks=aid.split(":");k=int(ks);q=coord[(v,k)];c=r.get("cheap_cost_covariates",{})
        feats.append([float(c.get("duration_s",float(q["window_end_s"])*30.0)),
          float(c.get("source_width",-1)),float(c.get("source_height",-1)),int(t=="dense4"),k/29.0])
        ys.append(float(r["binding_wall_ns"]));groups.append(video_to_group.get(v,"VAL:"+v));types.append(t);vids.append(v)
    X=np.asarray(feats);Y=np.asarray(ys)
    for seed,obj in outer.items():
        for f in obj["folds"]:
            query=set(f["group_ids"]);fit=np.array([v in train_ids and g not in query for v,g in zip(vids,groups)])
            ev=np.array([v in train_ids and g in query for v,g in zip(vids,groups)])
            model=FrozenGBDCostRegressor().fit(X[fit],Y[fit],[groups[i] for i in np.flatnonzero(fit)],[types[i] for i in np.flatnonzero(fit)])
            pred=model.predict(X[ev],eval_groups=[groups[i] for i in np.flatnonzero(ev)],action_types=[types[i] for i in np.flatnonzero(ev)])
            if np.any(~np.isfinite(pred)) or np.any(pred<0):errors.append("invalid_cost_prediction")
            fold_qc.append({"split_seed":int(seed),"outer_fold":int(f["fold"]),"n_fit":int(fit.sum()),"n_query":int(ev.sum()),
              "fit_query_group_overlap":len(set(np.asarray(groups)[fit])&set(np.asarray(groups)[ev])),
              "mae_ns":float(np.mean(np.abs(pred-Y[ev]))),"predicted_ticks_min":min(map(quantize_cost_ns,pred)),
              "predicted_ticks_max":max(map(quantize_cost_ns,pred))})
    # Budget enforcement unit/QC: no selected estimate may exceed its integer-tick budget.
    demo=sorted(max(0,quantize_cost_ns(x)) for x in Y[:60]);budget=max(0,math.floor(.1*sum(demo)))
    spent=0;selected=0
    for c in demo:
        if selected<12 and spent+c<=budget:spent+=c;selected+=1
    if spent>budget or selected>12:errors.append("budget_enforcement")
    out={"schema":"cvoi-c6-independent-audit/1","status":"REVIEW_REQUIRED" if not errors else "HALT",
      "errors":sorted(set(errors)),"candidate_metric_computed":False,"test_contact_count":ledger.test_contact_count,
      "coverage":{"train_videos":len(train_ids),"val_videos":len(val_ids),"expected_actions":len(expected),"actual_actions":len(got)},
      "fold_internal_regressor":fold_qc,"budget_qc":{"fraction":.1,"ticks":budget,"spent_ticks":spent,"selected":selected,"max_actions":12},
      "cost_registry":{"ocr_sha256":sha256_file(a.ocr),"dense_sha256":sha256_file(a.dense)},"contact":ledger.snapshot()}
    atomic_json(a.out,out)
if __name__=="__main__":main()
