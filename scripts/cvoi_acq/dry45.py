from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np
from .artifacts import validate_full_prediction_rows,validate_trace_join,write_jsonl
from .bootstrap import bootstrap_complete_runs,pareto_dominates,resample_complete_run_ids
from .common import ContactLedger,atomic_json,canonical_bytes,sha256_file
from .selectors import b2_executions,execute_registered_arm

def unit(*parts):return int.from_bytes(hashlib.sha256(canonical_bytes(parts)).digest()[:8],"big")/2**64
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--schedule",type=Path,required=True);ap.add_argument("--template",type=Path,required=True);ap.add_argument("--out-dir",type=Path,required=True);a=ap.parse_args()
    if a.out_dir.exists():raise FileExistsError(a.out_dir)
    a.out_dir.mkdir(parents=True);ledger=ContactLedger();ledger.register(a.schedule,"formal_45_schedule");ledger.register(a.template,"signed_closed_loop_template")
    schedule=json.loads(a.schedule.read_text());template=json.loads(a.template.read_text());vids=[f"sv{i:02d}" for i in range(20)];groups=[f"sg{i//2}" for i in range(20)];labels=[i%2 for i in range(20)]
    actions=[f"v:{kind}:{k:02d}" for k in range(30) for kind in ("ocr","dense4")];costs={x:1+(int(x[-2:])%3) for x in actions};signals={n:{x:unit(n,x) for x in actions} for n in ("salience","uncertainty","B7","B8","singleton","set_utility","singleton_ridge")};arms=tuple(f"B{i}" for i in range(2,13));pred=[];traces=[]
    for run in schedule["runs"]:
      query_fold=run["outer_fold"]
      for i,v in enumerate(vids):
       if i//4!=query_fold:continue
       for arm in arms:
        if arm=="B2":
          def rex(rng,d):
            q=list(actions);rng.shuffle(q);return {"actions":tuple(q[:4])}
          executions=b2_executions(rex,run["split_seed"],run["refit_id"],v,.5)
        else:executions=[{"draw_id":None,"actions":execute_registered_arm(arm,actions,costs,12,signals,router_positive=i%2==0)}]
        for ex in executions:
          draw=ex["draw_id"];chosen=ex["actions"];dec=[{"step":j,"action_id":q,"status":"ACQUIRE","estimated_cost_ms":costs[q],"realized_cost_ms":costs[q]} for j,q in enumerate(chosen)];th=hashlib.sha256(canonical_bytes(dec)).hexdigest();score=unit(run["run_id"],v,arm,draw);mh=hashlib.sha256(canonical_bytes(["synthetic-model",run["run_id"],arm])).hexdigest()
          pred.append({"schema":"cvoi-prediction/1","run_id":run["run_id"],"dataset":"SYNTHETIC","split_role":"train_oof","video_id":v,"group_id":groups[i],"outer_split_seed":run["split_seed"],"outer_fold":query_fold,"refit_seed":run["refit_id"],"arm_id":arm,"budget_fraction":.5,"score":score,"prediction":int(score>=.5),"threshold":.5,"threshold_source":"synthetic_inner_oof","estimated_budget_ms":12,"realized_cost_ms":sum(costs[q] for q in chosen),"action_trace_sha256":th,"config_id":"synthetic-registered","epoch":3,"model_sha256":mh,"payload_sha256":template["payload_sha256"],"draw_id":draw})
          traces.append({"schema":"cvoi-acquisition-trace/1","run_id":run["run_id"],"outer_split_seed":run["split_seed"],"refit_seed":run["refit_id"],"video_id":v,"arm_id":arm,"budget_fraction":.5,"draw_id":draw,"ordered_actions":chosen,"estimated_cost_ms":12,"realized_cost_ms":sum(costs[q] for q in chosen),"decision_records":dec,"trace_sha256":th})
    validate_full_prediction_rows(pred,5400);validate_trace_join(pred,traces)
    complete=[]
    for ss in (20260811,20260812,20260813):
      for rr in range(3):
       rows=[p for p in pred if p["outer_split_seed"]==ss and p["refit_seed"]==rr and p["draw_id"] is None];by={(p["video_id"],p["arm_id"]):p for p in rows};used=("B10","B3","B4")
       complete.append({"split_seed":ss,"refit":rr,"groups":groups,"profiles":{g:(1,1) for g in set(groups)},"y":labels,
        "scores":{arm:[by[(v,arm)]["score"] for v in vids] for arm in used},"costs":{arm:[by[(v,arm)]["realized_cost_ms"] for v in vids] for arm in used},"thresholds":{arm:[by[(v,arm)]["threshold"] for v in vids] for arm in used}})
    boot=bootstrap_complete_runs(complete,"B10",["B3","B4"]);sens=resample_complete_run_ids([unit("run",i) for i in range(9)]);assert pareto_dominates((.6,1),(.5,2))
    write_jsonl(a.out_dir/"predictions.jsonl",pred);write_jsonl(a.out_dir/"traces.jsonl",traces)
    atomic_json(a.out_dir/"audit.json",{"schema":"cvoi-dry45/1","synthetic_only":True,"candidate_performance_emitted":False,"schedule_sha256":sha256_file(a.schedule),"template_sha256":sha256_file(a.template),"runs":45,"arms":list(arms),"prediction_rows":len(pred),"trace_rows":len(traces),"bootstrap_replicates":len(boot),"run_sensitivity_replicates":len(sens),"bootstrap_values_persisted":False,"pareto_fixture":True,"prediction_trace_join":True,"contact":ledger.snapshot()})
if __name__=="__main__":main()
