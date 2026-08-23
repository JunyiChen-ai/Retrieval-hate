from __future__ import annotations

import argparse,hashlib,io,json
from pathlib import Path

import numpy as np
import torch

from .artifacts import assert_run_contract,expected_train_cardinality,validate_full_prediction_rows,write_jsonl,validate_targets,validate_trace_join
from .bootstrap import bootstrap_complete_runs,interval_gate,resample_complete_run_ids,pareto_dominates
from .common import atomic_json,atomic_write,canonical_bytes,sha256_file
from .costs import benchmark_action,charge_execution,enforce_estimated_budget,summarize_costs
from .models import ActionTokenizer,AdditiveSingletonPolicy,SharedSpecialistPolicy,StateClassifier
from .protocol import b12_applicability,select_threshold,utility
from .selectors import b2_executions,b3_order,b6_route,b12_first,execute_registered_arm
from .state import candidate_targets,dagger_schedule,frozen_state_mixture,utility_target_rows
from .nested import nested_fit_predict,predict_refit


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    run_id=a.out.name
    source_paths=[Path("scripts/cvoi_acq")/x for x in ("synthetic_pipeline.py","models.py","state.py","selectors.py","protocol.py","bootstrap.py","artifacts.py","costs.py","nested.py")]
    source_sha={str(p):sha256_file(p) for p in source_paths};payload_sha=hashlib.sha256(canonical_bytes({"run_root":run_id,"source_sha256":source_sha})).hexdigest()
    for d in ("groups","actions","selection","targets","predictions","traces","metrics","fixtures"):(a.out/d).mkdir(parents=True)
    vids=[f"s{i:02d}" for i in range(10)];labels=[i%2 for i in range(10)];groups=[f"g{i//2}" for i in range(10)]
    actions=[f"v:{kind}:{k:02d}" for k in range(30) for kind in ("ocr","dense4")];costs={x:1+(int(x[-2:])%3) for x in actions}
    order=b3_order(actions);budgeted=enforce_estimated_budget(order,costs,12,max_actions=12)
    mix=[]
    for v in vids:mix.extend(frozen_state_mixture(v,actions,{x:int(x[-2:]) for x in actions},costs,12,on_policy=lambda r:order[r:]))
    candidates=candidate_targets(vids[0],(),order[:4],actions,12,costs)
    fit_trace=[]
    _,dagger=dagger_schedule(lambda rows,e,r:fit_trace.append((len(rows),e,r)) or {"r":r},lambda p,r:[{"r":r}],mix[:2])
    torch.manual_seed(20260817);tokenizer=ActionTokenizer().eval();classifier=StateClassifier(4).eval();z=torch.arange(40,dtype=torch.float32).reshape(10,4)/40
    token=tokenizer(torch.zeros(2,1024),torch.tensor([0,1]),torch.tensor([0,1]),torch.tensor([0,1],dtype=torch.bool),torch.tensor([0,1]))
    logits=classifier(z[:2],token[:,None],torch.zeros(2,1,dtype=torch.bool))
    shared_model=SharedSpecialistPolicy(4);shared=torch.stack([shared_model(torch.zeros(2,4),arm,spec).mean() for arm in shared_model.arms for spec in ("ocr","dense","joint")])
    b9=AdditiveSingletonPolicy(2)(torch.zeros(2,2),token,torch.ones(2),torch.ones(2))
    model_buf=io.BytesIO();torch.save({"tokenizer":tokenizer.state_dict(),"classifier":classifier.state_dict(),"shared":shared_model.state_dict()},model_buf);model_sha=hashlib.sha256(model_buf.getvalue()).hexdigest()
    def action_tokens(chosen):
        if not chosen:return torch.empty((0,256))
        outcome=torch.zeros(len(chosen),1024);kind=[];window=[]
        for j,x in enumerate(chosen):
            k=int(x[-2:]);outcome[j,k]=1;kind.append(0 if ":ocr:" in x else 1);window.append(k)
        return tokenizer(outcome,torch.tensor(kind),torch.tensor(window),torch.zeros(len(chosen),dtype=torch.bool),torch.arange(len(chosen)))
    @torch.inference_mode()
    def state_probability(i,chosen):
        t=action_tokens(chosen);return float(torch.sigmoid(classifier(z[i:i+1],t[None],torch.zeros(1,len(t),dtype=torch.bool))).item())
    arms=tuple(f"B{i}" for i in range(2,13));thresholds={};score_base=np.asarray([state_probability(i,()) for i in range(10)])
    for arm in arms:thresholds[arm]=select_threshold(labels,score_base)
    signals={name:{x:float((int(x[-2:])+j)%31) for x in actions} for j,name in enumerate(("salience","uncertainty","B7","B8","singleton","set_utility","singleton_ridge"))}
    def random_exec(rng,draw_id):
        shuffled=list(actions);rng.shuffle(shuffled);picked=[];spent=0
        for x in shuffled:
            if len(picked)<12 and spent+costs[x]<=12:picked.append(x);spent+=costs[x]
        return {"score":rng.random(),"actions":tuple(picked)}
    def state_features(i,chosen):
        o=sum(":ocr:" in x for x in chosen);d=len(chosen)-o;w=sum(int(x[-2:]) for x in chosen);c=sum(costs[x] for x in chosen)
        return np.r_[z[i].numpy(),[o/12,d/12,w/348,c/12]]
    targets=[]
    for i,v in enumerate(vids):
        states=[r for r in mix if r["video_id"]==v]
        def before_after(state,action,i=i):
            b=state_probability(i,state);a=state_probability(i,tuple(state)+(action,));return b,a,utility(labels[i],b,a)
        targets.extend(utility_target_rows(v,groups[i],[g for g in sorted(set(groups)) if g!=groups[i]],states,actions,costs,12,
          before_after,lambda x:[int(":dense4:" in x),int(x[-2:]),costs[x]],lambda row:order[:4]))
    validate_targets(targets)
    nested_results={};selection=[]
    for ss in (20260811,20260812,20260813):
      for fold in range(5):
       for rr in range(3):
        for arm in arms:
         chosen_by=[]
         for i,v in enumerate(vids):
          chosen=(b2_executions(random_exec,ss,rr,v,.5)[0]["actions"] if arm=="B2" else execute_registered_arm(arm,actions,costs,12,signals,router_positive=(i%2==0)))
          chosen_by.append(chosen)
         X=np.stack([state_features(i,q) for i,q in enumerate(chosen_by)])
         nr=nested_fit_predict(X,labels,groups,[f"g{fold}"],ss,rr);nested_results[(ss,fold,rr,arm)]=nr
         selection.append({"schema":"cvoi-inner-selection/2","procedure_run_id":f"{run_id}-s{ss}-r{rr}","split_seed":ss,"arm_id":arm,
           "outer_fold":fold,"refit_seed":rr,**nr["selection"],"threshold":nr["threshold"],"threshold_source":"pooled_inner_oof",
           "model_sha256":nr["refit_model_sha256"],"payload_sha256":payload_sha})
    preds=[];traces=[]
    for ss in (20260811,20260812,20260813):
      for rr in range(3):
       for i,v in enumerate(vids):
        for arm in arms:
         if arm=="B2": executions=b2_executions(random_exec,ss,rr,v,.5)
         else:executions=[{"draw_id":None,"score":float(score_base[i]),"actions":execute_registered_arm(arm,actions,costs,12,signals,router_positive=(i%2==0))}]
         for execution in executions:
          draw=execution["draw_id"];chosen=execution["actions"];nr=nested_results[(ss,i%5,rr,arm)];score=float(predict_refit(nr["refit_state"],state_features(i,chosen)[None])[0])
          decisions=[{"step":j,"action_id":x,"estimated_cost_ms":costs[x],"realized_cost_ms":costs[x]} for j,x in enumerate(chosen)]
          trace_id=hashlib.sha256(canonical_bytes(decisions)).hexdigest()
          procedure_run_id=f"{run_id}-s{ss}-r{rr}"
          frozen_threshold=nr["threshold"]
          preds.append({"schema":"cvoi-prediction/1","run_id":procedure_run_id,"dataset":"SYNTHETIC","split_role":"train_oof",
            "video_id":v,"group_id":groups[i],"outer_split_seed":ss,"outer_fold":i%5,"refit_seed":rr,"arm_id":arm,
            "budget_fraction":.5,"score":score,"prediction":int(score>=frozen_threshold),"threshold":frozen_threshold,
            "threshold_source":"inner_oof","estimated_budget_ms":12.0,"realized_cost_ms":float(sum(costs[x] for x in chosen)),
            "action_trace_sha256":trace_id,"config_id":nr["selection"]["selected_config"],"epoch":10,
            "model_sha256":nr["refit_model_sha256"],"payload_sha256":payload_sha,"draw_id":draw})
          traces.append({"schema":"cvoi-acquisition-trace/1","run_id":procedure_run_id,"outer_split_seed":ss,"refit_seed":rr,
            "video_id":v,"arm_id":arm,"budget_fraction":.5,"draw_id":draw,"ordered_actions":chosen,
            "estimated_cost_ms":float(sum(costs[x] for x in chosen)),"realized_cost_ms":float(sum(costs[x] for x in chosen)),
            "decision_records":decisions,"trace_sha256":trace_id})
    expected=expected_train_cardinality(10,len(arms),1);validate_full_prediction_rows(preds,expected);validate_trace_join(preds,traces)
    runs=[]
    for ss in (20260811,20260812,20260813):
     for rr in range(3):
      rows=[p for p in preds if p["outer_split_seed"]==ss and p["refit_seed"]==rr and p["draw_id"] is None]
      by={(p["video_id"],p["arm_id"]):p for p in rows};used=("B10","B3","B4")
      runs.append({"split_seed":ss,"refit":rr,"groups":groups,"profiles":{g:(1,1) for g in set(groups)},"y":labels,
       "scores":{a:[by[(v,a)]["score"] for v in vids] for a in used},"costs":{a:[by[(v,a)]["realized_cost_ms"] for v in vids] for a in used},
       "thresholds":{a:[by[(v,a)]["threshold"] for v in vids] for a in used}})
    boot=bootstrap_complete_runs(runs,"B10",["B3","B4"]);gate=interval_gate([x["delta"] for x in boot]);run_points=[float(np.mean(r["scores"]["B10"])) for r in runs];run_sensitivity=resample_complete_run_ids(run_points)
    pareto={"performance":gate,"point_dominates":pareto_dominates((float(np.mean(score_base)),2.0),(float(np.mean(score_base))-.01,3.0)),
            "run_sensitivity_ci":[float(np.quantile(run_sensitivity,.025)),float(np.quantile(run_sensitivity,.975))]}
    action_cost=benchmark_action("synthetic:action",lambda:sum(range(10)));over=[benchmark_action("overhead:"+x,lambda:sum(range(3))) for x in ("policy","encoding","retrieval","solver")]
    charged=charge_execution([action_cost],over,10**9);summary=summarize_costs([action_cost]+over)
    heter=b12_applicability({s:[{"ocr":[1,1,2,3]} for _ in range(5)] for s in (1,2,3)});assert b12_first([1,2],[1,2],2)==1 and b6_route(False,order)==()
    # Complete real artifact layout, all written through collision-refusing atomic helpers.
    atomic_json(a.out/"manifest.json",{"schema":"cvoi-synthetic-manifest/1","synthetic":True,"run_root":run_id,"source_sha256":source_sha,"payload_sha256":payload_sha,"model_sha256":model_sha})
    atomic_json(a.out/"asset_registry.json",{"schema":"cvoi-synthetic-assets/1","assets":[]})
    write_jsonl(a.out/"groups/sources.jsonl",[{"video_id":v,"group_id":g,"label":y} for v,g,y in zip(vids,groups,labels)]);write_jsonl(a.out/"groups/edges.jsonl",[{"a":"s00","b":"s01","rules":["synthetic_duplicate"]}])
    atomic_json(a.out/"groups/components.json",{g:[v for v,q in zip(vids,groups) if q==g] for g in sorted(set(groups))});atomic_json(a.out/"groups/folds.json",{"folds":5})
    write_jsonl(a.out/"actions/ocr_actions.jsonl",[{"action_id":x} for x in actions if ":ocr:" in x]);atomic_write(a.out/"actions/dense_frames.f32",np.zeros((30,4,1024),"<f4").tobytes())
    write_jsonl(a.out/"actions/dense_sidecar.jsonl",[{"action_id":x} for x in actions if ":dense4:" in x]);write_jsonl(a.out/"actions/cost_actions.jsonl",[action_cost]+over);atomic_json(a.out/"actions/cost_summary.json",summary)
    write_jsonl(a.out/"selection/inner_selection.jsonl",selection)
    import pyarrow as pa,pyarrow.parquet as pq
    normalized=[{**r,"state":list(r["state"]),"fit_groups":list(r["fit_groups"]),"generator_provenance":list(r["generator_provenance"]),
                 "classifier_sha256":model_sha,"source_payload_sha256":payload_sha} for r in targets]
    sink=pa.BufferOutputStream();pq.write_table(pa.Table.from_pylist(normalized),sink);atomic_write(a.out/"targets/utility_targets.parquet",sink.getvalue().to_pybytes())
    val_rows=[]
    for rr in range(3):
      for i in range(2):
       for arm in arms:
        val_rows.append({"schema":"cvoi-val-confirmation/1","run_id":f"{run_id}-val-r{rr}","video_id":f"val{i}","arm_id":arm,"refit_seed":rr,
                         "score":state_probability(i,()),"threshold":thresholds[arm],"threshold_source":"train_inner_oof","payload_sha256":payload_sha})
    write_jsonl(a.out/"predictions/train_oof.jsonl",preds);write_jsonl(a.out/"predictions/val_confirmation.jsonl",val_rows);write_jsonl(a.out/"traces/acquisitions.jsonl",traces)
    atomic_json(a.out/"metrics/metrics.json",{"synthetic":True,"gate":gate,"pareto":pareto,"heterogeneity":heter,"nine_run_ids":sorted({p["run_id"] for p in preds})});bio=io.BytesIO();np.savez_compressed(bio,delta=np.asarray([x["delta"] for x in boot]),run_sensitivity=run_sensitivity);atomic_write(a.out/"metrics/bootstrap.npz",bio.getvalue())
    atomic_json(a.out/"fixtures/report.json",{"synthetic":True,"passed":True});write_jsonl(a.out/"resources.jsonl",[{"event":"complete"}]);atomic_write(a.out/"RESULTS.md",b"# Synthetic protocol run\n")
    assert_run_contract(a.out)
    atomic_json(a.out/"synthetic_audit.json",{"schema":"cvoi-synthetic-e2e/1","prediction_rows":len(preds),"expected":expected,
      "states":len(mix),"targets":len(targets),"candidate_limit":len(candidates),"dagger":dagger,"tensor_shapes":[list(token.shape),list(logits.shape),list(shared.shape),list(b9.shape)],
      "bootstrap_replicates":len(boot),"selection_rows":len(selection),"run_ids":len({p["run_id"] for p in preds}),"val_rows":len(val_rows),"cost":charged,"contract":True})

if __name__=="__main__":main()
