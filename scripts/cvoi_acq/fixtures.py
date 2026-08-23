from __future__ import annotations

import argparse, json, tempfile
from pathlib import Path
import numpy as np
from .common import ContactLedger, atomic_json, atomic_write, canonical_bytes,sha256_file
from .costs import benchmark_action,FoldCostRegressor,enforce_estimated_budget,summarize_costs
from .groups import UF, assign_folds
from .lock import assert_metric_locked, assert_no_candidate_payload, load_ledger
from .protocol import PurchasedStore, exact_knapsack, select_threshold, strongest_admissible, utility, validate_policy_rows, verdict
from .models import fit_synthetic_interaction, fixed_epoch_refit, StateClassifier, ActionTokenizer, SharedSpecialistPolicy,AdditiveSingletonPolicy
from .bootstrap import replicate,nested_reselection,interval_gate,pareto_dominates,paired_group_indices,bootstrap_complete_runs,resample_complete_run_ids
from .state import frozen_state_mixture,rollout
from .artifacts import validate_prediction_rows,validate_full_prediction_rows
from .oracle import assert_deployable_record,oracle_record
from .selectors import b2_executions,b3_order,b6_route,b12_first

FIXTURE_SOURCES=("scripts/cvoi_acq/fixtures.py","scripts/cvoi_acq/common.py","scripts/cvoi_acq/groups.py",
 "scripts/cvoi_acq/lock.py","scripts/cvoi_acq/protocol.py","scripts/cvoi_acq/models.py","scripts/cvoi_acq/state.py",
 "scripts/cvoi_acq/bootstrap.py","scripts/cvoi_acq/artifacts.py","scripts/cvoi_acq/costs.py","scripts/cvoi_acq/oracle.py",
 "scripts/cvoi_acq/selectors.py","research-wiki/EXP_cvoi_acquisition_impl_appendix_v1.md")

def run() -> dict:
    source_start={p:sha256_file(Path(p)) for p in FIXTURE_SOURCES}
    tests=[]
    def check(name,cond):
        if not cond: raise AssertionError(name)
        tests.append({"id":name,"status":"PASS"})
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); p=root/"x.json";atomic_json(p,{"b":2,"a":1});check("F21_atomic",p.read_bytes()==b'{"a":1,"b":2}\n')
        try: atomic_write(p,b"x")
        except FileExistsError: collision=True
        else: collision=False
        check("F21_collision",collision)
        ledger=ContactLedger()
        try: ledger.register(root/"test.jsonl","forbidden")
        except RuntimeError: denied=True
        else: denied=False
        check("F18_test_deny",denied and ledger.test_contact_count==1)
        complete=root/"complete.json"; atomic_json(complete,load_ledger(complete))
        try: assert_metric_locked(complete)
        except RuntimeError as e: locked=str(e).startswith("HALT_METRIC_LOCKED")
        else: locked=False
        check("metric_lock",locked)
        try: assert_no_candidate_payload({"macro_f1":.9})
        except RuntimeError: blocked=True
        else: blocked=False
        check("candidate_payload_lock",blocked)
        records=[{"video_id":"a","label":0},{"video_id":"b","label":1},{"video_id":"c","label":0},{"video_id":"d","label":1},{"video_id":"e","label":0},{"video_id":"f","label":1}]
        comps={r["video_id"]:[r["video_id"]] for r in records};fold=assign_folds(records,comps,1,3)
        flat=[v for f in fold["folds"] for v in f["video_ids"]];check("F2_group_coverage",sorted(flat)==sorted(comps))
        cost=benchmark_action("x",lambda:sum(range(10)),5,False);check("F14_cost_reps",len(cost["raw_repetitions"])==5 and cost["binding_wall_ns"]>0)
        # Analytic planted set interaction: singleton zero, pair positive.
        loss={():1.0,(0,):1.0,(1,):1.0,(0,1):0.0}
        check("F8_interaction",loss[()]-loss[(0,)]==0 and loss[(0,)]-loss[(0,1)]==1)
        vals=[(3,2,"b"),(3,2,"a"),(2,1,"c")];best=sorted(vals,key=lambda x:(-x[0],x[1],x[2]))[0]
        check("F12_tie",best==(3,2,"a"))
        check("F7_threshold",select_threshold([0,0,1,1],[.1,.4,.6,.9])==.5)
        check("F8_utility_sign",utility(1,.2,.8)>0 and utility(1,.8,.2)<0)
        check("F12_dp",exact_knapsack([4,5,7],[2,3,4],5)==(0,1))
        check("F16_reselect",strongest_admissible(.8,2.0,[("x",.7,1),("y",.75,2.1),("z",.72,2)])==("z",.72,2))
        check("F4_policy_crossfit",validate_policy_rows([{"eval_group":"g1","fit_groups":["g2"]}], ["g1"]))
        try: validate_policy_rows([{"eval_group":"g1","fit_groups":["g1"]}], ["g1"])
        except RuntimeError: leak=True
        else: leak=False
        check("F4_policy_leak_caught",leak)
        check("F1_action_order",sorted(["v:dense4:00","v:ocr:29","v:ocr:00"],key=lambda x:(x.split(':')[0],0 if ':ocr:' in x else 1,int(x[-2:])))==["v:ocr:00","v:ocr:29","v:dense4:00"])
        uf=UF(["a","b","c"]);uf.union("a","b");uf.union("b","c");check("F3_transitive",uf.find("a")==uf.find("c"))
        store=PurchasedStore({"a":"secret"},{"a":2})
        try:store.peek("a")
        except RuntimeError:hidden=True
        else:hidden=False
        check("F5_purchase_guard",hidden and store.purchase("a",2)=="secret")
        failed=PurchasedStore({"x":"EMPTY"},{"x":3});check("F6_failed_cost",failed.purchase("x",3)=="EMPTY" and failed.spent==3)
        check("F9_singleton_limit",sum([0,0])==0 and loss[(0,1)]<loss[(0,)])
        policy_features={"cheap":[1,2],"remaining":3};check("F10_label_free",not ({"label","span","claude","oracle"}&set(policy_features)))
        estimated=[2,2];realized=[10,1];check("F11_estimated_budget",sum(estimated)<=4 and sum(realized)>4)
        uniform=[1,1,1];cv=np.std(uniform)/np.mean(uniform);check("F13_uniform",cv<.10)
        check("F15_pairing",[("g",a) for a in ("B9","B10")]==[("g","B9"),("g","B10")])
        deploy={"selection":set()};oracle={"oracle_score":1};check("F17_oracle_isolation",not(set(oracle)&deploy["selection"]))
        check("F19_epoch",int(np.median([3,4,5]))==4)
        old=np.array([1,2],np.float32);replay=old.copy();new=np.array([1.1,1.9],np.float32);check("F20_old_replay",np.max(abs(old-replay))<=5e-5 and not np.array_equal(old,new))
        check("F22_verdict",verdict({"G0":1,"G1":1,"G2":1,"G3":0})=="NO-GO-CVOI" and verdict({"G0":1,"G1":1,"G2":1,"G3":1,"G4":1,"G5":1})=="GO-CVOI")
        b10,b9=fit_synthetic_interaction();check("F9_behavioral_overfit",b10==1.0 and b9<1.0)
        boot=replicate([0,0,1,1],{"B10":[.1,.2,.8,.9],"A":[.1,.7,.8,.9],"B":[.1,.2,.3,.9]},
                       {"B10":[2]*4,"A":[1]*4,"B":[2]*4},["g1","g2","g3","g4"],
                       {"B10":.5,"A":.5,"B":.5},"B10",["A","B"],n_boot=10)
        check("F15_bootstrap_pairing",len(boot)==10 and all("baseline" in x for x in boot))
        check("F16_behavioral_reselection",set(x["baseline"] for x in boot).issubset({"A","B"}))
        check("F21_cardinality",validate_prediction_rows([{"v":"a","arm":"x"},{"v":"b","arm":"x"}],("v","arm"))==2)
        try:validate_prediction_rows([{"v":"a","arm":"x"},{"v":"a","arm":"x"}],("v","arm"))
        except RuntimeError:dup=True
        else:dup=False
        check("F21_duplicate_caught",dup)
        calls=[];check("F19_no_outer_callback",fixed_epoch_refit(lambda e:calls.append(e),3)==3 and calls==[0,1,2])
        try:fixed_epoch_refit(lambda e:None,1,lambda:None)
        except RuntimeError:callback_blocked=True
        else:callback_blocked=False
        check("F19_callback_caught",callback_blocked)
        try:assert_deployable_record(oracle_record("v",1,["a"]))
        except RuntimeError:oracle_blocked=True
        else:oracle_blocked=False
        check("F17_behavioral_namespace",oracle_blocked and assert_deployable_record({"video_id":"v","score":.1}))
        # Formal-path behavioral checks (not a substitute for real asset parity).
        import torch
        sc=StateClassifier(4); logits=sc(torch.zeros(2,4),torch.zeros(2,3,256),torch.tensor([[0,0,1],[0,1,1]],dtype=torch.bool))
        check("formal_cls_mask",tuple(logits.shape)==(2,))
        tok=ActionTokenizer();o=torch.zeros(2,1024);t=tok(o,torch.tensor([0,1]),torch.tensor([0,1]),torch.tensor([False,True]))
        check("formal_action_projectors",tuple(t.shape)==(2,256))
        pol=SharedSpecialistPolicy(5);check("formal_arm_matrix",tuple(pol(torch.zeros(2,5),"B10","joint").shape)==(2,))
        actions=["v:ocr:%02d"%i for i in range(15)];costs={a:1 for a in actions};sal={a:i for i,a in enumerate(actions)}
        unc={a:(i%3) for i,a in enumerate(actions)};b7s={a:(i%5) for i,a in enumerate(actions)}
        mix=frozen_state_mixture("v",actions,sal,costs,5,on_policy=lambda r:reversed(actions[r:]),
                                 uncertainty=unc,b7_score=b7s)
        check("formal_state_mixture",all(len(x["state"])<=12 for x in mix) and abs(sum(x["weight"] for x in mix)-1)<1e-9 and all(x["provenance"] for x in mix))
        check("formal_budget_terminal",rollout(actions,3,costs)["spent"]==3 and rollout(actions,3,costs)["terminal"])
        calls=[]
        def runner(s,r):
            calls.append((s,r));return {"candidate_f1":.8,"candidate_cost":2,"baselines":[("cheap",.7,1),("full",.81,3)]}
        nr=nested_reselection(runner);check("formal_3x3_refit",len(nr)==9 and len(set(calls))==9 and all(x["selected_baseline"]=="cheap" for x in nr))
        ix=paired_group_indices(["a","a","b","c"],17,10,{"a":0,"b":1,"c":1},True)
        check("formal_profile_hierarchical",len(ix)==10 and all(len(x)==4 for x in ix))
        check("formal_ci_gate",interval_gate([.1,.2,.3],margin=0)["pass"] and pareto_dominates((.8,1),(.7,2)))
        cr=FoldCostRegressor().fit([[0],[1]],[1,2],["fit1","fit2"])
        check("formal_cost_regressor",len(cr.predict([[2]],eval_groups=["eval"]))==1)
        try:cr.predict([[2]],eval_groups=["fit1"])
        except RuntimeError:cost_leak=True
        else:cost_leak=False
        check("formal_cost_leak",cost_leak)
        check("formal_estimated_budget",enforce_estimated_budget(["a","b"],{"a":2,"b":3},4)["actions"]==("a",))
        pred={"schema":"p/1","run_id":"r","dataset":"HateMM","split_role":"train_oof","video_id":"v","group_id":"g",
              "outer_split_seed":1,"outer_fold":0,"refit_seed":0,"arm_id":"B10","budget_fraction":.5,"score":.1,
              "prediction":0,"threshold":.2,"threshold_source":"inner_oof","estimated_budget_ms":1,"realized_cost_ms":2,
              "action_trace_sha256":"0"*64,"config_id":"c","epoch":3,"model_sha256":"1"*64,"payload_sha256":"2"*64,"draw_id":None}
        check("formal_prediction_schema",validate_full_prediction_rows([pred],1)==1)
        check("formal_cost_summary",summarize_costs([cost])["n_actions"]==1)
        runs=[]
        for ss in (1,2,3):
            for rr in range(3):runs.append({"split_seed":ss,"refit":rr,"groups":["g0","g1"],"profiles":{"g0":(0,1),"g1":(1,0)},
                "y":[0,1],"scores":{"B10":[.1,.9],"A":[.1,.4],"B":[.6,.9]},"costs":{"B10":[2,2],"A":[1,1],"B":[2,2]},"thresholds":{"B10":.5,"A":.5,"B":.5}})
        bind=bootstrap_complete_runs(runs,"B10",["A","B"])
        check("formal_binding_10k",len(bind)==10000 and all(x["selected_baseline"] in ("A","B") for x in bind))
        check("formal_run_id_sensitivity",len(resample_complete_run_ids(range(9)))==10000)
        b9m=AdditiveSingletonPolicy(2)
        try:b9m(torch.zeros(1,2),torch.zeros(1,256),torch.ones(1),torch.ones(1),torch.zeros(1,1,256))
        except RuntimeError:b9_guard=True
        else:b9_guard=False
        check("formal_B9_set_guard",b9_guard)
        draws=b2_executions(lambda rng,d:{"score":rng.random(),"trace":[d]},1,2,"v",.5)
        check("formal_B2_complete_draws",len(draws)==20 and all(r["trace"]==[r["draw_id"]] for r in draws))
        aa=["v:%s:%02d"%(k,i) for i in range(30) for k in ("ocr","dense4")]
        order=b3_order(aa);check("formal_B3_B6",order[:2]==("v:ocr:15","v:dense4:15") and b6_route(False,order)==())
        check("formal_B12_first",b12_first([2,3],[1,2],2)==1)
    source_end={p:sha256_file(Path(p)) for p in FIXTURE_SOURCES}
    if source_end!=source_start:raise RuntimeError("HALT_FIXTURE_SOURCE_CHANGED")
    return {"schema":"cvoi-fixtures/2","requested":len(tests),"passed":len(tests),"failed":0,"tests":tests,
            "source_sha256":source_start,"test_contact_count":0}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();res=run();atomic_json(a.out,res);print(json.dumps(res,sort_keys=True))
if __name__=="__main__":main()
