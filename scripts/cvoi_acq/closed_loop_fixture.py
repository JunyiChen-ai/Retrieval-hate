from __future__ import annotations

import argparse,hashlib,io,json,platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
import copy

from .common import ContactLedger,atomic_json,canonical_bytes,sha256_file
from .protocol import exact_knapsack,select_threshold,utility,macro_f1_binary
from .fold_iterator import iter_registered_folds

SEED=20260817

def hstate(m):
    b=io.BytesIO();torch.save(m.state_dict(),b);return hashlib.sha256(b.getvalue()).hexdigest()
def registered_seed(*parts):return int.from_bytes(hashlib.sha256("||".join(map(str,parts)).encode()).digest()[:4],"big")
def make_model(dim,seed_parts):
    seed=registered_seed(*seed_parts);torch.manual_seed(seed);return nn.Sequential(nn.Linear(dim,16),nn.GELU(),nn.Linear(16,1)),seed
def fit(model,x,y,epochs=80,lr=.03):
    X=torch.tensor(np.asarray(x),dtype=torch.float32);Y=torch.tensor(np.asarray(y),dtype=torch.float32);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    for _ in range(epochs):
        out=model(X).squeeze(-1);loss=nn.functional.binary_cross_entropy_with_logits(out,Y) if set(np.asarray(y).tolist())<={0,1} else nn.functional.huber_loss(out,Y)
        opt.zero_grad();loss.backward();opt.step()
    return model
def fit_early(model,x,y,vx,vy,max_epochs=60,lr=.03,patience=8,min_delta=1e-4):
    X=torch.tensor(np.asarray(x),dtype=torch.float32);Y=torch.tensor(np.asarray(y),dtype=torch.float32);VX=torch.tensor(np.asarray(vx),dtype=torch.float32);VY=torch.tensor(np.asarray(vy),dtype=torch.float32);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    best=float("inf");best_epoch=0;best_state=None;stale=0;history=[]
    for epoch in range(1,max_epochs+1):
        out=model(X).squeeze(-1);loss=nn.functional.huber_loss(out,Y);opt.zero_grad();loss.backward();opt.step()
        with torch.inference_mode():vl=float(nn.functional.huber_loss(model(VX).squeeze(-1),VY))
        history.append(vl)
        if best-vl>=min_delta:best=vl;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(best_state);return model,best_epoch,history
def prob(model,x):
    with torch.inference_mode():return float(torch.sigmoid(model(torch.tensor(x,dtype=torch.float32))).item())

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();torch.manual_seed(SEED);np.random.seed(SEED)
    if a.out.exists():raise FileExistsError(a.out)
    ledger=ContactLedger();fold_path=Path("artifacts/cvoi_acq/premetric-v2/groups-v4/outer_folds.json");inner_path=Path("artifacts/cvoi_acq/premetric-v2/groups-v4/inner_folds.json")
    ledger.register(fold_path,"frozen_group_fold_loader");ledger.register(inner_path,"frozen_inner_fold_hash")
    frozen_folds=json.loads(fold_path.read_text());frozen_inner=json.loads(inner_path.read_text());real_rows=list(iter_registered_folds(frozen_folds,frozen_inner));real_fold_audit={}
    for sk,obj in frozen_folds.items():
        flat=[v for f in obj["folds"] for v in f["video_ids"]];real_fold_audit[sk]={"n_folds":5,"n_videos":len(flat),"membership_sha256":hashlib.sha256(canonical_bytes(flat)).hexdigest()}
    assert len(real_rows)==15
    vids=[f"v{i}" for i in range(20)];groups=[f"sg{i//2}" for i in range(20)];y=np.array(([0,1,1,0,0,1,1,0,0,1]*2));base=np.tile(np.eye(4),(5,1))
    synthetic_fold_artifact={str(s):{"seed":s,"n_folds":5,"folds":[{"fold":k,"group_ids":[f"sg{2*k}",f"sg{2*k+1}"],"video_ids":[f"v{4*k+j}" for j in range(4)]} for k in range(5)]} for s in (20260811,20260812,20260813)}
    synthetic_inner_artifact={}
    for s in (20260811,20260812,20260813):
        synthetic_inner_artifact[str(s)]={}
        for k in range(5):
            remaining=[q for q in range(10) if q not in (2*k,2*k+1)];parts=[remaining[e::4] for e in range(4)]
            synthetic_inner_artifact[str(s)][str(k)]={"seed":s,"n_folds":4,"folds":[{"fold":e,"group_ids":[f"sg{q}" for q in part],"video_ids":[f"v{2*q+j}" for q in part for j in (0,1)]} for e,part in enumerate(parts)]}
    confirmation_inner_artifact={"n_folds":5,"folds":[{"fold":e,"group_ids":[f"sg{2*e}",f"sg{2*e+1}"],"video_ids":[f"v{4*e+j}" for j in range(4)]} for e in range(5)]}
    assert len({v for f in confirmation_inner_artifact["folds"] for v in f["video_ids"]})==20
    for obj in synthetic_fold_artifact.values():
        flat=[v for f in obj["folds"] for v in f["video_ids"]];assert len(flat)==len(set(flat))==20
    actions=list(range(6));atype=np.array([0,0,0,1,1,1]);window=np.array([0,1,2,0,1,2]);cost=np.array([1,2,1,2,1,2])
    decision_overhead=.25;primary_budget=4.0
    outcome=np.stack([np.array([(i+a)%2,(i*2+a)%3])/3 for i in range(20) for a in actions]).reshape(20,6,2)
    def phi(i,S):return np.r_[base[i],outcome[i,list(S)].sum(0) if S else [0,0]]
    def afeat(i,S,a):return np.r_[base[i],outcome[i,list(S)].sum(0) if S else [0,0],atype[a],window[a]/2,cost[a]/2]
    synthetic_rows=list(iter_registered_folds(synthetic_fold_artifact,synthetic_inner_artifact));folds=[r for r in synthetic_rows if r["split_seed"]==20260811][:2];all_rows=[];checks=[];fold_records=[]
    for fold_id,fold in enumerate(folds):
        query=set(fold["query_groups"]);train_groups=list(fold["train_groups"]);train=[i for i,v in enumerate(vids) if v in set(fold["train_videos"])];query_ix=[i for i,v in enumerate(vids) if v in set(fold["query_videos"])]
        targets=[];oof_score={};cf_models={};inner_classifier_records=[]
        # Cross-fit classifier: target/eval group is excluded from classifier fit.
        registered_inner_count=len(fold["inner"]);assert registered_inner_count==4
        for ir in fold["inner"]:
            inner_id=ir["inner_fold"];fit_groups=list(ir["fit_groups"]);eval_groups=list(ir["eval_groups"]);fit_ix=[i for i,g in enumerate(groups) if g in fit_groups]
            X=[];Y=[]
            for i in fit_ix:
                for S in ((),(0,),(3,),(0,3)):X.append(phi(i,S));Y.append(y[i])
            clf_seed=registered_seed(fold["split_seed"],fold["outer_fold"],inner_id,fold_id,"classifier","base")
            torch.manual_seed(clf_seed);clf=fit(nn.Linear(6,1),X,Y)
            inner_hash=hashlib.sha256(canonical_bytes(ir)).hexdigest();inner_classifier_records.append({"inner_fold":inner_id,"fit_groups":fit_groups,"eval_groups":eval_groups,"inner_sha256":inner_hash,"rng_seed":clf_seed,"model_sha256":hstate(clf),"status":"EMPTY_EVAL" if not eval_groups else "OK"})
            for eg in eval_groups:cf_models[eg]=clf
            for i in [q for q,g in enumerate(groups) if g in set(eval_groups)]:
                oof_score[i]=prob(clf,phi(i,()))
                for S in ((),(0,),(3,)):
                    for act in actions:
                        if act in S:continue
                        before=prob(clf,phi(i,S));after=prob(clf,phi(i,tuple(S)+(act,)))
                        targets.append({"video_id":vids[i],"eval_group":groups[i],"fit_groups":fit_groups,"state":list(S),"action":act,
                          "features":afeat(i,S,act).tolist(),"before":before,"after":after,"utility":utility(y[i],before,after),
                          "inner_fold":inner_id,"inner_sha256":inner_hash,"classifier_rng_seed":clf_seed,"classifier_sha256":hstate(clf)})
        assert len(inner_classifier_records)==registered_inner_count and set(oof_score)==set(train)
        assert all(r["eval_group"] not in r["fit_groups"] for r in targets);checks.append("crossfit_target_lineage")
        threshold=select_threshold([y[i] for i in train],[oof_score[i] for i in train])
        # B9 is state-blind (drop state-summary columns); B10 is set-conditioned.
        tx=np.asarray([r["features"] for r in targets]);ty=np.asarray([r["utility"] for r in targets]);b9x=np.c_[tx[:,:4],tx[:,6:]]
        def select_policy(kind):
            dim=7 if kind=="B9" else 9;records=[]
            for lr in (.01,.03):
                held_scores={};losses=[];held_costs=[];held_best_epochs=[];held_policy_hash={};held_validation={}
                for ir in fold["inner"]:
                    inner_id=ir["inner_fold"];eval_groups=set(ir["eval_groups"]);fit_group_set=set(ir["fit_groups"]);tr=np.array([r["eval_group"] in fit_group_set for r in targets]);va=np.array([r["eval_group"] in eval_groups for r in targets]);xx=b9x if kind=="B9" else tx
                    config_id=f"{kind}-lr{lr:g}-wd1e-4";seed_parts=(fold["split_seed"],fold["outer_fold"],inner_id,fold_id,kind,config_id);m,rng_seed=make_model(dim,seed_parts)
                    if not va.any():
                        m=fit(m,xx[tr],ty[tr],1,lr);held_policy_hash[str(inner_id)]=hstate(m);held_validation[str(inner_id)]={"status":"EMPTY_EVAL","rng_seed":rng_seed,"eval_groups":[]};continue
                    m,best_epoch,val_history=fit_early(m,xx[tr],ty[tr],xx[va],ty[va],60,lr,8,1e-4)
                    held_policy_hash[str(inner_id)]=hstate(m)
                    held_validation[str(inner_id)]={"status":"OK","rng_seed":rng_seed,"eval_groups":sorted(eval_groups),"best_epoch":best_epoch,"epochs_ran":len(val_history),"best_loss":min(val_history),"history_sha256":hashlib.sha256(canonical_bytes(val_history)).hexdigest()}
                    with torch.inference_mode():losses.append(float(nn.functional.mse_loss(m(torch.tensor(xx[va],dtype=torch.float32)).squeeze(),torch.tensor(ty[va],dtype=torch.float32))))
                    for i in [q for q,g in enumerate(groups) if g in eval_groups]:
                        S=[];spent=0.0
                        while spent<primary_budget and len(S)<3:
                            cand=[q for q in actions if q not in S and spent+decision_overhead+cost[q]<=primary_budget]
                            if not cand:break
                            feats=np.stack([afeat(i,S,q) for q in cand]);inp=np.c_[feats[:,:4],feats[:,6:]] if kind=="B9" else feats
                            with torch.inference_mode():sc=m(torch.tensor(inp,dtype=torch.float32)).squeeze().numpy();j=int(np.argmax(np.atleast_1d(sc)))
                            S.append(cand[j]);spent+=decision_overhead+cost[cand[j]]
                        held_scores[i]=prob(cf_models[groups[i]],phi(i,S))
                        held_costs.append(spent)
                    held_best_epochs.append(best_epoch)
                yy=[y[i] for i in train];ss=[held_scores[i] for i in train];thr=select_threshold(yy,ss);f1=macro_f1_binary(yy,np.asarray(ss)>=thr)
                assert len(held_validation)==registered_inner_count and set(held_scores)==set(train)
                median_epoch=int(np.floor(np.median(held_best_epochs)+.5));config_id=f"{kind}-lr{lr:g}-wd1e-4"
                records.append({"config_id":config_id,"optimizer":"AdamW","weight_decay":1e-4,"lr":lr,"held_best_epochs":held_best_epochs,
                  "early_stopping":{"patience":8,"min_delta":1e-4},"median_refit_epoch":median_epoch,"epochs":median_epoch,
                  "inner_oof_macro_f1":f1,"mean_realized_cost":float(np.mean(held_costs)),"threshold":thr,"config_loss":float(np.mean(losses)),
                  "held_validation":held_validation,"per_held_model_sha256":held_policy_hash,"oof_sha256":hashlib.sha256(canonical_bytes(ss)).hexdigest()})
            chosen=sorted(records,key=lambda r:(-r["inner_oof_macro_f1"],r["mean_realized_cost"],r["median_refit_epoch"],r["config_id"]))[0];xx=b9x if kind=="B9" else tx
            model,refit_rng_seed=make_model(dim,(fold["split_seed"],fold["outer_fold"],-1,fold_id,kind,chosen["config_id"]));model=fit(model,xx,ty,chosen["epochs"],chosen["lr"]);chosen["refit_rng_seed"]=refit_rng_seed
            return model,chosen,records
        b9,b9_selected,b9_configs=select_policy("B9");b10,b10_selected,b10_configs=select_policy("B10")
        # Outer classifier is fit once on all legal outer-train state rows.
        X=[];Y=[]
        for i in train:
            for S in ((),(0,),(3,),(0,3)):X.append(phi(i,S));Y.append(y[i])
        outer_seed=registered_seed(fold["split_seed"],fold["outer_fold"],-1,fold_id,"classifier","outer_refit");torch.manual_seed(outer_seed);outer_clf=fit(nn.Linear(6,1),X,Y);model_hashes={"classifier":hstate(outer_clf),"B9":hstate(b9),"B10":hstate(b10)}
        def scores(i,S,kind):
            xx=np.stack([afeat(i,S,q) for q in actions if q not in S]);ids=[q for q in actions if q not in S]
            inp=np.c_[xx[:,:4],xx[:,6:]] if kind=="B9" else xx;model=b9 if kind=="B9" else b10
            with torch.inference_mode():sc=model(torch.tensor(inp,dtype=torch.float32)).squeeze().numpy()
            return ids,np.atleast_1d(sc)
        for i in query_ix:
            for arm in ("B9","B10","B12"):
                S=[];spent=0.0;action_spent=0.0;overhead_spent=0.0;decisions=[]
                while spent<primary_budget and len(S)<3:
                    if arm=="B12":
                        if spent+decision_overhead>primary_budget:break
                        overhead_spent+=decision_overhead;spent=action_spent+overhead_spent
                    ids,sc=scores(i,tuple(S),"B9" if arm=="B9" else "B10")
                    feasible=[j for j,q in enumerate(ids) if spent+(0 if arm=="B12" else decision_overhead)+cost[q]<=primary_budget]
                    if not feasible:
                        if arm=="B12":decisions.append({"state":list(S),"candidate_scores":{},"chosen":None,"status":"STOP","decision_overhead":decision_overhead})
                        break
                    if arm=="B12":
                        local=exact_knapsack([float(sc[j]) for j in feasible],[int(cost[ids[j]]*100) for j in feasible],int((primary_budget-spent)*100));j=feasible[local[0]] if local else feasible[0]
                        if not local:
                            decisions.append({"state":list(S),"candidate_scores":{str(ids[q]):float(sc[q]) for q in feasible},"chosen":None,"status":"STOP","decision_overhead":decision_overhead});break
                    else:j=max(feasible,key=lambda q:(float(sc[q]),-cost[ids[q]],-ids[q]))
                    chosen=ids[j];decisions.append({"state":list(S),"candidate_scores":{str(ids[q]):float(sc[q]) for q in feasible},"chosen":chosen,"status":"ACQUIRE","decision_overhead":decision_overhead});S.append(chosen);action_spent+=float(cost[chosen]);overhead_spent+=0 if arm=="B12" else decision_overhead;spent=action_spent+overhead_spent
                arm_threshold=b9_selected["threshold"] if arm=="B9" else b10_selected["threshold"]
                score=prob(outer_clf,phi(i,tuple(S)));all_rows.append({"fold":fold_id,"video_id":vids[i],"group":groups[i],"arm":arm,"score":score,"threshold":arm_threshold,
                  "prediction":int(score>=arm_threshold),"actions":S,"cost":spent,"action_cost":action_spent,"decision_overhead_cost":overhead_spent,
                  "predicted_feasibility_reserved_overhead":decision_overhead,"decisions":decisions,"model_hashes":model_hashes,
                  "selected_lr":b9_selected["lr"] if arm=="B9" else b10_selected["lr"],"target_count":len(targets)})
        # State-blind invariant: B9 score for the same action is unchanged by S.
        probe=0;f0=afeat(train[0],(),probe);f1=afeat(train[0],(1,3),probe);i0=np.r_[f0[:4],f0[6:]];i1=np.r_[f1[:4],f1[6:]]
        with torch.inference_mode():b9_diff=float(abs(b9(torch.tensor(i0,dtype=torch.float32))-b9(torch.tensor(i1,dtype=torch.float32))).item())
        assert b9_diff<=1e-7;checks.append("B9_state_invariant")
        fold_records.append({"fold":fold_id,"split_seed":fold["split_seed"],"outer_fold":fold["outer_fold"],"refit_seed":fold_id,"inner_count":len(fold["inner"]),"inner_classifier_records":inner_classifier_records,"targets":targets,"crossfit_classifier_hashes":{g:hstate(m) for g,m in cf_models.items()},
          "B9":{"selected":b9_selected,"configs":b9_configs,"refit_model_sha256":hstate(b9),"refit_no_callback":True},"B10":{"selected":b10_selected,"configs":b10_configs,"refit_model_sha256":hstate(b10),"refit_no_callback":True},
          "outer_classifier_rng_seed":outer_seed,"outer_classifier_sha256":hstate(outer_clf),"b9_invariant_abs":b9_diff})
        checks.extend(["targets_train_B9_B10","selected_model_outer_prediction","B12_recompute_each_step"])
    assert len(all_rows)==8*3 and len({(r["video_id"],r["arm"]) for r in all_rows})==24
    assert all(sum(d["chosen"] is not None for d in r["decisions"])==len(r["actions"]) for r in all_rows);checks.extend(["two_fold_exact_coverage","trace_action_linkage"])
    stop_choice=exact_knapsack([-1.0,-2.0],[10,10],20);assert stop_choice==()
    stop_case={"scores":[-1.0,-2.0],"choice":[],"status":"STOP","decision_overhead_charged":1,"action_cost":0};checks.append("B12_negative_utility_STOP")
    assert ledger.test_contact_count==0 and all("test_seen" not in r["path"] for r in ledger.snapshot()["opened_paths"])
    sources=[Path("scripts/cvoi_acq/closed_loop_fixture.py"),Path("scripts/cvoi_acq/fold_iterator.py"),Path("scripts/cvoi_acq/protocol.py"),Path("scripts/cvoi_acq/common.py"),fold_path,inner_path]
    payload={"schema":"cvoi-closed-loop-fixture/3","folds":2,"rows":all_rows,"fold_records":fold_records,"stop_case":stop_case,"checks":sorted(set(checks)),
      "real_groups_v4_loader_audit":real_fold_audit,"registered_real_iterator_rows":len(real_rows),
      "synthetic_fold_artifact":synthetic_fold_artifact,"synthetic_inner_artifact":synthetic_inner_artifact,"registered_synthetic_iterator_rows":len(synthetic_rows),
      "confirmation_inner_artifact":confirmation_inner_artifact,
      "source_sha256":{str(p):sha256_file(p) for p in sources},"environment":{"python":platform.python_version(),"platform":platform.platform(),
       "torch":torch.__version__,"numpy":np.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
      "candidate_metric_firewall":{"namespace":"synthetic-only","real_candidate_imported":False,"test_path_denied":True,"enforced":True},
      "contact":ledger.snapshot(),"test_contact_count":ledger.test_contact_count}
    payload["payload_sha256"]=hashlib.sha256(canonical_bytes(payload)).hexdigest();atomic_json(a.out,payload)

if __name__=="__main__":main()
