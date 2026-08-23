"""Fresh five-fold, all-development confirmation refit for the synthetic audit."""
from __future__ import annotations
import hashlib
import numpy as np
import torch

from .common import canonical_bytes
from .closed_loop_fixture import fit, prob, registered_seed
from .nested import predict_refit
from .protocol import macro_f1_binary, select_threshold, utility
from .selector_trainer import fit_router, fit_selector, nested_selector_fit, predict_selector
from .selectors import execute_random_arm, execute_stateful_arm


def run(template, split_seed, refit, arms, vids, groups, labels, base, outcome,
        action_ids, action_cost, phi, feature_fn, test_contact_count):
    fresh=template["confirmation_inner_artifact"]
    if fresh.get("n_folds")!=5 or len(fresh.get("folds",[]))!=5:raise RuntimeError("HALT_FRESH5_FOLD_CONTRACT")
    all_groups=set(groups);folds=[]
    for q in fresh["folds"]:
        held=set(q["group_ids"]);folds.append({"inner_fold":int(q["fold"]),"eval_groups":sorted(held),"fit_groups":sorted(all_groups-held)})
    flat=[g for q in folds for g in q["eval_groups"]]
    if len(flat)!=len(set(flat)) or set(flat)!=all_groups:raise RuntimeError("HALT_FRESH5_GROUP_COVERAGE")

    targets=[];classifier_records=[];held_classifiers={}
    for q in folds:
        fit_groups=set(q["fit_groups"]);ix=[i for i,g in enumerate(groups) if g in fit_groups];x=[];y=[]
        for i in ix:
            for state in ((),(0,),(3,),(0,3)):x.append(phi(i,state));y.append(labels[i])
        seed=registered_seed(split_seed,refit,q["inner_fold"],"fresh5-classifier");torch.manual_seed(seed)
        model=fit(torch.nn.Linear(6,1),x,y);held_classifiers[q["inner_fold"]]=model
        mh=hashlib.sha256(canonical_bytes({k:v.detach().tolist() for k,v in model.state_dict().items()})).hexdigest()
        classifier_records.append({**q,"rng_seed":seed,"model_sha256":mh})
        for i,g in enumerate(groups):
            if g not in set(q["eval_groups"]):continue
            local=[]
            for state in ((),(0,),(3,)):
                for j,a in enumerate(action_ids):
                    if j in state:continue
                    before=prob(model,phi(i,state));after=prob(model,phi(i,tuple(state)+(j,)))
                    local.append({"video_id":vids[i],"eval_group":g,"fit_groups":q["fit_groups"],"state":list(state),"action_id":a,
                      "action_features":np.r_[base[i],j/5,len(state)/3].tolist(),"cheap_features":base[i].tolist(),
                      "state_features":(outcome[i,list(state)].sum(0) if state else np.zeros(2)).tolist(),
                      "before_probability":before,"after_probability":after,"utility":utility(labels[i],before,after),
                      "estimated_cost":action_cost[a],
                      "remaining_budget":4.-sum(action_cost[action_ids[z]] for z in state)-.25*(len(state)+1),
                      "generator_provenance":[mh,"fresh5",str(q["inner_fold"])],"terminal":False})
            for r in local:r["weight"]=1/len(local)
            targets.extend(local)
    if len({r["video_id"] for r in targets})!=20:raise RuntimeError("HALT_FRESH5_DEV_CARDINALITY")

    learned={a:nested_selector_fit(targets,folds,a,split_seed,-1,refit,expected_folds=5)
             for a in ("B7","B8","B9","B10","B11","B12")}
    router=fit_router(base,labels,(split_seed,-1,refit,"fresh5"));rw=np.asarray(router["state"]["weight"])[0];rb=float(router["state"]["bias"][0])

    def signals(b,o,i,models,state=()):
        remaining=4-sum(action_cost[action_ids[z]] for z in state)-.25*(len(state)+1)
        if remaining < -1e-9:raise RuntimeError("HALT_REMAINING_BUDGET_SEMANTICS")
        rows=[{"cheap_features":b[i].tolist(),"state_features":(o[i,list(state)].sum(0) if state else np.zeros(2)).tolist(),
          "action_features":np.r_[b[i],j/5,len(state)/3].tolist(),"remaining_budget":remaining,"estimated_cost":action_cost[a]}
          for j,a in enumerate(action_ids)]
        z={"salience":{a:float(o[i,j].sum()) for j,a in enumerate(action_ids)},"uncertainty":{a:float(1-abs(o[i,j].mean()-.5)) for j,a in enumerate(action_ids)},"remaining_budget":remaining}
        for a,key in (("B7","B7"),("B8","B8"),("B9","singleton"),("B10","set_utility"),("B11","singleton_ridge")):
            sc=predict_selector(models[a],rows,a);z[key]={q:float(sc[j]) for j,q in enumerate(action_ids)}
        b12=predict_selector(models["B12"],rows,"B12");z["B12_set_utility"]={q:float(b12[j]) for j,q in enumerate(action_ids)}
        return z

    def choose(b,o,i,arm,models,route=True,draw_id=0,video_id="dev"):
        if arm=="B2":
            rng=np.random.default_rng(registered_seed(split_seed,refit,video_id,draw_id,"fresh5-B2"))
            return execute_random_arm(action_ids,action_cost,4,rng,decision_overhead=.25)
        def callback(state):
            z=signals(b,o,i,models,tuple(action_ids.index(a) for a in state))
            if arm=="B12":z["set_utility"]=z["B12_set_utility"]
            return z
        return execute_stateful_arm(arm,action_ids,action_cost,4,callback,
            router_positive=route,decision_overhead=.25)

    # Each held fold gets selectors fit without its target groups; these scores alone select thresholds.
    oof={a:np.full(20,np.nan) for a in arms}
    for q in folds:
        held=set(q["eval_groups"]);tr=[r for r in targets if r["eval_group"] in set(q["fit_groups"])]
        fold_models={a:fit_selector(tr,a,(split_seed,-1,refit,q["inner_fold"],"fresh5-held"),
                     learned[a]["selected"]["epochs"],learned[a]["selected"]["lr"])
                     for a in learned}
        ridx=[i for i,g in enumerate(groups) if g in set(q["fit_groups"])]
        held_router=fit_router(base[ridx],labels[ridx],(split_seed,-1,refit,q["inner_fold"],"fresh5-held-router"))
        hrw=np.asarray(held_router["state"]["weight"])[0];hrb=float(held_router["state"]["bias"][0])
        clf=held_classifiers[q["inner_fold"]]
        for i,g in enumerate(groups):
            if g not in held:continue
            for arm in arms:
                chosen=choose(base,outcome,i,arm,fold_models,route=bool(1/(1+np.exp(-(base[i]@hrw+hrb)))>=.5))["actions"]
                oof[arm][i]=prob(clf,phi(i,[action_ids.index(a) for a in chosen]))
    if any(not np.isfinite(x).all() for x in oof.values()):raise RuntimeError("HALT_FRESH5_OOF_COVERAGE")
    thresholds={a:select_threshold(labels,oof[a]) for a in arms}

    # All-development refits: state classifier, selectors and B6 router.
    sx=[];sy=[]
    for i in range(20):
        for state in ((),(0,),(3,),(0,3)):sx.append(phi(i,state));sy.append(labels[i])
    seed=registered_seed(split_seed,refit,"fresh5-all-dev-classifier");torch.manual_seed(seed);state_clf=fit(torch.nn.Linear(6,1),sx,sy)
    state={k:v.detach().reshape(-1).tolist() for k,v in state_clf.state_dict().items()};state_hash=hashlib.sha256(canonical_bytes(state)).hexdigest()

    # Independent, inference-only synthetic confirmation population.
    vvid=[f"confirm_v{i}" for i in range(10)];vgroup=[f"confirm_g{i}" for i in range(10)]
    vb=np.roll(np.tile(np.eye(4),(3,1))[:10],1,axis=1)*.85+.03
    vo=np.stack([np.asarray([((3*i+2*j+1)%5)/5,((i+3*j+2)%7)/7]) for i in range(10) for j in range(6)]).reshape(10,6,2)
    rows=[];traces=[]
    for i,vid in enumerate(vvid):
        for arm in arms:
            for draw in (range(20) if arm=="B2" else (None,)):
                ex=choose(vb,vo,i,arm,{a:learned[a]["refit_model"] for a in learned},route=bool(1/(1+np.exp(-(vb[i]@rw+rb)))>=.5),draw_id=draw or 0,video_id=vid);chosen=ex["actions"]
                ids=[action_ids.index(a) for a in chosen];vf=np.r_[vb[i],vo[i,ids].sum(0) if ids else [0,0]]
                score=float(predict_refit(state,vf[None])[0]);dec=[{"step":j,"action_id":a,"status":"ACQUIRE","estimated_cost_ms":action_cost[a],"realized_cost_ms":action_cost[a]} for j,a in enumerate(chosen)]
                th=hashlib.sha256(canonical_bytes(ex["decisions"])).hexdigest()
                rows.append({"schema":"cvoi-prediction/1","video_id":vid,"group_id":vgroup[i],"arm_id":arm,"draw_id":draw,"score":score,"prediction":int(score>=thresholds[arm]),"threshold":thresholds[arm],"threshold_source":"fresh_five_fold_all_dev_oof","realized_cost_ms":ex["realized_cost"],"action_trace_sha256":th})
                traces.append({"schema":"cvoi-acquisition-trace/1","video_id":vid,"arm_id":arm,"draw_id":draw,"ordered_actions":chosen,"estimated_cost_ms":ex["estimated_cost"],"realized_cost_ms":ex["realized_cost"],"decision_overhead_ms":ex["overhead_cost"],"decision_records":ex["decisions"],"trace_sha256":th})
    lineage={"schema":"cvoi-fresh5-confirmation-lineage/1","development_video_count":20,"development_group_count":10,"fold_count":5,
      "target_count":len(targets),"confirmation_video_count":10,"confirmation_group_count":10,"confirmation_inference_only":True,
      "confirmation_labels_loaded":False,"disjoint_video_ids":not bool(set(vids)&set(vvid)),"disjoint_group_ids":not bool(set(groups)&set(vgroup)),
      "state_classifier_refit":"all_development","selector_refit":"all_crossfit_targets","b6_router_refit":"all_development",
      "state_classifier_sha256":state_hash,"selector_sha256":{a:learned[a]["refit_model"]["sha256"] for a in learned},"router_sha256":router["sha256"],
      "classifier_records":classifier_records,"thresholds":thresholds,"test_contact_count":test_contact_count}
    return rows,traces,lineage
