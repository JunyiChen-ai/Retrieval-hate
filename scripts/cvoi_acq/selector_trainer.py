"""Leakage-safe selector training primitives for the CVoI closed loop."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch import nn
from .common import canonical_bytes
from .closed_loop_fixture import fit_early, registered_seed

STATEFUL=("B7","B8","B10","B12")

def _arrays(rows,arm):
    cheap=np.asarray([r["cheap_features"] for r in rows],np.float32)
    action=np.asarray([r["action_features"] for r in rows],np.float32)
    state=np.asarray([r["state_features"] for r in rows],np.float32)
    budget=np.asarray([[r["remaining_budget"],r["estimated_cost"]] for r in rows],np.float32)
    # B9 and B11 are provably singleton/state blind.
    x=np.c_[cheap,action,budget] if arm in ("B9","B11") else np.c_[cheap,state,action,budget]
    y=np.asarray([r["utility"] for r in rows],np.float32)
    return x,y

def fit_selector(rows,arm,seed_parts,epochs=30,lr=.01):
    x,y=_arrays(rows,arm); torch.manual_seed(registered_seed(*seed_parts,arm))
    if arm=="B11":
        a=np.c_[np.ones(len(x)),x]; coef=np.linalg.lstsq(a,y,rcond=None)[0]
        state={"kind":"ridge","coef":coef.tolist()}
    else:
        m=nn.Sequential(nn.Linear(x.shape[1],32),nn.GELU(),nn.Linear(32,1))
        X=torch.as_tensor(x);Y=torch.as_tensor(y);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
        for _ in range(epochs):
            loss=nn.functional.huber_loss(m(X).squeeze(-1),Y);opt.zero_grad();loss.backward();opt.step()
        state={"kind":"mlp","input_dim":x.shape[1],"state":{k:v.detach().tolist() for k,v in m.state_dict().items()}}
    state["sha256"]=hashlib.sha256(canonical_bytes(state)).hexdigest();return state

def predict_selector(model,rows,arm):
    x,_=_arrays([{**r,"utility":r.get("utility",0.)} for r in rows],arm)
    if model["kind"]=="ridge":
        c=np.asarray(model["coef"]);return np.c_[np.ones(len(x)),x]@c
    m=nn.Sequential(nn.Linear(model["input_dim"],32),nn.GELU(),nn.Linear(32,1))
    m.load_state_dict({k:torch.tensor(v) for k,v in model["state"].items()});m.eval()
    with torch.inference_mode():return m(torch.as_tensor(x)).squeeze(-1).numpy()

def fit_router(train_features,train_labels,seed_parts):
    """B6 router accepts train rows only; callers never pass outer-query labels."""
    x=np.asarray(train_features,np.float32);y=np.asarray(train_labels,np.float32)
    torch.manual_seed(registered_seed(*seed_parts,"B6-router"));m=nn.Linear(x.shape[1],1)
    opt=torch.optim.AdamW(m.parameters(),lr=.03)
    X=torch.as_tensor(x);Y=torch.as_tensor(y)
    for _ in range(40):
        loss=nn.functional.binary_cross_entropy_with_logits(m(X).squeeze(-1),Y);opt.zero_grad();loss.backward();opt.step()
    state={k:v.detach().tolist() for k,v in m.state_dict().items()}
    return {"state":state,"sha256":hashlib.sha256(canonical_bytes(state)).hexdigest()}

def nested_selector_fit(rows,inner_records,arm,split_seed,outer_fold,refit_id,
                        configs=((.01,20),(.03,30)),expected_folds=4):
    """Registered-fold OOF selection followed by callback-free refit."""
    if len(inner_records)!=expected_folds:raise RuntimeError("HALT_SELECTOR_INNER_COUNT")
    eval_once=[]
    for q in inner_records:eval_once.extend(q["eval_groups"])
    if len(eval_once)!=len(set(eval_once)):raise RuntimeError("HALT_SELECTOR_INNER_OVERLAP")
    records=[]
    for lr,epochs in configs:
        oof={};models={};loss=[]
        for q in inner_records:
            held=set(q["eval_groups"]);fit_groups=set(q["fit_groups"])
            tr=[r for r in rows if r["eval_group"] in fit_groups]
            va=[r for r in rows if r["eval_group"] in held]
            if not tr or not va:raise RuntimeError("HALT_SELECTOR_EMPTY_INNER")
            if any(r["eval_group"] in fit_groups for r in va):raise RuntimeError("HALT_SELECTOR_GROUP_LEAK")
            model=fit_selector(tr,arm,(split_seed,outer_fold,refit_id,q["inner_fold"],lr,epochs),epochs,lr)
            pred=predict_selector(model,va,arm);truth=np.asarray([r["utility"] for r in va])
            loss.append(float(np.mean((pred-truth)**2)));models[str(q["inner_fold"])]=model["sha256"]
            for r,p in zip(va,pred):
                key=(r["video_id"],tuple(r["state"]),r["action_id"])
                if key in oof:raise RuntimeError("HALT_SELECTOR_OOF_DUPLICATE")
                oof[key]=float(p)
        records.append({"config_id":f"lr{lr:g}-e{epochs}","lr":lr,"epochs":epochs,
                        "oof_mse":float(np.mean(loss)),"inner_model_sha256":models,
                        "oof_sha256":hashlib.sha256(canonical_bytes(sorted(oof.items(),key=str))).hexdigest(),
                        "oof_count":len(oof)})
    selected=min(records,key=lambda q:(q["oof_mse"],q["epochs"],q["config_id"]))
    model=fit_selector(rows,arm,(split_seed,outer_fold,refit_id,-1,selected["config_id"]),
                       selected["epochs"],selected["lr"])
    return {"arm_id":arm,"selection_metric":"inner_oof_utility_mse",
            "selected":selected,"configs":records,"refit_model":model,
            "outer_query_callback":False}
