from __future__ import annotations

import hashlib
import numpy as np
import torch

from .common import canonical_bytes
from .protocol import macro_f1_binary,select_threshold


def _fit_linear(x,y,lr,epochs,seed):
    torch.manual_seed(int(seed));m=torch.nn.Linear(x.shape[1],1);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    X=torch.as_tensor(x,dtype=torch.float32);Y=torch.as_tensor(y,dtype=torch.float32)
    for _ in range(int(epochs)):
        loss=torch.nn.functional.binary_cross_entropy_with_logits(m(X).squeeze(-1),Y);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    return m

def nested_fit_predict(x,y,groups,query_groups,split_seed,refit,configs=(("lr1e-4",1e-4),("lr3e-4",3e-4)),epochs=10):
    x=np.asarray(x,float);y=np.asarray(y,int);groups=np.asarray(groups);query=set(query_groups);fit_mask=np.asarray([g not in query for g in groups]);query_mask=~fit_mask
    fit_groups=sorted(set(groups[fit_mask]));inner={g:i%min(4,len(fit_groups)) for i,g in enumerate(fit_groups)};rank=[];records=[]
    for config_id,lr in configs:
        oof=np.full(len(y),np.nan)
        for fold in sorted(set(inner.values())):
            held={g for g,f in inner.items() if f==fold};tr=fit_mask&np.asarray([g not in held for g in groups]);va=fit_mask&np.asarray([g in held for g in groups])
            model=_fit_linear(x[tr],y[tr],lr,epochs,split_seed+refit*101+fold)
            with torch.inference_mode():oof[va]=torch.sigmoid(model(torch.as_tensor(x[va],dtype=torch.float32))).squeeze(-1).numpy()
        legal=oof[fit_mask];threshold=select_threshold(y[fit_mask],legal);f1=macro_f1_binary(y[fit_mask],legal>=threshold)
        records.append({"config_id":config_id,"inner_oof_f1":f1,"threshold":threshold,
                        "inner_oof_sha256":hashlib.sha256(canonical_bytes(legal.tolist())).hexdigest()})
        rank.append((-f1,config_id,lr,threshold))
    _,config_id,lr,threshold=min(rank);model=_fit_linear(x[fit_mask],y[fit_mask],lr,epochs,split_seed+refit*101+99)
    with torch.inference_mode():query_score=torch.sigmoid(model(torch.as_tensor(x[query_mask],dtype=torch.float32))).squeeze(-1).numpy()
    state={k:v.detach().cpu().reshape(-1).tolist() for k,v in model.state_dict().items()};model_hash=hashlib.sha256(canonical_bytes(state)).hexdigest()
    return {"query_indices":np.flatnonzero(query_mask).tolist(),"query_scores":query_score.tolist(),"threshold":threshold,"refit_state":state,"refit_model_sha256":model_hash,
            "selection":{"selected_config":config_id,"outer_refit_epoch":epochs,"outer_callback":False,"fit_groups":fit_groups,
                         "query_groups":sorted(query),"configs":records,"query_score_sha256":hashlib.sha256(canonical_bytes(query_score.tolist())).hexdigest()}}

def predict_refit(state,x):
    X=np.asarray(x,float);w=np.asarray(state["weight"],float).reshape(-1);b=float(state["bias"][0]);return 1/(1+np.exp(-(X@w+b)))
