from __future__ import annotations

import numpy as np

def macro_f1_binary(y, pred):
    y=np.asarray(y,int);pred=np.asarray(pred,int);vals=[]
    for c in (0,1):
        tp=np.sum((y==c)&(pred==c));fp=np.sum((y!=c)&(pred==c));fn=np.sum((y==c)&(pred!=c))
        vals.append(0.0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
    return float(np.mean(vals))

def select_threshold(y, score):
    raw=[float(x) for x in score]
    candidates=sorted(set(raw)|{0.5,float(np.nextafter(max(raw),np.inf))});rank=[]
    for t in candidates: rank.append((-macro_f1_binary(y,np.asarray(score)>=t),abs(t-.5),t))
    return min(rank)[2]

def logloss(label,prob):
    p=min(max(float(prob),1e-7),1-1e-7)
    return -np.log(p if label else 1-p)

def utility(label,before,after): return float(logloss(label,before)-logloss(label,after))

def exact_knapsack(scores,cost_ticks,budget):
    dp={0:(0.0,())}
    for idx,(score,cost) in enumerate(zip(scores,cost_ticks)):
        nxt=dict(dp)
        for used,(value,chosen) in dp.items():
            nu=used+int(cost)
            if nu>budget:continue
            cand=(value+float(score),chosen+(idx,));old=nxt.get(nu)
            if old is None or cand[0]>old[0] or (cand[0]==old[0] and cand[1]<old[1]):nxt[nu]=cand
        dp=nxt
        if len(dp)>5_000_000:raise RuntimeError("HALT_DP_STATES")
    candidates=[(value,-used,chosen) for used,(value,chosen) in dp.items()]
    return max(candidates,key=lambda x:(x[0],x[1],tuple(-q for q in x[2])))[2]

def cost_heterogeneous(values):
    x=np.asarray(values,float)
    if x.size<2 or np.any(x<=0):return False
    return bool(np.std(x)/np.mean(x)>=.10 and np.percentile(x,90)/np.percentile(x,10)>=1.25)

def b12_applicability(fold_costs):
    """Require a live heterogeneous type in >=4/5 folds for every split seed."""
    result={}
    for seed,folds in fold_costs.items():
        if len(folds)!=5:raise RuntimeError("HALT_B12_FOLD_COUNT")
        live=sum(any(cost_heterogeneous(v) for v in f.values() if len(v)) for f in folds)
        result[str(seed)]={"heterogeneous_folds":live,"pass":live>=4}
    return {"applicable":all(x["pass"] for x in result.values()),"seeds":result,
            "label":"knapsack" if all(x["pass"] for x in result.values()) else "sequential_feasible_topk"}

def strongest_admissible(candidate_f1,candidate_cost,baselines):
    legal=[(name,f1,cost) for name,f1,cost in baselines if cost<=candidate_cost]
    if not legal:raise RuntimeError("HALT_NO_ADMISSIBLE_BASELINE")
    return sorted(legal,key=lambda x:(-x[1],x[2],x[0]))[0]

def validate_policy_rows(rows,heldout_groups):
    heldout=set(heldout_groups)
    for row in rows:
        if row["eval_group"] in set(row["fit_groups"]):raise RuntimeError("HALT_POLICY_TARGET_LEAKAGE")
        if row["eval_group"] not in heldout:raise RuntimeError("HALT_WRONG_EVAL_GROUP")
    return True

class PurchasedStore:
    def __init__(self,outcomes,costs):self._outcomes=dict(outcomes);self._costs=dict(costs);self.purchased=set();self.spent=0
    def purchase(self,action,budget):
        cost=self._costs[action]
        if self.spent+cost>budget:return None
        self.spent+=cost;self.purchased.add(action);return self._outcomes[action]
    def peek(self,action):
        if action not in self.purchased:raise RuntimeError("HALT_UNPURCHASED_ACTION_ACCESS")
        return self._outcomes[action]

def verdict(gates):
    order=[("G0","HALT-MEASUREMENT"),("G1","NO-GO-ACTION-HEADROOM"),("G2","NO-GO-SPARSE-HEADROOM"),
           ("G3","NO-GO-CVOI"),("G4","CVOI-PREDICTIVE-BUT-NOT-EFFICIENT")]
    for key,label in order:
        if not gates.get(key,False):return label
    return "GO-CVOI" if gates.get("G5",False) else "TRAIN-OOF-ONLY"
