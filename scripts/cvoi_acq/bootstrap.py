from __future__ import annotations
import numpy as np
from .protocol import macro_f1_binary,strongest_admissible
def paired_group_indices(group_ids,seed,n_boot,profiles=None,hierarchical=False):
    """Paired group bootstrap, optionally stratified by a frozen group profile.

    Hierarchical sensitivity resamples observations within each sampled group;
    the binding analysis keeps each sampled group intact.
    """
    unique=sorted(set(group_ids));members={g:np.asarray([i for i,x in enumerate(group_ids) if x==g]) for g in unique};rng=np.random.default_rng(seed)
    strata={}
    for g in unique: strata.setdefault((profiles or {}).get(g,"all"),[]).append(g)
    out=[]
    for _ in range(n_boot):
        sampled=[]
        for key in sorted(strata,key=str):
            gs=strata[key]
            for g in rng.choice(gs,len(gs),replace=True):
                ix=members[g]
                sampled.append(rng.choice(ix,len(ix),replace=True) if hierarchical else ix)
        out.append(np.concatenate(sampled))
    return out
def replicate(y,scores,costs,groups,thresholds,candidate,baselines,seed=1,n_boot=100,profiles=None,hierarchical=False):
    out=[]
    for idx in paired_group_indices(groups,seed,n_boot,profiles,hierarchical):
        f={a:macro_f1_binary(np.asarray(y)[idx],np.asarray(scores[a])[idx]>=thresholds[a]) for a in scores}
        c={a:float(np.mean(np.asarray(costs[a])[idx])) for a in costs}
        best=strongest_admissible(f[candidate],c[candidate],[(a,f[a],c[a]) for a in baselines])
        out.append({"delta":f[candidate]-best[1],"baseline":best[0]})
    return out

def nested_reselection(split_refit_runner, seeds=(20260811,20260812,20260813), n_refits=3):
    """Execute every split/refit path; callback must return candidate and baselines.

    Keeping reselection inside the callback makes it impossible to reuse the
    original-sample winner in a bootstrap replicate.
    """
    rows=[]
    for split_seed in seeds:
        for refit in range(n_refits):
            row=dict(split_refit_runner(int(split_seed),int(refit)))
            required={"candidate_f1","candidate_cost","baselines"}
            if not required<=set(row): raise RuntimeError("HALT_BOOTSTRAP_PATH_SCHEMA")
            b=strongest_admissible(row["candidate_f1"],row["candidate_cost"],row["baselines"])
            row.update({"split_seed":int(split_seed),"refit":refit,"selected_baseline":b[0],
                        "delta":float(row["candidate_f1"]-b[1])})
            rows.append(row)
    return rows

def interval_gate(values,alpha=.05,margin=0.0):
    a=np.asarray(values,dtype=float)
    if a.size==0 or not np.isfinite(a).all(): raise RuntimeError("HALT_BOOTSTRAP_VALUES")
    lo,hi=np.quantile(a,[alpha/2,1-alpha/2])
    return {"mean":float(a.mean()),"ci_low":float(lo),"ci_high":float(hi),
            "margin":float(margin),"pass":bool(lo>margin)}

def pareto_dominates(candidate,baseline,eps=1e-12):
    """Higher F1/lower cost; at least one strict improvement."""
    cf,cc=candidate;bf,bc=baseline
    return cf+eps>=bf and cc<=bc+eps and (cf>bf+eps or cc<bc-eps)

def bootstrap_complete_runs(runs,candidate,baselines,n_boot=10000,seed=20260819):
    """Binding bootstrap over 3 split seeds x 3 complete refits.

    Each run provides y/groups/profiles/scores/costs/frozen thresholds. Group
    multiplicities are drawn once per split and applied to every arm/refit.
    The admissible baseline is reselected from replicate-mean run metrics.
    """
    if n_boot!=10000:raise ValueError("binding bootstrap requires 10000 replicates")
    keys={(int(r["split_seed"]),int(r["refit"])) for r in runs}
    if len(keys)!=9 or len({k[0] for k in keys})!=3:raise RuntimeError("HALT_COMPLETE_RUN_IDS")
    by_split={}
    for r in runs:by_split.setdefault(int(r["split_seed"]),[]).append(r)
    rng=np.random.default_rng(seed);out=[]
    for rep_id in range(n_boot):
        arm_f={a:[] for a in [candidate]+list(baselines)};arm_c={a:[] for a in arm_f}
        for split,rr in sorted(by_split.items()):
            exemplar=rr[0];groups=list(exemplar["groups"]);profiles=exemplar["profiles"]
            strata={}
            for g in sorted(set(groups)):strata.setdefault(profiles[g],[]).append(g)
            sampled=[]
            for profile in sorted(strata,key=str):
                gs=strata[profile];sampled.extend(rng.choice(gs,len(gs),replace=True).tolist())
            members={g:np.asarray([i for i,x in enumerate(groups) if x==g]) for g in set(groups)}
            idx=np.concatenate([members[g] for g in sampled])
            for r in rr:
                if list(r["groups"])!=groups:raise RuntimeError("HALT_RUN_PAIRING")
                y=np.asarray(r["y"])[idx]
                for a in arm_f:
                    threshold=r["thresholds"][a]
                    applied=(np.asarray(threshold)[idx] if np.ndim(threshold)>0 else threshold)
                    arm_f[a].append(macro_f1_binary(y,np.asarray(r["scores"][a])[idx]>=applied))
                    arm_c[a].append(float(np.mean(np.asarray(r["costs"][a])[idx])))
        mf={a:float(np.mean(v)) for a,v in arm_f.items()};mc={a:float(np.mean(v)) for a,v in arm_c.items()}
        best=strongest_admissible(mf[candidate],mc[candidate],[(a,mf[a],mc[a]) for a in baselines])
        out.append({"replicate":rep_id,"candidate_f1":mf[candidate],"candidate_cost":mc[candidate],
                    "selected_baseline":best[0],"delta":mf[candidate]-best[1],"cost_saving":best[2]-mc[candidate]})
    return out

def resample_complete_run_ids(values,n_boot=10000,seed=20260820):
    a=np.asarray(values,float)
    if a.shape!=(9,):raise RuntimeError("HALT_EXPECTED_NINE_RUNS")
    rng=np.random.default_rng(seed);return rng.choice(a,(n_boot,9),replace=True).mean(1)
