#!/usr/bin/env python3
"""Cross-corpus locator information and validation-only role audit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from frame_eval_common import average_precision, rank_roc_auc
from relation_v4.io import apply_ecdf, fit_ecdf, load_manifest
from relation_v8.run import atomic_json, load_split_exact
from relation_v8.model import UnifiedRelationV8
from relation_v10.diagnostic import components, score


def mixed_macro(scores, gt):
    rows=[]
    for vid in sorted(gt):
        y=gt[vid]
        if y.any() and not y.all():
            rows.append((rank_roc_auc(scores[vid],y),average_precision(scores[vid],y)))
    return {"macro_roc":float(np.mean([x[0] for x in rows])) if rows else None,
            "macro_ap":float(np.mean([x[1] for x in rows])) if rows else None,
            "eligible_videos":len(rows)}


def centered_hateful_pooled(scores,gt):
    ids=[v for v in sorted(gt) if gt[v].any()]
    s=np.concatenate([scores[v]-np.mean(scores[v]) for v in ids]);y=np.concatenate([gt[v] for v in ids])
    return {"centered_pooled_roc":rank_roc_auc(s,y),"centered_pooled_ap":average_precision(s,y),
            "hateful_videos":len(ids)}


def pooled(scores,gt):
    s=np.concatenate([scores[v] for v in sorted(gt)]);y=np.concatenate([gt[v] for v in sorted(gt)])
    return {"pooled_roc":rank_roc_auc(s,y),"pooled_ap":average_precision(s,y)}


def shuffle(scores):
    out={}
    for vid,value in scores.items():
        seed=int.from_bytes(hashlib.sha256(vid.encode()).digest()[:8],"little")
        out[vid]=np.asarray(value)[np.random.default_rng(seed).permutation(len(value))]
    return out


def metrics(scores,gt):
    return {**pooled(scores,gt),**mixed_macro(scores,gt),**centered_hateful_pooled(scores,gt)}


def sources(values,names):
    result={name:{v:x[:,i] for v,x in values.items()} for i,name in enumerate(names)}
    result["equal_consensus"]={v:x.mean(1) for v,x in values.items()}
    return result


def add_frozen_full(result,values,manifest):
    path=Path(f"results/reproduction/relation_v10/diagnostic_stable/{manifest['corpus']}/frozen_config.json")
    config=json.loads(path.read_text());n=len(manifest["experts"])
    model=UnifiedRelationV8(n,manifest.get("window",12),manifest.get("temperature",.2)).eval();parts=components(model,values)
    fallback=config["identity_v8_fallback"];identity=np.full(n,1/n)
    result["v8_full"]=score(parts,identity,fallback["beta"],fallback["gamma"])
    chosen=config["selected"];weights=np.asarray(config["candidate_weights"][chosen["aggregation"]])
    result["v10_full"]=score(parts,weights,chosen["beta"],chosen["gamma"])
    return result


def prior(source): return {v:np.full(len(x),np.mean(x)) for v,x in source.items()}
def locator(source): return {v:x-np.mean(x) for v,x in source.items()}
def combine(prior_score,locator_score,beta):
    return {v:prior_score[v]+beta*locator_score[v] for v in prior_score}


def audit_sources(all_sources,gt):
    report={}
    for name,value in all_sources.items():
        report[name]={"observed":metrics(value,gt),"time_shuffle":metrics(shuffle(value),gt)}
    return report


def select_roles(val_sources,val_gt,beta_grid):
    prior_rows=[]
    for name,value in val_sources.items():
        p=prior(value);m=pooled(p,val_gt);prior_rows.append({"source":name,**m})
    prior_choice=max(prior_rows,key=lambda x:(x["pooled_ap"],x["pooled_roc"],x["source"]=="equal_consensus"))
    p=prior(val_sources[prior_choice["source"]]);fallback=pooled(p,val_gt);rows=[]
    for name,value in val_sources.items():
        loc=locator(value)
        for beta in beta_grid:
            m=pooled(combine(p,loc,beta),val_gt)
            rows.append({"locator_source":name,"beta":float(beta),**m})
    eligible=[x for x in rows if x["pooled_ap"]>=fallback["pooled_ap"] and x["pooled_roc"]>=fallback["pooled_roc"]]
    selected=max(eligible,key=lambda x:(x["pooled_ap"],x["pooled_roc"],-abs(x["beta"]),x["locator_source"]=="equal_consensus"))
    return {"prior":prior_choice,"prior_fallback":fallback,"locator":selected,
            "validation_locator_gain_ap":selected["pooled_ap"]-fallback["pooled_ap"],"grid":rows}


def complementarity(all_sources,gt):
    names=[x for x in all_sources if x!="equal_consensus"];rows=[]
    mixed=[v for v in gt if gt[v].any() and not gt[v].all()]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            av=np.concatenate([locator(all_sources[a])[v] for v in mixed]);bv=np.concatenate([locator(all_sources[b])[v] for v in mixed])
            corr=float(np.corrcoef(av,bv)[0,1]) if np.std(av)>0 and np.std(bv)>0 else 0.
            mean={v:(all_sources[a][v]+all_sources[b][v])/2 for v in gt}
            rows.append({"a":a,"b":b,"centered_correlation":corr,**mixed_macro(mean,gt)})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    manifest=load_manifest(a.manifest);names=[x["name"] for x in manifest["experts"]]
    val_raw,val_gt,_=load_split_exact(manifest,"val");refs=fit_ecdf(val_raw);val=apply_ecdf(val_raw,refs);base_val_sources=sources(val,names)
    # One preregistered, corpus-agnostic locator scale grid. Small scales let
    # the locator break within-video ties without erasing cross-video prior.
    beta_grid=(-3.2,-1.6,-1.,-.8,-.4,-.2,-.1,-.05,-.025,-.01,0.,.01,.025,.05,.1,.2,.4,.8,1.,1.6,3.2)
    role=select_roles(base_val_sources,val_gt,beta_grid)
    val_base_audit=audit_sources(base_val_sources,val_gt)
    information_candidates=[]
    for name,item in val_base_audit.items():
        observed=item["observed"]["centered_pooled_roc"];shuffled=item["time_shuffle"]["centered_pooled_roc"]
        if observed is not None and observed>.5 and observed>shuffled:
            information_candidates.append({"source":name,"centered_roc":observed,
                                           "shuffle_roc":shuffled,"contrast":observed-shuffled})
    information_role=max(information_candidates,key=lambda x:(x["contrast"],x["centered_roc"])) if information_candidates else None
    val_sources=add_frozen_full(dict(base_val_sources),val,manifest)
    test_raw,test_gt,_=load_split_exact(manifest,"test");test=apply_ecdf(test_raw,refs);base_test_sources=sources(test,names);test_sources=add_frozen_full(dict(base_test_sources),test,manifest)
    prior_test=prior(base_test_sources[role["prior"]["source"]]);locator_test=locator(base_test_sources[role["locator"]["locator_source"]])
    full=combine(prior_test,locator_test,role["locator"]["beta"]);shuffled=combine(prior_test,shuffle(locator_test),role["locator"]["beta"])
    prior_metric=metrics(prior_test,test_gt);full_metric=metrics(full,test_gt);shuffle_metric=metrics(shuffled,test_gt)
    information_test=None
    if information_role:
        iloc=locator(base_test_sources[information_role["source"]]);ifull=combine(prior_test,iloc,.01);ishuffle=combine(prior_test,shuffle(iloc),.01)
        im,ism=metrics(ifull,test_gt),metrics(ishuffle,test_gt)
        information_test={"validation_selected_locator":information_role,"fixed_beta":.01,
                          "full":im,"time_shuffled":ism,
                          "mixed_macro_ap_gain":im["macro_ap"]-prior_metric["macro_ap"],
                          "shuffled_mixed_macro_ap_gain":ism["macro_ap"]-prior_metric["macro_ap"]}
    payload={"corpus":manifest["corpus"],"calibration":"validation-reference ECDF","role_rule":"best validation prior-only pooled AP; then locator source on fixed shared scale grid maximizing validation pooled AP subject to pooled ROC nondecrease","locator_scale_grid":list(beta_grid),"validation_role_selection":role,
             "locator_information_role_rule":"max validation centered-pooled ROC time-shuffle contrast among sources with observed ROC>0.5; fixed beta=0.01",
             "validation_source_audit":{**val_base_audit,**{k:v for k,v in audit_sources(val_sources,val_gt).items() if k not in val_base_audit}},"validation_complementarity":complementarity(val_sources,val_gt),
             "test_source_audit":audit_sources(test_sources,test_gt),"test_complementarity":complementarity(test_sources,test_gt),
             "test_role_result":{"prior_only":prior_metric,"full":full_metric,"time_shuffled_locator":shuffle_metric,
             "locator_gain_ap":full_metric["pooled_ap"]-prior_metric["pooled_ap"],"shuffle_gain_ap":shuffle_metric["pooled_ap"]-prior_metric["pooled_ap"]},
             "test_locator_information_role":information_test,
             "test_labels_used_for_selection":False}
    atomic_json(a.out,payload);print(json.dumps(payload["test_role_result"],indent=2))


if __name__=="__main__":main()
