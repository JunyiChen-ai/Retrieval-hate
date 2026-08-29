#!/usr/bin/env python3
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from frame_eval_common import rank_roc_auc,average_precision
from relation_v4.io import load_manifest,fit_ecdf,apply_ecdf
from relation_v4.io import sha256
from relation_v8.run import load_split_exact,atomic_json
from relation_v10.diagnostic import components,score
from relation_v8.model import UnifiedRelationV8
from relation_v11.robust import fit,huber_barycenter

def center(x):return {v:z-z.mean(0,keepdims=True) for v,z in x.items()}
def flatten(x,ids):return np.concatenate([x[v] for v in ids])
def macro(scores,gt):
 ids=[v for v in gt if gt[v].any() and not gt[v].all()]
 return np.asarray([rank_roc_auc(scores[v],gt[v]) for v in ids]),ids
def shuffled_auc(value,y,vid,n=200):
 out=[];base=int.from_bytes(hashlib.sha256(vid.encode()).digest()[:8],'little')
 for i in range(n):out.append(rank_roc_auc(value[np.random.default_rng(base+i).permutation(len(value))],y))
 return float(np.mean(out))
def lower_ci(values,seed=11,n=2000):
 values=np.asarray(values,float);rng=np.random.default_rng(seed)
 draws=np.mean(values[rng.integers(0,len(values),(n,len(values)))],1)
 return float(np.percentile(draws,2.5))
def pooled(scores,gt):
 s=np.concatenate([scores[v] for v in sorted(gt)]);y=np.concatenate([gt[v] for v in sorted(gt)])
 return average_precision(s,y),rank_roc_auc(s,y)

def candidate_stats(name,values,gt):
 aucs,mixed=macro(values,gt);shuffle=np.asarray([shuffled_auc(values[v],gt[v],v) for v in mixed]);contrast=aucs-shuffle
 return {'source':name,'macro_roc':float(aucs.mean()),'roc_ci_lower':lower_ci(aucs),
         'shuffle_macro_roc':float(shuffle.mean()),'contrast':float(contrast.mean()),
         'contrast_ci_lower':lower_ci(contrast),'eligible':len(mixed)}
def state_json(state):
 return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in state.items()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 m=load_manifest(a.manifest);names=[e['name'] for e in m['experts']];raw,gt,_=load_split_exact(m,'val');refs=fit_ecdf(raw);val=apply_ecdf(raw,refs);ids=sorted(val);folds=[ids[i::5] for i in range(5)]
 loc=center(val);outer_scores={};outer_rows=[]
 for fold_index,held in enumerate(folds):
  inner=[v for v in ids if v not in set(held)]
  inner_refs=fit_ecdf({v:raw[v] for v in inner});fold_values=apply_ecdf(raw,inner_refs);fold_loc=center(fold_values)
  locator_state=fit(flatten(fold_loc,inner))
  inner_values={name:{v:fold_loc[v][:,i] for v in inner} for i,name in enumerate(names)}
  inner_values['robust_copula']={v:huber_barycenter(fold_loc[v],locator_state) for v in inner}
  stats=[candidate_stats(name,value,{v:gt[v] for v in inner}) for name,value in inner_values.items()]
  passing=[x for x in stats if x['roc_ci_lower']>.5 and x['contrast_ci_lower']>0]
  choice=max(passing,key=lambda x:(x['contrast'],x['macro_roc'])) if passing else None
  for v in held:
   if choice is None:outer_scores[v]=np.zeros(len(fold_values[v]))
   elif choice['source']=='robust_copula':outer_scores[v]=huber_barycenter(fold_loc[v],locator_state)
   else:outer_scores[v]=fold_loc[v][:,names.index(choice['source'])]
  outer_rows.append({'outer_fold':fold_index,'inner_ids':len(inner),'heldout_ids':len(held),
                     'inner_candidates':stats,'inner_selected':choice})
 outer_stat=candidate_stats('nested_oof_selected_role',outer_scores,gt)
 outer_gate=outer_stat['roc_ci_lower']>.5 and outer_stat['contrast_ci_lower']>0
 full_state=fit(flatten(loc,ids));full_values={name:{v:loc[v][:,i] for v in ids} for i,name in enumerate(names)}
 full_values['robust_copula']={v:huber_barycenter(loc[v],full_state) for v in ids}
 candidates=[candidate_stats(name,value,gt) for name,value in full_values.items()]
 passing=[x for x in candidates if x['roc_ci_lower']>.5 and x['contrast_ci_lower']>0]
 selected=(max(passing,key=lambda x:(x['contrast'],x['macro_roc'])) if passing and outer_gate else None)
 prior_state=fit(np.stack([val[v].mean(0) for v in ids]))
 frozen={'protocol':'V11 pilot nested 5-fold validation cross-fitting','shuffle_B':200,'bootstrap_B':2000,
         'outer_folds':outer_rows,'outer_heldout_gate':outer_stat,'outer_gate_pass':outer_gate,
         'full_validation_candidates':candidates,'locator_activation':selected,
         'ecdf_and_dependence':'fit on each outer inner-train scores; final state fit full validation only',
         'manifest':str(Path(a.manifest).resolve()),'manifest_sha256':sha256(a.manifest),
         'full_validation_ecdf':refs,'full_locator_state':state_json(full_state),
         'full_prior_state':state_json(prior_state),
         'test_opened':False}
 freeze_path=Path(a.out).with_suffix('.frozen.json');atomic_json(freeze_path,frozen)
 test_raw,test_gt,_=load_split_exact(m,'test');test=apply_ecdf(test_raw,refs);test_loc=center(test)
 test_prior={v:np.full(len(test[v]),float(huber_barycenter(test[v].mean(0,keepdims=True),prior_state)[0])) for v in test}
 if selected:
  if selected['source']=='robust_copula':loc={v:huber_barycenter(test_loc[v],full_state) for v in test}
  else:idx=names.index(selected['source']);loc={v:test_loc[v][:,idx] for v in test}
  role={v:test_prior[v]+.01*loc[v] for v in test};role_shuffle={v:test_prior[v]+.01*loc[v][np.random.default_rng(int.from_bytes(hashlib.sha256(v.encode()).digest()[:8],'little')).permutation(len(loc[v]))] for v in test}
 else:loc={v:np.zeros(len(test[v])) for v in test};role=dict(test_prior);role_shuffle=dict(test_prior)
 prior_ap,prior_roc=pooled(test_prior,test_gt);role_ap,role_roc=pooled(role,test_gt);aucs,mixed=macro(role,test_gt);paucs,_=macro(test_prior,test_gt);saucs,_=macro(role_shuffle,test_gt)
 # Exact identity V8 remains the performance candidate/fallback.
 cfg=json.loads(Path(f"results/reproduction/relation_v10/diagnostic_stable/{m['corpus']}/frozen_config.json").read_text());model=UnifiedRelationV8(len(names),m.get('window',12),m.get('temperature',.2)).eval();parts=components(model,test)
 fb=cfg['identity_v8_fallback'];identity=score(parts,np.full(len(names),1/len(names)),fb['beta'],fb['gamma']);iap,iroc=pooled(identity,test_gt)
 performance='role_fusion' if selected and role_ap>=iap and role_roc>=iroc else 'identity_v8'
 payload={'corpus':m['corpus'],'crossfit_folds':5,'shuffle_repeats':200,'bootstrap_repeats':2000,
 'outer_folds':outer_rows,'outer_heldout_gate':outer_stat,'outer_gate_pass':outer_gate,'validation_candidates':candidates,'locator_activation':selected,
 'test_role':{'prior_ap':prior_ap,'prior_roc':prior_roc,'full_ap':role_ap,'full_roc':role_roc,'pooled_ap_gain':role_ap-prior_ap,'mixed_macro_roc':float(aucs.mean()),'mixed_macro_roc_gain':float((aucs-paucs).mean()),'shuffled_mixed_macro_roc_gain':float((saucs-paucs).mean()),'eligible':len(mixed)},
 'performance_fallback':{'selected':performance,'identity_ap':iap,'identity_roc':iroc,'role_ap':role_ap,'role_roc':role_roc},'frozen_before_test':str(freeze_path.resolve()),'test_labels_used_for_selection':False}
 atomic_json(a.out,payload);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
