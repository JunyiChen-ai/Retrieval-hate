#!/usr/bin/env python3
"""Four-arm, three-seed V25 trainer. Never reads temporal labels."""
import argparse,json,random
from pathlib import Path
import torch
from core import *
from reference_builder import load_bags,verify
ARMS=('real','permuted','negative_reference_only','global_only')
def verify_declaration(path=None):
 p=Path(path or Path(__file__).with_name('FINAL_PROTOCOL_DECLARATION.json'));d=json.load(open(p));keys={'schema','status','supersedes','canonical_derivation','identities','identity_set_sha256','labels_or_gt_read','test_read'}
 if set(d)!=keys or d['schema']!='v25_final_pretraining_declaration_v1' or d['status']!='FINAL_AUTHORITATIVE_PRETRAINING_DECLARATION' or d['labels_or_gt_read'] is not False or d['test_read'] is not False or d['identity_set_sha256']!=canon_hash(d['identities']):raise RuntimeError('protocol declaration schema/hash')
 for x in d['identities'].values():
  if set(x)!={'path','sha256'} or sha(x['path'])!=x['sha256']:raise RuntimeError('protocol declared identity tamper')
 return d
def refs(root,fam,k=None):return json.load(open(Path(root)/(f'{fam}_full.json' if k is None else f'{fam}_exclude_fold{k}.json')))
def transform(rows,root,crossfit=True):
 out=[]
 for r in rows:
  k=fold(r['video_id']) if crossfit else None;z=np.stack([ecdf_logit(np.array(r['families'][f][0]),refs(root,f,k)) for f in ('text','multimodal')]);out.append({'id':r['video_id'],'y':r['video_label'],'g':r['global_causal_score'],'z':z})
 return out
def permute(rows,seed):
 groups={}
 for r in rows:groups.setdefault(r['z'].shape[1],[]).append(r)
 out=[];moved_v=moved_i=total_i=0;mapping={}
 for key,rs in groups.items():
  rs=sorted(rs,key=lambda x:x['id']);n=len(rs);shift=0 if n==1 else 1+seed%(n-1)
  for i,r in enumerate(rs):
   d=rs[(i+shift)%n];q={**r,'z':d['z'].copy()};out.append(q);mapping[r['id']]=d['id'];total_i+=r['z'].shape[1]
   if d['id']!=r['id']:moved_v+=1;moved_i+=r['z'].shape[1]
 ids=sorted(x['id'] for x in rows);lengths={r['id']:r['z'].shape[1] for r in rows};donor_folds={v:fold(d) for v,d in mapping.items()};zh={r['id']:canon_hash(r['z'].tolist()) for r in rows};man={'schema':'v25_permutation_seed_v1','seed':seed,'mapping':mapping,'donor_folds':donor_folds,'donor_z_sha256':{v:zh[d] for v,d in mapping.items()},'donor_scores_are_oof_by_donor_fold':True,'pre_ids_sha256':canon_hash(ids),'post_ids_sha256':canon_hash(sorted(x['id'] for x in out)),'pre_lengths_sha256':canon_hash(lengths),'post_lengths_sha256':canon_hash({r['id']:r['z'].shape[1] for r in out}),'pre_instances_sha256':canon_hash([[r['id'],r['z'].shape[1]] for r in sorted(rows,key=lambda x:x['id'])]),'post_instances_sha256':canon_hash([[r['id'],r['z'].shape[1]] for r in sorted(out,key=lambda x:x['id'])]),'nonself_ids':[v for v in ids if mapping[v]!=v],'n_videos':len(rows),'n_instances':total_i,'moved_video_fraction':moved_v/len(rows),'moved_instance_fraction':moved_i/total_i,'lengths':lengths}
 return sorted(out,key=lambda x:x['id']),man
def verify_permutation_manifest(rows,pm):
 if set(pm)!=set(map(str,SEEDS)):raise RuntimeError('permutation seeds')
 base={r['id']:r for r in rows};ids=sorted(base);lengths={v:base[v]['z'].shape[1] for v in ids};inst=canon_hash([[v,lengths[v]] for v in ids]);required={'schema','seed','mapping','donor_folds','donor_z_sha256','donor_scores_are_oof_by_donor_fold','pre_ids_sha256','post_ids_sha256','pre_lengths_sha256','post_lengths_sha256','pre_instances_sha256','post_instances_sha256','nonself_ids','n_videos','n_instances','moved_video_fraction','moved_instance_fraction','lengths'}
 for s in SEEDS:
  m=pm[str(s)]
  if set(m)!=required or m['schema']!='v25_permutation_seed_v1' or m['seed']!=s or set(m['mapping'])!=set(ids) or set(m['mapping'].values())!=set(ids) or m['lengths']!=lengths:raise RuntimeError('permutation schema')
  if any(m[k]!=v for k,v in {'pre_ids_sha256':canon_hash(ids),'post_ids_sha256':canon_hash(ids),'pre_lengths_sha256':canon_hash(lengths),'post_lengths_sha256':canon_hash(lengths),'pre_instances_sha256':inst,'post_instances_sha256':inst}.items()):raise RuntimeError('permutation hashes')
  non=[v for v in ids if m['mapping'][v]!=v]
  if m['nonself_ids']!=non or any(lengths[v]!=lengths[d] for v,d in m['mapping'].items()) or any(m['donor_folds'][v]!=fold(d) for v,d in m['mapping'].items()) or any(m['donor_z_sha256'][v]!=canon_hash(base[d]['z'].tolist()) for v,d in m['mapping'].items()):raise RuntimeError('donor identity')
  mv=len(non)/len(ids);mi=sum(lengths[v] for v in non)/sum(lengths.values())
  if m['moved_video_fraction']!=mv or m['moved_instance_fraction']!=mi or mv<.8 or mi<.8 or m['donor_scores_are_oof_by_donor_fold'] is not True:raise RuntimeError('permutation identifiability')
 return True
def train_arm(rows,arm,seed,out):
 torch.manual_seed(seed);random.seed(seed);m=V25(arm in ('real','permuted'));
 if arm=='global_only':m.gamma.requires_grad_(False)
 opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-2);hist=[]
 for epoch in range(6):
  if epoch:
   order=list(rows);random.Random(seed+epoch).shuffle(order)
   for r in order:
    opt.zero_grad();loss=loss_one(m,r['g'],torch.tensor(r['z']),r['y'])
    if arm=='global_only':
     L,_=m(r['g'],torch.tensor(r['z']));loss=torch.nn.functional.binary_cross_entropy_with_logits(L,torch.tensor(float(r['y']),dtype=torch.float64))
    loss.backward();opt.step();m.project()
  hist.append({'epoch':epoch,'state':{k:v.detach().clone() for k,v in m.state_dict().items()}})
 payload={'schema':'v25_checkpoint_v1','arm':arm,'seed':seed,'epochs':list(range(6)),'history':hist};payload['state_sha256']=[canon_hash({k:v.tolist() for k,v in h['state'].items()}) for h in hist];torch.save(payload,out)
def main():
 p=argparse.ArgumentParser();p.add_argument('--bags',required=True);p.add_argument('--reference',required=True);p.add_argument('--out',required=True);a=p.parse_args();verify_declaration();verify(a.reference,a.bags);rows=transform(load_bags(a.bags),a.reference,True);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);perms={};prs={}
 for seed in SEEDS:
  pr,pm=permute(rows,seed);perms[str(seed)]=pm;prs[seed]=pr
 verify_permutation_manifest(rows,perms)
 for seed in SEEDS:
  for arm in ARMS:train_arm(prs[seed] if arm=='permuted' else rows,arm,seed,out/f'{arm}_seed{seed}.pt')
 verify_permutation_manifest(rows,perms);pp=out/'permutation_manifest.json';pp.write_text(json.dumps(perms,indent=2,sort_keys=True)+'\n');prot={'schema':'v25_train_protocol_v1','status':'TRAINED_NO_VAL_OR_TEST','bags_sha256':sha(a.bags),'reference_manifest_sha256':sha(Path(a.reference)/'manifest.json'),'trainer_sha256':sha(__file__),'core_sha256':sha(Path(__file__).with_name('core.py')),'selector_sha256':sha(Path(__file__).with_name('selector.py')),'inference_sha256':sha(Path(__file__).with_name('inference.py')),'reference_builder_sha256':sha(Path(__file__).with_name('reference_builder.py')),'val_predict_sha256':sha(Path(__file__).with_name('val_predict.py')),'permutation_manifest_sha256':sha(pp),'protocol_declaration_sha256':sha(Path(__file__).with_name('FINAL_PROTOCOL_DECLARATION.json')),'seeds':list(SEEDS),'epochs':[0,1,2,3,4,5],'arms':list(ARMS)};(out/'protocol.json').write_text(json.dumps(prot,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
