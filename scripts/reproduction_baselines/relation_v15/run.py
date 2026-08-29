#!/usr/bin/env python3
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE.parent),str(HERE.parent.parent/'duplex')]
from hate_common import data as hdata
from relation_v2.protocol import frozen_splits,scoped_labels
from relation_v4.io import sha256
from relation_v8.run import load_split_exact,atomic_json
from relation_v11.score_stream_benchmark import metrics
from relation_v12.diagnostic import frozen_v10_identity
from relation_v15.data import FrozenFeatures,DIMS
from relation_v15.model import EmissionMIL,exact_bag_nll

LAMBDAS=(0.,.01,.025,.05,.1,.2,.4);EPOCHS=5
def infer(model,store,ids):
 model.eval();out={};audit={}
 with torch.no_grad():
  for v in ids:
   x,miss,t=store.load(v);z=(model(x)-np.log(t)).numpy();out[v]=z-z.mean();audit[v]={'n_frames':t,'missing_modalities':miss}
 return out,audit
def add(base,res,lam):return {v:base[v]+lam*res[v] for v in base}
def shuffle(res):
 out={}
 for v,x in res.items():
  if len(x)<2:out[v]=x.copy();continue
  k=1+int.from_bytes(hashlib.sha256(v.encode()).digest()[:4],'little')%(len(x)-1);out[v]=np.roll(x,k)
 return out
def shuffle_controls(base,res,gt,b=200):
 rows=[]
 for j in range(b):
  shifted={}
  for v,x in res.items():
   if len(x)<2:shifted[v]=x.copy();continue
   k=1+int.from_bytes(hashlib.sha256(f'{j}:{v}'.encode()).digest()[:4],'little')%(len(x)-1);shifted[v]=np.roll(x,k)
  rows.append(metrics(add(base,shifted,1.),gt))
 keys=('frame_ap','frame_roc','within_centered_ap','within_centered_roc','within_macro_ap','within_macro_roc')
 return {'B':b,**{k:{'mean':float(np.mean([r[k] for r in rows])),'q025':float(np.quantile([r[k] for r in rows],.025)),'q975':float(np.quantile([r[k] for r in rows],.975))} for k in keys}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--seed',type=int,default=234);a=p.parse_args();torch.manual_seed(a.seed);np.random.seed(a.seed);m=json.load(open(a.manifest));out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False) if not out.exists() else (_ for _ in ()).throw(RuntimeError('fresh out-dir required'))
 original_gt_arrays=hdata.gt_arrays
 def guarded_gt_arrays(corpus,split):
  if split=='train':raise RuntimeError('protocol violation: train temporal GT accessor forbidden')
  return original_gt_arrays(corpus,split)
 hdata.gt_arrays=guarded_gt_arrays
 corpus=m['corpus'];store=FrozenFeatures(Path.cwd(),corpus);train_ids=list(frozen_splits(corpus)['train']);labels,_=scoped_labels(corpus,'train');val_raw,vg,_=load_split_exact(m,'val');val_ids=sorted(vg);basev,v10_path=frozen_v10_identity(m,val_raw);basev={v:x[:,0] for v,x in basev.items()};base_metric=metrics(basev,vg)
 model=EmissionMIL(DIMS);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);history=[];best=None
 # Epoch zero is the exact immutable identity candidate.
 for epoch in range(EPOCHS+1):
  if epoch:
   model.train();losses=[]
   for v in np.random.default_rng(a.seed+epoch).permutation(train_ids):
    x,_,_=store.load(v);logit=model(x);loss=exact_bag_nll(logit,bool(labels[v]));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
  else:losses=[]
  residual,audit=infer(model,store,val_ids);rows=[]
  for lam in LAMBDAS:
   mm=metrics(add(basev,residual,lam),vg);rows.append({'lambda':lam,**mm,'eligible':bool(mm['frame_ap']+1e-12>=base_metric['frame_ap'] and mm['frame_roc']+1e-12>=base_metric['frame_roc'])})
  eligible=[r for r in rows if r['eligible']];chosen=max(eligible,key=lambda r:(r['frame_ap'],r['frame_roc'],-r['lambda'])) if eligible else next(r for r in rows if r['lambda']==0);entry={'epoch':epoch,'mean_exact_bag_nll':None if not losses else float(np.mean(losses)),'lambda_rows':rows,'chosen':chosen};history.append(entry);key=(chosen['frame_ap'],chosen['frame_roc'],-chosen['lambda'],-epoch)
  if best is None or key>best[0]:best=(key,epoch,chosen['lambda'],copy.deepcopy(model.state_dict()),audit)
 torch.save(best[3],out/'model.pth');frozen={'method':'ncas_mil_stage1_cardinality_corrected','corpus':corpus,'seed':a.seed,'frozen_encoders':['CLIP-B16 1fps','VGGish 1s','I3D RGB 5crop'],'train_temporal_gt_accessor':'runtime monkeypatch forbids split=train','train_objective':'exact independent latent-mixture bag likelihood with float64 log-domain accumulation and -log(T) rare-state prior: negative all-background; positive at least one latent positive frame','positive_bag':'latent mixture; no top-k or fixed length; permutation invariant','missing_modality':'masked/excluded from emission average; never interpreted as negative','k16_vera_observation':{'enabled':False,'reason':'stage1 unified evidence incomplete; missing observation is not negative'},'heads':'modality-specific linear density-ratio/emission logits','epochs':EPOCHS,'lr':1e-3,'lambda_grid':LAMBDAS,'selection':'max validation pooled Frame AP, ROC tie-break among AP and ROC noninferior to immutable V10; lambda0 exact fallback','selected_epoch':best[1],'selected_lambda':best[2],'history':history,'selected_validation_missing_audit':best[4],'external_v10':str(v10_path.resolve()),'external_v10_sha256':sha256(v10_path),'manifest':str(Path(a.manifest).resolve()),'manifest_sha256':sha256(a.manifest),'test_opened':False,'test_sources_or_gt_loaded_before_freeze':False};atomic_json(out/'frozen_config.json',frozen)
 # Test source and temporal GT are intentionally unopened until the complete
 # validation-selected configuration above has been atomically frozen.
 test_raw,tg,_=load_split_exact(m,'test');test_ids=sorted(tg);model.load_state_dict(best[3]);rt,test_audit=infer(model,store,test_ids);baset,_=frozen_v10_identity(m,test_raw);baset={v:x[:,0] for v,x in baset.items()};pred=add(baset,rt,best[2]);shuf=add(baset,shuffle(rt),best[2]);test={'selected':metrics(pred,tg),'identity':metrics(baset,tg),'time_shuffle':metrics(shuf,tg)};controls=shuffle_controls(baset,{v:best[2]*x for v,x in rt.items()},tg,200);payload={'method':frozen['method'],'corpus':corpus,'selected_epoch':best[1],'selected_lambda':best[2],'test':test,'time_shuffle_controls':controls,'delta_vs_v10':{k:test['selected'][k]-test['identity'][k] for k in ('frame_ap','frame_roc','within_centered_ap','within_centered_roc','within_macro_ap','within_macro_roc')},'test_missing_audit':test_audit,'test_labels_used_for_selection':False,'frozen_config':str((out/'frozen_config.json').resolve()),'frozen_config_sha256':sha256(out/'frozen_config.json')};atomic_json(out/'test_eval.json',payload);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
