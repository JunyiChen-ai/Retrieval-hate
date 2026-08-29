#!/usr/bin/env python3
import argparse,hashlib,json,math,random
from pathlib import Path
import numpy as np,torch
from artifacts import atomic,sha,verify_manifest
from core import CTW,Probe,DESIGN_SHA,MIGRATION_SHA,ARCH
from reference import verify_reference
from train import load_rows
from train_control_v3 import CKPT_SCHEMA, CONTROL_SCHEMA, source_sha as trainer_sha, verify_train_run
def model_from(path,arm,epoch,seed,device):
 c=torch.load(path,map_location='cpu',weights_only=False)
 if set(c)!={'schema','design_sha256','migration_sha256','architecture','control_schema','trainer_sha256','seed','arm','epoch','steps','inputs','state'} or c['schema']!=CKPT_SCHEMA or c['control_schema']!=CONTROL_SCHEMA or c['trainer_sha256']!=trainer_sha() or (c['design_sha256'],c['migration_sha256'],c['architecture'])!=(DESIGN_SHA,MIGRATION_SHA,ARCH) or (c['arm'],c['epoch'],c['seed'])!=(arm,epoch,seed):raise RuntimeError('checkpoint')
 m=CTW(model_seed=seed);m.load_state_dict(c['state']);m.to(device).eval();return m
def run(features,reference,train_manifest,out):
 rows,_=load_rows(features);tm=verify_train_run(train_manifest);ref=verify_reference(reference,tm['features']['path'],tm['labels']['path'],features);seed=tm['seed'];pc=torch.load(tm['probe']['path'],map_location='cpu',weights_only=False)
 if sha(tm['probe']['path'])!=tm['probe']['sha256'] or pc['arm']!='probe' or pc['seed']!=26027:raise RuntimeError('probe')
 if not torch.cuda.is_available() or 'RTX 5090' not in torch.cuda.get_device_name(0):raise RuntimeError('V26 validation requires RTX 5090')
 device=torch.device('cuda');probe=Probe();probe.load_state_dict(pc['state']);probe.to(device).eval()
 backgrounds={}
 for r in rows:
  e=ref['val_backgrounds'][r['id']]
  if sha(e['path'])!=e['sha256']:raise RuntimeError('val background tamper')
  backgrounds[r['id']]=[x.to(device) for x in torch.load(e['path'],map_location='cpu',weights_only=True)]
  r['X']=[x.to(device) for x in r['X']];r['masks']=[x.to(device) for x in r['masks']];r['G']=r['G'].to(device)
 out=Path(out);out.mkdir(parents=True,exist_ok=False);files={}
 for arm in ('real','permuted','negative_mean'):
  files[arm]={}
  for ep,c in enumerate(tm['arms'][arm]):
   if sha(c['path'])!=c['sha256']:raise RuntimeError('checkpoint tamper')
   m=model_from(c['path'],arm,ep,seed,device);pred={}
   for r in rows:
    b=backgrounds[r['id']] if arm!='negative_mean' else [torch.zeros_like(x) for x in r['X']]
    with torch.no_grad():logit=m(r['X'],r['masks'],r['G']);e=torch.clamp(m.effects(r['X'],r['masks'],b,r['G']),-12,12);local=torch.sigmoid(e)
    sh=[]
    for k in range(100):
     rng=np.random.default_rng(26030000+1000*k+seed);sh.append(local[rng.permutation(len(local))].tolist())
    te=sum(bool(torch.stack(r['masks'],1).any(1)[t]) for t in range(r['T']));k=max(1,math.ceil(.2*te));starts=range(max(1,r['T']-k+1));top=max(starts,key=lambda s:float(e[s:s+k].sum()));base=float(probe(r['X'],r['masks']));
    def drop(s):
     xx=[x.clone() for x in r['X']]
     for t in range(s,min(s+k,r['T'])):
      for f in range(3):
       if r['masks'][f][t]:xx[f][t]=b[f][t]
     return base-float(probe(xx,r['masks']))
    rng=np.random.default_rng(26028+int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16));random_starts=rng.integers(0,max(1,r['T']-k+1),100);faith={'k':k,'top_start':top,'drop_top':drop(top),'drop_random100':[drop(int(s)) for s in random_starts]}
    pred[r['id']]={'G':float(r['G']),'video_logit':float(logit),'local':local.tolist(),'effects':e.tolist(),'shuffle100':sh,'faithfulness':faith,'duration':r['T'],'epoch0_G_exact':bool(ep!=0 or float(logit)==float(r['G']))}
   p=out/f'{arm}.epoch{ep}.json';atomic(p,{'schema':'v26_val_predictions_v1','design_sha256':DESIGN_SHA,'arm':arm,'epoch':ep,'seed':seed,'records':pred,'labels_or_gt_read':False});files[arm][str(ep)]={'path':str(p.resolve()),'sha256':sha(p)}
 man={'schema':'v26_val_prediction_manifest_v1','design_sha256':DESIGN_SHA,'seed':seed,'features':{'path':str(Path(features).resolve()),'sha256':sha(features)},'reference':{'path':str(Path(reference).resolve()),'sha256':sha(reference)},'train_run':{'path':str(Path(train_manifest).resolve()),'sha256':sha(train_manifest)},'files':files,'source_sha256':sha(__file__),'test_read':False};atomic(out/'manifest.json',man);return man
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--reference',required=True);p.add_argument('--train-manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.features,a.reference,a.train_manifest,a.out)
if __name__=='__main__':main()
