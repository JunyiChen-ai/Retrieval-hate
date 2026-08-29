#!/usr/bin/env python3
"""Five-epoch HateMM weak-label MACIL-AV locator residual pilot."""
import argparse,copy,json
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from frame_eval_common import evaluate
from relation_v2.protocol import scoped_labels
from relation_v4.io import apply_ecdf
from relation_v8.model import UnifiedRelationV8
from relation_v10.diagnostic import components,score
from relation_v9.train import load

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default='results/reproduction/relation_v9/manifests/hatemm_macil_vera.json');p.add_argument('--out-dir',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=234);a=p.parse_args()
 torch.manual_seed(a.seed);np.random.seed(a.seed);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise RuntimeError('fresh out-dir required')
 manifest=json.loads(Path(a.manifest).read_text());cfg=json.loads(Path('results/reproduction/relation_v10/diagnostic_stable/hatemm/frozen_config.json').read_text());refs=cfg['calibration'];chosen=cfg['identity_v8_fallback']
 model=UnifiedRelationV8(3,manifest.get('window',12),manifest.get('temperature',.2)).eval()
 data={};provenance={}
 for split in ('train','val','test'):
  ids,raw,prov=load(manifest,split);cal=apply_ecdf(raw,refs);parts=components(model,cal);base=score(parts,np.full(3,1/3),chosen['beta'],chosen['gamma']);loc={v:cal[v][:,0]-cal[v][:,0].mean() for v in ids};data[split]=(ids,base,loc);provenance[split]=prov
 labels,label_path=scoped_labels('hatemm','train');alpha=torch.nn.Parameter(torch.zeros((),device=a.device));opt=torch.optim.Adam([alpha],lr=.05);gt=__import__('hate_common.data',fromlist=['data']).gt_arrays('hatemm','val')
 def valrow(epoch):
  per={}
  for v in data['val'][0]:
   b=torch.tensor(data['val'][1][v],device=a.device);l=torch.tensor(data['val'][2][v],device=a.device);s=(b+alpha*l).detach().cpu().numpy();per[v]=(s,gt[v])
  m=evaluate(per);return {'epoch':epoch,'alpha':float(alpha.detach()),'validation_frame_ap':m['pr_auc'],'validation_frame_roc':m['roc_auc']}
 history=[valrow(0)];states=[float(alpha.detach())]
 for epoch in range(1,6):
  order=np.random.permutation(data['train'][0]);losses=[]
  for v in order:
   b=torch.tensor(data['train'][1][v],device=a.device);l=torch.tensor(data['train'][2][v],device=a.device);frame=(b+alpha*l).clamp(1e-5,1-1e-5);k=max(1,len(frame)//8);video=frame.topk(k).values.mean();target=torch.tensor(float(labels[v]),device=a.device);loss=F.binary_cross_entropy(video,target);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
  row=valrow(epoch);row['train_weak_bce']=float(np.mean(losses));history.append(row);states.append(float(alpha.detach()))
 best=max(range(6),key=lambda i:(history[i]['validation_frame_ap'],history[i]['validation_frame_roc']));selected_alpha=states[best]
 # Test is a development diagnostic after the validation-selected epoch freezes.
 test_gt=__import__('hate_common.data',fromlist=['data']).gt_arrays('hatemm','test');per={};score_path=out/'scores.jsonl'
 with score_path.open('w') as handle:
  for v in data['test'][0]:
   base=data['test'][1][v];locator=data['test'][2][v];final=base+selected_alpha*locator
   per[v]=(final,test_gt[v]);handle.write(json.dumps({'video_id':v,'score_epoch0_identity':base.tolist(),'score_weak_residual':final.tolist(),'locator_macilsd_av':locator.tolist()})+'\n')
 metric=evaluate(per);payload={'method':'hatemm_weak_label_macil_av_residual_short_pilot','seed':a.seed,'epochs':5,'no_sweep':True,'epoch0_exact_identity':states[0]==0.,'selected_epoch':best,'selected_alpha':selected_alpha,'selected_by':'validation Frame AP; ROC tie-break','history':history,'test':{'frame_ap':metric['pr_auc'],'frame_roc':metric['roc_auc']},'test_informed_development_diagnostic':True,'test_used_for_training_or_checkpoint_selection':False,'weak_train_labels':str(Path(label_path).resolve()),'sources':provenance}
 (out/'pilot.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
