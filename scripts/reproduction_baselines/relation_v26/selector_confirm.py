#!/usr/bin/env python3
"""Conditional three-seed shared-epoch selector; never opens frame labels."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from artifacts import atomic,sha
from core import DESIGN_SHA,SEEDS
def run(predictions,val_labels,out):
 if len(predictions)!=3 or len(val_labels)!=3:raise RuntimeError('three matched seeds required')
 per={};ident=None
 for pp,ll in zip(predictions,val_labels):
  pm=json.load(open(pp));lm=json.load(open(ll));seed=pm['seed']
  if seed not in SEEDS or lm.get('prediction_manifest_sha256')!=sha(pp) or lm.get('temporal_labels_read') is not False:raise RuntimeError('seed/label binding')
  ids=sorted(lm['ids']);y=np.array([lm['labels'][v] for v in ids]);cur=[]
  for ep in range(9):
   f=pm['files']['real'][str(ep)]
   if sha(f['path'])!=f['sha256']:raise RuntimeError('prediction tamper')
   r=json.load(open(f['path']))['records'];s=[r[v]['video_logit'] for v in ids];cur.append({'ap':float(average_precision_score(y,s)),'roc':float(roc_auc_score(y,s))})
  per[str(seed)]=cur;ident=(ids,y.tolist()) if ident is None else ident
 if set(map(int,per))!=set(SEEDS):raise RuntimeError('seed coverage')
 mean=[{'epoch':e,'ap':float(np.mean([per[str(s)][e]['ap'] for s in SEEDS])),'roc':float(np.mean([per[str(s)][e]['roc'] for s in SEEDS]))} for e in range(9)];chosen=max(mean,key=lambda x:(x['ap'],x['roc'],-x['epoch']))
 x={'schema':'v26_three_seed_selection_v1','design_sha256':DESIGN_SHA,'seeds':list(SEEDS),'selected_epoch':chosen['epoch'],'selection_rule':'mean_seed_(AP,ROC,-epoch)','mean_metrics':mean,'seed_metrics':per,'prediction_hashes':[sha(p) for p in predictions],'status':'PENDING_TEMPORAL_CONFIRMATION','test_authorized':False};atomic(out,x);return x
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',nargs=3,required=True);p.add_argument('--val-labels',nargs=3,required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.predictions,a.val_labels,a.out)
if __name__=='__main__':main()
