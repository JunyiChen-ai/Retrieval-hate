#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(HERE.parent),str(ROOT/'scripts/duplex')]
from hate_common import data as hdata
from relation_v4.io import sha256
from relation_v8.run import atomic_json
def auc(y,s):return float(roc_auc_score(y,s)) if len(set(y))==2 else None
def evaluate(rows,gt,key):
 rows=[r for r in rows if r.get('temporal_span_valid',True) and r['video_id'] in gt and r['start']<len(gt[r['video_id']]) and r['end']>0]
 for r in rows:
  y=gt[r['video_id']];a=min(len(y)-1,max(0,int(np.floor(r['start']))));b=min(len(y),max(a+1,int(np.ceil(r['end']))));r['_gold']=int(y[a:b].max()>0);r['_video_positive']=int(y.max()>0)
 pooled=auc([r['_gold'] for r in rows],[r['scores'][key] for r in rows]);mac=[]
 grouped={}
 for r in rows:grouped.setdefault(r['video_id'],[]).append(r)
 for v,q in sorted(grouped.items()):
  z=auc([x['_gold'] for x in q],[x['scores'][key] for x in q]);
  if z is not None:mac.append(z)
 pos=[r for r in rows if r['_gold']];neg=[r for r in rows if not r['_video_positive']];cross=auc([1]*len(pos)+[0]*len(neg),[r['scores'][key] for r in pos+neg]);return {'pooled_auc':pooled,'within_macro_auc':float(np.mean(mac)) if mac else None,'within_macro_n':len(mac),'cross_video_auc':cross}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);a=p.parse_args();out=Path(a.out_dir);cfg=json.load(open(out/'preregistered_config.json'));man=json.load(open(out/'raw_manifest.json'));raw=out/'per_chunk_raw.jsonl';assert man['raw_sha256']==sha256(raw) and man['raw_frozen_before_gt'];rows=list(map(json.loads,open(raw)));gt=hdata.gt_arrays(cfg['corpus'],'test');base={arm:evaluate([dict(r) for r in rows],gt,arm) for arm in cfg['arms']};rng=np.random.default_rng(1600);vids=sorted(set(r['video_id'] for r in rows if r['video_id'] in gt));boots={a:{k:[] for k in ('pooled_auc','within_macro_auc','cross_video_auc')} for a in cfg['arms']}
 by_video={v:[r for r in rows if r['video_id']==v] for v in vids}
 for _ in range(cfg['bootstrap_videos']):
  sample=rng.choice(vids,len(vids),replace=True);q=[]
  for j,v in enumerate(sample):
   for r in by_video[v]:q.append({**r,'video_id':f'{j}:{v}'})
  # GT aliases for resampled video identities.
  bg={f'{j}:{v}':gt[v] for j,v in enumerate(sample)}
  for arm in cfg['arms']:
   z=evaluate([dict(r) for r in q],bg,arm)
   for k in boots[arm]:
    if z[k] is not None:boots[arm][k].append(z[k])
 ci={a:{k:{'q025':float(np.quantile(x,.025)),'q975':float(np.quantile(x,.975))} for k,x in d.items()} for a,d in boots.items()};ref=[r for r in rows if r['sequential_reference'] is not None];fidelity={a:float(spearmanr([r['scores'][a] for r in ref],[r['sequential_reference'] for r in ref]).statistic) for a in cfg['arms']};report={'method':cfg['method'],'corpus':cfg['corpus'],'test_informed':True,'raw_manifest_sha256':sha256(out/'raw_manifest.json'),'metrics':base,'video_bootstrap_ci':ci,'sequential_subset_spearman':fidelity,'gt_opened_only_after_raw_hash_verified':True};atomic_json(out/'report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
