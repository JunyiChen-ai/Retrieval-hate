#!/usr/bin/env python3
"""Val-only evaluation. Verifies the frozen raw hash before opening temporal GT."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'scripts/reproduction_baselines')]
from hate_common import data as hdata
from relation_v4.io import sha256

def metrics(y,s):
    if len(set(y))<2:return None
    return {'roc':float(roc_auc_score(y,s)),'ap':float(average_precision_score(y,s))}

def attach(rows,gt):
    out=[]
    for r in rows:
        if r['video_id'] not in gt:continue
        y=gt[r['video_id']];a=max(0,min(len(y)-1,int(np.floor(r['start']))));b=max(a+1,min(len(y),int(np.ceil(r['end']))))
        out.append({**r,'gold':int(np.mean(y[a:b])>=.5)})
    return out

def evaluate(rows,key):
    pooled=metrics([r['gold'] for r in rows],[r['scores'][key] for r in rows]); by={}
    for r in rows:by.setdefault(r['video_id'],[]).append(r)
    ms=[metrics([x['gold'] for x in q],[x['scores'][key] for x in q]) for q in by.values()]
    ms=[x for x in ms if x]
    return {'pooled':pooled,'within_macro':{k:float(np.mean([x[k] for x in ms])) for k in ('roc','ap')},'eligible_videos':len(ms)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);a=p.parse_args();out=Path(a.out_dir)
 cfg=json.load(open(out/'preregistered_config.json'));man=json.load(open(out/'raw_manifest.json'));raw=out/'per_chunk_raw.jsonl'
 assert cfg['split']=='val' and cfg['test_access'] is False and man['raw_frozen_before_gt'] and sha256(raw)==man['raw_sha256']
 rows=attach(list(map(json.loads,open(raw))),hdata.gt_arrays(cfg['corpus'],'val')); arms=cfg['arms']
 base={k:evaluate(rows,k) for k in arms}; rng=np.random.default_rng(2100); shuffled={k:[] for k in arms}
 by={};
 for r in rows:by.setdefault(r['video_id'],[]).append(r)
 for _ in range(200):
  q=[]
  for vv in by.values():
   perms={k:rng.permutation([r['scores'][k] for r in vv]) for k in arms}
   for i,r in enumerate(vv):q.append({**r,'scores':{k:float(perms[k][i]) for k in arms}})
  for k in arms:shuffled[k].append(evaluate(q,k)['within_macro']['roc'])
 paired=[]
 for vid,vv in sorted(by.items()):
  ma=metrics([x['gold'] for x in vv],[x['scores']['asr_only'] for x in vv]);mm=metrics([x['gold'] for x in vv],[x['scores']['frame_ocr_asr'] for x in vv])
  if ma and mm:paired.append({k:mm[k]-ma[k] for k in ('roc','ap')})
 boot={k:[] for k in ('roc','ap')}
 if paired:
  for _ in range(2000):
   z=rng.choice(len(paired),len(paired),replace=True)
   for k in boot:boot[k].append(float(np.mean([paired[i][k] for i in z])))
 report={'method':cfg['method'],'corpus':cfg['corpus'],'split':'val','test_opened':False,'metrics':base,
         'shuffle_within_macro_roc':{k:{'mean':float(np.mean(v)),'q95':float(np.quantile(v,.95))} for k,v in shuffled.items()},
         'multimodal_minus_asr':{m:base['frame_ocr_asr']['within_macro'][m]-base['asr_only']['within_macro'][m] for m in ('roc','ap')},
         'paired_video_bootstrap_delta_ci':{k:{'mean':float(np.mean(v)),'q025':float(np.quantile(v,.025)),'q975':float(np.quantile(v,.975))} for k,v in boot.items()},
         'packed_sequential_fidelity':'not_applicable_sequential_prototype','v8_fallback_preserved':'lambda=0 is mandatory in subsequent val fusion'}
 (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
