#!/usr/bin/env python3
"""Label-blind inference; test split requires a frozen selector artifact."""
import argparse,json,math,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import V24
from train import load_id_manifest,load_bags,load_global_source,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--frozen-config',required=True);p.add_argument('--bags',required=True);p.add_argument('--id-manifest',required=True);p.add_argument('--v23-global-source',required=True);p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--out-dir',required=True);a=p.parse_args();f=json.load(open(a.frozen_config))
 if a.split=='test' and (f.get('status')!='FINAL_PASS' or not f.get('temporal_steward_gate_signed')):raise RuntimeError('test forbidden before signed FINAL_PASS')
 im=load_id_manifest(a.id_manifest,f['corpus'],a.split)
 if im['producer_sha256']!=f['producer_sha256']:raise RuntimeError('inference producer identity mismatch')
 bags=load_bags(a.bags,im['ids'],f['corpus'],a.split,im['v23_global_source_sha256'],False);gs=load_global_source(a.v23_global_source,im['ids'],im['v23_global_source_sha256'])
 if any(bags[v]['global']!=gs[v] for v in gs):raise RuntimeError('inference global source mismatch')
 m=V24();m.load_state_dict({k:torch.tensor(v,dtype=torch.float64) for k,v in f['selected_state'].items()});out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False);rows=[]
 with torch.no_grad():
  for vid,b in sorted(bags.items()):
   z,s=m(b['global'],b['families']);ss=[float(x) for x in s]
   if not all(math.isfinite(x) for x in ss+[float(z)]):raise RuntimeError('nonfinite inference')
   if f['selected_epoch']==0 and (float(z)!=b['global'] or any(x!=b['global'] for x in ss)):raise RuntimeError('epoch0 not exact source fallback')
   rows.append({'video_id':vid,'video_score':float(z),'window_scores':ss})
 rp=out/'predictions.jsonl';rp.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows));(out/'raw_manifest.json').write_text(json.dumps({'raw_frozen_before_labels':True,'split':a.split,'n_videos':len(rows),'predictions_sha256':sha(rp),'frozen_config_sha256':sha(a.frozen_config),'bags_sha256':sha(a.bags),'id_manifest_sha256':sha(a.id_manifest)},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
