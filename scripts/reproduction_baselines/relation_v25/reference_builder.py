#!/usr/bin/env python3
"""Build immutable full and five-fold train-negative references."""
import argparse,json
from pathlib import Path
import numpy as np
from core import sha,canon_hash,fold
FAMS=('text','multimodal')
def load_bags(path,split='train'):
 out=[]
 for line in open(path):
  r=json.loads(line)
  if r.get('split')!=split or r.get('video_label') not in (0,1) or set(r.get('families',{}))!=set(FAMS):raise RuntimeError('invalid weak bag')
  if not isinstance(r.get('global_causal_score'),(int,float)) or not np.isfinite(r['global_causal_score']):raise RuntimeError('nonfinite G')
  xs=[]
  for f in FAMS:
   q=r['families'][f]
   if not isinstance(q,list) or len(q)!=1 or not isinstance(q[0],list) or not q[0] or not np.isfinite(q[0]).all():raise RuntimeError('invalid/nonfinite family')
   xs.append(len(q[0]))
  if len(set(xs))!=1:raise RuntimeError('family length mismatch')
  out.append(r)
 if len({r['video_id'] for r in out})!=len(out):raise RuntimeError('duplicate IDs')
 return out
def build(path,out):
 bags=load_bags(path);neg=[r for r in bags if r['video_label']==0];root=Path(out);root.mkdir(parents=True,exist_ok=False);files={}
 for f in FAMS:
  full=sorted(x for r in neg for x in r['families'][f][0]);p=root/f'{f}_full.json';p.write_text(json.dumps(full)+'\n');files[p.name]=sha(p)
  for k in range(5):
   held=[r for r in neg if fold(r['video_id'])==k];ref=sorted(x for r in neg if fold(r['video_id'])!=k for x in r['families'][f][0])
   if not held or not ref or not np.isfinite(ref).all():raise RuntimeError('empty/nonfinite crossfit negative fold')
   p=root/f'{f}_exclude_fold{k}.json';p.write_text(json.dumps(ref)+'\n');files[p.name]=sha(p)
 manifest={'schema':'v25_reference_v1','source_bags_sha256':sha(path),'families':list(FAMS),'fold_rule':'sha256-prefix8-mod5','files':files,'full_reference_materializations':1,'producer_sha256':sha(__file__)};manifest['aggregate_sha256']=canon_hash(files);(root/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return manifest
def verify(root,source=None):
 root=Path(root);m=json.load(open(root/'manifest.json'));keys={'schema','source_bags_sha256','families','fold_rule','files','full_reference_materializations','producer_sha256','aggregate_sha256'}
 if set(m)!=keys or m['schema']!='v25_reference_v1' or m['aggregate_sha256']!=canon_hash(m['files']) or m['full_reference_materializations']!=1:raise RuntimeError('reference manifest')
 if m['producer_sha256']!=sha(__file__):raise RuntimeError('reference producer identity')
 if source and sha(source)!=m['source_bags_sha256']:raise RuntimeError('source tamper')
 for n,h in m['files'].items():
  if sha(root/n)!=h:raise RuntimeError('reference tamper')
  x=json.load(open(root/n));
  if not x or not np.isfinite(x).all() or x!=sorted(x):raise RuntimeError('invalid reference')
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--bags',required=True);p.add_argument('--out',required=True);a=p.parse_args();build(a.bags,a.out)
if __name__=='__main__':main()
