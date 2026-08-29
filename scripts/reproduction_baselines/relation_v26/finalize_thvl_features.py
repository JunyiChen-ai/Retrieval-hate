#!/usr/bin/env python3
import argparse,json,hashlib,os,tempfile
from pathlib import Path
import numpy as np
from core import DESIGN_SHA
from feature_manifest import MODELS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atom(p,s):
 p=Path(p);fd,t=tempfile.mkstemp(dir=p.parent,prefix=p.name+'.')
 try:
  with os.fdopen(fd,'w') as f:f.write(s);f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def atom_npz(p,**x):
 p=Path(p);fd,t=tempfile.mkstemp(dir=p.parent,prefix=p.name+'.',suffix='.npz');os.close(fd)
 try:
  np.savez(t,**x)
  with open(t,'rb') as f:os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=('train','val'),required=True);p.add_argument('--raw',required=True);p.add_argument('--raw-seal',required=True);p.add_argument('--qc-dir',required=True);p.add_argument('--asr-dir',required=True);p.add_argument('--g-jsonl',required=True);p.add_argument('--split-manifest',required=True);p.add_argument('--train-moments');p.add_argument('--out',required=True);a=p.parse_args();raw=Path(a.raw)
 from raw_verifier import verify as verify_raw,sha as rsha,ids_hash
 rm,_=verify_raw(raw,a.split,a.split_manifest,a.qc_dir,a.asr_dir,a.g_jsonl);seal=json.load(open(a.raw_seal))
 sm=json.load(open(a.split_manifest));ids=sorted(sm['ids'])
 sk={'schema','split','raw_path','raw_manifest_sha256','raw_root_sha256','split_manifest_path','split_manifest_sha256','g_source_sha256','ids_sha256','n','producer_sha256','verifier_sha256'}
 if set(seal)!=sk or seal['schema']!='v26_raw_seal_v2' or seal['split']!=a.split or seal['raw_path']!=str(raw.resolve()) or seal['raw_manifest_sha256']!=rsha(raw/'manifest.json') or seal['raw_root_sha256']!=rm['root_sha256'] or seal['split_manifest_path']!=str(Path(a.split_manifest).resolve()) or seal['split_manifest_sha256']!=rsha(a.split_manifest) or seal['g_source_sha256']!=rsha(a.g_jsonl) or seal['ids_sha256']!=ids_hash(ids) or seal['n']!=len(ids) or seal['producer_sha256']!=rsha(Path(__file__).with_name('extract_thvl_1hz.py')) or seal['verifier_sha256']!=rsha(Path(__file__).with_name('raw_verifier.py')):raise RuntimeError('raw seal')
 if rm['split']!=a.split or rm['design_sha256']!=DESIGN_SHA or rm['n']!=len(ids):raise RuntimeError('raw/full coverage')
 arr={v:np.load(raw/(v+'.npz')) for v in ids};dims={'visual':512,'audio':128,'text':768}
 if a.split=='train':
  mom={}
  for fi,f in enumerate(dims):
   z=np.concatenate([x[f][x['availability'][:,fi].astype(bool)] for x in arr.values()]);mom[f]={'mean':z.mean(0),'std':np.maximum(z.std(0),1e-6)}
  mp=Path(a.out)/'train_moments.npz';Path(a.out).mkdir(parents=True,exist_ok=False);atom_npz(mp,**{f+'_'+k:v for f,q in mom.items() for k,v in q.items()})
 else:
  if not a.train_moments:raise RuntimeError('val requires frozen train moments')
  Path(a.out).mkdir(parents=True,exist_ok=False);mp=Path(a.train_moments);z=np.load(mp);mom={f:{'mean':z[f+'_mean'],'std':z[f+'_std']} for f in dims}
 rec={};rd=Path(a.out)/'records';rd.mkdir()
 for v,x in arr.items():
  T=len(x['availability']);secs=[]
  for j in range(T):
   av=x['availability'][j].astype(int).tolist();row={'second':j,'availability':av}
   for fi,f in enumerate(dims):row[f]=(((x[f][j]-mom[f]['mean'])/mom[f]['std']).astype(float).tolist() if av[fi] else [])
   if not any(av):raise RuntimeError('T_eff coverage')
   secs.append(row)
  r={'corpus':'thvl','split':a.split,'opaque_id':v,'duration':float(x['duration']),'G':float(x['G']),'G_domain':'signed_logit','seconds':secs,'source_hashes':{'raw_npz_sha256':sha(raw/(v+'.npz')),'finalizer_sha256':sha(__file__)}};rp=rd/(v+'.json');atom(rp,json.dumps(r,sort_keys=True)+'\n');rec[v]=str(rp.resolve())
 norm={'path':str(mp.resolve()),'sha256':sha(mp),'fit_split':'train'};root=hashlib.sha256((rm['root_sha256']+sha(a.raw_seal)+sha(__file__)+sha(a.split_manifest)+norm['sha256']+''.join(v+'\t'+sha(rec[v])+'\n' for v in ids)).encode()).hexdigest();man={'schema':'v26_features_v1','design_sha256':DESIGN_SHA,'corpus':'thvl','split':a.split,'ids':ids,'records':rec,'models':MODELS,'normalization':norm,'G_binding':{'domain':'signed_logit','source_sha256':rm['g_source_sha256'],'raw_root_sha256':rm['root_sha256'],'raw_seal_sha256':sha(a.raw_seal),'split_manifest_sha256':sha(a.split_manifest),'finalizer_sha256':sha(__file__)},'root_sha256':root,'labels_or_gt_read':False};atom(Path(a.out)/'manifest.json',json.dumps(man,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
