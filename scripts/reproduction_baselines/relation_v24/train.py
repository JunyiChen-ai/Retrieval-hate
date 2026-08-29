#!/usr/bin/env python3
"""Fail-closed fixed seed234/5-epoch V24 trainer; video labels only."""
import argparse,hashlib,json,math,random
from pathlib import Path
import torch
from model import V24
ROW_KEYS={'corpus','split','video_id','video_label','global_causal_score','families','source_hashes'};INFER_ROW_KEYS=ROW_KEYS-{'video_label'};FAMILIES={'text','multimodal'};SOURCE_KEYS={'text_scores_sha256','multimodal_scores_sha256','v23_global_source_sha256'}
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sha_ids(ids):return hashlib.sha256(''.join(x+'\n' for x in sorted(ids)).encode()).hexdigest()
def corpus_sha(corpus):return hashlib.sha256((corpus+'\n').encode()).hexdigest()
def load_id_manifest(path,corpus,split):
 m=json.load(open(path));allowed={'corpus','split','ids','v23_global_source_sha256','producer_sha256'}
 if set(m)!=allowed or m['corpus']!=corpus or m['split']!=split:raise RuntimeError('invalid exact ID manifest schema/identity')
 ids=m['ids']
 if not ids or len(ids)!=len(set(ids)) or any(not isinstance(x,str) for x in ids):raise RuntimeError('invalid IDs')
 for k in ('v23_global_source_sha256','producer_sha256'):
  if not isinstance(m[k],str) or len(m[k])!=64:raise RuntimeError('invalid source hash')
 return m
def _finite_vector(x):return isinstance(x,list) and len(x)>0 and all(isinstance(v,(int,float)) and math.isfinite(float(v)) for v in x)
def load_global_source(path,ids,expected_sha):
 if sha(path)!=expected_sha:raise RuntimeError('V23 global values source hash mismatch')
 out={}
 with open(path) as fh:
  for line in fh:
   r=json.loads(line)
   if set(r)!={'video_id','global_causal_score'}:raise RuntimeError('global source exact schema violation')
   if r['video_id'] in out or not isinstance(r['global_causal_score'],(int,float)) or not math.isfinite(float(r['global_causal_score'])):raise RuntimeError('invalid global source row')
   out[r['video_id']]=float(r['global_causal_score'])
 if set(out)!=set(ids):raise RuntimeError('global source coverage mismatch')
 return out
def load_bags(path,expected_ids,corpus,split,source_global_sha,require_labels=True):
 expected=set(expected_ids);out={};allowed=ROW_KEYS if require_labels else INFER_ROW_KEYS
 with open(path) as fh:
  for n,line in enumerate(fh,1):
   r=json.loads(line)
   if set(r)!=allowed:raise RuntimeError(f'row {n} violates exact allowlist: {sorted(set(r)^allowed)}')
   if r['corpus']!=corpus or r['split']!=split:raise RuntimeError('cross-corpus/split row')
   vid=r['video_id']
   if vid not in expected or vid in out:raise RuntimeError('non-scoped or duplicate video')
   if require_labels and r['video_label'] not in (0,1):raise RuntimeError('binary video label required')
   if not isinstance(r['global_causal_score'],(int,float)) or not math.isfinite(float(r['global_causal_score'])):raise RuntimeError('nonfinite global')
   if set(r['families'])!=FAMILIES or set(r['source_hashes'])!=SOURCE_KEYS:raise RuntimeError('family/source schema mismatch')
   if r['source_hashes']['v23_global_source_sha256']!=source_global_sha:raise RuntimeError('V23 global source mismatch')
   if any(not isinstance(r['source_hashes'][k],str) or len(r['source_hashes'][k])!=64 for k in SOURCE_KEYS):raise RuntimeError('invalid source hash')
   lengths=[]
   for fam in sorted(FAMILIES):
    channels=r['families'][fam]
    if not isinstance(channels,list) or not channels or not all(_finite_vector(x) for x in channels):raise RuntimeError('invalid family channels')
    lengths.extend(map(len,channels))
   if len(set(lengths))!=1:raise RuntimeError('unaligned family/window lengths')
   out[vid]={'global':float(r['global_causal_score']),'families':r['families'],'label':int(r['video_label']) if require_labels else None,'source_hashes':r['source_hashes']}
 if set(out)!=expected:raise RuntimeError('exact full scoped ID coverage required')
 return out
def state(model):return {k:v.detach().clone() for k,v in model.state_dict().items()}
def train(bags,out,mode='real'):
 random.seed(234);torch.manual_seed(234);model=V24();params=list(model.parameters()) if mode!='global_only' else [model.global_delta,model.global_bias];opt=torch.optim.Adam(params,lr=1e-2);lossfn=torch.nn.BCEWithLogitsLoss();history=[]
 for epoch in range(6):
  if epoch==0:
   for b in bags.values():
    z,f=model(b['global'],b['families']);g=torch.tensor(b['global'],dtype=torch.float64)
    if not torch.equal(z,g) or not torch.equal(f,torch.full_like(f,g)):raise RuntimeError('epoch0 not exact V23 global fallback')
  else:
   order=sorted(bags);random.Random(234+epoch).shuffle(order)
   for vid in order:
    b=bags[vid];opt.zero_grad();z,_=model(b['global'],b['families']);loss=lossfn(z.reshape(1),torch.tensor([b['label']],dtype=torch.float64));loss.backward();opt.step();model.project_()
   if mode=='global_only':
    with torch.no_grad():model.gamma.zero_();model.family_logits.zero_()
  history.append({'epoch':epoch,'state':state(model)})
 torch.save({'seed':234,'epochs':5,'mode':mode,'history':history},out)
def negative_control(bags):
 buckets={}
 for vid,b in bags.items():
  n=len(next(iter(b['families'].values()))[0]);bucket=1 if n<=1 else 2**round(math.log2(n));buckets.setdefault(bucket,[]).append(vid)
 out={v:{**b,'families':{k:[list(x) for x in vv] for k,vv in b['families'].items()}} for v,b in bags.items()};rng=random.Random(24024)
 for ids in buckets.values():
  ids=sorted(ids);donors=ids[:];rng.shuffle(donors)
  if len(ids)>1 and donors==ids:donors=donors[1:]+donors[:1]
  for target,source in zip(ids,donors):out[target]['families']=bags[source]['families']
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--bags',required=True);p.add_argument('--id-manifest',required=True);p.add_argument('--producer',required=True);p.add_argument('--v23-global-source',required=True);p.add_argument('--corpus',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args();m=load_id_manifest(a.id_manifest,a.corpus,'train')
 if sha(a.producer)!=m['producer_sha256']:raise RuntimeError('producer hash mismatch')
 bags=load_bags(a.bags,m['ids'],a.corpus,'train',m['v23_global_source_sha256'],True);gs=load_global_source(a.v23_global_source,m['ids'],m['v23_global_source_sha256'])
 if any(bags[v]['global']!=gs[v] for v in gs):raise RuntimeError('bag global differs elementwise from frozen V23 source')
 out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False);train(bags,out/'real_local.pt','real');train(negative_control(bags),out/'permuted_local_negative_control.pt','permuted');train(bags,out/'matched_global_only.pt','global_only')
 protocol={'status':'TRAINED_NO_VAL_OR_TEST_ACCESS','corpus':a.corpus,'corpus_sha256':corpus_sha(a.corpus),'seed':234,'epochs':5,'bags':str(Path(a.bags).resolve()),'bags_sha256':sha(a.bags),'id_manifest':str(Path(a.id_manifest).resolve()),'id_manifest_sha256':sha(a.id_manifest),'producer':str(Path(a.producer).resolve()),'producer_sha256':sha(a.producer),'train_ids_sha256':sha_ids(m['ids']),'v23_global_source':str(Path(a.v23_global_source).resolve()),'v23_global_source_sha256':m['v23_global_source_sha256'],'epoch0_elementwise_source_check':True}
 (out/'train_protocol.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
