#!/usr/bin/env python3
from __future__ import annotations
import dataclasses,hashlib,hmac,json,math,os
from pathlib import Path
from typing import Callable,Iterable
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score,roc_auc_score
FIXED_REMOTE_IDENTITY={'repository_url':'https://github.com/Multimodal-Intelligence-Lab-MIL/DeHate','commit':'8b3ecac98223ef953ad657b319cf90ffcff9ada1','path':'DeHate.xlsx','bytes':1267040,'git_blob_sha1':'1f959059c6f96d1d46e580743f8a1b7ac02ec0e3'}
QC_SCHEMA={"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","required":["hashed_id","probe_status","duration_seconds","decode_ok","audio_present","sha256"],"properties":{"hashed_id":{"type":"string","pattern":"^[0-9a-f]{64}$"},"probe_status":{"enum":["ok","missing","corrupt","unsupported"]},"duration_seconds":{"type":["number","null"],"minimum":0},"decode_ok":{"type":"boolean"},"audio_present":{"type":["boolean","null"]},"sha256":{"type":["string","null"],"pattern":"^[0-9a-f]{64}$"},"width":{"type":["integer","null"],"minimum":1},"height":{"type":["integer","null"],"minimum":1},"fps":{"type":["number","null"],"exclusiveMinimum":0}},"additionalProperties":False}
PUBLIC_ROW_KEYS={'hashed_id','duplicate_group_hash','split','media_qc_status'};PRIVATE_ROW_KEYS={'raw_id','hashed_id','internal_media_hash','canonical_group_internal_hash','duration_seconds','media_sha256','media_qc_status'}
def canonical_json(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha256_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def verify_remote_identity(observed,expected=None):
 expected=FIXED_REMOTE_IDENTITY if expected is None else expected
 if expected!=FIXED_REMOTE_IDENTITY:raise ValueError('expected identity must equal fixed release identity')
 if set(observed)!=set(expected):raise ValueError('remote identity keys must be exact')
 bad={k:(expected[k],observed[k]) for k in expected if observed[k]!=expected[k]}
 if bad:raise RuntimeError(f'remote identity mismatch: {bad}')
 return {'verified':True,'identity_sha256':hashlib.sha256(canonical_json(observed)).hexdigest(),'network_accessed':False}
def public_id(x,key):
 if not key:raise ValueError('private HMAC key required')
 return hmac.new(key,x.encode(),hashlib.sha256).hexdigest()
def exact_group_split(records,salt,ratios=(.7,.1,.2)):
 if ratios!=(.7,.1,.2):raise ValueError('frozen split exactly 70/10/20')
 ids=[r['raw_id'] for r in records]
 if len(ids)!=len(set(ids)):raise RuntimeError('duplicate raw ID')
 groups={}
 hash_to_group={}
 for r in records:
  if len(r['internal_media_hash'])!=64:raise ValueError('invalid internal SHA')
  ih=r['internal_media_hash'];g=r['duplicate_group']
  if ih in hash_to_group and hash_to_group[ih]!=g:raise RuntimeError('same internal_media_hash appears in multiple duplicate groups')
  hash_to_group[ih]=g
  groups.setdefault(r['duplicate_group'],[]).append(r['canonical_id'])
 group_keys={g:'\n'.join(sorted(v)) for g,v in groups.items()};canon={g:hashlib.sha256(group_keys[g].encode()).hexdigest() for g in groups};thresholds={'train':[0.,.7],'validation':[.7,.8],'test':[.8,1.]};gsp={};u_values={}
 for g in groups:
  digest=hashlib.sha256((salt+'\n'+group_keys[g]).encode()).digest();u=int.from_bytes(digest[:8],'big')/2**64;u_values[canon[g]]=u;gsp[g]='train' if u<.7 else ('validation' if u<.8 else 'test')
 out={r['raw_id']:gsp[r['duplicate_group']] for r in records};counts={s:list(gsp.values()).count(s) for s in ('train','validation','test')};audit={'ratios':[.7,.1,.2],'hash_bytes':'SHA256((salt+"\\n"+group_key).encode("utf-8")); first 8 bytes big-endian / 2^64','hash_u_thresholds':thresholds,'allocation':'independent frozen threshold; counts archived, never rebalanced','n_groups':len(groups),'group_counts':counts,'canonical_group_internal_hashes':sorted(canon.values()),'u_by_internal_group_hash':u_values,'assignment_sha256':hashlib.sha256(canonical_json(sorted((canon[g],gsp[g]) for g in groups))).hexdigest()};return out,audit
def build_manifests(records,split,key):
 ids={r['raw_id'] for r in records}
 if set(split)!=ids:raise RuntimeError('split coverage mismatch')
 gm={}
 for r in records:gm.setdefault(r['duplicate_group'],[]).append(r['canonical_id'])
 canon={g:hashlib.sha256('\n'.join(sorted(x)).encode()).hexdigest() for g,x in gm.items()};pub=[];priv=[]
 for r in records:
  v=r['raw_id'];hid=public_id(v,key);status=r['media_qc_status'];pub.append({'hashed_id':hid,'duplicate_group_hash':public_id('group:'+canon[r['duplicate_group']],key),'split':split[v],'media_qc_status':status});priv.append({'raw_id':v,'hashed_id':hid,'internal_media_hash':r['internal_media_hash'],'canonical_group_internal_hash':canon[r['duplicate_group']],'duration_seconds':float(r['duration_seconds']),'media_sha256':r.get('media_sha256'),'media_qc_status':status})
 pub.sort(key=lambda x:x['hashed_id']);priv.sort(key=lambda x:x['hashed_id'])
 if any(set(x)!=PUBLIC_ROW_KEYS for x in pub) or any(set(x)!=PRIVATE_ROW_KEYS for x in priv) or {x['hashed_id'] for x in pub}!={x['hashed_id'] for x in priv} or len(pub)!=len(ids):raise AssertionError('manifest schema/coverage')
 return {'schema_version':2,'visibility':'public','id_scheme':'HMAC-SHA256; key withheld','n_records':len(pub),'records':pub},{'schema_version':2,'visibility':'steward-private','n_records':len(priv),'records':priv}
def self_sealed_validation_proposal(pub,opaque_steward_manifest_sha256):
 if len(opaque_steward_manifest_sha256)!=64:raise ValueError('opaque steward manifest SHA-256 required')
 return {'protocol':'selfsealed_validation','official_validation':False,'train':'train labels only','selfsealed_validation':'aggregate-only steward service; bounded preregistered candidates and immutable query ledger','opaque_steward_manifest_sha256':opaque_steward_manifest_sha256,'test':'encrypted steward-only one-shot after signed freeze','public_split_manifest_sha256':hashlib.sha256(canonical_json(pub)).hexdigest(),'test_selection_forbidden':True}
def rasterize_1hz(duration,spans):
 if not np.isfinite(duration) or duration<=0:raise ValueError('invalid duration')
 T=max(1,int(math.ceil(duration)));y=np.zeros(T,dtype=np.uint8)
 for s,e in spans:
  if not np.isfinite(s) or not np.isfinite(e):raise ValueError('nonfinite span')
  s=max(0.,min(duration,float(s)));e=max(0.,min(duration,float(e)))
  if e>s:
   t=np.arange(T,dtype=float);y[(t<e)&((t+1)>s)]=1
 return y
def _validate_frame(pred,gt):
 if not gt or set(pred)!=set(gt):raise RuntimeError('exact nonempty cohort required')
 for v,y in gt.items():
  y=np.asarray(y);s=np.asarray(pred[v])
  if len(y)==0 or len(s)!=len(y) or not np.isfinite(s).all() or not np.isin(y,[0,1]).all():raise RuntimeError(f'invalid frame data {v}')
 if len(np.unique(np.concatenate([gt[v] for v in sorted(gt)])))!=2:raise RuntimeError('pooled GT needs both classes')
def stable_within(pred,gt):
 ids=sorted(gt);mixed=[v for v in ids if len(np.unique(gt[v]))==2];r={v:rankdata(pred[v],method='average')/len(pred[v]) for v in ids}
 if not mixed:return {'within_rank_ap':None,'within_rank_roc':None,'within_macro_ap':None,'within_macro_roc':None,'mixed_videos':0}
 y=np.concatenate([gt[v] for v in mixed]);z=np.concatenate([r[v] for v in mixed]);return {'within_rank_ap':float(average_precision_score(y,z)),'within_rank_roc':float(roc_auc_score(y,z)),'within_macro_ap':float(np.mean([average_precision_score(gt[v],r[v]) for v in mixed])),'within_macro_roc':float(np.mean([roc_auc_score(gt[v],r[v]) for v in mixed])),'mixed_videos':len(mixed)}
def frame_metrics(pred,gt):
 _validate_frame(pred,gt);ids=sorted(gt);y=np.concatenate([gt[v] for v in ids]);s=np.concatenate([pred[v] for v in ids]);return {'frame_ap':float(average_precision_score(y,s)),'frame_roc':float(roc_auc_score(y,s)),**stable_within(pred,gt)}
def _seg(x,conf):
 if len(x)!=(3 if conf else 2):raise ValueError('segment arity')
 x=tuple(map(float,x))
 if not all(np.isfinite(x)) or x[0]<0 or x[1]<=x[0]:raise ValueError('invalid segment')
 return x
def iou(a,b):
 inter=max(0.,min(a[1],b[1])-max(a[0],b[0]));return inter/(max(a[1],b[1])-min(a[0],b[0]))
def temporal_ap(pred,gt,thresholds=(.1,.3,.5,.7)):
 if set(pred)!=set(gt):raise RuntimeError('segment cohort mismatch')
 pred={v:[_seg(x,True) for x in q] for v,q in pred.items()};gt={v:[_seg(x,False) for x in q] for v,q in gt.items()};n=sum(map(len,gt.values()))
 if n==0:return {'tAP':{str(x):None for x in thresholds},'mean_tAP':None,'n_gt':0}
 out={}
 for th in thresholds:
  det=sorted([(c,v,(s,e)) for v,q in pred.items() for s,e,c in q],reverse=True);used={v:set() for v in gt};yy=[];ss=[]
  for c,v,x in det:
   cand=[(iou(x,g),j) for j,g in enumerate(gt[v]) if j not in used[v]];best=max(cand,default=(0.,-1));hit=best[0]>=th
   if hit:used[v].add(best[1])
   yy.append(int(hit));ss.append(c)
  if not det:out[str(th)]=0.;continue
  o=np.argsort(-np.asarray(ss));tp=np.cumsum(np.asarray(yy)[o]);fp=np.cumsum(1-np.asarray(yy)[o]);rec=tp/n;prec=tp/np.maximum(tp+fp,1);out[str(th)]=float(np.sum((rec-np.r_[0,rec[:-1]])*prec))
 return {'tAP':out,'mean_tAP':float(np.mean(list(out.values()))),'n_gt':n}
def paired_video_cluster_bootstrap(base,pred,gt,clusters,B=10000,seed=2026):
 if B<1 or not (set(base)==set(pred)==set(gt)==set(clusters)):raise ValueError('base/pred/gt/cluster keys must be exact')
 _validate_frame(base,gt);_validate_frame(pred,gt)
 mem={}
 for v,g in clusters.items():mem.setdefault(g,[]).append(v)
 keys=sorted(mem);rng=np.random.default_rng(seed);vals={k:[] for k in ('frame_ap','frame_roc','within_rank_ap','within_rank_roc','within_macro_ap','within_macro_roc')}
 for _ in range(B):
  bb={};pp={};gg={}
  for i,g in enumerate(rng.choice(keys,len(keys),replace=True)):
   for v in mem[g]:k=f'{i}:{v}';bb[k]=base[v];pp[k]=pred[v];gg[k]=gt[v]
  try:mb=frame_metrics(bb,gg);mp=frame_metrics(pp,gg)
  except RuntimeError:continue  # a resample may contain only one pooled class
  for k in vals:
   if mb[k] is not None and mp[k] is not None:vals[k].append(mp[k]-mb[k])
 return {'B':B,'unit':'duplicate-video cluster','metrics':{k:{'n_valid':len(x),'mean':float(np.mean(x)) if x else None,'q025':float(np.quantile(x,.025)) if x else None,'q975':float(np.quantile(x,.975)) if x else None} for k,x in vals.items()}}
def sign_freeze_manifest(m,key):
 if set(m)!={'checkpoint_sha256','source_sha256','evaluator_sha256','environment_sha256','prediction_sha256','config_sha256','split_sha256','selection_sha256'}:raise ValueError('freeze schema')
 return {'freeze_manifest':m,'freeze_manifest_sha256':hashlib.sha256(canonical_json(m)).hexdigest(),'signature_hmac_sha256':hmac.new(key,canonical_json(m),hashlib.sha256).hexdigest()}
@dataclasses.dataclass(frozen=True)
class SealedEvaluator:
 decryptor:Callable[[Path],dict];signing_key:bytes;steward_mode:bool=False
 def evaluate(self,predictions,encrypted_labels,freeze,provenance_out,bootstrap_B=10000):
  if not self.steward_mode:raise PermissionError('steward mode required')
  if provenance_out.exists():raise FileExistsError('any prior OPEN_STARTED permanently forbids re-entry')
  signed=sign_freeze_manifest(freeze['freeze_manifest'],self.signing_key)
  if signed!=freeze:raise RuntimeError('invalid signed freeze')
  ph=hashlib.sha256(canonical_json(predictions)).hexdigest()
  if ph!=freeze['freeze_manifest']['prediction_sha256']:raise RuntimeError('prediction freeze mismatch')
  provenance_out.parent.mkdir(parents=True,exist_ok=True);fh=provenance_out.open('x');started={'event':'OPEN_STARTED','freeze_manifest_sha256':freeze['freeze_manifest_sha256'],'freeze_signature':freeze['signature_hmac_sha256'],'encrypted_bundle_sha256':sha256_file(encrypted_labels),'prediction_sha256':ph};fh.write(json.dumps(started,sort_keys=True)+'\n');fh.flush();os.fsync(fh.fileno())
  try:
   lab=self.decryptor(encrypted_labels);gt={v:np.asarray(x,dtype=np.uint8) for v,x in lab['frame_gt'].items()};pred={v:np.asarray(x,dtype=float) for v,x in predictions['frame_scores'].items()};res={**frame_metrics(pred,gt),**temporal_ap(predictions.get('segments',{}),lab.get('segments',{}))}
   if 'baseline_scores' in lab:res['paired_video_cluster_bootstrap']=paired_video_cluster_bootstrap({v:np.asarray(x) for v,x in lab['baseline_scores'].items()},pred,gt,lab['duplicate_clusters'],bootstrap_B)
   final={'event':'COMPLETED','aggregate_metrics_only':True,'metrics_sha256':hashlib.sha256(canonical_json(res)).hexdigest()}
  except BaseException as exc:
   final={'event':'FAILED','error_type':type(exc).__name__};fh.write(json.dumps(final,sort_keys=True)+'\n');fh.flush();os.fsync(fh.fileno());fh.close();raise
  fh.write(json.dumps(final,sort_keys=True)+'\n');fh.flush();os.fsync(fh.fileno());fh.close();return res
def guarded_annotation_path(path,steward_mode):
 if not steward_mode:raise PermissionError('explicit steward mode required')
 if Path(path).suffix.lower() not in ('.xlsx','.xls','.csv','.json'):raise ValueError('unsupported type')
 return Path(path)
