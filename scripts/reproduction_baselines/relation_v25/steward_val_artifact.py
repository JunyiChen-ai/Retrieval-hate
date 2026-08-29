#!/usr/bin/env python3
"""Steward-only THVL val GT generator; non-val rows expose raw videoID only."""
import argparse,ast,hashlib,json,math,os,tempfile
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from temporal_eval_rule import overlaps
SCHEMA='thvl_val_temporal_steward_v2';HEADER=('videoID','segment-level annotation','segment-level timestamp','contributing modalities');HB=','.join(HEADER).encode();HSH='ec5feb87f0e9419e59889a8f42bb9160a640ca70c17b20b3d2859b17f92b4d93'
RK={'canonical_id','hashed_id','raw_id','repository_paths','source_group','split'}
def hb(b):return hashlib.sha256(b).hexdigest()
def sha(p):return hb(Path(p).read_bytes())
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def atomic(p,b):
 p=Path(p);fd,t=tempfile.mkstemp(prefix=p.name+'.',dir=p.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 except BaseException:
  try:os.unlink(t)
  except FileNotFoundError:pass
  raise
def public(vp,qp,tp):
 v=json.load(open(vp));q=json.load(open(qp));t=json.load(open(tp))
 if set(v)!={'schema','ids'} or v['schema']!='thvl_public_val_ids_v1' or len(v['ids'])!=32 or len(set(v['ids']))!=32:raise RuntimeError('exact32 val')
 if set(q)!={'schema','durations'} or q['schema']!='thvl_qc_durations_v1' or set(q['durations'])!=set(v['ids']) or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in q['durations'].values()):raise RuntimeError('QC')
 if set(t)!={'schema','category_count','target_indices','other_harm_indices'} or t['schema']!='thvl_taxonomy_indices_v1' or t['category_count']!=11 or t['target_indices']!=[1,2,3] or set(t['target_indices'])&set(t['other_harm_indices']):raise RuntimeError('taxonomy')
 return v,q,t
def idmap(p,source,wanted):
 m=json.load(open(p))
 if set(m)!={'remote_identity','records'} or len(m['records'])!=450 or m['remote_identity'].get('annotation_sha256')!=sha(source):raise RuntimeError('map/source')
 out={};opaque=set()
 for r in m['records']:
  if set(r)!=RK or r['split'] not in ('train','validation','test') or r['raw_id'] in out or r['hashed_id'] in opaque:raise RuntimeError('map duplicate/schema')
  out[r['raw_id']]=(r['hashed_id'],r['split']);opaque.add(r['hashed_id'])
 if {x for x,s in out.values() if s=='validation'}!=wanted:raise RuntimeError('map swap/membership')
 return out
def records(p):
 b=Path(p).read_bytes();s=i=0;q=False
 while i<len(b):
  if b[i]==34:
   if q and i+1<len(b) and b[i+1]==34:i+=2;continue
   q=not q
  if not q and b[i] in (10,13):yield b[s:i];i+=2 if b[i]==13 and i+1<len(b) and b[i+1]==10 else 1;s=i;continue
  i+=1
 if q:raise RuntimeError('unterminated quote')
 if s<len(b):yield b[s:]
def field(r,n):
 f=i=0;q=False;o=bytearray()
 while i<=len(r):
  c=r[i] if i<len(r) else 44
  if c==34:
   if q and i+1<len(r) and r[i+1]==34:
    if f==n:o.append(34)
    i+=2;continue
   q=not q;i+=1;continue
  if c==44 and not q:
   if f==n:return bytes(o)
   f+=1;o.clear();i+=1;continue
  if f==n:o.append(c)
  i+=1
 raise RuntimeError('missing field')
def literal(s,name):
 try:x=ast.literal_eval(s)
 except (ValueError,SyntaxError):raise RuntimeError('malformed '+name)
 if not isinstance(x,list):raise RuntimeError(name+' list')
 return x
def tm(x):
 if not isinstance(x,str) or len(x.strip().split(':')) not in (2,3):raise RuntimeError('timestamp')
 try:a=[float(z) for z in x.strip().split(':')]
 except ValueError:raise RuntimeError('timestamp')
 if any(not math.isfinite(z) or z<0 for z in a):raise RuntimeError('timestamp')
 return a[-1]+60*a[-2]+(3600*a[-3] if len(a)==3 else 0)
def parse(ls,ts,d,target,other):
 ll=literal(ls,'labels');tt=literal(ts,'timestamps')
 if len(ll)!=len(tt):raise RuntimeError('label/timestamp length mismatch')
 spans=[]
 for a,z in zip(ll,tt):
  if not isinstance(a,list) or len(a)!=11 or any(type(x)is not int or x not in (0,1) for x in a) or not isinstance(z,list) or len(z)!=2:raise RuntimeError('official serialization')
  s,e=map(tm,z)
  if not 0<=s<e:raise RuntimeError('span')
  e=min(e,d)
  if s<d:
   spans.extend({'start':s,'end':min(e,d),'label':i} for i,x in enumerate(a) if x and i in target|other)
 y=[0]*math.ceil(d);valid=[1]*len(y)
 for j in range(len(y)):
  labs={z['label'] for z in spans if overlaps(z['start'],z['end'],j,d)}
  if labs&target:y[j]=1
  elif labs&other:valid[j]=0
 return y,valid
def bindings(v,q,t,src,mp):return {'val_manifest_sha256':sha(v),'qc_sha256':sha(q),'taxonomy_sha256':sha(t),'private_source_sha256':sha(src),'raw_id_map_path':str(Path(mp).resolve()),'raw_id_map_sha256':sha(mp),'generator_sha256':sha(__file__),'header_sha256':HSH,'header_fields':list(HEADER),'modalities_extracted':False}
def generate(vp,qp,tp,src,mp,key,out):
 v,q,t=public(vp,qp,tp);want=set(v['ids']);m=idmap(mp,src,want);it=iter(records(src));h=next(it,None)
 if h!=HB or hb(h)!=HSH:raise RuntimeError('official exact4 header/hash')
 got={};seen=set();non=[]
 for row in it:
  rawb=field(row,0);raw=rawb.decode() # sole pre-membership extraction
  if raw in seen:raise RuntimeError('duplicate source ID')
  seen.add(raw)
  if raw not in m:raise RuntimeError('unmapped ID')
  oid,split=m[raw]
  if split!='validation':non.append(hb(rawb));continue
  # Only val extracts labels/timestamps; modality field 3 is never extracted.
  d=q['durations'][oid];y,z=parse(field(row,1).decode(),field(row,2).decode(),d,set(t['target_indices']),set(t['other_harm_indices']));got[oid]={'duration':d,'target_1hz':y,'valid_1hz':z}
 if len(seen)!=450 or set(got)!=want:raise RuntimeError('coverage')
 b=bindings(vp,qp,tp,src,mp);payload={'schema':SCHEMA,'ids':sorted(got),'records':{x:got[x] for x in sorted(got)},'bindings':b,'test_annotations_materialized':False};ledger={'schema':'thvl_val_access_ledger_v2','source_rows_seen':450,'val_rows_annotation_parsed':32,'nonval_filtered_before_other_fields_extract':418,'nonval_raw_id_hashes_sha256':hb(''.join(sorted(x+'\n' for x in non)).encode()),'nonval_other_fields_extracted':False,'test_annotations_materialized':False,'test_labels_materialized':False}
 k=Path(key).read_bytes()
 if len(k)!=32:raise RuntimeError('key')
 nonce=os.urandom(12);aad=canon({'schema':SCHEMA,'bindings':b});cipher=nonce+AESGCM(k).encrypt(nonce,canon(payload),aad);o=Path(out)
 if o.exists():raise RuntimeError('output exists')
 o.mkdir(parents=True);atomic(o/'artifact.aesgcm',cipher);atomic(o/'access_ledger.json',json.dumps(ledger,sort_keys=True).encode()+b'\n');man={'schema':'thvl_val_encrypted_manifest_v2','cipher_sha256':sha(o/'artifact.aesgcm'),'aad_sha256':hb(aad),'access_ledger_sha256':sha(o/'access_ledger.json'),'bindings':b,'n_val':32};atomic(o/'manifest.json',json.dumps(man,sort_keys=True).encode()+b'\n');return man
def decrypt_and_verify(out,key,vp,qp,tp,src,mp):
 o=Path(out);m=json.load(open(o/'manifest.json'));b=bindings(vp,qp,tp,src,mp)
 if set(m)!={'schema','cipher_sha256','aad_sha256','access_ledger_sha256','bindings','n_val'} or m['schema']!='thvl_val_encrypted_manifest_v2' or m['n_val']!=32 or m['bindings']!=b or sha(o/'artifact.aesgcm')!=m['cipher_sha256'] or sha(o/'access_ledger.json')!=m['access_ledger_sha256']:raise RuntimeError('manifest/binding')
 l=json.load(open(o/'access_ledger.json'))
 if set(l)!={'schema','source_rows_seen','val_rows_annotation_parsed','nonval_filtered_before_other_fields_extract','nonval_raw_id_hashes_sha256','nonval_other_fields_extracted','test_annotations_materialized','test_labels_materialized'} or l['schema']!='thvl_val_access_ledger_v2' or l['source_rows_seen']!=450 or l['val_rows_annotation_parsed']!=32 or l['nonval_other_fields_extracted'] is not False or l['test_annotations_materialized'] is not False or l['test_labels_materialized'] is not False:raise RuntimeError('ledger')
 aad=canon({'schema':SCHEMA,'bindings':b})
 if hb(aad)!=m['aad_sha256']:raise RuntimeError('AAD')
 c=(o/'artifact.aesgcm').read_bytes();x=json.loads(AESGCM(Path(key).read_bytes()).decrypt(c[:12],c[12:],aad));v,q,_=public(vp,qp,tp)
 if set(x)!={'schema','ids','records','bindings','test_annotations_materialized'} or x['schema']!=SCHEMA or x['bindings']!=b or x['ids']!=sorted(v['ids']) or set(x['records'])!=set(v['ids']):raise RuntimeError('payload')
 for oid,r in x['records'].items():
  if set(r)!={'duration','target_1hz','valid_1hz'} or r['duration']!=q['durations'][oid] or len(r['target_1hz'])!=math.ceil(r['duration']) or len(r['valid_1hz'])!=math.ceil(r['duration']) or any(a not in (0,1) for a in r['target_1hz']+r['valid_1hz']) or any(a and not z for a,z in zip(r['target_1hz'],r['valid_1hz'])):raise RuntimeError('array')
 return x
def main():
 p=argparse.ArgumentParser()
 for x in ('val-manifest','qc','taxonomy','private-source','raw-id-map','key','out'):p.add_argument('--'+x,required=True)
 a=p.parse_args();generate(a.val_manifest,a.qc,a.taxonomy,a.private_source,a.raw_id_map,a.key,a.out)
if __name__=='__main__':main()
