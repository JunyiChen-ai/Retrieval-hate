#!/usr/bin/env python3
"""Label-blind conversion of frozen THVL val QC/ASR into V24 atomic allowlist."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'results/steward_private/thvl_bench';BASE=P/'val32_v24_atomic';QO=BASE/'qc_records';AO=BASE/'asr_records';QC=P/'val32_download_qc.json';ASR=P/'val32_timestamped_chunks.jsonl'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise RuntimeError(f'fresh atomic output required: {p}')
 t=p.with_suffix('.tmp')
 with open(t,'x') as f:json.dump(x,f,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def main():
 if BASE.exists():raise RuntimeError('fresh val atomic root required')
 q=json.load(open(QC));a={r['hashed_id']:r for r in map(json.loads,open(ASR))}
 if len(q['rows'])!=32 or len(a)!=32:raise RuntimeError('exact val32 required')
 for r in q['rows']:
  v=r['hashed_id'];pp=[x for x in r['paths'] if x.get('wav_path')]
  if r['status']!='ok' or len(pp)!=1 or v not in a:raise RuntimeError(f'invalid val source {v}')
  x=pp[0];qr={'opaque_id':v,'hf_path':x['repository_path'],'hf_revision':q['revision'],'source_manifest_sha256':sha(QC),'status':'ok','cache_path':x['cache_path'],'bytes':x['bytes'],'media_sha256':x['sha256'],'duration_seconds':x['duration_seconds'],'full_decode_ok':x['decode_ok'],'audio_available':x['audio_available'],'video_available':x['video_available'],'wav_path':x['wav_path'],'wav_sha256':x['wav_sha256']}
  ar={'opaque_id':v,'model':'openai/whisper-large-v3','text':a[v]['text'],'chunks':a[v]['chunks'],'n_chunks':a[v]['n_chunks'],'language':a[v].get('language'),'labels_or_temporal_gt_opened':False}
  if ar['wav_sha256']!=a[v]['wav_sha256']:raise RuntimeError('wav provenance mismatch')
  write(QO/f'{v}.json',qr);write(AO/f'{v}.json',ar)
 write(BASE/'manifest.json',{'status':'COMPLETE_LABEL_BLIND_ATOMIC_CONVERSION','n_videos':32,'qc_source_sha256':sha(QC),'asr_source_sha256':sha(ASR),'qc_root_hash':hashlib.sha256(''.join(p.name+'\t'+sha(p)+'\n' for p in sorted(QO.glob('*.json'))).encode()).hexdigest(),'asr_root_hash':hashlib.sha256(''.join(p.name+'\t'+sha(p)+'\n' for p in sorted(AO.glob('*.json'))).encode()).hexdigest(),'temporal_gt_opened':False})
 print(json.dumps({'videos':32,'out':str(BASE)}))
if __name__=='__main__':main()
