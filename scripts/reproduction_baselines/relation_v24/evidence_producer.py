#!/usr/bin/env python3
"""Label-blind atomic THVL-train evidence producer for V24."""
import argparse,hashlib,json,math,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'src/duplex'),str(ROOT/'scripts/reproduction_baselines')]
MODEL='Qwen/Qwen3-VL-8B-Instruct';MODEL_REVISION='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b';CAP=3000;WIDTH=30.0
V16_FORWARD=ROOT/'scripts/reproduction_baselines/relation_v16/forward.py';V16_PROMPT=ROOT/'scripts/duplex/masked_parallel_isolation_pilot.py';V23_FORWARD=ROOT/'scripts/reproduction_baselines/relation_v23/forward.py';V23_OFFSETS=[0,-.5,.5,-1,1]
MM_POLICY='Judge only this synchronized local video moment. Answer Yes if it contains targeted hostility or severe derogatory conduct; otherwise answer No. Use the image and synchronized speech jointly. Do not infer from other moments.'
TEXT_POLICY='Judge only the supplied local video speech. Answer Yes if it contains targeted hostility or severe derogatory conduct; otherwise answer No. Do not infer from other moments.'
QUESTION='Does this local moment violate the policy? Answer Yes or No.'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_write(path,text):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with open(tmp,'x') as f:f.write(text);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def windows(duration):
 if not math.isfinite(duration) or duration<=0:raise ValueError('invalid duration')
 out=[];s=0.;i=0
 while s<duration:e=min(duration,s+WIDTH);out.append({'window_index':i,'start':s,'end':e,'center':(s+e)/2});s=e;i+=1
 return out
def window_speech(chunks,s,e,last):
 text=[]
 for c in chunks:
  try:a=float(c['start']);b=float(c['end'])
  except (KeyError,TypeError,ValueError):continue
  if not(math.isfinite(a) and math.isfinite(b) and b>a):continue
  m=(a+b)/2
  if s<=m<e or (last and m==e):
   z=' '.join(str(c.get('text','')).split())
   if z:text.append(z)
 return (' '.join(text) or '[NO SPEECH]')[:CAP]
def load_atomic(qc_dir,asr_dir):
 q={p.stem:json.loads(p.read_text()) for p in sorted(Path(qc_dir).glob('*.json'))};a={p.stem:json.loads(p.read_text()) for p in sorted(Path(asr_dir).glob('*.json'))}
 if not q or set(q)!=set(a):raise RuntimeError('atomic QC/ASR coverage incomplete or mismatched')
 qkeys={'opaque_id','hf_path','hf_revision','source_manifest_sha256','status','cache_path','bytes','media_sha256','duration_seconds','full_decode_ok','audio_available','video_available','wav_path','wav_sha256'};akeys={'opaque_id','wav_sha256','model','text','chunks','n_chunks','language','labels_or_temporal_gt_opened'}
 for vid in q:
  if set(q[vid])!=qkeys or set(a[vid])!=akeys or any(set(c)!={'start','end','text'} for c in a[vid].get('chunks',[])):raise RuntimeError('atomic QC/ASR exact allowlist violation')
  if q[vid].get('opaque_id')!=vid or a[vid].get('opaque_id')!=vid or q[vid].get('status')!='ok':raise RuntimeError('atomic identity/status mismatch')
  if a[vid].get('labels_or_temporal_gt_opened') is not False:raise RuntimeError('ASR provenance is not label blind')
 return q,a
def root_hash(root):
 rows=''.join(p.name+'\t'+sha(p)+'\n' for p in sorted(Path(root).glob('*.json')));return hashlib.sha256(rows.encode()).hexdigest()
def validate_v16(v16_dir,ids):
 d=Path(v16_dir);cfg=json.load(open(d/'preregistered_config.json'));man=json.load(open(d/'raw_manifest.json'));raw=d/'per_chunk_raw.jsonl'
 bound=cfg.get('forward_implementation_sha256')
 if cfg.get('model_revision')!=MODEL_REVISION or not isinstance(bound,str) or len(bound)!=64 or cfg.get('prompt_implementation_sha256')!=sha(V16_PROMPT) or 'causal_continuous' not in cfg.get('arms',[]):raise RuntimeError('V16 semantic identity mismatch')
 if man.get('raw_sha256')!=sha(raw) or man.get('config_sha256')!=sha(d/'preregistered_config.json') or man.get('model_revision')!=MODEL_REVISION:raise RuntimeError('V16 frozen chain mismatch')
 rows=list(map(json.loads,open(raw)));by={}
 for r in rows:
  if 'causal_continuous' not in r.get('scores',{}) or not math.isfinite(float(r['scores']['causal_continuous'])):raise RuntimeError('invalid V16 global row')
  by.setdefault(r['video_id'],[]).append(float(r['scores']['causal_continuous']))
 if set(by)!=set(ids) or any(not x for x in by.values()):raise RuntimeError('V16 global video coverage mismatch')
 return {v:sum(x)/len(x) for v,x in by.items()},{'config_sha256':sha(d/'preregistered_config.json'),'raw_sha256':sha(raw),'raw_manifest_sha256':sha(d/'raw_manifest.json'),'bound_forward_implementation_sha256':bound,'current_forward_implementation_sha256':sha(V16_FORWARD)}
def prompt_spec_hash():
 return hashlib.sha256(json.dumps({'mm':MM_POLICY,'text':TEXT_POLICY,'question':QUESTION},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def assert_runtime_config(cfg):
 if cfg.get('producer_sha256')!=sha(__file__) or cfg.get('local_forward_sha256')!=sha(__file__):raise RuntimeError('producer/local forward source changed after preregistration')
 if cfg.get('prompt_spec_sha256')!=prompt_spec_hash():raise RuntimeError('prompt spec changed after preregistration')
 if cfg.get('model')!=MODEL or cfg.get('model_revision')!=MODEL_REVISION:raise RuntimeError('model identity changed after preregistration')
 if cfg.get('v23_forward_sha256')!=sha(V23_FORWARD) or cfg.get('v16_forward_sha256')!=sha(V16_FORWARD) or cfg.get('v16_prompt_sha256')!=sha(V16_PROMPT):raise RuntimeError('bound V16/V23 implementation changed')
 from huggingface_hub import snapshot_download
 snapshot=Path(snapshot_download(MODEL,revision=MODEL_REVISION,local_files_only=True))
 if snapshot.name!=MODEL_REVISION or not (snapshot/'config.json').is_file():raise RuntimeError('local model snapshot identity mismatch')
 return snapshot
def preregister(qc_dir,asr_dir,v16_dir,out,split='train'):
 q,a=load_atomic(qc_dir,asr_dir);out=Path(out);out.mkdir(parents=True,exist_ok=False);items=[]
 for vid in sorted(q):
  ww=windows(float(q[vid]['duration_seconds']))
  rows=[]
  for j,w in enumerate(ww):
   sp=window_speech(a[vid].get('chunks',[]),w['start'],w['end'],j==len(ww)-1);rows.append({**w,'speech':sp,'speech_sha256':hashlib.sha256(sp.encode()).hexdigest()})
  items.append({'video_id':vid,'media_path':q[vid]['cache_path'],'media_sha256':q[vid]['media_sha256'],'duration':float(q[vid]['duration_seconds']),'asr_record_sha256':sha(Path(asr_dir)/(vid+'.json')),'windows':rows})
 ip=out/'frozen_inputs.jsonl';ip.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in items));globals_,v16hash=validate_v16(v16_dir,[x['video_id'] for x in items]);gp=out/'v16_global_means.jsonl';gp.write_text(''.join(json.dumps({'video_id':v,'global_causal_score':globals_[v]},sort_keys=True)+'\n' for v in sorted(globals_)));cfg={'method':'v24_train_val_evidence_label_blind','status':'PREREGISTERED_BEFORE_MODEL_FORWARD','split':split,'labels_read':False,'model':MODEL,'model_revision':MODEL_REVISION,'model_id_sha256':hashlib.sha256(MODEL.encode()).hexdigest(),'prompt_spec_sha256':prompt_spec_hash(),'local_forward_sha256':sha(__file__),'v23_forward_sha256':sha(V23_FORWARD),'v23_frame_fallback_offsets':V23_OFFSETS,'v16_forward_sha256':sha(V16_FORWARD),'v16_prompt_sha256':sha(V16_PROMPT),'v16_frozen':v16hash,'window_seconds':WIDTH,'asr_cap':CAP,'asr_truncation':'prefix','mm_policy':MM_POLICY,'text_policy':TEXT_POLICY,'question':QUESTION,'global_definition':'exact V16 packed causal_continuous ASR-chunk margins using V16 prompt/token/attention/continuous positions; arithmetic mean per video','required_execution_order':['freeze exact V16 raw for this split','prepare/freeze V24 config and windows','run V24 local text/MM forward'],'n_videos':len(items),'n_windows':sum(len(x['windows']) for x in items),'frozen_inputs_sha256':sha(ip),'qc_atomic_root_sha256':root_hash(qc_dir),'asr_atomic_root_sha256':root_hash(asr_dir),'v16_global_means_sha256':sha(gp),'producer_sha256':sha(__file__)};(out/'preregistered_config.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n');return cfg
def run(out):
 from transformers import AutoModelForImageTextToText,AutoProcessor
 import torch
 from score_duplex_probe import build_binary_token_ids
 from relation_v23.forward import frame_at
 out=Path(out);cfg=json.load(open(out/'preregistered_config.json'));ip=out/'frozen_inputs.jsonl'
 if cfg['frozen_inputs_sha256']!=sha(ip):raise RuntimeError('frozen inputs changed')
 snapshot=assert_runtime_config(cfg)
 proc=AutoProcessor.from_pretrained(MODEL,revision=MODEL_REVISION,local_files_only=True);ids=build_binary_token_ids(proc.tokenizer);yes=torch.tensor(sorted(ids['Yes']),device='cuda:0');no=torch.tensor(sorted(ids['No']),device='cuda:0');model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=MODEL_REVISION,local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval();rd=out/'records';rd.mkdir(exist_ok=True)
 def score(policy,speech,image=None):
  text=policy+'\n\nSynchronized speech: '+speech+'\n\n'+QUESTION;content=([{'type':'image','image':image}] if image is not None else [])+[{'type':'text','text':text}];prompt=proc.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True);enc=proc(text=[prompt],images=[image] if image is not None else None,return_tensors='pt');enc={k:v.to(model.device) for k,v in enc.items()}
  with torch.inference_mode():lg=model(**enc,use_cache=False,logits_to_keep=1).logits[0,-1].float()
  z=float(torch.logsumexp(lg[yes],0)-torch.logsumexp(lg[no],0));return z,hashlib.sha256(prompt.encode()).hexdigest()
 def valid(item,r):
  return r.get('video_id')==item['video_id'] and r.get('input_item_sha256')==hashlib.sha256(json.dumps(item,sort_keys=True,separators=(',',':')).encode()).hexdigest() and r.get('config_sha256')==sha(out/'preregistered_config.json') and r.get('producer_source_sha256')==sha(__file__) and r.get('model_revision')==MODEL_REVISION and r.get('prompt_spec_sha256')==cfg['prompt_spec_sha256'] and r.get('labels_read') is False and len(r.get('windows',[]))==len(item['windows']) and all(all(math.isfinite(float(x[k])) for k in ('text_isolated_score','multimodal_isolated_score')) for x in r['windows']) and math.isfinite(float(r.get('global_causal_score',float('nan'))))
 for item in map(json.loads,open(ip)):
  dst=rd/(item['video_id']+'.json')
  if dst.exists():
   r=json.load(open(dst))
   if valid(item,r):continue
   raise RuntimeError('stale atomic output')
  rows=[]
  for w in item['windows']:
   image,idx,fps,off,actual=frame_at(item['media_path'],w['center'],w['start'],w['end'],cfg['v23_frame_fallback_offsets']);zt,pt=score(TEXT_POLICY,w['speech']);zm,pm=score(MM_POLICY,w['speech'],image)
   if not all(math.isfinite(x) for x in (zt,zm)):raise RuntimeError('nonfinite evidence')
   rows.append({k:w[k] for k in ('window_index','start','end','center','speech_sha256')}|{'text_isolated_score':zt,'multimodal_isolated_score':zm,'frame_index':idx,'frame_time':actual,'frame_fallback_offset':off,'fps':fps,'prompt_sha256':{'text':pt,'multimodal':pm}})
  globals_={r['video_id']:r['global_causal_score'] for r in map(json.loads,open(out/'v16_global_means.jsonl'))};rec={'video_id':item['video_id'],'duration':item['duration'],'media_sha256':item['media_sha256'],'asr_record_sha256':item['asr_record_sha256'],'input_item_sha256':hashlib.sha256(json.dumps(item,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'config_sha256':sha(out/'preregistered_config.json'),'producer_source_sha256':sha(__file__),'model_revision':MODEL_REVISION,'prompt_spec_sha256':cfg['prompt_spec_sha256'],'windows':rows,'global_causal_score':globals_[item['video_id']],'labels_read':False};atomic_write(dst,json.dumps(rec,sort_keys=True)+'\n')
 # Complete-only aggregate.
 expected=[x['video_id'] for x in map(json.loads,open(ip))]
 if set(p.stem for p in rd.glob('*.json'))!=set(expected):raise RuntimeError('incomplete evidence coverage')
 items={x['video_id']:x for x in map(json.loads,open(ip))}
 for v in expected:
  if not valid(items[v],json.load(open(rd/(v+'.json')))):raise RuntimeError('invalid resumed/final atomic evidence')
 manifest={'status':'COMPLETE_LABEL_BLIND','n_videos':len(expected),'n_windows':cfg['n_windows'],'records':{v:sha(rd/(v+'.json')) for v in sorted(expected)},'config_sha256':sha(out/'preregistered_config.json'),'frozen_inputs_sha256':sha(ip)};atomic_write(out/'evidence_manifest.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--qc-dir',required=True);p.add_argument('--asr-dir',required=True);p.add_argument('--v16-dir',required=True);p.add_argument('--split',choices=['train','val'],required=True);p.add_argument('--out-dir',required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 if not Path(a.out_dir).exists():preregister(a.qc_dir,a.asr_dir,a.v16_dir,a.out_dir,a.split)
 if not a.prepare_only:run(a.out_dir)
if __name__=='__main__':main()
