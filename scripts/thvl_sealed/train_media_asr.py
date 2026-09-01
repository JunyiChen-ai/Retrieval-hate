#!/usr/bin/env python3
"""Resumable pinned THVL train media QC and per-video atomic local Whisper ASR."""
import argparse,concurrent.futures,hashlib,json,math,os,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];PRIVATE=ROOT/'results/steward_private/thvl_bench'
MANIFEST=ROOT/'results/reproduction/thvl_sealed/train_media_manifest.json'
REPO='THVL/THVL-Bench';REV=None;MODEL='openai/whisper-large-v3'
CACHE=PRIVATE/'hf_media_cache';BASE=PRIVATE/'train314';QC_DIR=BASE/'qc_records';WAV_DIR=BASE/'wav16k';ASR_DIR=BASE/'asr_records'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def fsync_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 with open(t,'x') as f:json.dump(x,f,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def load_manifest():
 m=json.load(open(MANIFEST));q=m['records']
 if m['n_records']!=314 or len(q)!=314 or len({r['opaque_id'] for r in q})!=314 or len({r['hf_path'] for r in q})!=314:raise RuntimeError('train manifest exact 314 IDs/paths required')
 if any(r['split']!='train' or r['hf_revision']!=REV for r in q):raise RuntimeError('split/revision mismatch')
 return sorted(q,key=lambda r:r['opaque_id'])
def valid_existing(p,expected):
 try:
  x=json.load(open(p));return all(x.get(k)==v for k,v in expected.items()) and x.get('status')=='ok' and Path(x['wav_path']).is_file() and sha(x['wav_path'])==x['wav_sha256']
 except Exception:return False
def qc_one(r):
 from huggingface_hub import hf_hub_download
 vid=r['opaque_id'];op=QC_DIR/f'{vid}.json';expected={'opaque_id':vid,'hf_path':r['hf_path'],'hf_revision':REV,'source_manifest_sha256':sha(MANIFEST)}
 if valid_existing(op,expected):return 'skip',vid
 p=Path(hf_hub_download(REPO,r['hf_path'],repo_type='dataset',revision=REV,cache_dir=CACHE))
 actual=sha(p)
 if p.stat().st_size!=r['expected_repo_size_bytes'] or actual!=r['expected_repo_sha256']:raise RuntimeError(f'{vid}: pinned bytes/SHA mismatch')
 probe=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)],capture_output=True,text=True,check=True);pj=json.loads(probe.stdout)
 dec=subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'],capture_output=True)
 if dec.returncode:raise RuntimeError(f'{vid}: full decode failed')
 audio=any(x.get('codec_type')=='audio' for x in pj.get('streams',[]));video=any(x.get('codec_type')=='video' for x in pj.get('streams',[]))
 if not audio or not video:raise RuntimeError(f'{vid}: audio/video stream required')
 duration=float((pj.get('format') or {}).get('duration') or 0)
 if not math.isfinite(duration) or duration<=0:raise RuntimeError(f'{vid}: invalid duration')
 WAV_DIR.mkdir(parents=True,exist_ok=True);wav=WAV_DIR/f'{vid}.wav';tmp=wav.with_suffix('.wav.tmp')
 if tmp.exists():tmp.unlink()
 subprocess.run(['ffmpeg','-y','-v','error','-i',str(p),'-vn','-ac','1','-ar','16000','-f','wav',str(tmp)],check=True);os.replace(tmp,wav)
 rec={**expected,'status':'ok','cache_path':str(p.resolve()),'bytes':p.stat().st_size,'media_sha256':actual,'duration_seconds':duration,'full_decode_ok':True,'audio_available':True,'video_available':True,'wav_path':str(wav.resolve()),'wav_sha256':sha(wav)};fsync_json(op,rec);return 'done',vid
def stage_qc(workers):
 q=load_manifest();QC_DIR.mkdir(parents=True,exist_ok=True);t=time.time();done=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
  fut={ex.submit(qc_one,r):r['opaque_id'] for r in q}
  for f in concurrent.futures.as_completed(fut):
   status,vid=f.result();done+=1;print(f'{done}/314 {vid[:10]} {status}',flush=True)
 print(json.dumps({'stage':'qc','records':done,'wall_seconds':time.time()-t}))
def load_qc():
 q=load_manifest();out=[]
 for r in q:
  p=QC_DIR/f'{r["opaque_id"]}.json';expected={'opaque_id':r['opaque_id'],'hf_path':r['hf_path'],'hf_revision':REV,'source_manifest_sha256':sha(MANIFEST)}
  if not valid_existing(p,expected):raise RuntimeError(f'incomplete/invalid QC {r["opaque_id"]}')
  out.append(json.load(open(p)))
 return out
def valid_asr(p,q):
 try:
  x=json.load(open(p));return x['opaque_id']==q['opaque_id'] and x['wav_sha256']==q['wav_sha256'] and x['model']==MODEL and isinstance(x['chunks'],list)
 except Exception:return False
def stage_asr(batch_size):
 import torch
 from transformers import AutoModelForSpeechSeq2Seq,AutoProcessor,pipeline
 q=load_qc();ASR_DIR.mkdir(parents=True,exist_ok=True);pending=[x for x in q if not valid_asr(ASR_DIR/f'{x["opaque_id"]}.json',x)]
 print(json.dumps({'qc':len(q),'already_complete':len(q)-len(pending),'pending':len(pending)}),flush=True)
 if not pending:return
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True);model=AutoModelForSpeechSeq2Seq.from_pretrained(MODEL,torch_dtype=torch.float16,low_cpu_mem_usage=True,local_files_only=True).to('cuda').eval();pipe=pipeline('automatic-speech-recognition',model=model,tokenizer=proc.tokenizer,feature_extractor=proc.feature_extractor,torch_dtype=torch.float16,device='cuda',chunk_length_s=30,batch_size=batch_size);t=time.time();n=0
 for i in range(0,len(pending),batch_size):
  b=pending[i:i+batch_size];outs=list(pipe([x['wav_path'] for x in b],batch_size=len(b),return_timestamps=True,return_language=True,generate_kwargs={'task':'transcribe'}))
  for qrow,x in zip(b,outs):
   chunks=[{'start':c.get('timestamp',(None,None))[0],'end':c.get('timestamp',(None,None))[1],'text':c.get('text','')} for c in x.get('chunks',[])]
   rec={'opaque_id':qrow['opaque_id'],'wav_sha256':qrow['wav_sha256'],'model':MODEL,'text':(x.get('text') or '').strip(),'chunks':chunks,'n_chunks':len(chunks),'language':x.get('language'),'labels_or_temporal_gt_opened':False};fsync_json(ASR_DIR/f'{qrow["opaque_id"]}.json',rec);n+=1;print(f'{len(q)-len(pending)+n}/314 {qrow["opaque_id"][:10]} chunks={len(chunks)} atomic',flush=True)
 print(json.dumps({'stage':'asr','new_records':n,'wall_seconds':time.time()-t}),flush=True)
def stage_finalize():
 q=load_qc();asr=[]
 for x in q:
  p=ASR_DIR/f'{x["opaque_id"]}.json'
  if not valid_asr(p,x):raise RuntimeError(f'incomplete ASR {x["opaque_id"]}')
  asr.append(json.load(open(p)))
 out=BASE/'timestamped_chunks.jsonl';tmp=out.with_suffix('.jsonl.tmp')
 with open(tmp,'x') as f:
  for x in asr:f.write(json.dumps(x,ensure_ascii=False)+'\n')
  f.flush();os.fsync(f.fileno())
 os.replace(tmp,out)
 windows=BASE/'windows30s_full_coverage.jsonl';tmp=windows.with_suffix('.jsonl.tmp');nw=0
 with open(tmp,'x') as f:
  for qrow,a in zip(q,asr):
   dur=qrow['duration_seconds'];n=max(1,int(math.ceil(dur/30)))
   for j in range(n):
    s=30.*j;e=min(dur,30.*(j+1));texts=[]
    for c in a['chunks']:
     cs,ce=c.get('start'),c.get('end')
     if isinstance(cs,(int,float)) and math.isfinite(cs) and (ce is None or isinstance(ce,(int,float))) and cs<e and (ce is None or ce>s):texts.append(c.get('text',''))
    text=' '.join(x.strip() for x in texts if x and x.strip())
    f.write(json.dumps({'opaque_id':qrow['opaque_id'],'window_index':j,'start':s,'end':e,'duration_seconds':dur,'text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'speech_text_available':bool(text)})+'\n');nw+=1
  f.flush();os.fsync(f.fileno())
 os.replace(tmp,windows)
 payload={'dataset':'THVL-Bench','split':'train','n_videos':314,'n_asr_nonempty':sum(bool(x['chunks']) for x in asr),'n_chunks':sum(x['n_chunks'] for x in asr),'n_windows30s':nw,'window_policy':'mechanical [30j,min(duration,30(j+1))) full coverage; ASR text from overlapping timestamp chunks; empty text retained','qwen_forward_status':'NOT_RUN_AWAIT_V24_PROTOCOL_FREEZE','labels_or_temporal_gt_opened':False,'source':{'train_media_manifest_sha256':sha(MANIFEST),'qc_record_hashes_sha256':hashlib.sha256(''.join(sha(QC_DIR/f'{x["opaque_id"]}.json')+'\n' for x in q).encode()).hexdigest(),'asr_record_hashes_sha256':hashlib.sha256(''.join(sha(ASR_DIR/f'{x["opaque_id"]}.json')+'\n' for x in q).encode()).hexdigest(),'tool_sha256':sha(__file__)},'timestamped_chunks_sha256':sha(out),'windows_sha256':sha(windows)};fsync_json(BASE/'final_manifest.json',payload);print(json.dumps(payload,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['qc','asr','finalize'],required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--batch-size',type=int,default=8);a=p.parse_args()
 {'qc':lambda:stage_qc(a.workers),'asr':lambda:stage_asr(a.batch_size),'finalize':stage_finalize}[a.stage]()
if __name__=='__main__':main()
