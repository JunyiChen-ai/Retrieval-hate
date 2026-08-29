#!/usr/bin/env python3
"""Local Whisper timestamped ASR over pinned/QC-passed THVL validation audio."""
import hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq,AutoProcessor,pipeline
ROOT=Path(__file__).resolve().parents[2];PRIVATE=ROOT/'results/steward_private/thvl_bench';MODEL='openai/whisper-large-v3'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 qcpath=PRIVATE/'val32_download_qc.json';qc=json.load(open(qcpath));out=PRIVATE/'val32_timestamped_chunks.jsonl'
 if out.exists():raise RuntimeError('fresh ASR output required')
 items=[]
 for r in qc['rows']:
  wavs=[p for p in r['paths'] if p.get('wav_path')]
  if wavs:items.append((r['hashed_id'],wavs[0]))
 if not torch.cuda.is_available():raise RuntimeError('CUDA required for local Whisper')
 proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True);model=AutoModelForSpeechSeq2Seq.from_pretrained(MODEL,torch_dtype=torch.float16,low_cpu_mem_usage=True,local_files_only=True).to('cuda').eval();pipe=pipeline('automatic-speech-recognition',model=model,tokenizer=proc.tokenizer,feature_extractor=proc.feature_extractor,torch_dtype=torch.float16,device='cuda',chunk_length_s=30,batch_size=8);t0=time.time();rows=[]
 for i in range(0,len(items),8):
  batch=items[i:i+8];outs=list(pipe([x[1]['wav_path'] for x in batch],batch_size=len(batch),return_timestamps=True,return_language=True,generate_kwargs={'task':'transcribe'}))
  for (vid,w),x in zip(batch,outs):
   chunks=[{'start':c.get('timestamp',(None,None))[0],'end':c.get('timestamp',(None,None))[1],'text':c.get('text','')} for c in x.get('chunks',[])];rows.append({'hashed_id':vid,'wav_sha256':w['wav_sha256'],'text':(x.get('text') or '').strip(),'chunks':chunks,'n_chunks':len(chunks),'language':x.get('language')});print(f'{len(rows)}/{len(items)} {vid[:10]} chunks={len(chunks)}',flush=True)
 with open(out,'x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 prov={'method':'local_whisper_timestamped_asr','model':MODEL,'model_local_cache_only':True,'qc_manifest_sha256':sha(qcpath),'n_qc_audio':len(items),'n_asr_records':len(rows),'n_nonempty_chunks':sum(bool(r['chunks']) for r in rows),'total_chunks':sum(r['n_chunks'] for r in rows),'elapsed_seconds':time.time()-t0,'paid_api_cost_usd':0,'labels_or_gt_opened':False,'output_sha256':sha(out)};(PRIVATE/'val32_asr_provenance.json').write_text(json.dumps(prov,indent=2)+'\n');print(json.dumps(prov,indent=2))
if __name__=='__main__':main()
