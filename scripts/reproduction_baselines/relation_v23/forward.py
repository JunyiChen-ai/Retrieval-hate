#!/usr/bin/env python3
"""V23 label-blind sequential multimodal forward and frozen derived scores."""
import argparse,hashlib,json,math,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'src/duplex')]
def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def frame_at(path,t,start,end,offsets):
 import decord,io,json as _json,subprocess
 from PIL import Image
 errors=[]
 try:
  vr=decord.VideoReader(path,num_threads=1);fps=max(float(vr.get_avg_fps()),1e-6)
  for off in offsets:
   tt=min(end,max(start,t+off));idx=min(len(vr)-1,max(0,int(round(tt*fps))))
   try:return Image.fromarray(vr[idx].asnumpy()).convert('RGB'),idx,fps,float(off),tt
   except Exception as e:errors.append('decord_frame_'+type(e).__name__)
 except Exception as e:errors.append('decord_init_'+type(e).__name__)
 # Deterministic interoperability fallback for media that passed frozen ffmpeg
 # full-decode QC but whose container/codec is unsupported by decord.
 probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate','-of','json',str(path)],capture_output=True,text=True,check=True);rate=(_json.loads(probe.stdout).get('streams') or [{}])[0].get('avg_frame_rate','0/1');a,b=rate.split('/');fps=max(float(a)/max(float(b),1e-12),1e-6)
 for off in offsets:
  tt=min(end,max(start,t+off));cmd=['ffmpeg','-v','error','-ss',f'{tt:.9f}','-i',str(path),'-frames:v','1','-f','image2pipe','-vcodec','png','-'];p=subprocess.run(cmd,capture_output=True)
  if p.returncode==0 and p.stdout:
   try:return Image.open(io.BytesIO(p.stdout)).convert('RGB'),max(0,int(round(tt*fps))),fps,float(off),tt
   except Exception as e:errors.append('ffmpeg_image_'+type(e).__name__)
  else:errors.append('ffmpeg_decode')
 raise RuntimeError(f'center frame decode failed offsets={offsets} errors={errors}')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);a=p.parse_args();out=Path(a.out_dir);cfg=json.load(open(out/'preregistered_config.json'));wp=out/'windows_frozen.jsonl'
 if cfg['windows_sha256']!=sha256(wp) or cfg['status']!='VAL_INFORMED_PREREGISTERED_BEFORE_FORWARD_NO_NEW_GT_ACCESS':raise RuntimeError('preregistration mismatch')
 from transformers import AutoModelForImageTextToText,AutoProcessor
 import torch
 from score_duplex_probe import build_binary_token_ids
 proc=AutoProcessor.from_pretrained(cfg['model']);ids=build_binary_token_ids(proc.tokenizer)
 if sorted(ids['Yes'])!=cfg['yes_token_ids'] or sorted(ids['No'])!=cfg['no_token_ids']:raise RuntimeError('binary token IDs changed since preregistration')
 yes=torch.tensor(cfg['yes_token_ids'],device='cuda:0');no=torch.tensor(cfg['no_token_ids'],device='cuda:0')
 model=AutoModelForImageTextToText.from_pretrained(cfg['model'],dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
 raw=out/'per_window_raw.jsonl';done=set()
 if raw.exists():
  for r in map(json.loads,open(raw)):done.add((r['video_id'],r['window_index']))
 with open(raw,'a') as f:
  for w in map(json.loads,open(wp)):
   key=(w['video_id'],w['window_index'])
   if key in done:continue
   image,idx,fps,off,actual_t=frame_at(w['media_path'],w['center'],w['start'],w['end'],cfg['decode']['center_fallback_offsets_seconds']);text=cfg['policy']+'\n\nSynchronized speech: '+w['speech']+'\n\n'+cfg['question']
   msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':text}]}];prompt=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
   enc=proc(text=[prompt],images=[image],return_tensors='pt');enc={k:v.to(model.device) for k,v in enc.items()}
   with torch.inference_mode():lg=model(**enc,use_cache=False,logits_to_keep=1).logits[0,-1].float()
   z=float(torch.logsumexp(lg[yes],0)-torch.logsumexp(lg[no],0));assert math.isfinite(z)
   rec={k:w[k] for k in ('video_id','window_index','start','end','center','duration','media_sha256','speech_sha256')};rec.update({'frame_index':idx,'frame_time':actual_t,'frame_fallback_offset':off,'fps':fps,'score_raw':z,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'input_tokens':int(enc['input_ids'].shape[1])})
   f.write(json.dumps(rec,sort_keys=True)+'\n');f.flush()
 # Freeze raw before any derived transformation; neither stage reads GT.
 man={'raw_frozen_before_steward':True,'raw_sha256':sha256(raw),'windows_sha256':sha256(wp),'n_rows':sum(1 for _ in open(raw))};(out/'raw_manifest.json').write_text(json.dumps(man,indent=2,sort_keys=True)+'\n')
 rows=list(map(json.loads,open(raw)));by={}
 for r in rows:by.setdefault(r['video_id'],[]).append(r)
 # Frozen V16 global reuse.
 vg={}
 for r in map(json.loads,open(cfg['global_v16_raw'])):vg.setdefault(r['video_id'],[]).append(float(r['scores']['causal_continuous']))
 derived=[]
 from scipy.stats import rankdata
 for vid,q in sorted(by.items()):
  q.sort(key=lambda x:x['window_index']);x=[r['score_raw'] for r in q];mean=sum(x)/len(x);ranks=rankdata(x,method='average')/len(x);glob=sum(vg[vid])/len(vg[vid])
  for r,rank in zip(q,ranks):derived.append({**r,'score_centered':r['score_raw']-mean,'score_rank':float(rank),'global_causal_text_mean':glob})
 rms=(sum(r['score_centered']**2 for r in derived)/len(derived))**.5
 if not math.isfinite(rms) or rms<=0:raise RuntimeError('invalid frozen local RMS')
 for r in derived:r['score_centered_rms_scaled']=r['score_centered']/rms
 dp=out/'scores_derived_no_gt.jsonl';dp.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in derived));(out/'derived_manifest.json').write_text(json.dumps({'source_raw_sha256':sha256(raw),'derived_sha256':sha256(dp),'n_rows':len(derived),'local_rms':rms,'local_rms_schema':cfg['local_rms_schema'],'v22_global_ecdf':cfg['v22_global_ecdf'],'ready_for_steward':True},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
