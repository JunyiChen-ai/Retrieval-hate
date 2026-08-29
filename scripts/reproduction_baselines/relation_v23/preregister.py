#!/usr/bin/env python3
"""Freeze V23 windows and prompt before model forward. Never imports GT code."""
import argparse,hashlib,json,math,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/duplex'))
PRIVATE=ROOT/'results/steward_private/thvl_bench'
PUBLIC=ROOT/'results/reproduction/thvl_sealed'
ASR=PRIVATE/'val32_timestamped_chunks.jsonl'; QC=PRIVATE/'val32_download_qc.json'
V16=PRIVATE/'v16_val_raw_frozen_32_v2/per_chunk_raw.jsonl'
V22=PRIVATE/'v22_val_components_frozen/manifest.json'
ASR_CAP=3000
POLICY="Judge only this synchronized local video moment. Answer Yes if it contains targeted hostility or severe derogatory conduct; otherwise answer No. Use the image and synchronized speech jointly. Do not infer from other moments."
QUESTION="Does this local moment violate the policy? Answer Yes or No."

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def make_windows(duration,width=30.0):
 if not math.isfinite(duration) or duration<=0:raise ValueError('positive finite duration required')
 out=[];start=0.0;i=0
 while start<duration:
  end=min(duration,start+width);out.append({'window_index':i,'start':start,'end':end,'center':(start+end)/2});start=end;i+=1
 return out

def speech_in_window(chunks,start,end,is_last=False):
 texts=[]
 for c in chunks:
  try:s=float(c['start']);e=float(c['end'])
  except (TypeError,ValueError,KeyError):continue
  if not(math.isfinite(s) and math.isfinite(e) and e>s):continue
  mid=(s+e)/2
  if (start<=mid<end) or (is_last and mid==end):
   t=' '.join(str(c.get('text','')).split())
   if t:texts.append(t)
 joined=' '.join(texts) if texts else '[NO SPEECH]'
 return joined[:ASR_CAP]

def main():
 p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
 asr={r['hashed_id']:r for r in map(json.loads,open(ASR))};qc=json.load(open(QC));rows=qc['rows'];ids=sorted(r['hashed_id'] for r in rows)
 if len(ids)!=32 or set(ids)!=set(asr):raise RuntimeError('exact reconciled 32-video ASR/QC coverage required')
 media={r['hashed_id']:r['paths'][0] for r in rows};windows=[]
 for vid in ids:
  m=media[vid];dur=float(m['duration_seconds']);ww=make_windows(dur)
  for j,w in enumerate(ww):
   speech=speech_in_window(asr[vid].get('chunks',[]),w['start'],w['end'],j==len(ww)-1)
   windows.append({'video_id':vid,**w,'duration':dur,'media_path':m['cache_path'],'media_sha256':m['sha256'],
                   'speech':speech,'speech_sha256':hashlib.sha256(speech.encode()).hexdigest()})
 wp=out/'windows_frozen.jsonl';wp.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in windows))
 from transformers import AutoProcessor
 import decord
 from score_duplex_probe import build_binary_token_ids
 proc=AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct');binary=build_binary_token_ids(proc.tokenizer)
 sentinel=POLICY+'\n\nSynchronized speech: [WINDOW_SPEECH]\n\n'+QUESTION
 msgs=[{'role':'user','content':[{'type':'image','image':'[IMAGE_OBJECT]'},{'type':'text','text':sentinel}]}]
 template=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
 v22=json.load(open(V22));ge=json.dumps(v22['calibration_state']['global_ecdf'],separators=(',',':'))
 cfg={'method':'relation_v23_thvl_multimodal_isolated_30s','status':'VAL_INFORMED_PREREGISTERED_BEFORE_FORWARD_NO_NEW_GT_ACCESS','test_informed':False,'val_informed':True,'corpus':'thvl','split':'validation_opaque_32',
      'n_videos':32,'n_windows':len(windows),'window_rule':'complete media timeline; deterministic nonoverlapping [30k,min(30(k+1),duration)); retain final short window',
      'speech_rule':f'valid ASR chunks assigned exactly once by midpoint; normalized whitespace; concatenate source order; retain first {ASR_CAP} Unicode codepoints; empty window literal [NO SPEECH]',
      'asr_cap_codepoints':ASR_CAP,'asr_truncation_direction':'keep prefix / discard tail',
      'frame_rule':'try deterministic offsets [0,-0.5,+0.5,-1.0,+1.0] seconds from window center; each clamped to [start,end] and media timeline; nearest decoded frame index round(t*fps); first successful decode wins',
      'decode':{'library':'decord','version':decord.__version__,'center_fallback_offsets_seconds':[0,-0.5,0.5,-1.0,1.0]},
      'policy':POLICY,'question':QUESTION,'model':'Qwen/Qwen3-VL-8B-Instruct','inference':'sequential isolated one image+text per window; no API',
      'prompt_template_sha256':hashlib.sha256(template.encode()).hexdigest(),'prompt_template_sentinel':'[WINDOW_SPEECH]','yes_token_ids':sorted(binary['Yes']),'no_token_ids':sorted(binary['No']),
      'score':'next-token logsumexp Yes minus logsumexp No','local_outputs':['per-video exact centered raw margin','per-video tie-aware percentile rank'],
      'local_rms_schema':'after raw freeze: sqrt(mean(score_centered^2)) over every validation window, each window weight 1; require finite and >0; record scalar and derived-manifest hash; rank is per-video scipy tie-aware average rank / n_windows',
      'global':'arithmetic per-video mean of frozen V16 causal_continuous ASR chunk margins','global_v16_raw':str(V16.resolve()),'global_v16_raw_sha256':sha256(V16),
      'v22_global_ecdf':{'manifest':str(V22.resolve()),'manifest_sha256':sha256(V22),'canonical_compact_json_sha256':hashlib.sha256(ge.encode()).hexdigest(),'n_reference':len(v22['calibration_state']['global_ecdf'])},
      'inputs':{'asr':str(ASR.resolve()),'asr_sha256':sha256(ASR),'media_qc':str(QC.resolve()),'media_qc_sha256':sha256(QC),'opaque_manifest_sha256':sha256(PUBLIC/'validation_opaque_manifest.json')},
      'windows_path':str(wp.resolve()),'windows_sha256':sha256(wp),'forbidden':['GT imports','THVL labels/taxonomy','test assets','dataset-specific prompt examples','API calls']}
 cp=out/'preregistered_config.json';cp.write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n');print(json.dumps({'config':str(cp.resolve()),'n_windows':len(windows),'no_speech':sum(x['speech']=='[NO SPEECH]' for x in windows)},indent=2))
if __name__=='__main__':main()
