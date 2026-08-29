#!/usr/bin/env python3
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/reproduction_baselines'))
from relation_v8.run import atomic_json
from relation_v4.io import sha256
MODEL_REV='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
def main():
 p=argparse.ArgumentParser();p.add_argument('--asr',required=True);p.add_argument('--qc-dir',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False);src=Path(a.asr);rows=list(map(json.loads,open(src)));ids=sorted(r['opaque_id'] for r in rows);dur={p.stem:float(json.load(open(p))['duration_seconds']) for p in Path(a.qc_dir).glob('*.json')}
 if len(ids)!=314 or len(set(ids))!=314:raise RuntimeError('exact 314 ASR IDs required')
 if set(dur)!=set(ids):raise RuntimeError('QC duration exact 314 coverage required')
 sanitized={};invalid=[];open_ended_repairs=[]
 for r in rows:
  good=[]
  for i,c in enumerate(r['chunks']):
   s,e=c.get('start'),c.get('end')
   if isinstance(s,(int,float)) and math.isfinite(s) and e is None and 0<=s<dur[r['opaque_id']]:
    e=dur[r['opaque_id']];open_ended_repairs.append({'video_id':r['opaque_id'],'source_index':i,'start':float(s),'end_from_qc_duration':e})
   if not isinstance(s,(int,float)) or not isinstance(e,(int,float)) or not math.isfinite(s) or not math.isfinite(e) or e<=s:invalid.append({'video_id':r['opaque_id'],'source_index':i});continue
   good.append({'start':float(s),'end':float(e),'text':c.get('text',''),'source_index':i})
  if not good:raise RuntimeError(f'zero valid ASR chunks: {r["opaque_id"]}')
  sanitized[r['opaque_id']]=good
 fw=ROOT/'scripts/reproduction_baselines/relation_v16/forward.py';prompt=ROOT/'scripts/duplex/masked_parallel_isolation_pilot.py';cfg={'method':'relation_v16_exact_causal_train_thvl','status':'FROZEN_BEFORE_V16_FORWARD','corpus':'thvl','split':'train','asr_source':str(src.resolve()),'asr_source_sha256':sha256(src),'qc_duration_policy':'label-free: finite open-ended ASR chunk with start<duration gets end=QC media duration; all other invalid spans masked','video_ids':ids,'video_ids_sha256':hashlib.sha256(''.join(v+'\n' for v in ids).encode()).hexdigest(),'n_videos':314,'sanitized_chunks':sanitized,'n_valid_chunks':sum(map(len,sanitized.values())),'n_invalid_chunks':len(invalid),'invalid_chunks':invalid,'open_ended_repairs':open_ended_repairs,'arms':['causal_continuous'],'sequential_reference_subset':[],'token_packing':'per-video exact LCP reconstruction asserted','prompt_implementation':str(prompt.resolve()),'prompt_implementation_sha256':sha256(prompt),'forward_implementation':str(fw.resolve()),'forward_implementation_sha256':sha256(fw),'model':'Qwen/Qwen3-VL-8B-Instruct','model_revision':MODEL_REV,'gt_or_temporal_labels_opened':False}
 atomic_json(out/'preregistered_config.json',cfg);print(json.dumps({'config_sha256':sha256(out/'preregistered_config.json'),'videos':314,'valid_chunks':cfg['n_valid_chunks'],'invalid':len(invalid),'open_ended_repairs':len(open_ended_repairs),'zero_valid_videos':sum(not x for x in sanitized.values())}))
if __name__=='__main__':main()
