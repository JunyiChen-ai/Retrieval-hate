#!/usr/bin/env python3
import argparse,hashlib,json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(ROOT/'scripts/duplex'),str(HERE.parent)]
from masked_parallel_isolation_pilot import Judge,block_mask,causal_mask,branch_position_ids,sequential_position_ids
from sentinel_localization_pilot import clean_chunk_text
from relation_v8.run import atomic_json
from relation_v4.io import sha256
def packed(judge,branches,attention,position):
 p=len(judge.prefix_ids);lens=[len(x) for x in branches];ids=list(judge.prefix_ids);ends=[]
 for b in branches:ids.extend(b);ends.append(len(ids)-1)
 mask=block_mask(p,lens,judge.dtype,judge.device) if attention=='masked' else causal_mask(len(ids),judge.dtype,judge.device);pos=branch_position_ids(p,lens,judge.device) if position=='reset' else sequential_position_ids(len(ids),judge.device);inp=torch.tensor([ids],device=judge.device);keep=torch.tensor(ends,device=judge.device)
 with torch.no_grad():logits=judge.model(input_ids=inp,attention_mask=mask,position_ids=pos,use_cache=False,logits_to_keep=keep).logits[0]
 return [judge.margin(logits[i]) for i in range(len(branches))],len(ids)
def exact_shared_branches(judge,texts):
 full=[judge.encode(judge._full_prompt(t)) for t in texts];p=min(len(x) for x in full)
 for i in range(p):
  if any(x[i]!=full[0][i] for x in full):p=i;break
 prefix=full[0][:p];branches=[x[p:] for x in full]
 if not prefix or any(prefix+b!=x for b,x in zip(branches,full)):raise AssertionError('exact LCP branch reconstruction failed')
 return prefix,branches
def main():
 p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);a=p.parse_args();out=Path(a.out_dir);cfg=json.load(open(out/'preregistered_config.json'));src=Path(cfg['asr_source']);rows=list(map(json.loads,open(src)))
 def record_id(r):
  v=r.get('video_id',r.get('hashed_id',r.get('opaque_id')))
  if not isinstance(v,str) or not v:raise RuntimeError('ASR row requires video_id, hashed_id, or opaque_id')
  return v
 records={record_id(r):r for r in rows}
 if len(records)!=len(rows):raise RuntimeError('duplicate ASR record ID')
 raw=out/'per_chunk_raw.jsonl'
 if raw.exists():raise RuntimeError('fresh raw output required')
 judge=Judge()
 if cfg.get('prompt_language')=='chinese':
  z=cfg['chinese_prompt_parts'];policy=z['policy_lead']+'\n'+z['policy_rules']
  def full_zh(text):
   user=z['wrapper'].format(policy_block=policy,chunk=text,question=z['question'])
   msgs=[{'role':'system','content':z['system']},{'role':'user','content':[{'type':'text','text':user}]}]
   return judge.processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
  judge._full_prompt=full_zh;judge.prefix_text='';judge.suffix_text='';judge.prefix_ids=[]
 tmp=raw.with_suffix(raw.suffix+'.tmp')
 if tmp.exists():raise RuntimeError('stale temporary raw output; inspect before retry')
 f=open(tmp,'x');n=0;tokens={k:0 for k in cfg['arms']};seqset=set(cfg['sequential_reference_subset'])
 for vi,v in enumerate(cfg['video_ids']):
  chunks=(cfg.get('sanitized_chunks') or {}).get(v)
  if chunks is None:chunks=records[v]['chunks']
  if not chunks:
   print(f'{vi+1}/{len(cfg["video_ids"])} {v} chunks=0 identity-missing',flush=True);continue
  texts=[clean_chunk_text(x.get('text')) for x in chunks];prefix,branches=exact_shared_branches(judge,texts);judge.prefix_ids=prefix;scores={}
  for arm in cfg['arms']:
   att='masked' if arm.startswith('masked') else 'causal';pos='reset' if arm.endswith('reset') else 'continuous';scores[arm],tok=packed(judge,branches,att,pos);tokens[arm]+=tok
  seq=[judge.score_sequential(t)[0] for t in texts] if v in seqset else [None]*len(texts)
  for i,(ch,sq) in enumerate(zip(chunks,seq)):
   start=None if ch.get('start') is None else float(ch['start']);end=None if ch.get('end') is None else float(ch['end']);rec={'video_id':v,'chunk_index':i,'source_chunk_index':ch.get('source_index',i),'start':start,'end':end,'temporal_span_valid':bool(start is not None and end is not None and end>start),'text_sha256':hashlib.sha256(texts[i].encode()).hexdigest(),'prompt_tokens_packed_branch':len(branches[i]),'scores':{arm:float(scores[arm][i]) for arm in cfg['arms']},'sequential_reference':None if sq is None else float(sq)};f.write(json.dumps(rec)+'\n');n+=1
  f.flush();print(f'{vi+1}/{len(cfg["video_ids"])} {v} chunks={len(chunks)}',flush=True)
 f.flush();f.close();tmp.replace(raw)
 manifest={'raw_frozen_before_gt':True,'atomic_raw_freeze':True,'config_sha256':sha256(out/'preregistered_config.json'),'raw_path':str(raw.resolve()),'raw_sha256':sha256(raw),'n_rows':n,'expected_sanitized_rows':cfg.get('n_valid_chunks',n),'zero_valid_video_ids':[v for v in cfg['video_ids'] if not (cfg.get('sanitized_chunks') or {}).get(v,records.get(v,{}).get('chunks'))],'packed_tokens':tokens,'model':'Qwen/Qwen3-VL-8B-Instruct','model_revision':cfg.get('model_revision'),'model_input':'ASR text only'}
 if n!=manifest['expected_sanitized_rows']:raise RuntimeError('sanitized row count mismatch')
 atomic_json(out/'raw_manifest.json',manifest);print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
