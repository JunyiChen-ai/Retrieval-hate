#!/usr/bin/env python3
"""Label-independent local inference and deterministic 1 Hz reduction."""
import argparse,json
from pathlib import Path
import torch
from core import V25,ecdf_logit,reduce_1hz,sha
from reference_builder import verify
from train import verify_declaration
def exact_global_fallback_1hz(global_score,duration):
 import math
 if not isinstance(global_score,(int,float)) or not math.isfinite(global_score) or not math.isfinite(duration) or duration<=0:raise RuntimeError('invalid frozen V16 fallback')
 return [global_score]*math.ceil(duration),[1]*math.ceil(duration)
def run(records,reference,states,out,fallback=False):
 verify(reference);root=Path(reference);refs=[json.load(open(root/f'{f}_full.json')) for f in ('text','multimodal')];models={s:V25() for s in map(str,SEEDS)}
 if set(states)!=set(models):raise RuntimeError('three selected states required')
 for s,m in models.items():m.load_state_dict(states[s])
 dest=Path(out);dest.mkdir(parents=True,exist_ok=False)
 seen=set()
 for line in open(records):
  r=json.loads(line);top={'video_id','duration','global_causal_score','windows'}
  import math
  if set(r)!=top or not isinstance(r['video_id'],str) or r['video_id'] in seen or not isinstance(r['duration'],(int,float)) or not math.isfinite(r['duration']) or r['duration']<=0 or not isinstance(r['global_causal_score'],(int,float)) or not math.isfinite(r['global_causal_score']):raise RuntimeError('input record exact schema/ID/finite');seen.add(r['video_id']);wins=r['windows']
  if not isinstance(wins,list) or not wins:raise RuntimeError('empty windows')
  wk={'window_index','start','end','text_isolated_score','multimodal_isolated_score'}
  for i,w in enumerate(wins):
   if set(w)!=wk or w['window_index']!=i or any(not isinstance(w[k],(int,float)) or not math.isfinite(w[k]) for k in wk-{'window_index'}) or not (0<=w['start']<w['end']<=r['duration']) or (i and w['start']!=wins[i-1]['end']):raise RuntimeError('window schema/order/finite/coverage')
  by={}
  if fallback:
   _,mask=reduce_1hz([{'start':w['start'],'end':w['end'],'logit':0.} for w in wins],r['duration']);raw,_=exact_global_fallback_1hz(r['global_causal_score'],r['duration']);by={s:list(raw) for s in models}
  else:
   z=torch.tensor([ecdf_logit([w[f'{f}_isolated_score'] for w in wins],refs[i]) for i,f in enumerate(('text','multimodal'))])
   for seed,m in models.items():
    ell=m.local(z).detach().tolist();ww=[{'start':w['start'],'end':w['end'],'logit':q} for w,q in zip(wins,ell)];by[seed],mask=reduce_1hz(ww,r['duration'])
  if not all(mask):raise RuntimeError('1Hz coverage below 1')
  mean=[sum(by[s][i] for s in by)/3 for i in range(len(mask))];(dest/f"{r['video_id']}.json").write_text(json.dumps({'video_id':r['video_id'],'scores_by_seed':by,'mean_scores':mean,'mask':mask,'duration':r['duration'],'fallback_global_constant':fallback},sort_keys=True)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--reference',required=True);p.add_argument('--frozen-state',required=True);p.add_argument('--train-protocol',required=True);p.add_argument('--protocol-declaration',required=True);p.add_argument('--out',required=True);a=p.parse_args();f=json.load(open(a.frozen_state));prot=json.load(open(a.train_protocol));decl=verify_declaration(a.protocol_declaration)
 if f.get('status')!='FINAL_PASS' or not f.get('test_seal_signed'):raise RuntimeError('inference forbidden before final seal')
 if f.get('reference_manifest_sha256')!=sha(Path(a.reference)/'manifest.json'):raise RuntimeError('frozen reference identity mismatch')
 chain=f.get('approved_test_input_chain');
 if not isinstance(chain,dict) or set(chain)!={'test_records','evidence_manifest','evidence_config'} or str(Path(a.records).resolve())!=chain['test_records']['path'] or any(sha(v['path'])!=v['sha256'] for v in chain.values()):raise RuntimeError('sealed test input chain mismatch')
 if f.get('train_protocol_sha256')!=sha(a.train_protocol) or f.get('selector_sha256')!=sha(Path(__file__).with_name('selector.py')) or f.get('inference_sha256')!=sha(__file__) or prot.get('trainer_sha256')!=sha(Path(__file__).with_name('train.py')) or prot.get('core_sha256')!=sha(Path(__file__).with_name('core.py')) or prot.get('protocol_declaration_sha256')!=sha(a.protocol_declaration) or decl.get('status')!='FINAL_AUTHORITATIVE_PRETRAINING_DECLARATION':raise RuntimeError('frozen source/protocol mismatch')
 run(a.records,a.reference,{s:{k:torch.tensor(v) for k,v in st.items()} for s,st in f['selected_states_by_seed'].items()},a.out,f.get('selected_epoch')==0)
if __name__=='__main__':main()
