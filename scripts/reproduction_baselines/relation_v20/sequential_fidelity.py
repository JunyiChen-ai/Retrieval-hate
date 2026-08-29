#!/usr/bin/env python3
"""Supplementary frozen-raw fidelity check; never edits the raw score file."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(ROOT/'scripts/duplex'),str(HERE.parent)]
from masked_parallel_isolation_pilot import Judge
from sentinel_localization_pilot import clean_chunk_text
from relation_v4.io import sha256
from relation_v8.run import atomic_json
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args();d=Path(a.raw_dir);cfg=json.load(open(d/'preregistered_config.json'));rm=json.load(open(d/'raw_manifest.json'));rows=list(map(json.loads,open(d/'per_chunk_raw.jsonl')));by={}
 for r in rows:by.setdefault(r['video_id'],[]).append(r)
 subset=sorted(by,key=lambda v:(-len(by[v]),v))[:12];judge=Judge();packed=[];seq=[];n=0
 src={r['video_id']:r for r in map(json.loads,open(cfg['asr_source']))};clean=cfg['sanitized_chunks']
 for v in subset:
  chunks=clean[v]
  if len(chunks)!=len(by[v]):raise RuntimeError('chunk coverage mismatch')
  for ch,r in zip(chunks,by[v]):
   text=clean_chunk_text(ch.get('text'))
   if hashlib.sha256(text.encode()).hexdigest()!=r['text_sha256']:raise RuntimeError('text hash mismatch')
   packed.append(r['scores']['masked_branch_reset']);seq.append(judge.score_sequential(text)[0]);n+=1
 payload={'method':'masked_reset_sequential_fidelity_supplement','subset_rule':'12 videos with most frozen sanitized chunks, video-id tie break','video_ids':subset,'n_chunks':n,'spearman':float(spearmanr(packed,seq).statistic),'max_abs_delta':float(np.max(np.abs(np.asarray(packed)-np.asarray(seq)))),'raw_sha256':rm['raw_sha256'],'raw_manifest_sha256':sha256(d/'raw_manifest.json'),'raw_edited':False,'model':'Qwen/Qwen3-VL-8B-Instruct','answer_margin':'identical Yes/No token sets'};atomic_json(a.out,payload);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
