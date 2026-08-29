#!/usr/bin/env python3
"""Only this post-freeze steward joins train weak labels into V24 bags."""
import argparse,hashlib,json
from pathlib import Path
from evidence_producer import sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence-dir',required=True);p.add_argument('--weak-manifest',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args();ed=Path(a.evidence_dir);em=json.load(open(ed/'evidence_manifest.json'));wm=json.load(open(a.weak_manifest));labels={r['opaque_id']:int(r['weak_video_label']) for r in wm['records']};ids=sorted(em['records'])
 if set(labels)!=set(ids) or any(x not in (0,1) for x in labels.values()):raise RuntimeError('weak label coverage/schema mismatch')
 out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False);globalp=out/'v23_global_values.jsonl';records=[json.load(open(ed/'records'/(v+'.json'))) for v in ids];globalp.write_text(''.join(json.dumps({'video_id':r['video_id'],'global_causal_score':r['global_causal_score']},sort_keys=True)+'\n' for r in records));gh=sha(globalp);bags=out/'bags.jsonl'
 with open(bags,'w') as f:
  for r in records:
   text=[x['text_isolated_score'] for x in r['windows']];mm=[x['multimodal_isolated_score'] for x in r['windows']];row={'corpus':'thvl','split':'train','video_id':r['video_id'],'video_label':labels[r['video_id']],'global_causal_score':r['global_causal_score'],'families':{'text':[text],'multimodal':[mm]},'source_hashes':{'text_scores_sha256':hashlib.sha256(json.dumps(text,separators=(',',':')).encode()).hexdigest(),'multimodal_scores_sha256':hashlib.sha256(json.dumps(mm,separators=(',',':')).encode()).hexdigest(),'v23_global_source_sha256':gh}};f.write(json.dumps(row,sort_keys=True)+'\n')
 prod=sha(Path(__file__).resolve());idm={'corpus':'thvl','split':'train','ids':ids,'v23_global_source_sha256':gh,'producer_sha256':prod};(out/'train_id_manifest.json').write_text(json.dumps(idm,indent=2,sort_keys=True)+'\n');(out/'join_manifest.json').write_text(json.dumps({'bags_sha256':sha(bags),'global_sha256':gh,'evidence_manifest_sha256':sha(ed/'evidence_manifest.json'),'weak_manifest_sha256':sha(a.weak_manifest),'join_producer_sha256':prod},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
