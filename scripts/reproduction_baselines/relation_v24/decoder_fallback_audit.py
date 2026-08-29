#!/usr/bin/env python3
"""Label-free audit of windows that require the frozen ffmpeg frame fallback."""
import argparse,hashlib,json,os
from pathlib import Path
def sha(p):return hashlib.sha256(open(p,'rb').read()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args();d=Path(a.evidence_dir);cfg=json.load(open(d/'preregistered_config.json'));items=list(map(json.loads,open(d/'frozen_inputs.jsonl')));rows=[];nw=0
 import decord
 for item in items:
  failed=[];reason=None
  try:
   vr=decord.VideoReader(item['media_path'],num_threads=1);fps=max(float(vr.get_avg_fps()),1e-6)
   for w in item['windows']:
    ok=False
    for off in cfg['v23_frame_fallback_offsets']:
     tt=min(w['end'],max(w['start'],w['center']+off));idx=min(len(vr)-1,max(0,int(round(tt*fps))))
     try:vr[idx].asnumpy();ok=True;break
     except Exception:pass
    if not ok:failed.append(w['window_index'])
  except Exception as e:reason='decord_init_'+type(e).__name__;failed=[w['window_index'] for w in item['windows']]
  if failed:rows.append({'video_id':item['video_id'],'n_windows':len(item['windows']),'fallback_window_indices':failed,'reason':reason or 'all_decord_offsets_failed'});nw+=len(failed)
 out=Path(a.out);payload={'status':'COMPLETE_LABEL_FREE_DECODER_AUDIT','evidence_dir':str(d.resolve()),'n_videos':len(items),'n_windows':sum(len(x['windows']) for x in items),'fallback_videos':len(rows),'fallback_windows':nw,'details':rows,'frozen_inputs_sha256':sha(d/'frozen_inputs.jsonl'),'evidence_manifest_sha256':sha(d/'evidence_manifest.json'),'v23_forward_sha256':cfg['v23_forward_sha256'],'labels_or_gt_opened':False};tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');os.replace(tmp,out);print(json.dumps({k:payload[k] for k in ('n_videos','n_windows','fallback_videos','fallback_windows')},indent=2))
if __name__=='__main__':main()
