from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .common import atomic_json,sha256_file
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();root=a.root
 start=json.loads((root/'start_manifest.json').read_text());changed=[]
 for q in start['sources']:
  p=Path(q['path']);h=sha256_file(p) if p.exists() else None
  if not p.exists() or p.stat().st_size!=q['size'] or p.stat().st_mtime_ns!=q['mtime_ns'] or h!=q['sha256']:changed.append(str(p))
 roles={};all_pass=not changed
 for role,n in [('train',744),('val',107)]:
  cr=[json.loads(x) for x in open(root/f'{role}_cheap_sidecar.jsonl')];dr=[json.loads(x) for x in open(root/f'{role}_dense_sidecar.jsonl')]
  ca=np.memmap(root/f'{role}_cheap_midpoint.f32',dtype='<f4',mode='r',shape=(n*30,1024));da=np.memmap(root/f'{role}_dense4.f32',dtype='<f4',mode='r',shape=(n*30,4,1024))
  bh=bz=fb=0
  for rows,arr,dense in [(cr,ca,False),(dr,da,True)]:
   for q in rows:
    v=arr[q['feature_row'],q.get('frame_slot',0)] if dense else arr[q['feature_row']]
    bh+=hashlib.sha256(v.tobytes()).hexdigest()!=q['feature_sha256'];fb+=q['fallback_source_slot'] is not None
    if q['decode_error'] is not None:bz+=bool(np.any(v))
  meta=json.loads((root/f'{role}_visual_meta.json').read_text());ok=len(cr)==n*30 and len(dr)==n*120 and bh==bz==fb==0 and np.isfinite(ca).all() and np.isfinite(da).all() and meta['contact']['test_contact_count']==0
  roles[role]={'passed':bool(ok),'cheap_rows':len(cr),'dense_rows':len(dr),'bad_feature_hashes':bh,'nonzero_decode_errors':bz,'fallback_rows':fb,'test_contact_count':meta['contact']['test_contact_count'],'files':{p.name:sha256_file(p) for p in root.glob(role+'_*')}};all_pass&=ok
 atomic_json(a.out,{'schema':'cvoi-visual-independent-ready/1','passed':bool(all_pass),'start_manifest_sha256':sha256_file(root/'start_manifest.json'),'source_count':len(start['sources']),'changed_sources':changed,'roles':roles})
if __name__=='__main__':main()
