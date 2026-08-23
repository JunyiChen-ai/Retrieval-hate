"""Independent C5 audit of the new interior-timestamp dense action assets."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .common import atomic_json,sha256_file
from .dense_asset_policy import assert_new_dense_asset,empty_dense_status

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();roles={}
 durations=json.load(open('artifacts/cvoi_acq/premetric-v2/durations/hatemm_train_val.json'))
 duration_by={str(x['video_id']):float(x['duration_s']) for x in durations['records']}
 for role,n in [('train',744),('val',107)]:
  fp=assert_new_dense_asset(a.root/f'{role}_dense4.f32');sp=a.root/f'{role}_dense_sidecar.jsonl';mp=a.root/f'{role}_visual_meta.json'
  rows=[json.loads(x) for x in sp.open() if x.strip()];arr=np.memmap(fp,dtype='<f4',mode='r',shape=(n*30,4,1024));by={};bad_hash=bad_time=bad_ref=0
  for q in rows:
   by.setdefault(q['action_id'],[]).append(q);v=arr[int(q['feature_row']),int(q['frame_slot'])]
   bad_hash+=hashlib.sha256(v.tobytes()).hexdigest()!=q['feature_sha256']
   D=duration_by[q['video_id']];expected=(int(q['window_id'])+(int(q['frame_slot'])+.5)/4)*D/30;bad_time+=abs(float(q['requested_t_s'])-expected)>1e-9
  empty=0
  for aid,qq in by.items():
   qq=sorted(qq,key=lambda x:x['frame_slot']);bad_ref+=len(qq)!=4 or [x['frame_slot'] for x in qq]!=[0,1,2,3] or len({x['feature_row'] for x in qq})!=1
   empty+=empty_dense_status(qq)=='EMPTY_DENSE'
  meta=json.load(open(mp));prov=meta['model_provenance'];ok=len(rows)==n*120 and len(by)==n*30 and not any((bad_hash,bad_time,bad_ref)) and np.isfinite(arr).all() and meta['contact']['test_contact_count']==0 and meta['model']=='openai/clip-vit-large-patch14-336' and 'pytorch_model.bin' in prov['resolved_files']
  roles[role]={'passed':bool(ok),'videos':n,'actions':len(by),'frame_rows':len(rows),'bad_feature_hashes':bad_hash,'bad_requested_timestamps':bad_time,'bad_action_feature_rows':bad_ref,'empty_dense_actions':empty,'empty_dense_contract':'all-four-fail=>EMPTY_DENSE','dense_sha256':sha256_file(fp),'sidecar_sha256':sha256_file(sp),'meta_sha256':sha256_file(mp),'model_weights_sha256':prov['resolved_files']['pytorch_model.bin']['sha256'],'test_contact_count':meta['contact']['test_contact_count']}
 # Required negative replay evidence remains part of a positive new-asset audit.
 replay=json.load(open('artifacts/cvoi_acq/premetric-v2/audits/c5_train_k30_parity_v1.json'));fixture_empty=empty_dense_status([{'decode_status':'failed'} for _ in range(4)])
 payload={'schema':'cvoi-c5-new-dense-independent-audit/1','passed':all(x['passed'] for x in roles.values()) and replay['passed'] is False and fixture_empty=='EMPTY_DENSE','roles':roles,'old_cache_comparability':'FAIL','old_train_replay_required_negative_evidence':{'path':'artifacts/cvoi_acq/premetric-v2/audits/c5_train_k30_parity_v1.json','sha256':sha256_file(Path('artifacts/cvoi_acq/premetric-v2/audits/c5_train_k30_parity_v1.json')),'passed':replay['passed'],'tolerance':replay['tolerance'],'max_abs':replay['max_abs']},'val_old_parity':'NOT_RUN_BY_D1','empty_dense_behavior_fixture':{'input':'four failed frames','output':fixture_empty,'passed':fixture_empty=='EMPTY_DENSE'},'candidate_metric_computed':False,'test_contact_count':sum(x['test_contact_count'] for x in roles.values())}
 atomic_json(a.out,payload)
if __name__=='__main__':main()
