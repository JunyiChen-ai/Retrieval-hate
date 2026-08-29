#!/usr/bin/env python3
import copy,hashlib,json,tempfile
from pathlib import Path
import numpy as np
from core import *
def fails(fn,exc):
 try:fn()
 except exc:return
 raise AssertionError(f'expected {exc.__name__}')
def main():
 assert verify_remote_identity(dict(FIXED_REMOTE_IDENTITY))['verified'];bad=dict(FIXED_REMOTE_IDENTITY);bad['commit']='0'*40;fails(lambda:verify_remote_identity(bad),RuntimeError);fails(lambda:verify_remote_identity({}),ValueError)
 rec=[]
 for i in range(10):
  for j in range(2 if i==0 else 1):rec.append({'raw_id':f'id-{i}-{j}','canonical_id':f'platform:{i}-{j}','internal_media_hash':hashlib.sha256(f'bytes-{i}-{j}'.encode()).hexdigest(),'duplicate_group':f'name-{i}','duration_seconds':3,'media_sha256':'0'*64,'media_qc_status':'available_decodable'})
 sp,a=exact_group_split(rec,'DeHate-selfsealed-v1-2026-08-29');assert a['hash_u_thresholds']=={'train':[0.,.7],'validation':[.7,.8],'test':[.8,1.]};assert sum(a['group_counts'].values())==10
 cross=[dict(r) for r in rec];cross[1]['internal_media_hash']=cross[0]['internal_media_hash'];cross[1]['duplicate_group']='definitely-another-group';fails(lambda:exact_group_split(cross,'DeHate-selfsealed-v1-2026-08-29'),RuntimeError)
 gold=[{'raw_id':'x','canonical_id':'platform:1','internal_media_hash':'0'*64,'duplicate_group':'g','duration_seconds':1,'media_sha256':'0'*64,'media_qc_status':'available_decodable'}];gsp,ga=exact_group_split(gold,'DeHate-selfsealed-v1-2026-08-29');u=next(iter(ga['u_by_internal_group_hash'].values()));assert abs(u-.8029727881971915)<1e-15 and gsp['x']=='test'
 renamed=[{**r,'raw_id':'renamed-'+r['raw_id'],'duplicate_group':'renamed-'+r['duplicate_group']} for r in reversed(rec)];_,a2=exact_group_split(renamed,'DeHate-selfsealed-v1-2026-08-29');assert a['assignment_sha256']==a2['assignment_sha256']
 pub,priv=build_manifests(rec,sp,b'fixture-key');assert pub['n_records']==priv['n_records']==len(rec);assert all(set(x)==PUBLIC_ROW_KEYS and 'raw_id' not in x and 'duration_seconds' not in x and 'media_sha256' not in x for x in pub['records']);assert self_sealed_validation_proposal(pub,'9'*64)['protocol']=='selfsealed_validation';fails(lambda:build_manifests(rec,{**sp,'extra':'train'},b'k'),RuntimeError)
 assert rasterize_1hz(4.2,[(.2,1.1),(3.5,5.)]).tolist()==[1,1,0,1,1]
 gt={'a':np.array([0,1,1,0]),'b':np.array([0,0,1])};pred={'a':np.array([.1,.8,.9,.2]),'b':np.array([.1,.2,.7])};assert frame_metrics(pred,gt)['frame_ap']>.99;fails(lambda:frame_metrics(pred,{'a':np.array([0,2,1,0]),'b':gt['b']}),RuntimeError);z=stable_within({'x':np.array([1.,1.])},{'x':np.array([0,0])});assert z['mixed_videos']==0 and z['within_macro_ap'] is None
 ta=temporal_ap({'a':[(1,3,.9)],'b':[(2,3,.8)]},{'a':[(1,3)],'b':[(2,3)]});assert set(ta['tAP'])=={'0.1','0.3','0.5','0.7'} and ta['mean_tAP']==1.;assert temporal_ap({'a':[]},{'a':[]})['mean_tAP'] is None;fails(lambda:temporal_ap({'a':[(2,1,.5)]},{'a':[]}),ValueError)
 boot=paired_video_cluster_bootstrap(pred,pred,gt,{'a':'g1','b':'g2'},B=20);assert boot['B']==20 and boot['metrics']['frame_ap']['mean']==0 and boot['metrics']['within_macro_roc']['n_valid']>0;fails(lambda:paired_video_cluster_bootstrap(pred,pred,gt,{'a':'g1'},B=2),ValueError);badpred={**pred,'a':np.array([.1,np.nan,.2,.3])};fails(lambda:paired_video_cluster_bootstrap(pred,badpred,gt,{'a':'g1','b':'g2'},B=2),RuntimeError)
 with tempfile.TemporaryDirectory() as x:
  td=Path(x);bundle=td/'labels.enc';bundle.write_bytes(b'synthetic ciphertext');predj={'frame_scores':{v:s.tolist() for v,s in pred.items()},'segments':{'a':[(1,3,.9)],'b':[(2,3,.8)]}};labels={'frame_gt':{v:y.tolist() for v,y in gt.items()},'segments':{'a':[(1,3)],'b':[(2,3)]}};fm={k:str(i)*64 for i,k in enumerate(('checkpoint_sha256','source_sha256','evaluator_sha256','environment_sha256','config_sha256','split_sha256','selection_sha256'),1)};fm['prediction_sha256']=hashlib.sha256(canonical_json(predj)).hexdigest();signed=sign_freeze_manifest(fm,b'signing-key');ev=SealedEvaluator(lambda _:labels,b'signing-key',True);ledger=td/'TEST_OPEN.jsonl';assert ev.evaluate(predj,bundle,signed,ledger,20)['frame_ap']>.99;events=[json.loads(z) for z in ledger.read_text().splitlines()];assert [z['event'] for z in events]==['OPEN_STARTED','COMPLETED'];fails(lambda:ev.evaluate(predj,bundle,signed,ledger),FileExistsError);tam=copy.deepcopy(signed);tam['signature_hmac_sha256']='0'*64;fails(lambda:SealedEvaluator(lambda _:labels,b'signing-key',True).evaluate(predj,bundle,tam,td/'tamper'),RuntimeError);fails(lambda:SealedEvaluator(lambda _:labels,b'signing-key').evaluate(predj,bundle,signed,td/'closed'),PermissionError)
  crash=td/'CRASH.jsonl';fails(lambda:SealedEvaluator(lambda _:(_ for _ in ()).throw(RuntimeError('decrypt crash')),b'signing-key',True).evaluate(predj,bundle,signed,crash),RuntimeError);assert [json.loads(z)['event'] for z in crash.read_text().splitlines()]==['OPEN_STARTED','FAILED'];fails(lambda:ev.evaluate(predj,bundle,signed,crash),FileExistsError)
 fails(lambda:guarded_annotation_path('/real/annotation.xlsx',False),PermissionError)
 print(json.dumps({'status':'PASS','coverage':['exact prereg bytes + platform:1 golden u=.8029728','rename/order invariance','real fixed remote/tamper','selfsealed_validation opaque manifest','atomic STARTED/COMPLETED/FAILED permanent ledger','tAP strict/noGT','public-private schema/coverage','binary/cohort/zero-mixed','exact-key cluster bootstrap']},indent=2))
if __name__=='__main__':main()
