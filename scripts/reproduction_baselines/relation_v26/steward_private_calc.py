#!/usr/bin/env python3
"""Private V26 validation metric calculator. This is the sole stats producer."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from artifacts import atomic,sha
from core import DESIGN_SHA
from steward import bootstrap_file,V25_MANIFEST,V25_MANIFEST_SHA
from video_selection import verify as verify_selection
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'relation_v25'));sys.path.insert(0,str(Path(__file__).resolve().parent.parent));from relation_v25.steward_val_artifact import decrypt_and_verify
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def metrics(pred,gt):
 yy=[];ss=[];wa=[];wr=[];ids=[]
 for v in sorted(gt):
  y=np.asarray(gt[v]['target_1hz']);valid=np.asarray(gt[v]['valid_1hz'],bool);s=np.asarray(pred[v],float)
  if len(s)!=len(y) or not np.isfinite(s).all():raise RuntimeError('prediction/GT length')
  yy.extend(y[valid]);ss.extend(s[valid])
  if len(np.unique(y[valid]))==2:wa.append(average_precision_score(y[valid],s[valid]));wr.append(roc_auc_score(y[valid],s[valid]));ids.append(v)
 return {'frame_ap':float(average_precision_score(yy,ss)),'frame_roc':float(roc_auc_score(yy,ss)),'within_ap':float(np.mean(wa)),'within_roc':float(np.mean(wr)),'mixed_ids':ids,'per_ap':wa,'per_roc':wr}
def qci(x):return np.quantile(np.asarray(x),[.025,.975],method='linear').tolist()
def constant_ap_paired(seed_metrics,constant_metrics,bm):
 d=np.mean([[s['per_ap'][i]-constant_metrics['per_ap'][i] for i in range(len(constant_metrics['per_ap']))] for s in seed_metrics],axis=0)
 return float(np.mean(d)),qci([float(np.mean(np.asarray(d)[ix])) for ix in bm])
def constant_ap_gate(point,ci):return point>=.01 and ci[0]>0
def load_boot(p,cohort,seed,ids):
 e={'path':str(Path(p).resolve()),'sha256':sha(p)};bootstrap_file(e,cohort,seed,ids);return json.load(open(p))['arrays'],e
def load_v25(ids):
 root=V25_MANIFEST.parent;m=json.load(open(V25_MANIFEST))
 if sha(V25_MANIFEST)!=V25_MANIFEST_SHA:raise RuntimeError('V25 authoritative manifest')
 req={'checkpoint_sha256_by_seed','epoch','evidence_config_sha256','evidence_manifest_sha256','files','ids_sha256','labels_read','producer_sha256','reducer_sha256','reference_manifest_sha256','schema','shuffle_rule','state_sha256_by_seed','status','test_read'}
 if set(m)!=req or m['schema']!='v25_val_predictions_v1' or m['epoch']!=2 or m['status']!='VAL_LABEL_GT_BLIND' or m['labels_read'] is not False or m['test_read'] is not False or m['producer_sha256']!=sha(Path(__file__).resolve().parent.parent/'relation_v25'/'val_predict.py') or m['reducer_sha256']!=sha(Path(__file__).resolve().parent.parent/'relation_v25'/'inference.py') or any(sha(root/n)!=h for n,h in m['files'].items()):raise RuntimeError('V25 live verifier')
 train=root.parents[0]/'train314'/'v25_training_seed234_2025_3407';ref=root.parents[0]/'train314'/'v25_train_negative_reference_frozen'/'manifest.json'
 if sha(ref)!=m['reference_manifest_sha256'] or any(sha(train/f'real_seed{s}.pt')!=m['checkpoint_sha256_by_seed'][str(s)] for s in (234,2025,3407)):raise RuntimeError('V25 checkpoint/reference replacement')
 by=[]
 for seed in (234,2025,3407):
  rows={}
  for line in open(root/f'seed{seed}_raw.jsonl'):
   r=json.loads(line);n=math.ceil(r['duration']);s=[];mask=[]
   for j in range(n):
    u=min(j+.5,r['duration']-1e-9);q=[z for a,b,z in zip(r['start'],r['end'],r['logits']) if a<=u<b or (b==r['duration'] and u==r['duration'])]
    s.append(float(1/(1+math.exp(-sum(q)/len(q)))) if q else float('nan'));mask.append(int(bool(q)))
   if not all(mask):raise RuntimeError('V25 coverage')
   rows[r['video_id']]=s
  if sorted(rows)!=ids:raise RuntimeError('V25 exact IDs')
  by.append(rows)
 return {v:np.mean([z[v] for z in by],axis=0).tolist() for v in ids},m
def run(args):
 gt_payload=decrypt_and_verify(args.encrypted_dir,args.key,args.val_manifest,args.qc,args.taxonomy,args.private_source,args.raw_id_map);gt=gt_payload['records'];ids=sorted(gt)
 if len(ids)!=32:raise RuntimeError('exact32')
 selection=verify_selection(args.selection,args.predictions,args.video_labels);selected_epoch=selection['selected_epoch'];v25,v25mfest=load_v25(ids)
 pms=[];seed_metrics=[];preds=[];shuffles=[];faith=[]
 for mp in args.predictions:
  from selector import verify_predictions
  pm,_=verify_predictions(mp);ep=str(selected_epoch);f=pm['files']['real'][ep]
  if sha(f['path'])!=f['sha256'] or any(set(pm[k])!={'path','sha256'} or sha(pm[k]['path'])!=pm[k]['sha256'] for k in ('features','reference','train_run')):raise RuntimeError('V26 prediction chain')
  r=json.load(open(f['path']))['records'];pred={v:r[v]['local'] for v in ids};preds.append(pred);seed_metrics.append(metrics(pred,gt));shuffles.append([{v:r[v]['shuffle100'][j] for v in ids} for j in range(100)]);faith.append({v:r[v]['faithfulness']['drop_top']-float(np.mean(r[v]['faithfulness']['drop_random100'])) for v in ids});pms.append({'path':str(Path(mp).resolve()),'sha256':sha(mp),'seed':pm['seed'],'prediction_file':{'path':f['path'],'sha256':f['sha256']},'features':pm['features'],'reference':pm['reference'],'train_run':pm['train_run']})
 mixed=seed_metrics[0]['mixed_ids'];pos=[v for v in ids if np.asarray(gt[v]['target_1hz']).any()];ba,eba=load_boot(args.bootstrap_all,'all32',26031,ids);bp,ebp=load_boot(args.bootstrap_positive,'positive',26032,pos);bm,ebm=load_boot(args.bootstrap_mixed,'mixed',26033,mixed)
 if any(x['mixed_ids']!=mixed for x in seed_metrics):raise RuntimeError('mixed cohort mismatch')
 v25m=metrics(v25,gt);mean=lambda k:float(np.mean([x[k] for x in seed_metrics]));droc=np.mean([[seed_metrics[s]['per_roc'][i]-v25m['per_roc'][i] for i in range(len(mixed))] for s in range(len(preds))],0);dap=np.mean([[seed_metrics[s]['per_ap'][i]-v25m['per_ap'][i] for i in range(len(mixed))] for s in range(len(preds))],0)
 broc=[float(np.mean(np.asarray(droc)[ix])) for ix in bm];bap=[float(np.mean(np.asarray(dap)[ix])) for ix in bm];shuffle_vals=[];paired=[]
 for j in range(100):shuffle_vals.append(float(np.mean([metrics(shuffles[s][j],gt)['within_roc'] for s in range(len(preds))])))
 for i,v in enumerate(mixed):paired.append(float(np.mean([seed_metrics[s]['per_roc'][i]-np.mean([metrics(shuffles[s][j],gt)['per_roc'][i] for j in range(100)]) for s in range(len(preds))])))
 bpair=[float(np.mean(np.asarray(paired)[ix])) for ix in bm];dur=np.asarray([gt[v]['duration'] for v in ids]);means=np.mean([[np.mean(pred[v]) for v in ids] for pred in preds],0);maxs=np.mean([[np.max(pred[v]) for v in ids] for pred in preds],0);v25means=np.asarray([np.mean(v25[v]) for v in ids]);v25max=np.asarray([np.max(v25[v]) for v in ids]);fv=np.mean([[faith[s][v] for v in pos] for s in range(len(preds))],0);bf=[float(np.mean(np.asarray(fv)[ix])) for ix in bp]
 first=json.load(open(pms[0]['prediction_file']['path']))['records'];constant={v:[float(first[v]['G'])]*len(gt[v]['target_1hz']) for v in ids};constantm=metrics(constant,gt);constant_gain,constant_ci=constant_ap_paired(seed_metrics,constantm,bm)
 stats={'frame_ap_by_seed':[x['frame_ap'] for x in seed_metrics],'frame_roc_by_seed':[x['frame_roc'] for x in seed_metrics],'frame_ap_mean':mean('frame_ap'),'frame_roc_mean':mean('frame_roc'),'within_roc_point':mean('within_roc'),'within_roc_ci':qci([float(np.mean(np.asarray([x['per_roc'] for x in seed_metrics])[:,ix])) for ix in bm]),'within_ap_point':mean('within_ap'),'constant_g_within_ap':constantm['within_ap'],'constant_g_within_ap_gain':constant_gain,'constant_g_within_ap_gain_ci':constant_ci,'within_ap_gain_v25':float(np.mean(dap)),'within_ap_gain_v25_ci':qci(bap),'ctw_v25_roc_diff':float(np.mean(droc)),'ctw_v25_roc_diff_ci':qci(broc),'shuffle_q975_gain':float(np.quantile(shuffle_vals,.975,method='linear')-.5),'paired_shuffle_ci':qci(bpair),'faithfulness_diff':float(np.mean(fv)),'faithfulness_ci':qci(bf),'duration_mean_abs':abs(float(spearmanr(dur,means).statistic)),'duration_max_abs':abs(float(spearmanr(dur,maxs).statistic)),'v25_duration_mean_abs':abs(float(spearmanr(dur,v25means).statistic)),'v25_duration_max_abs':abs(float(spearmanr(dur,v25max).statistic)),'variance_video_fraction':float(np.mean([np.std(np.mean([pred[v] for pred in preds],0))>0 for v in ids])),'coverage':1.,'eligible_mixed':len(mixed),'eligible_positive':len(pos)}
 gates={'within_roc':stats['within_roc_point']>=.5868 and stats['within_roc_ci'][0]>.5,'within_ap_constant':constant_gain>=.01 and constant_ci[0]>0,'within_ap_v25':stats['within_ap_gain_v25']>=.01 and stats['within_ap_gain_v25_ci'][0]>0,'v25_roc':stats['ctw_v25_roc_diff']>=.02 and stats['ctw_v25_roc_diff_ci'][0]>0,'shuffle':stats['shuffle_q975_gain']<=.2*(stats['within_roc_point']-.5) and stats['paired_shuffle_ci'][0]>0,'faithfulness':stats['faithfulness_diff']>=.02 and stats['faithfulness_ci'][0]>0,'duration':stats['duration_mean_abs']<=.20 and stats['duration_max_abs']<=.20 and stats['duration_mean_abs']<=stats['v25_duration_mean_abs']+.05 and stats['duration_max_abs']<=stats['v25_duration_max_abs']+.05,'variance':stats['variance_video_fraction']>=.95,'coverage':True}
 bind=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}
 v25_train=V25_MANIFEST.parents[1]/'train314'/'v25_training_seed234_2025_3407';v25_ref=V25_MANIFEST.parents[1]/'train314'/'v25_train_negative_reference_frozen'/'manifest.json'
 inputs={'encrypted_cipher':bind(Path(args.encrypted_dir)/'artifact.aesgcm'),'encrypted_manifest':bind(Path(args.encrypted_dir)/'manifest.json'),'ledger':bind(Path(args.encrypted_dir)/'access_ledger.json'),'public_val_manifest':bind(args.val_manifest),'qc':bind(args.qc),'taxonomy':bind(args.taxonomy),'private_source':bind(args.private_source),'raw_id_map':bind(args.raw_id_map),'gt_reducer':bind(args.gt_reducer),'calculator':bind(__file__),'selection':bind(args.selection),'predictions':pms,'v25_manifest':bind(V25_MANIFEST),'v25_provenance':bind(V25_MANIFEST.parent/'approved_run_provenance.json'),'v25_reference':bind(v25_ref),'v25_checkpoints':[bind(v25_train/f'real_seed{s}.pt') for s in (234,2025,3407)],'v25_verifier':bind(Path(__file__).resolve().parent.parent/'relation_v25'/'val_predict.py'),'v25_reducer':bind(Path(__file__).resolve().parent.parent/'relation_v25'/'inference.py'),'bootstraps':[eba,ebp,ebm]};payload={'schema':'v26_signed_temporal_report_v2','design_sha256':DESIGN_SHA,'selected_epoch':selected_epoch,'calculator_sha256':sha(__file__),'inputs':inputs,'all_ids':ids,'positive_ids':pos,'mixed_ids':mixed,'stats':stats,'gates':gates,'all_gates_pass':all(gates.values()),'test_opened':False};priv=Ed25519PrivateKey.from_private_bytes(Path(args.signing_key).read_bytes());payload['signature_hex']=priv.sign(canon(payload)).hex();atomic(args.out,payload);return payload
def main():
 p=argparse.ArgumentParser()
 for x in ('encrypted-dir','key','val-manifest','qc','taxonomy','private-source','raw-id-map','gt-reducer','selection','video-labels','bootstrap-all','bootstrap-positive','bootstrap-mixed','signing-key','out'):p.add_argument('--'+x,required=True)
 p.add_argument('--predictions',nargs='+',required=True);run(p.parse_args())
if __name__=='__main__':main()
