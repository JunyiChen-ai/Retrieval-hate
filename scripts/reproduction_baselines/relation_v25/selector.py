#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from core import *
from reference_builder import load_bags,verify
from train import transform,ARMS,verify_permutation_manifest,verify_declaration
from val_predict import verify_outputs
from post_training_migration import verify as verify_post_training_migration
from post_access_correction import verify as verify_post_access_correction,correction_verify_outputs
def model(path,e,arm):
 x=torch.load(path,weights_only=False);keys={'schema','arm','seed','epochs','history','state_sha256'}
 if set(x)!=keys or x['schema']!='v25_checkpoint_v1' or x['arm']!=arm or not Path(path).stem.endswith(f"seed{x.get('seed')}") or not Path(path).stem.startswith(arm+'_') or x['epochs']!=list(range(6)) or len(x['history'])!=6 or len(x['state_sha256'])!=6 or any(set(h)!={'epoch','state'} or h['epoch']!=i for i,h in enumerate(x['history'])) or x['history'][e]['epoch']!=e:raise RuntimeError('checkpoint schema')
 if x['state_sha256'][e]!=canon_hash({k:v.tolist() for k,v in x['history'][e]['state'].items()}):raise RuntimeError('checkpoint tamper')
 m=V25(arm in ('real','permuted'));m.load_state_dict(x['history'][e]['state']);return m,x
def scores(m,rows):return np.array([float(m(r['g'],torch.tensor(r['z']))[0]) for r in rows])
def metric(y,s):return {'ap':float(average_precision_score(y,s)),'roc':float(roc_auc_score(y,s))}
def paired_ci(y,aa,bb,B=2000):
 rng=np.random.default_rng(25025);d=[];bad=0;n=len(y)
 for _ in range(B):
  ix=rng.integers(0,n,n)
  if len(set(y[ix]))<2:bad+=1;continue
  d.append(np.mean([average_precision_score(y[ix],a[ix])-average_precision_score(y[ix],b[ix]) for a,b in zip(aa,bb)]))
 return [float(x) for x in np.quantile(d,[.025,.975])],bad
def removal_ratio(real,shuffled):
 if len(real)!=3 or len(shuffled)!=3 or not np.isfinite(real+shuffled).all():raise RuntimeError('matched seeds')
 d=np.mean(np.asarray(real)-.5)
 if d<=0:raise RuntimeError('nonpositive gain')
 return float(1-np.mean(np.asarray(shuffled)-.5)/d)
def main():
 p=argparse.ArgumentParser();p.add_argument('--train-run',required=True);p.add_argument('--val-bags',required=True);p.add_argument('--reference',required=True);p.add_argument('--temporal-steward-report',required=True);p.add_argument('--val-prediction-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args();run=Path(a.train_run);prot=json.load(open(run/'protocol.json'));verify(a.reference)
 migration=verify_post_training_migration(allow_post_access=True);correction=verify_post_access_correction();old=migration['old_training_identities']
 checks={'trainer_sha256':sha(Path(__file__).with_name('train.py')),'core_sha256':sha(Path(__file__).with_name('core.py')),'selector_sha256':old['selector_sha256'],'inference_sha256':sha(Path(__file__).with_name('inference.py')),'reference_builder_sha256':sha(Path(__file__).with_name('reference_builder.py')),'val_predict_sha256':sha(Path(__file__).with_name('val_predict.py')),'permutation_manifest_sha256':sha(run/'permutation_manifest.json'),'protocol_declaration_sha256':old['protocol_declaration_sha256']}
 exact={'schema','status','bags_sha256','reference_manifest_sha256','trainer_sha256','core_sha256','selector_sha256','inference_sha256','reference_builder_sha256','val_predict_sha256','permutation_manifest_sha256','protocol_declaration_sha256','seeds','epochs','arms'}
 if set(prot)!=exact or prot.get('schema')!='v25_train_protocol_v1' or prot.get('status')!='TRAINED_NO_VAL_OR_TEST' or prot.get('seeds')!=list(SEEDS) or prot.get('epochs')!=list(range(6)) or prot.get('arms')!=list(ARMS) or prot.get('bags_sha256')!='16906241a8c4de70aa54bdca4f390ca84988ac27affa6e2835c0d02ff25f3e86' or prot.get('reference_manifest_sha256')!=sha(Path(a.reference)/'manifest.json') or any(prot.get(k)!=v for k,v in checks.items()):raise RuntimeError('protocol integrity')
 dp=migration['new_evaluation_identities']['current_declaration']['path'];decl=json.load(open(dp));tb=decl['identities']['train_bags']['path']
 if sha(dp)!=migration['new_evaluation_identities']['current_declaration']['sha256'] or decl.get('status')!='FINAL_AUTHORITATIVE_PRETRAINING_DECLARATION' or sha(tb)!=decl['identities']['train_bags']['sha256']:raise RuntimeError('historical declaration/train bags')
 train_rows=transform(load_bags(tb),a.reference,True);verify_permutation_manifest(train_rows,json.load(open(run/'permutation_manifest.json')))
 rows=transform(load_bags(a.val_bags,'val'),a.reference,False);y=np.array([r['y'] for r in rows]);surface={}
 for arm in ARMS:
  surface[arm]={}
  for e in range(6):
   ss=[scores(model(run/f'{arm}_seed{s}.pt',e,arm)[0],rows) for s in SEEDS];sm=[metric(y,x) for x in ss];surface[arm][str(e)]={'ap':float(np.mean([q['ap'] for q in sm])),'roc':float(np.mean([q['roc'] for q in sm])),'seed_metrics':sm,'scores':[x.tolist() for x in ss]}
 best=max(range(6),key=lambda e:(surface['real'][str(e)]['ap'],surface['real'][str(e)]['roc'],-e));real=[np.asarray(x) for x in surface['real'][str(best)]['scores']];epoch0=[np.asarray(x) for x in surface['real']['0']['scores']];gates={};cis={}
 for arm in ('permuted','negative_reference_only'):
  c=[np.asarray(x) for x in surface[arm][str(best)]['scores']];ci,bad=paired_ci(y,real,c);cis[arm]={'ci':ci,'invalid':bad};gates['ap_vs_'+arm]=surface['real'][str(best)]['ap']>=surface[arm][str(best)]['ap']+.005 and ci[0]>0
 ci,bad=paired_ci(y,real,epoch0);cis['epoch0']={'ci':ci,'invalid':bad};gates['ap_vs_epoch0']=surface['real'][str(best)]['ap']>=surface['real']['0']['ap']+.005 and ci[0]>0;gates['roc_noninferior']=surface['real'][str(best)]['roc']>=surface['real']['0']['roc']-.005
 pm=json.load(open(run/'permutation_manifest.json'));gates['permutation_identifiable']=all(pm[str(s)]['moved_video_fraction']>=.8 and pm[str(s)]['moved_instance_fraction']>=.8 for s in SEEDS);params=[]
 for s in SEEDS:
  m,_=model(run/f'real_seed{s}.pt',best,'real');params.append({'gamma':float(torch.clamp(m.gamma,0)),'weights':torch.softmax(m.wraw,0).tolist()})
 gates['activation']=all(x['gamma']>=.01 and max(x['weights'])<=.95 for x in params)
 vd=Path(a.val_prediction_dir);vm=json.load(open(vd/'manifest.json'));tr=json.load(open(a.temporal_steward_report));required={'status','post_access_provenance_correction_sha256','post_training_pre_temporal_migration_sha256','pre_temporal_eval_addendum_sha256','val_ids_sha256','val_bags_sha256','selected_epoch','checkpoint_sha256_by_seed','state_sha256_by_seed','reference_manifest_sha256','val_prediction_manifest_sha256','val_prediction_producer_sha256','raw_prediction_sha256_by_seed','shuffle_prediction_sha256_by_seed','within_macro_roc','within_gain','real_within_roc_by_seed','shuffled_within_roc_by_seed','paired_ci_lower'}
 add=Path(__file__).with_name('PRE_TEMPORAL_EVAL_ADDENDUM.json');ad=json.load(open(add))
 if set(tr)!=required or tr['status']!='VAL_TEMPORAL_STEWARD' or tr['post_access_provenance_correction_sha256']!=sha(Path(__file__).with_name('POST_ACCESS_PROVENANCE_CORRECTION_V2.json')) or tr['post_training_pre_temporal_migration_sha256']!=sha(Path(__file__).with_name('POST_TRAINING_PRE_TEMPORAL_MIGRATION.json')) or tr['pre_temporal_eval_addendum_sha256']!=sha(add) or ad.get('status')!='FROZEN_BEFORE_TEMPORAL_METRICS_ACCESS':raise RuntimeError('temporal schema/correction/migration/addendum')
 stale=json.load(open(correction['frozen_artifacts']['stale_temporal_report']['path']));repaired=dict(tr);repaired.pop('post_access_provenance_correction_sha256')
 if repaired!=stale or tr['val_prediction_manifest_sha256']!=correction['unchanged_evaluation_bindings']['prediction_manifest_sha256'] or tr['reference_manifest_sha256']!=correction['unchanged_evaluation_bindings']['reference_manifest_sha256']:raise RuntimeError('post-access metric/provenance invariant changed')
 ck={str(s):sha(run/f'real_seed{s}.pt') for s in SEEDS};st={str(s):model(run/f'real_seed{s}.pt',best,'real')[1]['state_sha256'][best] for s in SEEDS}
 vm=correction_verify_outputs(vd,canon_hash(sorted(r['id'] for r in rows)),best,ck,st,sha(Path(a.reference)/'manifest.json'));raw={str(s):sha(vd/f'seed{s}_raw.jsonl') for s in SEEDS};shuf={str(s):sha(vd/f'seed{s}_shuffle.jsonl') for s in SEEDS}
 if tr['val_ids_sha256']!=canon_hash(sorted(r['id'] for r in rows)) or tr['val_bags_sha256']!=sha(a.val_bags) or tr['selected_epoch']!=best or tr['checkpoint_sha256_by_seed']!=ck or tr['state_sha256_by_seed']!=st or tr['reference_manifest_sha256']!=sha(Path(a.reference)/'manifest.json') or tr['val_prediction_manifest_sha256']!=sha(vd/'manifest.json') or tr['val_prediction_producer_sha256']!=sha(Path(__file__).with_name('val_predict.py')) or tr['raw_prediction_sha256_by_seed']!=raw or tr['shuffle_prediction_sha256_by_seed']!=shuf:raise RuntimeError('temporal provenance')
 rr=removal_ratio(tr['real_within_roc_by_seed'],tr['shuffled_within_roc_by_seed']);gates['within']=tr['within_macro_roc']>.5 and tr['within_gain']>=.01 and tr['paired_ci_lower']>0;gates['shuffle']=rr>=.8;passed=all(gates.values());chosen=best if passed else 0
 states={str(s):{k:v.tolist() for k,v in model(run/f'real_seed{s}.pt',chosen,'real')[0].state_dict().items()} for s in SEEDS}
 if not passed:states={str(s):{k:v.tolist() for k,v in V25().state_dict().items()} for s in SEEDS}
 out={'status':'VIDEO_VAL_PASS_PENDING_TEST_SEAL' if passed else 'VAL_FAIL_EXACT_GLOBAL_FALLBACK','selected_epoch':chosen,'gates':gates,'paired_bootstrap':cis,'shuffle_removal_ratio':rr,'params':params,'surface':surface,'selected_states_by_seed':states,'fallback_frame_policy':'bit_exact_per_video_v16_global','within_roc_on_fallback':.5,'train_protocol_sha256':sha(run/'protocol.json'),'reference_manifest_sha256':sha(Path(a.reference)/'manifest.json'),'val_bags_sha256':sha(a.val_bags),'temporal_steward_report_sha256':sha(a.temporal_steward_report),'selector_sha256':sha(__file__),'inference_sha256':sha(Path(__file__).with_name('inference.py')),'test_opened':False,'test_seal_signed':False};Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
