#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from artifacts import atomic,sha
from core import DESIGN_SHA,bootstrap_indices
from feature_manifest import verify as verify_features
from train_control_v3 import verify_train_run
from reference import verify_reference
from steward import verify_signed_report
PRED_KEYS={'schema','design_sha256','seed','features','reference','train_run','files','source_sha256','test_read'}
def verify_predictions(path):
 pm=json.load(open(path))
 if set(pm)!=PRED_KEYS or pm['schema']!='v26_val_prediction_manifest_v1' or pm['design_sha256']!=DESIGN_SHA or pm['test_read'] is not False or pm['source_sha256']!=sha(Path(__file__).with_name('val_predict.py')) or set(pm['files'])!={'real','permuted','negative_mean'}:raise RuntimeError('prediction manifest')
 for k in ('features','reference','train_run'):
  if set(pm[k])!={'path','sha256'} or sha(pm[k]['path'])!=pm[k]['sha256']:raise RuntimeError('prediction input')
 fm=verify_features(pm['features']['path']);ids=sorted(fm['ids'])
 if fm['split']!='val' or len(ids)!=32:raise RuntimeError('exact32 val')
 tm=verify_train_run(pm['train_run']['path']);verify_reference(pm['reference']['path'],tm['features']['path'],tm['labels']['path'],pm['features']['path'])
 if pm['seed']!=tm['seed'] or pm['reference']['path']!=tm['reference']['path']:raise RuntimeError('prediction chain')
 for arm,eps in pm['files'].items():
  if set(eps)!=set(map(str,range(9))):raise RuntimeError('prediction epochs')
  for ep,e in eps.items():
   if set(e)!={'path','sha256'} or sha(e['path'])!=e['sha256']:raise RuntimeError('prediction file')
   x=json.load(open(e['path']))
   if set(x)!={'schema','design_sha256','arm','epoch','seed','records','labels_or_gt_read'} or x['schema']!='v26_val_predictions_v1' or (x['design_sha256'],x['arm'],x['epoch'],x['seed'],x['labels_or_gt_read'])!=(DESIGN_SHA,arm,int(ep),pm['seed'],False) or sorted(x['records'])!=ids:raise RuntimeError('prediction file schema')
   for v,r in x['records'].items():
    T=len(json.load(open(fm['records'][v]))['seconds']);req={'G','video_logit','local','effects','shuffle100','faithfulness','duration','epoch0_G_exact'}
    if set(r)!=req or r['duration']!=T or not np.isfinite([r['G'],r['video_logit'],*r['local'],*r['effects']]).all() or len(r['local'])!=T or len(r['effects'])!=T or len(r['shuffle100'])!=100 or any(len(z)!=T or not np.isfinite(z).all() for z in r['shuffle100']):raise RuntimeError('prediction record')
 return pm,ids
def select(real):return max(real,key=lambda e:(e['ap'],e['roc'],-e['epoch']))
def ci(x):return np.quantile(np.asarray(x),[.025,.975],method='linear').tolist()
def run(pred_manifest,val_labels,out,temporal_report=None):
 out_path=Path(out)
 pm,pids=verify_predictions(pred_manifest);lm=json.load(open(val_labels));req={'schema','design_sha256','split','ids','labels','prediction_manifest_sha256','temporal_labels_read'}
 if set(lm)!=req or lm['schema']!='v26_val_video_labels_v1' or lm['design_sha256']!=DESIGN_SHA or lm['split']!='val' or lm['prediction_manifest_sha256']!=sha(pred_manifest) or lm['temporal_labels_read'] is not False:raise RuntimeError('val video labels')
 ids=sorted(lm['ids']);
 if ids!=pids:raise RuntimeError('label/prediction IDs')
 y=np.array([lm['labels'][v] for v in ids]);metrics={}
 for arm,eps in pm['files'].items():
  metrics[arm]=[]
  for ep in range(9):
   f=eps[str(ep)]
   if sha(f['path'])!=f['sha256']:raise RuntimeError('prediction tamper')
   x=json.load(open(f['path']));s=np.array([x['records'][v]['video_logit'] for v in ids]);metrics[arm].append({'epoch':ep,'ap':float(average_precision_score(y,s)),'roc':float(roc_auc_score(y,s))})
 chosen=select(metrics['real']);ep=chosen['epoch'];fallback=json.load(open(pm['files']['real']['0']['path']));real=json.load(open(pm['files']['real'][str(ep)]['path']));controls={a:json.load(open(pm['files'][a][str(ep)]['path'])) for a in ('permuted','negative_mean')};B=bootstrap_indices(len(ids),2000,26031,y);diff=[];control_diff={a:{'ap':[],'roc':[]} for a in controls}
 for ix in B:
  yy=y[ix]
  if len(set(yy))<2:raise RuntimeError('bootstrap')
  diff.append(average_precision_score(yy,[real['records'][ids[i]]['video_logit'] for i in ix])-average_precision_score(yy,[fallback['records'][ids[i]]['video_logit'] for i in ix]))
  for a,c in controls.items():
   rs=[real['records'][ids[i]]['video_logit'] for i in ix];cs=[c['records'][ids[i]]['video_logit'] for i in ix];control_diff[a]['ap'].append(average_precision_score(yy,rs)-average_precision_score(yy,cs));control_diff[a]['roc'].append(roc_auc_score(yy,rs)-roc_auc_score(yy,cs))
 video_gate=chosen['ap']-metrics['real'][0]['ap']>=.005 and ci(diff)[0]>0 and chosen['roc']>=metrics['real'][0]['roc']-.005
 status='VIDEO_VAL_PASS_PENDING_TEMPORAL' if video_gate else 'FINAL_FALLBACK'
 control_point={a:{'ap':chosen['ap']-metrics[a][ep]['ap'],'roc':chosen['roc']-metrics[a][ep]['roc']} for a in controls};gates={'video':video_gate,'ap_diff':chosen['ap']-metrics['real'][0]['ap'],'ap_diff_ci':ci(diff),'roc_diff':chosen['roc']-metrics['real'][0]['roc'],'control_point':control_point,'control_paired_ci':{a:{k:ci(x) for k,x in z.items()} for a,z in control_diff.items()},'controls_pass':all(control_point[a]['ap']>=.005 and ci(z['ap'])[0]>0 for a,z in control_diff.items())}
 video_gate=video_gate and gates['controls_pass'];gates['video']=video_gate;status='VIDEO_VAL_PASS_PENDING_TEMPORAL' if video_gate else 'FINAL_FALLBACK'
 if temporal_report:
  tr=verify_signed_report(temporal_report,pred_manifest)
  if tr.get('selected_epoch')!=ep:raise RuntimeError('temporal report epoch')
  gates['temporal']=tr['all_gates_pass'];status='FINAL_PASS' if video_gate and tr['all_gates_pass'] else 'FINAL_FALLBACK'
 result={'schema':'v26_selection_v1','design_sha256':DESIGN_SHA,'selected_epoch':ep,'selection_rule':'real_only_(AP,ROC,-epoch)','metrics':metrics,'passive_controls':{a:metrics[a][ep] for a in ('permuted','negative_mean')},'gates':gates,'status':status,'prediction_manifest_sha256':sha(pred_manifest),'test_authorized':status=='FINAL_PASS'};atomic(out_path,result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--val-labels',required=True);p.add_argument('--temporal-report');p.add_argument('--out',required=True);a=p.parse_args();run(a.predictions,a.val_labels,a.out,a.temporal_report)
if __name__=='__main__':main()
