#!/usr/bin/env python3
"""Signed, video-label-only V26 epoch selection artifact."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from artifacts import atomic,sha
from core import DESIGN_SHA,bootstrap_indices
from selector import verify_predictions

KEY_MANIFEST=Path('/home/jehc223/Hate-follow-up/results/steward_private/thvl_bench/v26_steward_key_frozen/public_manifest.json')
KEY_MANIFEST_SHA='38a510ab4dca9bdfacb3551385d01ad8053eb157176f6ce20c7b8ba3730d0f35'
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def labels(path,ids):
 x=json.load(open(path));req={'corpus','label_semantics','records','schema_version','split'}
 if set(x)!=req or x['schema_version']!='v24_video_labels_v1' or x['corpus']!='thvl' or x['split']!='val' or x['label_semantics']!='any_target_video_level' or any(set(r)!={'video_id','any_target_label'} or r['any_target_label'] not in (0,1) for r in x['records']):raise RuntimeError('video-label-only artifact')
 d={r['video_id']:r['any_target_label'] for r in x['records']}
 if len(d)!=32 or sorted(d)!=ids:raise RuntimeError('video label exact32')
 return np.asarray([d[v] for v in ids])
def calculate(predictions,label_path):
 verified=[verify_predictions(p) for p in predictions];ids=verified[0][1]
 if any(z[1]!=ids for z in verified) or len({z[0]['seed'] for z in verified})!=len(verified):raise RuntimeError('prediction cohort/seeds')
 y=labels(label_path,ids);metrics={a:[] for a in ('real','permuted','negative_mean')};scores={}
 for arm in metrics:
  for ep in range(9):
   per=[]
   for pm,_ in verified:
    r=json.load(open(pm['files'][arm][str(ep)]['path']))['records'];s=np.asarray([r[v]['video_logit'] for v in ids]);per.append({'ap':float(average_precision_score(y,s)),'roc':float(roc_auc_score(y,s))});scores[(pm['seed'],arm,ep)]=s
   metrics[arm].append({'epoch':ep,'ap':float(np.mean([q['ap'] for q in per])),'roc':float(np.mean([q['roc'] for q in per])),'by_seed':per})
 ep=max(metrics['real'],key=lambda q:(q['ap'],q['roc'],-q['epoch']))['epoch'];B=bootstrap_indices(len(ids),2000,26031,y);g={}
 base=np.mean([scores[(pm['seed'],'real',0)] for pm,_ in verified],0);real=np.mean([scores[(pm['seed'],'real',ep)] for pm,_ in verified],0)
 def ci(z):return np.quantile(z,[.025,.975],method='linear').tolist()
 dif=[average_precision_score(y[ix],real[ix])-average_precision_score(y[ix],base[ix]) for ix in B]
 control={}
 for a in ('permuted','negative_mean'):
  c=np.mean([scores[(pm['seed'],a,ep)] for pm,_ in verified],0);d=[average_precision_score(y[ix],real[ix])-average_precision_score(y[ix],c[ix]) for ix in B];control[a]={'ap_point':metrics['real'][ep]['ap']-metrics[a][ep]['ap'],'ap_ci':ci(d),'roc_point':metrics['real'][ep]['roc']-metrics[a][ep]['roc']}
 g={'video_ap_gain':metrics['real'][ep]['ap']-metrics['real'][0]['ap'],'video_ap_ci':ci(dif),'video_roc_noninferiority':metrics['real'][ep]['roc']>=metrics['real'][0]['roc']-.005,'controls':control};g['pass']=g['video_ap_gain']>=.005 and g['video_ap_ci'][0]>0 and g['video_roc_noninferiority'] and all(v['ap_point']>=.005 and v['ap_ci'][0]>0 for v in control.values())
 return verified,ids,metrics,ep,g
def verify(path,predictions,label_path):
 from steward import pinned_public_key
 x=json.load(open(path));sig=x.pop('signature_hex',None)
 if set(x)!={'schema','design_sha256','status','selected_epoch','selection_rule','metrics','gates','prediction_manifests','video_labels','producer_sha256','test_read'} or x['schema']!='v26_signed_video_selection_v1' or x['design_sha256']!=DESIGN_SHA or x['producer_sha256']!=sha(__file__) or x['test_read'] is not False:raise RuntimeError('selection schema')
 pinned_public_key().verify(bytes.fromhex(sig),canon(x));verified,ids,m,e,g=calculate(predictions,label_path)
 bindings=[{'path':str(Path(p).resolve()),'sha256':sha(p),'seed':z[0]['seed']} for p,z in zip(predictions,verified)]
 if x['prediction_manifests']!=bindings or x['video_labels']!={'path':str(Path(label_path).resolve()),'sha256':sha(label_path)} or x['selected_epoch']!=e or x['metrics']!=m or x['gates']!=g or x['selection_rule']!='mean_seed_real_only_(AP,ROC,-epoch)' or x['status']!=('VIDEO_VAL_PASS_PENDING_TEMPORAL' if g['pass'] else 'FINAL_FALLBACK'):raise RuntimeError('selection recomputation')
 x['signature_hex']=sig;return x
def run(predictions,label_path,key,out):
 verified,ids,m,e,g=calculate(predictions,label_path);x={'schema':'v26_signed_video_selection_v1','design_sha256':DESIGN_SHA,'status':'VIDEO_VAL_PASS_PENDING_TEMPORAL' if g['pass'] else 'FINAL_FALLBACK','selected_epoch':e,'selection_rule':'mean_seed_real_only_(AP,ROC,-epoch)','metrics':m,'gates':g,'prediction_manifests':[{'path':str(Path(p).resolve()),'sha256':sha(p),'seed':z[0]['seed']} for p,z in zip(predictions,verified)],'video_labels':{'path':str(Path(label_path).resolve()),'sha256':sha(label_path)},'producer_sha256':sha(__file__),'test_read':False};x['signature_hex']=Ed25519PrivateKey.from_private_bytes(Path(key).read_bytes()).sign(canon(x)).hex();atomic(out,x);return verify(out,predictions,label_path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',nargs='+',required=True);p.add_argument('--video-labels',required=True);p.add_argument('--signing-key',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.predictions,a.video_labels,a.signing_key,a.out)
if __name__=='__main__':main()
