#!/usr/bin/env python3
"""Label/GT-free validation prediction producer for temporal steward."""
import argparse,json,random,hashlib
from pathlib import Path
import torch
from core import *
from reference_builder import verify
from train import ARMS,verify_declaration
sys_path=Path(__file__).resolve().parents[1]/'relation_v24'
import sys;sys.path.insert(0,str(sys_path))
from steward_val_join import load_evidence
def model(path,e,arm):
 x=torch.load(path,weights_only=False)
 if x.get('schema')!='v25_checkpoint_v1' or x.get('arm')!=arm or x.get('epochs')!=list(range(6)) or x['history'][e]['epoch']!=e or x['state_sha256'][e]!=canon_hash({k:v.tolist() for k,v in x['history'][e]['state'].items()}):raise RuntimeError('checkpoint identity')
 m=V25(True);m.load_state_dict(x['history'][e]['state']);return m,x
def verify_outputs(root,ids_sha,epoch,checkpoints,states,reference_sha):
 root=Path(root);m=json.load(open(root/'manifest.json'));keys={'schema','status','epoch','ids_sha256','evidence_manifest_sha256','evidence_config_sha256','reference_manifest_sha256','checkpoint_sha256_by_seed','state_sha256_by_seed','files','shuffle_rule','producer_sha256','reducer_sha256','labels_read','test_read'}
 d=verify_declaration();ve=d['identities']['authoritative_val_evidence_manifest'];vc=d['identities']['authoritative_val_evidence_config']
 if set(m)!=keys or m['schema']!='v25_val_predictions_v1' or m['status']!='VAL_LABEL_GT_BLIND' or m['labels_read'] is not False or m['test_read'] is not False or m['epoch']!=epoch or m['ids_sha256']!=ids_sha or m['checkpoint_sha256_by_seed']!=checkpoints or m['state_sha256_by_seed']!=states or m['reference_manifest_sha256']!=reference_sha or m['evidence_manifest_sha256']!=ve['sha256'] or m['evidence_config_sha256']!=vc['sha256'] or m['producer_sha256']!=sha(__file__) or m['reducer_sha256']!=sha(Path(__file__).with_name('inference.py')):raise RuntimeError('val prediction identity')
 expected={f'seed{s}_{k}.jsonl' for s in SEEDS for k in ('raw','shuffle')}
 if set(m['files'])!=expected or any(sha(root/n)!=m['files'][n] for n in expected):raise RuntimeError('val prediction raw tamper')
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--val-evidence',required=True);p.add_argument('--reference',required=True);p.add_argument('--train-run',required=True);p.add_argument('--epoch',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
 if a.epoch not in range(6):raise RuntimeError('epoch')
 verify(a.reference)
 decl=verify_declaration();ve=decl['identities']['authoritative_val_evidence_manifest'];vc=decl['identities']['authoritative_val_evidence_config']
 if str(Path(a.val_evidence).resolve()/'evidence_manifest.json')!=ve['path'] or str(Path(a.val_evidence).resolve()/'preregistered_config.json')!=vc['path'] or sha(ve['path'])!=ve['sha256'] or sha(vc['path'])!=vc['sha256']:raise RuntimeError('non-authoritative validation evidence')
 em,cfg,recs,ep,cp=load_evidence(a.val_evidence)
 run=Path(a.train_run);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 refs=[json.load(open(Path(a.reference)/f'{f}_full.json')) for f in ('text','multimodal')];files={};states={};cks={}
 for seed in SEEDS:
  ck=run/f'real_seed{seed}.pt';m,x=model(ck,a.epoch,'real');cks[str(seed)]=sha(ck);states[str(seed)]=x['state_sha256'][a.epoch];raw=[];shuf=[]
  for vid in sorted(recs):
   r=recs[vid];ws=r['windows'];z=torch.tensor([ecdf_logit([w[f'{f}_isolated_score'] for w in ws],refs[i]) for i,f in enumerate(('text','multimodal'))]);ell=m.local(z).detach().tolist();row={'video_id':vid,'duration':r['duration'],'window_index':[w['window_index'] for w in ws],'start':[w['start'] for w in ws],'end':[w['end'] for w in ws],'logits':ell};raw.append(row);q=list(ell);random.Random(25026+seed+int(hashlib.sha256(vid.encode()).hexdigest()[:8],16)).shuffle(q);shuf.append({**row,'logits':q})
  for kind,data in (('raw',raw),('shuffle',shuf)):
   fp=out/f'seed{seed}_{kind}.jsonl';fp.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in data));files[fp.name]=sha(fp)
 man={'schema':'v25_val_predictions_v1','status':'VAL_LABEL_GT_BLIND','epoch':a.epoch,'ids_sha256':canon_hash(sorted(recs)),'evidence_manifest_sha256':sha(ep),'evidence_config_sha256':sha(cp),'reference_manifest_sha256':sha(Path(a.reference)/'manifest.json'),'checkpoint_sha256_by_seed':cks,'state_sha256_by_seed':states,'files':files,'shuffle_rule':'seed=25026+model_seed+sha256(video_id)prefix8','producer_sha256':sha(__file__),'reducer_sha256':sha(Path(__file__).with_name('inference.py')),'labels_read':False,'test_read':False};(out/'manifest.json').write_text(json.dumps(man,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
