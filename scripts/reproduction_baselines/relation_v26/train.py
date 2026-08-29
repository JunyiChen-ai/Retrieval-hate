#!/usr/bin/env python3
import argparse,json,math,os,random,tempfile,copy
from pathlib import Path
import numpy as np,torch
from feature_manifest import verify as verify_features
from artifacts import atomic,sha
from core import CTW,Probe,ctw_loss,permutation,DESIGN_SHA,MIGRATION_SHA,ARCH,ch,fold,tensor_ch
from reference import verify_reference
def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.use_deterministic_algorithms(True)
def cuda5090():
 if not torch.cuda.is_available() or 'RTX 5090' not in torch.cuda.get_device_name(0):raise RuntimeError('V26 kill pilot requires RTX 5090')
 return torch.device('cuda')
def preload(rows,device):
 for r in rows:
  for k in ('X','masks','oof_b'):r[k]=[x.to(device=device) for x in r[k]]
  r['G']=r['G'].to(device=device,dtype=torch.float32);r['y']=r['y'].to(device=device,dtype=torch.float32)
def load_rows(feature_manifest,labels=None):
 m=verify_features(feature_manifest);lab=None
 if labels:
  lm=json.load(open(labels));req={'schema','design_sha256','split','ids','labels','feature_manifest_sha256','temporal_labels_read'}
  if set(lm)!=req or lm['schema']!='v26_train_video_labels_v1' or lm['design_sha256']!=DESIGN_SHA or lm['split']!='train' or lm['temporal_labels_read'] is not False or lm['feature_manifest_sha256']!=sha(feature_manifest) or sorted(lm['ids'])!=sorted(m['ids']) or set(lm['labels'])!=set(m['ids']) or any(lm['labels'][v] not in (0,1) for v in m['ids']):raise RuntimeError('weak label join')
  lab=lm['labels']
 rows=[]
 for v in m['ids']:
  r=json.load(open(m['records'][v]));T=len(r['seconds']);xs=[];ms=[]
  for f in ('visual','audio','text'):
   mask=torch.tensor([bool(z['availability'][('visual','audio','text').index(f)]) for z in r['seconds']])
   d={'visual':512,'audio':128,'text':768}[f];x=torch.zeros(T,d)
   for t,z in enumerate(r['seconds']):
    if mask[t]:x[t]=torch.tensor(z[f])
   xs.append(x);ms.append(mask)
  row={'id':v,'T':T,'X':xs,'masks':ms,'G':torch.tensor(float(r['G']))}
  if lab is not None:row['y']=torch.tensor(float(lab[v]))
  rows.append(row)
 return rows,m
def save_ckpt(path,model,seed,arm,epoch,steps,inputs):
 state={k:v.detach().cpu() for k,v in model.state_dict().items()} if model else {};payload={'schema':'v26_finite_rf_checkpoint_v2','design_sha256':DESIGN_SHA,'migration_sha256':MIGRATION_SHA,'architecture':ARCH,'seed':seed,'arm':arm,'epoch':epoch,'steps':steps,'inputs':inputs,'state':state};fd,t=tempfile.mkstemp(dir=Path(path).parent,prefix=Path(path).name+'.');os.close(fd);torch.save(payload,t);os.replace(t,path);return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def attach_oof(rows,ref):
 for r in rows:
  x=ref['oof'][r['id']]
  if x['fold']!=fold(r['id']) or sha(x['path'])!=x['sha256']:raise RuntimeError('OOF binding')
  r['oof_b']=torch.load(x['path'],map_location='cpu',weights_only=True)
  if ch([z.tolist() for z in r['oof_b']])!=x['tensor_sha256'] or any(z.shape!=r['X'][i].shape for i,z in enumerate(r['oof_b'])):raise RuntimeError('OOF tensor')
def train_arm(rows,arm,seed,out,inputs,epochs=8,initial_state=None,device=None):
 seed_all(seed);model=CTW(model_seed=seed);model.load_state_dict(copy.deepcopy(initial_state) if initial_state is not None else model.state_dict());model.to(device or cuda5090());opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);out=Path(out);out.mkdir(parents=True,exist_ok=False);ck=[];steps=0
 ck.append(save_ckpt(out/'epoch0.pt',model,seed,arm,0,0,inputs))
 use=rows
 if arm=='permuted':
  use,pm=permutation(rows,seed);non=sum(x['nonself'] for x in pm)/len(pm);inst=sum(x['T'] for x in pm if x['nonself'])/sum(x['T'] for x in pm)
  donors=[x['donor'] for x in pm];pre_av=sorted(tuple(int(x.sum()) for x in r['masks']) for r in rows);post_av=sorted(tuple(int(x.sum()) for x in r['masks']) for r in use)
  if non<.8 or inst<.8 or sorted(donors)!=sorted(r['id'] for r in rows) or sorted(x['T'] for x in use)!=sorted(x['T'] for x in rows) or pre_av!=post_av or any(x['intervention_coverage']<=0 for x in pm):raise RuntimeError('permutation coverage')
  atomic(out/'permutation.json',{'schema':'v26_permutation_v2','design_sha256':DESIGN_SHA,'seed':seed,'rows':pm,'video_fraction':non,'instance_fraction':inst,'donor_bijection':True,'availability_multiset_sha256':ch(pre_av),'pre_seq_sha256':ch([(r['id'],r['T'],tensor_ch(r['X']),tensor_ch(r['masks']),tensor_ch(r['oof_b'])) for r in rows]),'post_seq_sha256':ch([(r['id'],r['T'],tensor_ch(r['X']),tensor_ch(r['masks']),tensor_ch(r['oof_b'])) for r in use])})
 for ep in range(1,epochs+1):
  for s in range(0,len(use),4):
   losses=[]
   for r in use[s:s+4]:
    b=[torch.zeros_like(x) for x in r['X']] if arm=='negative_mean' else r['oof_b'];loss,_,_=ctw_loss(model,r,b);losses.append(loss)
   opt.zero_grad();torch.stack(losses).mean().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1
  ck.append(save_ckpt(out/f'epoch{ep}.pt',model,seed,arm,ep,steps,inputs))
 return ck,steps
def train_probe(rows,out,inputs,device=None):
 seed_all(26027);m=Probe().to(device or cuda5090());opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
 for _ in range(8):
  for s in range(0,len(rows),4):
   ls=[torch.nn.functional.binary_cross_entropy_with_logits(m(r['X'],r['masks']),r['y']) for r in rows[s:s+4]];opt.zero_grad();torch.stack(ls).mean().backward();opt.step()
 return save_ckpt(Path(out)/'probe.pt',m,26027,'probe',8,math.ceil(len(rows)/4)*8,inputs)
def run(features,labels,reference,out,seed=234,epochs=8,device='cuda'):
 rows,fm=load_rows(features,labels);rawref=json.load(open(reference));ref=verify_reference(reference,features,labels,rawref['inputs']['val_features']['path']);attach_oof(rows,ref);out=Path(out);out.mkdir(parents=True,exist_ok=False);arms={};steps={};row_sha=ch([(r['id'],r['T'],tensor_ch(r['X']),tensor_ch(r['masks']),tensor_ch(r['oof_b']),float(r['G']),float(r['y'])) for r in rows]);inputs={'features_sha256':sha(features),'features_root_sha256':fm['root_sha256'],'labels_sha256':sha(labels),'reference_manifest_sha256':sha(reference),'rows_sha256':row_sha,'core_sha256':sha(Path(__file__).with_name('core.py')),'trainer_sha256':sha(__file__),'design_sha256':DESIGN_SHA,'migration_sha256':MIGRATION_SHA,'architecture':ARCH}
 atomic(out/'fallback.json',{'schema':'v26_fallback_v1','design_sha256':DESIGN_SHA,'scores':{r['id']:float(r['G']) for r in rows},'raw_G_bit_exact':True})
 dev=cuda5090() if device=='cuda' else (_ for _ in ()).throw(RuntimeError('only --device cuda is legal'));preload(rows,dev);canonical=CTW(model_seed=seed).state_dict();epoch0_hash=ch({k:v.tolist() for k,v in canonical.items()})
 for arm in ('real','permuted','negative_mean'):arms[arm],steps[arm]=train_arm(rows,arm,seed,out/arm,inputs,epochs,canonical,dev)
 probe=train_probe(rows,out,inputs);man={'schema':'v26_finite_rf_train_run_v2','design_sha256':DESIGN_SHA,'migration_sha256':MIGRATION_SHA,'architecture':ARCH,'epoch0_state_sha256':epoch0_hash,'seed':seed,'epochs':list(range(epochs+1)),'arms':arms,'steps':steps,'matched_steps':len(set(steps.values()))==1,'probe':probe,'features':{'path':str(Path(features).resolve()),'sha256':sha(features),'root_sha256':fm['root_sha256']},'labels':{'path':str(Path(labels).resolve()),'sha256':sha(labels)},'reference':{'path':str(Path(reference).resolve()),'sha256':sha(reference)},'rows_sha256':row_sha,'source_hashes':{'core':sha(Path(__file__).with_name('core.py')),'train':sha(__file__)},'test_read':False};atomic(out/'manifest.json',man);verify_train_run(out/'manifest.json');return man
def verify_train_run(path):
 m=json.load(open(path));keys={'schema','design_sha256','migration_sha256','architecture','epoch0_state_sha256','seed','epochs','arms','steps','matched_steps','probe','features','labels','reference','rows_sha256','source_hashes','test_read'}
 if set(m)!=keys or m['schema']!='v26_finite_rf_train_run_v2' or (m['design_sha256'],m['migration_sha256'],m['architecture'])!=(DESIGN_SHA,MIGRATION_SHA,ARCH) or m['test_read'] is not False or m['source_hashes']!={'core':sha(Path(__file__).with_name('core.py')),'train':sha(__file__)} or sha(m['features']['path'])!=m['features']['sha256'] or sha(m['labels']['path'])!=m['labels']['sha256'] or sha(m['reference']['path'])!=m['reference']['sha256'] or not m['matched_steps'] or len(set(m['steps'].values()))!=1:raise RuntimeError('train manifest')
 rawref=json.load(open(m['reference']['path']));verify_reference(m['reference']['path'],m['features']['path'],m['labels']['path'],rawref['inputs']['val_features']['path'])
 expected={'features_sha256':m['features']['sha256'],'features_root_sha256':m['features']['root_sha256'],'labels_sha256':m['labels']['sha256'],'reference_manifest_sha256':m['reference']['sha256'],'rows_sha256':m['rows_sha256'],'core_sha256':m['source_hashes']['core'],'trainer_sha256':m['source_hashes']['train'],'design_sha256':DESIGN_SHA,'migration_sha256':MIGRATION_SHA,'architecture':ARCH}
 for arm,cks in m['arms'].items():
  for ep,e in enumerate(cks):
   if sha(e['path'])!=e['sha256']:raise RuntimeError('checkpoint bytes')
   c=torch.load(e['path'],map_location='cpu',weights_only=False)
   if set(c)!={'schema','design_sha256','migration_sha256','architecture','seed','arm','epoch','steps','inputs','state'} or c['schema']!='v26_finite_rf_checkpoint_v2' or (c['design_sha256'],c['migration_sha256'],c['architecture'],c['seed'],c['arm'],c['epoch'],c['inputs'])!=(DESIGN_SHA,MIGRATION_SHA,ARCH,m['seed'],arm,ep,expected):raise RuntimeError('checkpoint nested')
   if ep==0 and ch({k:v.tolist() for k,v in c['state'].items()})!=m['epoch0_state_sha256']:raise RuntimeError('epoch0 arm state mismatch')
 if sha(m['probe']['path'])!=m['probe']['sha256']:raise RuntimeError('probe bytes')
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--labels',required=True);p.add_argument('--reference',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=234,choices=(234,2025,3407));p.add_argument('--epochs',type=int,default=8);p.add_argument('--device',required=True,choices=('cuda',));a=p.parse_args();run(a.features,a.labels,a.reference,a.out,a.seed,a.epochs,a.device)
if __name__=='__main__':main()
