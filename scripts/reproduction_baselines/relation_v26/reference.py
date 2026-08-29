#!/usr/bin/env python3
import argparse,json,random,os,tempfile,platform
from pathlib import Path
import numpy as np,torch
from core import Decoder,fold,DESIGN_SHA,ch,tensor_ch
from artifacts import atomic,sha
DIMS=(512,128,768);CONFIG={'epochs':20,'batch_videos':16,'lr':1e-3,'weight_decay':1e-4,'huber_delta':1.0,'seed':26026,'last_short_batch':'mean_over_present_videos'}
MANIFEST_KEYS={'schema','design_sha256','config','runtime','sources','inputs','folds','oof','full','val_backgrounds','labels_used','status'}
FROZEN_REFERENCE=Path('/home/jehc223/Hate-follow-up/results/steward_private/thvl_bench/train314/v26_negative_reference_frozen/manifest.json');FROZEN_REFERENCE_SHA='0cba3535d2098444ee50e7a548012296d5b7989809a72e9d6f9c82abc428d5b9';FROZEN_REFERENCE_SOURCES={'core_sha256':'c8b4668db3f1585c484e7b946578d32b7412f9ec05850318b28334991c6e421a','reference_sha256':'f1f81d44fa7248f0faf0cfd9803abf24257df964d3570d977418f26439086f09','train_loader_sha256':'190c0d1508825efad4dc81cd5797f8d11da6d1c83299ea580a595dd007f3dd3b'}
def source_identities(train_path=None):
 train_path=Path(train_path) if train_path is not None else Path(__file__).with_name('train.py')
 return {'core_sha256':sha(Path(__file__).with_name('core.py')),'reference_sha256':sha(__file__),'train_loader_sha256':sha(train_path)}
def seed_all(s=26026):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.use_deterministic_algorithms(True)
def row_identity(r):return {'T':r['T'],'X_sha256':tensor_ch(r['X']),'mask_sha256':tensor_ch(r['masks']),'availability':[int(x.sum()) for x in r['masks']],'G':float(r['G'])}
def predict(models,row):
 out=[]
 with torch.no_grad():
  for f,d in enumerate(DIMS):out.append(torch.stack([models[f](row['X'][f],row['masks'][f],t) if row['masks'][f][t] else torch.zeros(d) for t in range(row['T'])]))
 return out
def video_recon_loss(mods,r):
 seconds=[]
 for t in range(r['T']):
  ls=[torch.nn.functional.huber_loss(mods[f](r['X'][f],r['masks'][f],t),r['X'][f][t],delta=1.,reduction='mean') for f in range(3) if r['masks'][f][t]]
  if ls:seconds.append(torch.stack(ls).mean())
 if not seconds:raise RuntimeError('negative has no available target')
 return torch.stack(seconds).mean()
def fit(neg):
 if not neg:raise RuntimeError('empty negative reference')
 seed_all();mods=[Decoder(d) for d in DIMS];opt=torch.optim.AdamW([p for m in mods for p in m.parameters()],lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
 rr=sorted(neg,key=lambda x:x['id'])
 for _ in range(CONFIG['epochs']):
  for s in range(0,len(rr),CONFIG['batch_videos']):
   losses=[video_recon_loss(mods,r) for r in rr[s:s+CONFIG['batch_videos']]];opt.zero_grad();torch.stack(losses).mean().backward();opt.step()
 return mods
def audit_zero(mods,rows):
 gates=[]
 for f,m in enumerate(mods):
  found=False
  for r in rows:
   for t in range(r['T']):
    if r['masks'][f][t]:
     x=r['X'][f].clone();a=x.clone();b=x.clone();a[max(0,t-1):min(r['T'],t+2)]=1.2345;b[max(0,t-1):min(r['T'],t+2)]=-2.3456
     with torch.no_grad():oa=m(a,r['masks'][f],t);ob=m(b,r['masks'][f],t)
     if not torch.equal(oa,ob):raise RuntimeError('center value influence')
     z=x.clone().requires_grad_(True);m(z,r['masks'][f],t).sum().backward();g=z.grad[max(0,t-1):min(r['T'],t+2)]
     if not torch.equal(g,torch.zeros_like(g)):raise RuntimeError('center gradient influence')
     gates.append({'modality':f,'video_id':r['id'],'t':t,'value_sha256':ch(oa.tolist()),'gradient_zero':True});found=True;break
   if found:break
  if not found:raise RuntimeError('gate target unavailable')
 return gates
def save_state(path,mods):
 fd,t=tempfile.mkstemp(dir=Path(path).parent,prefix=Path(path).name+'.');os.close(fd);torch.save([m.state_dict() for m in mods],t);os.replace(t,path)
def save_tensor(path,x):
 fd,t=tempfile.mkstemp(dir=Path(path).parent,prefix=Path(path).name+'.');os.close(fd);torch.save(x,t);os.replace(t,path)
def tensor_entry(path,b,r,source,fold_id):
 return {'path':str(Path(path).resolve()),'sha256':sha(path),'tensor_sha256':tensor_ch(b),'fold':fold_id,'source_state_sha256':source,'row_identity':row_identity(r),'availability':[int(x.sum()) for x in r['masks']],'intervention_coverage':int(torch.stack(r['masks'],1).any(1).sum())}
def build(rows,val_rows,out,features,labels,val_features):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);oof={};folds={};coverage={r['id']:0 for r in rows}
 for k in range(5):
  targets=[r for r in rows if fold(r['id'])==k];neg=[r for r in rows if float(r['y'])==0 and fold(r['id'])!=k];mc=[sum(int(r['masks'][f].sum()) for r in neg) for f in range(3)]
  if not targets or not neg or any(x<=0 for x in mc):raise RuntimeError('fold coverage')
  mods=fit(neg);gates=audit_zero(mods,neg);sp=out/f'fold{k}.pt';save_state(sp,mods);ss=sha(sp);[g.update(state_sha256=ss) for g in gates];folds[str(k)]={'target_ids':sorted(r['id'] for r in targets),'negative_ids':sorted(r['id'] for r in neg),'modality_target_counts':mc,'input_rows':{r['id']:row_identity(r) for r in neg},'zero_influence_gates':gates,'state':{'path':str(sp.resolve()),'sha256':ss}}
  for r in targets:
   b=predict(mods,r);bp=out/f"{r['id']}.oof.pt";save_tensor(bp,b);oof[r['id']]=tensor_entry(bp,b,r,ss,k);coverage[r['id']]+=1
 neg=[r for r in rows if float(r['y'])==0];mods=fit(neg);gates=audit_zero(mods,neg);sp=out/'full.pt';save_state(sp,mods);full_sha=sha(sp);[g.update(state_sha256=full_sha) for g in gates];valb={}
 for r in val_rows:
  b=predict(mods,r);bp=out/f"{r['id']}.val_full.pt";save_tensor(bp,b);valb[r['id']]=tensor_entry(bp,b,r,full_sha,'full')
 if any(v!=1 for v in coverage.values()):raise RuntimeError('OOF coverage')
 sources=source_identities()
 inputs={'train_features':{'path':str(Path(features).resolve()),'sha256':sha(features),'root_sha256':json.load(open(features))['root_sha256']},'weak_labels':{'path':str(Path(labels).resolve()),'sha256':sha(labels)},'val_features':{'path':str(Path(val_features).resolve()),'sha256':sha(val_features),'root_sha256':json.load(open(val_features))['root_sha256']}}
 man={'schema':'v26_reference_v2','design_sha256':DESIGN_SHA,'config':CONFIG,'runtime':{'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version()},'sources':sources,'inputs':inputs,'folds':folds,'oof':oof,'full':{'state':{'path':str(sp.resolve()),'sha256':full_sha},'negative_ids':sorted(r['id'] for r in neg),'modality_target_counts':[sum(int(r['masks'][f].sum()) for r in neg) for f in range(3)],'input_rows':{r['id']:row_identity(r) for r in neg},'zero_influence_gates':gates},'val_backgrounds':valb,'labels_used':'train_video_negative_only','status':'FROZEN'};atomic(out/'manifest.json',man);verify_reference(out/'manifest.json',features,labels,val_features);return man
def load_models(path):
 mods=[Decoder(d) for d in DIMS];states=torch.load(path,map_location='cpu',weights_only=True)
 for m,s in zip(mods,states):m.load_state_dict(s);m.eval();[p.requires_grad_(False) for p in m.parameters()]
 return mods
def verify_reference(path,features,labels,val_features):
 m=json.load(open(path))
 runtime={'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version()}
 frozen=Path(path).resolve()==FROZEN_REFERENCE.resolve() and sha(path)==FROZEN_REFERENCE_SHA;expected_sources=FROZEN_REFERENCE_SOURCES if frozen else source_identities()
 if set(m)!=MANIFEST_KEYS or m['schema']!='v26_reference_v2' or m['design_sha256']!=DESIGN_SHA or m['config']!=CONFIG or m['runtime']!=runtime or m['status']!='FROZEN' or m['labels_used']!='train_video_negative_only' or m['sources']!=expected_sources or set(m['inputs'])!={'train_features','weak_labels','val_features'}:raise RuntimeError('reference schema/source')
 from train import load_rows
 rows,_=load_rows(features,labels);vr,_=load_rows(val_features);rm={r['id']:r for r in rows};vm={r['id']:r for r in vr}
 for key,p in (('train_features',features),('weak_labels',labels),('val_features',val_features)):
  expected={'path','sha256'} if key=='weak_labels' else {'path','sha256','root_sha256'}
  if set(m['inputs'][key])!=expected or m['inputs'][key]['path']!=str(Path(p).resolve()) or m['inputs'][key]['sha256']!=sha(p) or (key!='weak_labels' and m['inputs'][key]['root_sha256']!=json.load(open(p))['root_sha256']):raise RuntimeError('reference input')
 if set(m['folds'])!=set(map(str,range(5))) or set(m['oof'])!=set(rm) or set(m['val_backgrounds'])!=set(vm):raise RuntimeError('reference coverage')
 for k,fm in m['folds'].items():
  kk=int(k);neg=sorted(v for v,r in rm.items() if float(r['y'])==0 and fold(v)!=kk);targets=sorted(v for v in rm if fold(v)==kk);counts=[sum(int(rm[v]['masks'][f].sum()) for v in neg) for f in range(3)];expected_rows={v:row_identity(rm[v]) for v in neg}
  if set(fm)!={'target_ids','negative_ids','modality_target_counts','input_rows','zero_influence_gates','state'} or set(fm['state'])!={'path','sha256'} or sha(fm['state']['path'])!=fm['state']['sha256'] or fm['target_ids']!=targets or fm['negative_ids']!=neg or fm['input_rows']!=expected_rows or fm['modality_target_counts']!=counts:raise RuntimeError('fold nested')
  mods=load_models(fm['state']['path']);gg=audit_zero(mods,[rm[v] for v in neg]);[g.update(state_sha256=fm['state']['sha256']) for g in gg]
  if fm['zero_influence_gates']!=gg or len(gg)!=3 or any(set(g)!={'modality','video_id','t','value_sha256','gradient_zero','state_sha256'} for g in gg) or sorted(g['modality'] for g in gg)!=[0,1,2] or any(not g['gradient_zero'] or g['video_id'] not in neg or g['state_sha256']!=fm['state']['sha256'] for g in gg):raise RuntimeError('fold gate')
 for pool,rr in ((m['oof'],rm),(m['val_backgrounds'],vm)):
  for v,e in pool.items():
   if set(e)!={'path','sha256','tensor_sha256','fold','source_state_sha256','row_identity','availability','intervention_coverage'}:raise RuntimeError('background entry schema')
   if sha(e['path'])!=e['sha256'] or e['row_identity']!=row_identity(rr[v]) or e['availability']!=[int(x.sum()) for x in rr[v]['masks']] or e['intervention_coverage']!=int(torch.stack(rr[v]['masks'],1).any(1).sum()):raise RuntimeError('background binding')
   b=torch.load(e['path'],map_location='cpu',weights_only=True)
   if tensor_ch(b)!=e['tensor_sha256'] or any(b[i].shape!=rr[v]['X'][i].shape for i in range(3)):raise RuntimeError('background tensor')
 for v,e in m['oof'].items():
  ff=str(fold(v));
  if e['fold']!=fold(v) or e['source_state_sha256']!=m['folds'][ff]['state']['sha256']:raise RuntimeError('OOF source')
 for v,e in m['val_backgrounds'].items():
  if e['fold']!='full' or e['source_state_sha256']!=m['full']['state']['sha256']:raise RuntimeError('val source')
 neg=sorted(v for v,r in rm.items() if float(r['y'])==0);counts=[sum(int(rm[v]['masks'][f].sum()) for v in neg) for f in range(3)];expected_rows={v:row_identity(rm[v]) for v in neg};fu=m['full']
 if set(fu)!={'state','negative_ids','modality_target_counts','input_rows','zero_influence_gates'} or set(fu['state'])!={'path','sha256'} or sha(fu['state']['path'])!=fu['state']['sha256'] or fu['negative_ids']!=neg or fu['input_rows']!=expected_rows or fu['modality_target_counts']!=counts:raise RuntimeError('full nested')
 mods=load_models(fu['state']['path']);gg=audit_zero(mods,[rm[v] for v in neg]);[g.update(state_sha256=fu['state']['sha256']) for g in gg]
 if fu['zero_influence_gates']!=gg or len(gg)!=3 or any(set(g)!={'modality','video_id','t','value_sha256','gradient_zero','state_sha256'} for g in gg):raise RuntimeError('full gate')
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--labels',required=True);p.add_argument('--val-features',required=True);p.add_argument('--out',required=True);a=p.parse_args();from train import load_rows;rows,_=load_rows(a.features,a.labels);vr,_=load_rows(a.val_features);build(rows,vr,a.out,a.features,a.labels,a.val_features)
if __name__=='__main__':main()
