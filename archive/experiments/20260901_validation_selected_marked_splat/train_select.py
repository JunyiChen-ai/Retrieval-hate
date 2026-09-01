#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as tdata

ROOT=Path(__file__).resolve().parents[2]; BASE=ROOT/'scripts/reproduction_baselines'; MM=BASE/'multihateloc'
sys.path[:0]=[str(ROOT),str(BASE),str(MM)]
import data as mdata
from hate_common import data as hdata
from src.marked_temporal_splat import MarkedTemporalSplatMIL
from src.scoped_video_protocol import scoped_video_labels

def ap(y,s):
    y=np.asarray(y,float); order=np.argsort(-np.asarray(s),kind='mergesort'); y=y[order]
    if y.sum()==0: return float('nan')
    precision=np.cumsum(y)/np.arange(1,len(y)+1); recall=np.cumsum(y)/y.sum()
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))

def loader(corpus,ids,labels,batch,shuffle,workers,generator=None):
    return tdata.DataLoader(mdata.MultiModalDataset(corpus,ids,labels),batch_size=batch,
        shuffle=shuffle,collate_fn=mdata.collate,num_workers=workers,generator=generator)

def train_epoch(model,batches,device,opt,args):
    model.train(); totals={}; seen=0
    for feats,labels,lengths,mask,_ in batches:
        feats={k:v.to(device,non_blocking=True) for k,v in feats.items()}; labels=labels.to(device); lengths=lengths.to(device); mask=mask.to(device)
        out=model(feats,mask); mil,_=model.mil_loss(out['prob'],mask,lengths,labels)
        smooth=model.smoothness_loss(out['prob'],mask); contrast=model.contrastive_loss(out['embeds'],mask)
        loss=mil+args.lambda_smooth*smooth+args.lambda_contrast*contrast
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); n=len(labels); seen+=n
        for key,value in {'loss':loss,'mil':mil,'smooth':smooth,'contrast':contrast}.items(): totals[key]=totals.get(key,0.)+float(value.detach())*n
    return {k:v/seen for k,v in totals.items()}

@torch.no_grad()
def val_scores(model,batches,device):
    model.eval(); scores={}; labels={}
    for feats,target,lengths,mask,ids in batches:
        feats={k:v.to(device) for k,v in feats.items()}; mask=mask.to(device); lengths=lengths.to(device)
        score=model.video_score(model(feats,mask)['prob'],mask,lengths)
        for i,vid in enumerate(ids): scores[vid]=float(score[i]); labels[vid]=int(target[i])
    return scores,labels

def main():
    p=argparse.ArgumentParser(); p.add_argument('--corpus',required=True,choices=('hatemm','hateclipseg')); p.add_argument('--output-dir',required=True); p.add_argument('--config-name',required=True)
    p.add_argument('--seed',type=int,default=234); p.add_argument('--lr',type=float,required=True); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--max-epoch',type=int,required=True); p.add_argument('--k-proportion',type=int,required=True)
    p.add_argument('--lambda-smooth',type=float,required=True); p.add_argument('--lambda-contrast',type=float,required=True); p.add_argument('--hidden',type=int,required=True); p.add_argument('--embed',type=int,required=True); p.add_argument('--dropout',type=float,required=True); p.add_argument('--temperature',type=float,required=True); p.add_argument('--device',default='cuda'); args=p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps({'date':'2026-09-01','method':'validation_selected_marked_splat','stage':'validation_selection_only','test_split_read':False,'args':vars(args)},indent=2)+'\n')
    train_ids=hdata.load_split(args.corpus,'train'); val_ids=hdata.load_split(args.corpus,'val')
    if set(train_ids)&set(val_ids): raise RuntimeError('train/validation overlap')
    gen=torch.Generator().manual_seed(args.seed)
    train=loader(args.corpus,train_ids,scoped_video_labels(args.corpus,'train',train_ids),args.batch_size,True,4,gen)
    val=loader(args.corpus,val_ids,scoped_video_labels(args.corpus,'val',val_ids),args.batch_size,False,2)
    model=MarkedTemporalSplatMIL({n:mdata.FEATURE_DIMS[n] for n in mdata.MODALITIES},args.hidden,args.embed,args.dropout,args.k_proportion,args.temperature).to(args.device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr); best=(-1.,None,None); history=[]; started=time.time()
    for epoch in range(1,args.max_epoch+1):
        stats=train_epoch(model,train,args.device,opt,args); scores,labels=val_scores(model,val,args.device); ids=sorted(scores); value=ap([labels[v] for v in ids],[scores[v] for v in ids]); stats.update(epoch=epoch,validation_video_ap=value); history.append(stats)
        if value==value and value>best[0]: best=(value,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        if epoch==1 or epoch%10==0: print(f'{args.corpus}/{args.config_name} epoch {epoch:03d} loss={stats["loss"]:.4f} val_ap={value:.4f}',flush=True)
    if best[2] is None: raise RuntimeError('no validation-selected checkpoint')
    torch.save(best[2],out/'checkpoint.pt'); (out/'train_log.json').write_text(json.dumps({'corpus':args.corpus,'config_name':args.config_name,'selected_validation_video_ap':best[0],'selected_epoch':best[1],'test_prediction_generated':False,'test_labels_used_for_gradient_or_checkpoint_selection':False,'history':history,'elapsed_seconds':round(time.time()-started,1)},indent=2)+'\n')
    print(f'selected epoch {best[1]}; validation AP {best[0]:.6f}; no test read',flush=True)
if __name__=='__main__': main()
