#!/usr/bin/env python3
"""Strict single-corpus Relation-V3 training with localization-aware MIL."""
from __future__ import annotations
import argparse, copy, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE=os.path.dirname(os.path.abspath(__file__)); PARENT=os.path.dirname(HERE)
sys.path[:0]=[PARENT, os.path.join(os.path.dirname(PARENT),"duplex")]
from hate_common import data as hdata
from hate_common import runtime
from macilsd.train import _seq_len_of
from powa_macil.dataset import PowaTestDataset,PowaTrainDataset,usable_text_ids
import frame_eval_common as fec
from relation_v2.protocol import frozen_splits,scoped_labels,sha256_file,sha256_ids,verify_macil_init
from relation_v3.model import RelationV3

def parser():
 p=argparse.ArgumentParser(); p.add_argument("--corpus",required=True,choices=hdata.CORPORA); p.add_argument("--out-dir",required=True); p.add_argument("--macil-init",required=True); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=234); p.add_argument("--num-workers",type=int,default=4); p.add_argument("--max-epoch",type=int,default=5); p.add_argument("--batch-size",type=int,default=24); p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--crop-repeat",type=int,default=1); p.add_argument("--max-seqlen",type=int,default=200); p.add_argument("--grid",default="snippet",choices=("snippet","second")); p.add_argument("--hid-dim",type=int,default=128); p.add_argument("--ffn-dim",type=int,default=128); p.add_argument("--nhead",type=int,default=4); p.add_argument("--dropout",type=float,default=.1); p.add_argument("--num-classes",type=int,default=1); p.add_argument("--a-feature-size",type=int,default=128); p.add_argument("--v-feature-size",type=int,default=1024); p.add_argument("--text-feature-size",type=int,default=768); p.add_argument("--n-relations",type=int,default=4); p.add_argument("--relation-dim",type=int,default=32); p.add_argument("--binding-window",type=int,default=24); p.add_argument("--binding-temperature",type=float,default=.2); p.add_argument("--topk-divisor",type=int,default=16); p.add_argument("--non-top-weight",type=float,default=.05); p.add_argument("--smooth-weight",type=float,default=.02); return p

def mask(lengths,width,device): return torch.arange(width,device=device)[None]<lengths.to(device)[:,None]
def build(a):
 s=frozen_splits(a.corpus); labels,path=scoped_labels(a.corpus,"train"); tr=usable_text_ids(a.corpus,s["train"]); va=usable_text_ids(a.corpus,s["val"])
 if tr!=s["train"] or va!=s["val"]: raise RuntimeError("feature coverage differs from frozen split")
 td=PowaTrainDataset(a.corpus,tr,labels,a.max_seqlen,a.grid,"av",a.crop_repeat); vd=PowaTestDataset(a.corpus,va,a.max_seqlen,a.grid,"av")
 return DataLoader(td,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers),DataLoader(vd,batch_size=1,shuffle=False,num_workers=a.num_workers),s,path

def localization_loss(prob,y,lengths,a):
 terms=[]; pos_top=[]; neg_top=[]; non=[]; smooth=[]
 for i in range(len(y)):
  p=prob[i,:int(lengths[i])]; k=max(1,len(p)//a.topk_divisor+1); top,idx=p.topk(k)
  if y[i]>0.5:
   primary=torch.nn.functional.binary_cross_entropy(top,torch.ones_like(top)); keep=torch.ones(len(p),dtype=torch.bool,device=p.device); keep[idx]=False; nontop=p[keep].mean() if keep.any() else p.new_zeros(()); pos_top.append(top.mean()); non.append(nontop)
   term=primary+a.non_top_weight*nontop
  else:
   primary=torch.nn.functional.binary_cross_entropy(top,torch.zeros_like(top)); neg_top.append(top.mean()); term=primary
  sm=(p[1:]-p[:-1]).abs().mean() if len(p)>1 else p.new_zeros(()); smooth.append(sm); terms.append(term+a.smooth_weight*sm)
 def avg(xs): return torch.stack(xs).mean() if xs else prob.new_zeros(())
 return torch.stack(terms).mean(),{"positive_topk":avg(pos_top),"negative_topk":avg(neg_top),"positive_nontop":avg(non),"smooth":avg(smooth)}

def train_epoch(m,loader,opt,a,device):
 m.train(); sums={k:0. for k in ("loss","positive_topk","negative_topk","positive_nontop","smooth","witness_saturation","transport_row_mass","transport_col_mass","delta_variance","base_variance")}; n=0
 for fv,fa,ft,y in loader:
  lengths=_seq_len_of(fv); keep=int(lengths.max()); valid=mask(lengths,keep,device); fv=fv[:,:keep].float().to(device); fa=fa[:,:keep].float().to(device); ft=ft[:,:keep].float().to(device); y=y.float().to(device)
  o=m(fa,fv,ft,lengths,valid); loss,parts=localization_loss(o["frame_prob"],y,lengths,a); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],5.); opt.step()
  vals={"loss":loss,**parts,"witness_saturation":(o["relation_witness"]>.95).float().mean(),"transport_row_mass":o["endpoint_outgoing"].mean(),"transport_col_mass":o["endpoint_incoming"].mean(),"delta_variance":o["delta_logit"][valid].var(unbiased=False),"base_variance":o["base_frame_logits"].squeeze(-1)[valid].var(unbiased=False)}
  for k,v in vals.items(): sums[k]+=float(v.detach())
  n+=1
 return {k:v/max(1,n) for k,v in sums.items()}|{"batches":n}

@torch.no_grad()
def validate(m,loader,corpus,device):
 m.eval(); gt=hdata.gt_arrays(corpus,"val"); pv={}; seen=set()
 for fv,fa,ft,index,nsec,vid in loader:
  vid=vid[0]
  if vid not in gt: continue
  if vid in seen: raise RuntimeError("duplicate validation ID")
  seen.add(vid); fv,fa,ft=fv[0].to(device),fa[0].to(device),ft[0].to(device); lengths=torch.full((fv.shape[0],),fv.shape[1],dtype=torch.long); score=m(fa,fv,ft,lengths)["frame_prob"].mean(0).cpu().numpy()[index[0].numpy()]
  if len(score)!=len(gt[vid]) or not np.isfinite(score).all(): raise RuntimeError("validation alignment/nonfinite")
  pv[vid]=(score,gt[vid])
 if seen!=set(gt): raise RuntimeError("validation coverage mismatch")
 x=fec.evaluate(pv); return {"frame_ap":x["pr_auc"],"frame_roc":x["roc_auc"],"n_eval_videos":x["n_videos"]}

def main(argv=None):
 a=parser().parse_args(argv); a.device=runtime.resolve_device(a.device); runtime.setup_seed(a.seed)
 if os.path.exists(a.out_dir) and os.listdir(a.out_dir): raise RuntimeError("out-dir must be absent or empty")
 init=verify_macil_init(a.corpus,a.macil_init); loader,vloader,splits,labelpath=build(a); m=RelationV3(a).to(a.device); m.macil.load_state_dict(torch.load(a.macil_init,map_location=a.device)); opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.max_epoch); os.makedirs(a.out_dir,exist_ok=True); best=-1.; be=-1; hist=[]
 for epoch in range(1,a.max_epoch+1):
  st=time.time(); tr=train_epoch(m,loader,opt,a,a.device); sch.step(); va=validate(m,vloader,a.corpus,a.device); row={"epoch":epoch,"train":tr,"validation":va,"seconds":round(time.time()-st,2)}; hist.append(row); print(json.dumps(row),flush=True)
  if va["frame_ap"]>best: best=va["frame_ap"]; be=epoch; torch.save(copy.deepcopy(m.state_dict()),os.path.join(a.out_dir,"model.pth.tmp")); os.replace(os.path.join(a.out_dir,"model.pth.tmp"),os.path.join(a.out_dir,"model.pth"))
 meta={"method":"relation_v3_performance","corpus":a.corpus,"args":vars(a),"selected_epoch":be,"selected_metric":"validation_pooled_frame_ap","selected_value":best,"history":hist,"train_ids":sorted(splits["train"]),"val_ids":sorted(splits["val"]),"test_manifest_ids":sorted(splits["test"]),"split_hashes":{k:sha256_ids(v) for k,v in splits.items()},"scoped_train_labels":{"path":labelpath,"sha256":sha256_file(labelpath)},"macil_init":init,"macil_immutable":True,"readout_inputs":"transport_endpoint_only","test_labels_used_in_gradient_training":False,"cross_corpus_data_or_parameters":False}; mp=os.path.join(a.out_dir,"model.pth"); meta["model_sha256"]=sha256_file(mp); jp=os.path.join(a.out_dir,"train_meta.json"); json.dump(meta,open(jp+".tmp","w"),indent=2); os.replace(jp+".tmp",jp); complete={"corpus":a.corpus,"model_sha256":meta["model_sha256"],"meta_sha256":sha256_file(jp)}; cp=os.path.join(a.out_dir,"COMPLETE.json"); json.dump(complete,open(cp+".tmp","w"),indent=2); os.replace(cp+".tmp",cp); print(f"selected epoch {be} val Frame AP {best:.6f}")
if __name__=="__main__": main()
