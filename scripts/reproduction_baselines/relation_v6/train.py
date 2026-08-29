#!/usr/bin/env python3
"""Train-only Relation-V6 with joint validation checkpoint/scale selection."""
import argparse,copy,json,os,sys,time
import numpy as np,torch
from torch.utils.data import DataLoader
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path[:0]=[os.path.dirname(HERE),os.path.join(os.path.dirname(os.path.dirname(HERE)),"duplex")]
from hate_common import runtime
import frame_eval_common as fec
from relation_v4.io import load_manifest,sha256
from relation_v6.audited_data import build,collate
from relation_v6.model import RelationV6
def bag(prob,valid,divisor=16):
 out=[]
 for i in range(len(prob)):
  p=prob[i,valid[i]]; k=max(1,len(p)//divisor+1); out.append(p.topk(k).values.mean())
 return torch.stack(out)
def train_epoch(model,loader,opt,device,reg,teacher_weight):
 model.train(); total={k:0. for k in ("loss","prior_bce","bag_bce","teacher_bce","locator_smooth","locator_variance")}; n=0
 for vids,scores,valid,lengths,label,teacher,tm,gold in loader:
  scores,valid,label,teacher,tm=scores.to(device),valid.to(device),label.to(device),teacher.to(device),tm.to(device); out=model(scores,valid,1.); prior=torch.nn.functional.binary_cross_entropy_with_logits(out["prior_logit"],label); bce=torch.nn.functional.binary_cross_entropy(bag(out["frame_prob"],valid),label); smooth=[]
  for i in range(len(scores)):
   loc=out["locator_logit"][i,valid[i]]; smooth.append((loc[1:]-loc[:-1]).abs().mean() if len(loc)>1 else loc.new_zeros(()))
  smooth=torch.stack(smooth).mean(); teacher_loss=(torch.nn.functional.binary_cross_entropy_with_logits(out["frame_logit"][tm],teacher[tm]) if tm.any() else scores.new_zeros(())); loss=prior+bce+float(reg)*smooth+float(teacher_weight)*teacher_loss
  opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); vals={"loss":loss,"prior_bce":prior,"bag_bce":bce,"teacher_bce":teacher_loss,"locator_smooth":smooth,"locator_variance":out["locator_logit"][valid].var(unbiased=False)}
  for k,v in vals.items(): total[k]+=float(v.detach())
  n+=1
 return {k:v/max(1,n) for k,v in total.items()}|{"batches":n}
@torch.no_grad()
def validate(model,loader,scales,device):
 model.eval(); store={float(s):{} for s in scales}
 for vids,scores,valid,lengths,label,teacher,tm,gold in loader:
  scores,valid=scores.to(device),valid.to(device); base=model(scores,valid,1.)
  for i,vid in enumerate(vids):
   n=int(lengths[i]); y=gold[i,:n].numpy()
   for scale in scales: score=torch.sigmoid(base["prior_logit"][i]+float(scale)*base["locator_logit"][i,:n]).cpu().numpy(); store[float(scale)][vid]=(score,y)
 rows=[]
 for scale in scales:
  m=fec.evaluate(store[float(scale)]); rows.append({"locator_scale":float(scale),"frame_ap":m["pr_auc"],"frame_roc":m["roc_auc"]})
 return rows
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--out-dir",required=True); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=234); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--hidden",type=int,default=32); p.add_argument("--heads",type=int,default=4); p.add_argument("--window",type=int,default=12); p.add_argument("--temperature",type=float,default=.2); p.add_argument("--dropout",type=float,default=.1); p.add_argument("--regularization",type=float,choices=(0.,.01),default=.01); p.add_argument("--teacher-weight",type=float,default=.1); p.add_argument("--lambda-grid",default="0,.25,.5,1,1.5,2"); a=p.parse_args(); m=load_manifest(a.manifest)
 if m["corpus"]!="hateclipseg": raise RuntimeError("current V6 manifest is HCS-bound")
 if m.get("train_sparse_manifest") and not os.path.isfile(m["train_sparse_manifest"]): raise RuntimeError("train sparse manifest is not ready")
 if os.path.exists(a.out_dir) and os.listdir(a.out_dir): raise RuntimeError("out-dir must be absent or empty")
 a.device=runtime.resolve_device(a.device); runtime.setup_seed(a.seed); train_ds,cal,train_prov=build(m,"train"); val_ds,_,val_prov=build(m,"val",cal); train_loader=DataLoader(train_ds,batch_size=a.batch_size,shuffle=True,collate_fn=collate); val_loader=DataLoader(val_ds,batch_size=a.batch_size,shuffle=False,collate_fn=collate); model=RelationV6(len(m["experts"]),a.hidden,a.heads,a.window,a.temperature,a.dropout).to(a.device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4); scales=[float(x) for x in a.lambda_grid.split(",")]; os.makedirs(a.out_dir,exist_ok=True); history=[]; best=None
 for epoch in range(1,a.epochs+1):
  start=time.time(); tr=train_epoch(model,train_loader,opt,a.device,a.regularization,a.teacher_weight); va=validate(model,val_loader,scales,a.device)
  choice=max(va,key=lambda r:(r["frame_ap"],r["frame_roc"])); row={"epoch":epoch,"train":tr,"validation":va,"epoch_choice":choice,"seconds":round(time.time()-start,2)}; history.append(row); print(json.dumps(row),flush=True)
  key=(choice["frame_ap"],choice["frame_roc"])
  if best is None or key>(best[0],best[1]): best=(choice["frame_ap"],choice["frame_roc"],epoch,choice["locator_scale"]); torch.save(copy.deepcopy(model.state_dict()),os.path.join(a.out_dir,"model.pth.tmp")); os.replace(os.path.join(a.out_dir,"model.pth.tmp"),os.path.join(a.out_dir,"model.pth"))
 sparse_provenance=train_ds.sparse_root; meta={"method":"relation_v6_train_only_candidate","corpus":m["corpus"],"args":vars(a),"selected_epoch":best[2],"selected_locator_scale":best[3],"selected_validation_frame_ap":best[0],"selected_validation_frame_roc":best[1],"history":history,"calibration":{"type":"train_frozen_ecdf","sorted_values":cal},"train_experts":train_prov,"validation_experts":val_prov,"manifest":os.path.abspath(a.manifest),"manifest_sha256":sha256(a.manifest),"train_sparse_provenance":sparse_provenance,"test_opened_during_training_or_selection":False,"single_corpus":True,"authoritative_only_after_outer_selection":True}; mp=os.path.join(a.out_dir,"model.pth"); meta["model_sha256"]=sha256(mp); jp=os.path.join(a.out_dir,"train_meta.json"); json.dump(meta,open(jp+".tmp","w"),indent=2); os.replace(jp+".tmp",jp); complete={"corpus":m["corpus"],"model_sha256":meta["model_sha256"],"meta_sha256":sha256(jp)}; cp=os.path.join(a.out_dir,"COMPLETE.json"); json.dump(complete,open(cp+".tmp","w"),indent=2); os.replace(cp+".tmp",cp); print(json.dumps({"selected_epoch":best[2],"locator_scale":best[3],"val_ap":best[0],"val_roc":best[1]}))
if __name__=="__main__": main()
