#!/usr/bin/env python3
"""Weak-label zero-mean locator over a frozen V10 identity consensus."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np,torch
import torch.nn as nn
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE.parent),str(HERE.parent.parent/"duplex")]
import frame_eval_common as fec
from hate_common import data as hdata
from relation_v4.io import apply_ecdf,sha256
from relation_v9.train import load

class Locator(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Conv1d(1,16,3,padding=1),nn.GELU(),nn.Conv1d(16,1,3,padding=1));self.scale=nn.Parameter(torch.zeros(()))
 def forward(self,identity,role):
  centered=role-role.mean();delta=self.net(centered[None,None]).squeeze();delta=delta-delta.mean();return identity+torch.tanh(self.scale)*delta,delta
def bag(logits):return logits.topk(max(1,len(logits)//8)).values.mean()
def score_model(model,split,identity,role,gt):
 per={};device=next(model.parameters()).device
 with torch.no_grad():
  for v in gt:
   base=torch.logit(torch.tensor(identity[v],dtype=torch.float32,device=device).clamp(1e-5,1-1e-5));out,_=model(base,torch.tensor(role[v],dtype=torch.float32,device=device));per[v]=(torch.sigmoid(out).cpu().numpy(),gt[v])
 return fec.evaluate(per),per
def main():
 p=argparse.ArgumentParser();p.add_argument("--corpus",required=True,choices=("mhclip_en","mhclip_zh"));p.add_argument("--manifest",required=True);p.add_argument("--v10-config",required=True);p.add_argument("--out-dir",required=True);p.add_argument("--seed",type=int,default=234);p.add_argument("--device",default="cuda");a=p.parse_args();torch.manual_seed(a.seed);np.random.seed(a.seed);out=Path(a.out_dir)
 if out.exists() and any(out.iterdir()):raise RuntimeError("fresh out-dir required")
 out.mkdir(parents=True);manifest=json.load(open(a.manifest));cfg=json.load(open(a.v10_config));selected=cfg["selected"]
 if cfg["corpus"]!=a.corpus or selected["beta"]!=0 or selected["gamma"]!=0:raise RuntimeError("pilot requires frozen V10 prior-only identity")
 weights=np.asarray(cfg["candidate_weights"][selected["aggregation"]],np.float32);refs=cfg["calibration"];role_index=next(i for i,x in enumerate(manifest["experts"]) if x["name"]=="cmhkf");data={};prov={}
 for split in ("train","val","test"):
  ids,raw,pv=load(manifest,split);cal=apply_ecdf(raw,refs);identity={v:np.full(len(cal[v]),float(cal[v].mean(0)@weights),np.float32) for v in ids};role={v:cal[v][:,role_index] for v in ids};data[split]=(ids,identity,role);prov[split]=pv
 labels=hdata.load_labels(a.corpus);val_gt=hdata.gt_arrays(a.corpus,"val");test_gt=hdata.gt_arrays(a.corpus,"test");model=Locator().to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
 # Bit-exact identity is a required epoch-zero candidate.
 for v in data["val"][0]:
  ident=torch.logit(torch.tensor(data["val"][1][v],device=a.device).clamp(1e-5,1-1e-5));role=torch.tensor(data["val"][2][v],device=a.device);pred,_=model(ident,role)
  if not torch.equal(pred,ident):raise RuntimeError("epoch0 is not exact V10 identity")
 metric,_=score_model(model,"val",data["val"][1],data["val"][2],val_gt);history=[{"epoch":0,"train_loss":None,"validation_frame_ap":metric["pr_auc"],"validation_frame_roc":metric["roc_auc"],"identity_fallback":True}];best=((metric["pr_auc"],metric["roc_auc"]),0,copy.deepcopy(model.state_dict()));torch.save(best[2],out/"model.pth")
 for epoch in range(1,6):
  model.train();losses=[]
  for v in np.random.permutation(data["train"][0]):
   identity=torch.logit(torch.tensor(data["train"][1][v],device=a.device).clamp(1e-5,1-1e-5));role=torch.tensor(data["train"][2][v],device=a.device);logit,delta=model(identity,role);noisy,_=model(identity,(role+.01*torch.randn_like(role)).clamp(0,1));label=torch.tensor(float(labels[v]),device=a.device);loss=F.binary_cross_entropy_with_logits(bag(logit),label)+.01*(delta[1:]-delta[:-1]).abs().mean()+.1*F.mse_loss(logit,noisy)
   opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
  model.eval();metric,_=score_model(model,"val",data["val"][1],data["val"][2],val_gt);row={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_frame_ap":metric["pr_auc"],"validation_frame_roc":metric["roc_auc"],"identity_fallback":False};history.append(row);key=(metric["pr_auc"],metric["roc_auc"])
  if key>best[0]:best=(key,epoch,copy.deepcopy(model.state_dict()));torch.save(best[2],out/"model.pth")
 model.load_state_dict(torch.load(out/"model.pth",map_location=a.device));model.eval();test_metric,test_per=score_model(model,"test",data["test"][1],data["test"][2],test_gt);scores_path=out/"test_scores.jsonl"
 with scores_path.open("w") as f:
  for v in sorted(test_per):f.write(json.dumps({"video_id":v,"score_role_residual":test_per[v][0].tolist(),"identity_score":data["test"][1][v].tolist()})+"\n")
 tag="test-informed role checkpoint; not a purely validation-selected method" if a.corpus=="mhclip_zh" else "predefined validation-only role"
 meta={"method":"relation_v9_role_separated_residual_pilot","corpus":a.corpus,"seed":a.seed,"fixed_epochs":5,"identity":{"source":"frozen V10 strong consensus","config":str(Path(a.v10_config).resolve()),"config_sha256":sha256(a.v10_config),"aggregation":selected["aggregation"],"weights":weights.tolist(),"beta":0,"gamma":0,"epoch0_exact":True},"locator_role":{"source":"cmhkf","selection_provenance":tag},"selected_epoch":best[1],"selected_by":"maximum validation Frame AP; ROC tie-break including epoch0","history":history,"train_sources":prov["train"],"validation_sources":prov["val"],"test_sources":prov["test"],"test":{"frame_ap":test_metric["pr_auc"],"frame_roc":test_metric["roc_auc"],"n_videos":test_metric["n_videos"],"n_frames":test_metric["n_frames"]},"model_sha256":sha256(out/"model.pth"),"scores_sha256":sha256(scores_path),"test_used_for_training_or_checkpoint_selection":False};json.dump(meta,open(out/"result.json","w"),indent=2);print(json.dumps({"selected_epoch":best[1],"validation":best[0],"test":meta["test"],"role_provenance":tag},indent=2))
if __name__=="__main__":main()
