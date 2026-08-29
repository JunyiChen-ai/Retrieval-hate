#!/usr/bin/env python3
"""Checkpoint-only dense train scorer for the V9 MHC expert pools."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path[:0]=[str(BASE),str(BASE.parent/"duplex")]
from hate_common import data as hdata
from relation_v2.protocol import frozen_splits
from relation_v4.io import sha256

def timeline(corpus):return json.load(open(Path(hdata.REPO_ROOT)/f"results/reproduction/features/vggish_1s/{corpus}/index.json"))
def cmhkf(meta,checkpoint,ids,device):
 import cmhkf_adapter as a
 cfg=SimpleNamespace(**meta["args"]);length=meta["visual_length"];window=meta["attn_window"]
 model=a.CMHKF(2,512,length,512,1,1,window,cfg.prompt_prefix,cfg.prompt_postfix,str(device)).to(device);a.reuse_identical_text_prompt_pass(model);model.load_state_dict(torch.load(checkpoint,map_location=device));return a.score_ids(model,cfg.corpus,ids,length,device,cfg.batch_size)
def fed(meta,checkpoint,ids,device):
 import fed_wsvad_adapter as a
 cfg=SimpleNamespace(**meta["args"]);length=meta["visual_length"];window=meta["attn_window"]
 model=a.Model(512,length,cfg.prompt_prefix,cfg.prompt_postfix,512,cfg.visual_layers,1,window,str(device)).to(device);model.load_state_dict(torch.load(checkpoint,map_location=device));scores,_=a.score_ids(model,cfg.corpus,ids,length,device);return {v:{"score_align":scores[v]} for v in ids}
def macil(meta,checkpoint,ids,device):
 from macilsd import option
 from macilsd.dataset import MacilTestDataset
 from macilsd.train import build_models
 from macilsd.infer import _to_gold
 cfg=SimpleNamespace(**meta["args"]);cfg.device=device;cfg.method=option.METHOD_NAME[cfg.modality]
 ds=MacilTestDataset(cfg.corpus,ids,cfg.max_seqlen,cfg.grid,cfg.modality);loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0);av,uni=build_models(cfg);model=av if av is not None else uni;model.load_state_dict(torch.load(checkpoint,map_location=device));model.to(device).eval();out={}
 with torch.no_grad():
  for fv,fa,index_map,n_seconds,vid in loader:
   vid=vid[0];idx=index_map[0].numpy();fv,fa=fv[0].to(device),fa[0].to(device)
   if cfg.modality=="av":
    _,al,vl,avl,_,_=model(fa,fv,seq_len=None);branches={"score_av":_to_gold(torch.sigmoid(avl.squeeze(-1)).mean(0).cpu(),idx),"score_audio":_to_gold(al.squeeze(-1).mean(0).cpu(),idx),"score_visual":_to_gold(vl.squeeze(-1).mean(0).cpu(),idx)}
   else:
    feat=fa if cfg.modality=="audio" else fv;branches={"score_mil":_to_gold(torch.sigmoid(model(feat,seq_len=None).squeeze(-1)).mean(0).cpu(),idx)}
   out[vid]=branches
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--method",required=True,choices=("macilsd_audio","macilsd","cmhkf","fed_wsvad_1client"));p.add_argument("--corpus",required=True,choices=("mhclip_en","mhclip_zh"));p.add_argument("--seed",required=True,type=int,choices=(234,2025,3407));p.add_argument("--out-dir",required=True);p.add_argument("--device",default="cuda");a=p.parse_args();out=Path(a.out_dir)
 if out.exists() and any(out.iterdir()):raise RuntimeError("fresh output directory required")
 source=Path(hdata.REPO_ROOT)/f"results/reproduction/official_val/final/{a.method}/{a.corpus}/seed_{a.seed}";mp=source/"train_meta.json";cp=source/"model.pth"
 if not mp.is_file() or not cp.is_file():raise FileNotFoundError("checkpoint/train_meta missing")
 meta=json.load(open(mp));args=meta.get("args",{});ids=list(frozen_splits(a.corpus)["train"])
 if meta.get("method")!=a.method or args.get("corpus")!=a.corpus or int(args.get("seed"))!=a.seed or set(meta.get("train_ids",[]))!=set(ids):raise RuntimeError("checkpoint train provenance mismatch")
 scorer=macil if a.method.startswith("macilsd") else cmhkf if a.method=="cmhkf" else fed;scores=scorer(meta,cp,ids,torch.device(a.device));index=timeline(a.corpus)
 if set(scores)!=set(ids):raise RuntimeError("producer output IDs not exact frozen train")
 out.mkdir(parents=True);target=out/"scores.jsonl"
 with target.open("w") as f:
  for vid in ids:
   expected=int(index[vid]["n_frames"]);row={"video_id":vid,"n_frames":expected}
   for key,value in scores[vid].items():
    value=np.asarray(value,float)
    if len(value)!=expected or not np.isfinite(value).all():raise RuntimeError(f"{vid}/{key} length/finite failure")
    row[key]=value.tolist()
   f.write(json.dumps(row)+"\n")
 score_keys=sorted(next(iter(scores.values())).keys());selected_key={"macilsd_audio":"score_mil","macilsd":"score_av","cmhkf":"score_align","fed_wsvad_1client":"score_align"}[a.method]
 manifest={"method":a.method,"corpus":a.corpus,"split":"train","seed":a.seed,"score_keys":score_keys,"selected_score_key":selected_key,"checkpoint_only":True,"ids_source":"frozen train manifest","n_ids":len(ids),"scores":str(target.resolve()),"scores_sha256":sha256(target),"checkpoint":str(cp.resolve()),"checkpoint_sha256":sha256(cp),"train_meta":str(mp.resolve()),"train_meta_sha256":sha256(mp),"source_files":[{"path":str(x.resolve()),"sha256":sha256(x)} for x in (cp,mp)]};producer=out/"producer_manifest.json";json.dump(manifest,open(producer,"w"),indent=2);complete={"producer_manifest":str(producer.resolve()),"producer_manifest_sha256":sha256(producer),"scores_sha256":sha256(target),"checkpoint_sha256":sha256(cp),"train_meta_sha256":sha256(mp)};json.dump(complete,open(out/"COMPLETE.json","w"),indent=2);print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
