#!/usr/bin/env python3
"""Corpus-bound test inference for a completed V6 checkpoint."""
import argparse,json,os,sys
from types import SimpleNamespace
import numpy as np,torch
from torch.utils.data import DataLoader
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path[:0]=[os.path.dirname(HERE),os.path.join(os.path.dirname(os.path.dirname(HERE)),"duplex")]
import frame_eval_common as fec
from relation_v4.io import load_manifest,sha256
from relation_v6.audited_data import build,collate
from relation_v6.model import RelationV6
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--checkpoint-dir",required=True); p.add_argument("--device",default="cuda"); p.add_argument("--scores-out",required=True); p.add_argument("--eval-out",required=True); a=p.parse_args(); m=load_manifest(a.manifest); jp=os.path.join(a.checkpoint_dir,"train_meta.json"); mp=os.path.join(a.checkpoint_dir,"model.pth"); cp=os.path.join(a.checkpoint_dir,"COMPLETE.json"); outer_path=os.path.join(a.checkpoint_dir,"OUTER_SELECTION.json")
 if not os.path.isfile(outer_path): raise RuntimeError("inference requires authoritative OUTER_SELECTION.json")
 outer=json.load(open(outer_path)); meta=json.load(open(jp)); complete=json.load(open(cp))
 if outer.get("selected",{}).get("model_sha256")!=sha256(mp): raise RuntimeError("outer-selected model hash mismatch")
 if meta["corpus"]!=m["corpus"] or complete["corpus"]!=m["corpus"] or meta["manifest_sha256"]!=sha256(a.manifest): raise RuntimeError("corpus/manifest mismatch")
 if complete["model_sha256"]!=sha256(mp) or complete["meta_sha256"]!=sha256(jp): raise RuntimeError("checkpoint hash mismatch")
 cfg=SimpleNamespace(**meta["args"]); ds,_,prov=build(m,"test",meta["calibration"]["sorted_values"]); loader=DataLoader(ds,batch_size=cfg.batch_size,shuffle=False,collate_fn=collate); model=RelationV6(len(m["experts"]),cfg.hidden,cfg.heads,cfg.window,cfg.temperature,cfg.dropout).to(a.device); model.load_state_dict(torch.load(mp,map_location=a.device)); model.eval(); per={}; os.makedirs(os.path.dirname(os.path.abspath(a.scores_out)),exist_ok=True)
 seen=set()
 with torch.no_grad(),open(a.scores_out,"w") as f:
  for vids,scores,valid,lengths,label,teacher,tm,gold in loader:
   scores,valid=scores.to(a.device),valid.to(a.device); out=model(scores,valid,meta["selected_locator_scale"])
   for i,vid in enumerate(vids):
    if vid in seen: raise RuntimeError("duplicate inference ID")
    seen.add(vid); n=int(lengths[i]); score=out["frame_prob"][i,:n].cpu().numpy()
    if len(score)!=len(gold[i,:n]) or not np.isfinite(score).all(): raise RuntimeError("inference alignment/nonfinite")
    per[vid]=(score,gold[i,:n].numpy()); f.write(json.dumps({"video_id":vid,"score_relation_v6":score.tolist(),"prior_logit":float(out["prior_logit"][i].cpu())})+"\n")
 if seen!=set(ds.ids): raise RuntimeError("inference full coverage mismatch")
 metric=fec.evaluate(per); payload={"method":"relation_v6_train_only","corpus":m["corpus"],"checkpoint":os.path.abspath(a.checkpoint_dir),"checkpoint_model_sha256":sha256(mp),"selected_epoch":meta["selected_epoch"],"selected_locator_scale":meta["selected_locator_scale"],"test_experts":prov,"results":{"frame_ap":metric["pr_auc"],"frame_roc":metric["roc_auc"],"n_videos":metric["n_videos"],"n_frames":metric["n_frames"]}}; json.dump(payload,open(a.eval_out+".tmp","w"),indent=2); os.replace(a.eval_out+".tmp",a.eval_out); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
