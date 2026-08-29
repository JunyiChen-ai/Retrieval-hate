#!/usr/bin/env python3
"""Corpus-bound Relation-V3 dense inference."""
import argparse,json,os,sys
from types import SimpleNamespace
import numpy as np, torch
from torch.utils.data import DataLoader
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE))
from hate_common import data as hdata
from powa_macil.dataset import PowaTestDataset,usable_text_ids
from relation_v2.protocol import checkpoint_corpus,frozen_splits,sha256_file
from relation_v3.model import RelationV3
def main():
 p=argparse.ArgumentParser(); p.add_argument("--checkpoint-dir",required=True); p.add_argument("--corpus",required=True,choices=hdata.CORPORA); p.add_argument("--split",default="test",choices=("val","test")); p.add_argument("--device",default="cuda"); p.add_argument("--out",required=True); a=p.parse_args(); mp=os.path.join(a.checkpoint_dir,"model.pth"); jp=os.path.join(a.checkpoint_dir,"train_meta.json"); cp=os.path.join(a.checkpoint_dir,"COMPLETE.json"); meta=json.load(open(jp)); complete=json.load(open(cp)); checkpoint_corpus(meta,a.corpus); checkpoint_corpus(complete,a.corpus)
 if complete["model_sha256"]!=sha256_file(mp) or complete["meta_sha256"]!=sha256_file(jp): raise RuntimeError("checkpoint completion hash mismatch")
 cfg=SimpleNamespace(**meta["args"]); ids=frozen_splits(a.corpus)[a.split]; available=usable_text_ids(a.corpus,ids)
 if available!=ids: raise RuntimeError("inference feature coverage mismatch")
 gt=hdata.gt_arrays(a.corpus,a.split); eval_ids=[v for v in ids if v in gt]; ds=PowaTestDataset(a.corpus,eval_ids,cfg.max_seqlen,cfg.grid,"av"); loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=cfg.num_workers); m=RelationV3(cfg).to(a.device); m.load_state_dict(torch.load(mp,map_location=a.device)); m.eval(); seen=set(); os.makedirs(os.path.dirname(a.out),exist_ok=True)
 with torch.no_grad(),open(a.out,"w") as fh:
  for fv,fa,ft,index,nsec,vid in loader:
   vid=vid[0]
   if vid in seen: raise RuntimeError("duplicate inference ID")
   seen.add(vid); fv,fa,ft=fv[0].to(a.device),fa[0].to(a.device),ft[0].to(a.device); lengths=torch.full((fv.shape[0],),fv.shape[1],dtype=torch.long); score=m(fa,fv,ft,lengths)["frame_prob"].mean(0).cpu().numpy()[index[0].numpy()]
   if len(score)!=len(gt[vid]) or not np.isfinite(score).all(): raise RuntimeError("inference alignment/nonfinite")
   fh.write(json.dumps({"video_id":vid,"score_relation_v3":score.tolist()})+"\n")
 if seen!=set(gt): raise RuntimeError("inference coverage mismatch")
 print(f"wrote {len(seen)} videos to {a.out}")
if __name__=="__main__": main()
