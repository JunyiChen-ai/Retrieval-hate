#!/usr/bin/env python3
"""Fail-closed synthetic tests for the complete producer proof chain."""
import json,tempfile
from pathlib import Path
from relation_v4.io import sha256
from relation_v9.build_mhc_manifests import verify_producer
from relation_v2.protocol import frozen_splits
def reject(fn):
 try:fn()
 except RuntimeError:return
 raise AssertionError("tampered producer was accepted")
def main():
 with tempfile.TemporaryDirectory() as td:
  repo=Path(td);method="cmhkf";corpus="mhclip_en";seed=234;root=repo/f"results/reproduction/relation_v9/train_dense/{method}/{corpus}/seed_{seed}";source=repo/f"results/reproduction/official_val/final/{method}/{corpus}/seed_{seed}";root.mkdir(parents=True);source.mkdir(parents=True);scores=root/"scores.jsonl";checkpoint=source/"model.pth";train_meta=source/"train_meta.json";ids=list(frozen_splits(corpus)["train"]);scores.write_text("".join(json.dumps({"video_id":v,"score_align":[.5]})+"\n" for v in ids));checkpoint.write_bytes(b'model');train_meta.write_text('{}\n')
  base={"method":method,"corpus":corpus,"split":"train","seed":seed,"n_ids":len(ids),"score_keys":["score_align"],"selected_score_key":"score_align","checkpoint_only":True,"scores":str(scores.resolve()),"scores_sha256":sha256(scores),"checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":sha256(checkpoint),"train_meta":str(train_meta.resolve()),"train_meta_sha256":sha256(train_meta)};mp=root/"producer_manifest.json";complete=root/"COMPLETE.json"
  def write(x,seal=True):
   mp.write_text(json.dumps(x))
   if seal:complete.write_text(json.dumps({"producer_manifest":str(mp.resolve()),"producer_manifest_sha256":sha256(mp),"scores_sha256":sha256(scores),"checkpoint_sha256":sha256(checkpoint),"train_meta_sha256":sha256(train_meta)}))
  write(base);assert verify_producer(repo,root,method,corpus,seed,"score_align")==scores
  for field,value in (("method","fed_wsvad_1client"),("corpus","mhclip_zh"),("split","val"),("seed",2025),("n_ids",len(ids)-1),("score_keys",["score_mil"]),("selected_score_key","score_mil"),("checkpoint_only",False),("scores_sha256","0"*64),("checkpoint_sha256","0"*64),("train_meta_sha256","0"*64),("scores",str(repo/"elsewhere")),("checkpoint",str(repo/"elsewhere")),("train_meta",str(repo/"elsewhere"))):
   bad=dict(base);bad[field]=value;write(bad);reject(lambda:verify_producer(repo,root,method,corpus,seed,"score_align"))
  write(base);mp.write_text(mp.read_text()+" ");reject(lambda:verify_producer(repo,root,method,corpus,seed,"score_align"))
 print("Relation-V9 producer tamper tests: PASS")
if __name__=="__main__":main()
