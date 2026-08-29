#!/usr/bin/env python3
"""Create V9 MHC manifests only after every train producer is complete."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from relation_v4.io import sha256
from relation_v2.protocol import frozen_splits
from hate_common import data as hdata
SEEDS=(234,2025,3407)
POOLS={"mhclip_en":[("macilsd_audio","score_mil"),("cmhkf","score_align")],"mhclip_zh":[("fed_wsvad_1client","score_align"),("macilsd","score_av"),("cmhkf","score_align")]}
def verify_producer(repo,root,method,corpus,seed,key):
 scores=root/"scores.jsonl";producer=root/"producer_manifest.json";complete_path=root/"COMPLETE.json";checkpoint=repo/f"results/reproduction/official_val/final/{method}/{corpus}/seed_{seed}/model.pth";train_meta=checkpoint.with_name("train_meta.json")
 for path in (scores,producer,complete_path,checkpoint,train_meta):
  if not path.is_file():raise RuntimeError(f"missing producer chain file: {path}")
 meta=json.load(open(producer));expected={"method":method,"corpus":corpus,"split":"train","seed":seed,"checkpoint_only":True}
 if any(meta.get(k)!=v for k,v in expected.items()):raise RuntimeError(f"producer identity mismatch: {root}")
 if key not in meta.get("score_keys",[]) or meta.get("selected_score_key")!=key:raise RuntimeError(f"producer score key mismatch: {root}")
 ids=list(frozen_splits(corpus)["train"]);records=hdata.load_scores_jsonl(scores)
 if meta.get("n_ids")!=len(ids) or set(records)!=set(ids):raise RuntimeError(f"producer n_ids/coverage mismatch: {root}")
 for vid in ids:
  if key not in records[vid]:raise RuntimeError(f"producer selected score key absent: {root}/{vid}")
 checks=(("scores","scores_sha256",scores),("checkpoint","checkpoint_sha256",checkpoint),("train_meta","train_meta_sha256",train_meta))
 for path_key,hash_key,actual in checks:
  if Path(meta.get(path_key,"")).resolve()!=actual.resolve() or meta.get(hash_key)!=sha256(actual):raise RuntimeError(f"producer {path_key} path/hash mismatch: {root}")
 complete=json.load(open(complete_path))
 if Path(complete.get("producer_manifest","")).resolve()!=producer.resolve() or complete.get("producer_manifest_sha256")!=sha256(producer):raise RuntimeError(f"producer manifest SHA mismatch: {root}")
 for field,actual in (("scores_sha256",scores),("checkpoint_sha256",checkpoint),("train_meta_sha256",train_meta)):
  if complete.get(field)!=sha256(actual):raise RuntimeError(f"COMPLETE {field} mismatch: {root}")
 return scores
def main():
 repo=HERE.parents[2];out=repo/"results/reproduction/relation_v9/manifests";out.mkdir(parents=True,exist_ok=True)
 for corpus,pool in POOLS.items():
  base=json.load(open(repo/f"results/reproduction/relation_v8/manifests/{corpus}_equal.json"));by_name={x["name"]:x for x in base["experts"]};experts=[]
  for method,key in pool:
   paths=[]
   for seed in SEEDS:
    root=repo/f"results/reproduction/relation_v9/train_dense/{method}/{corpus}/seed_{seed}";scores=verify_producer(repo,root,method,corpus,seed,key)
    paths.append(str(scores.relative_to(repo)))
   old=by_name[method];experts.append({"name":method,"score_key":key,"train_scores":paths,"train_producer_manifests":[str((repo/f"results/reproduction/relation_v9/train_dense/{method}/{corpus}/seed_{seed}/producer_manifest.json").relative_to(repo)) for seed in SEEDS],"train_producer_manifest_sha256":[sha256(repo/f"results/reproduction/relation_v9/train_dense/{method}/{corpus}/seed_{seed}/producer_manifest.json") for seed in SEEDS],"val_scores":old["val_scores"],"test_scores":old["test_scores"]})
  target=out/f"{corpus}_train.json"
  if target.exists():raise RuntimeError(f"refuse overwrite: {target}")
  target.write_text(json.dumps({"corpus":corpus,"experts":experts,"construction":"written only after all checkpoint-only train producers passed"},indent=2)+"\n");print(target)
if __name__=="__main__":main()
