from __future__ import annotations

import argparse,json
from pathlib import Path

import torch

from .actions import split_rows
from .common import ContactLedger,atomic_json,sha256_file


def flat_ids(value):
    if isinstance(value,(list,tuple)) and len(value)==1 and isinstance(value[0],(list,tuple)):value=value[0]
    return [str(x) for x in value]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--train",type=Path,required=True);ap.add_argument("--val",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    ledger=ContactLedger();roles={}
    for role,path in (("train",a.train),("val",a.val)):
        ledger.register(path,"frozen_base_feature_cache")
        obj=torch.load(path,map_location="cpu",weights_only=False);ids=flat_ids(obj["ids"])
        expected=[str(r["id"]) for r in split_rows(role,ledger)]
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected):raise RuntimeError("HALT_BASE_ID_ALIGNMENT:"+role)
        for key in ("img_feats","text_feats"):
            if len(obj[key])!=len(ids):raise RuntimeError("HALT_BASE_SHAPE:"+role+":"+key)
        roles[role]={"path":str(path.resolve()),"sha256":sha256_file(path),"n_ids":len(ids),
                     "id_order_sha256":__import__("hashlib").sha256("\n".join(ids).encode()).hexdigest(),
                     "feature_shapes":{"img_feats":list(obj["img_feats"].shape),"text_feats":list(obj["text_feats"].shape)}}
    atomic_json(a.out,{"schema":"cvoi-base-selection/1","selection_rule":"appendix fallback: frozen Qwen2.5-VL-7B non-LoRA because historical train-OOF winner ordering is ambiguous",
                       "model":"Qwen2.5-VL-7B-Instruct_HF","roles":roles,"contact":ledger.snapshot(),"candidate_metrics_read":False})

if __name__=="__main__":main()
