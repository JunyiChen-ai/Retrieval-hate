#!/usr/bin/env python3
"""Create a provenance addendum without modifying/retraining approved V24 artifacts."""
import argparse,json
from pathlib import Path
from train import sha

OLD_ID_KEYS={"corpus","split","ids","v23_global_source_sha256","producer_sha256"}
ADD_KEYS={"schema_version","status","corpus","train_protocol_sha256","train_id_manifest_sha256","train_join_manifest_sha256","train_join_producer_sha256","train_evidence_manifest_sha256","train_evidence_config_sha256","train_evidence_producer_sha256","selector_source_sha256","val_join_source_sha256"}

def create(train_dir, train_id, train_join, train_evidence_dir, selector, val_join, out):
    td=Path(train_dir); tp_path=td/"train_protocol.json"; tp=json.load(open(tp_path)); im=json.load(open(train_id)); jm=json.load(open(train_join)); ed=Path(train_evidence_dir); ep=ed/"evidence_manifest.json"; cp=ed/"preregistered_config.json"; cfg=json.load(open(cp)); em=json.load(open(ep))
    if set(im)!=OLD_ID_KEYS or im.get("split")!="train": raise RuntimeError("approved train ID manifest is not expected immutable schema")
    if sha(train_id)!=tp.get("id_manifest_sha256") or sha(train_join)=="": raise RuntimeError("train protocol hash mismatch")
    if jm.get("join_producer_sha256")!=im.get("producer_sha256") or tp.get("producer_sha256")!=im.get("producer_sha256"): raise RuntimeError("train join provenance mismatch")
    if jm.get("evidence_manifest_sha256")!=sha(ep) or em.get("config_sha256")!=sha(cp): raise RuntimeError("train evidence hash mismatch")
    if cfg.get("split")!="train" or cfg.get("labels_read") is not False or cfg.get("producer_sha256")!=cfg.get("local_forward_sha256"): raise RuntimeError("train evidence identity mismatch")
    a={"schema_version":"v24_protocol_addendum_v1","status":"NO_RETRAIN_PROVENANCE_MIGRATION","corpus":tp["corpus"],"train_protocol_sha256":sha(tp_path),"train_id_manifest_sha256":sha(train_id),"train_join_manifest_sha256":sha(train_join),"train_join_producer_sha256":jm["join_producer_sha256"],"train_evidence_manifest_sha256":sha(ep),"train_evidence_config_sha256":sha(cp),"train_evidence_producer_sha256":cfg["producer_sha256"],"selector_source_sha256":sha(selector),"val_join_source_sha256":sha(val_join)}
    Path(out).write_text(json.dumps(a,indent=2,sort_keys=True)+"\n"); return a

def load(path):
    a=json.load(open(path))
    if set(a)!=ADD_KEYS or a.get("schema_version")!="v24_protocol_addendum_v1" or a.get("status")!="NO_RETRAIN_PROVENANCE_MIGRATION": raise RuntimeError("invalid exact protocol addendum")
    for k,v in a.items():
        if k.endswith("sha256") and (not isinstance(v,str) or len(v)!=64): raise RuntimeError("invalid addendum hash")
    return a

def main():
    p=argparse.ArgumentParser();p.add_argument("--train-dir",required=True);p.add_argument("--train-id-manifest",required=True);p.add_argument("--train-join-manifest",required=True);p.add_argument("--train-evidence-dir",required=True);p.add_argument("--selector",required=True);p.add_argument("--val-join",required=True);p.add_argument("--out",required=True);a=p.parse_args();create(a.train_dir,a.train_id_manifest,a.train_join_manifest,a.train_evidence_dir,a.selector,a.val_join,a.out)
if __name__=="__main__":main()
