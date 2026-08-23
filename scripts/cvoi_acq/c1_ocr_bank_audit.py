"""Independent structural audit for the C1 OCR bank/unified registry."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from .common import atomic_json,sha256_file
from .artifacts import write_jsonl
from .ocr_embedding_bank import dense_outcome_ref

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--bank-root",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    audit=json.loads((a.bank_root/"audit.json").read_text());report={"schema":"cvoi-c1-ocr-bank-independent-audit/1","bank_root":str(a.bank_root.resolve()),"splits":{},"candidate_metric_computed":False,"test_contact_count":audit["preflight"]["contact"]["test_contact_count"]}
    for split,n in (("train",744),("val",107)):
        side=[json.loads(x) for x in (a.bank_root/f"{split}_ocr_embedding_sidecar.jsonl").open() if x.strip()];reg=[json.loads(x) for x in (a.bank_root/f"{split}_unified_actions.jsonl").open() if x.strip()]
        dense_path=Path("artifacts/cvoi_acq/premetric-v2/visual-v10")/f"{split}_dense_sidecar.jsonl"
        dense=[json.loads(x) for x in dense_path.open() if x.strip()];dense_by={}
        dense_sidecar_sha256=sha256_file(dense_path)
        for d in dense:dense_by.setdefault(d["action_id"],[]).append(d)
        bank=np.memmap(a.bank_root/f"{split}_ocr_embeddings.f32",dtype="<f4",mode="r",shape=(n*30,768));norm=np.linalg.norm(bank,axis=1);empty=np.asarray([x["empty"] for x in side])
        ids={x["video_id"] for x in side};ocr=[x for x in reg if x["action_type"]=="ocr"];dense=[x for x in reg if x["action_type"]=="dense4"]
        if len(side)!=n*30 or len(reg)!=n*60 or len(ids)!=n or len(ocr)!=len(dense)!=n*30:raise RuntimeError("HALT_C1_AUDIT_CARDINALITY")
        if not np.all(norm[empty]==0) or not np.allclose(norm[~empty],1,atol=2e-5):raise RuntimeError("HALT_C1_AUDIT_NORM")
        if any(x["cost_join_status"]!="PENDING_C6" for x in reg):raise RuntimeError("HALT_C1_AUDIT_COST_STATUS")
        if any("test" in x["video_id"].lower() for x in reg):raise RuntimeError("HALT_TEST_CONTACT")
        for i,x in enumerate(side):
            if x["feature_row"]!=i or x["feature_sha256"]!=hashlib.sha256(bank[i].tobytes()).hexdigest():raise RuntimeError("HALT_C1_AUDIT_FEATURE_HASH")
        side_by={x["action_id"]:x for x in side};enriched=[]
        for row in reg:
            q=dict(row)
            if q["action_type"]=="ocr":
                s=side_by[q["action_id"]];q["outcome_sha256"]=s["feature_sha256"];q["provenance"]={"normalized_text_sha256":s["normalized_text_sha256"],"encoder_weights_sha256":audit["preflight"]["weights_sha256"],"encoder_revision":audit["preflight"]["revision"]}
            else:
                ds=dense_by[q["action_id"]];q["outcome_ref"]=dense_outcome_ref(split,ds)
                expected_row=int(q["outcome_ref"].rsplit(":",1)[1])
                if any(int(x["feature_row"])!=expected_row for x in ds):raise RuntimeError("HALT_C1_AUDIT_DENSE_REF")
                q["outcome_sha256"]=hashlib.sha256("".join(x["feature_sha256"] for x in sorted(ds,key=lambda z:z["frame_slot"])).encode()).hexdigest();q["provenance"]={"dense_sidecar_sha256":dense_sidecar_sha256,"action_feature_row":expected_row,"frame_feature_sha256":[x["feature_sha256"] for x in sorted(ds,key=lambda z:z["frame_slot"])]}
            q["join_status"]={"outcome":"JOINED","cost":"PENDING_C6","base":"REFERENCE_C7_ONLY"};enriched.append(q)
        enriched_path=a.out.parent/f"c1_{split}_unified_actions_provenance_v2.jsonl";write_jsonl(enriched_path,enriched)
        report["splits"][split]={"videos":n,"embedding_rows":len(side),"unified_rows":len(reg),"empty_rows":int(empty.sum()),"nonempty_unit_norm":True,"feature_hashes_replayed":True,"dense_outcome_refs_rebuilt_and_replayed":True,"cost_join_status":"PENDING_C6","bank_sha256":sha256_file(a.bank_root/f"{split}_ocr_embeddings.f32"),"sidecar_sha256":sha256_file(a.bank_root/f"{split}_ocr_embedding_sidecar.jsonl"),"source_registry_sha256":sha256_file(a.bank_root/f"{split}_unified_actions.jsonl"),"provenance_registry_path":str(enriched_path),"provenance_registry_sha256":sha256_file(enriched_path)}
    report["base_asset_reference"]=audit["base_asset_reference"];report["encoder"]={k:audit["preflight"][k] for k in ("model_id","revision","output_dim","model_config_sha256","tokenizer_tree_sha256","weights_sha256")};atomic_json(a.out,report)
if __name__=="__main__":main()
