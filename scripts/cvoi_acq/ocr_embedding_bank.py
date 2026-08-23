"""Build frozen OCR action embeddings and a non-candidate unified action join."""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np

from .common import ContactLedger, atomic_json, atomic_write, canonical_bytes, sha256_file
from .artifacts import write_jsonl

MODEL_ID="openai/clip-vit-large-patch14-336"
REVISION="ce19dc912ca5cd21c8a653c79e251e808ccabcd1"

def dense_outcome_ref(split:str,frames:list[dict])->str:
    """Dense sidecar feature_row is already the action row, not a frame row."""
    rows={int(x["feature_row"]) for x in frames}
    if len(frames)!=4 or len(rows)!=1:raise RuntimeError("HALT_DENSE_ACTION_FEATURE_ROW")
    return f"artifacts/cvoi_acq/premetric-v2/visual-v10/{split}_dense4.f32:{rows.pop()}"

def tree_hash(root:Path,names:list[str]):
    rows=[]
    for name in names:
        p=root/name
        if not p.exists():raise RuntimeError("HALT_OCR_ENCODER_FILE:"+name)
        rows.append({"name":name,"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    return hashlib.sha256(canonical_bytes(rows)).hexdigest(),rows

def load_rows(path,ledger,role):
    ledger.register(path,role);return [json.loads(x) for x in path.open() if x.strip()]

def audit_inputs(root:Path):
    ledger=ContactLedger();actions={};dense={}
    for split,n in (("train",744),("val",107)):
        op=root/f"artifacts/cvoi_acq/premetric-v2/actions/{split}_ocr_actions.jsonl"
        dp=root/f"artifacts/cvoi_acq/premetric-v2/visual-v10/{split}_dense_sidecar.jsonl"
        rr=load_rows(op,ledger,f"{split}_ocr_actions");dd=load_rows(dp,ledger,f"{split}_dense_sidecar")
        if len(rr)!=n*30 or len(dd)!=n*30*4:raise RuntimeError("HALT_C1_ACTION_CARDINALITY:"+split)
        okeys={(r["video_id"],int(r["window_id"])) for r in rr};dkeys={(r["video_id"],int(r["window_id"])) for r in dd}
        if len(okeys)!=n*30 or okeys!=dkeys:raise RuntimeError("HALT_C1_ACTION_JOIN:"+split)
        if any("test" in str(r.get("video_id","")).lower() for r in rr):raise RuntimeError("HALT_TEST_CONTACT")
        actions[split]=rr;dense[split]=dd
    return actions,dense,ledger

def encode_texts(texts,model_path:Path,batch_size=64):
    import torch
    from transformers import CLIPTextModel,CLIPTokenizerFast
    tok=CLIPTokenizerFast.from_pretrained(model_path,local_files_only=True)
    model=CLIPTextModel.from_pretrained(model_path,local_files_only=True).eval().to("cpu")
    if int(model.config.hidden_size)!=768:raise RuntimeError("HALT_OCR_ENCODER_DIM")
    unique=sorted({x for x in texts if x});encoded={};chunks=[];owners=[]
    for text in unique:
        ids=tok.encode(text,add_special_tokens=False)
        for j in range(0,max(1,len(ids)),75):
            content=ids[j:j+75];chunks.append([tok.bos_token_id]+content+[tok.eos_token_id]);owners.append(text)
    accum={x:[] for x in unique}
    with torch.inference_mode():
        for j in range(0,len(chunks),batch_size):
            part=chunks[j:j+batch_size];m=max(map(len,part));mask=[[1]*len(q)+[0]*(m-len(q)) for q in part];pad=[q+[tok.pad_token_id]*(m-len(q)) for q in part]
            out=model(input_ids=torch.tensor(pad),attention_mask=torch.tensor(mask)).last_hidden_state
            mm=torch.tensor(mask,dtype=out.dtype).unsqueeze(-1);vec=(out*mm).sum(1)/mm.sum(1)
            for owner,v in zip(owners[j:j+batch_size],vec):accum[owner].append(v.cpu().numpy())
    for text,vs in accum.items():
        v=np.mean(vs,axis=0).astype("<f4");norm=float(np.linalg.norm(v));encoded[text]=v/max(norm,1e-12)
    return encoded

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--preflight-only",action="store_true");ap.add_argument("--batch-size",type=int,default=64);a=ap.parse_args()
    root=Path(__file__).resolve().parents[2];actions,dense,ledger=audit_inputs(root)
    snap=Path.home()/f".cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/{REVISION}"
    config_hash,config_files=tree_hash(snap,["config.json"]);tokenizer_hash,tokenizer_files=tree_hash(snap,["tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","special_tokens_map.json"]);weights_hash,weight_files=tree_hash(snap,["pytorch_model.bin"])
    pre={"schema":"cvoi-ocr-embedding-preflight/1","model_id":MODEL_ID,"revision":REVISION,"output_dim":768,"pooling":"nonoverlap_75_content_tokens_masked_token_mean_then_chunk_mean_l2","empty_outcome":"zero_bank_row_plus_learned_EMPTY_OCR_downstream","model_config_sha256":config_hash,"tokenizer_tree_sha256":tokenizer_hash,"weights_sha256":weights_hash,"model_files":config_files+tokenizer_files+weight_files,"counts":{s:len(r) for s,r in actions.items()},"contact":ledger.snapshot(),"candidate_metric_computed":False}
    if a.preflight_only:atomic_json(a.out,pre);return
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    encoded=encode_texts([r["normalized_text"] for s in actions.values() for r in s],snap,a.batch_size)
    audit={"schema":"cvoi-c1-ocr-unified-audit/1","preflight":pre,"splits":{},"base_asset_reference":{"gate":"C7","path":"artifacts/cvoi_acq/premetric-v2/base/base_selection_v1.json","sha256":sha256_file(root/"artifacts/cvoi_acq/premetric-v2/base/base_selection_v1.json")},"cost_status":"PENDING_C6","dense_status":"REFERENCES_C5_WITHOUT_PROMOTION","candidate_metric_computed":False,"environment":{"python":platform.python_version(),"numpy":np.__version__}}
    for split,rr in actions.items():
        rr=sorted(rr,key=lambda r:(r["video_id"],int(r["window_id"])));bank=np.zeros((len(rr),768),dtype="<f4");side=[]
        for i,r in enumerate(rr):
            text=r["normalized_text"]
            if text:bank[i]=encoded[text]
            side.append({"schema":"cvoi-ocr-embedding-row/1","action_id":r["action_id"],"video_id":r["video_id"],"window_id":r["window_id"],"feature_row":i,"empty":not bool(text),"normalized_text_sha256":hashlib.sha256(text.encode()).hexdigest(),"feature_sha256":hashlib.sha256(bank[i].tobytes()).hexdigest()})
        atomic_write(a.out/f"{split}_ocr_embeddings.f32",bank.tobytes());write_jsonl(a.out/f"{split}_ocr_embedding_sidecar.jsonl",side)
        by_dense={}
        for d in dense[split]:by_dense.setdefault((d["video_id"],int(d["window_id"])),[]).append(d)
        registry=[]
        for r in rr:registry.append({"schema":"cvoi-unified-action-registry/1","split":split,"video_id":r["video_id"],"window_id":r["window_id"],"action_id":r["action_id"],"action_type":"ocr","outcome_status":r["engine_status"],"outcome_ref":f"{split}_ocr_embeddings.f32:{side[len(registry)][ 'feature_row']}","cost_join_status":"PENDING_C6"})
        for v,w in sorted(by_dense):
            ds=sorted(by_dense[(v,w)],key=lambda x:x["frame_slot"]);registry.append({"schema":"cvoi-unified-action-registry/1","split":split,"video_id":v,"window_id":w,"action_id":f"{v}:dense4:{w:02d}","action_type":"dense4","outcome_status":"ok" if any(x["decode_status"]=="ok" for x in ds) else "missing","outcome_ref":dense_outcome_ref(split,ds),"cost_join_status":"PENDING_C6"})
        registry=sorted(registry,key=lambda x:(x["video_id"],0 if x["action_type"]=="ocr" else 1,x["window_id"]));write_jsonl(a.out/f"{split}_unified_actions.jsonl",registry)
        audit["splits"][split]={"videos":len({r["video_id"] for r in rr}),"ocr_rows":len(rr),"dense_action_rows":len(by_dense),"unified_rows":len(registry),"embedding_sha256":sha256_file(a.out/f"{split}_ocr_embeddings.f32"),"sidecar_sha256":sha256_file(a.out/f"{split}_ocr_embedding_sidecar.jsonl"),"unified_sha256":sha256_file(a.out/f"{split}_unified_actions.jsonl")}
    atomic_json(a.out/"audit.json",audit)

if __name__=="__main__":main()
