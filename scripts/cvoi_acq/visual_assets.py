from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from .actions import K, durations, read_frame, split_rows, video_path
from .common import ContactLedger, atomic_json, atomic_write, canonical_bytes
from .common import sha256_file

def model_provenance(model,proc):
    import hashlib
    from huggingface_hub.constants import HF_HUB_CACHE
    repo=Path(HF_HUB_CACHE)/"models--openai--clip-vit-large-patch14-336"/"snapshots"
    files={};weight_found=False
    if repo.exists():
        snapshots=sorted([p for p in repo.iterdir() if p.is_dir()])
        ref=repo.parent/"refs/main"
        snap=(repo/ref.read_text().strip()) if ref.exists() else (snapshots[-1] if snapshots else None)
        if snap is not None:
            for name in ("config.json","preprocessor_config.json","model.safetensors","pytorch_model.bin"):
                p=snap/name
                if p.exists():
                    files[name]={"resolved_path":str(p.resolve()),"sha256":sha256_file(p)}
                    weight_found=weight_found or name in ("model.safetensors","pytorch_model.bin")
    config_hash=hashlib.sha256(canonical_bytes(model.config.to_dict())).hexdigest()
    proc_hash=hashlib.sha256(canonical_bytes(proc.to_dict())).hexdigest()
    if not weight_found:
        raise RuntimeError("HALT_MODEL_WEIGHTS_UNPINNED")
    return {"resolved_files":files,"config_object_sha256":config_hash,"processor_object_sha256":proc_hash}

def decode_points(path,times):
    from decord import VideoReader,cpu
    try:vr=VideoReader(str(path),ctx=cpu(0),num_threads=2);fps=float(vr.get_avg_fps() or 0);n=len(vr)
    except Exception as exc:return [(None,None,type(exc).__name__)]*len(times)
    if fps<=0 or n<=0:return [(None,None,"invalid_stream")]*len(times)
    idx=[max(0,min(n-1,int(np.floor(t*fps+0.5)))) for t in times]
    try:
        batch=vr.get_batch(idx).asnumpy();return [(batch[j],idx[j]/fps,None) for j in range(len(idx))]
    except Exception:
        out=[]
        for j,q in enumerate(idx):
            try:out.append((vr[q].asnumpy(),q/fps,None))
            except Exception as exc:out.append((None,None,type(exc).__name__))
        return out

def build(role, out_dir, duration_registry):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, CLIPVisionModel
    ledger=ContactLedger(); rows=split_rows(role,ledger); dur=durations(ledger,duration_registry)
    model_id="openai/clip-vit-large-patch14-336"
    proc=AutoImageProcessor.from_pretrained(model_id,local_files_only=True)
    model=CLIPVisionModel.from_pretrained(model_id,local_files_only=True).eval().cuda()
    if int(model.config.hidden_size)!=1024:raise RuntimeError("HALT_VISUAL_DIM")
    provenance=model_provenance(model,proc)
    side=[]; cheap_side=[]; cheap=[]; dense=[]; t0=time.time()
    with torch.inference_mode():
        for vi,row in enumerate(sorted(rows,key=lambda r:str(r["id"]))):
            vid=str(row["id"]); path=video_path(vid); D=dur.get(vid)
            if path is None or D is None: raise RuntimeError("HALT_VISUAL_SOURCE:"+vid)
            ledger.register(path,"raw_video_visual_action")
            all_times=[]
            for q in range(K):
                ql,qr=q*D/K,(q+1)*D/K
                all_times.extend([(ql+qr)/2]+[ql+(m+.5)*(qr-ql)/4 for m in range(4)])
            decoded=decode_points(path,all_times)
            for k in range(K):
                left,right=k*D/K,(k+1)*D/K
                times=[(left+right)/2]+[left+(m+.5)*(right-left)/4 for m in range(4)]
                frames=[]; metadata=[]
                for slot,t in enumerate(times):
                    rgb,actual,decode_error=decoded[5*k+slot]
                    digest=hashlib.sha256(rgb.tobytes()).hexdigest() if rgb is not None else None
                    shape=list(rgb.shape[:2][::-1]) if rgb is not None else [None,None]
                    metadata.append([slot,t,actual,"ok" if rgb is not None else "failed",digest,shape,None,decode_error])
                    frames.append(Image.fromarray(rgb) if rgb is not None else None)
                cheap_ok=frames[0] is not None;dense_good=[j for j in range(1,5) if frames[j] is not None]
                cheap_status="ok" if cheap_ok else "EMPTY"
                dense_status="ok" if len(dense_good)==4 else ("fallback" if dense_good else "EMPTY")
                # Keep the two action namespaces isolated: dense fallback may
                # only use another dense slot; cheap never borrows a dense frame.
                model_frames=list(frames);placeholder=next((x for x in frames if x is not None),None)
                if placeholder is None:
                    encoded=np.zeros((5,1024),np.float32)
                else:
                    if model_frames[0] is None:model_frames[0]=placeholder
                    for j in range(1,5):
                        if model_frames[j] is None:
                            if dense_good:
                                source=min(dense_good,key=lambda q:abs(q-j));model_frames[j]=model_frames[source]
                                metadata[j][3]="fallback";metadata[j][6]=source-1
                            else:model_frames[j]=placeholder
                    pixels=proc(images=model_frames,return_tensors="pt")["pixel_values"].cuda()
                    encoded=model(pixel_values=pixels).pooler_output.float().cpu().numpy()
                    if not cheap_ok:encoded[0]=0
                    if not dense_good:encoded[1:]=0
                cheap.append(encoded[0]); dense.append(encoded[1:]); aid=vid+":dense4:"+format(k,"02d")
                cslot,ct,cactual,cstatus,cdigest,cshape,cfallback,cerror=metadata[0]
                cheap_side.append({"schema":"cvoi-cheap-midpoint/1","video_id":vid,"window_id":k,
                                   "requested_t_s":ct,"actual_t_s":cactual,"decode_status":cstatus,
                                   "fallback_source_slot":cfallback,"decode_error":cerror,"action_status":cheap_status,
                                   "width":cshape[0],"height":cshape[1],"frame_rgb_sha256":cdigest,
                                   "feature_row":len(cheap)-1,
                                   "feature_sha256":hashlib.sha256(encoded[0].astype("<f4").tobytes()).hexdigest()})
                for m in range(4):
                    slot,t,actual,status,digest,shape,fallback,decode_error=metadata[m+1]
                    side.append({"schema":"cvoi-dense-frame/1","video_id":vid,"action_id":aid,
                                 "window_id":k,"frame_slot":m,"requested_t_s":t,"actual_t_s":actual,
                                 "decode_status":status,"fallback_source_slot":fallback,"decode_error":decode_error,
                                 "action_status":dense_status,"width":shape[0],"height":shape[1],
                                 "frame_rgb_sha256":digest,"feature_row":len(dense)-1,
                                 "feature_sha256":hashlib.sha256(encoded[m+1].astype("<f4").tobytes()).hexdigest()})
            if (vi+1)%10==0: print("[visual] %s %d/%d elapsed_min=%.1f"%(role,vi+1,len(rows),(time.time()-t0)/60),flush=True)
    out_dir.mkdir(parents=True,exist_ok=True)
    cheap_arr=np.asarray(cheap,dtype="<f4");dense_arr=np.asarray(dense,dtype="<f4")
    atomic_write(out_dir/(role+"_cheap_midpoint.f32"),cheap_arr.tobytes())
    atomic_write(out_dir/(role+"_dense4.f32"),dense_arr.tobytes())
    atomic_write(out_dir/(role+"_dense_sidecar.jsonl"),b"".join(canonical_bytes(x) for x in side))
    atomic_write(out_dir/(role+"_cheap_sidecar.jsonl"),b"".join(canonical_bytes(x) for x in cheap_side))
    meta={"schema":"cvoi-visual-assets/1","role":role,"n_videos":len(rows),"K":K,
          "cheap_shape":[len(rows)*K,1024],"dense_shape":[len(rows)*K,4,1024],
          "dtype":"little-endian-float32","model":model_id,"model_provenance":provenance,"wall_seconds":time.time()-t0,
          "files":{"cheap":role+"_cheap_midpoint.f32","dense":role+"_dense4.f32",
                   "sidecar":role+"_dense_sidecar.jsonl","cheap_sidecar":role+"_cheap_sidecar.jsonl"},
          "contact":ledger.snapshot()}
    atomic_json(out_dir/(role+"_visual_meta.json"),meta); return meta

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--role",choices=["train","val"],required=True);ap.add_argument("--out-dir",type=Path,required=True);ap.add_argument("--duration-registry",type=Path,required=True);args=ap.parse_args()
    print(json.dumps(build(args.role,args.out_dir,args.duration_registry),sort_keys=True))
if __name__=="__main__":main()
