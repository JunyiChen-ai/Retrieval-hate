#!/usr/bin/env python3
"""Sequential single-frame + synchronized OCR/ASR V21 forward; never reads GT."""
import argparse, hashlib, json, math, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src/duplex'),str(ROOT/'scripts/reproduction_baselines')]

def nearest_ocr(rows, center, max_chars=1200):
    if not rows:return ""
    r=min(rows,key=lambda x:(abs(float(x['t_mid'])-center),int(x['window_k'])))
    texts=[x['text'].strip() for x in r.get('texts',[]) if float(x.get('conf',0))>=.5 and x.get('text','').strip()]
    return " ".join(texts)[:max_chars]

def frame_at(path, t):
    import decord
    from PIL import Image
    vr=decord.VideoReader(str(path),num_threads=1); fps=max(float(vr.get_avg_fps()),1e-6)
    idx=min(len(vr)-1,max(0,int(round(t*fps))))
    return Image.fromarray(vr[idx].asnumpy()).convert('RGB'), idx, fps

def user_text(policy, question, asr, ocr=None):
    parts=[policy]
    if ocr is not None: parts.append("Visible OCR (may contain recognition errors): " + (ocr or "[none]"))
    parts.append("Synchronized speech: " + (asr.strip() or "[none]")); parts.append(question)
    return "\n\n".join(parts)

def main():
    import torch
    from score_duplex_probe import build_binary_token_ids
    from relation_v4.io import sha256
    from vera_adapter import video_path
    p=argparse.ArgumentParser();p.add_argument('--out-dir',required=True);a=p.parse_args();out=Path(a.out_dir)
    cfg=json.load(open(out/'preregistered_config.json')); assert cfg['split']=='val' and not cfg['test_access']
    asr={r['video_id']:r for r in map(json.loads,open(cfg['asr']['path']))}
    ocr={}
    for r in map(json.loads,open(cfg['ocr']['path'])): ocr.setdefault(r['video_id'],[]).append(r)
    from transformers import AutoModelForImageTextToText,AutoProcessor
    proc=AutoProcessor.from_pretrained(cfg['model']); tok=proc.tokenizer; ids=build_binary_token_ids(tok)
    model=AutoModelForImageTextToText.from_pretrained(cfg['model'],dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
    yes=torch.tensor(sorted(ids['Yes']),device=model.device); no=torch.tensor(sorted(ids['No']),device=model.device)
    raw=out/'per_chunk_raw.jsonl'; done=set()
    if raw.exists():
        for r in map(json.loads,open(raw)):done.add((r['video_id'],r['chunk_index']))
    def score(text,image=None):
        content=[]
        if image is not None:content.append({'type':'image','image':image})
        content.append({'type':'text','text':text}); msgs=[{'role':'user','content':content}]
        prompt=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        enc=proc(text=[prompt],images=[image] if image is not None else None,return_tensors='pt',padding=True)
        enc={k:v.to(model.device) for k,v in enc.items()}
        with torch.inference_mode():lg=model(**enc,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        z=float(torch.logsumexp(lg[yes],0)-torch.logsumexp(lg[no],0))
        return z,hashlib.sha256(prompt.encode()).hexdigest(),int(enc['input_ids'].shape[1])
    with open(raw,'a') as f:
      for vid in cfg['cohort']:
       path=video_path(cfg['corpus'],vid)
       for i,ch in enumerate(asr[vid]['chunks']):
        if (vid,i) in done:continue
        if not isinstance(ch.get('start'),(int,float)) or not isinstance(ch.get('end'),(int,float)) or not math.isfinite(float(ch['start'])) or not math.isfinite(float(ch['end'])) or float(ch['end'])<=float(ch['start']):continue
        st,en=float(ch['start']),float(ch['end']);center=(st+en)/2; speech=ch.get('text','')[:cfg['asr']['max_chars']]
        image,fi,fps=frame_at(path,center); ot=nearest_ocr(ocr.get(vid,[]),center)
        za,sha_a,na=score(user_text(cfg['policy'],cfg['question'],speech))
        zm,sha_m,nm=score(user_text(cfg['policy'],cfg['question'],speech,ot),image)
        rec={'video_id':vid,'chunk_index':i,'start':st,'end':en,'center':center,'frame_index':fi,'fps':fps,
             'ocr_sha256':hashlib.sha256(ot.encode()).hexdigest(),'asr_sha256':hashlib.sha256(speech.encode()).hexdigest(),
             'scores':{'asr_only':za,'frame_ocr_asr':zm},'prompt_sha256':{'asr_only':sha_a,'frame_ocr_asr':sha_m},'tokens':{'asr_only':na,'frame_ocr_asr':nm}}
        assert all(math.isfinite(x) for x in rec['scores'].values());f.write(json.dumps(rec)+'\n');f.flush()
    man={'raw_frozen_before_gt':True,'raw':str(raw.resolve()),'raw_sha256':sha256(raw),'n_rows':sum(1 for _ in open(raw)),'config_sha256':sha256(out/'preregistered_config.json')}
    (out/'raw_manifest.json').write_text(json.dumps(man,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
