#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/reproduction_baselines'))
from relation_v8.run import atomic_json
from relation_v4.io import sha256

PUBLIC=ROOT/'results/reproduction/thvl_sealed/validation_opaque_manifest.json'
ASR=ROOT/'results/steward_private/thvl_bench/val32_timestamped_chunks.jsonl'
OUT=ROOT/'results/steward_private/thvl_bench/v16_val_raw_frozen_32_v2'
FORWARD=ROOT/'scripts/reproduction_baselines/relation_v16/forward.py'
PROMPT=ROOT/'scripts/duplex/masked_parallel_isolation_pilot.py'
MODEL_REV='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
ARMS=['masked_branch_reset','masked_continuous','causal_branch_reset','causal_continuous']

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    pub=json.load(open(PUBLIC)); rows=[json.loads(x) for x in open(ASR)]
    ids=sorted(r['hashed_id'] for r in pub['records'])
    if len(ids)!=32 or len(set(ids))!=32: raise RuntimeError('opaque validation cohort must be 32 unique IDs')
    rec={r['hashed_id']:r for r in rows}
    if len(rec)!=len(rows): raise RuntimeError('duplicate ASR hashed_id')
    extra=sorted(set(rec)-set(ids)); missing=sorted(set(ids)-set(rec))
    if extra: raise RuntimeError(f'ASR IDs outside opaque validation cohort: {extra}')
    sanitized={}; invalid=[]
    for v in ids:
        good=[]
        for i,ch in enumerate(rec.get(v,{}).get('chunks',[])):
            s,e=ch.get('start'),ch.get('end')
            if not isinstance(s,(int,float)) or not isinstance(e,(int,float)) or not math.isfinite(s) or not math.isfinite(e) or e<=s:
                invalid.append({'hashed_id':v,'source_index':i,'reason':'requires finite start/end and end>start'})
                continue
            good.append({'start':float(s),'end':float(e),'text':ch.get('text',''),'source_index':i})
        sanitized[v]=good
    if sum(map(len,sanitized.values()))+len(invalid)!=sum(len(r.get('chunks',[])) for r in rows): raise RuntimeError('timestamp sanitation accounting mismatch')
    subset=sorted([v for v in ids if sanitized[v]],key=lambda v:(-len(sanitized[v]),v))[:12]
    snap=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots'/MODEL_REV
    if not snap.is_dir(): raise RuntimeError('pinned local Qwen snapshot unavailable')
    media=ROOT/'results/reproduction/thvl_sealed/validation_media_manifest.json'; correction=ROOT/'results/reproduction/thvl_sealed/validation_media_reconciliation_correction.json'
    if missing: raise RuntimeError(f'formal reconciled forward requires 32/32 ASR, missing {missing}')
    cfg={'method':'relation_v16_asr_attention_position_2x2_thvl_external_confirmation','status':'PREREGISTERED_BEFORE_FORWARD_NO_LABEL_ACCESS','corpus':'thvl','split':'self-sealed validation opaque cohort','input':'timestamped ASR text only; labels/GT unopened','public_opaque_manifest':str(PUBLIC.resolve()),'public_opaque_manifest_sha256':sha256(PUBLIC),'reconciled_media_manifest':str(media.resolve()),'reconciled_media_manifest_sha256':sha256(media),'reconciliation_correction_sha256':sha256(correction),'asr_source':str(ASR.resolve()),'asr_source_sha256':sha256(ASR),'asr_provenance_sha256':sha256(ROOT/'results/steward_private/thvl_bench/val32_asr_provenance.json'),'video_ids':ids,'video_ids_sha256':hashlib.sha256(''.join(v+'\n' for v in ids).encode()).hexdigest(),'n_videos':32,'source_present_videos':len(rec),'missing_video_ids':missing,'missing_policy':'formal run requires none; fail closed','timestamp_sanitation':'finite numeric start/end and end>start; invalid chunks masked before forward','invalid_chunks':invalid,'n_invalid_chunks':len(invalid),'sanitized_chunks':sanitized,'n_valid_chunks':sum(map(len,sanitized.values())),'arms':ARMS,'position_attention_factorial':{'attention':['masked branch isolation','full causal'],'position':['branch reset','continuous']},'token_packing':'per-video exact longest-common-prefix of complete sequential prompt token IDs; reconstruction asserted','sequential_reference_subset':subset,'same_tokens_prompt_chunks_all_arms':True,'prompt_language':'generic frozen English','prompt_implementation':str(PROMPT.resolve()),'prompt_implementation_sha256':sha256(PROMPT),'forward_implementation':str(FORWARD.resolve()),'forward_implementation_sha256':sha256(FORWARD),'model':'Qwen/Qwen3-VL-8B-Instruct','model_revision':MODEL_REV,'model_snapshot_path':str(snap.resolve()),'model_snapshot_identity':'HuggingFace snapshot commit directory; immutable model blobs addressed by repository revision','gt_or_labels_opened':False,'selection_or_evaluation_permitted_here':False,'raw_freeze':'temporary file then atomic rename; manifest written after raw SHA256'}
    atomic_json(OUT/'preregistered_config.json',cfg)
    print(json.dumps({'out':str(OUT),'config_sha256':sha256(OUT/'preregistered_config.json'),'videos':32,'present':len(rec),'missing':missing,'valid_chunks':cfg['n_valid_chunks'],'invalid_chunks':len(invalid),'arms':ARMS},indent=2))
if __name__=='__main__':main()
