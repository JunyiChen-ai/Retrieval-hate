#!/usr/bin/env python3
"""Freeze the MHC localization cohort and sanitized ASR timeline before Qwen."""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE.parent),str(ROOT/'scripts/duplex'),str(ROOT/'src/duplex'),str(ROOT/'src/our_method')]
from hate_common import data as hdata
from relation_v4.io import sha256
from relation_v8.run import atomic_json
from score_duplex_probe import SYSTEM_MESSAGE,YOUTUBE_RULES
from isolated_chunk_diag import RULES_LEAD_IN,QUESTION,user_text

ARMS=['masked_branch_reset','masked_continuous','causal_branch_reset','causal_continuous']
ZH_TRANSLATION={
 'system':'你是一名内容审核分析员。请严格依据可观察到的证据作答。',
 'policy_lead':'根据该平台的仇恨言论政策，禁止以下类型的内容：',
 'policy_rules':'1. 鼓励针对受保护身份群体的暴力\n2. 煽动对受保护身份群体的仇恨\n3. 通过将群体比作非人实体来贬低其人性\n4. 赞扬或美化针对受保护群体的暴力\n5. 使用煽动仇恨的侮辱性称谓和刻板印象（包括通过幽默、反讽或暗语）\n6. 声称某些群体因受保护身份而低人一等\n7. 宣扬仇恨至上主义\n8. 对受保护身份群体提出阴谋论式指控\n9. 否认或淡化重大暴力事件',
 'wrapper':'{policy_block}\n\n视频中的转录片段：\n{chunk}\n\n{question}',
 'question':'该片段是否违反上述规则？\n\n请回答“Yes”或“No”。'}
def prompt_translation_manifest():
 src={'system':SYSTEM_MESSAGE,'policy_lead':RULES_LEAD_IN,'policy_rules':YOUTUBE_RULES,'wrapper':'{policy_block}\n\nTranscript excerpt from a video:\n{chunk}\n\n{question}','question':QUESTION}
 return {k:{'english':src[k],'chinese':ZH_TRANSLATION[k],'english_sha256':hashlib.sha256(src[k].encode()).hexdigest(),'chinese_sha256':hashlib.sha256(ZH_TRANSLATION[k].encode()).hexdigest()} for k in src}

def sanitize(chunks, duration):
    out=[]; reasons={'none_or_nonfinite':0,'nonpositive_before_clip':0,'invalid_after_clip':0}
    for i,ch in enumerate(chunks or []):
        s,e=ch.get('start'),ch.get('end')
        if s is None or e is None:
            reasons['none_or_nonfinite']+=1; continue
        try:s=float(s);e=float(e)
        except (TypeError,ValueError):reasons['none_or_nonfinite']+=1;continue
        if not math.isfinite(s) or not math.isfinite(e):reasons['none_or_nonfinite']+=1;continue
        if e<=s:reasons['nonpositive_before_clip']+=1;continue
        s=max(0.,min(float(duration),s));e=max(0.,min(float(duration),e))
        if e<=s:reasons['invalid_after_clip']+=1;continue
        out.append({'source_index':i,'start':s,'end':e,'text':ch.get('text')})
    return out,reasons

def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',choices=['mhclip_en','mhclip_zh'],required=True);p.add_argument('--prompt-language',choices=['english','chinese'],default='english')
    p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--out-dir',required=True);a=p.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
    src=ROOT/f'results/reproduction/asr/{a.corpus}_all/timestamped_chunks.jsonl'
    records={}
    for line_no,r in enumerate(map(json.loads,open(src)),1):
        v=r.get('video_id')
        if v in records:raise RuntimeError(f'duplicate video_id at ASR line {line_no}: {v}')
        records[v]=r
    split_ids=hdata.load_split(a.corpus,a.split);gt_path=Path(hdata.GT_ROOT)/f'{a.corpus}_{a.split}.npz'
    # IDs only: no temporal GT array is materialized and no GT length is read.
    with np.load(gt_path) as z: cohort=list(z.files)
    if not set(cohort).issubset(split_ids):raise RuntimeError('GT cohort is not within frozen split')
    missing_source=sorted(set(cohort)-set(records))
    if missing_source:raise RuntimeError(f'GT cohort missing ASR sources: {missing_source[:10]}')
    clean={};drop={};missing=[]
    for v in cohort:
        duration=records[v].get('wav_duration')
        if duration is None or not math.isfinite(float(duration)) or float(duration)<=0:raise RuntimeError(f'invalid ASR wav_duration: {v} {duration}')
        q,rr=sanitize(records[v].get('chunks'),float(duration))
        clean[v]=q;drop[v]=rr
    cfg={'method':'relation_v18_mhc_qwen_2x2','status':'PREREGISTERED_FROZEN_GT_COHORT_BEFORE_FORWARD',
         'test_informed_design_from_v16':True,'corpus':a.corpus,'split':a.split,
         'policy_language_invariant':'exact same English prompt/Qwen/arms as HM/HCS; no language-specific policy',
         'input':'ASR chunk text only','asr_source':str(src.resolve()),'asr_source_sha256':sha256(src),
         'gt_cohort_source':str(gt_path.resolve()),'gt_cohort_ids_only_opened':True,
         'video_ids':cohort,'video_ids_sha256':hashlib.sha256(''.join(v+'\n' for v in cohort).encode()).hexdigest(),
         'duration_source':'ASR wav_duration only; temporal GT arrays not loaded','sanitized_chunks':clean,'sanitation':{'order':'drop None/nonfinite/end<=start; clip to [0,ASR wav_duration]; drop invalid after clip','dropped_by_video':drop,'missing_asr_videos':missing,'uncovered_frames':'nearest-center extrapolation, exactly matching HM/HCS formal','zero_valid_video':'exact identity; local and global missing','duplicate_video_id':'hard fail','GT_cohort_missing_ASR_source':'hard fail'},
         'n_videos':len(cohort),'n_valid_chunks':sum(map(len,clean.values())),'n_zero_valid':sum(not x for x in clean.values()),
         'arms':ARMS,'same_tokens_prompt_chunks_all_arms':True,'sequential_reference_subset':[],
         'prompt_implementation':str((ROOT/'scripts/duplex/masked_parallel_isolation_pilot.py').resolve()),
         'prompt_implementation_sha256':sha256(ROOT/'scripts/duplex/masked_parallel_isolation_pilot.py'),
         'forward_implementation':str((ROOT/'scripts/reproduction_baselines/relation_v16/forward.py').resolve()),
         'forward_implementation_sha256':sha256(ROOT/'scripts/reproduction_baselines/relation_v16/forward.py'),
         'raw_frozen_before_temporal_gt_values':True}
    if a.prompt_language=='chinese':
        if a.corpus!='mhclip_zh':raise RuntimeError('Chinese adaptation pilot is scoped to mhclip_zh')
        cfg.update({'method':'relation_v20_language_native_judge','TEST_INFORMED_LANGUAGE_ADAPTATION':True,'prompt_language':'chinese','translation_frozen_before_fresh_raw':True,'translation_no_dataset_keywords':True,'answer_tokens_unchanged':'Yes/No using identical token-margin readout','prompt_translation':prompt_translation_manifest(),'chinese_prompt_parts':ZH_TRANSLATION})
    else:cfg['prompt_language']='english'
    atomic_json(out/'preregistered_config.json',cfg);print(json.dumps({'sha256':sha256(out/'preregistered_config.json'),'videos':len(cohort),'chunks':cfg['n_valid_chunks'],'zero':cfg['n_zero_valid']},indent=2))
if __name__=='__main__':main()
