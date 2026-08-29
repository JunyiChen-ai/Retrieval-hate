#!/usr/bin/env python3
"""Freeze the V21 multimodal-local-judge validation pilot before inference."""
import argparse, hashlib, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(ROOT / "scripts/reproduction_baselines")]
from relation_v2.protocol import frozen_splits
from relation_v4.io import sha256

ASR = {c: ROOT / f"results/reproduction/asr/{c}_all/timestamped_chunks.jsonl"
       for c in ("hatemm", "hateclipseg")}
OCR = {
    "hatemm": Path("/home/jehc223/Retrieval-hate/data/OCR/HateMM/ocr_windows_K30.jsonl"),
    "hateclipseg": Path("/home/jehc223/Retrieval-hate/data/OCR/HateClipSeg/ocr_windows_K30.jsonl"),
}
VIDEO = {"hatemm": Path("/home/jehc223/data/HateMM/video"),
         "hateclipseg": Path("/home/jehc223/data/HateClipSeg/videos")}
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
POLICY = ("Judge only the synchronized local video moment. Answer Yes if that moment "
          "contains hateful or severely offensive content; otherwise answer No. "
          "Use the image, visible text, and speech jointly. Do not infer from other moments.")
QUESTION = "Does this synchronized local moment contain hateful or severely offensive content? Answer Yes or No."

def digest_ids(ids):
    return hashlib.sha256("".join(x + "\n" for x in ids).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--corpus",choices=sorted(ASR),required=True)
    p.add_argument("--out-dir",required=True); p.add_argument("--max-videos",type=int,default=12)
    a=p.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=False)
    rows={r["video_id"]:r for r in map(json.loads,open(ASR[a.corpus]))}
    val=set(frozen_splits(a.corpus)["val"])
    def valid_chunks(v):
        return [x for x in rows.get(v,{}).get("chunks",[]) if
                isinstance(x.get('start'),(int,float)) and isinstance(x.get('end'),(int,float)) and
                math.isfinite(float(x['start'])) and math.isfinite(float(x['end'])) and float(x['end'])>float(x['start'])]
    eligible=sorted(v for v in val if valid_chunks(v))
    # Label-free, deterministic cohort: highest ASR branch count, then ID.
    cohort=sorted(eligible,key=lambda v:(-len(valid_chunks(v)),v))[:a.max_videos]
    cfg={"method":"relation_v21_multimodal_isolated_local_judge","status":"PREREGISTERED_BEFORE_FORWARD_OR_TEMPORAL_GT",
         "corpus":a.corpus,"split":"val","test_access":False,"model":MODEL,
         "cohort":cohort,"cohort_sha256":digest_ids(cohort),"selection_rule":"val videos with nonempty ASR; descending ASR chunk count, video-id tie break",
         "asr":{"path":str(ASR[a.corpus].resolve()),"sha256":sha256(ASR[a.corpus]),"max_chars":3000,"validity_rule":"finite numeric start/end and end>start; invalid timestamp rows excluded without labels"},
         "ocr":{"path":str(OCR[a.corpus].resolve()),"sha256":sha256(OCR[a.corpus]),"rule":"nearest t_mid to ASR span center; ties earliest window_k; texts confidence>=0.5; reading order as stored; max 1200 chars"},
         "frame":{"video_root":str(VIDEO[a.corpus].resolve()),"rule":"one RGB frame at ASR span center, clamped to decodable timeline; no label-dependent sampling"},
         "policy":POLICY,"question":QUESTION,"arms":["asr_only","frame_ocr_asr"],
         "score":"logsumexp next-token Yes IDs minus logsumexp next-token No IDs",
         "packed_status":"DEFERRED: visual token expansion and multimodal mRoPE require exact sequential equivalence proof",
         "evaluation_after_raw_freeze":["within-video macro ROC/AP","pooled ROC/AP","within-video time-shuffle B=200","paired video bootstrap B=2000"],
         "fallback":{"base":"ASR-only/V8 identity","candidate_lambda_includes_zero":True},
         "forbidden":["test media","test labels","six-category evidence prompts","API calls"]}
    (out/"preregistered_config.json").write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"config":str((out/'preregistered_config.json').resolve()),"n_videos":len(cohort),"n_chunks":sum(len(valid_chunks(v)) for v in cohort)},indent=2))
if __name__=="__main__": main()
