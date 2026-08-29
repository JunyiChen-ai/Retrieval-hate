#!/usr/bin/env python3
"""Decompose a V4 correction into video-prior and within-video residual effects."""
import argparse,json,os,sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path[:0]=[os.path.dirname(HERE),os.path.join(os.path.dirname(os.path.dirname(HERE)),"duplex")]
from hate_common import data as hdata
from eval_baseline_scores import evaluate_scores
def main():
 p=argparse.ArgumentParser(); p.add_argument("--corpus",required=True); p.add_argument("--scores",required=True); p.add_argument("--out",required=True); a=p.parse_args(); records=hdata.load_scores_jsonl(a.scores); gt=hdata.gt_arrays(a.corpus,"test"); dynamic={v:np.asarray(records[v]["score_relation_v4"],float) for v in gt}; static={v:np.asarray(records[v]["score_static_fusion"],float) for v in gt}
 variants={"static":static,"full":dynamic,"dynamic_mean_static_residual":{v:dynamic[v].mean()+static[v]-static[v].mean() for v in gt},"static_mean_dynamic_residual":{v:static[v].mean()+dynamic[v]-dynamic[v].mean() for v in gt}}; result={}
 for name,value in variants.items():
  metric=evaluate_scores(value,gt); result[name]={"frame_ap":metric["pr_auc"],"frame_roc":metric["roc_auc"]}
 delta=[dynamic[v]-static[v] for v in gt]; result["correction_statistics"]={"mean_video_mean_shift":float(np.mean([x.mean() for x in delta])),"sd_video_mean_shift":float(np.std([x.mean() for x in delta])),"mean_within_video_delta_sd":float(np.mean([x.std() for x in delta])),"mean_absolute_delta":float(np.mean([np.abs(x).mean() for x in delta]))}; payload={"corpus":a.corpus,"scores":os.path.abspath(a.scores),"diagnostic_only_test_not_used_for_selection":True,"results":result}; json.dump(payload,open(a.out+".tmp","w"),indent=2); os.replace(a.out+".tmp",a.out); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
