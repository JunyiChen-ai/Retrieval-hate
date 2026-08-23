"""C6 timing entry point for the four separately charged per-decision overheads."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .common import atomic_json,atomic_write,canonical_bytes,sha256_file
from .costs import benchmark_action,hardware_software_lock
from .models import ActionTokenizer,UtilityPolicy
from .protocol import exact_knapsack

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    torch.manual_seed(20260816);device="cuda"
    tokenizer=ActionTokenizer().eval().to(device);policy=UtilityPolicy(256+256+4).eval().to(device)
    ocr=torch.zeros((1,1024),device=device);kind=torch.zeros(1,dtype=torch.long,device=device)
    window=torch.zeros(1,dtype=torch.long,device=device);policy_x=torch.zeros((60,516),device=device)
    bank=torch.zeros((44640,256),device=device);query=torch.zeros(256,device=device)
    scores=[float(i%13)/13 for i in range(60)];ticks=[1+(i%17) for i in range(60)]
    def encoding():
        with torch.inference_mode():return tokenizer(ocr,kind,window)
    def policy_score():
        with torch.inference_mode():return policy(policy_x)
    def retrieval():
        with torch.inference_mode():return torch.topk(bank@query,k=12).indices
    def solver():return exact_knapsack(scores,ticks,180)
    rows=[]
    for name,fn,use_cuda in (("policy",policy_score,True),("encoding",encoding,True),
                             ("retrieval",retrieval,True),("solver",solver,False)):
        for _ in range(100):fn()
        rows.append(benchmark_action("overhead:"+name,fn,5,use_cuda))
    atomic_write(a.out,b"".join(canonical_bytes(x) for x in rows))
    atomic_json(a.out.with_suffix(".meta.json"),{"schema":"cvoi-overhead-cost-run/1","warmups_per_operation":100,
      "repetitions":5,"binding_repetitions":[2,3,4,5],"shapes":{"policy_candidates":60,"policy_input":516,
      "encoding_input":1024,"retrieval_bank":[44640,256],"retrieval_k":12,"solver_actions":60,"solver_budget_ticks":180},
      "cost_sha256":sha256_file(a.out),"hardware_software_lock":hardware_software_lock(),
      "candidate_metric_computed":False,"test_contact_count":0})
if __name__=="__main__":main()
