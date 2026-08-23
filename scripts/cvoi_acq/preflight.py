from __future__ import annotations

import argparse, json, os, shutil
from pathlib import Path
from .common import ROOT, atomic_json
from .lock import load_ledger

def inspect(completeness: Path, projected_bytes: int) -> dict:
    disk=shutil.disk_usage(ROOT); gpu=[]
    try:
        import torch
        gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception: pass
    state=load_ledger(completeness)
    pending=[k for k,v in state["gates"].items() if v["status"]!="PASS"]
    return {"schema":"cvoi-resource-preflight/1","cpu_count":os.cpu_count(),"gpu_count":len(gpu),
            "gpu_models":gpu,"disk_free_bytes":disk.free,"projected_new_asset_bytes":projected_bytes,
            "disk_requirement_bytes":2*projected_bytes+20*(1<<30),
            "disk_ok":disk.free >= 2*projected_bytes+20*(1<<30),"pending_gates":pending,
            "metric_locked":bool(pending),"limits":{"cpu":16,"memory_gb":128,"gpu":2,"slurm_time_forbidden":True}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--completeness",type=Path,required=True);ap.add_argument("--projected-bytes",type=int,default=8*(1<<30));ap.add_argument("--out",type=Path,required=True);ap.add_argument("--enforce",action="store_true");a=ap.parse_args()
    result=inspect(a.completeness,a.projected_bytes);atomic_json(a.out,result);print(json.dumps(result,sort_keys=True))
    if a.enforce and (not result["disk_ok"] or result["gpu_count"]<1 or result["cpu_count"]<1):raise SystemExit(41)
if __name__=="__main__":main()
