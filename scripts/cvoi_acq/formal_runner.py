from __future__ import annotations

import argparse,hashlib,json
from pathlib import Path

from .common import ContactLedger,atomic_json,canonical_bytes,sha256_file
from .fold_iterator import iter_registered_folds
from .lock import assert_metric_locked

def seed_for(split_seed,outer_fold,inner_fold,refit_id,arm,config_id):
    key=(int(split_seed),int(outer_fold),int(inner_fold),int(refit_id),str(arm),str(config_id))
    return int.from_bytes(hashlib.sha256(canonical_bytes(key)).digest()[:4],"big")

def schedule(outer_path,inner_path):
    ledger=ContactLedger();ledger.register(outer_path,"registered_outer_folds");ledger.register(inner_path,"registered_inner_folds")
    outer=json.loads(outer_path.read_text());inner=json.loads(inner_path.read_text());folds=list(iter_registered_folds(outer,inner))
    if len(folds)!=15:raise RuntimeError("HALT_EXPECTED_15_OUTER")
    runs=[]
    for f in folds:
        registered_n=int(inner[str(f["split_seed"])][str(f["outer_fold"])]["n_folds"])
        if len(f["inner"])!=registered_n:raise RuntimeError("HALT_REGISTERED_INNER_COUNT")
        for refit_id in range(3):
            run_id="s%d-o%d-r%d"%(f["split_seed"],f["outer_fold"],refit_id)
            seeds=[]
            for arm in tuple("B%d"%i for i in range(2,13)):
                for inner_row in f["inner"]:seeds.append({"arm":arm,"inner_fold":inner_row["inner_fold"],"seed":seed_for(f["split_seed"],f["outer_fold"],inner_row["inner_fold"],refit_id,arm,"GRID")})
                seeds.append({"arm":arm,"inner_fold":-1,"seed":seed_for(f["split_seed"],f["outer_fold"],-1,refit_id,arm,"SELECTED")})
            runs.append({"run_id":run_id,"split_seed":f["split_seed"],"outer_fold":f["outer_fold"],"refit_id":refit_id,
              "query_groups":f["query_groups"],"train_groups":f["train_groups"],"inner_count":registered_n,"rng":seeds})
    if len(runs)!=45 or len({r["run_id"] for r in runs})!=45:raise RuntimeError("HALT_EXPECTED_45_RUNS")
    return {"schema":"cvoi-formal-schedule/1","metric_locked":True,"n_outer":15,"n_refits":3,"n_runs":45,
      "outer_sha256":sha256_file(outer_path),"inner_sha256":sha256_file(inner_path),"runs":runs,"contact":ledger.snapshot()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--outer",type=Path,required=True);ap.add_argument("--inner",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--premetric-schedule",action="store_true");ap.add_argument("--completeness",type=Path);ap.add_argument("--frozen-config",type=Path);a=ap.parse_args()
    plan=schedule(a.outer,a.inner)
    if a.premetric_schedule:atomic_json(a.out,plan);return
    if a.completeness is None or a.frozen_config is None:raise RuntimeError("HALT_FORMAL_LOCK_ARGS")
    assert_metric_locked(a.completeness,a.frozen_config)
    raise RuntimeError("HALT_EXECUTION_BODY_NOT_FROZEN")

if __name__=="__main__":main()
