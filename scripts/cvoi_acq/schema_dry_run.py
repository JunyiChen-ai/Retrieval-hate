from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from .artifacts import validate_full_prediction_rows,validate_targets,validate_trace_join
from .common import atomic_json,canonical_bytes,sha256_file

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--template",type=Path,required=True);ap.add_argument("--schedule",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    src=json.loads(a.template.read_text());sched=json.loads(a.schedule.read_text());payload=src["payload_sha256"];pred=[];traces=[]
    for r in src["rows"]:
        run_id="synthetic-schema-s%d-o%d-r0"%(src["fold_records"][r["fold"]]["split_seed"],r["fold"]);dec=r["decisions"];th=hashlib.sha256(canonical_bytes(dec)).hexdigest();mh=r["model_hashes"]["B10" if r["arm"]=="B12" else r["arm"]]
        pred.append({"schema":"cvoi-prediction/1","run_id":run_id,"dataset":"SYNTHETIC","split_role":"train_oof","video_id":r["video_id"],"group_id":r["group"],
          "outer_split_seed":src["fold_records"][r["fold"]]["split_seed"],"outer_fold":r["fold"],"refit_seed":0,"arm_id":r["arm"],"budget_fraction":.5,
          "score":r["score"],"prediction":r["prediction"],"threshold":r["threshold"],"threshold_source":"pooled_inner_oof","estimated_budget_ms":4,
          "realized_cost_ms":r["cost"],"action_trace_sha256":th,"config_id":"template-selected","epoch":src["fold_records"][r["fold"]]["B10" if r["arm"]!="B9" else "B9"]["selected"]["median_refit_epoch"],
          "model_sha256":mh,"payload_sha256":payload,"draw_id":None})
        traces.append({"schema":"cvoi-acquisition-trace/1","run_id":run_id,"outer_split_seed":src["fold_records"][r["fold"]]["split_seed"],"refit_seed":0,
          "video_id":r["video_id"],"arm_id":r["arm"],"budget_fraction":.5,"draw_id":None,"ordered_actions":r["actions"],"estimated_cost_ms":4,
          "realized_cost_ms":r["cost"],"decision_records":dec,"trace_sha256":th})
    validate_full_prediction_rows(pred,len(pred));validate_trace_join(pred,traces)
    target_counts=[]
    for fr in src["fold_records"]:
        rows=[];counts={}
        for r in fr["targets"]:counts[r["video_id"]]=counts.get(r["video_id"],0)+1
        for r in fr["targets"]:rows.append({"video_id":r["video_id"],"eval_group":r["eval_group"],"fit_groups":r["fit_groups"],"state":r["state"],"action_id":r["action"],
          "action_features":r["features"],"before_probability":r["before"],"after_probability":r["after"],"utility":r["utility"],"estimated_cost":1,
          "remaining_budget":4,"weight":1/counts[r["video_id"]],"generator_provenance":["registered_inner"],"terminal":False})
        target_counts.append(validate_targets(rows))
    if sched["n_runs"]!=45 or any(r["inner_count"]!=4 for r in sched["runs"]):raise RuntimeError("HALT_FORMAL_SCHEDULE_SCHEMA")
    atomic_json(a.out,{"schema":"cvoi-schema-dry-run/1","synthetic_only":True,"candidate_metrics_imported":False,"template_sha256":sha256_file(a.template),
      "schedule_sha256":sha256_file(a.schedule),"prediction_rows":len(pred),"trace_rows":len(traces),"target_rows_by_fold":target_counts,
      "prediction_trace_join":True,"target_schema_and_weights":True,"formal_runs":45,"formal_inner_count":4,"test_contact_count":0})
if __name__=="__main__":main()
