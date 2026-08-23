from __future__ import annotations
import hashlib,json
from pathlib import Path
from .common import atomic_write,canonical_bytes
REQUIRED=("manifest.json","asset_registry.json","groups/sources.jsonl","groups/edges.jsonl","groups/components.json",
 "groups/folds.json","actions/ocr_actions.jsonl","actions/dense_frames.f32","actions/dense_sidecar.jsonl",
 "actions/cost_actions.jsonl","actions/cost_summary.json","selection/inner_selection.jsonl","targets",
 "predictions/train_oof.jsonl","predictions/val_confirmation.jsonl","traces/acquisitions.jsonl",
 "metrics/metrics.json","metrics/bootstrap.npz","fixtures/report.json","resources.jsonl","RESULTS.md")
PREDICTION_REQUIRED=("schema","run_id","dataset","split_role","video_id","group_id","outer_split_seed",
 "outer_fold","refit_seed","arm_id","budget_fraction","score","prediction","threshold","threshold_source",
 "estimated_budget_ms","realized_cost_ms","action_trace_sha256","config_id","epoch","model_sha256",
 "payload_sha256","draw_id")
TRACE_REQUIRED=("schema","run_id","outer_split_seed","refit_seed","video_id","arm_id","budget_fraction","draw_id",
 "ordered_actions","estimated_cost_ms","realized_cost_ms","decision_records","trace_sha256")
TARGET_REQUIRED=("video_id","eval_group","fit_groups","state","action_id","action_features","before_probability",
 "after_probability","utility","estimated_cost","remaining_budget","weight","generator_provenance","terminal")
def validate_prediction_rows(rows,expected_keys):
    seen=set()
    for r in rows:
        key=tuple(r[k] for k in expected_keys)
        if key in seen:raise RuntimeError("HALT_DUPLICATE_PREDICTION_ROW")
        seen.add(key)
    return len(seen)
def write_jsonl(path,rows):
    data=b"".join(canonical_bytes(r) for r in rows);atomic_write(path,data);return hashlib.sha256(data).hexdigest()
def assert_run_contract(root):
    missing=[p for p in REQUIRED if not (root/p).exists()]
    if missing:raise RuntimeError("HALT_ARTIFACT_CONTRACT:"+",".join(missing))
def validate_full_prediction_rows(rows,expected_count=None):
    for i,r in enumerate(rows):
        missing=set(PREDICTION_REQUIRED)-set(r)
        if missing:raise RuntimeError("HALT_PREDICTION_SCHEMA:%d:%s"%(i,",".join(sorted(missing))))
        if r["estimated_budget_ms"]<0 or r["realized_cost_ms"]<0:raise RuntimeError("HALT_NEGATIVE_COST")
    n=validate_prediction_rows(rows,("run_id","outer_split_seed","refit_seed","video_id","arm_id","budget_fraction","draw_id"))
    if expected_count is not None and n!=expected_count:raise RuntimeError("HALT_CARDINALITY:%d!=%d"%(n,expected_count))
    return n
def file_record(path):
    p=Path(path);return {"path":str(p.resolve()),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def expected_train_cardinality(n_videos,n_arms,n_budgets,n_split_seeds=3,n_refits=3,b2_draws=20):
    # B2 emits 20 complete executions; every other arm emits one.
    if n_arms<1:raise ValueError("n_arms")
    return int(n_videos*n_budgets*n_split_seeds*n_refits*((n_arms-1)+b2_draws))
def validate_targets(rows):
    seen=set();by_video={}
    for i,r in enumerate(rows):
        missing=set(TARGET_REQUIRED)-set(r)
        if missing:raise RuntimeError("HALT_TARGET_SCHEMA:%d:%s"%(i,",".join(sorted(missing))))
        if r["eval_group"] in set(r["fit_groups"]):raise RuntimeError("HALT_TARGET_GROUP_LEAKAGE")
        key=(r["video_id"],tuple(r["state"]),r["action_id"])
        if key in seen:raise RuntimeError("HALT_TARGET_DUPLICATE")
        seen.add(key);by_video[r["video_id"]]=by_video.get(r["video_id"],0.0)+float(r["weight"])
    if any(abs(x-1)>1e-6 for x in by_video.values()):raise RuntimeError("HALT_TARGET_VIDEO_WEIGHT")
    return len(rows)
def validate_trace_join(predictions,traces):
    by={}
    for i,t in enumerate(traces):
        missing=set(TRACE_REQUIRED)-set(t)
        if missing:raise RuntimeError("HALT_TRACE_SCHEMA:%d:%s"%(i,",".join(sorted(missing))))
        key=(t["run_id"],t["outer_split_seed"],t["refit_seed"],t["video_id"],t["arm_id"],t["budget_fraction"],t["draw_id"])
        if key in by:raise RuntimeError("HALT_TRACE_DUPLICATE")
        digest=hashlib.sha256(canonical_bytes(t["decision_records"])).hexdigest()
        if digest!=t["trace_sha256"]:raise RuntimeError("HALT_TRACE_HASH")
        by[key]=t
    for p in predictions:
        key=(p["run_id"],p["outer_split_seed"],p["refit_seed"],p["video_id"],p["arm_id"],p["budget_fraction"],p["draw_id"])
        if key not in by or p["action_trace_sha256"]!=by[key]["trace_sha256"]:raise RuntimeError("HALT_PREDICTION_TRACE_JOIN")
    if len(by)!=len(predictions):raise RuntimeError("HALT_TRACE_CARDINALITY")
    return len(by)
def validate_run_deep(root,expected_train_rows=None):
    import pyarrow.parquet as pq
    assert_run_contract(root)
    manifest=json.loads((root/"manifest.json").read_text())
    payload={"run_root":manifest["run_root"],"source_sha256":manifest["source_sha256"]}
    if hashlib.sha256(canonical_bytes(payload)).hexdigest()!=manifest["payload_sha256"]:raise RuntimeError("HALT_PAYLOAD_HASH")
    for p,h in manifest["source_sha256"].items():
        q=Path(p)
        if not q.exists() or hashlib.sha256(q.read_bytes()).hexdigest()!=h:raise RuntimeError("HALT_SOURCE_HASH:"+p)
    load=lambda p:[json.loads(x) for x in p.open() if x.strip()]
    pred=load(root/"predictions/train_oof.jsonl");traces=load(root/"traces/acquisitions.jsonl");val=load(root/"predictions/val_confirmation.jsonl")
    validate_full_prediction_rows(pred,expected_train_rows);validate_full_prediction_rows(val)
    validate_trace_join(pred+val,traces)
    targets=pq.read_table(root/"targets/utility_targets.parquet").to_pylist();validate_targets(targets)
    selection=load(root/"selection/inner_selection.jsonl")
    if not selection or not val:raise RuntimeError("HALT_EMPTY_SELECTION_OR_VAL")
    selection_keys={(x["procedure_run_id"],x["outer_fold"],x.get("arm_id")) for x in selection}
    for p in pred:
        if (p["run_id"],p["outer_fold"],p["arm_id"]) not in selection_keys:raise RuntimeError("HALT_PREDICTION_SELECTION_JOIN")
        if p["payload_sha256"]!=manifest["payload_sha256"]:raise RuntimeError("HALT_PREDICTION_PAYLOAD")
    return {"prediction_rows":len(pred),"trace_rows":len(traces),"target_rows":len(targets),"selection_rows":len(selection),"val_rows":len(val)}
