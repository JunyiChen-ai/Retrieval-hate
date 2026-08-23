from __future__ import annotations
def oracle_record(video_id,score,actions):
    return {"video_id":video_id,"oracle_score":float(score),"oracle_actions":list(actions),"namespace":"oracle"}
def assert_deployable_record(record):
    forbidden=[k for k in record if k.startswith("oracle") or any(x in k.lower() for x in ("span","claude","required_modalities"))]
    if forbidden:raise RuntimeError("HALT_ORACLE_NAMESPACE:"+",".join(forbidden))
    return True
