#!/usr/bin/env python3
"""Join frozen V24 validation evidence with *video-level only* steward labels.

This is deliberately separate from the immutable train join producer.  The label
manifest is an exact allowlist and cannot carry intervals, frames or timestamps.
"""
import argparse, hashlib, json, math
from pathlib import Path

from evidence_producer import sha

EVIDENCE_PRODUCER = "6290c3d8c6f7faf1b12e01c21778f6c1e2fc3b4412b1e28e29034c075c5365bd"
LABEL_KEYS = {"schema_version", "corpus", "split", "label_semantics", "records"}
LABEL_ROW_KEYS = {"video_id", "any_target_label"}
ID_KEYS = {"schema_version", "corpus", "split", "ids", "v23_global_source_sha256",
           "evidence_producer_sha256", "join_producer_sha256",
           "evidence_manifest_sha256", "evidence_config_sha256",
           "labels_manifest_sha256", "bags_sha256"}
EM_KEYS={"status","n_videos","n_windows","records","config_sha256","frozen_inputs_sha256"}
REC_KEYS={"video_id","duration","media_sha256","asr_record_sha256","input_item_sha256","config_sha256","producer_source_sha256","model_revision","prompt_spec_sha256","windows","global_causal_score","labels_read"}
WIN_KEYS={"window_index","start","end","center","frame_time","frame_index","fps","frame_fallback_offset","speech_sha256","prompt_sha256","text_isolated_score","multimodal_isolated_score"}
INPUT_KEYS={"video_id","media_path","media_sha256","duration","asr_record_sha256","windows"}
INPUT_WIN_KEYS={"window_index","start","end","center","speech","speech_sha256"}

def _hash64(x): return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)

def load_video_labels(path, corpus="thvl"):
    m = json.load(open(path))
    if set(m) != LABEL_KEYS or m["schema_version"] != "v24_video_labels_v1":
        raise RuntimeError("invalid exact video-label schema")
    if m["corpus"] != corpus or m["split"] != "val" or m["label_semantics"] != "any_target_video_level":
        raise RuntimeError("wrong validation label identity/semantics")
    out = {}
    if not isinstance(m["records"], list) or not m["records"]: raise RuntimeError("empty labels")
    for r in m["records"]:
        if not isinstance(r, dict) or set(r) != LABEL_ROW_KEYS: raise RuntimeError("temporal/extra label fields forbidden")
        if not isinstance(r["video_id"], str) or r["video_id"] in out or r["any_target_label"] not in (0, 1):
            raise RuntimeError("invalid/duplicate video label")
        out[r["video_id"]] = int(r["any_target_label"])
    return out

def load_evidence(ed, corpus="thvl"):
    ed = Path(ed); ep = ed / "evidence_manifest.json"; cp = ed / "preregistered_config.json"
    em, cfg = json.load(open(ep)), json.load(open(cp))
    if set(em)!=EM_KEYS or em.get("status") != "COMPLETE_LABEL_BLIND" or em.get("config_sha256") != sha(cp):
        raise RuntimeError("evidence not frozen/hash-valid")
    if cfg.get("split") != "val" or cfg.get("labels_read") is not False: raise RuntimeError("wrong/nonblind evidence split")
    if cfg.get("producer_sha256") != EVIDENCE_PRODUCER or cfg.get("local_forward_sha256") != EVIDENCE_PRODUCER:
        raise RuntimeError("swapped evidence producer")
    if not isinstance(em["records"],dict) or not em["records"] or any(not _hash64(x) for x in em["records"].values()): raise RuntimeError("invalid manifest records")
    ip=ed/"frozen_inputs.jsonl"
    if em["frozen_inputs_sha256"]!=sha(ip) or cfg.get("frozen_inputs_sha256")!=sha(ip): raise RuntimeError("frozen inputs hash mismatch")
    inputs={}
    with open(ip) as f:
      for line in f:
        x=json.loads(line)
        if set(x)!=INPUT_KEYS or not isinstance(x["windows"],list) or not x["windows"]: raise RuntimeError("invalid frozen input schema")
        if x["video_id"] in inputs or any(set(w)!=INPUT_WIN_KEYS for w in x["windows"]): raise RuntimeError("invalid frozen input identity/window schema")
        inputs[x["video_id"]]=x
    disk={p.stem:p for p in (ed/"records").glob("*.json")}
    if set(disk)!=set(em["records"]) or set(inputs)!=set(em["records"]): raise RuntimeError("disk/manifest/frozen ID mismatch")
    records={}; total=0
    for vid in sorted(em["records"]):
        p=disk[vid]
        if sha(p)!=em["records"][vid]: raise RuntimeError("record file hash mismatch")
        r=json.load(open(p)); item=inputs[vid]
        if set(r)!=REC_KEYS or r.get("video_id")!=vid or r.get("labels_read") is not False: raise RuntimeError("record top schema/identity violation")
        item_hash=hashlib.sha256(json.dumps(item,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        if r.get("input_item_sha256")!=item_hash or r.get("config_sha256")!=em["config_sha256"]: raise RuntimeError("record input/config mismatch")
        if r.get("producer_source_sha256")!=EVIDENCE_PRODUCER or r.get("model_revision")!=cfg.get("model_revision") or r.get("prompt_spec_sha256")!=cfg.get("prompt_spec_sha256"): raise RuntimeError("record producer/model/prompt mismatch")
        if r.get("media_sha256")!=item["media_sha256"] or r.get("asr_record_sha256")!=item["asr_record_sha256"] or r.get("duration")!=item["duration"]: raise RuntimeError("record frozen identity mismatch")
        if not isinstance(r.get("global_causal_score"),(int,float)) or not math.isfinite(float(r["global_causal_score"])): raise RuntimeError("nonfinite global score")
        ws=r.get("windows")
        if not isinstance(ws,list) or len(ws)!=len(item["windows"]): raise RuntimeError("window count mismatch")
        for i,(w,iw) in enumerate(zip(ws,item["windows"])):
            if set(w)!=WIN_KEYS or w["window_index"]!=i or iw["window_index"]!=i: raise RuntimeError("window schema/order violation")
            if any(w[k]!=iw[k] for k in ("start","end","center","speech_sha256")): raise RuntimeError("window differs from frozen input")
            if i and w["start"]!=ws[i-1]["end"]: raise RuntimeError("noncontiguous windows")
            if not (0<=w["start"]<w["end"]<=r["duration"]+1e-9): raise RuntimeError("invalid window bounds")
            if set(w["prompt_sha256"])!={"text","multimodal"} or any(not _hash64(x) for x in w["prompt_sha256"].values()): raise RuntimeError("invalid prompt hashes")
            if any(not isinstance(w[k],(int,float)) or not math.isfinite(float(w[k])) for k in ("text_isolated_score","multimodal_isolated_score")): raise RuntimeError("nonfinite local score")
        total+=len(ws);records[vid]=r
    if len(records)!=em["n_videos"] or total!=em["n_windows"] or total!=cfg.get("n_windows") or len(records)!=cfg.get("n_videos"): raise RuntimeError("manifest/config aggregate count mismatch")
    return em, cfg, records, ep, cp

def join(evidence_dir, labels_path, out_dir, corpus="thvl"):
    em, cfg, records, ep, cp = load_evidence(evidence_dir, corpus); labels = load_video_labels(labels_path, corpus)
    ids = sorted(records)
    if set(labels) != set(ids): raise RuntimeError("label/evidence coverage mismatch")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=False)
    gp, bp = out / "v23_global_values.jsonl", out / "bags.jsonl"
    gp.write_text("".join(json.dumps({"video_id":v,"global_causal_score":records[v]["global_causal_score"]},sort_keys=True)+"\n" for v in ids))
    gh = sha(gp)
    with open(bp, "w") as f:
        for v in ids:
            r=records[v]; text=[x["text_isolated_score"] for x in r["windows"]]; mm=[x["multimodal_isolated_score"] for x in r["windows"]]
            row={"corpus":corpus,"split":"val","video_id":v,"video_label":labels[v],"global_causal_score":r["global_causal_score"],"families":{"text":[text],"multimodal":[mm]},"source_hashes":{"text_scores_sha256":hashlib.sha256(json.dumps(text,separators=(",",":")).encode()).hexdigest(),"multimodal_scores_sha256":hashlib.sha256(json.dumps(mm,separators=(",",":")).encode()).hexdigest(),"v23_global_source_sha256":gh}}
            f.write(json.dumps(row,sort_keys=True)+"\n")
    prod=sha(Path(__file__).resolve()); idm={"schema_version":"v24_join_manifest_v2","corpus":corpus,"split":"val","ids":ids,"v23_global_source_sha256":gh,"evidence_producer_sha256":cfg["producer_sha256"],"join_producer_sha256":prod,"evidence_manifest_sha256":sha(ep),"evidence_config_sha256":sha(cp),"labels_manifest_sha256":sha(labels_path),"bags_sha256":sha(bp)}
    ip=out/"val_id_manifest.json"; ip.write_text(json.dumps(idm,indent=2,sort_keys=True)+"\n")
    jm={**idm,"id_manifest_sha256":sha(ip),"global_sha256":gh}; (out/"join_manifest.json").write_text(json.dumps(jm,indent=2,sort_keys=True)+"\n")
    return idm

def load_val_manifest(path, corpus="thvl"):
    m=json.load(open(path))
    if set(m)!=ID_KEYS or m.get("schema_version")!="v24_join_manifest_v2" or m.get("corpus")!=corpus or m.get("split")!="val": raise RuntimeError("invalid exact v2 val manifest")
    if not m["ids"] or len(m["ids"])!=len(set(m["ids"])): raise RuntimeError("invalid val IDs")
    for k in ID_KEYS-{"schema_version","corpus","split","ids"}:
        if not _hash64(m[k]): raise RuntimeError("invalid provenance hash")
    return m

def main():
    p=argparse.ArgumentParser(); p.add_argument("--evidence-dir",required=True);p.add_argument("--video-label-manifest",required=True);p.add_argument("--out-dir",required=True);p.add_argument("--corpus",default="thvl");a=p.parse_args();join(a.evidence_dir,a.video_label_manifest,a.out_dir,a.corpus)
if __name__=="__main__": main()
