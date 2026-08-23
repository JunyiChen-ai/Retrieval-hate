from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess, shutil
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import ContactLedger, ROOT, atomic_json, atomic_write, canonical_bytes, sha256_file

TRAIN = ROOT / "data/gt/HateMM/train.jsonl"
VIDEO_DIR = ROOT / "data/video/HateMM/All"
PERMS = 128
PERM_SEED = 20260810


def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def minhash128(text: str) -> list[int]:
    grams = {text[i:i + 5] for i in range(max(0, len(text) - 4))}
    if not grams:
        return [0] * PERMS
    out = []
    for j in range(PERMS):
        person = hashlib.sha256(f"{PERM_SEED}:{j}".encode()).digest()[:8]
        out.append(min(int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8,
                                                       person=person).digest(), "big")
                       for g in grams))
    return out


def video_path(vid: str) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".avi"):
        p = VIDEO_DIR / f"{vid}{ext}"
        if p.exists():
            return p.resolve()
    return None


def probe_duration_ms(path: Path) -> int | None:
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", str(path)], capture_output=True,
                           text=True, timeout=60, check=True)
        return round(float(p.stdout.strip()) * 1000)
    except Exception:
        return None


def phash30(path: Path) -> tuple[list[int] | None, str]:
    try:
        from decord import VideoReader,cpu
        from PIL import Image
        from scipy.fft import dctn
        vr=VideoReader(str(path),ctx=cpu(0),num_threads=1);n=len(vr)
        if n <= 0:
            return None, "decode_failed"
        vals = []
        for k in range(30):
            idx = round((k + 0.5) * n / 30 - 0.5)
            frame=vr[max(0,min(n-1,idx))].asnumpy()
            gray=np.asarray(Image.fromarray(frame).convert("L"));h,w=gray.shape
            scale = min(256 / max(w, 1), 256 / max(h, 1))
            resized=np.asarray(Image.fromarray(gray).resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.BILINEAR))
            canvas = np.zeros((256, 256), np.uint8)
            y, x = (256 - resized.shape[0]) // 2, (256 - resized.shape[1]) // 2
            canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            coeff=dctn(canvas.astype(np.float32),type=2,norm="ortho")[:8,:8]
            med=float(np.median(coeff.reshape(-1)[1:]));bits=(coeff>=med).reshape(-1)
            value = sum(int(bit) << q for q, bit in enumerate(bits))
            vals.append(value)
        return vals, "ok"
    except Exception:
        return None, "decode_failed"


def build_sources(out: Path) -> list[dict]:
    ledger = ContactLedger()
    ledger.register(TRAIN, "train_group_source")
    rows = [json.loads(line) for line in TRAIN.open() if line.strip()]
    records = []
    for i, row in enumerate(rows):
        vid = str(row["id"])
        p = video_path(vid)
        if p is not None: ledger.register(p,"train_group_raw_video")
        text = norm_text(str(row.get("text") or ""))
        ph, status = (phash30(p) if p else (None, "missing"))
        rec = {
            "video_id": vid,
            "video_sha256": sha256_file(p) if p else "",
            "duration_ms": probe_duration_ms(p) if p else None,
            "phash30": ph,
            "chromaprint": None,
            "normalized_transcript_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "transcript_minhash128": minhash128(text),
            "creator_namespace": None, "creator_id": None,
            "decode_status": status,
            "source_paths_sha256": hashlib.sha256((str(TRAIN.relative_to(ROOT)) + "\n" +
                                                    (str(p) if p else "missing")).encode()).hexdigest(),
            "label": int(row["label"]),
        }
        records.append(rec)
        if (i + 1) % 50 == 0:
            print(f"[groups] sources {i+1}/{len(rows)}", flush=True)
    payload = b"".join(canonical_bytes(r) for r in sorted(records, key=lambda x: x["video_id"]))
    atomic_write(out / "group_sources.jsonl", payload)
    atomic_json(out / "group_source_contact.json", ledger.snapshot())
    atomic_json(out / "group_capabilities.json",{
        "schema":"cvoi-group-capabilities/1","creator_field_present":False,
        "creator_fallback":"null; no trusted creator field in audited train JSONL",
        "fpcalc_available":shutil.which("fpcalc") is not None,
        "chromaprint_fallback":"null; deterministic missing-feature/no-edge fallback",
        "note":"Chromaprint is null in v1 sources; enabling it requires a pre-metric versioned rebuild."})
    return records


class UF:
    def __init__(self, ids): self.p = {x: x for x in ids}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.p[max(a,b)] = min(a,b)


def estimated_jaccard(a: list[int], b: list[int]) -> float:
    return sum(x == y for x, y in zip(a, b)) / PERMS


def build_components(records: list[dict], out: Path) -> dict[str, list[str]]:
    ids = sorted(r["video_id"] for r in records); by = {r["video_id"]: r for r in records}
    uf = UF(ids); edges = []
    sha_buckets = defaultdict(list); creator_buckets = defaultdict(list)
    for r in records:
        if r["video_sha256"]: sha_buckets[r["video_sha256"]].append(r["video_id"])
        if r["creator_id"]: creator_buckets[(r["creator_namespace"],r["creator_id"])].append(r["video_id"])
    candidates = set()
    for bucket in list(sha_buckets.values()) + list(creator_buckets.values()):
        for a in bucket:
            for b in bucket:
                if a < b: candidates.add((a,b))
    # Pairwise pHash is acceptable at n=744 and is performed only pre-metric.
    for ia, a in enumerate(ids):
        ra = by[a]
        for b in ids[ia+1:]:
            rb = by[b]; rules=[]
            if ra["video_sha256"] and ra["video_sha256"] == rb["video_sha256"]: rules.append("sha")
            if ra["creator_id"] and (ra["creator_namespace"],ra["creator_id"]) == (rb["creator_namespace"],rb["creator_id"]): rules.append("creator")
            if ra["phash30"] and rb["phash30"]:
                close=sum((int(x)^int(y)).bit_count() <= 6 for x,y in zip(ra["phash30"],rb["phash30"]))
                if close >= 24 and estimated_jaccard(ra["transcript_minhash128"],rb["transcript_minhash128"]) >= .80:
                    rules.append("phash_transcript")
            if rules:
                uf.union(a,b); edges.append({"a":a,"b":b,"rules":rules})
    comps=defaultdict(list)
    for vid in ids: comps[uf.find(vid)].append(vid)
    components={min(vs):sorted(vs) for vs in comps.values()}
    atomic_write(out/"group_edges.jsonl", b"".join(canonical_bytes(x) for x in edges))
    atomic_json(out/"group_components.json", components)
    return components


def assign_folds(records: list[dict], components: dict[str,list[str]], seed: int, requested: int) -> dict:
    labels={r["video_id"]:int(r["label"]) for r in records}
    groups=[]
    for gid, vids in components.items():
        groups.append((gid,vids,sum(labels[v] for v in vids),len(vids)))
    order=sorted(groups,key=lambda g:hashlib.sha256((str(seed)+"||"+g[0]).encode("utf-8")).hexdigest())
    for nfold in range(requested,2,-1):
        folds=[[] for _ in range(nfold)]; pos=[0]*nfold; total=[0]*nfold
        target_p=sum(g[2] for g in groups)/nfold
        target_neg=sum(g[3]-g[2] for g in groups)/nfold
        target_n=sum(g[3] for g in groups)/nfold
        for gid,vids,np_,nt in order:
            def objective(q):
                pp=pos.copy();tt=total.copy();pp[q]+=np_;tt[q]+=nt
                joint=sum(abs(pp[j]-target_p)/max(target_p,1)+abs((tt[j]-pp[j])-target_neg)/max(target_neg,1) for j in range(nfold))
                total_dev=sum(abs(tt[j]-target_n) for j in range(nfold))
                return (joint,total_dev,total[q],q)
            f=min(range(nfold),key=objective)
            folds[f].append(gid);pos[f]+=np_;total[f]+=nt
        if all(0 < pos[f] < total[f] for f in range(nfold)):
            return {"seed":seed,"n_folds":nfold,"folds":[{"fold":f,"group_ids":sorted(folds[f]),"video_ids":sorted(v for g in folds[f] for v in components[g]),"n":total[f],"n_positive":pos[f]} for f in range(nfold)]}
    raise RuntimeError("HALT_GROUP_FOLDS")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--sources",type=Path);args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.sources:
        records=[json.loads(x) for x in args.sources.open() if x.strip()]
    else: records=build_sources(args.out_dir)
    comps=build_components(records,args.out_dir)
    outer={str(s):assign_folds(records,comps,s,5) for s in (20260811,20260812,20260813)}
    atomic_json(args.out_dir/"outer_folds.json",outer)
    inner={}
    by={r["video_id"]:r for r in records}
    for s,assignment in outer.items():
        inner[s]={}
        for fold in assignment["folds"]:
            held=set(fold["group_ids"]);sub={g:v for g,v in comps.items() if g not in held}
            sub_records=[by[v] for vs in sub.values() for v in vs]
            derived=(1000003+1009*int(s)+int(fold["fold"]))%(1<<64)
            inner[s][str(fold["fold"])]=assign_folds(sub_records,sub,derived,4)
    atomic_json(args.out_dir/"inner_folds.json",inner)
    print(json.dumps({"n_videos":len(records),"n_groups":len(comps),"folds":{s:v["n_folds"] for s,v in outer.items()}},sort_keys=True))

if __name__ == "__main__": main()
