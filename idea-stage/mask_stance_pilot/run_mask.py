"""MASK STANCE PILOT -- two-step runner (extract -> programmatic mask -> masked stance).

Every stage caches to idea-stage/mask_stance_pilot/ and is idempotent: re-running a stage whose
output file already exists is a no-op unless --force is given.

  python run_mask.py extract  --split smoke --tag s1                 # realtime
  python run_mask.py extract  --split eval  --tag m1 --batch         # Batch API submit
  python run_mask.py poll     --tag m1 --stage ext
  python run_mask.py fetch    --tag m1 --stage ext
  python run_mask.py mask     --split eval  --tag m1
  python run_mask.py stance   --split smoke --tag s1                 # realtime
  python run_mask.py stance   --split eval  --tag m1 --batch
  python run_mask.py poll     --tag m1 --stage stn
  python run_mask.py fetch    --tag m1 --stage stn

The API key is read from ~/.dashscope_api_key and is never written to any output.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SP = os.path.join(ROOT, "idea-stage", "stance_pilot")
sys.path.insert(0, HERE)
sys.path.insert(0, SP)

from run_pilot import client, frame_urls, load_texts, parse_json  # noqa: E402
from mask_prompts import EXTRACT, MASKED_STANCE, MASK_NOTE, SYSTEM  # noqa: E402

TITLE_DS = {"MHC", "MHC_zh"}
SEED = 20260811
MATCH_BAR = 0.80
LOCK = threading.Lock()


def sample(split):
    return json.load(open(os.path.join(SP, "sample.json")))[split]


def p(tag, name):
    return os.path.join(HERE, f"{name}_{tag}.jsonl")


def title_note(ds):
    return "; its first sentence, before the first ' . ', is the video TITLE" if ds in TITLE_DS else ""


def texts_cache(items):
    c = {}
    for it in items:
        if it["dataset"] not in c:
            c[it["dataset"]] = load_texts(it["dataset"])
    return c


# ================================================================ step 1: extraction
def extract_body(ds, transcript):
    return EXTRACT.format(title_note=title_note(ds), transcript=transcript or "(empty)")


def extract_messages(ds, transcript):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": extract_body(ds, transcript)}]


def cmd_extract(a):
    out_p = p(a.tag, "extract")
    if os.path.exists(out_p) and not a.force and not a.batch:
        print("[extract] cached", out_p)
        return
    items = sample(a.split)
    T = texts_cache(items)
    reqs = [(it, (T[it["dataset"]].get(it["id"]) or "").strip()) for it in items]

    if a.batch:
        in_p = os.path.join(HERE, f"batch_in_ext_{a.tag}.jsonl")
        with open(in_p, "w") as f:
            for it, tx in reqs:
                f.write(json.dumps({
                    "custom_id": f"{it['dataset']}::{it['id']}",
                    "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": a.model,
                             "messages": extract_messages(it["dataset"], tx),
                             "max_tokens": 2000, "temperature": 0.0, "seed": SEED}},
                    ensure_ascii=False) + "\n")
        print("[extract] batch input", in_p, round(os.path.getsize(in_p) / 1e6, 3), "MB",
              len(reqs), "requests", flush=True)
        cli = client()
        fo = cli.files.create(file=open(in_p, "rb"), purpose="batch")
        b = cli.batches.create(input_file_id=fo.id, endpoint="/v1/chat/completions",
                               completion_window="24h")
        json.dump({"tag": a.tag, "stage": "ext", "model": a.model, "file_id": fo.id,
                   "batch_id": b.id, "n": len(reqs),
                   "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  open(os.path.join(HERE, f"batch_meta_ext_{a.tag}.json"), "w"), indent=1)
        print("[extract] batch", b.id, b.status, flush=True)
        return

    cli = client()

    def work(arg):
        it, tx = arg
        try:
            r = cli.chat.completions.create(model=a.model, messages=extract_messages(it["dataset"], tx),
                                            max_tokens=2000, temperature=0.0, seed=SEED)
            txt = r.choices[0].message.content
            obj, st = parse_json(txt)
            u = r.usage.model_dump()
        except Exception as e:
            txt, obj, st, u = None, None, "err:" + str(e)[:120], None
        rec = {"dataset": it["dataset"], "id": it["id"], "group": it["group"],
               "parse": st, "parsed": obj, "raw": txt, "usage": u, "model": a.model,
               "stage": "extract"}
        with LOCK:
            ns = len((obj or {}).get("spans") or [])
            print(f"[extract] {it['id']:<22} {st:<10} spans={ns}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        recs = list(ex.map(work, reqs))
    with open(out_p, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[extract] wrote", out_p, len(recs))


# ================================================================ step 1.5: masking
WS = re.compile(r"\s+")
CJK = re.compile(r"[一-鿿㐀-䶿]")


def normalise(s):
    """lowercase + collapse whitespace; return (norm, idx_map) with idx_map[k] = orig index."""
    out, idx = [], []
    i, n = 0, len(s)
    prev_ws = True
    while i < n:
        ch = s[i]
        if ch.isspace():
            if not prev_ws:
                out.append(" ")
                idx.append(i)
            prev_ws = True
            i += 1
            continue
        out.append(ch.lower())
        idx.append(i)
        prev_ws = False
        i += 1
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def locate(nt, ns):
    """return (start, end, ratio) in normalised-transcript coordinates, or None."""
    if not ns or not nt:
        return None
    i = nt.find(ns)
    if i >= 0:
        return (i, i + len(ns), 1.0)
    L = len(ns)
    best = None
    k = min(16, max(5, L // 4))
    head = ns[:k]
    cands, start = [], 0
    while len(cands) < 60:
        j = nt.find(head, start)
        if j < 0:
            break
        cands.append(j)
        start = j + 1
    for j in cands:
        for LL in (L, int(L * 1.2) + 1, max(1, int(L * 0.85))):
            seg = nt[j:j + LL]
            if not seg:
                continue
            r = SequenceMatcher(None, seg, ns).ratio()
            if best is None or r > best[2]:
                best = (j, j + len(seg), r)
    if best is None or best[2] < MATCH_BAR:
        step = max(1, L // 8)
        for j in range(0, max(1, len(nt)), step):
            seg = nt[j:j + L]
            if len(seg) < L * 0.6:
                break
            sm = SequenceMatcher(None, seg, ns)
            if sm.quick_ratio() < 0.7:
                continue
            r = sm.ratio()
            if best is None or r > best[2]:
                best = (j, j + len(seg), r)
    return best


def placeholder(target, cjk):
    t = (target or "").strip()
    if not t:
        t = "某一群体" if cjk else "a group"
    t = WS.sub(" ", t)[:40]
    return f"[一段针对{t}的攻击性言论]" if cjk else f"[a passage of attacking speech targeting {t}]"


def mask_one(transcript, spans):
    """returns (masked_text, report)."""
    rep = {"n_spans": len(spans), "matched": 0, "unmatched": 0, "ratios": [],
           "masked_chars": 0, "orig_chars": len(transcript), "unmatched_spans": [],
           "targets": []}
    if not transcript:
        return transcript, rep
    cjk = len(CJK.findall(transcript)) >= 0.15 * max(1, len(WS.sub("", transcript)))
    rep["cjk"] = cjk
    nt, idx = normalise(transcript)
    hits = []

    def add(s, e, r, sp, stext):
        s = max(0, min(s, len(idx) - 1))
        e = max(s + 1, min(e, len(idx)))
        hits.append({"os": idx[s], "oe": idx[e - 1] + 1, "ratio": round(r, 3),
                     "target": (sp or {}).get("target"), "text": stext[:120]})

    for sp in spans:
        stext = (sp or {}).get("text") or ""
        ns, _ = normalise(stext)
        if len(ns) < 2:
            rep["unmatched"] += 1
            rep["unmatched_spans"].append(stext[:80])
            continue
        # E1.1: mask EVERY exact occurrence of the span, not just the first, so that a
        # repeated slur the extractor listed once does not leak back into step 2's context.
        occ, start = [], 0
        while True:
            j = nt.find(ns, start)
            if j < 0 or len(occ) >= 200:
                break
            occ.append(j)
            start = j + max(1, len(ns))
        if occ:
            for j in occ:
                add(j, j + len(ns), 1.0, sp, stext)
            rep["ratios"].append(1.0)
            rep["matched"] += 1
            rep["n_exact_occurrences"] = rep.get("n_exact_occurrences", 0) + len(occ)
            continue
        if len(ns) < 5:            # too short for fuzzy matching to be safe
            rep["unmatched"] += 1
            rep["unmatched_spans"].append(stext[:80])
            continue
        loc = locate(nt, ns)
        if not loc or loc[2] < MATCH_BAR:
            rep["unmatched"] += 1
            rep["unmatched_spans"].append(stext[:80])
            rep["ratios"].append(round(loc[2], 3) if loc else 0.0)
            continue
        s, e, r = loc
        add(s, e, r, sp, stext)
        rep["ratios"].append(round(r, 3))
        rep["matched"] += 1
    # drop overlaps, prefer longer then higher ratio
    hits.sort(key=lambda h: (-(h["oe"] - h["os"]), -h["ratio"]))
    keep = []
    for h in hits:
        if all(h["oe"] <= k["os"] or h["os"] >= k["oe"] for k in keep):
            keep.append(h)
    keep.sort(key=lambda h: h["os"])
    # merge adjacent same-target placeholders separated by <= 3 chars
    merged = []
    for h in keep:
        if merged and (h["target"] or "") == (merged[-1]["target"] or "") \
                and h["os"] - merged[-1]["oe"] <= 3:
            merged[-1]["oe"] = h["oe"]
        else:
            merged.append(dict(h))
    parts, cur, mc = [], 0, 0
    for h in merged:
        parts.append(transcript[cur:h["os"]])
        parts.append(placeholder(h["target"], cjk))
        mc += h["oe"] - h["os"]
        cur = h["oe"]
        rep["targets"].append(h["target"])
    parts.append(transcript[cur:])
    masked = WS.sub(" ", "".join(parts)).strip()
    rep["masked_chars"] = mc
    rep["n_placeholders"] = len(merged)
    rep["masked_frac"] = round(mc / max(1, len(transcript)), 3)
    rep["spans_kept"] = merged
    # residual-leak audit: is any extracted span still verbatim in the masked transcript?
    nm, _ = normalise(masked)
    leaks = []
    for sp in spans:
        ns, _ = normalise((sp or {}).get("text") or "")
        if len(ns) >= 2 and ns in nm:
            leaks.append(ns[:60])
    rep["residual_leaks"] = leaks
    return masked, rep


TEXT_RE = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"')
TARGET_RE = re.compile(r'"target"\s*:\s*"((?:[^"\\]|\\.)*)"')


def salvage_spans(raw):
    """Deviation D1: recover spans from a malformed extraction reply.

    qwen3-vl-plus occasionally corrupts the "text" KEY of a span object into a bare commentary
    string (e.g. {" irresponsibly transcribed as: \\"...\\"", "target": "..."}), which makes the
    whole reply unparseable and would send an item to step 2 completely unmasked. This recovers
    every well-formed "text" field and pairs it with the next "target" field that occurs before
    the following "text" field; a span with no such target keeps target=None and gets the generic
    placeholder. Conservative: it never invents a span and never mis-pairs across span boundaries.
    """
    if not raw:
        return []
    texts = list(TEXT_RE.finditer(raw))
    targets = list(TARGET_RE.finditer(raw))
    out = []
    for i, m in enumerate(texts):
        nxt = texts[i + 1].start() if i + 1 < len(texts) else len(raw)
        tgt = None
        for t in targets:
            if m.end() <= t.start() < nxt:
                tgt = t.group(1)
                break
        try:
            txt = json.loads('"' + m.group(1) + '"')
        except Exception:
            txt = m.group(1)
        out.append({"text": txt, "target": tgt, "salvaged": True})
    return out


def cmd_mask(a):
    out_p = p(a.tag, "masked")
    items = {(it["dataset"], it["id"]): it for it in sample(a.split)}
    T = texts_cache(list(items.values()))
    recs = []
    n_salv = 0
    for line in open(p(a.tag, "extract"), encoding="utf-8"):
        r = json.loads(line)
        it = items[(r["dataset"], r["id"])]
        tx = (T[r["dataset"]].get(r["id"]) or "").strip()
        spans = ((r.get("parsed") or {}).get("spans") or [])
        if not isinstance(spans, list):
            spans = []
        if not spans and r.get("parse") != "ok":
            spans = salvage_spans(r.get("raw"))
            if spans:
                n_salv += 1
        masked, rep = mask_one(tx, spans)
        recs.append({"dataset": r["dataset"], "id": r["id"], "group": it["group"],
                     "extract_parse": r["parse"],
                     "salvaged": bool(spans and spans[0].get("salvaged")),
                     "any_hate_surface": (r.get("parsed") or {}).get("any_hate_surface"),
                     "orig": tx, "masked": masked, "report": rep})
    with open(out_p, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    nm = sum(1 for r in recs if r["report"].get("n_placeholders", 0) > 0)
    tot_s = sum(r["report"]["n_spans"] for r in recs)
    tot_u = sum(r["report"]["unmatched"] for r in recs)
    print(f"[mask] wrote {out_p}  items={len(recs)}  with>=1 placeholder={nm}  "
          f"spans={tot_s} unmatched={tot_u} ({tot_u / max(1, tot_s):.3f})  salvaged_items={n_salv}")


# ================================================================ step 2: masked stance
def stance_messages(it, masked, n_placeholders, n_frames):
    ds = it["dataset"]
    urls = frame_urls(ds, it["id"], n_frames)
    frames_note = (f"{len(urls)} evenly spaced frames of the video are attached above, "
                   "in temporal order." if urls else
                   "No video frames are available for this item; judge from the transcript alone.")
    body = MASKED_STANCE.format(frames_note=frames_note, title_note=title_note(ds),
                                transcript=masked or "(empty)",
                                mask_note=(MASK_NOTE if n_placeholders > 0 else ""))
    content = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    content.append({"type": "text", "text": body})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}], len(urls)


def cmd_stance(a):
    out_p = p(a.tag, "pred")
    if os.path.exists(out_p) and not a.force and not a.batch:
        print("[stance] cached", out_p)
        return
    items = {(it["dataset"], it["id"]): it for it in sample(a.split)}
    rows = [json.loads(l) for l in open(p(a.tag, "masked"), encoding="utf-8")]

    if a.batch:
        in_p = os.path.join(HERE, f"batch_in_stn_{a.tag}.jsonl")
        with open(in_p, "w") as f:
            for r in rows:
                it = items[(r["dataset"], r["id"])]
                msgs, nf = stance_messages(it, r["masked"],
                                           r["report"].get("n_placeholders", 0), a.frames)
                f.write(json.dumps({
                    "custom_id": f"{r['dataset']}::{r['id']}",
                    "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": a.model, "messages": msgs, "max_tokens": 400,
                             "temperature": 0.0, "seed": SEED}}, ensure_ascii=False) + "\n")
        print("[stance] batch input", in_p, round(os.path.getsize(in_p) / 1e6, 3), "MB",
              len(rows), "requests", flush=True)
        cli = client()
        fo = cli.files.create(file=open(in_p, "rb"), purpose="batch")
        b = cli.batches.create(input_file_id=fo.id, endpoint="/v1/chat/completions",
                               completion_window="24h")
        json.dump({"tag": a.tag, "stage": "stn", "model": a.model, "file_id": fo.id,
                   "batch_id": b.id, "n": len(rows), "frames": a.frames,
                   "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  open(os.path.join(HERE, f"batch_meta_stn_{a.tag}.json"), "w"), indent=1)
        print("[stance] batch", b.id, b.status, flush=True)
        return

    cli = client()

    def work(r):
        it = items[(r["dataset"], r["id"])]
        npl = r["report"].get("n_placeholders", 0)
        msgs, nf = stance_messages(it, r["masked"], npl, a.frames)
        try:
            resp = cli.chat.completions.create(model=a.model, messages=msgs, max_tokens=400,
                                               temperature=0.0, seed=SEED)
            txt = resp.choices[0].message.content
            obj, st = parse_json(txt)
            u = resp.usage.model_dump()
        except Exception as e:
            txt, obj, st, u = None, None, "err:" + str(e)[:120], None
        rec = {"dataset": r["dataset"], "id": r["id"], "n_frames_sent": nf,
               "n_placeholders": npl, "parse": st, "parsed": obj, "raw": txt,
               "usage": u, "model": a.model, "prompt": "M1_masked"}
        with LOCK:
            print(f"[stance] {r['id']:<22} pl={npl} {st:<10} "
                  f"{(obj or {}).get('stance')}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        recs = list(ex.map(work, rows))
    with open(out_p, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[stance] wrote", out_p, len(recs))


# ================================================================ batch plumbing
def _meta(tag, stage):
    return json.load(open(os.path.join(HERE, f"batch_meta_{stage}_{tag}.json")))


def cmd_poll(a):
    cli = client()
    m = _meta(a.tag, a.stage)
    while True:
        b = cli.batches.retrieve(m["batch_id"])
        print(time.strftime("%H:%M:%S"), a.stage, b.status, b.request_counts, flush=True)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            m.update({"status": b.status, "output_file_id": b.output_file_id,
                      "error_file_id": b.error_file_id})
            json.dump(m, open(os.path.join(HERE, f"batch_meta_{a.stage}_{a.tag}.json"), "w"),
                      indent=1)
            return
        time.sleep(30)


def cmd_fetch(a):
    cli = client()
    m = _meta(a.tag, a.stage)
    b = cli.batches.retrieve(m["batch_id"])
    raw_p = os.path.join(HERE, f"batch_out_{a.stage}_{a.tag}.jsonl")
    for kind, fid in (("out", b.output_file_id), ("err", b.error_file_id)):
        if not fid:
            continue
        raw = cli.files.content(fid).content.decode()
        open(os.path.join(HERE, f"batch_{kind}_{a.stage}_{a.tag}.jsonl"), "w").write(raw)
        print(kind, len(raw.splitlines()), "lines")
    if not os.path.exists(raw_p):
        print("no output file")
        return
    items = {(it["dataset"], it["id"]): it for it in sample("eval")}
    recs = []
    for line in open(raw_p, encoding="utf-8"):
        r = json.loads(line)
        ds, vid = r["custom_id"].split("::", 1)
        body = (r.get("response") or {}).get("body") or {}
        ch = (body.get("choices") or [{}])[0]
        txt = (ch.get("message") or {}).get("content")
        obj, st = parse_json(txt)
        rec = {"dataset": ds, "id": vid, "parse": st, "parsed": obj, "raw": txt,
               "usage": body.get("usage"), "model": m["model"]}
        if a.stage == "ext":
            rec.update({"group": items[(ds, vid)]["group"], "stage": "extract"})
        else:
            rec["prompt"] = "M1_masked"
        recs.append(rec)
    out = p(a.tag, "extract" if a.stage == "ext" else "pred")
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in recs if r["parse"] == "ok")
    print("parsed", ok, "/", len(recs), "->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "mask", "stance", "poll", "fetch"])
    ap.add_argument("--split", default="eval", choices=["smoke", "eval"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="qwen3-vl-plus")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stage", default="ext", choices=["ext", "stn"])
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
