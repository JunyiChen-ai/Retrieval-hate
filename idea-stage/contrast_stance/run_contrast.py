"""CONTRAST STANCE PILOT -- pinned-comment two-alternative forced choice.

Rules frozen in idea-stage/MASK_STANCE_PILOT_FREEZE.md Appendix B.
Every stage caches to idea-stage/contrast_stance/ and is idempotent.

  python run_contrast.py smoke  --tag s1                     # realtime, 11 items
  python run_contrast.py submit --tag c1 --pair 0            # Batch API, one pair index
  python run_contrast.py poll   --tag c1 --pair 0
  python run_contrast.py fetch  --tag c1 --pair 0
  python run_contrast.py merge  --tag c1                     # pred_c1.jsonl

The API key is read from ~/.dashscope_api_key and is never written to any output.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SP = os.path.join(ROOT, "idea-stage", "stance_pilot")
MSP = os.path.join(ROOT, "idea-stage", "mask_stance_pilot")
sys.path.insert(0, HERE)
sys.path.insert(0, SP)

from run_pilot import client, frame_urls, load_texts, MAX_SIDE, JPEG_Q  # noqa: E402
from prompts import SYSTEM  # noqa: E402
from contrast_prompts import BANK, FRAME, N_PAIRS  # noqa: E402

SEED = 20260813
SALT = "20260813"
TITLE_DS = {"MHC", "MHC_zh"}
CJK = re.compile(r"[一-鿿㐀-䶿]")
WS = re.compile(r"\s+")
LOCK = threading.Lock()

# The 3 eval items the user named as mandatory qualitative smoke checks (freeze B.8).
# They are seen before the freeze is consumed and are therefore removed from the
# primary metric denominator by the scorer.
SMOKE_EVAL = [("MHC", "KDcCiUU8q5E"), ("HateMM", "non_hate_video_32"),
              ("HateMM", "non_hate_video_16")]


# ---------------------------------------------------------------- inputs
def sample(split):
    return json.load(open(os.path.join(SP, "sample.json")))[split]


def smoke_items():
    ev = {(x["dataset"], x["id"]): x for x in sample("eval")}
    out = list(sample("smoke"))
    for k in SMOKE_EVAL:
        it = dict(ev[k])
        it["_from_eval"] = True
        out.append(it)
    return out


def texts_cache(items):
    c = {}
    for it in items:
        if it["dataset"] not in c:
            c[it["dataset"]] = load_texts(it["dataset"])
    return c


def targets_map():
    """Most frequent non-empty `target` per item from the previous pilot's extraction step."""
    out = {}
    p = os.path.join(MSP, "extract_m1.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        spans = ((r.get("parsed") or {}).get("spans") or [])
        if not isinstance(spans, list):
            continue
        ts = [WS.sub(" ", str(s.get("target")).strip())
              for s in spans if isinstance(s, dict) and s.get("target")]
        ts = [t for t in ts if t and t.lower() not in ("null", "none")]
        if ts:
            out[(r["dataset"], r["id"])] = Counter(ts).most_common(1)[0][0][:40]
    return out


def lang_of(ds, transcript):
    """freeze B.4: ZH iff CJK >= 15% of non-whitespace chars; <10 chars -> dataset default."""
    t = transcript or ""
    nw = len(WS.sub("", t))
    if nw < 10:
        return "zh" if ds == "MHC_zh" else "en"
    return "zh" if len(CJK.findall(t)) >= 0.15 * nw else "en"


def swap(ds, vid, variant, pair):
    """True  -> slot A carries the ENDORSING template (freeze B.4)."""
    h = hashlib.sha256(f"{SALT}|{ds}|{vid}|{variant}|{pair}".encode()).digest()
    return (h[0] & 1) == 0


def build_one(it, transcript, variant, pair, tgt):
    ds, vid = it["dataset"], it["id"]
    lang = lang_of(ds, transcript)
    end_t, opp_t = BANK[(variant, lang)][pair]
    if variant == "v2":
        end_t, opp_t = end_t.format(t=tgt), opp_t.format(t=tgt)
    a_is_endorse = swap(ds, vid, variant, pair)
    opt_a, opt_b = (end_t, opp_t) if a_is_endorse else (opp_t, end_t)
    urls = frame_urls(ds, vid, 8)
    frames_note = (f"{len(urls)} evenly spaced frames of the video are attached above, "
                   "in temporal order." if urls else
                   "No video frames are available for this item; judge from the transcript alone.")
    title_note = ("; its first sentence, before the first ' . ', is the video TITLE"
                  if ds in TITLE_DS else "")
    body = FRAME.format(frames_note=frames_note, title_note=title_note,
                        transcript=(transcript or "").strip() or "(empty)",
                        opt_a=opt_a, opt_b=opt_b)
    content = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    content.append({"type": "text", "text": body})
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    meta = {"dataset": ds, "id": vid, "group": it["group"], "variant": variant, "pair": pair,
            "lang": lang, "a_is_endorse": a_is_endorse, "n_frames": len(urls),
            "target": tgt if variant == "v2" else None,
            "opt_a": opt_a, "opt_b": opt_b, "endorse_text": end_t, "oppose_text": opp_t}
    return msgs, meta


def build_all(items, pair=None):
    """Yield (msgs, meta) for every (item, variant, pair) request in a frozen order."""
    T = texts_cache(items)
    TG = targets_map()
    out = []
    for it in items:
        tx = (T[it["dataset"]].get(it["id"]) or "").strip()
        tgt = TG.get((it["dataset"], it["id"]))
        for variant in ("v1", "v2"):
            if variant == "v2" and not tgt:
                continue
            for p in range(N_PAIRS):
                if pair is not None and p != pair:
                    continue
                out.append(build_one(it, tx, variant, p, tgt))
    return out


VOTE_RE = re.compile(r"\b([AB])\b")


def parse_vote(txt):
    if not txt:
        return None
    t = txt.strip().upper()
    m = VOTE_RE.search(t)
    if m:
        return m.group(1)
    t2 = re.sub(r"[^AB]", "", t)
    return t2[0] if t2 else None


def rec_of(meta, txt, usage, model):
    v = parse_vote(txt)
    side = None
    if v:
        side = "ENDORSE" if (v == "A") == meta["a_is_endorse"] else "OPPOSE"
    return {**meta, "reply": txt, "vote_slot": v, "vote_side": side,
            "usage": usage, "model": model}


# ---------------------------------------------------------------- realtime smoke
def cmd_smoke(a):
    out_p = os.path.join(HERE, f"smoke_{a.tag}.jsonl")
    if os.path.exists(out_p) and not a.force:
        print("[smoke] cached", out_p)
        return
    reqs = build_all(smoke_items())
    print(f"[smoke] {len(reqs)} requests over {len(smoke_items())} items", flush=True)
    cli = client()

    def work(rm):
        msgs, meta = rm
        try:
            r = cli.chat.completions.create(model=a.model, messages=msgs, max_tokens=8,
                                            temperature=0.0, seed=SEED)
            txt = r.choices[0].message.content
            u = r.usage.model_dump()
        except Exception as e:
            txt, u = None, {"error": str(e)[:160]}
        rec = rec_of(meta, txt, u, a.model)
        with LOCK:
            print(f"[smoke] {meta['id']:<22} {meta['variant']} p{meta['pair']} "
                  f"{meta['lang']} A={'E' if meta['a_is_endorse'] else 'O'} "
                  f"-> {rec['vote_slot']} = {rec['vote_side']}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        recs = list(ex.map(work, reqs))
    with open(out_p, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[smoke] wrote", out_p, len(recs))


# ---------------------------------------------------------------- batch
def _mp(tag, pair):
    return os.path.join(HERE, f"batch_meta_{tag}_p{pair}.json")


def cmd_submit(a):
    if os.path.exists(_mp(a.tag, a.pair)) and not a.force:
        print("[submit] already submitted", _mp(a.tag, a.pair))
        return
    reqs = build_all(sample("eval"), pair=a.pair)
    in_p = os.path.join(HERE, f"batch_in_{a.tag}_p{a.pair}.jsonl")
    with open(in_p, "w") as f:
        for msgs, meta in reqs:
            f.write(json.dumps({
                "custom_id": f"{meta['dataset']}::{meta['id']}::{meta['variant']}::{meta['pair']}",
                "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": a.model, "messages": msgs, "max_tokens": 8,
                         "temperature": 0.0, "seed": SEED}}, ensure_ascii=False) + "\n")
    meta_p = os.path.join(HERE, f"reqmeta_{a.tag}_p{a.pair}.jsonl")
    with open(meta_p, "w") as f:
        for _, meta in reqs:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"[submit] pair {a.pair}: {len(reqs)} requests, "
          f"{os.path.getsize(in_p)/1e6:.1f} MB", flush=True)
    if a.dry:
        return
    cli = client()
    t0 = time.time()
    fo = cli.files.create(file=open(in_p, "rb"), purpose="batch")
    print(f"[submit] uploaded in {time.time()-t0:.0f}s", flush=True)
    b = cli.batches.create(input_file_id=fo.id, endpoint="/v1/chat/completions",
                           completion_window="24h")
    json.dump({"tag": a.tag, "pair": a.pair, "model": a.model, "file_id": fo.id,
               "batch_id": b.id, "n": len(reqs),
               "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(_mp(a.tag, a.pair), "w"), indent=1)
    print("[submit] batch", b.id, b.status, flush=True)


def cmd_poll(a):
    cli = client()
    m = json.load(open(_mp(a.tag, a.pair)))
    while True:
        b = cli.batches.retrieve(m["batch_id"])
        print(time.strftime("%H:%M:%S"), f"p{a.pair}", b.status, b.request_counts, flush=True)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            m.update({"status": b.status, "output_file_id": b.output_file_id,
                      "error_file_id": b.error_file_id})
            json.dump(m, open(_mp(a.tag, a.pair), "w"), indent=1)
            return
        time.sleep(30)


def cmd_fetch(a):
    out_p = os.path.join(HERE, f"pred_{a.tag}_p{a.pair}.jsonl")
    if os.path.exists(out_p) and not a.force:
        print("[fetch] cached", out_p)
        return
    cli = client()
    m = json.load(open(_mp(a.tag, a.pair)))
    b = cli.batches.retrieve(m["batch_id"])
    raw_p = os.path.join(HERE, f"batch_out_{a.tag}_p{a.pair}.jsonl")
    for kind, fid in (("out", b.output_file_id), ("err", b.error_file_id)):
        if not fid:
            continue
        raw = cli.files.content(fid).content.decode()
        open(os.path.join(HERE, f"batch_{kind}_{a.tag}_p{a.pair}.jsonl"), "w").write(raw)
        print(kind, len(raw.splitlines()), "lines")
    if not os.path.exists(raw_p):
        print("no output file")
        return
    metas = {}
    for line in open(os.path.join(HERE, f"reqmeta_{a.tag}_p{a.pair}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        metas[(r["dataset"], r["id"], r["variant"], r["pair"])] = r
    recs = []
    for line in open(raw_p, encoding="utf-8"):
        r = json.loads(line)
        ds, vid, variant, pair = r["custom_id"].split("::")
        meta = metas[(ds, vid, variant, int(pair))]
        body = (r.get("response") or {}).get("body") or {}
        ch = (body.get("choices") or [{}])[0]
        txt = (ch.get("message") or {}).get("content")
        recs.append(rec_of(meta, txt, body.get("usage"), m["model"]))
    with open(out_p, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in recs if r["vote_side"])
    print("votes parsed", ok, "/", len(recs), "->", out_p)


def cmd_merge(a):
    recs = []
    for p in range(N_PAIRS):
        fp = os.path.join(HERE, f"pred_{a.tag}_p{p}.jsonl")
        if not os.path.exists(fp):
            print("[merge] MISSING", fp)
            continue
        recs += [json.loads(l) for l in open(fp, encoding="utf-8")]
    out_p = os.path.join(HERE, f"pred_{a.tag}.jsonl")
    with open(out_p, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[merge] wrote", out_p, len(recs), "votes;",
          "parsed", sum(1 for r in recs if r["vote_side"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "submit", "poll", "fetch", "merge"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--model", default="qwen3-vl-plus")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
