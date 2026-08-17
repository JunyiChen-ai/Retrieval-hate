"""STANCE PILOT -- request builder + DashScope runner (realtime smoke / Batch API for the paid run).

Usage
  python run_pilot.py smoke  --model M --prompt V1 [--tag t]
  python run_pilot.py submit --model M --prompt V1 [--frames 8] [--ocr] [--tag t]
  python run_pilot.py poll   --tag t
  python run_pilot.py fetch  --tag t

The API key is read from ~/.dashscope_api_key and is never written to any output.
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import time
from collections import OrderedDict

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from prompts import BANK, SYSTEM  # noqa: E402

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_SIDE = 512          # frozen low-resolution tier
JPEG_Q = 80
TITLE_DS = {"MHC", "MHC_zh"}   # test.jsonl `text` = "<title> . <transcript>"


def key():
    return open(os.path.expanduser("~/.dashscope_api_key")).read().strip()


def client():
    from openai import OpenAI
    # the link to DashScope from this host runs ~30 KB/s, so a 17 MB batch file needs ~10 min
    # to upload; the SDK's 600 s default timeout silently retried the whole upload. Infra fix
    # only -- no change to any input, prompt or judgement rule.
    return OpenAI(api_key=key(), base_url=BASE_URL, timeout=3600.0, max_retries=1)


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "")


def load_texts(ds):
    o = {}
    for line in open(os.path.join(ROOT, "data", "gt", ds, "test.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        o[r["id"]] = strip_html(r.get("text", ""))
    return o


def load_ocr(ds):
    p = os.path.join(ROOT, "data", "OCR", ds, "ocr_video_test.jsonl")
    if not os.path.exists(p):
        return {}
    o = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        seen, keep = OrderedDict(), []
        for ln in (r.get("text") or "").split("\n"):
            ln = ln.strip()
            if ln and ln not in seen:
                seen[ln] = 1
                keep.append(ln)
        o[r["video_id"]] = " | ".join(keep)
    return o


def frame_urls(ds, vid, n_frames):
    d = os.path.join(ROOT, "data", "lora_frames", ds, vid)
    if not os.path.isdir(d):
        return []
    fs = sorted(os.listdir(d), key=lambda x: int(re.findall(r"\d+", x)[0]))
    if n_frames < len(fs):
        step = len(fs) / n_frames
        fs = [fs[int(i * step)] for i in range(n_frames)]
    out = []
    for f in fs:
        im = Image.open(os.path.join(d, f)).convert("RGB")
        if max(im.size) > MAX_SIDE:
            s = MAX_SIDE / max(im.size)
            im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_Q)
        out.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    return out


def build_messages(item, texts, ocrs, prompt_key, n_frames, use_ocr):
    ds, vid = item["dataset"], item["id"]
    urls = frame_urls(ds, vid, n_frames)
    frames_note = (f"{len(urls)} evenly spaced frames of the video are attached above, "
                   "in temporal order." if urls else
                   "No video frames are available for this item; judge from the transcript alone.")
    title_note = ("; its first sentence, before the first ' . ', is the video TITLE"
                  if ds in TITLE_DS else "")
    extra = ""
    if use_ocr and vid in ocrs and ocrs[vid]:
        extra = ("On-screen text recovered by OCR (deduplicated, may be noisy):\n<<<\n"
                 + ocrs[vid][:4000] + "\n>>>\n")
    body = BANK[prompt_key].format(frames_note=frames_note, title_note=title_note,
                                   transcript=(texts.get(vid) or "").strip() or "(empty)",
                                   extra=extra)
    content = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    content.append({"type": "text", "text": body})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}], len(urls)


def iter_items(which):
    s = json.load(open(os.path.join(HERE, "sample.json")))
    return s[which]


def make_requests(items, model, prompt_key, n_frames, use_ocr):
    cache_t, cache_o = {}, {}
    reqs = []
    for it in items:
        ds = it["dataset"]
        if ds not in cache_t:
            cache_t[ds] = load_texts(ds)
            cache_o[ds] = load_ocr(ds)
        msgs, nf = build_messages(it, cache_t[ds], cache_o[ds], prompt_key, n_frames, use_ocr)
        reqs.append((it, msgs, nf))
    return reqs


def parse_json(txt):
    if txt is None:
        return None, "empty"
    t = txt.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None, "no_json"
    try:
        return json.loads(m.group(0)), "ok"
    except Exception as e:
        return None, "bad_json:" + str(e)[:60]


# ---------------------------------------------------------------- realtime smoke
def cmd_smoke(a):
    cli = client()
    reqs = make_requests(iter_items("smoke"), a.model, a.prompt, a.frames, a.ocr)
    out = []
    for it, msgs, nf in reqs:
        t0 = time.time()
        r = cli.chat.completions.create(model=a.model, messages=msgs, max_tokens=400,
                                        temperature=0.0, seed=20260811)
        txt = r.choices[0].message.content
        obj, st = parse_json(txt)
        rec = {"id": it["id"], "dataset": it["dataset"], "group": it["group"],
               "label": it["label"], "n_frames_sent": nf, "parse": st, "parsed": obj,
               "raw": txt, "usage": r.usage.model_dump(), "secs": round(time.time() - t0, 1),
               "model": a.model, "prompt": a.prompt}
        out.append(rec)
        print(json.dumps({k: rec[k] for k in ("id", "group", "n_frames_sent", "parse", "parsed",
                                              "usage")}, ensure_ascii=False)[:600], flush=True)
    p = os.path.join(HERE, f"smoke_{a.tag}.jsonl")
    with open(p, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", p)


# ---------------------------------------------------------------- batch
def cmd_submit(a):
    cli = client()
    reqs = make_requests(iter_items("eval"), a.model, a.prompt, a.frames, a.ocr)
    p_in = os.path.join(HERE, f"batch_in_{a.tag}.jsonl")
    with open(p_in, "w") as f:
        for it, msgs, nf in reqs:
            f.write(json.dumps({
                "custom_id": f"{it['dataset']}::{it['id']}",
                "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": a.model, "messages": msgs, "max_tokens": 400,
                         "temperature": 0.0, "seed": 20260811}}, ensure_ascii=False) + "\n")
    print("input file", p_in, os.path.getsize(p_in) / 1e6, "MB", len(reqs), "requests", flush=True)
    fo = cli.files.create(file=open(p_in, "rb"), purpose="batch")
    b = cli.batches.create(input_file_id=fo.id, endpoint="/v1/chat/completions",
                           completion_window="24h")
    meta = {"tag": a.tag, "model": a.model, "prompt": a.prompt, "frames": a.frames,
            "ocr": a.ocr, "file_id": fo.id, "batch_id": b.id, "n": len(reqs),
            "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(meta, open(os.path.join(HERE, f"batch_meta_{a.tag}.json"), "w"), indent=1)
    print("batch", b.id, b.status, flush=True)


def _meta(tag):
    return json.load(open(os.path.join(HERE, f"batch_meta_{tag}.json")))


def cmd_poll(a):
    cli = client()
    m = _meta(a.tag)
    while True:
        b = cli.batches.retrieve(m["batch_id"])
        print(time.strftime("%H:%M:%S"), b.status, b.request_counts, flush=True)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            m["status"] = b.status
            m["output_file_id"] = b.output_file_id
            m["error_file_id"] = b.error_file_id
            json.dump(m, open(os.path.join(HERE, f"batch_meta_{a.tag}.json"), "w"), indent=1)
            return
        time.sleep(30)


def cmd_fetch(a):
    cli = client()
    m = _meta(a.tag)
    b = cli.batches.retrieve(m["batch_id"])
    for kind, fid in (("out", b.output_file_id), ("err", b.error_file_id)):
        if not fid:
            continue
        raw = cli.files.content(fid).content.decode()
        open(os.path.join(HERE, f"batch_{kind}_{a.tag}.jsonl"), "w").write(raw)
        print(kind, len(raw.splitlines()), "lines")
    # normalise
    p = os.path.join(HERE, f"batch_out_{a.tag}.jsonl")
    if os.path.exists(p):
        recs = []
        for line in open(p):
            r = json.loads(line)
            ds, vid = r["custom_id"].split("::", 1)
            body = (r.get("response") or {}).get("body") or {}
            ch = (body.get("choices") or [{}])[0]
            txt = (ch.get("message") or {}).get("content")
            obj, st = parse_json(txt)
            recs.append({"dataset": ds, "id": vid, "parse": st, "parsed": obj, "raw": txt,
                         "usage": body.get("usage"), "model": m["model"], "prompt": m["prompt"]})
        o = os.path.join(HERE, f"pred_{a.tag}.jsonl")
        with open(o, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ok = sum(1 for r in recs if r["parse"] == "ok")
        print("parsed", ok, "/", len(recs), "->", o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "submit", "poll", "fetch"])
    ap.add_argument("--model", default="qwen3-vl-plus-2025-12-19")
    ap.add_argument("--prompt", default="V1")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--tag", default="t")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
