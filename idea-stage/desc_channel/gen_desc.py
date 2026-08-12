"""DESC_CHANNEL step 1 -- perceptual description generation for all 1066 HateMM videos.

Spec frozen in idea-stage/DESC_CHANNEL_FREEZE.md section 2.

  python gen_desc.py smoke   [--prompt V1] [--tag v1]
  python gen_desc.py size                              # payload accounting, no API call
  python gen_desc.py submit  [--prompt V2] [--shard N] # Batch API, one file per shard
  python gen_desc.py poll
  python gen_desc.py fetch
  python gen_desc.py realtime [--workers 12]           # realtime fallback / gap filler
  python gen_desc.py repair                            # one regeneration pass for violations
  python gen_desc.py report

Idempotent: descriptions_hatemm.jsonl is append-only and keyed by id; any id already present
is never re-requested.  The API key is read from ~/.dashscope_api_key and never written out.
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
from collections import OrderedDict

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from defect import load_gt  # noqa: E402
from prompts import BANK, FIELDS, SYSTEM  # noqa: E402

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_SIDE = 512
JPEG_Q = 80
N_FRAMES = 8
MAX_TOKENS = 700
SEED = 20260813
OUT = os.path.join(HERE, "descriptions_hatemm.jsonl")

BANNED = [
    "hateful", "hatred", "hate", "offensive", "offend", "racist", "racism", "slur",
    "derogatory", "bigot", "bigoted", "bigotry", "supremacist", "supremacy",
    "extremist", "extremism", "propaganda", "harmful", "harm", "toxic", "abusive",
    "abuse", "discriminat", "antisemit", "anti-semit", "xenophob", "homophob",
    "transphob", "misogyn", "sexist", "sexism", "nazi", "kkk", "radicalis", "radicaliz",
    "仇恨", "攻击", "冒犯", "歧视", "种族主义", "极端", "有害", "辱骂",
]
SCAN_FIELDS = [f for f in FIELDS if f != "on_screen_text"]


# ------------------------------------------------------------------ infra
def key():
    return open(os.path.expanduser("~/.dashscope_api_key")).read().strip()


def client():
    from openai import OpenAI
    return OpenAI(api_key=key(), base_url=BASE_URL, timeout=3600.0, max_retries=2)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ inputs
def load_ocr():
    o = {}
    for fn in ("ocr_video.jsonl", "ocr_video_test.jsonl"):
        p = os.path.join(ROOT, "data", "OCR", "HateMM", fn)
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                seen, keep = OrderedDict(), []
                for ln in (r.get("text") or "").split("\n"):
                    ln = ln.strip()
                    if ln and ln not in seen:
                        seen[ln] = 1
                        keep.append(ln)
                o[r["video_id"]] = " | ".join(keep)
    return o


_frame_cache = {}


def frame_urls(vid):
    d = os.path.join(ROOT, "data", "lora_frames", "HateMM", vid)
    fs = sorted(os.listdir(d), key=lambda x: int(re.findall(r"\d+", x)[0]))
    if N_FRAMES < len(fs):
        step = len(fs) / N_FRAMES
        fs = [fs[int(i * step)] for i in range(N_FRAMES)]
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


def build_messages(vid, ocrs, prompt_key):
    urls = frame_urls(vid)
    frames_note = ("%d evenly spaced frames of the video are attached above, in temporal "
                   "order." % len(urls))
    txt = ocrs.get(vid) or ""
    extra = ""
    if txt.strip():
        extra = ("On-screen text recovered by OCR (deduplicated, may be noisy):\n<<<\n"
                 + txt[:4000] + "\n>>>\n\n")
    body = BANK[prompt_key].format(frames_note=frames_note, extra=extra)
    content = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    content.append({"type": "text", "text": body})
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": content}], len(urls)


def all_ids():
    gt = load_gt(ROOT)
    return sorted(gt), gt


# ------------------------------------------------------------------ parsing / rules
def parse_json(txt):
    if txt is None:
        return None, "empty"
    t = re.sub(r"^```(?:json)?", "", txt.strip()).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None, "no_json"
    try:
        o = json.loads(m.group(0))
    except Exception as e:
        return None, "bad_json:" + str(e)[:60]
    if not isinstance(o, dict):
        return None, "not_dict"
    return {k: ("" if o.get(k) is None else str(o.get(k))) for k in FIELDS}, "ok"


def violations(fields):
    """-> list of field names containing a banned term (on_screen_text exempt)."""
    if not fields:
        return []
    bad = []
    for f in SCAN_FIELDS:
        low = (fields.get(f) or "").lower()
        if any(b in low for b in BANNED):
            bad.append(f)
    return bad


def load_done():
    done = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done[r["id"]] = r
    return done


_wlock = threading.Lock()


def append(rec):
    with _wlock:
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_rec(vid, fields, parse, raw, usage, model, prompt, regenerated=0):
    v = violations(fields)
    return {"id": vid, "fields": fields, "parse": parse, "violations": v,
            "regenerated": regenerated, "raw_len": len(raw or ""),
            "usage": usage, "model": model, "prompt": prompt,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}


# ------------------------------------------------------------------ commands
SMOKE_IDS = [
    # 2 strictly empty-transcript videos (frozen requirement)
    "hate_video_1", "non_hate_video_119",
    # 2 more DEFECT videos (near-empty / garbled ASR)
    "non_hate_video_318", "hate_video_383",
    # 4 ordinary videos with usable transcripts
    "hate_video_104", "non_hate_video_30", "hate_video_100", "non_hate_video_590",
]


def cmd_size(a):
    ids, _ = all_ids()
    ocrs = load_ocr()
    tot = 0
    n = 25
    for vid in ids[:n]:
        msgs, _ = build_messages(vid, ocrs, a.prompt)
        tot += len(json.dumps({"custom_id": vid, "body": {"messages": msgs}},
                              ensure_ascii=False).encode())
    log("mean request bytes %.0f  -> full 1066-item batch file ≈ %.1f MB"
        % (tot / n, tot / n * len(ids) / 1e6))


def _one_realtime(cli, vid, ocrs, prompt, model, max_tokens=None):
    msgs, nf = build_messages(vid, ocrs, prompt)
    r = cli.chat.completions.create(model=model, messages=msgs,
                                    max_tokens=max_tokens or MAX_TOKENS,
                                    temperature=0.0, seed=SEED)
    txt = r.choices[0].message.content
    fields, st = parse_json(txt)
    return make_rec(vid, fields, st, txt, r.usage.model_dump(), model, prompt), txt


def cmd_smoke(a):
    cli = client()
    ocrs = load_ocr()
    out = []
    for vid in SMOKE_IDS:
        t0 = time.time()
        try:
            rec, txt = _one_realtime(cli, vid, ocrs, a.prompt, a.model)
        except Exception as e:
            log(vid, "ERROR", type(e).__name__, str(e)[:200])
            continue
        rec["secs"] = round(time.time() - t0, 1)
        out.append(rec)
        log(vid, rec["parse"], "viol=", rec["violations"], "tok=",
            rec["usage"]["prompt_tokens"], "/", rec["usage"]["completion_tokens"])
    p = os.path.join(HERE, "smoke_%s.jsonl" % a.tag)
    with open(p, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log("wrote", p)


def cmd_realtime(a):
    """Concurrent realtime generation for every id not already in OUT."""
    from concurrent.futures import ThreadPoolExecutor
    ids, _ = all_ids()
    done = load_done()
    todo = [v for v in ids if v not in done]
    log("realtime: %d todo / %d total (%d already done)" % (len(todo), len(ids), len(done)))
    ocrs = load_ocr()
    cli = client()
    state = {"ok": 0, "err": 0, "in": 0, "out": 0}

    def work(vid):
        for attempt in range(3):
            try:
                rec, _ = _one_realtime(cli, vid, ocrs, a.prompt, a.model)
                append(rec)
                with _wlock:
                    state["ok"] += 1
                    state["in"] += rec["usage"]["prompt_tokens"]
                    state["out"] += rec["usage"]["completion_tokens"]
                return
            except Exception as e:
                msg = "%s: %s" % (type(e).__name__, str(e)[:160])
                if "DataInspectionFailed" in msg or "inappropriate" in msg:
                    append(make_rec(vid, None, "moderation_refused", "", None,
                                    a.model, a.prompt))
                    with _wlock:
                        state["err"] += 1
                    log(vid, "MODERATION_REFUSED")
                    return
                if attempt == 2:
                    append(make_rec(vid, None, "error:" + msg, "", None, a.model, a.prompt))
                    with _wlock:
                        state["err"] += 1
                    log(vid, "FAILED", msg)
                    return
                time.sleep(3 * (attempt + 1))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, v) for v in todo]
        last = 0
        while any(not f.done() for f in futs):
            time.sleep(20)
            d = state["ok"] + state["err"]
            if d != last:
                el = time.time() - t0
                log("PROGRESS %d/%d ok=%d err=%d elapsed=%ds eta=%ds tok_in=%d tok_out=%d"
                    % (d, len(todo), state["ok"], state["err"], el,
                       (len(todo) - d) * el / max(d, 1), state["in"], state["out"]))
                last = d
    log("DONE ok=%d err=%d tok_in=%d tok_out=%d elapsed=%ds"
        % (state["ok"], state["err"], state["in"], state["out"], time.time() - t0))


def cmd_submit(a):
    cli = client()
    ids, _ = all_ids()
    done = load_done()
    todo = [v for v in ids if v not in done]
    ocrs = load_ocr()
    shards = [todo[i:i + a.shard] for i in range(0, len(todo), a.shard)]
    meta = {"model": a.model, "prompt": a.prompt, "shards": [],
            "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for si, sh in enumerate(shards):
        p_in = os.path.join(HERE, "batch_in_%d.jsonl" % si)
        with open(p_in, "w", encoding="utf-8") as f:
            for vid in sh:
                msgs, _ = build_messages(vid, ocrs, a.prompt)
                f.write(json.dumps({
                    "custom_id": vid, "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": a.model, "messages": msgs, "max_tokens": MAX_TOKENS,
                             "temperature": 0.0, "seed": SEED}}, ensure_ascii=False) + "\n")
        log("shard %d: %s %.1f MB %d requests" % (si, p_in, os.path.getsize(p_in) / 1e6, len(sh)))
        fo = cli.files.create(file=open(p_in, "rb"), purpose="batch")
        b = cli.batches.create(input_file_id=fo.id, endpoint="/v1/chat/completions",
                               completion_window="24h")
        meta["shards"].append({"i": si, "n": len(sh), "file_id": fo.id, "batch_id": b.id})
        log("shard %d submitted batch=%s status=%s" % (si, b.id, b.status))
        json.dump(meta, open(os.path.join(HERE, "batch_meta.json"), "w"), indent=1)


def cmd_poll(a):
    cli = client()
    meta = json.load(open(os.path.join(HERE, "batch_meta.json")))
    while True:
        allo = True
        for s in meta["shards"]:
            b = cli.batches.retrieve(s["batch_id"])
            s["status"] = b.status
            s["output_file_id"] = b.output_file_id
            s["error_file_id"] = b.error_file_id
            log("shard", s["i"], b.status, b.request_counts)
            if b.status not in ("completed", "failed", "expired", "cancelled"):
                allo = False
        json.dump(meta, open(os.path.join(HERE, "batch_meta.json"), "w"), indent=1)
        if allo:
            return
        time.sleep(60)


def cmd_fetch(a):
    cli = client()
    meta = json.load(open(os.path.join(HERE, "batch_meta.json")))
    done = load_done()
    n_new = 0
    for s in meta["shards"]:
        b = cli.batches.retrieve(s["batch_id"])
        for kind, fid in (("out", b.output_file_id), ("err", b.error_file_id)):
            if not fid:
                continue
            raw = cli.files.content(fid).content.decode()
            open(os.path.join(HERE, "batch_%s_%d.jsonl" % (kind, s["i"])), "w").write(raw)
            if kind != "out":
                log("shard", s["i"], "errors", len(raw.splitlines()))
                continue
            for line in raw.splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                vid = r["custom_id"]
                if vid in done:
                    continue
                body = (r.get("response") or {}).get("body") or {}
                ch = (body.get("choices") or [{}])[0]
                txt = (ch.get("message") or {}).get("content")
                fields, st = parse_json(txt)
                append(make_rec(vid, fields, st, txt, body.get("usage"),
                                meta["model"], meta["prompt"]))
                done[vid] = 1
                n_new += 1
    log("fetched %d new rows" % n_new)


def cmd_repair(a):
    """One regeneration pass over rows with a violation or a failed parse (frozen: once)."""
    cli = client()
    ocrs = load_ocr()
    done = load_done()
    todo = [v for v, r in done.items()
            if r.get("regenerated", 0) == 0
            and (r.get("violations") or r.get("parse") != "ok")
            and r.get("parse") != "moderation_refused"]
    log("repair: %d rows to regenerate once" % len(todo))
    from concurrent.futures import ThreadPoolExecutor
    rows = []

    def one(vid):
        try:
            rec, _ = _one_realtime(cli, vid, ocrs, done[vid].get("prompt", a.prompt),
                                   a.model, max_tokens=a.max_tokens)
        except Exception as e:
            log(vid, "repair ERROR", type(e).__name__, str(e)[:150])
            return
        rec["regenerated"] = 1
        with _wlock:
            rows.append(rec)
        log(vid, "->", rec["parse"], rec["violations"])

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, todo))
    # rewrite the file with repaired rows replacing the originals
    for r in rows:
        prev = done[r["id"]]
        if r["parse"] != "ok":
            r = prev  # keep the original if the retry did not even parse
            r["regenerated"] = 1
        elif r["violations"]:
            for f in r["violations"]:
                r["fields"][f] = ""
            r["blanked_fields"] = list(r["violations"])
        done[r["id"]] = r
    with open(OUT, "w", encoding="utf-8") as f:
        for vid in sorted(done):
            f.write(json.dumps(done[vid], ensure_ascii=False) + "\n")
    log("repair pass written")


def cmd_report(a):
    ids, gt = all_ids()
    done = load_done()
    miss = [v for v in ids if v not in done]
    ok = [r for r in done.values() if r["parse"] == "ok"]
    viol1 = [r for r in done.values() if r.get("violations")]
    blank = [r for r in done.values() if r.get("blanked_fields")]
    ti = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in done.values())
    to = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in done.values())
    log("rows=%d missing=%d parse_ok=%d still_violating=%d blanked=%d"
        % (len(done), len(miss), len(ok), len(viol1), len(blank)))
    log("tokens in=%d out=%d" % (ti, to))
    if miss:
        log("missing ids (first 10):", miss[:10])
    for r in done.values():
        if r["parse"] not in ("ok",):
            log("non-ok:", r["id"], r["parse"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "size", "submit", "poll", "fetch",
                                    "realtime", "repair", "report"])
    ap.add_argument("--model", default="qwen3-vl-plus")
    ap.add_argument("--prompt", default="V2")
    ap.add_argument("--tag", default="t")
    ap.add_argument("--shard", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
