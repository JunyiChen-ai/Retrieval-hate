"""ARBITER step 2 -- binary hateful/not-hateful judgement from qwen3-vl-plus on the
uncertainty band.

  python mllm_judge.py smoke                 # <=8 TRAIN videos, prompt wording only
  python mllm_judge.py run [--workers 8]     # the frozen band union, one call per video
  python mllm_judge.py report

Inputs per video: the 8 evenly spaced frames already cached in data/lora_frames/HateMM/<id>
(resized to <=512 px, JPEG q80) plus the complete transcript from data/gt/HateMM/*.jsonl.
Output: one JSON object {"hateful": true|false, "confidence": 0..1}.

judgements.jsonl is append-only and keyed by id; any id already present is never
re-requested.  The API key is read from ~/.dashscope_api_key and never written out.
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

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "..", "desc_channel"))
from defect import load_gt  # noqa: E402

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-vl-plus"
MAX_SIDE = 512
JPEG_Q = 80
N_FRAMES = 8
MAX_TOKENS = 200
TEMPERATURE = 0.0
SEED = 20260813
OUT = os.path.join(HERE, "judgements.jsonl")
BAND = os.path.join(HERE, "band_union.json")

SYSTEM = (
    "You are a content-moderation classifier for a hateful-video research benchmark. "
    "You always reply with a single JSON object and nothing else."
)

PROMPT = """{n} evenly spaced frames of a video are attached above, in temporal order.

The complete speech transcript of the same video is:
<<<
{transcript}
>>>

Decide whether this video is HATEFUL.

HATEFUL means the video, taken as a whole, attacks, demeans, dehumanises, threatens, \
ridicules or incites hostility against a person or group because of a protected \
characteristic - race, ethnicity, national origin, religion, caste, immigration status, \
sex, gender, gender identity, sexual orientation or disability. It includes slurs used as \
attacks, dehumanising comparisons, hateful stereotyping, calls for exclusion or violence, \
and celebration or endorsement of such attacks or of hate groups.

NOT HATEFUL covers everything else, including: violence, crime, accidents, insults or \
profanity with no protected-characteristic target; political, religious or social \
criticism that does not attack people for a protected characteristic; news reporting, \
documentary or educational coverage of hate; counter-speech, satire or commentary that \
quotes hateful material in order to condemn it.

Judge the video as a whole, using both what you see in the frames and what is said in the \
transcript. If the transcript is empty or unhelpful, judge from the frames alone.

Reply with exactly this JSON object and nothing else:
{{"hateful": true or false, "confidence": a number between 0 and 1}}

"confidence" is how certain you are about the label you just gave, where 0.5 means a coin \
flip and 1.0 means completely certain. No markdown, no code fences, no explanation."""


def key():
    return open(os.path.expanduser("~/.dashscope_api_key")).read().strip()


def client():
    from openai import OpenAI
    return OpenAI(api_key=key(), base_url=BASE_URL, timeout=600.0, max_retries=2)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


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


def build_messages(vid, gt):
    urls = frame_urls(vid)
    tr = (gt[vid]["text"] or "").strip()
    body = PROMPT.format(n=len(urls), transcript=tr if tr else "(no speech transcribed)")
    content = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    content.append({"type": "text", "text": body})
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": content}]


def parse_json(txt):
    """-> ({'hateful':bool,'confidence':float}, status)"""
    if not txt:
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
    if not isinstance(o, dict) or "hateful" not in o or "confidence" not in o:
        return None, "missing_keys"
    h = o["hateful"]
    if isinstance(h, str):
        if h.strip().lower() in ("true", "yes"):
            h = True
        elif h.strip().lower() in ("false", "no"):
            h = False
        else:
            return None, "bad_hateful"
    if not isinstance(h, bool):
        return None, "bad_hateful"
    try:
        c = float(o["confidence"])
    except Exception:
        return None, "bad_confidence"
    if not (0.0 <= c <= 1.0):
        return None, "bad_confidence_range"
    return {"hateful": bool(h), "confidence": c}, "ok"


_wlock = threading.Lock()


def append(rec):
    with _wlock:
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_done():
    done = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[r["id"]] = r
    return done


def call_one(cli, vid, gt, model):
    msgs = build_messages(vid, gt)
    r = cli.chat.completions.create(
        model=model, messages=msgs, max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE, seed=SEED,
        extra_body={"enable_thinking": False})
    txt = r.choices[0].message.content
    j, st = parse_json(txt)
    return {"id": vid, "judgement": j, "parse": st, "raw": (txt or "")[:400],
            "usage": r.usage.model_dump(), "model": model,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, st


# 8 TRAIN-split videos for prompt smoke only.  Never enters any metric.
SMOKE_IDS = ["hate_video_2", "hate_video_11", "hate_video_200", "hate_video_400",
             "non_hate_video_5", "non_hate_video_44", "non_hate_video_300",
             "non_hate_video_500"]


def cmd_smoke(a):
    gt = load_gt(ROOT)
    cli = client()
    ok = 0
    for vid in SMOKE_IDS:
        if gt[vid]["split"] != "train":
            log(vid, "NOT TRAIN -- skipped", gt[vid]["split"])
            continue
        try:
            rec, st = call_one(cli, vid, gt, a.model)
        except Exception as e:
            log(vid, "ERROR", type(e).__name__, str(e)[:200])
            continue
        ok += st == "ok"
        log(vid, "gold=%d" % gt[vid]["label"], st, rec["judgement"],
            "tok=%d/%d" % (rec["usage"]["prompt_tokens"], rec["usage"]["completion_tokens"]),
            "| raw:", (rec["raw"] or "").replace("\n", " ")[:120])
    log("smoke parsed ok: %d" % ok)


def cmd_run(a):
    from concurrent.futures import ThreadPoolExecutor
    ids = json.load(open(BAND))
    gt = load_gt(ROOT)
    done = load_done()
    todo = [v for v in ids if v not in done]
    log("run: %d todo / %d band (%d already done)" % (len(todo), len(ids), len(done)))
    cli = client()
    state = {"ok": 0, "err": 0, "refused": 0, "in": 0, "out": 0, "retry": 0}

    def work(vid):
        # network errors: up to 3 attempts.  a response that arrives but fails to parse:
        # exactly one identical retry (frozen).
        parse_retry_used = False
        for attempt in range(3):
            try:
                rec, st = call_one(cli, vid, gt, a.model)
            except Exception as e:
                msg = "%s: %s" % (type(e).__name__, str(e)[:180])
                if "DataInspectionFailed" in msg or "inappropriate" in msg \
                        or "data_inspection_failed" in msg:
                    append({"id": vid, "judgement": None, "parse": "moderation_refused",
                            "raw": "", "usage": None, "model": a.model,
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
                    with _wlock:
                        state["refused"] += 1
                    log(vid, "MODERATION_REFUSED")
                    return
                if attempt == 2:
                    append({"id": vid, "judgement": None, "parse": "error:" + msg,
                            "raw": "", "usage": None, "model": a.model,
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
                    with _wlock:
                        state["err"] += 1
                    log(vid, "FAILED", msg)
                    return
                time.sleep(3 * (attempt + 1))
                continue
            with _wlock:
                state["in"] += rec["usage"]["prompt_tokens"]
                state["out"] += rec["usage"]["completion_tokens"]
            if st != "ok" and not parse_retry_used:
                parse_retry_used = True
                with _wlock:
                    state["retry"] += 1
                log(vid, "PARSE_FAIL", st, "-- one identical retry")
                continue
            append(rec)
            with _wlock:
                state["ok" if st == "ok" else "err"] += 1
            return

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, v) for v in todo]
        last = -1
        while any(not f.done() for f in futs):
            time.sleep(15)
            d = state["ok"] + state["err"] + state["refused"]
            if d != last:
                el = time.time() - t0
                log("PROGRESS %d/%d ok=%d err=%d refused=%d retry=%d elapsed=%ds "
                    "eta=%ds tok_in=%d tok_out=%d"
                    % (d, len(todo), state["ok"], state["err"], state["refused"],
                       state["retry"], el, (len(todo) - d) * el / max(d, 1),
                       state["in"], state["out"]))
                last = d
    log("DONE ok=%d err=%d refused=%d retry=%d tok_in=%d tok_out=%d elapsed=%ds"
        % (state["ok"], state["err"], state["refused"], state["retry"],
           state["in"], state["out"], time.time() - t0))


def cmd_report(a):
    done = load_done()
    ids = json.load(open(BAND))
    import collections
    c = collections.Counter(r["parse"] for r in done.values())
    ti = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in done.values())
    to = sum((r["usage"] or {}).get("completion_tokens", 0) for r in done.values())
    print(json.dumps({"band": len(ids), "have": len(done), "parse": dict(c),
                      "tok_in": ti, "tok_out": to,
                      "cost_cny_at_0.002_0.008": round(ti / 1000 * 0.002
                                                       + to / 1000 * 0.008, 4)}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "run", "report"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"smoke": cmd_smoke, "run": cmd_run, "report": cmd_report}[a.cmd](a)
