"""CAD step 1 -- counterfactual (de-hating) rewrites of the HateMM TRAIN-split hate
transcripts.

Spec frozen in idea-stage/CAD_FREEZE.md sections 2/3.

  python cadgen.py smoke   [--prompt V1] [--tag v1]
  python cadgen.py realtime [--workers 8]      # full eligible set, idempotent
  python cadgen.py report

Idempotent: rewrites_train_hate.jsonl is append-only and keyed by video id; any id
already present is never re-requested.  The API key is read from ~/.dashscope_api_key
and is never written to any file, log or report.

Direction is one-way by construction: the model is only ever asked to REMOVE identity
attacks from an existing transcript.  It is never asked to produce hateful content.
"""
import argparse
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "idea-stage", "desc_channel"))
from defect import load_gt  # noqa: E402
from prompts_cad import BANK, SYSTEM  # noqa: E402

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"
SEED = 20260813
MAX_TOKENS = 4096
MIN_CHARS = 40                       # G0 eligibility gate (frozen)
OUT = os.path.join(HERE, "rewrites_train_hate.jsonl")

_wlock = threading.Lock()


def key():
    return open(os.path.expanduser("~/.dashscope_api_key")).read().strip()


def client():
    from openai import OpenAI
    return OpenAI(api_key=key(), base_url=BASE_URL, timeout=600.0, max_retries=2)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ target set
def targets(root=ROOT):
    """-> (eligible ids sorted, skipped_short ids sorted, gt)

    Frozen target set: every TRAIN-split video with label == 1.
    G0: a transcript with < MIN_CHARS non-whitespace characters carries nothing to
    minimally edit and is skipped before any API call.
    """
    gt = load_gt(root)
    hate = sorted(v for v, r in gt.items() if r["split"] == "train" and r["label"] == 1)
    elig = [v for v in hate if len((gt[v]["text"] or "").strip()) >= MIN_CHARS]
    short = [v for v in hate if v not in set(elig)]
    return elig, short, gt


# ------------------------------------------------------------------ parsing
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
    if not isinstance(o, dict) or "rewritten" not in o:
        return None, "no_rewritten_key"
    rw = o.get("rewritten")
    if rw is None:
        return None, "null_rewritten"
    # defensive: some responses echo the <<< >>> transcript markers back
    rw = re.sub(r"^\s*<<<\s*", "", str(rw))
    rw = re.sub(r"\s*>>>\s*$", "", rw)
    try:
        ne = int(o.get("n_edits", -1))
    except Exception:
        ne = -1
    return {"rewritten": str(rw), "n_edits": ne}, "ok"


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


def append(rec):
    with _wlock:
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_rec(vid, obj, parse, raw, usage, prompt):
    return {"id": vid,
            "rewritten": (obj or {}).get("rewritten"),
            "n_edits": (obj or {}).get("n_edits", -1),
            "parse": parse, "raw_len": len(raw or ""),
            "usage": usage, "model": MODEL, "prompt": prompt,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}


def build_messages(transcript, prompt_key):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": BANK[prompt_key].format(transcript=transcript)}]


def _one(cli, vid, gt, prompt):
    msgs = build_messages(gt[vid]["text"].strip(), prompt)
    r = cli.chat.completions.create(model=MODEL, messages=msgs, max_tokens=MAX_TOKENS,
                                    temperature=0.0, seed=SEED)
    txt = r.choices[0].message.content
    obj, st = parse_json(txt)
    return make_rec(vid, obj, st, txt, r.usage.model_dump(), prompt), txt


# ------------------------------------------------------------------ commands
def cmd_smoke(a):
    """Prompt iteration only, on <= 8 TRAIN videos. Writes smoke_<tag>.jsonl, which is
    NOT an input to anything downstream: the frozen run regenerates every id."""
    elig, short, gt = targets()
    # deterministic spread over the transcript-length distribution
    order = sorted(elig, key=lambda v: len(gt[v]["text"].strip()))
    idx = [int(i * (len(order) - 1) / 7) for i in range(8)]
    ids = [order[i] for i in idx][: a.limit or 8]
    cli = client()
    rows = []
    for vid in ids:
        t0 = time.time()
        try:
            rec, _ = _one(cli, vid, gt, a.prompt)
        except Exception as e:
            log(vid, "ERROR", type(e).__name__, str(e)[:200])
            continue
        rec["secs"] = round(time.time() - t0, 1)
        o = len(gt[vid]["text"].strip())
        n = len((rec["rewritten"] or "").strip())
        rec["orig_chars"], rec["new_chars"] = o, n
        rows.append(rec)
        log(vid, rec["parse"], "n_edits=", rec["n_edits"],
            "chars %d->%d ratio %.2f" % (o, n, n / max(o, 1)),
            "tok=%d/%d" % (rec["usage"]["prompt_tokens"],
                           rec["usage"]["completion_tokens"]))
    p = os.path.join(HERE, "smoke_%s.jsonl" % a.tag)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log("wrote", p)


def cmd_realtime(a):
    from concurrent.futures import ThreadPoolExecutor
    elig, short, gt = targets()
    done = load_done()
    todo = [v for v in elig if v not in done]
    log("targets: %d train-hate eligible (%d skipped by G0 short), %d already done, "
        "%d todo" % (len(elig), len(short), len(done), len(todo)))
    if not todo:
        log("nothing to do")
        return
    cli = client()
    state = {"ok": 0, "err": 0, "in": 0, "out": 0}

    def work(vid):
        for attempt in range(3):
            try:
                rec, _ = _one(cli, vid, gt, a.prompt)
                append(rec)
                with _wlock:
                    state["ok"] += 1
                    state["in"] += rec["usage"]["prompt_tokens"]
                    state["out"] += rec["usage"]["completion_tokens"]
                return
            except Exception as e:
                msg = "%s: %s" % (type(e).__name__, str(e)[:160])
                if ("DataInspectionFailed" in msg or "data_inspection_failed" in msg
                        or "inappropriate" in msg):
                    append(make_rec(vid, None, "moderation_refused", "", None, a.prompt))
                    with _wlock:
                        state["err"] += 1
                    log(vid, "MODERATION_REFUSED")
                    return
                if attempt == 2:
                    append(make_rec(vid, None, "error:" + msg, "", None, a.prompt))
                    with _wlock:
                        state["err"] += 1
                    log(vid, "FAILED", msg)
                    return
                time.sleep(3 * (attempt + 1))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, v) for v in todo]
        last = -1
        while any(not f.done() for f in futs):
            time.sleep(15)
            d = state["ok"] + state["err"]
            if d != last:
                el = time.time() - t0
                log("PROGRESS %d/%d ok=%d err=%d elapsed=%ds eta=%ds tok_in=%d tok_out=%d"
                    % (d, len(todo), state["ok"], state["err"], el,
                       (len(todo) - d) * el / max(d, 1), state["in"], state["out"]))
                last = d
    log("DONE ok=%d err=%d tok_in=%d tok_out=%d elapsed=%ds"
        % (state["ok"], state["err"], state["in"], state["out"], time.time() - t0))


def cmd_report(a):
    elig, short, gt = targets()
    done = load_done()
    miss = [v for v in elig if v not in done]
    ok = [r for r in done.values() if r["parse"] == "ok"]
    ref = [r for r in done.values() if r["parse"] == "moderation_refused"]
    err = [r for r in done.values()
           if r["parse"] not in ("ok", "moderation_refused")]
    ti = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in done.values())
    to = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in done.values())
    log("eligible=%d skipped_short=%d rows=%d missing=%d parse_ok=%d refused=%d other_err=%d"
        % (len(elig), len(short), len(done), len(miss), len(ok), len(ref), len(err)))
    # qwen-plus list price (CNY / 1k tokens), 2026-08 tiered entry rate
    log("tokens in=%d out=%d  approx cost CNY %.3f"
        % (ti, to, ti / 1000 * 0.0008 + to / 1000 * 0.002))
    for r in err:
        log("non-ok:", r["id"], r["parse"][:120])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "realtime", "report"])
    ap.add_argument("--prompt", default="V1")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
