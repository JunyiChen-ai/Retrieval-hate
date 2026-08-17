"""STANCE PILOT -- pre-registered fallback step 2 (freeze doc s7 item 2): task decomposition.

Stage A: ask ONLY whether identity-based hateful material is present, who utters it, and a short
         quote of it.  (no stance question at all)
Stage B: conditioned on stage A's answer, ask ONLY the stance question.  Items where stage A
         says no hateful surface skip stage B and are recorded as no_hate_content.

Deviation logged in the result doc: this fallback runs on the REALTIME endpoint with 8 parallel
workers, not the Batch API. Reason: each batch submission costs ~25 min of upload on this link
(17 MB of base64 frames) and the fallback needs two passes; at full realtime price the whole
fallback is well under USD 1, so the 50 % batch discount is not worth ~1 h of wall clock.

The stage-B prompt keeps the class definitions of V1.3 verbatim but drops V1.3's
"DEFAULT RULE" line, which was a Q1->Q3 coupling ("no distancing => endorses"). That coupling
cannot survive decomposition by construction: in stage B the Q1 answer is already given, so the
default rule would be the only instruction left and would force the collapse it is meant to be
tested for.
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_pilot import (build_messages, client, iter_items, load_ocr, load_texts,  # noqa: E402
                       parse_json)
from prompts import SYSTEM  # noqa: E402

STAGE_A = """VIDEO EVIDENCE
{frames_note}
Speech transcript (full automatic transcript{title_note}; it may be in English or Chinese, and may be noisy or empty):
<<<
{transcript}
>>>
{extra}
ONE QUESTION ONLY. Do not judge anybody's intent. Do not reason out loud. Emit only the JSON object.

Q1 hate_surface_present -- Does the video contain ANY material that attacks, demeans, \
dehumanises or slurs a person or group **because of a group identity** -- race, ethnicity, \
nationality, immigration status, religion, gender, sexual orientation, gender identity, \
disability, caste or age -- REGARDLESS of who utters it and regardless of why it is shown? \
true or false.
  Answer FALSE for: ordinary profanity, personal insults or threats with no group-identity basis, \
political or institutional criticism, sexual content, violence, rudeness or general negativity.
  Answer TRUE for: identity-based slurs, dehumanising comparisons, stereotyping attacks, \
calls for exclusion or harm against an identity group -- including when they are quoted, \
subtitled, archived, or shown in order to criticise them. Gendered insults and \
gender-stereotype epithets aimed at a person (e.g. calling a woman a "bitch", or labelling \
someone with a mocking gender stereotype) DO count as identity-based.

Q2 primary_voice -- If Q1 is true, whose mouth or text does that material come from? Exactly one of:
  uploader | on_screen_speaker | quoted_third_party | archival_source | caption_overlay | none

Q3 surface -- If Q1 is true, quote at most 25 words of the material itself. Else null.

OUTPUT: {{"hate_surface_present": true|false, "primary_voice": "<one of the six>", \
"surface": "<=25 words"|null}}"""

STAGE_B = """VIDEO EVIDENCE
{frames_note}
Speech transcript (full automatic transcript{title_note}; it may be in English or Chinese, and may be noisy or empty):
<<<
{transcript}
>>>
{extra}
ALREADY ESTABLISHED by a previous pass (take these as given, do not re-litigate them):
  - identity-based hateful material IS present in this video
  - it is uttered by: {voice}
  - the material is: {surface}

ONE QUESTION ONLY. Do not reason out loud. Emit only the JSON object.

stance -- What is the relation of THIS VIDEO'S OWN AUTHORIAL VOICE (the creator/uploader, i.e. \
the party responsible for making this video and posting it) to that established material? \
Note that the creator and the person who utters the material need not be the same party. \
Exactly one of:
  endorses         = the creator asserts, promotes, incites, jokes along with, or deploys it \
against the target; includes hateful lyrics performed or used as the video's own soundtrack, and \
hateful material posted as entertainment with no critical, educational or journalistic framing
  quotes_mentions  = the material is attributed to someone else and is reproduced or displayed \
while the creator visibly takes no side (use-vs-mention, screenshotted quote, unlabelled \
third-party clip)
  condemns         = the creator criticises, refutes, exposes, mocks the bigot, counter-speaks, \
gives an educational takedown, or satirises the hater
  reports          = the creator presents it as neutral news, documentary or archival material

target -- the group or person attacked, short noun phrase, or null.
evidence -- ONE sentence, at most 25 words, naming the decisive cue about the CREATOR's position.

OUTPUT: {{"stance": "<one of the four>", "target": "<short phrase>"|null, "evidence": "<one sentence>"}}"""

LOCK = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-plus")
    ap.add_argument("--tag", default="fb2")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cli = client()
    items = iter_items("eval")
    cache_t, cache_o = {}, {}
    for it in items:
        ds = it["dataset"]
        if ds not in cache_t:
            cache_t[ds], cache_o[ds] = load_texts(ds), load_ocr(ds)

    def ask(it, template, fmt):
        msgs, nf = build_messages(it, cache_t[it["dataset"]], cache_o[it["dataset"]],
                                  "V1", a.frames, False)
        # rebuild the text block from the requested template, keep the image parts
        ds = it["dataset"]
        frames_note = ("%d evenly spaced frames of the video are attached above, in temporal "
                       "order." % nf) if nf else ("No video frames are available for this item; "
                                                  "judge from the transcript alone.")
        title_note = ("; its first sentence, before the first ' . ', is the video TITLE"
                      if ds in ("MHC", "MHC_zh") else "")
        body = template.format(frames_note=frames_note, title_note=title_note,
                               transcript=(cache_t[ds].get(it["id"]) or "").strip() or "(empty)",
                               extra="", **fmt)
        msgs[1]["content"][-1]["text"] = body
        r = cli.chat.completions.create(model=a.model, messages=msgs, max_tokens=400,
                                        temperature=0.0, seed=20260811)
        obj, st = parse_json(r.choices[0].message.content)
        return obj, st, r.usage.model_dump(), r.choices[0].message.content, nf

    def work(it):
        try:
            A, stA, uA, rawA, nf = ask(it, STAGE_A, {})
        except Exception as e:
            return {"dataset": it["dataset"], "id": it["id"], "parse": "err_A:" + str(e)[:80],
                    "parsed": None, "raw": None, "usage": None}
        rec = {"dataset": it["dataset"], "id": it["id"], "n_frames_sent": nf,
               "stageA": A, "parseA": stA, "rawA": rawA, "usage": uA}
        if not A or not A.get("hate_surface_present"):
            rec.update({"parse": stA if A else "no_json_A",
                        "parsed": {"hate_surface_present": False, "primary_voice": "none",
                                   "stance": "no_hate_content", "target": None,
                                   "evidence": (A or {}).get("surface")}})
            return rec
        try:
            B, stB, uB, rawB, _ = ask(it, STAGE_B,
                                      {"voice": A.get("primary_voice"),
                                       "surface": json.dumps(A.get("surface"),
                                                             ensure_ascii=False)})
        except Exception as e:
            rec.update({"parse": "err_B:" + str(e)[:80], "parsed": None})
            return rec
        rec["stageB"] = B
        rec["rawB"] = rawB
        rec["usage"] = {k: (uA.get(k, 0) or 0) + (uB.get(k, 0) or 0)
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        rec.update({"parse": stB, "parsed": None if not B else {
            "hate_surface_present": True, "primary_voice": A.get("primary_voice"),
            "stance": B.get("stance"), "target": B.get("target"),
            "evidence": B.get("evidence")}})
        return rec

    out = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rec in enumerate(ex.map(work, items)):
            out.append(rec)
            with LOCK:
                print(i, rec["id"], rec.get("parse"),
                      (rec.get("parsed") or {}).get("stance"), flush=True)
    p = os.path.join(HERE, f"pred_{a.tag}.jsonl")
    with open(p, "w") as f:
        for r in out:
            r2 = dict(r)
            r2["model"] = a.model
            r2["prompt"] = "FB2_decomposed"
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    print("wrote", p, len(out))


if __name__ == "__main__":
    main()
