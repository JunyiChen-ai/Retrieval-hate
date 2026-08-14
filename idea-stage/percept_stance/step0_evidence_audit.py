"""PERCEPT_STANCE step 0 -- offline audit of the round-1 `evidence` field.

Zero API calls, zero GPU. Question (frozen in the task spec before any code was written):

    For every item whose round-1 stance answer is WRONG in the binary reading, does the
    `evidence` sentence state a fact of the class that would have pointed to the CORRECT
    direction?

This is the direct test of the premise behind the perception-questionnaire design
("the bias lives only in the judgement step; the perception / transcription step is clean").

Coding rule R-STRICT (this is the criterion named in the task spec, incl. its parenthetical
"evidence 是否写出了 criticizes / reports / 引用 类事实"):

  gold = OPPOSE (group S_FP)  -> YES iff the evidence sentence explicitly states at least one
      fact of the ATTRIBUTION / QUOTATION / REPORTAGE / CRITICISM class:
        (a) it names a speaker/source other than the uploader as the utterer, or
        (b) it states the material is quoted / archival / broadcast / embedded from elsewhere, or
        (c) it states that someone in the video criticises, condemns, reports or disavows it, or
        (d) it states a fact showing the author holds a supportive stance toward the group.
      A clause of the form "with no criticism / attribution / distancing" is a CONTRARY fact and
      never scores YES, but its presence does not veto a supporting fact stated elsewhere in the
      same sentence.
      "lyrics" / "a song" / "a performance" alone does NOT count -- one can perform one's own song.

  gold = ENDORSE (groups S_FN, CTRL_HATE) -> YES iff the evidence sentence explicitly states a
      fact of the mirror class: the uploader / speaker asserts it in their own voice, first
      person, unattributed.

  CTRL_NONHATE (no gold direction; wrong = answered `endorses`) -> YES iff the evidence sentence
      states a fact showing the material is benign or is a mention rather than an assertion.

Secondary reading R-LOOSE, reported but NOT the gate: is the evidence sentence's description of
the *content* factually accurate (regardless of whether it points anywhere)?  Adjudicated against
the transcript in data/gt/<ds>/test.jsonl plus the hand-coded provenance notes in
idea-stage/voice_field_analysis.py::GOLD_VOICE.

R-VOICE: a third column recording whether the sibling `primary_voice` field (not the evidence
sentence) independently carries the direction-relevant fact.  Reported as a lenient variant.

Gate (frozen in the task spec before this file existed):
    S_FP ratio < 0.30  ->  premise does not hold  ->  STOP, spend nothing.

Codes below were written down by reading, for every wrong item, the evidence sentence, the
transcript, and the hand-coded GOLD_VOICE note.  Usage:  python step0_evidence_audit.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "idea-stage"))

GOLD = {"S_FP": "OPPOSE", "S_FN": "ENDORSE", "CTRL_HATE": "ENDORSE", "CTRL_NONHATE": None}
SMOKE = {("MHC", "KDcCiUU8q5E"), ("HateMM", "non_hate_video_32"), ("HateMM", "non_hate_video_16")}
FRAME_DS = {"HateMM", "MHC", "MHC_zh"}

# key -> (R_STRICT, R_LOOSE, R_VOICE, note)
#   R_STRICT : evidence sentence states a fact pointing to the correct direction
#   R_LOOSE  : evidence sentence's description of the content is factually accurate
#   R_VOICE  : the sibling primary_voice field carries the direction-relevant fact
CODES = {
    # ---------------- S_FP wrong (gold OPPOSE, answered `endorses`) : 18 items
    ("HateMM", "non_hate_video_121"): (True, True, False,
        "names 'Greg Anglin' as the utterer -> third-party attribution present (class a); the "
        "trailing 'presented without criticism or distancing' is false (it is a sting call)"),
    ("HateMM", "non_hate_video_16"): (False, True, False,
        "1956 segregationist newsreel; content described accurately, but zero archival / "
        "attribution fact and no reportage cue"),
    ("HateMM", "non_hate_video_149"): (False, True, True,
        "archival minstrel recording; evidence asserts 'without any critical framing or "
        "attribution' -- contrary fact. primary_voice='archival_source' IS correct -> R_VOICE"),
    ("HateMM", "non_hate_video_400"): (False, True, False,
        "Ellen-show broadcast clip; 'performing a song' is not attribution; no third party named"),
    ("HateMM", "non_hate_video_32"): (False, True, False,
        "Lennon/Ono song; evidence explicitly asserts 'no critical framing, attribution, or "
        "distancing by the uploader'"),
    ("HateMM", "non_hate_video_642"): (False, False, False,
        "third-party Kiffness music video; evidence says 'the uploader performs lyrics' -- the "
        "speaker attribution is wrong"),
    ("MHC", "8zLoOqXvk64"): (False, True, False,
        "title-only item; the title is quoted verbatim and correctly, but the fact points nowhere"),
    ("MHC", "YDEsYXYlB8o"): (False, True, False,
        "the transcript attributes the claim to 'a STEM professor ... published their findings'; "
        "the evidence sentence attributes it to the uploader instead"),
    ("MHC", "XlJCNPi5inM"): (False, True, False,
        "title names Kevin Samuels; evidence says only 'the speaker asserts' -- no attribution"),
    ("MHC", "ga1r2cweP80"): (False, True, False, "vlogger's own narration; no direction fact"),
    ("MHC", "pofgIFZpR7c"): (False, False, False,
        "title byline '- Hamza Ahmed'; evidence attributes the monologue to 'the uploader'"),
    ("MHC_zh", "BV1to4y177df"): (False, True, False, "90-char skit; no direction fact"),
    ("ImpliHateVid", "NH_44"): (False, True, False, "own monologue; no direction fact"),
    ("ImpliHateVid", "NH_394"): (False, True, False,
        "evidence explicitly asserts 'first-person ... without distancing or attribution'"),
    ("ImpliHateVid", "NH_988"): (True, True, False,
        "'self-identifies as supporting gay rights for Palestinians' -> class (d) supportive-stance "
        "fact stated correctly, then overridden by the stance field"),
    ("ImpliHateVid", "NH_887"): (False, True, False,
        "the pro-immigrant contrast is stated, but the alleged target is 'American people', so it "
        "is not a direction fact for the alleged attack"),
    ("ImpliHateVid", "NH_650"): (False, True, False,
        "scripted two-voice debate skit; evidence attributes everything to 'the uploader' as a "
        "single asserting voice and states no reaction / dialogue fact"),
    ("ImpliHateVid", "NH_875"): (False, False, False,
        "the hateful line is inside the embedded Starbucks parody being mocked; the evidence "
        "attributes it to the uploader and says 'without criticism or attribution'"),

    # ---------------- S_FN wrong (gold ENDORSE, answered non-`endorses`) : 9 items
    ("HateMM", "hate_video_365"): (False, True, False,
        "answered `condemns`; evidence states the reportage fact -- correct in itself but the "
        "opposite class to the gold direction"),
    ("MHC", "EEC98aHSgIY"): (True, True, False,
        "'the video presents biblical verses and commentary' -> own-voice fact, correct"),
    ("MHC", "_qldaPBgkk0"): (True, True, False,
        "'The speaker states ... and describes' -> own-voice assertion, unattributed, correct"),
    ("MHC_zh", "BV12G4y1S7mN"): (False, True, False,
        "film-recap of third-party drama; evidence says only 'the subtitles describe'"),
    ("MHC_zh", "BV1Vy4y1p7x2"): (False, True, False, "'the transcript contains' -- no voice fact"),
    ("MHC_zh", "BV1qZ4y1T71a"): (False, True, False,
        "'the transcript and frames show' -- no voice fact"),
    ("ImpliHateVid", "IM_176"): (False, True, False, "'the transcript discusses' -- no voice fact"),
    ("ImpliHateVid", "IM_308"): (False, True, False,
        "'the transcript reports student concerns' -- reportage class, opposite direction"),
    ("ImpliHateVid", "IM_57"): (True, True, False,
        "'The speaker criticizes churches ...' -> own-voice assertion, correct"),

    # ---------------- CTRL_HATE wrong (gold ENDORSE, answered `no_hate_content`) : 2 items
    ("MHC", "uPJtlBAOT_U"): (False, True, False, "no own-voice assertion fact stated"),
    ("ImpliHateVid", "IM_31"): (False, True, False, "no own-voice assertion fact stated"),

    # ---------------- CTRL_NONHATE wrong (answered `endorses`) : 4 items
    ("MHC", "RAGvPqMoW-4"): (False, True, False, "'uttered as the video's own framing' -- contrary"),
    ("ImpliHateVid", "NH_443"): (True, True, False,
        "'to refer to people targeted for depopulation ... part of a globalist agenda' -> states "
        "the term is the described agenda's, i.e. a mention"),
    ("ImpliHateVid", "NH_412"): (True, True, False,
        "verbatim quote of a platform-moderation complaint; states no attack"),
    ("ImpliHateVid", "NH_892"): (False, True, False, "asserted as the uploader's own claim"),
}


# evidence sentences that volunteer a provenance verdict of the form
# "presented without criticism / attribution / distancing / critical framing"
NEG_CLAUSE = re.compile(r"\b(without|with no|no)\b[^.]{0,70}?"
                        r"\b(critic|critiq|attribut|distanc|framing|context)", re.I)

# of the items carrying that clause, the ones where it is demonstrably false against the
# transcript or against voice_field_analysis.py::GOLD_VOICE (hand-adjudicated, see notes above)
NEG_CLAUSE_FALSE = {("HateMM", "non_hate_video_121"), ("HateMM", "non_hate_video_149"),
                    ("HateMM", "non_hate_video_400"), ("ImpliHateVid", "NH_988"),
                    ("ImpliHateVid", "NH_875")}


def binarise(s):
    return None if s is None else ("ENDORSE" if s == "endorses" else "OPPOSE")


def main():
    sample = {(x["dataset"], x["id"]): x
              for x in json.load(open(os.path.join(ROOT, "idea-stage", "stance_pilot",
                                                   "sample.json")))["eval"]}
    rows = [json.loads(l) for l in
            open(os.path.join(ROOT, "idea-stage", "stance_pilot", "pred_strong.jsonl"),
                 encoding="utf-8")]

    wrong, right, out = {}, {}, {}
    per_item = []
    for r in rows:
        k = (r["dataset"], r["id"])
        g = sample[k]["group"]
        st = (r.get("parsed") or {}).get("stance")
        b = binarise(st)
        gd = GOLD[g]
        is_wrong = (b == "ENDORSE") if gd is None else (b != gd)
        (wrong if is_wrong else right).setdefault(g, []).append(k)
        if is_wrong:
            if k not in CODES:
                raise SystemExit(f"uncoded wrong item {k}")
            s, l, v, note = CODES[k]
            ev = (r.get("parsed") or {}).get("evidence") or ""
            per_item.append({"ds": k[0], "id": k[1], "group": g, "gold": gd, "stance": st,
                             "smoke": k in SMOKE, "frame": k[0] in FRAME_DS,
                             "R_STRICT": s, "R_LOOSE": l, "R_VOICE": v,
                             "neg_clause": bool(NEG_CLAUSE.search(ev)),
                             "neg_clause_false": k in NEG_CLAUSE_FALSE, "note": note})

    for g in ("S_FP", "S_FN", "CTRL_HATE", "CTRL_NONHATE"):
        w = wrong.get(g, [])
        sub = [p for p in per_item if p["group"] == g]
        ns = sum(p["R_STRICT"] for p in sub)
        nl = sum(p["R_LOOSE"] for p in sub)
        nv = sum(p["R_STRICT"] or p["R_VOICE"] for p in sub)
        out[g] = {
            "n_items": len(w) + len(right.get(g, [])),
            "n_wrong": len(w), "n_right": len(right.get(g, [])),
            "R_STRICT_yes": ns,
            "ratio_of_wrong": round(ns / len(w), 4) if w else None,
            "ratio_of_all": round(ns / (len(w) + len(right.get(g, []))), 4),
            "R_STRICT_or_VOICE_yes": nv,
            "ratio_lenient_of_wrong": round(nv / len(w), 4) if w else None,
            "R_LOOSE_content_accurate": nl,
            "ratio_content_accurate_of_wrong": round(nl / len(w), 4) if w else None,
            "evidence_volunteers_no_criticism_clause": sum(p["neg_clause"] for p in sub),
            "of_which_demonstrably_false": sum(p["neg_clause_false"] for p in sub),
        }

    # frame-bearing / no-frame split on S_FP
    for lab, pred in (("S_FP_frame", lambda p: p["group"] == "S_FP" and p["frame"]),
                      ("S_FP_noframe", lambda p: p["group"] == "S_FP" and not p["frame"]),
                      ("S_FP_no_smoke", lambda p: p["group"] == "S_FP" and not p["smoke"])):
        sub = [p for p in per_item if pred(p)]
        out[lab] = {"n_wrong": len(sub), "R_STRICT_yes": sum(p["R_STRICT"] for p in sub),
                    "ratio_of_wrong": round(sum(p["R_STRICT"] for p in sub) / len(sub), 4)
                    if sub else None}

    gate = out["S_FP"]["ratio_of_wrong"]
    out["GATE"] = {"criterion": "S_FP ratio of wrong items with a correct direction-pointing "
                                "evidence fact",
                   "value": gate, "bar": 0.30,
                   "verdict": "PREMISE HOLDS -> proceed" if gate >= 0.30
                   else "PREMISE DOES NOT HOLD -> STOP, spend nothing"}
    out["per_item"] = sorted(per_item, key=lambda p: (p["group"], p["ds"], p["id"]))
    json.dump(out, open(os.path.join(HERE, "step0_audit.json"), "w"), indent=1,
              ensure_ascii=False)

    for g in ("S_FP", "S_FP_no_smoke", "S_FP_frame", "S_FP_noframe", "S_FN", "CTRL_HATE",
              "CTRL_NONHATE"):
        print(g, json.dumps(out[g], ensure_ascii=False))
    print()
    print("GATE:", json.dumps(out["GATE"], ensure_ascii=False))


if __name__ == "__main__":
    main()
