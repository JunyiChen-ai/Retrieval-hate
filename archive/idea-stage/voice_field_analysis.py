"""VOICE FIELD ANALYSIS -- can the `primary_voice` field alone separate stance-class
detector errors from correctly-classified items?

Pure offline re-analysis of the already-paid-for stance pilot outputs. Zero API calls,
zero GPU. Nothing here re-scores the stance pilot; its KILL verdict stands.

=======================================================================================
JUDGEMENT CRITERIA -- FROZEN BEFORE ANY NUMBER IN THIS FILE WAS COMPUTED
(written 2026-08-12, author: subagent Opus 5; no cell count was looked at first)
=======================================================================================

F0. POPULATION / STRATIFICATION. Inherited unchanged from STANCE_PILOT_FREEZE.md §6.1 (D1):
    view A = the 72 frame-bearing items (HateMM + MHC + MHC_zh) -- THE PRIMARY READ;
    view B = all 99 (adds the 27 transcript-only ImpliHateVid items) -- appendix;
    view C = the 27 ImpliHateVid items alone -- appendix.
    Round 1 has 71 rows in view A (1 item moderation-rejected, D6); fallback (2) has 72
    rows of which 1 has a stage-A error and therefore a null voice.

F1. BINARISATION OF `primary_voice` (primary definition, "V-strict"):
      OWN     = {uploader}
      NOT_OWN = {on_screen_speaker, quoted_third_party, archival_source, caption_overlay}
      NO_VOICE= {none}  -> EXCLUDED from the primary 2x2.
    Rationale, fixed in advance: (i) `none` is mechanically forced by the prompt
    ("none = there is no such material (Q1 false)"), so it is a restatement of
    hate_surface_present, not an observation about voice; including it would smuggle the
    hate/non-hate axis back into a test that is supposed to isolate voice.
    (ii) STANCE_PILOT_RESULT.md §3(a) already treats on_screen_speaker as non-uploader;
    this analysis keeps that reading so the two documents are commensurable.
    Two pre-registered sensitivity variants are also reported and may NOT be promoted to
    primary after the fact:
      V-loose : OWN = {uploader, on_screen_speaker}; NOT_OWN = the other three.
      V-incl  : primary split, but `none` folded into NOT_OWN.

F2. CONTRASTS. Primary contrast = all S-bucket errors (S_FP + S_FN) vs all controls
    (CTRL_HATE + CTRL_NONHATE), 2x2 on {NOT_OWN, OWN}. Two decision-relevant
    sub-contrasts are also reported: S_FP vs CTRL_NONHATE (both label=0) and
    S_FP vs CTRL_HATE (the pair the proposed rule actually trades off).

F3. STATISTIC. Odds ratio of NOT_OWN in the error group vs the control group, with the
    Haldane-Anscombe +0.5 correction applied to every cell if and only if some cell is 0
    (the uncorrected OR is also printed). p = two-sided Fisher exact test.

F4. DECISION BAR (frozen, as instructed):
      SIGNAL  iff OR >= 3.0 AND p < 0.05  on the PRIMARY contrast in the PRIMARY view (A),
              under the PRIMARY binarisation V-strict, in at least one of the two rounds.
      Otherwise BURY.
    A sensitivity variant or an appendix view clearing the bar while the primary does not
    is recorded as "not sufficient" and explicitly does not overturn BURY.

F5. NET FLIP PROJECTION. Frozen rule R_voice: `primary_voice in NOT_OWN` => push the item
    towards NON-HATE. One-directional (OWN never pushes towards hate; the whole proposal
    is a suppression rule). Then, restricted to items the rule fires on:
      gains  = #{S_FP with NOT_OWN}                     (false positive, correctly suppressed)
      neutral= #{S_FN with NOT_OWN}                     (already wrong, stays wrong)
      damage = #{CTRL_HATE with NOT_OWN}                (was right, now wrong)
      CTRL_NONHATE is unaffected by a one-directional non-hate push.
    Sample-level net = gains - damage. Population projection uses the same estimator as
    STANCE_PILOT_FREEZE.md §6 P3: damage_pop = sum_ds (rate_hate_ds * n_correct_hate_ds),
    gains_pop = the complete enumeration of S_FP errors actually in the sample scaled by
    (all S_FP errors in the population / S_FP errors sampled) per dataset -- because unlike
    P3 this sample does NOT contain every S error, so the raw count would understate gains.
    Both the raw-count net and the scaled-population net are reported; the verdict on P3'
    uses the scaled-population net.

F6. TWO-ROUND STABILITY. Raw 6-way agreement + Cohen's kappa + binary (V-strict)
    agreement between round 1 and fallback (2) on items where both emitted a voice.
    Declared in advance as a *prompt-robustness* measure, not a sampling-noise measure:
    both runs are temperature 0, but fallback (2) asks the voice question in a different
    (decomposed) prompt. Reported, not used as a gate.

F7. VOICE ACCURACY (part 4). Gold voice form is coded by hand from the r5_error_dump
    transcript + OCR text ONLY, BEFORE the model's voice output for that item is read, and
    only for items where the text makes the utterance source determinable; undeterminable
    items are recorded as such and dropped from the denominator, with n stated. Adjudication
    is at the binary OWN/NOT_OWN level (the 6-way call is often undecidable from text).
    This is a small hand-coded n and is reported as an estimate with its n, not as a metric.

F8. Nothing in this file feeds back into STANCE_PILOT_FREEZE.md's P1/P2/P3 or the KILL.
=======================================================================================
"""
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SP = os.path.join(HERE, "stance_pilot")

VOICES = ["uploader", "on_screen_speaker", "quoted_third_party",
          "archival_source", "caption_overlay", "none"]
OWN_STRICT = {"uploader"}
NOTOWN_STRICT = {"on_screen_speaker", "quoted_third_party", "archival_source", "caption_overlay"}
OWN_LOOSE = {"uploader", "on_screen_speaker"}
NOTOWN_LOOSE = {"quoted_third_party", "archival_source", "caption_overlay"}
FRAME_DS = ["HateMM", "MHC", "MHC_zh"]
TEXT_DS = ["ImpliHateVid"]
ALL_DS = FRAME_DS + TEXT_DS
OR_BAR, P_BAR = 3.0, 0.05


# ----------------------------------------------------------------- statistics
def fisher_exact_2x2(a, b, c, d):
    """two-sided Fisher exact p for [[a,b],[c,d]] (exact, integer arithmetic)."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def pr(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    p0 = pr(a)
    tol = p0 * (1 + 1e-9)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= tol)


def odds_ratio(a, b, c, d):
    """a=err&NOT_OWN b=err&OWN c=ctrl&NOT_OWN d=ctrl&OWN"""
    raw = None if (b == 0 or c == 0) else (a * d) / (b * c)
    if 0 in (a, b, c, d):
        A, B, C, D = a + .5, b + .5, c + .5, d + .5
        return raw, (A * D) / (B * C), True
    return raw, (a * d) / (b * c), False


def contrast(rows, err_groups, ctrl_groups, own, notown, include_none_as_notown=False):
    def cls(v):
        if v in own:
            return "OWN"
        if v in notown:
            return "NOT_OWN"
        if v == "none":
            return "NOT_OWN" if include_none_as_notown else None
        return None
    e = [cls(r["voice"]) for r in rows if r["group"] in err_groups]
    c_ = [cls(r["voice"]) for r in rows if r["group"] in ctrl_groups]
    a = sum(1 for x in e if x == "NOT_OWN"); b = sum(1 for x in e if x == "OWN")
    c = sum(1 for x in c_ if x == "NOT_OWN"); d = sum(1 for x in c_ if x == "OWN")
    raw, adj, corrected = odds_ratio(a, b, c, d)
    p = fisher_exact_2x2(a, b, c, d) if (a + b) and (c + d) else 1.0
    return {"a_err_notown": a, "b_err_own": b, "c_ctrl_notown": c, "d_ctrl_own": d,
            "err_rate_notown": a / (a + b) if a + b else None,
            "ctrl_rate_notown": c / (c + d) if c + d else None,
            "OR_raw": raw, "OR": adj, "haldane_corrected": corrected, "p_fisher": p,
            "n_dropped_none": sum(1 for r in rows
                                  if r["group"] in (err_groups + ctrl_groups)
                                  and cls(r["voice"]) is None)}


def cohen_kappa(pairs, cats):
    n = len(pairs)
    if not n:
        return None
    obs = sum(1 for x, y in pairs if x == y) / n
    m1 = Counter(x for x, _ in pairs); m2 = Counter(y for _, y in pairs)
    exp = sum((m1[c] / n) * (m2[c] / n) for c in cats)
    return obs, (obs - exp) / (1 - exp) if exp < 1 else None


# ----------------------------------------------------------------- data
def load(tag):
    sample = {(x["dataset"], x["id"]): x for x in
              json.load(open(os.path.join(SP, "sample.json")))["eval"]}
    rows = []
    for line in open(os.path.join(SP, f"pred_{tag}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        it = sample[(r["dataset"], r["id"])]
        p = r.get("parsed") or {}
        v = p.get("primary_voice")
        rows.append({**it, "voice": v if v in VOICES else None,
                     "stance": p.get("stance"), "surface": p.get("hate_surface_present"),
                     "evidence": p.get("evidence"), "parse": r["parse"]})
    return rows


def population_counts():
    A = json.load(open(os.path.join(HERE, "r5_phase_a.json")))["A2_error_attribution"]
    out = {}
    for ds in ALL_DS:
        err = set(A[ds]["err_ids"])
        pos = neg = 0
        for line in open(os.path.join(ROOT, "data", "gt", ds, "test.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in err:
                continue
            pos += int(r["label"]) == 1
            neg += int(r["label"]) == 0
        out[ds] = (pos, neg)
    return out


def s_error_population():
    """per-dataset count of ALL primary-S errors in the test splits, split FP/FN."""
    buckets = json.load(open(os.path.join(HERE, "r5_buckets.json")))
    dump = json.load(open(os.path.join(HERE, "r5_error_dump.json")))
    out = {}
    for ds in ALL_DS:
        kind = {r["id"]: r["kind"] for r in dump[ds]}
        fp = fn = 0
        for vid, b in buckets[ds].items():
            if b != "S":
                continue
            if kind.get(vid) == "FP":
                fp += 1
            elif kind.get(vid) == "FN":
                fn += 1
        out[ds] = {"S_FP": fp, "S_FN": fn}
    return out


# ----------------------------------------------------------------- views
def view_block(rows, datasets, tag, name, pop, spop):
    R = {"view": name, "tag": tag, "datasets": datasets}
    rows = [r for r in rows if r["dataset"] in datasets]
    R["n"] = len(rows)

    # ---- 1. distribution tables
    dist = {}
    for grp in ["S_FP", "S_FN", "CTRL_HATE", "CTRL_NONHATE"]:
        c = Counter(r["voice"] for r in rows if r["group"] == grp)
        dist[grp] = {v: c.get(v, 0) for v in VOICES}
        dist[grp]["_null"] = c.get(None, 0)
        dist[grp]["_n"] = sum(1 for r in rows if r["group"] == grp)
    R["voice_dist"] = dist

    # ---- 2. separability
    R["contrasts"] = {}
    for cname, eg, cg in [("S_all_vs_CTRL_all", ["S_FP", "S_FN"], ["CTRL_HATE", "CTRL_NONHATE"]),
                          ("S_FP_vs_CTRL_NONHATE", ["S_FP"], ["CTRL_NONHATE"]),
                          ("S_FP_vs_CTRL_HATE", ["S_FP"], ["CTRL_HATE"]),
                          ("S_FN_vs_CTRL_HATE", ["S_FN"], ["CTRL_HATE"])]:
        R["contrasts"][cname] = {
            "V_strict": contrast(rows, eg, cg, OWN_STRICT, NOTOWN_STRICT),
            "V_loose": contrast(rows, eg, cg, OWN_LOOSE, NOTOWN_LOOSE),
            "V_incl": contrast(rows, eg, cg, OWN_STRICT, NOTOWN_STRICT, True)}

    # ---- 3. net flip projection (rule R_voice, V-strict)
    def fires(r):
        return r["voice"] in NOTOWN_STRICT
    g = defaultdict(list)
    for r in rows:
        g[r["group"]].append(r)
    gains_raw = sum(1 for r in g["S_FP"] if fires(r))
    neutral = sum(1 for r in g["S_FN"] if fires(r))
    dmg_raw = sum(1 for r in g["CTRL_HATE"] if fires(r))
    per_ds, dmg_pop, gain_pop = {}, 0.0, 0.0
    for ds in datasets:
        h = [r for r in g["CTRL_HATE"] if r["dataset"] == ds]
        fp = [r for r in g["S_FP"] if r["dataset"] == ds]
        rate_h = (sum(1 for r in h if fires(r)) / len(h)) if h else 0.0
        rate_fp = (sum(1 for r in fp if fires(r)) / len(fp)) if fp else 0.0
        d = rate_h * pop[ds][0]
        gp = rate_fp * spop[ds]["S_FP"]
        per_ds[ds] = {"rate_notown_ctrl_hate": round(rate_h, 3),
                      "n_correct_hate_pop": pop[ds][0], "projected_damage": round(d, 2),
                      "rate_notown_S_FP": round(rate_fp, 3),
                      "n_S_FP_pop": spop[ds]["S_FP"], "projected_gain": round(gp, 2),
                      "n_S_FP_sampled": len(fp), "n_ctrl_hate_sampled": len(h)}
        dmg_pop += d
        gain_pop += gp
    R["flip"] = {"rule": "primary_voice in NOT_OWN(V-strict) -> push non-hate",
                 "sample_gains_S_FP": gains_raw, "sample_neutral_S_FN": neutral,
                 "sample_damage_CTRL_HATE": dmg_raw,
                 "sample_net": gains_raw - dmg_raw,
                 "pop_projected_gain": round(gain_pop, 2),
                 "pop_projected_damage": round(dmg_pop, 2),
                 "pop_net": round(gain_pop - dmg_pop, 2),
                 "per_dataset": per_ds}
    return R


# ------------------------------------------------------- part 4: voice accuracy
# Hand-coded gold voice FORM for the 49 sampled S-bucket items, per F7.
# Coded by the analysing agent from the r5_error_dump transcript + OCR text ONLY,
# written down BEFORE the corresponding model `primary_voice` output was read.
# Values: "OWN"   = the hate-associated surface is produced by the video's own author
#         "NOT_OWN" = it comes from archive / broadcast / a named third party / an
#                     embedded clip the author is reacting to
#         "UNDET" = the text does not determine it (dropped from the denominator)
# `_blind: False` marks the three items whose model output had already been seen in
# this session before coding (declared, not hidden).
GOLD_VOICE = {
    ("HateMM", "hate_video_365"): ("NOT_OWN", "public-meeting / MLive news footage; the slur is an audience member's, recorded by a media outlet", False),
    ("HateMM", "non_hate_video_121"): ("OWN", "uploader's own recorded prank/expose phone call; he is the speaker driving it", False),
    ("HateMM", "non_hate_video_149"): ("NOT_OWN", "archival minstrel-era 'Swanee River' recording reposted", False),
    ("HateMM", "non_hate_video_16"): ("NOT_OWN", "1956 segregationist newsreel; OCR carries a GlobalImageWorks stock-footage timecode watermark", True),
    ("HateMM", "non_hate_video_32"): ("NOT_OWN", "Lennon/Ono song 'Woman Is the Nigger of the World' — a third-party recording", True),
    ("HateMM", "non_hate_video_400"): ("NOT_OWN", "Ellen show broadcast clip (OCR: ellentube / Watch Ellen Weekdays)", True),
    ("HateMM", "non_hate_video_528"): ("OWN", "German activist video; the charged material is the video's OWN burned-in editorial text", True),
    ("HateMM", "non_hate_video_642"): ("UNDET", "third-party Kiffness music video with the re-uploader's own antisemitic site URLs overlaid — two sources interleaved", True),
    ("MHC", "03qOelm_dK8"): ("UNDET", "pundit commentary short; cannot tell from text whether the pundit is the uploader", True),
    ("MHC", "8zLoOqXvk64"): ("UNDET", "no speech, title only", True),
    ("MHC", "DxcRdnzBZoo"): ("OWN", "Shane & Hannah answering questions about themselves in first person", True),
    ("MHC", "EEC98aHSgIY"): ("OWN", "'Welcome to Proverbs Club ... visit Proverbs.club' — the channel's own narration", True),
    ("MHC", "KDcCiUU8q5E"): ("UNDET", "news/podcast commentator clip; uploader identity not determinable", True),
    ("MHC", "N68vmAE5s_g"): ("OWN", "'subscribe Amelia Time for more videos' — the channel's own narrator", True),
    ("MHC", "OMSByZ-o3Ww"): ("OWN", "uploader's own reaction voiceover over footage", True),
    ("MHC", "XlJCNPi5inM"): ("NOT_OWN", "title attributes to a named third party (Kevin Samuels); clip channel", True),
    ("MHC", "YDEsYXYlB8o"): ("OWN", "'This is gay science, in today's episode' — the channel's own satirical episode", True),
    ("MHC", "_qldaPBgkk0"): ("UNDET", "podcast clip, #podcast tag; speakers vs uploader not determinable", True),
    ("MHC", "cXRgVEENkPA"): ("OWN", "commentator narrating the Dodgers story on his own show", True),
    ("MHC", "dK43yHIUMKA"): ("NOT_OWN", "title names Dave Allen; archival stand-up routine", True),
    ("MHC", "ga1r2cweP80"): ("OWN", "vlogger's own first-person narration in the red-light district", True),
    ("MHC", "h_wKRDyoG_c"): ("OWN", "kpop commentary channel's own narration reporting netizen claims", True),
    ("MHC", "j_foVftOOs4"): ("NOT_OWN", "news package audio ('This is the moment a Saint Louis police officer...')", True),
    ("MHC", "pofgIFZpR7c"): ("NOT_OWN", "title names Hamza Ahmed; clip of a third party's monologue", True),
    ("MHC_zh", "BV12G4y1S7mN"): ("NOT_OWN", "film-recap (解说) of third-party drama footage", True),
    ("MHC_zh", "BV15h4y157Km"): ("OWN", "uploader's own comedic commentary on a news item", True),
    ("MHC_zh", "BV1Kh411T7FJ"): ("UNDET", "multi-speaker talk clip; uploader identity not determinable", True),
    ("MHC_zh", "BV1Km4y1u7ri"): ("OWN", "single-speaker performance to camera", True),
    ("MHC_zh", "BV1Qk4y1g7PM"): ("OWN", "uploader performs the misogynistic phrases herself as specimens", True),
    ("MHC_zh", "BV1Vy4y1p7x2"): ("UNDET", "classroom skit; participant/uploader relation not determinable", True),
    ("MHC_zh", "BV1aP4y1E7PF"): ("UNDET", "music compilation, no determining cue", True),
    ("MHC_zh", "BV1ch411L7VP"): ("NOT_OWN", "archival Audrey Hepburn footage redubbed with a fabricated line", True),
    ("MHC_zh", "BV1m8411z7mV"): ("NOT_OWN", "excerpt of a third-party costume drama", True),
    ("MHC_zh", "BV1qZ4y1T71a"): ("UNDET", "short skit/drama excerpt, source not determinable", True),
    ("MHC_zh", "BV1to4y177df"): ("UNDET", "short skit, source not determinable", True),
    ("MHC_zh", "BV1vK41177zi"): ("NOT_OWN", "excerpt of a third-party drama (爽剧片段)", True),
    ("ImpliHateVid", "EX_150"): ("OWN", "'This is not me judging people, I'm just saying' — first-person monologue", True),
    ("ImpliHateVid", "IM_176"): ("OWN", "'This may come as a surprise to my younger viewers' — essayist's own narration", True),
    ("ImpliHateVid", "IM_308"): ("NOT_OWN", "CTV news package ('CTV's Kamal Kramali explains')", True),
    ("ImpliHateVid", "IM_5"): ("NOT_OWN", "Australian Senate footage + broadcast commentary", True),
    ("ImpliHateVid", "IM_57"): ("OWN", "first-person monologue ('I'm mad about that')", True),
    ("ImpliHateVid", "NH_394"): ("UNDET", "two-line threat clip, no provenance cue", True),
    ("ImpliHateVid", "NH_396"): ("OWN", "uploader interviewing a guest on his own channel", True),
    ("ImpliHateVid", "NH_44"): ("OWN", "'Hi everyone today I'm gonna talk about' — own monologue", True),
    ("ImpliHateVid", "NH_650"): ("OWN", "scripted two-voice debate skit authored by the uploader", True),
    ("ImpliHateVid", "NH_736"): ("OWN", "first-person monologue", True),
    ("ImpliHateVid", "NH_875"): ("NOT_OWN", "the hateful surface is in the embedded Starbucks-parody clip being reacted to", True),
    ("ImpliHateVid", "NH_887"): ("OWN", "first-person rant", True),
    ("ImpliHateVid", "NH_988"): ("UNDET", "two-speaker comedy bit, uploader relation not determinable", True),
}


def voice_accuracy(rows_by_tag, loose=False):
    OWN_S = OWN_LOOSE if loose else OWN_STRICT
    NOT_S = NOTOWN_LOOSE if loose else NOTOWN_STRICT

    def b(v):
        if v in OWN_S:
            return "OWN"
        if v in NOT_S:
            return "NOT_OWN"
        return "NONE" if v == "none" else None
    out = {}
    for tag, rows in rows_by_tag.items():
        m = {(r["dataset"], r["id"]): r for r in rows}
        per_view = {}
        for name, ds in [("A_frames_primary", FRAME_DS), ("B_all99", ALL_DS), ("C_textonly", TEXT_DS)]:
            hit = tot = none_out = 0
            errs = []
            for (d, i), (g, why, blind) in GOLD_VOICE.items():
                if d not in ds or g == "UNDET":
                    continue
                r = m.get((d, i))
                if not r:
                    continue
                p = b(r["voice"])
                if p == "NONE":
                    none_out += 1        # model says "no hate surface" -> no voice call to score
                    continue
                if p is None:
                    continue
                tot += 1
                if p == g:
                    hit += 1
                else:
                    errs.append({"item": f"{d}::{i}", "gold": g, "pred": r["voice"], "why": why})
            per_view[name] = {"n_scored": tot, "n_correct": hit,
                              "acc": round(hit / tot, 3) if tot else None,
                              "n_model_said_none": none_out,
                              "n_gold_undet": sum(1 for (d, i), (g, _, _) in GOLD_VOICE.items()
                                                  if d in ds and g == "UNDET"),
                              "errors": errs}
        out[tag] = per_view
    gold_ds = Counter(d for (d, i) in GOLD_VOICE)
    out["_gold_summary"] = {
        "n_coded": len(GOLD_VOICE),
        "by_class": dict(Counter(g for g, _, _ in GOLD_VOICE.values())),
        "by_dataset": dict(gold_ds),
        "n_not_blind": sum(1 for g, _, bl in GOLD_VOICE.values() if not bl)}
    return out


def main():
    pop = population_counts()
    spop = s_error_population()
    out = {"frozen_criteria": "see module docstring F0-F8",
           "population_correct_counts": pop, "S_error_population": spop}

    r1 = load("strong")
    fb = load("fb2")
    for tag, rows in [("strong", r1), ("fb2", fb)]:
        out[tag] = {
            "A_frames_primary": view_block(rows, FRAME_DS, tag, "A_frames_primary", pop, spop),
            "B_all99": view_block(rows, ALL_DS, tag, "B_all99", pop, spop),
            "C_textonly": view_block(rows, TEXT_DS, tag, "C_textonly", pop, spop)}

    # ---- 6. two-round stability
    m1 = {(r["dataset"], r["id"]): r for r in r1}
    stab = {}
    for name, ds in [("A_frames_primary", FRAME_DS), ("B_all99", ALL_DS), ("C_textonly", TEXT_DS)]:
        pairs6, pairs2 = [], []
        for r in fb:
            if r["dataset"] not in ds:
                continue
            o = m1.get((r["dataset"], r["id"]))
            if not o or not o["voice"] or not r["voice"]:
                continue
            pairs6.append((o["voice"], r["voice"]))

            def b(v):
                return "OWN" if v in OWN_STRICT else ("NOT_OWN" if v in NOTOWN_STRICT else "NONE")
            pairs2.append((b(o["voice"]), b(r["voice"])))
        a6 = cohen_kappa(pairs6, VOICES)
        a2 = cohen_kappa(pairs2, ["OWN", "NOT_OWN", "NONE"])
        conf = Counter(pairs6)
        stab[name] = {"n": len(pairs6),
                      "agree6": round(a6[0], 3) if a6 else None,
                      "kappa6": round(a6[1], 3) if a6 and a6[1] is not None else None,
                      "agree_binary3": round(a2[0], 3) if a2 else None,
                      "kappa_binary3": round(a2[1], 3) if a2 and a2[1] is not None else None,
                      "confusion": {f"{k[0]}->{k[1]}": v for k, v in sorted(conf.items(),
                                                                           key=lambda x: -x[1])}}
    out["two_round_stability"] = stab
    out["voice_accuracy"] = voice_accuracy({"strong": r1, "fb2": fb})
    out["voice_accuracy_Vloose"] = voice_accuracy({"strong": r1, "fb2": fb}, loose=True)

    # ---- EXPLORATORY (declared after the fact, NOT part of the F4 verdict):
    # does voice separate S_FP from CTRL_HATE *conditional on the model saying `endorses`*?
    exp = {}
    for tag, rows in [("strong", r1), ("fb2", fb)]:
        sub = [r for r in rows if r["dataset"] in FRAME_DS and r["stance"] == "endorses"]
        exp[tag] = contrast(sub, ["S_FP"], ["CTRL_HATE"], OWN_STRICT, NOTOWN_STRICT)
        exp[tag]["n_endorses_items"] = len(sub)
    out["EXPLORATORY_voice_given_endorses"] = exp

    # ---- verdict against F4
    prim = out["strong"]["A_frames_primary"]["contrasts"]["S_all_vs_CTRL_all"]["V_strict"]
    prim2 = out["fb2"]["A_frames_primary"]["contrasts"]["S_all_vs_CTRL_all"]["V_strict"]
    ok = [(t, c["OR"], c["p_fisher"], c["OR"] >= OR_BAR and c["p_fisher"] < P_BAR)
          for t, c in [("round1", prim), ("fb2", prim2)]]
    out["VERDICT_F4"] = {"bar": f"OR>={OR_BAR} and p<{P_BAR}", "rounds": ok,
                         "verdict": "SIGNAL" if any(x[3] for x in ok) else "BURY"}

    json.dump(out, open(os.path.join(HERE, "voice_field_analysis.json"), "w"), indent=1)

    # ---- console
    for tag in ("strong", "fb2"):
        for vw in ("A_frames_primary", "B_all99", "C_textonly"):
            V = out[tag][vw]
            print("=" * 78)
            print(f"[{tag}] {vw}  n={V['n']}")
            print(f"{'group':<14}" + "".join(f"{v[:11]:>13}" for v in VOICES) + f"{'null':>7}{'n':>5}")
            for grp, d in V["voice_dist"].items():
                print(f"{grp:<14}" + "".join(f"{d[v]:>13}" for v in VOICES)
                      + f"{d['_null']:>7}{d['_n']:>5}")
            for cn, cc in V["contrasts"].items():
                for bn, c in cc.items():
                    print(f"  {cn:<24} {bn:<9} err {c['a_err_notown']}/{c['a_err_notown']+c['b_err_own']}"
                          f"  ctrl {c['c_ctrl_notown']}/{c['c_ctrl_notown']+c['d_ctrl_own']}"
                          f"  OR={c['OR']:.3f}{'*' if c['haldane_corrected'] else ''}"
                          f"  p={c['p_fisher']:.4f}")
            f = V["flip"]
            print(f"  FLIP sample: gains={f['sample_gains_S_FP']} damage={f['sample_damage_CTRL_HATE']}"
                  f" neutral={f['sample_neutral_S_FN']} net={f['sample_net']}"
                  f" | pop gain={f['pop_projected_gain']} dmg={f['pop_projected_damage']}"
                  f" net={f['pop_net']}")
    print("=" * 78)
    print("STABILITY", json.dumps(out["two_round_stability"], indent=1))
    print("VOICE ACCURACY", json.dumps(out["voice_accuracy"], indent=1, ensure_ascii=False))
    print("VERDICT", json.dumps(out["VERDICT_F4"], indent=1))


if __name__ == "__main__":
    main()
