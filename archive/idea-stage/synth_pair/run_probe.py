"""SYNTH_PAIR_PROBE -- rule-synthesised quoting/self-utterance pairs -> real ASR transfer.

Everything here follows idea-stage/SYNTH_PAIR_PROBE_FREEZE.md, committed as 73971c1
BEFORE this file trained anything. Nothing below may be edited after the first frozen run.

Test-set handling: data/gt/<ds>/test.jsonl is read ONLY in the scoring stage (stage 4), by
an already-frozen classifier, and never influences pair construction, training, model
selection or thresholding.
"""
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GT = os.path.join(ROOT, "data", "gt")
DSS = ["HateMM", "MHC", "MHC_zh", "ImpliHateVid"]
SEED = 0
SMOKE = os.environ.get("SYNTH_PAIR_SMOKE") == "1"
MAX_SRC = 200 if SMOKE else 2600
MIN_PAIRS = 1 if SMOKE else 2000

# ------------------------------------------------------------------ FREEZE §2 templates
EN_OWN = [
    "so anyway {s} and that's just how it is",
    "look guys {s} i'm telling you right now",
    "here's what i think {s}",
    "welcome back to the channel today {s}",
    "{s} that's my honest opinion on this",
    "let me be clear about something {s}",
    "i've been saying this for years {s}",
    "okay so listen {s} alright moving on",
    "and honestly {s} i stand by that",
    "{s} that's coming straight from me",
    "i want to say something to you {s}",
    "you know what {s} and i mean every word",
]
EN_NOT = [
    "he said {s} end of quote",
    "she literally said {s} can you believe that",
    "the guy in this clip says {s}",
    "reports say that {s}",
    "here's the clip where he says {s}",
    "someone commented {s} which is disgusting",
    "this is what they posted {s} and it's vile",
    "the video shows him saying {s}",
    "according to the article {s}",
    "one user wrote {s} and got destroyed in the replies",
    "let me read you what he wrote {s} unbelievable",
    "in the footage the speaker states {s}",
    "{s} that's a direct quote from his speech",
    "he was recorded saying {s} and then he doubled down",
]
ZH_OWN = [
    "那我今天就说一句 {s} 就这样",
    "大家好 今天我想说 {s}",
    "我跟你们讲 {s} 我是认真的",
    "{s} 这就是我的看法",
    "说实话 {s} 我就这么想的",
    "我一直觉得 {s}",
    "咱们继续说 {s} 好吧",
    "{s} 我把话放在这里",
    "先讲一件事 {s}",
    "反正我的意思就是 {s}",
]
ZH_NOT = [
    "他说 {s} 原话就是这样",
    "她当时说 {s} 你敢信吗",
    "视频里这个人说 {s}",
    "有报道称 {s}",
    "网友评论说 {s} 真的很恶心",
    "他发的原文是 {s} 太离谱了",
    "根据这篇文章 {s}",
    "画面中的发言人表示 {s}",
    "{s} 这是他演讲里的原话",
    "我念一下他写的 {s} 简直无语",
]

MARKERS_EN = ["he said", "she said", "they said", "he says", "she says", "they say",
              "quote", "quoted", "according to", "reports say", "reported that", "wrote",
              "posted", "tweeted", "commented", "speaking to", "told reporters",
              "in his words", "in her words", "end quote"]
MARKERS_ZH = ["他说", "她说", "他们说", "据报道", "原话", "引用", "网友说", "评论说",
              "报道称", "表示", "写道", "发文"]


# ------------------------------------------------------------------ FREEZE §3 normaliser
def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "")


def cjk_ratio(s):
    if not s:
        return 0.0
    n = sum(1 for c in s if "一" <= c <= "鿿")
    return n / len(s)


def lang_of(s):
    return "zh" if cjk_ratio(s) > 0.15 else "en"


def normalise(s):
    """FREEZE §3: NFKC -> drop P*/S* -> lowercase -> collapse whitespace."""
    s = unicodedata.normalize("NFKC", s or "")
    out = []
    for c in s:
        cat = unicodedata.category(c)
        if cat[0] in ("P", "S"):
            out.append(" ")
        else:
            out.append(c)
    return re.sub(r"\s+", " ", "".join(out)).strip().lower()


def has_marker(norm_text, lang):
    lex = MARKERS_ZH if lang == "zh" else MARKERS_EN
    return any(m in norm_text for m in lex)


# ------------------------------------------------------------------ FREEZE §1 extraction
def source_sentences():
    rng = random.Random(SEED)
    seen, cands = set(), []
    per_ds = Counter()
    for ds in DSS:
        p = os.path.join(GT, ds, "train.jsonl")
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if str(r.get("label")) != "1":
                continue
            txt = strip_html(r.get("text", "")).replace("\U0001f3bc", " ")
            lang = lang_of(txt)
            parts = re.split(r"[。！？；!?\n]", txt) if lang == "zh" else re.split(r"[.!?;\n]", txt)
            for raw in parts:
                raw = raw.strip()
                if not raw:
                    continue
                nrm = normalise(raw)
                if not nrm:
                    continue
                if lang == "zh":
                    if not (8 <= len(nrm) <= 40):
                        continue
                else:
                    if not (5 <= len(nrm.split()) <= 25):
                        continue
                if nrm in seen:
                    continue
                seen.add(nrm)
                cands.append({"text": nrm, "lang": lang, "ds": ds})
                per_ds[ds] += 1
    rng.shuffle(cands)
    return cands[:MAX_SRC], per_ds


def build_pairs(srcs):
    rng = random.Random(SEED)
    rows = []
    for i, s in enumerate(srcs):
        own_t = ZH_OWN if s["lang"] == "zh" else EN_OWN
        not_t = ZH_NOT if s["lang"] == "zh" else EN_NOT
        rows.append({"pair": i, "y": 0, "lang": s["lang"],
                     "text": normalise(rng.choice(own_t).format(s=s["text"]))})
        rows.append({"pair": i, "y": 1, "lang": s["lang"],
                     "text": normalise(rng.choice(not_t).format(s=s["text"]))})
    return rows


# ------------------------------------------------------------------ FREEZE §5 eval sets
def load_gold_voice():
    src = open(os.path.join(ROOT, "idea-stage", "voice_field_analysis.py"), encoding="utf-8").read()
    blk = src.split("GOLD_VOICE = {")[1].split("\n}")[0]
    out = {}
    for ds, vid, g in re.findall(r'\("([A-Za-z_]+)",\s*"([^"]+)"\):\s*\("(OWN|NOT_OWN|UNDET)"', blk):
        out[(ds, vid)] = g
    return out


def load_test_texts():
    o = {}
    for ds in DSS:
        for line in open(os.path.join(GT, ds, "test.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            o[(ds, r["id"])] = strip_html(r.get("text", ""))
    return o


def chunks_of(norm, lang):
    """FREEZE §5.1 chunking."""
    if lang == "zh":
        cs, W = [], 60
        for i in range(0, len(norm), W):
            cs.append(norm[i:i + W])
        if len(cs) > 1 and len(cs[-1]) < 15:
            cs[-2] = cs[-2] + cs[-1]
            cs.pop()
    else:
        w = norm.split()
        cs, W = [], 40
        for i in range(0, len(w), W):
            cs.append(" ".join(w[i:i + W]))
        if len(cs) > 1 and len(cs[-1].split()) < 10:
            cs[-2] = cs[-2] + " " + cs[-1]
            cs.pop()
    return [c for c in cs if c.strip()] or [norm or " "]


# ------------------------------------------------------------------ metrics
def roc_auc(y, p):
    pairs = sorted(zip(p, y))
    ranks, i, n = {}, 0, len(pairs)
    order = [0.0] * n
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            order[k] = r
        i = j + 1
    npos = sum(y)
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return None
    spos = sum(order[k] for k in range(n) if pairs[k][1] == 1)
    return (spos - npos * (npos + 1) / 2.0) / (npos * nneg)


def acc_at(y, p, thr=0.5):
    if not y:
        return None
    return sum(1 for a, b in zip(y, p) if int(b > thr) == a) / len(y)


def summarise(y, p):
    if not y:
        return {"n": 0}
    pos = [q for a, q in zip(y, p) if a == 1]
    neg = [q for a, q in zip(y, p) if a == 0]
    return {"n": len(y), "n_not_own": len(pos), "n_own": len(neg),
            "auc": round(roc_auc(y, p), 4) if roc_auc(y, p) is not None else None,
            "acc@0.5": round(acc_at(y, p), 4),
            "recall_not_own": round(sum(1 for q in pos if q > 0.5) / len(pos), 4) if pos else None,
            "recall_own": round(sum(1 for q in neg if q <= 0.5) / len(neg), 4) if neg else None,
            "mean_p_not_own": round(sum(pos) / len(pos), 4) if pos else None,
            "mean_p_own": round(sum(neg) / len(neg), 4) if neg else None}


# ------------------------------------------------------------------ main
def log(*a):
    print(*a, flush=True)


def main():
    import numpy as np

    out = {"freeze": "idea-stage/SYNTH_PAIR_PROBE_FREEZE.md @ 73971c1"}

    # ---- stage 1: synthesis (train.jsonl only)
    srcs, per_ds = source_sentences()
    if len(srcs) < MIN_PAIRS:
        log("HALT: only %d source sentences, freeze requires >= %d" % (len(srcs), MIN_PAIRS))
        json.dump({"HALT": "insufficient source sentences", "n": len(srcs)},
                  open(os.path.join(ROOT, "idea-stage", "synth_pair_probe.json"), "w"), indent=2)
        return 1
    rows = build_pairs(srcs)
    out["synthesis"] = {"n_source_sentences": len(srcs), "n_pairs": len(srcs),
                        "n_examples": len(rows),
                        "by_lang": dict(Counter(s["lang"] for s in srcs)),
                        "candidates_by_dataset_before_cap": dict(per_ds)}
    log("synthesis:", json.dumps(out["synthesis"]))
    log("sample OWN :", rows[0]["text"][:160])
    log("sample NOT :", rows[1]["text"][:160])

    npairs = len(srcs)
    cut = int(round(0.85 * npairs))
    tr = [r for r in rows if r["pair"] < cut]
    dv = [r for r in rows if r["pair"] >= cut]
    log("train examples %d / dev examples %d" % (len(tr), len(dv)))

    # ---- stage 2: eval item preparation (test.jsonl read here, read-only)
    gold = load_gold_voice()
    sample = json.load(open(os.path.join(ROOT, "idea-stage", "stance_pilot", "sample.json")))
    group = {(r["dataset"], r["id"]): r["group"] for r in sample["eval"]}
    if SMOKE:
        # SMOKE never opens test.jsonl. Fake "transcripts" are stitched from the synthetic
        # dev split so that every downstream code path (chunking, aggregation, metrics,
        # decision) is exercised without any test-set contact.
        log("SMOKE MODE: test.jsonl is NOT read; eval texts are synthetic dev strings")
        pool = [r["text"] for r in dv]
        rr = random.Random(1)
        texts = {}
        for k in list(gold.keys()) + list(group.keys()):
            texts[k] = " ".join(rr.sample(pool, rr.randint(2, 12)))
    else:
        texts = load_test_texts()

    items = []
    for (ds, vid), g in gold.items():
        raw = texts.get((ds, vid))
        if raw is None:
            log("WARN missing test text", ds, vid)
            continue
        lang = lang_of(raw)
        nrm = normalise(raw)
        items.append({"key": "%s::%s" % (ds, vid), "ds": ds, "gold": g, "lang": lang,
                      "group": group.get((ds, vid)), "norm": nrm,
                      "chunks": chunks_of(nrm, lang), "marker": has_marker(nrm, lang),
                      "role": "GOLD"})
    for (ds, vid), grp in group.items():
        if grp not in ("CTRL_HATE", "CTRL_NONHATE"):
            continue
        raw = texts.get((ds, vid))
        if raw is None:
            log("WARN missing test text", ds, vid)
            continue
        lang = lang_of(raw)
        nrm = normalise(raw)
        items.append({"key": "%s::%s" % (ds, vid), "ds": ds, "gold": None, "lang": lang,
                      "group": grp, "norm": nrm, "chunks": chunks_of(nrm, lang),
                      "marker": has_marker(nrm, lang), "role": "CTRL"})
    out["eval_sets"] = {
        "gold_items": sum(1 for i in items if i["role"] == "GOLD"),
        "gold_by_class": dict(Counter(i["gold"] for i in items if i["role"] == "GOLD")),
        "ctrl_items": sum(1 for i in items if i["role"] == "CTRL"),
        "ctrl_by_group": dict(Counter(i["group"] for i in items if i["role"] == "CTRL")),
        "median_chunks": float(np.median([len(i["chunks"]) for i in items])),
        "max_chunks": max(len(i["chunks"]) for i in items)}
    log("eval_sets:", json.dumps(out["eval_sets"]))

    flat, span = [], []
    for i in items:
        span.append((len(flat), len(flat) + len(i["chunks"])))
        flat.extend(i["chunks"])
    trunc256 = [" ".join(i["norm"].split()[:256]) if i["lang"] == "en" else i["norm"][:256]
                for i in items]

    tiers = {}

    # ---- Tier A: multilingual sentence embeddings + logistic regression
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        log("Tier A: encoding ...")
        st = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        Xtr = st.encode([r["text"] for r in tr], batch_size=128, show_progress_bar=False)
        Xdv = st.encode([r["text"] for r in dv], batch_size=128, show_progress_bar=False)
        Xch = st.encode(flat, batch_size=128, show_progress_bar=False)
        Xtc = st.encode(trunc256, batch_size=64, show_progress_bar=False)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
        clf.fit(Xtr, [r["y"] for r in tr])
        dev_acc = float(clf.score(Xdv, [r["y"] for r in dv]))
        tiers["A_mpnet_logreg"] = {"synthetic_dev_acc": round(dev_acc, 4),
                                   "chunk_p": clf.predict_proba(Xch)[:, 1].tolist(),
                                   "trunc_p": clf.predict_proba(Xtc)[:, 1].tolist()}
        log("Tier A synthetic dev acc = %.4f" % dev_acc)
        del Xtr, Xdv, Xch, Xtc, st
    except Exception as e:
        log("Tier A FAILED:", repr(e))

    # ---- length-only control
    try:
        from sklearn.linear_model import LogisticRegression
        L = lambda t, lg: (len(t) if lg == "zh" else len(t.split()))
        lc = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
        lc.fit([[L(r["text"], r["lang"])] for r in tr], [r["y"] for r in tr])
        flat_lang = []
        for i in items:
            flat_lang.extend([i["lang"]] * len(i["chunks"]))
        tiers["LEN_control"] = {
            "synthetic_dev_acc": round(float(lc.score([[L(r["text"], r["lang"])] for r in dv],
                                                      [r["y"] for r in dv])), 4),
            "chunk_p": lc.predict_proba([[L(t, lg)] for t, lg in zip(flat, flat_lang)])[:, 1].tolist(),
            "trunc_p": lc.predict_proba([[L(t, i["lang"])] for t, i in zip(trunc256, items)])[:, 1].tolist()}
        log("LEN control synthetic dev acc = %.4f" % tiers["LEN_control"]["synthetic_dev_acc"])
    except Exception as e:
        log("LEN control FAILED:", repr(e))

    # ---- Tier B: distilbert-base-multilingual-cased fine-tune
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        assert torch.cuda.is_available(), "no CUDA"
        torch.manual_seed(SEED)
        dev = "cuda"
        name = "distilbert-base-multilingual-cased"
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(dev)
        opt = torch.optim.AdamW(mdl.parameters(), lr=2e-5)

        def batches(rs, bs, shuffle):
            idx = list(range(len(rs)))
            if shuffle:
                random.Random(SEED).shuffle(idx)
            for i in range(0, len(idx), bs):
                yield [rs[j] for j in idx[i:i + bs]]

        mdl.train()
        for ep in range(2):
            tot = nb = 0
            for b in batches(tr, 32, True):
                enc = tok([r["text"] for r in b], truncation=True, max_length=64,
                          padding=True, return_tensors="pt").to(dev)
                y = torch.tensor([r["y"] for r in b], device=dev)
                loss = mdl(**enc, labels=y).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                tot += float(loss)
                nb += 1
            log("Tier B epoch %d mean loss %.4f" % (ep, tot / max(nb, 1)))
        mdl.eval()

        @torch.no_grad()
        def score(strs, maxlen):
            ps = []
            for i in range(0, len(strs), 64):
                enc = tok(strs[i:i + 64], truncation=True, max_length=maxlen,
                          padding=True, return_tensors="pt").to(dev)
                ps.extend(torch.softmax(mdl(**enc).logits, -1)[:, 1].tolist())
            return ps

        dp = score([r["text"] for r in dv], 64)
        dacc = sum(1 for q, r in zip(dp, dv) if int(q > 0.5) == r["y"]) / len(dv)
        tiers["B_distilbert_ft"] = {"synthetic_dev_acc": round(dacc, 4),
                                    "chunk_p": score(flat, 64),
                                    "trunc_p": score(trunc256, 256)}
        log("Tier B synthetic dev acc = %.4f" % dacc)
    except Exception as e:
        log("Tier B SKIPPED/FAILED:", repr(e))

    # ---- stage 4: aggregation + frozen metrics
    res = {}
    for tname, t in tiers.items():
        cp = t["chunk_p"]
        agg = {}
        for i, (a, b) in zip(items, span):
            sl = cp[a:b]
            agg[i["key"]] = {"mean": sum(sl) / len(sl), "max": max(sl)}
        for i, q in zip(items, t["trunc_p"]):
            agg[i["key"]]["trunc256"] = q

        block = {"synthetic_dev_acc": t["synthetic_dev_acc"]}
        gitems = [i for i in items if i["role"] == "GOLD" and i["gold"] in ("OWN", "NOT_OWN")]
        for how in ("mean", "max", "trunc256"):
            y = [1 if i["gold"] == "NOT_OWN" else 0 for i in gitems]
            p = [agg[i["key"]][how] for i in gitems]
            block["GOLD_VOICE_" + how] = summarise(y, p)
        # secondary: S_FP / S_FN stratification, PRIMARY aggregation only
        for g in ("S_FP", "S_FN"):
            sub = [i for i in gitems if i["group"] == g]
            block["strat_" + g] = summarise([1 if i["gold"] == "NOT_OWN" else 0 for i in sub],
                                            [agg[i["key"]]["mean"] for i in sub])
        # R1(a): gold==OWN AND marker present -> trap cases
        trap = [i for i in gitems if i["gold"] == "OWN" and i["marker"]]
        notrap = [i for i in gitems if i["gold"] == "OWN" and not i["marker"]]
        block["R1a_trap_gold_OWN_with_marker"] = {
            "n": len(trap),
            "acc": round(sum(1 for i in trap if agg[i["key"]]["mean"] <= 0.5) / len(trap), 4) if trap else None,
            "mean_p_not_own": round(sum(agg[i["key"]]["mean"] for i in trap) / len(trap), 4) if trap else None}
        block["R1a_gold_OWN_no_marker"] = {
            "n": len(notrap),
            "acc": round(sum(1 for i in notrap if agg[i["key"]]["mean"] <= 0.5) / len(notrap), 4) if notrap else None,
            "mean_p_not_own": round(sum(agg[i["key"]]["mean"] for i in notrap) / len(notrap), 4) if notrap else None}
        # R1(b): 50 controls, marker-driven shift
        r1b = {}
        for grp in ("CTRL_HATE", "CTRL_NONHATE"):
            for mk in (True, False):
                sub = [i for i in items if i["role"] == "CTRL" and i["group"] == grp and i["marker"] == mk]
                r1b["%s_%s" % (grp, "marker" if mk else "nomarker")] = {
                    "n": len(sub),
                    "mean_p_not_own": round(sum(agg[i["key"]]["mean"] for i in sub) / len(sub), 4) if sub else None,
                    "rate_p_gt_0.5": round(sum(1 for i in sub if agg[i["key"]]["mean"] > 0.5) / len(sub), 4) if sub else None}
            a = r1b["%s_marker" % grp]["mean_p_not_own"]
            b_ = r1b["%s_nomarker" % grp]["mean_p_not_own"]
            r1b["%s_shift" % grp] = round(a - b_, 4) if (a is not None and b_ is not None) else None
        block["R1b_controls"] = r1b
        block["per_item"] = {i["key"]: {"gold": i["gold"], "group": i["group"],
                                        "marker": i["marker"], "n_chunks": len(i["chunks"]),
                                        "p_mean": round(agg[i["key"]]["mean"], 4)}
                             for i in items}
        res[tname] = block
        log("== %s GOLD_VOICE(mean) %s" % (tname, json.dumps(block["GOLD_VOICE_mean"])))

    out["results"] = res

    # ---- frozen decision rule (FREEZE §6)
    aucs = {k: res[k]["GOLD_VOICE_mean"]["auc"] for k in ("A_mpnet_logreg", "B_distilbert_ft")
            if k in res and res[k]["GOLD_VOICE_mean"].get("auc") is not None}
    best = max(aucs.values()) if aucs else None
    verdict = ("NO_TIER_RAN" if best is None else
               "SIGNAL" if best >= 0.70 else
               "WEAK" if best >= 0.60 else "TRANSFER_FAILURE")
    out["decision"] = {"tier_aucs_primary": aucs, "best_auc": best, "verdict": verdict,
                       "bar": "SIGNAL >=0.70 / WEAK 0.60-0.70 / TRANSFER_FAILURE <0.60"}
    log("DECISION:", json.dumps(out["decision"]))

    dest = os.path.join(ROOT, "idea-stage",
                        "synth_pair_probe_SMOKE.json" if SMOKE else "synth_pair_probe.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log("wrote " + dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
