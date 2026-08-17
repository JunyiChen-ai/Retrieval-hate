"""S_PRIZE_DECOMP cross-checks: annotator-split enrichment, voice=none rate, showcase rows."""
import ast, csv, json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = json.load(open(os.path.join(HERE, "s_prize_decomp.json")))
rows = D["rows"]
POS = {"Hateful", "Offensive"}


def load_votes(lang):
    V = {}
    for sp in ("train", "valid", "test"):
        p = os.path.join(ROOT, "data", "gt", "mhc_votes", f"mhc_{lang}_{sp}.tsv")
        for r in csv.DictReader(open(p, newline="", encoding="utf-8"), delimiter="\t"):
            v = [x for x in ast.literal_eval(r["Label"])
                 if x in ("Hateful", "Offensive", "Normal", "Counter Narrative")]
            V[r["Video_ID"].strip()] = [1 if x in POS else 0 for x in v]
    return V


votes = {"MHC": load_votes("English"), "MHC_zh": load_votes("Chinese")}
out = {}
tab = collections.defaultdict(lambda: {"n": 0, "split": 0})
for r in rows:
    if r["dataset"] not in votes:
        continue
    v = votes[r["dataset"]].get(r["id"])
    if not v:
        continue
    is_split = 0 < sum(v) < len(v)
    tab[r["cls"]]["n"] += 1
    tab[r["cls"]]["split"] += int(is_split)
    tab[f'{r["dataset"]}/{r["cls"]}']["n"] += 1
    tab[f'{r["dataset"]}/{r["cls"]}']["split"] += int(is_split)
    r["human_votes"] = v
    r["human_split"] = is_split
out["annot_split_by_cls_MHC_both"] = {k: {**v, "rate": round(v["split"] / v["n"], 3)}
                                      for k, v in sorted(tab.items()) if v["n"]}

# voice = none majority among the 3 Claude raters
for r in rows:
    vs = [x for x in r["voices"].values() if x]
    c = collections.Counter(vs)
    r["voice_maj"] = c.most_common(1)[0][0] if c and (len(c) == 1 or c.most_common(1)[0][1] >
                                                      c.most_common(2)[1][1]) else "TIE"
out["voice_none_rate_by_cls"] = {}
for cls in ("RECOVERABLE", "CONTESTED", "SPLIT"):
    sub = [r for r in rows if r["cls"] == cls]
    out["voice_none_rate_by_cls"][cls] = {
        "n": len(sub), "voice_none_maj": sum(1 for r in sub if r["voice_maj"] == "none"),
        "rate": round(sum(1 for r in sub if r["voice_maj"] == "none") / len(sub), 3)}

# qwen R2 5-way label distribution per class
out["qwen_r2_5way_by_cls"] = {
    cls: dict(collections.Counter(r["qwen_r2_5way"] for r in rows if r["cls"] == cls))
    for cls in ("RECOVERABLE", "CONTESTED", "SPLIT")}

# transcripts for showcase
dump = json.load(open(os.path.join(HERE, "r5_error_dump.json")))
tx = {(ds, e["id"]): e for ds, es in dump.items() for e in es}
for r in rows:
    e = tx.get((r["dataset"], r["id"]), {})
    r["transcript"] = (e.get("transcript") or "")
    r["ocr"] = (e.get("ocr") or "")
    r["err_kind"] = e.get("kind")
    r["score"] = e.get("score")

json.dump({"rows": rows, **out}, open(os.path.join(HERE, "s_prize_xcheck.json"), "w"),
          indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
print("\nCONTESTED items (4-0 first):")
for r in sorted([r for r in rows if r["cls"] == "CONTESTED"],
                key=lambda x: (x["k_for_gold"], x["dataset"])):
    print(f"  {r['dataset']}/{r['id']:<22} {r['group']} gold={r['gold']:<10} "
          f"k={r['k_for_gold']}/{r['n_valid']} voice={r['voice_maj']:<18} "
          f"hsplit={r.get('human_split')} tx_len={len(r['transcript'])}")
