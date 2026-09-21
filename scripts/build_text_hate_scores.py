"""Per-second hate probability from a frozen text classifier over the ASR chunks
and the K=30 OCR windows (module-1 iteration 2 "text evidence", 2026-09-10; word-level
Whisper records are merged into utterances first, see utterances()).

--asr-source asr (default): the ASR chunks of data/ASR (word records merged into utterances, see
utterances()); output data/text_hate/. --asr-source chunks (module-1 iteration 6b, 2026-09-22): the
sentence-level Whisper chunks of results/reproduction/asr/<corpus>_all/timestamped_chunks.jsonl (the
transcript the backbone's BERT rows were built from; read with the same load_chunks as
scripts/reproduction_baselines/multihateloc/extract_bert_sentence_features.py, spans < 1 s widened to 1 s);
output data/text_hate_chunks/. OCR windows are the same in both.

Output: <out root>/<corpus>/<video_id>.npz with
  p_asr (T,)  hate probability of the ASR chunk covering second t, NaN = no speech
  w_asr (T,)  1 / (seconds covered by that chunk): one chunk counts once in total
  p_ocr (T,)  hate probability of the OCR text of the K=30 window covering t, NaN = no text
  w_ocr (T,)  1 / (seconds of that window)
T = seconds of the video = rows of its VGGish array (hier_evidence_common.video_duration),
the same D the verdict windows use. No label is read. Model: cardiffnlp/twitter-roberta-base-hate-latest
(frozen, HF cache, offline), P(HATE) of the softmax, max_length 128, truncation.
"""
import argparse, json, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts/reproduction_baselines")); sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts/reproduction_baselines/multihateloc")); sys.path.insert(0, os.path.join(ROOT, "scripts/duplex"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from hate_common import data as hdata
import hier_evidence_common as hc
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "cardiffnlp/twitter-roberta-base-hate-latest"
ASR = {"hatemm": ["data/ASR/HateMM/train_asrK30_whisper-large-v3.jsonl", "data/ASR/HateMM/dev_seen_asrK30_whisper-large-v3.jsonl",
                  "data/ASR/HateMM/test_seen_asrK30_whisper-large-v3.jsonl"],
       "hateclipseg": ["data/ASR/HateClipSeg/test_seen_asrK30_whisper-large-v3.jsonl"]}
OCR = {"hatemm": ["data/OCR/HateMM/ocr_windows_K30.jsonl", "data/OCR/HateMM/ocr_windows_K30_test.jsonl"],
       "hateclipseg": ["data/OCR/HateClipSeg/ocr_windows_K30.jsonl"]}
K = 30
WORD_GAP = 1.0        # word-level records: a pause longer than this starts a new utterance
WORD_MAX = 40         # ... or 40 words


def utterances(rec):
    """[(start, end, text)] with real spans for both Whisper record kinds:
    timestamps == "chunk": the chunks as they are; timestamps == "word": consecutive
    words merged into utterances at sentence-final punctuation (. ? !), a pause
    longer than WORD_GAP s, or WORD_MAX words (a single word is not a scoring unit:
    the classifier gives ~.002 to any isolated word). Spans shorter than 1 s are
    widened to 1 s."""
    chunks = [(float(s), float(e), str(t).strip()) for s, e, t in rec.get("chunks", []) if str(t).strip()]
    if rec.get("timestamps") != "word":
        out = chunks
    else:
        out, cur = [], []
        for s, e, t in chunks:
            if cur and (s - cur[-1][1] > WORD_GAP or len(cur) >= WORD_MAX):
                out.append(cur); cur = []
            cur.append((s, e, t))
            if t[-1] in ".?!":
                out.append(cur); cur = []
        if cur:
            out.append(cur)
        out = [(u[0][0], max(w[1] for w in u), " ".join(w[2] for w in u)) for u in out]
    return [(s, max(e, s + 1.0), t) for s, e, t in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-ocr-conf", type=float, default=0.5)
    ap.add_argument("--asr-source", default="asr", choices=("asr", "chunks"))
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, "data/text_hate" if a.asr_source == "asr" else "data/text_hate_chunks", a.corpus)
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModelForSequenceClassification.from_pretrained(MODEL).to(a.device).eval()

    @torch.no_grad()
    def hate_prob(texts):
        out = []
        for i in range(0, len(texts), 64):
            x = tok(texts[i:i + 64], return_tensors="pt", padding=True, truncation=True, max_length=128).to(a.device)
            out.append(torch.softmax(mod(**x).logits, -1)[:, 1].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    ids = []
    for split in ("train", "val", "test"):
        ids += hdata.load_split(a.corpus, split)
    ids = [v for v in dict.fromkeys(ids)]
    T = {}
    for v in ids:
        try:
            T[v] = int(hc.video_duration(a.corpus, v))
        except Exception:
            pass
    asr = {}
    if a.asr_source == "asr":
        for p in ASR[a.corpus]:
            for line in open(os.path.join(ROOT, p)):
                r = json.loads(line)
                if r["id"] in T and r["id"] not in asr:
                    asr[r["id"]] = utterances(r)
    else:
        from extract_bert_sentence_features import load_chunks
        from extract_clip_features import CORPORA
        for v, chunks in load_chunks(CORPORA[a.corpus]).items():
            if v in T:
                asr[v] = [(float(c["start"]), max(float(c["end"]), float(c["start"]) + 1.0), c["text"]) for c in chunks]
    ocr = {}
    for p in OCR[a.corpus]:
        for line in open(os.path.join(ROOT, p)):
            r = json.loads(line)
            v = r["video_id"]
            if v in T:
                t = " ".join(x["text"] for x in r["texts"] if float(x.get("conf", 1.0)) >= a.min_ocr_conf).strip()
                if t:
                    ocr.setdefault(v, {})[int(r["window_k"])] = t
    texts, owners = [], []
    for v in T:
        for (s, e, t) in asr.get(v, []):
            if str(t).strip():
                texts.append(str(t).strip()); owners.append(("asr", v, float(s), float(e)))
        for k, t in ocr.get(v, {}).items():
            texts.append(t); owners.append(("ocr", v, k, None))
    probs = hate_prob(texts)
    arr = {v: {"p_asr": np.full(T[v], np.nan, np.float32), "w_asr": np.zeros(T[v], np.float32),
               "p_ocr": np.full(T[v], np.nan, np.float32), "w_ocr": np.zeros(T[v], np.float32)} for v in T}
    acc = {v: (np.zeros(T[v]), np.zeros(T[v])) for v in T}          # asr: sum p, count (overlapping chunks averaged)
    for (fam, v, s, e), p in zip(owners, probs):
        n = T[v]
        if fam == "asr":
            a0, b0 = max(0, int(np.floor(s))), min(n, max(int(np.ceil(e)), int(np.floor(s)) + 1))
            if b0 <= a0:
                continue
            acc[v][0][a0:b0] += p; acc[v][1][a0:b0] += 1
            arr[v]["w_asr"][a0:b0] = np.maximum(arr[v]["w_asr"][a0:b0], 1.0 / (b0 - a0))
        else:
            a0, b0 = int(s * n / K), max(int((s + 1) * n / K), int(s * n / K) + 1)
            b0 = min(b0, n)
            arr[v]["p_ocr"][a0:b0] = p
            arr[v]["w_ocr"][a0:b0] = 1.0 / (b0 - a0)
    n_asr = n_ocr = 0
    for v in T:
        sp, c = acc[v]
        m = c > 0
        arr[v]["p_asr"][m] = (sp[m] / c[m]).astype(np.float32)
        n_asr += int(m.any()); n_ocr += int(np.isfinite(arr[v]["p_ocr"]).any())
        np.savez(os.path.join(out_dir, "%s.npz" % v), **arr[v])
    print("%s [asr-source %s -> %s]: %d videos written, %d with speech text, %d with OCR text, %d texts scored"
          % (a.corpus, a.asr_source, out_dir, len(T), n_asr, n_ocr, len(texts)))


if __name__ == "__main__":
    main()
