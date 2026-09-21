"""Per-second transcript embedding on the SAME utterance units that scripts/build_text_hate_scores.py
scores for the text term x_t (module-1 iteration 6, "unified text encoder", 2026-09-21).

--encoder hate_roberta (default): cardiffnlp/twitter-roberta-base-hate-latest (frozen, HF cache, offline),
  <s> token of the last hidden state (768-d) = exactly what the checkpoint's classification head reads
  (RobertaClassificationHead takes features[:, 0]); the head's probability over the same utterances is
  x_t, so the backbone's text rows and the prior's text term are two read-outs of one hidden state.
  Output data/hate_text_1fps/. Self-check: the head applied to the stored vectors reproduces the
  model's own logits (max abs diff printed; must be ~0).
--encoder bert_utterance: bert-base-uncased CLS (last hidden state [:, 0], no pooler) on the same
  utterances -- the control that isolates the encoder from the transcript units. Output
  data/bert_utterance_1fps/.

Both: max_length 128, truncation. Units: utterances() of build_text_hate_scores applied to the data/ASR
records (both corpora mix word-level and chunk-level Whisper records; word records are merged at
sentence punctuation / a pause > 1 s / 40 words, chunk records are used as they are). Frame rule as
scripts/reproduction_baselines/multihateloc/extract_bert_sentence_features.assign_frames: frame i covers
[i, i + 1); the utterance with the largest overlap owns the frame (ties: the earlier one), zero vector
where no speech; note x_t instead averages all utterances covering a second (22-24% of speech seconds
are covered by more than one). T = hier_evidence_common.video_duration (VGGish rows), the grid
data/text_hate uses. ASR only: OCR window text is not embedded (x_t keeps ASR + OCR).

The BERT rows these replace (results/reproduction/features/bert_sentence_1fps, MultiHateLoc
reproduction) come from a DIFFERENT Whisper manifest (results/reproduction/asr/<corpus>_all/
timestamped_chunks.jsonl, chunk-level, max 64 tokens), so "bert" -> "hate_roberta" changes the
transcript source, the units and the encoder at once; "bert_utterance" -> "hate_roberta" changes the
encoder only.

Output: <root>/<corpus>/<video_id>.npy float32 (T, 768); <corpus>/index.json with per-video coverage.
No label is read.
"""
import argparse, json, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts/reproduction_baselines")); sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "scripts/reproduction_baselines/multihateloc"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from hate_common import data as hdata
import hier_evidence_common as hc
from build_text_hate_scores import MODEL, ASR, utterances
from extract_bert_sentence_features import assign_frames
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

ENCODERS = {"hate_roberta": (MODEL, os.path.join(ROOT, "data", "hate_text_1fps")),
            "bert_utterance": ("bert-base-uncased", os.path.join(ROOT, "data", "bert_utterance_1fps"))}
DIM = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--encoder", default="hate_roberta", choices=sorted(ENCODERS))
    a = ap.parse_args()
    model_id, out_root = ENCODERS[a.encoder]
    out_dir = os.path.join(out_root, a.corpus)
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if a.encoder == "hate_roberta":
        mod = AutoModelForSequenceClassification.from_pretrained(model_id).to(a.device).eval()
    else:
        mod = AutoModel.from_pretrained(model_id, add_pooling_layer=False).to(a.device).eval()
    assert int(mod.config.hidden_size) == DIM, mod.config.hidden_size

    @torch.no_grad()
    def encode(texts):
        """(first-token hidden state (n, 768), model logits (n, 2), head-on-stored-vector logits (n, 2));
        the two logit arrays are zeros for the plain BERT encoder (no head)."""
        vecs, logits, relog = [], [], []
        for i in range(0, len(texts), a.batch):
            x = tok(texts[i:i + a.batch], return_tensors="pt", padding=True, truncation=True, max_length=128).to(a.device)
            if a.encoder == "hate_roberta":
                out = mod(**x, output_hidden_states=True)
                h = out.hidden_states[-1]                   # (b, L, 768); [:, 0] is the head's input
                logits.append(out.logits.float().cpu().numpy())
                relog.append(mod.classifier(h[:, :1]).float().cpu().numpy())
            else:
                h = mod(**x).last_hidden_state
                logits.append(np.zeros((h.shape[0], 2), np.float32)); relog.append(np.zeros((h.shape[0], 2), np.float32))
            vecs.append(h[:, 0].float().cpu().numpy())
        return (np.concatenate(vecs), np.concatenate(logits), np.concatenate(relog)) if vecs else \
            (np.zeros((0, DIM), np.float32), np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))

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
    for p in ASR[a.corpus]:
        for line in open(os.path.join(ROOT, p)):
            r = json.loads(line)
            if r["id"] in T and r["id"] not in asr:
                asr[r["id"]] = [(s, e, t) for s, e, t in utterances(r) if str(t).strip()]
    index, max_diff, n_units, n_cov = {}, 0.0, 0, 0
    for v in T:
        n = T[v]
        feats = np.zeros((n, DIM), np.float32)
        units = asr.get(v, [])
        covered = 0
        if units:
            chunks = [{"start": float(s), "end": float(e), "text": str(t).strip()} for s, e, t in units]
            owner = assign_frames(chunks, n)
            used = sorted({int(c) for c in owner if c >= 0})
            if used:
                vecs, logits, relog = encode([chunks[c]["text"] for c in used])
                max_diff = max(max_diff, float(np.abs(logits - relog).max()))
                slot = {c: k for k, c in enumerate(used)}
                for f in range(n):
                    c = int(owner[f])
                    if c >= 0:
                        feats[f] = vecs[slot[c]]
                n_units += len(used)
            covered = int((owner >= 0).sum())
        tmp = os.path.join(out_dir, v + ".tmp.npy")
        np.save(tmp, feats)
        os.replace(tmp, os.path.join(out_dir, v + ".npy"))
        index[v] = {"n_frames": n, "dim": DIM, "model": model_id, "n_utterances": len(units),
                    "n_frames_covered": covered, "coverage": round(covered / n, 6) if n else 0.0}
        n_cov += int(covered > 0)
    with open(os.path.join(out_dir, "index.json"), "w") as fh:
        json.dump(index, fh, indent=1, sort_keys=True)
    print("%s [%s]: %d videos written, %d with speech rows, %d utterances embedded, head-vs-model logit max abs diff %.2e"
          % (a.corpus, a.encoder, len(T), n_cov, n_units, max_diff))


if __name__ == "__main__":
    main()
