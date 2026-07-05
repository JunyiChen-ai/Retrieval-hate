"""P4 archive-field schema: single source of truth for the four auxiliary-distillation
targets (target group, attack mechanism, evidence modality, explicitness).

Both the probe-gate (scripts/analysis/p4_probe_gate.py) and the training-time aux loss
(src/run_rac.py) import from here so the vocabulary and encoding are guaranteed identical.

Fields (all derived from the MLLM structured archive, which is generated from the video
content alone -- NO gold labels -- so using it on TRAIN videos is leakage-clean):
  explicitness  : single-label, 3 exhaustive classes {none, implicit, explicit} (CE).
  modality      : multi-label over the 3 evidence cues {visual, speech, on_screen_text},
                  label = that cue string is non-empty (BCE).
  mechanism     : multi-label over top-N mechanisms (>=90% train mass) + OTHER (BCE).
  target_group  : multi-label over top-N named groups + OTHER (BCE). Long-tailed; OTHER is
                  large -- the probe gate is the arbiter of whether it is usable.

Vocabularies are DERIVED from the TRAIN-split archives only and FROZEN to a JSON file so a
run always loads the same vocab (never re-derives silently).
"""
import json
import os
from collections import Counter

ROOT = "/data/jehc223/RGCL"
ARCHIVE_TAG = "Qwen2.5-VL-7B-Instruct_archive"

EXPLICIT_CLASSES = ["none", "implicit", "explicit"]   # exhaustive, order frozen
MODALITY_KEYS = ["visual", "speech", "on_screen_text"]  # exhaustive, order frozen


def _norm(s):
    return str(s).strip().lower()


def archive_path(ds, split, version="v2"):
    if version == "v1":
        return os.path.join(ROOT, "data/Archive", ds, "{}_{}.jsonl".format(split, ARCHIVE_TAG))
    return os.path.join(ROOT, "data/Archive", ds, version,
                        "{}_{}.jsonl".format(split, ARCHIVE_TAG))


def load_archive_records(ds, split, version="v2"):
    """id -> full archive record (last write wins)."""
    recs = {}
    with open(archive_path(ds, split, version)) as f:
        for line in f:
            r = json.loads(line)
            recs[r["id"]] = r
    return recs


def derive_vocab(train_records, topn_mech=3, topn_tg=3):
    """Deterministic top-N (by count desc, name asc tie-break) + OTHER for the two
    open-vocabulary multi-label fields. explicitness/modality are exhaustive."""
    mech, tg = Counter(), Counter()
    for r in train_records.values():
        a = r.get("archive") or {}
        for m in (a.get("mechanism") or []):
            mech[_norm(m)] += 1
        for t in (a.get("target_groups") or []):
            tg[_norm(t)] += 1

    def topn(counter, n):
        items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return [k for k, _ in items[:n]]

    def coverage(counter, chosen):
        tot = sum(counter.values())
        cov = sum(counter[k] for k in chosen)
        return (cov / tot) if tot else 0.0

    mech_top = topn(mech, topn_mech)
    tg_top = topn(tg, topn_tg)
    return {
        "explicitness": {"type": "single", "classes": list(EXPLICIT_CLASSES)},
        "modality": {"type": "multi", "classes": list(MODALITY_KEYS),
                     "coverage": 1.0},
        "mechanism": {"type": "multi", "classes": mech_top + ["OTHER"],
                      "coverage": round(coverage(mech, mech_top), 4),
                      "n_unique": len(mech)},
        "target_group": {"type": "multi", "classes": tg_top + ["OTHER"],
                         "coverage": round(coverage(tg, tg_top), 4),
                         "n_unique": len(tg)},
    }


def freeze_vocab(vocab, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)


def load_vocab(path):
    with open(path) as f:
        return json.load(f)


def field_dims(vocab):
    """Output width per field for the aux linear heads."""
    dims = {}
    for name, spec in vocab.items():
        if spec["type"] == "single":
            dims[name] = len(spec["classes"])
        else:
            dims[name] = len(spec["classes"])
    return dims


def encode_record(rec, vocab):
    """Return {field: (target_list, valid_bool)} for one archive record.

    rec = archive record dict (must have 'archive'); if None or not parse_ok, every field
    is invalid (masked). Multi-label targets are 0/1 lists over the frozen classes;
    explicitness target is the class index (or -1 when invalid).
    """
    out = {}
    ok = rec is not None and rec.get("parse_ok", True) and (rec.get("archive") is not None)
    a = (rec.get("archive") if rec else None) or {}

    # explicitness (single-label CE)
    ev = _norm(a.get("explicitness")) if a.get("explicitness") is not None else None
    if ok and ev in EXPLICIT_CLASSES:
        out["explicitness"] = (EXPLICIT_CLASSES.index(ev), True)
    else:
        out["explicitness"] = (-1, False)

    # modality (multi-label presence of each cue)
    mc = a.get("modality_cues") or {}
    mod = [1.0 if (mc.get(k) or "").strip() else 0.0 for k in MODALITY_KEYS]
    out["modality"] = (mod, ok)

    # mechanism / target_group (multi-label over top-N + OTHER)
    for name, raw_key in (("mechanism", "mechanism"),
                          ("target_group", "target_groups")):
        classes = vocab[name]["classes"]
        named = classes[:-1]           # last is OTHER
        vec = [0.0] * len(classes)
        vals = [_norm(x) for x in (a.get(raw_key) or [])]
        for v in vals:
            if v in named:
                vec[named.index(v)] = 1.0
            else:
                vec[-1] = 1.0           # OTHER
        out[name] = (vec, ok)

    return out
