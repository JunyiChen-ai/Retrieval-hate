"""TEXT_MERGE -- frozen arm text construction (idea-stage/TEXT_MERGE_FREEZE.md section 3).

No model, no GPU: this module only builds, per video and per arm, the string that is
placed in the `Transcript: ` slot of the *unchanged* deployed encoder prompt.

Defect rule, description text template and the mismatch permutation are imported /
reproduced verbatim from the previous experiment (idea-stage/desc_channel/) and are
NOT redefined here.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DESC_DIR = os.path.join(ROOT, "idea-stage", "desc_channel")
sys.path.insert(0, DESC_DIR)
from defect import is_defect, load_gt  # noqa: E402

DESC_JSONL = os.path.join(DESC_DIR, "descriptions_hatemm.jsonl")
FIELDS = ["scene", "people", "actions", "on_screen_text",
          "production_format", "audio_visible_cues"]
DESC_TMPL = ("Scene: {scene}\nPeople: {people}\nActions: {actions}\n"
             "On-screen text: {on_screen_text}\nFormat: {production_format}\n"
             "Audio cues: {audio_visible_cues}")
PERM_SEED = 20260813          # identical to desc_channel/build_desc_feats.py
ARMS = ["A0", "TMt", "TMall", "TMshuf"]
SPLIT_OUT = {"train": "train", "val": "dev_seen", "test": "test_seen"}


def desc_text(fields):
    if not fields:
        return ""
    return DESC_TMPL.format(**{k: (fields.get(k) or "") for k in FIELDS})


def load_desc():
    d = {}
    with open(DESC_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[r["id"]] = desc_text(r.get("fields"))
    return d


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def merge(transcript, description):
    """Frozen merge rule.

    - no description available -> transcript unchanged
    - transcript empty/whitespace -> the description REPLACES it
    - otherwise -> the description is APPENDED after the transcript, one newline apart
    """
    if not description:
        return transcript
    if not (transcript or "").strip():
        return description
    return transcript + "\n" + description


def build(root=ROOT):
    """-> (ids sorted, gt, arm_text: {arm: {vid: str}}, defect: {vid: bool})"""
    gt = load_gt(root)
    ids = sorted(gt)
    n = len(ids)
    desc = load_desc()
    missing = [v for v in ids if v not in desc]
    if missing:
        raise SystemExit("descriptions missing for %d ids (first: %s)"
                         % (len(missing), missing[:5]))

    rng = np.random.default_rng(PERM_SEED)
    perm = derangement(n, rng)           # video i receives the description of ids[perm[i]]
    mis = {ids[i]: desc[ids[perm[i]]] for i in range(n)}

    defect = {v: is_defect(gt[v]["text"]) for v in ids}
    arm_text = {a: {} for a in ARMS}
    for v in ids:
        t = gt[v]["text"]
        arm_text["A0"][v] = t
        arm_text["TMall"][v] = merge(t, desc[v])
        arm_text["TMt"][v] = merge(t, desc[v]) if defect[v] else t
        arm_text["TMshuf"][v] = merge(t, mis[v]) if defect[v] else t
    return ids, gt, arm_text, defect


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    ids, gt, arm_text, defect = build()
    dfl = sorted(v for v in ids if defect[v])
    print("videos %d  defect %d" % (len(ids), len(dfl)))
    print("DEFECT_LIST_SHA256 %s" % digest(dfl))
    print("DESC_JSONL_SHA256  %s" % hashlib.sha256(
        open(DESC_JSONL, "rb").read()).hexdigest())
    for a in ARMS:
        ch = sum(1 for v in ids if arm_text[a][v] != arm_text["A0"][v])
        print("arm %-7s rows differing from A0: %4d   text_sha256 %s"
              % (a, ch, digest({v: arm_text[a][v] for v in ids})))
    import collections
    c = collections.Counter()
    for v in dfl:
        c["empty" if not (gt[v]["text"] or "").strip() else "garbled"] += 1
    print("defect breakdown:", dict(c))
