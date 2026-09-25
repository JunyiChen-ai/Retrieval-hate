"""Query-tree manifest (README section 2.1): for every video of HateMM and HateClipSeg (all splits), the binary
partition tree of its 1-fps timeline and, for each node the VLM can be asked about, the 4 frame indices and the
node's transcript.

Tree: root = [0, T) with T = rows of the video's 1-fps grid (= GT length); a node [a, b) splits at
m = a + (b - a) // 2. A node is queryable when b - a >= F (F = 4 frames per question at 1 fps, so the 4 frames
are distinct seconds); the tree itself continues to single seconds for inference.
Frames: 1-fps frame a + floor((b - a) * (i + .5) / F), i = 0..F-1, clipped to the frames on disk.
Transcript: Whisper large-v3 word chunks (data/ASR/<Corpus>/*_asrK30_whisper-large-v3.jsonl, `chunks`) whose
midpoint lies in [a, b) (the midpoint rule of src/utils/generate_segment_asr_HF.py).

    python experiments/20260925_query_paradigm/build_tree_manifest.py [--corpus hatemm hateclipseg dehate]
Output: data/vlm_tree/<Corpus>/manifest.jsonl (+ PROVENANCE.md written by hand)

DeHate (external validation, 2026-09-26; experiments/20260926_dehate_external/README.md): the transcript is the
sentence-level Whisper large-v3 run of the reproduction study (results/reproduction/asr/dehate_all, the chunks behind
the BERT rows), assigned to nodes by the same midpoint rule. The word-level run above does not fit on the 32 GB GPU
for DeHate's 5-minute videos (it fell back to sentence chunks for 51% of HateMM and 62% of HateClipSeg videos
already). The DeHate manifest is built before the I3D features exist, so it lists every video with a VGGish array
(which fixes T); training and evaluation keep only videos with all features (hc.usable).
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hate_common import data as hdata  # noqa: E402
import hier_evidence_common as hc      # noqa: E402
from macilsd import align              # noqa: E402

F = 4
CORPUS_DIR = {"hatemm": "HateMM", "hateclipseg": "HateClipSeg", "dehate": "DeHate"}
REPRO_ASR = {"dehate": os.path.join(ROOT, "results", "reproduction", "asr", "dehate_all", "timestamped_chunks.jsonl")}


def tree_nodes(T, min_len=F):
    """Queryable nodes of the binary partition tree of [0, T), breadth-first (root first)."""
    out, level = [], [(0, T)]
    while level:
        nxt = []
        for a, b in level:
            if b - a < min_len:
                continue
            out.append((a, b))
            m = a + (b - a) // 2
            nxt += [(a, m), (m, b)]
        level = nxt
    return out


def frame_idx(a, b, n_frames):
    return [min(a + int((b - a) * (i + 0.5) / F), n_frames - 1) for i in range(F)]


def load_words(corpus):
    words = {}
    if corpus in REPRO_ASR:
        for line in open(REPRO_ASR[corpus]):
            r = json.loads(line)
            words[r["video_id"]] = [(0.5 * (c["start"] + c["end"]), c["text"]) for c in (r.get("chunks") or [])
                                    if c.get("text") and c["text"].strip() and c.get("start") is not None
                                    and c.get("end") is not None]
        return words
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "ASR", CORPUS_DIR[corpus], "*_asrK30_whisper-large-v3.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if r["id"] not in words:
                words[r["id"]] = [(0.5 * (s + e), t) for s, e, t in (r.get("chunks") or []) if t and t.strip()]
    return words


def main(corpora):
    for corpus in corpora:
        d = CORPUS_DIR[corpus]
        words = load_words(corpus)
        out_dir = os.path.join(ROOT, "data", "vlm_tree", d)
        os.makedirs(out_dir, exist_ok=True)
        n_vid = n_node = 0
        missing_asr = 0
        with open(os.path.join(out_dir, "manifest.jsonl"), "w") as fh:
            for split in ("train", "val", "test"):
                ids = hdata.load_split(corpus, split)
                ids = ([v for v in ids if os.path.exists(os.path.join(align.AUDIO_ROOT, corpus, v + ".npy"))]
                       if corpus in REPRO_ASR else hc.usable(corpus, ids))
                for v in ids:
                    fdir = os.path.join(ROOT, "data", "frames_1fps", d, v)
                    if not os.path.isdir(fdir):
                        print("no frames", corpus, v)
                        continue
                    n_frames = len([f for f in os.listdir(fdir) if f.endswith(".jpg")])
                    T = int(hc.video_duration(corpus, v))
                    w = words.get(v)
                    if w is None:
                        missing_asr += 1
                        w = []
                    nodes = []
                    for a, b in tree_nodes(T):
                        text = "".join(t for mid, t in w if a <= mid < b or (b == T and mid >= T)).strip()
                        nodes.append([a, b, frame_idx(a, b, n_frames), text])
                    fh.write(json.dumps({"id": v, "split": split, "T": T, "n_frames": n_frames, "nodes": nodes}) + "\n")
                    n_vid += 1
                    n_node += len(nodes)
        print("%s: %d videos, %d queryable nodes, %d videos without ASR record" % (corpus, n_vid, n_node, missing_asr))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", default=["hatemm", "hateclipseg"], choices=tuple(CORPUS_DIR))
    main(ap.parse_args().corpus)
