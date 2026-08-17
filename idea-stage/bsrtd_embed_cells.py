#!/usr/bin/env python
"""B-SRTD cell embedder — TEXT STREAM ONLY, train + val only.

Rules frozen in research-wiki/EXP_bsrtd_prereg.md.

WHY ONLY THE TEXT STREAM
    In src/utils/generate_VideoMLLM_embedding_HF.py the img stream is produced from
    8 frames + a FIXED neutral instruction and never sees the title or the transcript
    (process_split, span="prefix").  It is therefore invariant under every B-SRTD text
    intervention BY CONSTRUCTION.  Re-extracting it would only inject GPU non-determinism
    into a quantity that must be identical across the four cells of a lattice.  The pilot
    pairs the BANKED img_feats from
        data/CLIP_Embedding/<DS>/{train,dev_seen}_Qwen2.5-VL-7B-Instruct_HF.pt
    with the per-cell text_feats produced here.  Same pattern as the C02 density views
    (src/utils/generate_c02_density_view_text_embedding_HF.py).

REUSE, NOT REWRITE
    The frame sampler, chat template, pooling span, instruction constants and prompt
    assembly are imported UNMODIFIED from the deployed extractor.  The only new surface
    here is (a) which STRING goes in the transcript slot and (b) the output contract.
    The extractor's sha256 is printed and, when --pin-sha is given, asserted.

OUTPUT (one file per dataset x split; never overwrites a deployed cache)
    data/CLIP_Embedding/<DS>/{train,dev_seen}_bsrtdcells_Qwen2.5-VL-7B-Instruct_HF.pt
      {"ids": [[seed_id, ...]], "cells": ["orig","targetsub","stancerev","both"],
       "text_feats": FloatTensor[M, 4, D], "labels": LongTensor[M],
       "cell_expected_labels": LongTensor[M, 4], "text_sha256": [[sha, ...], ...],
       "student_tag": ..., "engine": "qwen2.5-vl-7b"}
    Deliberately NO "img_feats" key, so src/run_rac.py's loader cannot consume a cell file
    as if it were a full cache.

RESUMABLE
    Per-seed shards under data/CLIP_Embedding/<DS>/_bsrtdcells_shards/<split>/<seed_id>.pt.
    Re-running skips any shard whose four text_sha256 match the current lattice text.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))

from bsrtd_lattice import (  # noqa: E402
    BSRTD_DIR, CELL_TAG, CELLS, LANG2DS, SPLIT2CACHE, STUDENT_TAG, load_lattices,
    normalise_row, text_sha)
import utils.generate_VideoMLLM_embedding_HF as EX  # noqa: E402

EXTRACTOR = os.path.join(_REPO, "src", "utils", "generate_VideoMLLM_embedding_HF.py")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_text_prompt(title, transcript):
    """Byte-identical to the deployed assembly in EX.process_split (English defaults)."""
    return (EX.TEXT_INSTRUCTION
            + "\n" + "Title: " + (title if title else "(none)")
            + "\n" + "Transcript: " + (transcript if transcript else "(none)"))


def main():
    ap = argparse.ArgumentParser(description="B-SRTD lattice cell text embedder")
    ap.add_argument("--lattice-root", default=BSRTD_DIR)
    ap.add_argument("--langs", default="en,zh")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out-tag", default=CELL_TAG)
    ap.add_argument("--video-dir", default=os.path.join(_REPO, "data", "video"))
    ap.add_argument("--emb-dir", default=os.path.join(_REPO, "data", "CLIP_Embedding"))
    ap.add_argument("--num-frames", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=360 * 420)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pin-sha", default=None,
                    help="if given, assert the deployed extractor's sha256 matches")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate lattices, video paths and output paths; load no model")
    a = ap.parse_args()

    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    assert "test" not in splits and "test_seen" not in splits, \
        "REFUSING: B-SRTD lattices are train/val only; the test split is never embedded"
    langs = [s.strip() for s in a.langs.split(",") if s.strip()]

    sha = sha256_file(EXTRACTOR)
    print(f"deployed extractor sha256 = {sha}", flush=True)
    if a.pin_sha:
        assert sha == a.pin_sha, (
            f"extractor changed: {sha} != pinned {a.pin_sha}; re-freeze before proceeding")

    if a.dry_run:
        bad = 0
        for lang in langs:
            ds = LANG2DS[lang]
            for split in splits:
                rows = [normalise_row(r, split, lang)
                        for r in load_lattices(split, lang, root=a.lattice_root)]
                vr = os.path.join(a.video_dir, ds, "All")
                miss = [r["seed_id"] for r in rows
                        if not os.path.exists(os.path.join(vr, f"{r['seed_id']}.mp4"))]
                bad += len(miss)
                out_path = os.path.join(a.emb_dir, ds,
                                        f"{SPLIT2CACHE[split]}_{a.out_tag}.pt")
                print(f"[{ds}/{split}] lattices={len(rows)} missing_videos={len(miss)} "
                      f"-> {out_path}", flush=True)
                if miss[:5]:
                    print(f"    e.g. missing: {miss[:5]}", flush=True)
                prompt = build_text_prompt("", rows[0]["texts"][0]) if rows else ""
                if rows:
                    print(f"    prompt[0] first 160 chars: {prompt[:160]!r}", flush=True)
        print(f"DRY RUN complete; {bad} missing video files", flush=True)
        return

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print(f"loading {a.model} ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    model.to(a.device).eval()
    processor = AutoProcessor.from_pretrained(a.model, max_pixels=a.max_pixels)

    for lang in langs:
        ds = LANG2DS[lang]
        out_dir = os.path.join(a.emb_dir, ds)
        os.makedirs(out_dir, exist_ok=True)
        for split in splits:
            rows = [normalise_row(r, split, lang)
                    for r in load_lattices(split, lang, root=a.lattice_root)]
            if a.limit:
                rows = rows[:a.limit]
            shard_dir = os.path.join(out_dir, "_bsrtdcells_shards", split)
            os.makedirs(shard_dir, exist_ok=True)
            video_root = os.path.join(a.video_dir, ds, "All")
            print(f"[{ds}/{split}] {len(rows)} lattices -> {shard_dir}", flush=True)

            t0, n_new, n_zero = time.time(), 0, 0
            for i, r in enumerate(rows):
                shas = [text_sha(t) for t in r["texts"]]
                sp = os.path.join(shard_dir, f"{r['seed_id']}.pt")
                if os.path.exists(sp):
                    try:
                        old = torch.load(sp, map_location="cpu", weights_only=False)
                        if old.get("text_sha256") == shas:
                            continue
                    except Exception:  # noqa: BLE001 - torn shard, re-extract
                        pass
                frames, ok = EX.load_video_frames(
                    os.path.join(video_root, f"{r['seed_id']}.mp4"), a.num_frames)
                if not ok:
                    n_zero += 1
                    feats = torch.zeros(4, model.config.hidden_size, dtype=torch.float32)
                else:
                    feats = torch.stack([
                        EX._encode(frames, build_text_prompt("", t), processor, model,
                                   a.device, a.max_pixels, span="response")
                        for t in r["texts"]], dim=0).float()
                torch.save({"seed_id": r["seed_id"], "text_sha256": shas,
                            "text_feats": feats, "seed_label": r["seed_label"],
                            "expected": r["expected"], "decoded": bool(ok)}, sp)
                n_new += 1
                if n_new % 10 == 0:
                    print(f"  PROGRESS {i+1}/{len(rows)} new={n_new} zero_guard={n_zero} "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)

            feats, labels, exp, ids, shas_all, dec = [], [], [], [], [], 0
            for r in rows:
                sd = torch.load(os.path.join(shard_dir, f"{r['seed_id']}.pt"),
                                map_location="cpu", weights_only=False)
                ids.append(sd["seed_id"]); feats.append(sd["text_feats"])
                labels.append(sd["seed_label"]); exp.append(sd["expected"])
                shas_all.append(sd["text_sha256"]); dec += int(not sd["decoded"])
            obj = {"ids": [ids], "cells": list(CELLS),
                   "text_feats": torch.stack(feats, 0).float(),
                   "labels": torch.tensor(labels, dtype=torch.long),
                   "cell_expected_labels": torch.tensor(exp, dtype=torch.long),
                   "text_sha256": shas_all, "student_tag": STUDENT_TAG,
                   "engine": "qwen2.5-vl-7b", "extractor_sha256": sha,
                   "zero_guard": dec}
            out_path = os.path.join(out_dir, f"{SPLIT2CACHE[split]}_{a.out_tag}.pt")
            assert "test" not in os.path.basename(out_path), "refusing a test-like output path"
            torch.save(obj, out_path)
            print(f"[{ds}/{split}] saved M={len(ids)} D={obj['text_feats'].shape[-1]} "
                  f"zero_guard={dec} -> {out_path}", flush=True)
            with open(out_path + ".meta.json", "w") as f:
                json.dump({"n": len(ids), "zero_guard": dec, "extractor_sha256": sha,
                           "model": a.model, "cells": list(CELLS)}, f, indent=2)


if __name__ == "__main__":
    main()
