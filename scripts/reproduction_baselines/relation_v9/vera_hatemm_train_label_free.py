#!/usr/bin/env python3
"""VERA HateMM train inference/postprocess with label-free 1 fps support."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

import vera_adapter as vera
from vera_fast_infer import ReusableVideoReader, predict_batch
from relation_v9.train_timeline import hatemm_train_timeline


def softmax(value):
    value = np.exp(value - np.max(value)); return value / value.sum()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--raw-dir", required=True)
    p.add_argument("--out", required=True); p.add_argument("--prompt-json", required=True)
    p.add_argument("--batch-size", type=int, default=1, choices=(1, 2)); a = p.parse_args()
    ids, lengths, _ = hatemm_train_timeline(); raw_root = Path(a.raw_dir); raw_root.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(a.prompt_json).read_text())
    model, tokenizer, _ = vera.load_model(selection["attention_backend"])
    for number, vid in enumerate(ids, 1):
        target = raw_root / f"{vid}.json"; starts = np.arange(lengths[vid], dtype=float)
        if vera.valid_raw_result(target, vid, len(starts)):
            print(f"[{number}/{len(ids)}] {vid}: already complete", flush=True); continue
        reader = ReusableVideoReader(vera.video_path("hatemm", vid)); segments = []
        for offset in range(0, len(starts), a.batch_size):
            batch = starts[offset:offset+a.batch_size]
            images = [reader.frames(float(start), 10., 8) for start in batch]
            for start, (score, response) in zip(batch, predict_batch(model, tokenizer, images, selection["prompts"], a.batch_size)):
                segments.append({"start": float(start), "end": min(reader.duration, start + 10.),
                                 "score": score, "response": response})
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps({"video_id": vid, "duration": reader.duration,
                                         "segments": segments}, ensure_ascii=False))
        os.replace(temporary, target)
    clip_root = Path("results/reproduction/features/clip_b16_1fps/hatemm")
    output = Path(a.out); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for vid in ids:
            record = json.loads((raw_root / f"{vid}.json").read_text())
            raw = np.asarray([x["score"] for x in record["segments"]], float)
            visual = np.load(clip_root / f"{vid}.npy")
            if len(raw) != lengths[vid] or len(visual) != lengths[vid]:
                raise RuntimeError(f"VERA label-free timeline mismatch: {vid}")
            visual = visual / np.maximum(np.linalg.norm(visual, axis=1, keepdims=True), 1e-12)
            similarity = visual @ visual.T; top_n = max(1, int(.15 * len(raw))); propagated = np.empty(len(raw))
            for i in range(len(raw)):
                ix = np.argsort(similarity[i])[-top_n:]
                propagated[i] = softmax(similarity[i, ix] * 10) @ raw[ix]
            neighbor = gaussian_filter1d(propagated, sigma=10, radius=7, mode="nearest")
            x = np.arange(len(raw)); center = np.exp(-.5 * ((x - len(raw)//2) / max(len(raw)/2, 1))**2)
            handle.write(json.dumps({"video_id": vid, "score_raw": raw.tolist(),
                                     "score_neighbor": neighbor.tolist(),
                                     "score_official_postprocessed": (neighbor*center).tolist()}) + "\n")
    os.replace(temporary, output)


if __name__ == "__main__": main()
