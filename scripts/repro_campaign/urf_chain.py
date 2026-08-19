#!/usr/bin/env python
"""REPRO campaign Wave 1 — URF-HVAA (NeurIPS 2025) ported to the four hate corpora.

Chain, stage for stage as `third_party/URF-HVAA/README.md` and its scripts define
it for the temporal-VAD task:

  1 caption   `src/video_pre_caption.py` — VideoLLaMA3-7B captions a 10 s window
              centred on every 16th native frame (fps=2, max 10 frames)
  2 score     `scripts/query_llm_vad.sh` -> `src/llm_anomaly_scorer.py` —
              Llama-3.1-8B-Instruct rates each caption 0..1
  3 filter    `src/score_filter.py` — sliding window over the round-1 scores gives
              each video its highest/lowest interval and score statistics
  4 tags      `src/summarize_window.py` — VideoLLaMA3-7B looks at the highest
              interval and returns a Python-style list of suspicious phrases
  5 refine    `scripts/refine_score.sh` -> `src/refine_with_tag.py` — Llama-3.1
              re-scores, with the tag list in the system prompt, but **only** the
              videos whose highest-window average lies inside 0.5 ± threshold;
              every other video keeps its round-1 score (the `cp -n` at the end
              of `refine_score.sh`)

Declared adaptations, all recorded in REPRO_CAMPAIGN_RESULTS §L:

  * **Coarser center grid.** Their center step is 16 native frames (0.533 s at
    30 fps) with a 10 s window, i.e. a 19x overlap; one VideoLLaMA3 generation
    per center over our 84 h of video is far outside a single-GPU budget (their
    own README quotes ~20 h on a 3090 for one much smaller corpus).  We step the
    center by `--step` seconds so the 10 s windows tile or half-overlap, and
    report `native_rate = 1/step`.
  * **Greedy decoding** for both models, as for LAVAD, so the run is
    deterministic and a single run under freeze §6.
  * **Refusals are masked, not interpolated** (`MODEL_ASSETS_STATUS §3.11a`).
  * **Llama-3.1-8B-Instruct** comes from the ungated `NousResearch` mirror
    through the `llama_hf` shim (§3.2, §3.8).
  * `score_filter`'s window is `max(max_frame // 10, 300)` *native frames*, i.e.
    max(10 % of the video, 10 s at 30 fps).  On our grid the same rule is applied
    in seconds — `max(D / 10, 10)` — and converted to index units.

Stages are separate subcommands so the GPU queue can serialise them; each is
idempotent (`.tmp` + `os.replace`, existing output skipped).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "scripts/repro_campaign/shim"))
sys.path.insert(0, str(ROOT / "scripts/repro_campaign"))

WORK = ROOT / "data/urf"
RUN_DIR = ROOT / "idea-stage/repro_urf"
DATASETS = ["HateMM", "MHC", "MHC_zh", "HateClipSeg"]
VLM = "DAMO-NLP-SG/VideoLLaMA3-7B"

# --- prompts, verbatim from the repo ----------------------------------------
CAPTION_SYSTEM = ("You are an AI assistant analyzing this video segment. "
                  "Summarize the main events or actions in a concise way.")
CONTEXT_PROMPT = (
    "How would you rate the scene described on a scale from 0 to 1, with 0 "
    "representing a standard scene and 1 denoting a scene with suspicious "
    "activities or potentially criminal activities?"
)
FORMAT_PROMPT = (
    "Please provide the response in the form of a Python list and respond with only "
    "one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, "
    "0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and "
    "end with  ']'."
)
TAG_SYSTEM = "You are an AI assistant analyzing a suspicious segment of a video."
TAG_USER = ("Analyze the video interval to identify any possible suspicious behaviors. "
            "Return your answer strictly as a Python-style list of phrases that could "
            "briefly describe the suspicious scene splited by commas."
            "No additional commentary or text, return only the list.")
# `refine_score.sh` builds this from `dataset_prior="or criminal"`.
REFINE_CONTEXT = (
    "How would you rate the scene on a scale from 0 to 1, where 0 is ordinary and 1 "
    "indicates suspicious or criminal activity? In addition, we have identified "
    "certain suspicious or criminal behaviors that may appear in the video. Please "
    "consider these carefully when deciding on the final anomaly rating."
)
REFINE_FORMAT = (
    "Please provide the response as a Python list with exactly one number in "
    "[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]. Include no extra text. "
    "The list should begin with '[' and end with ']'."
)
WINDOW = 10.0        # their `--interval 10`
THRESHOLD = 0.05     # `refine_score.sh --threshold 0.05`
MAX_SEQ_LEN = 512
SCORE_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]")


def videos(ds: str, split: str):
    from blip2_caption import NO_VIDEO_STREAM, find_video

    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    out = []
    for i, v in enumerate(z["video_ids"]):
        if split != "all" and str(z["split"][i]) != split:
            continue
        vid = str(v)
        if vid in NO_VIDEO_STREAM:
            continue
        p = find_video(ds, vid)
        if p is not None:
            out.append((vid, p, float(z["duration"][i])))
    return sorted(out)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def centers_of(dur: float, step: float) -> list[int]:
    """Center k is at t = k * step seconds; the first center sits half a window in
    so the clamped window is a full one, exactly as their clamping produces."""
    n = max(int(dur // step), 1)
    return list(range(n))


# ----------------------------------------------------------- stage 1: VLM ---
def load_vlm():
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        VLM, trust_remote_code=True, device_map="cuda:0",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    proc = AutoProcessor.from_pretrained(VLM, trust_remote_code=True)
    return model.eval(), proc


@torch.inference_mode()
def vlm_infer(model, proc, conversation, max_new_tokens=256):
    inputs = proc(conversation=conversation, add_system_prompt=True,
                  add_generation_prompt=True, return_tensors="pt")
    dev, dt = next(model.parameters()).device, next(model.parameters()).dtype
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(dev, dtype=dt) if k == "pixel_values" else v.to(dev)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(out, skip_special_tokens=True)[0]


def stage_caption(args) -> None:
    model, proc = load_vlm()
    jobs = [(ds, v, p, d) for ds in args.datasets.split(",")
            for v, p, d in videos(ds, args.split)]
    todo = [j for j in jobs if not (WORK / "captions" / j[0] / f"{j[1]}.json").exists()]
    print(f"PROGRESS caption plan={len(todo)}/{len(jobs)} step={args.step}s", flush=True)
    t0, n, ncall = time.time(), 0, 0
    for ds, vid, path, dur in todo:
        res = {}
        try:
            for c in centers_of(dur, args.step):
                ct = c * args.step + args.step / 2.0
                st, en = max(ct - WINDOW / 2, 0.0), min(ct + WINDOW / 2, dur)
                if en - st <= 0:
                    continue
                conv = [
                    {"role": "system", "content": CAPTION_SYSTEM},
                    {"role": "user", "content": [{
                        "type": "video",
                        "video": {"video_path": str(path), "fps": 2,
                                  "start_time": st, "end_time": en, "max_frames": 10},
                    }]},
                ]
                r = vlm_infer(model, proc, conv).strip()
                res[str(c)] = r or "No detected activity in this segment."
                ncall += 1
        except Exception as e:
            print(f"[fail] {ds}/{vid} {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        if not res:
            continue
        write_json(WORK / "captions" / ds / f"{vid}.json", res)
        n += 1
        if n % 5 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"PROGRESS caption {n}/{len(todo)} calls={ncall} "
                  f"{ncall/max(el,1e-9):.2f} call/s elapsed={el/60:.1f}min "
                  f"eta={(len(todo)-n)*el/n/60:.1f}min", flush=True)
    print(f"[done] caption videos={n} calls={ncall} "
          f"wall={(time.time()-t0)/60:.1f}min", flush=True)


# --------------------------------------------------------- stage 2/5: LLM ---
class Llama31:
    def __init__(self, batch_size: int):
        from llama_hf import Llama

        os.environ.pop("LLAMA_HF_4BIT", None)   # 8B bf16 = 16 GiB, fits the card
        g = Llama.build(ckpt_dir="libs/llama/llama3.1-8b/",
                        max_seq_len=MAX_SEQ_LEN, max_batch_size=batch_size)
        self.model, self.tok = g.model, g.tokenizer
        self.bs = batch_size
        self.cache: dict[tuple[str, str], str] = {}
        self.n_gen = self.n_hit = 0

    @torch.inference_mode()
    def _run(self, pairs):
        prompts = [self.tok.apply_chat_template(
            [{"role": "system", "content": s}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True) for s, u in pairs]
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=MAX_SEQ_LEN - 1, add_special_tokens=False
                       ).to(self.model.device)
        plen = enc["input_ids"].shape[1]
        out = self.model.generate(**enc, max_new_tokens=max(MAX_SEQ_LEN - plen, 1),
                                  do_sample=False, pad_token_id=self.tok.pad_token_id)
        return [t.strip() for t in
                self.tok.batch_decode(out[:, plen:], skip_special_tokens=True)]

    def __call__(self, pairs):
        uniq = list(dict.fromkeys(p for p in pairs if p not in self.cache))
        self.n_hit += len(pairs) - len(uniq)
        uniq.sort(key=lambda p: len(p[1]))
        for i in range(0, len(uniq), self.bs):
            chunk = uniq[i: i + self.bs]
            for k, v in zip(chunk, self._run(chunk)):
                self.cache[k] = v
            self.n_gen += len(chunk)
        return [self.cache[p] for p in pairs]


def _score_pass(args, sub_in: str, sub_out: str, system_of, keep=None) -> None:
    llm = Llama31(args.batch_size)
    jobs = [(ds, v) for ds in args.datasets.split(",")
            for v, _, _ in videos(ds, args.split)]
    todo = [(ds, v) for ds, v in jobs
            if (WORK / sub_in / ds / f"{v}.json").exists()
            and not (WORK / sub_out / ds / f"{v}.json").exists()
            and (keep is None or keep(ds, v))]
    print(f"PROGRESS {sub_out} plan={len(todo)}", flush=True)
    t0, n, ncall = time.time(), 0, 0
    for ds, vid in todo:
        caps = json.loads((WORK / sub_in / ds / f"{vid}.json").read_text())
        centers = sorted(caps, key=int)
        system = system_of(ds, vid)
        if system is None:
            continue
        # `_prepare_dialogs`: system = context + " " + format, user = the caption
        outs = llm([(system, caps[c]) for c in centers])
        scores, refusals = {}, {}
        for c, o in zip(centers, outs):
            m = SCORE_RE.search(o)
            if m:
                scores[c] = float(m.group(1))
            else:
                scores[c] = -1.0
                refusals[c] = o[:400]
        write_json(WORK / sub_out / ds / f"{vid}.json", scores)
        write_json(WORK / f"{sub_out}_refusals" / ds / f"{vid}.json", refusals)
        n += 1
        ncall += len(centers)
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"PROGRESS {sub_out} {n}/{len(todo)} calls={ncall} gen={llm.n_gen} "
                  f"cachehit={llm.n_hit} {llm.n_gen/max(el,1e-9):.2f} gen/s "
                  f"elapsed={el/60:.1f}min eta={(len(todo)-n)*el/n/60:.1f}min",
                  flush=True)
    print(f"[done] {sub_out} videos={n} calls={ncall} gen={llm.n_gen} "
          f"cachehit={llm.n_hit} wall={(time.time()-t0)/60:.1f}min", flush=True)


def stage_score(args) -> None:
    system = CONTEXT_PROMPT + " " + FORMAT_PROMPT
    _score_pass(args, "captions", "scores", lambda ds, v: system)


# ----------------------------------------------------------- stage 3 -------
def find_extreme_intervals(scores: dict[int, float], win: int):
    """`score_filter.find_extreme_intervals`, with the window passed in index
    units (their `max(max_frame//10, 300)` native frames = max(D/10, 10) s)."""
    ks = sorted(scores)
    if not ks:
        return 0, 0, 0.0, 0, 0, 0.0
    best, worst = float("-inf"), float("inf")
    bs = be = ws = we = 0
    for i, s in enumerate(ks):
        e = s + win
        w = [scores[k] for k in ks[i:] if k < e]
        if not w:
            continue
        a = sum(w) / len(w)
        if a > best:
            best, bs, be = a, s, e
        if a < worst:
            worst, ws, we = a, s, e
    return bs, be, best, ws, we, worst


def stage_filter(args) -> None:
    out = {}
    for ds in args.datasets.split(","):
        out[ds] = {}
        for vid, _, dur in videos(ds, args.split):
            p = WORK / "scores" / ds / f"{vid}.json"
            if not p.exists():
                continue
            raw = {int(k): v for k, v in json.loads(p.read_text()).items()}
            # a refusal is not a score: it is excluded, not treated as -1
            raw = {k: v for k, v in raw.items() if v >= 0}
            if not raw:
                continue
            win = max(int(round(max(dur / 10.0, 10.0) / args.step)), 1)
            bs, be, best, ws, we, worst = find_extreme_intervals(raw, win)
            vals = np.array(list(raw.values()), dtype=np.float32)
            hi = vals[np.abs(vals - best) <= np.abs(vals - worst)]
            lo = vals[np.abs(vals - best) > np.abs(vals - worst)]
            hi = hi if hi.size else vals
            lo = lo if lo.size else vals
            out[ds][vid] = {
                "highest_interval": [bs, be], "highest_avg_score": round(best, 3),
                "lowest_interval": [ws, we], "lowest_avg_score": round(worst, 3),
                "std": round(float(vals.std()), 5),
                "avg_high_group": round(float(hi.mean()), 3),
                "avg_low_group": round(float(lo.mean()), 3),
                "gap_high_low": round(float(hi.mean() - lo.mean()), 3),
            }
        print(f"[filter] {ds} videos={len(out[ds])}", flush=True)
    write_json(WORK / "highest_lowest_intervals.json", out)


# ----------------------------------------------------------- stage 4 -------
def stage_tags(args) -> None:
    iv = json.loads((WORK / "highest_lowest_intervals.json").read_text())
    model, proc = load_vlm()
    outp = WORK / "suspicious_part_phrases.json"
    res = json.loads(outp.read_text()) if outp.exists() else {}
    t0, n = time.time(), 0
    jobs = [(ds, v, p, d) for ds in args.datasets.split(",")
            for v, p, d in videos(ds, args.split)
            if v in iv.get(ds, {}) and v not in res.get(ds, {})]
    print(f"PROGRESS tags plan={len(jobs)}", flush=True)
    for ds, vid, path, dur in jobs:
        s, e = iv[ds][vid]["highest_interval"]
        st = min(s * args.step, dur)
        en = min(e * args.step, dur)
        if en - st <= 0:
            continue
        conv = [
            {"role": "system", "content": TAG_SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": {"video_path": str(path), "fps": 18,
                                            "start_time": st, "end_time": en,
                                            "max_frames": args.tag_max_frames}},
                {"type": "text", "text": TAG_USER}]},
        ]
        try:
            r = vlm_infer(model, proc, conv, max_new_tokens=1024).strip()
        except Exception as ex:
            print(f"[fail] tags {ds}/{vid} {type(ex).__name__}: {ex}", flush=True)
            torch.cuda.empty_cache()
            continue
        res.setdefault(ds, {})[vid] = r
        n += 1
        if n % 10 == 0 or n == len(jobs):
            write_json(outp, res)
            el = time.time() - t0
            print(f"PROGRESS tags {n}/{len(jobs)} {n/max(el,1e-9):.2f} call/s "
                  f"elapsed={el/60:.1f}min eta={(len(jobs)-n)*el/n/60:.1f}min",
                  flush=True)
    write_json(outp, res)
    print(f"[done] tags videos={n} wall={(time.time()-t0)/60:.1f}min", flush=True)


# ----------------------------------------------------------- stage 5 -------
def clean_phrase(p: str) -> str:
    p = p.strip()
    if p.startswith("["):
        p = p[1:]
    if p.endswith("]"):
        p = p[:-1]
    return p.strip()


def stage_refine(args) -> None:
    iv = json.loads((WORK / "highest_lowest_intervals.json").read_text())
    tags = json.loads((WORK / "suspicious_part_phrases.json").read_text())

    def gate(ds, vid):
        st = iv.get(ds, {}).get(vid)
        if st is None:
            return False
        h = float(st["highest_avg_score"])
        lo, hi = max(0.0, 0.5 - THRESHOLD), min(1.0, 0.5 + THRESHOLD)
        return lo <= h <= hi and bool(tags.get(ds, {}).get(vid, "").strip())

    def system_of(ds, vid):
        t = clean_phrase(tags.get(ds, {}).get(vid, ""))
        low = t.lower()
        if not t or low == "none" or "no " in low or "no_" in low:
            return None
        return (f"{REFINE_CONTEXT}\n[Potentially reported suspicious activities: "
                f"{t}]\n\n{REFINE_FORMAT}")

    _score_pass(args, "captions", "refined", system_of, keep=gate)
    n = sum(1 for ds in args.datasets.split(",")
            for v, _, _ in videos(ds, args.split) if gate(ds, v))
    print(f"[gate] videos inside 0.5+-{THRESHOLD}: {n}", flush=True)


# ----------------------------------------------------------- curves --------
def stage_curves(args) -> None:
    from lavad_chain import _gt, _y_at

    stats = {}
    for ds in args.datasets.split(","):
        gt = _gt(ds)
        out_dir = RUN_DIR / "curves" / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        st = defaultdict(float)
        for vid, _, dur in videos(ds, args.split):
            sp = WORK / "scores" / ds / f"{vid}.json"
            if not sp.exists():
                continue
            raw = {int(k): v for k, v in json.loads(sp.read_text()).items()}
            if not raw:
                continue
            T = max(raw) + 1
            r1 = np.full(T, np.nan)
            for k, v in raw.items():
                if v >= 0:
                    r1[k] = v
            # `refine_score.sh` ends with `cp -n scores/*.json refined_scores/`:
            # a video the gate skipped keeps its round-1 curve.
            base = r1.copy()
            rp = WORK / "refined" / ds / f"{vid}.json"
            if rp.exists():
                for k, v in json.loads(rp.read_text()).items():
                    k = int(k)
                    if k < T:
                        base[k] = v if v >= 0 else np.nan
            np.savez(out_dir / f"{vid}.npz", rate=np.float64(1.0 / args.step),
                     base=base, round1=r1)
            g = gt.get(vid)
            st["n_frames"] += T
            st["n_refused"] += int(np.isnan(r1).sum())
            st["n_videos"] += 1
            if g is not None:
                y = _y_at(g, T, step=args.step, offset=args.step / 2.0)
                st["n_pos"] += int(y.sum())
                st["n_neg"] += int((1 - y).sum())
                st["n_refused_pos"] += int((np.isnan(r1) & (y == 1)).sum())
                st["n_refused_neg"] += int((np.isnan(r1) & (y == 0)).sum())
        stats[ds] = dict(st)
        print(f"[curves] {ds} {dict(st)}", flush=True)
    write_json(RUN_DIR / "refusal_stats.json", stats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["caption", "score", "filter", "tags",
                                      "refine", "curves"])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tag-max-frames", type=int, default=180)
    args = ap.parse_args()
    {"caption": stage_caption, "score": stage_score, "filter": stage_filter,
     "tags": stage_tags, "refine": stage_refine, "curves": stage_curves}[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
