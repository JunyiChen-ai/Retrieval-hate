#!/usr/bin/env python3
"""C3 NON-TARGET content pilot — generate dense world-knowledge/implicit-reasoning
text per video AND embed it via the pipeline's frozen text pathway. One job does
generate -> embed (the embedding needs the GPU, so it is folded in here).

Pre-registration: refine-logs/C3_NONTARGET_PILOT_DESIGN.md (frozen BEFORE any probe number).

Design (all fixed by the design doc):
  * Sample = 300 HateMM-train + 300 MHC-EN-train videos, stratified by gold label
    (labels used for SAMPLING ONLY; never enter the generation prompt).
  * Evidence pack = 8 frames + title + ASR, formatted exactly as the banked extractor.
  * ONE generation prompt (GEN_PROMPT below), greedy, max_new_tokens=256, Qwen2.5-VL-7B local.
  * A_text = _encode(frames, TEXT_INSTRUCTION + "\nAnalysis: " + generated_text, span="response")
    reused VERBATIM from src/utils/generate_VideoMLLM_embedding_HF.py -> 3584-d L2-normed,
    identical space/pooling as the banked text_feats.

Resumable per-video (skip if text json + emb npy both exist); atomic writes (os.replace).
Symlink-tolerant video loading (the extractor's loader follows symlinks via os.path.exists).
conda HateVideo; single GPU; HF_HUB_OFFLINE. No gold label is ever written into a generation
artifact. Not committed.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import torch

REPO = '/data/jehc223/RGCL'
EXT_PATH = os.path.join(REPO, 'src/utils/generate_VideoMLLM_embedding_HF.py')
ART_ROOT = os.path.join(REPO, 'artifacts/c3_nontarget')

# ---- import the banked extractor verbatim (loader + _encode + instructions) ----
_spec = importlib.util.spec_from_file_location('vmllm_ext', EXT_PATH)
ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ext)

# ---- THE single frozen generation prompt (design doc §3) ----
GEN_PROMPT = (
    "You are an expert analyst of implicit and coded hateful content in short videos. "
    "You are shown several frames sampled from a video, its title, and an automatic speech "
    "transcript. Write ONE dense analytical paragraph (about 150-220 words) that decodes the "
    "IMPLICIT and CODED signals a casual viewer would miss. Reason about: coded language, "
    "dog-whistles, slurs-by-allusion, euphemisms and in-group jargon; the real-world meaning of "
    "any symbols, flags, insignia, hand gestures, memes, or numeric codes that appear; and how the "
    "scene, setting, or behaviour reframes otherwise-neutral words. Use world knowledge to explain "
    "what these things REFERENCE and why they could signal hostility. Constraints: do NOT make "
    "\"naming the targeted group/category\" your main content - decode mechanisms, not labels; do "
    "NOT restate, quote, or paraphrase the transcript; do NOT transcribe or read out any on-screen "
    "text. If nothing implicit or coded is present, say so and briefly explain why the content reads "
    "as benign. Output only the paragraph."
)
GEN_PROMPT_SHA = hashlib.sha1(GEN_PROMPT.encode()).hexdigest()[:12]

MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'
MAX_PIXELS = 360 * 420
NUM_FRAMES = 8
MAX_NEW_TOKENS = 256
SAMPLE_SEED = 20260714
HATE_RATE = {'HateMM': 0.4005, 'MHC': 0.3060}
SAMPLE_N = 300


def build_sample_manifest(ds):
    """Deterministic stratified-by-label sample of SAMPLE_N ids (labels used for SAMPLING ONLY).
    Written once to artifacts/c3_nontarget/<ds>_sample300.json; reused if present."""
    mpath = os.path.join(ART_ROOT, f'{ds}_sample300.json')
    if os.path.exists(mpath):
        return json.load(open(mpath))
    items = ext.read_gt(os.path.join(REPO, f'data/gt/{ds}/train.jsonl'))
    pos = sorted([it['id'] for it in items if int(it['label']) == 1])
    neg = sorted([it['id'] for it in items if int(it['label']) == 0])
    n_pos = int(round(SAMPLE_N * HATE_RATE[ds]))
    n_neg = SAMPLE_N - n_pos
    n_pos = min(n_pos, len(pos)); n_neg = min(n_neg, len(neg))
    rng = np.random.default_rng(SAMPLE_SEED)
    sel_pos = sorted(rng.choice(pos, size=n_pos, replace=False).tolist())
    sel_neg = sorted(rng.choice(neg, size=n_neg, replace=False).tolist())
    by_id = {it['id']: it for it in items}
    chosen = sorted(sel_pos + sel_neg)
    manifest = {
        'dataset': ds, 'seed': SAMPLE_SEED, 'n_total': len(chosen),
        'n_pos': n_pos, 'n_neg': n_neg, 'hate_rate_target': HATE_RATE[ds],
        'note': 'labels used for SAMPLING ONLY (probe-power balancing); never in the generation prompt',
        'ids': chosen,
        'labels': {i: int(by_id[i]['label']) for i in chosen},
        'title': {i: by_id[i].get('title', '') for i in chosen},
        'text': {i: by_id[i].get('text', '') for i in chosen},
    }
    os.makedirs(ART_ROOT, exist_ok=True)
    tmp = mpath + '.tmp'
    json.dump(manifest, open(tmp, 'w'))
    os.replace(tmp, mpath)
    print(f'[manifest] wrote {mpath}: N={len(chosen)} (pos={n_pos} neg={n_neg})', flush=True)
    return manifest


@torch.no_grad()
def generate_text(frames, title, transcript, processor, model, device):
    """Greedy dense analysis paragraph from 8 frames + title + ASR. Returns the decoded string."""
    context = (
        GEN_PROMPT
        + "\nTitle: " + (title if title else "(none)")
        + "\nTranscript: " + (transcript if transcript else "(none)")
    )
    messages = ext._build_messages(frames, context)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=[frames], return_tensors='pt').to(device)
    gen_ids = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
        use_cache=True,
    )
    trimmed = gen_ids[:, inputs['input_ids'].shape[1]:]
    out = processor.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    return out.strip()


def process_dataset(ds, processor, model, device):
    manifest = build_sample_manifest(ds)
    ids = manifest['ids']
    titles = manifest['title']; texts = manifest['text']
    txt_dir = os.path.join(ART_ROOT, ds, 'text')
    emb_dir = os.path.join(ART_ROOT, ds, 'emb')
    os.makedirs(txt_dir, exist_ok=True); os.makedirs(emb_dir, exist_ok=True)
    video_root = os.path.join(REPO, f'data/video/{ds}/All')
    d = model.config.hidden_size

    done = 0; zero_guard = 0; t0 = time.time()
    for n, vid in enumerate(ids):
        tpath = os.path.join(txt_dir, f'{vid}.json')
        epath = os.path.join(emb_dir, f'{vid}.npy')
        if os.path.exists(tpath) and os.path.exists(epath):
            done += 1
            continue
        vpath = os.path.join(video_root, f'{vid}.mp4')
        frames, ok = ext.load_video_frames(vpath, NUM_FRAMES)
        if ok:
            try:
                gen = generate_text(frames, titles.get(vid, ''), texts.get(vid, ''),
                                    processor, model, device)
                embed_prompt = ext.TEXT_INSTRUCTION + "\nAnalysis: " + (gen if gen else "(none)")
                vec = ext._encode(frames, embed_prompt, processor, model, device,
                                  MAX_PIXELS, span='response').numpy().astype(np.float32)
            except Exception as e:  # noqa: BLE001
                print(f'[ERR] {ds}/{vid} generation/embed failed: {repr(e)}', flush=True)
                ok = False; gen = ''; vec = np.zeros(d, dtype=np.float32)
        else:
            gen = ''; vec = np.zeros(d, dtype=np.float32)
        if not ok:
            zero_guard += 1
        # atomic writes (NO gold label in the text artifact)
        rec = {'id': vid, 'dataset': ds, 'ok': bool(ok), 'gen_text': gen,
               'gen_prompt_sha': GEN_PROMPT_SHA, 'model': MODEL,
               'n_frames': NUM_FRAMES, 'max_new_tokens': MAX_NEW_TOKENS,
               'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}
        ttmp = tpath + '.tmp'; json.dump(rec, open(ttmp, 'w')); os.replace(ttmp, tpath)
        etmp = epath + '.tmp.npy'; np.save(etmp, vec); os.replace(etmp, epath)
        done += 1
        if done % 10 == 0:
            el = time.time() - t0
            print(f'  [{ds}] {done}/{len(ids)} done (zero-guard {zero_guard}) '
                  f'{el:.0f}s elapsed, {el/max(done,1):.1f}s/vid', flush=True)
    print(f'[{ds}] COMPLETE: {done}/{len(ids)} (zero-vector videos={zero_guard})', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', type=str, default='HateMM,MHC')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.manual_seed(SAMPLE_SEED)
    device = torch.device(args.device)
    print(f'[gen] loading {MODEL} on {device} (prompt sha={GEN_PROMPT_SHA})', flush=True)
    model = ext.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map=None)
    model.to(device).eval()
    processor = ext.AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)

    for ds in [d.strip() for d in args.datasets.split(',') if d.strip()]:
        print(f'\n===== dataset {ds} =====', flush=True)
        process_dataset(ds, processor, model, device)
    print('\n[gen] ALL DONE', flush=True)


if __name__ == '__main__':
    sys.exit(main())
