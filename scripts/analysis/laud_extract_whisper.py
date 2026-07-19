#!/usr/bin/env python3
"""LEARNED-AUDIO axis — Stage A: extract Whisper-ENCODER hidden-state embeddings.

Design frozen in refine-logs/AUDIO_AXIS_FORENSIC_RECON.md sections 2-3 (commit 166f9e2b),
followed VERBATIM. The GO verdict there ranks this the #1 in-box gap (cheapest, blessed gain
source): a LEARNED Whisper-encoder audio representation as a NEW input stream, replacing the
classical eGeMAPS-88-d prosody vector that F41/APX killed (HateMM-only, whole-video functionals).

Representation (recon section 3):
  For each video: PyAV-decode 16 kHz mono float32 (reuse decode_audio_pyav from the ASR extractor,
  no ffmpeg binary / no torchaudio) -> split into 30 s (n_samples=480000) chunks -> Whisper
  get_encoder() last_hidden_state [1500, d] per chunk -> mean-pool (+) max-pool over the 1500 time
  steps = 2d per chunk -> mean over chunks -> ONE 2d-dim video vector.
  Primary model = openai/whisper-large-v3 (d_model=1280 -> 2560-d), weights already on disk.
  d_model is read from the loaded encoder config (NOT hardcoded): the pooled dim = 2*d_model.

  ALSO banked (recon section 3, NOT used in the $0 gate): a K=4 segment-level variant — the
  concatenated encoder frame sequence [n_chunks*1500, d] split into K=4 uniform contiguous windows,
  mean (+) max per window = [K, 2d]. Reserved for a future localization tie-in; frozen now so the
  segment representation is fixed before any use.

Scope: train (union) val ONLY, per dataset (HateMM, MHC=EN, MHC_zh=ZH). ZERO test-touch — the test
split is never enumerated. Canonical id order = gt train.jsonl (+) val.jsonl (train then val, file
order). Labels are banked into the cache alongside (as eGeMAPS did) but are consumed only PROBE-ONLY
by the downstream gate; this extractor uses labels for nothing.

Per-video CHECKPOINTING: each id's vectors are written to
data/audio/<DS>/whisper_<tag>/<id>.npz (vid=[2d], seg=[K,2d]) the moment they are computed; a re-run
(after a SLURM reap/requeue) skips every id already cached. After all ids of a dataset are present
the script aggregates them into one ordered cache + a status manifest, mirroring the eGeMAPS layout
(scripts/analysis/apx_extract_egemaps.py).

Raw videos NEVER leave local (CLAUDE.md hard rule); only derived float .pt caches are produced.
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch

REPO = '/data/jehc223/RGCL'
DATASETS = ['HateMM', 'MHC', 'MHC_zh']
CHUNK_SAMPLES = 480000          # 30 s at 16 kHz (WhisperFeatureExtractor.n_samples)
TARGET_RATE = 16000
MIN_SAMPLES = 160               # >10 ms -> treat as audio (mirrors the ASR extractor audio_ok gate)


# reuse decode_audio_pyav VERBATIM from the deployed ASR extractor (no drift, no copy)
def _load_decode_fn():
    path = f'{REPO}/src/utils/generate_segment_asr_HF.py'
    spec = importlib.util.spec_from_file_location('gen_asr_hf', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decode_audio_pyav


def read_gt_ids(dataset, splits=('train', 'val')):
    """Canonical id order = gt train.jsonl (+) val.jsonl (train then val, file order).
    Returns (ids, labels). Zero test-touch: 'test' is never passed."""
    ids, labels = [], []
    for split in splits:
        gt = f'{REPO}/data/gt/{dataset}/{split}.jsonl'
        with open(gt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                ids.append(str(o['id']))
                labels.append(int(o['label']))
    return ids, np.asarray(labels, dtype=np.int64)


def chunk_audio(audio):
    """Split a 1-D float32 waveform into consecutive 30 s chunks (last chunk short -> FE pads)."""
    n = len(audio)
    return [audio[i:i + CHUNK_SAMPLES] for i in range(0, n, CHUNK_SAMPLES)]


def pool_video(hidden_chunks, K):
    """hidden_chunks: list of numpy [1500, d] (one per 30 s chunk, in time order).
    Returns (vid_vec [2d], seg_vec [K, 2d]).
      vid_vec = mean over chunks of (mean (+) max over the 1500 frames of each chunk).
      seg_vec = mean (+) max over K uniform contiguous windows of the concatenated frame sequence.
    """
    d = hidden_chunks[0].shape[1]
    # ---- video-level: per-chunk mean(+)max, then mean over chunks ----
    per_chunk = np.empty((len(hidden_chunks), 2 * d), dtype=np.float64)
    for c, h in enumerate(hidden_chunks):
        per_chunk[c] = np.concatenate([h.mean(axis=0), h.max(axis=0)])
    vid_vec = per_chunk.mean(axis=0).astype(np.float32)
    # ---- segment-level: K uniform windows over the concatenated [n_chunks*1500, d] sequence ----
    T = len(hidden_chunks) * hidden_chunks[0].shape[0]   # every chunk yields exactly 1500 frames
    seg_sum = np.zeros((K, d), dtype=np.float64)
    seg_cnt = np.zeros(K, dtype=np.int64)
    seg_max = np.full((K, d), -np.inf, dtype=np.float64)
    off = 0
    for h in hidden_chunks:
        t = h.shape[0]
        wid = np.minimum(K - 1, (np.arange(off, off + t) * K) // T)
        for w in np.unique(wid):
            rows = h[wid == w]
            seg_sum[w] += rows.sum(axis=0)
            seg_cnt[w] += rows.shape[0]
            seg_max[w] = np.maximum(seg_max[w], rows.max(axis=0))
        off += t
    seg_mean = seg_sum / np.maximum(seg_cnt, 1)[:, None]
    seg_vec = np.concatenate([seg_mean, seg_max], axis=1).astype(np.float32)   # [K, 2d]
    return vid_vec, seg_vec


def encode_video(audio, feat_ext, encoder, device, dtype, batch_chunks, K, d_model):
    """audio -> (vid_vec [2d], seg_vec [K,2d], n_chunks). audio must be non-empty."""
    chunks = chunk_audio(audio)
    hidden = []
    for i in range(0, len(chunks), batch_chunks):
        batch = chunks[i:i + batch_chunks]
        feats = feat_ext([np.asarray(c, dtype=np.float32) for c in batch],
                         sampling_rate=TARGET_RATE, return_tensors='pt')
        inp = feats.input_features.to(device=device, dtype=dtype)
        with torch.no_grad():
            h = encoder(inp).last_hidden_state          # [b, 1500, d]
        hidden.extend(h.float().cpu().numpy())          # list of [1500, d] float32
    assert hidden and hidden[0].shape[1] == d_model, (len(hidden), hidden[0].shape if hidden else None)
    vid_vec, seg_vec = pool_video(hidden, K)
    return vid_vec, seg_vec, len(chunks)


def main(args):
    from transformers import WhisperFeatureExtractor, WhisperModel

    decode_audio_pyav = _load_decode_fn()
    model_tag = str(args.model).split('/')[-1]
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == 'cuda' else torch.float32

    print(f'[laud-extract] model={args.model} tag={model_tag} device={device} dtype={dtype} '
          f'K={args.num_subclips} batch_chunks={args.batch_chunks}', flush=True)
    feat_ext = WhisperFeatureExtractor.from_pretrained(args.model)
    encoder = WhisperModel.from_pretrained(args.model).get_encoder().to(device=device, dtype=dtype).eval()
    d_model = int(encoder.config.d_model)
    pooled_dim = 2 * d_model
    print(f'[laud-extract] d_model={d_model} -> pooled video dim={pooled_dim} '
          f'(FE nmels={feat_ext.feature_size} n_samples={feat_ext.n_samples} chunk_len={feat_ext.chunk_length})',
          flush=True)

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    limit = int(os.environ.get('LAUD_LIMIT', '0'))   # 0 = all (smoke affordance; mirrors the ASR --limit)
    for ds in datasets:
        ids, labels = read_gt_ids(ds)
        if limit > 0:
            ids, labels = ids[:limit], labels[:limit]
        vid_root = f'{REPO}/data/video/{ds}/All'
        shard_dir = f'{REPO}/data/audio/{ds}/whisper_{model_tag}'
        os.makedirs(shard_dir, exist_ok=True)
        cache = f'{REPO}/data/audio/{ds}/whisper_{model_tag}_trainval.pt'
        manifest = f'{REPO}/data/audio/{ds}/whisper_{model_tag}_manifest.json'

        missing_vid = [i for i in ids if not os.path.exists(f'{vid_root}/{i}.mp4')]
        assert not missing_vid, f'{ds}: missing raw videos {missing_vid[:10]} (+{len(missing_vid)})'
        todo = [i for i in ids if not os.path.exists(f'{shard_dir}/{i}.npz')]
        print(f'[laud-extract][{ds}] N={len(ids)} cached={len(ids)-len(todo)} todo={len(todo)}', flush=True)

        t0 = time.time()
        status = {}
        for n, vid in enumerate(todo, 1):
            try:
                audio, _dur = decode_audio_pyav(f'{vid_root}/{vid}.mp4', target_rate=TARGET_RATE)
                if audio is None or len(audio) < MIN_SAMPLES:
                    vv = np.zeros(pooled_dim, dtype=np.float32)
                    sv = np.zeros((args.num_subclips, pooled_dim), dtype=np.float32)
                    st = 'no_audio'
                else:
                    vv, sv, nck = encode_video(audio, feat_ext, encoder, device, dtype,
                                               args.batch_chunks, args.num_subclips, d_model)
                    st = 'ok'
                    if np.isnan(vv).any() or np.isnan(sv).any():
                        vv = np.nan_to_num(vv, nan=0.0); sv = np.nan_to_num(sv, nan=0.0)
                        st = 'ok_nanfix'
            except Exception as e:  # noqa: BLE001
                vv = np.zeros(pooled_dim, dtype=np.float32)
                sv = np.zeros((args.num_subclips, pooled_dim), dtype=np.float32)
                st = f'ERR:{type(e).__name__}:{e}'
                print(f'[laud-extract][{ds}]  !! {vid}: {st}', flush=True)
            np.savez(f'{shard_dir}/{vid}.npz', vid=vv, seg=sv)
            status[vid] = st
            if st in ('no_audio',) or st.startswith('ERR'):
                print(f'[laud-extract][{ds}]  !! {vid}: {st}', flush=True)
            if n % 50 == 0 or n == len(todo):
                print(f'[laud-extract][{ds}]  {n}/{len(todo)} [{time.time()-t0:.0f}s]', flush=True)

        # ---- aggregate (all ids now cached) ----
        emb = np.zeros((len(ids), pooled_dim), dtype=np.float32)
        seg = np.zeros((len(ids), args.num_subclips, pooled_dim), dtype=np.float32)
        for i, vid in enumerate(ids):
            z = np.load(f'{shard_dir}/{vid}.npz')
            emb[i] = z['vid']; seg[i] = z['seg']
        n_zero = int((~emb.any(axis=1)).sum())
        n_nan = int(np.isnan(emb).sum() + np.isnan(seg).sum())
        norms = np.linalg.norm(emb, axis=1)
        torch.save({'ids': [ids], 'emb': torch.tensor(emb),
                    'seg_emb': torch.tensor(seg),
                    'labels': torch.tensor(labels),
                    'model_tag': model_tag, 'd_model': d_model, 'pooled_dim': pooled_dim,
                    'pool': 'mean_cat_max', 'num_subclips': args.num_subclips,
                    'scope': 'train_union_val', 'sample_rate': TARGET_RATE, 'channels': 1},
                   cache)
        man = {'dataset': ds, 'N': len(ids), 'n_pos': int(labels.sum()),
               'pooled_dim': pooled_dim, 'd_model': d_model, 'num_subclips': args.num_subclips,
               'model': args.model, 'model_tag': model_tag, 'pool': 'mean_cat_max',
               'scope': 'train_union_val', 'n_zero_vector_rows': n_zero, 'n_nan': n_nan,
               'status_counts': _count(status if todo else {'(all-cached)': len(ids)}),
               'example_norms': {ids[0]: float(norms[0]),
                                 ids[len(ids)//2]: float(norms[len(ids)//2]),
                                 ids[-1]: float(norms[-1])},
               'elapsed_s': round(time.time() - t0, 1), 'cache': cache}
        json.dump(man, open(manifest, 'w'), indent=1)
        print(f'[laud-extract][{ds}] DONE emb={emb.shape} seg={seg.shape} n_zero={n_zero} n_nan={n_nan} '
              f'status={man["status_counts"]} -> {cache}', flush=True)
        print(f'[laud-extract][{ds}] wrote {manifest}', flush=True)


def _count(status):
    c = {}
    for v in status.values():
        key = v if (v in ('ok', 'ok_nanfix', 'no_audio', '(all-cached)')) else 'ERR'
        c[key] = c.get(key, 0) + 1
    return c


def parse_args():
    ap = argparse.ArgumentParser(description='Whisper-encoder audio embedding extractor (Stage A).')
    ap.add_argument('--datasets', type=str, default=','.join(DATASETS))
    ap.add_argument('--model', type=str, default='openai/whisper-large-v3')
    ap.add_argument('--num_subclips', type=int, default=4)
    ap.add_argument('--batch_chunks', type=int, default=16)
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    return ap.parse_args()


if __name__ == '__main__':
    t = time.time()
    main(parse_args())
    print(f'[laud-extract] elapsed {time.time()-t:.0f}s', flush=True)
