#!/usr/bin/env python3
"""APX audio gate — Stage A: extract openSMILE eGeMAPSv02 functionals (88-d) for HateMM train U val.

Design: refine-logs/WAVE3_CANDIDATES.md CANDIDATE 3 (d) (commit 0ee06df). Pure CPU: ffmpeg decodes the
LOCAL mp4 audio to 16 kHz mono wav, openSMILE computes the 88-d eGeMAPSv02 whole-clip functionals
(pitch, loudness, jitter/shimmer, spectral). NO model download, NO GPU. Raw videos NEVER leave local.

Zero test-touch: only the train + dev_seen HateMM ids (canonical order = frameset train (+) dev_seen,
the same 851 ids as Z_best) are extracted; test_seen is never enumerated.

Per-video CHECKPOINTING: each id's 88-d vector is written to data/audio/HateMM/egemaps_v02/<id>.npy the
moment it is computed; a re-run (after a SLURM reap/requeue) skips every id already cached. After all ids
are present the script aggregates them into a single ordered cache + a status manifest.

Parallel: a multiprocessing pool (APX_NPROC workers, default 8) each with its own Smile instance.
"""
import os, sys, subprocess, tempfile, json, time
import numpy as np
import torch

REPO = '/data/jehc223/RGCL'
VID = f'{REPO}/data/video/HateMM/All'
OUT_DIR = f'{REPO}/data/audio/HateMM/egemaps_v02'
CACHE = f'{REPO}/data/audio/HateMM/egemaps_v02_trainval.pt'
MANIFEST = f'{REPO}/data/audio/HateMM/egemaps_v02_manifest.json'
NPROC = int(os.environ.get('APX_NPROC', '8'))
NFEAT = 88

_smile = None


def _init_worker():
    global _smile
    import opensmile
    _smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                             feature_level=opensmile.FeatureLevel.Functionals)


def extract_one(vid):
    """ffmpeg mp4->wav(16k mono) -> eGeMAPSv02 88-d. Failures -> zero vector (kept, honest 'no prosody').
    Checkpoint: writes <id>.npy immediately; returns fast if already cached."""
    global _smile
    out = f'{OUT_DIR}/{vid}.npy'
    if os.path.exists(out):
        return (vid, 'cached')
    import soundfile as sf
    wav = None
    try:
        fd, wav = tempfile.mkstemp(suffix='.wav'); os.close(fd)
        r = subprocess.run(['ffmpeg', '-y', '-i', f'{VID}/{vid}.mp4', '-vn', '-ac', '1', '-ar', '16000',
                            '-f', 'wav', wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0 or not os.path.exists(wav) or os.path.getsize(wav) < 100:
            v = np.zeros(NFEAT, dtype=np.float64); status = 'ffmpeg_fail'
        else:
            data, sr = sf.read(wav)
            if data.size == 0:
                v = np.zeros(NFEAT, dtype=np.float64); status = 'empty_audio'
            else:
                df = _smile.process_signal(data, sr); v = df.values[0].astype(np.float64)
                status = 'ok'
                if np.isnan(v).any():
                    v = np.nan_to_num(v, nan=0.0); status = 'ok_nanfix'
        np.save(out, v)
        return (vid, status)
    except Exception as e:  # noqa
        return (vid, f'ERR:{type(e).__name__}:{e}')
    finally:
        if wav and os.path.exists(wav):
            os.remove(wav)


def canonical_ids():
    fst = torch.load(f'{REPO}/data/CLIP_Embedding/HateMM/frameset_qwen7b_8f/train_frameset.pt',
                     map_location='cpu', weights_only=False)
    fsv = torch.load(f'{REPO}/data/CLIP_Embedding/HateMM/frameset_qwen7b_8f/dev_seen_frameset.pt',
                     map_location='cpu', weights_only=False)
    ids = list(fst['ids'][0]) + list(fsv['ids'][0])
    y = torch.cat([fst['labels'], fsv['labels']], dim=0).numpy().astype(int)
    return ids, y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ids, y = canonical_ids()
    missing_vid = [i for i in ids if not os.path.exists(f'{VID}/{i}.mp4')]
    assert not missing_vid, f'missing raw videos: {missing_vid[:10]} (+{len(missing_vid)})'
    todo = [i for i in ids if not os.path.exists(f'{OUT_DIR}/{i}.npy')]
    print(f'[apx-extract] N={len(ids)} cached={len(ids)-len(todo)} todo={len(todo)} nproc={NPROC}', flush=True)
    import opensmile
    print(f'[apx-extract] opensmile={opensmile.__version__} feature_set=eGeMAPSv02 level=Functionals '
          f'n_feat={NFEAT}', flush=True)
    t0 = time.time()
    status = {}
    if todo:
        from multiprocessing import Pool
        with Pool(NPROC, initializer=_init_worker) as pool:
            for n, (vid, st) in enumerate(pool.imap_unordered(extract_one, todo, chunksize=1), 1):
                status[vid] = st
                if st.startswith('ERR') or st in ('ffmpeg_fail', 'empty_audio'):
                    print(f'[apx-extract]  !! {vid}: {st}', flush=True)
                if n % 100 == 0 or n == len(todo):
                    print(f'[apx-extract]  {n}/{len(todo)} [{time.time()-t0:.0f}s]', flush=True)
    # aggregate (all ids now cached)
    feats = np.zeros((len(ids), NFEAT), dtype=np.float64)
    for i, vid in enumerate(ids):
        feats[i] = np.load(f'{OUT_DIR}/{vid}.npy')
    n_zero = int((~feats.any(axis=1)).sum())
    torch.save({'ids': [ids], 'egemaps': torch.tensor(feats, dtype=torch.float32),
                'labels': torch.tensor(y, dtype=torch.int64), 'n_feat': NFEAT,
                'feature_set': 'eGeMAPSv02', 'level': 'Functionals',
                'opensmile_version': opensmile.__version__}, CACHE)
    manifest = {'N': len(ids), 'n_feat': NFEAT, 'n_pos': int(y.sum()), 'n_zero_vector_rows': n_zero,
                'opensmile_version': opensmile.__version__, 'feature_set': 'eGeMAPSv02',
                'level': 'Functionals', 'sample_rate': 16000, 'channels': 1,
                'status_counts': _count(status), 'elapsed_s': round(time.time() - t0, 1),
                'cache': CACHE}
    json.dump(manifest, open(MANIFEST, 'w'), indent=1)
    print(f'[apx-extract] DONE feats={feats.shape} n_zero_rows={n_zero} '
          f'status={manifest["status_counts"]} -> {CACHE}', flush=True)
    print(f'[apx-extract] wrote {MANIFEST}', flush=True)


def _count(status):
    c = {}
    for v in status.values():
        c[v] = c.get(v, 0) + 1
    return c


if __name__ == '__main__':
    t = time.time(); main(); print(f'[apx-extract] elapsed {time.time()-t:.0f}s', flush=True)
