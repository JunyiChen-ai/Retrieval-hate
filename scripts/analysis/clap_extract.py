#!/usr/bin/env python3
"""CLAP general-audio axis — Stage A: extract CLAP audio embeddings for HateMM.

Design FROZEN in refine-logs/CLAP_GATE_SPEC_2026-07-27.md (commit 6c8929d) sections 1-2,
followed VERBATIM. Structure mirrors scripts/analysis/laud_extract_whisper.py (the F64 precedent).

The candidate: CLAP (laion/larger_clap_general) general-audio SEMANTIC embeddings as a new input
stream, targeting the ERRPAT FN1 cluster ("speech-poor visual hate", 7/27 stable HateMM errors,
ceiling +0.0326 acc). CLAP is the designated CLOSER for the learned-audio axis that both prior
audio kills named: F41/APX killed classical eGeMAPS prosody, F64/LAUD killed the Whisper speech-ASR
encoder, and the LAUD record scopes its own kill to "the Whisper realization, not the axis".

Representation (spec section 2):
  Per video: PyAV-decode 48 kHz mono float32 (decode_audio_pyav imported VERBATIM from the ASR
  extractor; CLAP's native rate) -> split into consecutive non-overlapping 10.0 s windows
  (480000 samples == ClapFeatureExtractor.nb_max_samples) -> per window one forward ->
  mean-over-windows (+) max-over-windows.

  TWO blocks are banked from the SAME forward (zero marginal cost):
    proj   : ClapModel.get_audio_features()  -> 512-d projected joint audio-text embedding
             -> pooled 1024-d.  ** BINDING PRIMARY ** (this IS the CLAP object: language-aligned
             general-audio semantics, the property that distinguishes it from eGeMAPS/Whisper).
    hidden : audio_model(...).pooler_output  -> 1024-d pre-projection HTSAT pooled hidden
             -> pooled 2048-d.  Pre-declared SECONDARY (spec section 4.4): CANNOT produce a PASS.

  Exact-10 s windowing means no input ever exceeds nb_max_samples, so the ClapFeatureExtractor
  default truncation='rand_trunc' (NONDETERMINISTIC for longer inputs) is never entered. This is
  VERIFIED by the --smoke determinism check (D-1), not assumed.

Splits: train (744) + val (107) -> the trainval cache (the ONLY file the gate opens);
        test (215) -> a SEPARATE cache, written then untouched by this lane.
No test label is read here and no test-set metric is computed anywhere in this lane.

Raw videos/audio NEVER leave local (CLAUDE.md hard rule); only derived float caches are produced.
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch

REPO = '/data/jehc223/RGCL'
DATASET = 'HateMM'
MODEL = 'laion/larger_clap_general'
TARGET_RATE = 48000             # CLAP native (ClapFeatureExtractor.sampling_rate)
WINDOW_SAMPLES = 480000         # 10.0 s at 48 kHz == ClapFeatureExtractor.nb_max_samples
MIN_SAMPLES = 480               # >10 ms -> treat as audio (mirrors the ASR/LAUD audio_ok gate)


# reuse decode_audio_pyav VERBATIM from the deployed ASR extractor (no drift, no copy)
def _load_decode_fn():
    path = f'{REPO}/src/utils/generate_segment_asr_HF.py'
    spec = importlib.util.spec_from_file_location('gen_asr_hf', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decode_audio_pyav


def read_gt_ids(splits):
    """Canonical id order = gt file order over the given splits. Returns (ids, labels)."""
    ids, labels = [], []
    for split in splits:
        with open(f'{REPO}/data/gt/{DATASET}/{split}.jsonl') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                ids.append(str(o['id']))
                labels.append(int(o['label']))
    return ids, np.asarray(labels, dtype=np.int64)


def window_audio(audio):
    """Consecutive non-overlapping 10 s windows; trailing short window kept (CLAP repeatpads it)."""
    return [audio[i:i + WINDOW_SAMPLES] for i in range(0, len(audio), WINDOW_SAMPLES)]


def encode_video(audio, feat_ext, model, device, dtype, batch_windows):
    """audio -> (proj_pooled [2*512], hidden_pooled [2*1024], n_windows).

    Per window: proj = get_audio_features (L2-normalised projected joint-space embedding),
    hidden = pooler_output (pre-projection). Pooling = mean-over-windows (+) max-over-windows,
    per spec section 2.

    Both blocks come from ONE forward: verified this session that
        F.normalize(audio_projection(pooler_output), dim=-1) == get_audio_features(...)
    exactly (max abs diff 0.0), so calling audio_model once and normalising is bit-identical to
    the spec's get_audio_features() at half the compute.
    """
    wins = window_audio(audio)
    projs, hids = [], []
    for i in range(0, len(wins), batch_windows):
        batch = [np.asarray(w, dtype=np.float32) for w in wins[i:i + batch_windows]]
        feats = feat_ext(batch, sampling_rate=TARGET_RATE, return_tensors='pt')
        inp = feats.input_features.to(device=device, dtype=dtype)
        kw = {}
        if 'is_longer' in feats:
            kw['is_longer'] = feats.is_longer.to(device=device)
        with torch.no_grad():
            ao = model.audio_model(input_features=inp, **kw)
            pooled = ao.pooler_output                              # [b, 1024] pre-projection
            proj = torch.nn.functional.normalize(
                model.audio_projection(pooled), dim=-1)            # [b, 512] == get_audio_features
        projs.append(proj.float().cpu().numpy())
        hids.append(pooled.float().cpu().numpy())
    P = np.concatenate(projs, axis=0).astype(np.float64)           # [n_win, 512]
    H = np.concatenate(hids, axis=0).astype(np.float64)            # [n_win, 1024]
    proj_vec = np.concatenate([P.mean(axis=0), P.max(axis=0)]).astype(np.float32)      # [1024]
    hid_vec = np.concatenate([H.mean(axis=0), H.max(axis=0)]).astype(np.float32)       # [2048]
    return proj_vec, hid_vec, len(wins)


def build_model(device):
    from transformers import ClapModel, ClapFeatureExtractor
    dtype = torch.float32                       # CPU path; HTSAT in fp32 (no autocast games)
    feat_ext = ClapFeatureExtractor.from_pretrained(MODEL)
    model = ClapModel.from_pretrained(MODEL).to(device=device, dtype=dtype).eval()
    # audio tower only; the RoBERTa text tower is loaded but never forwarded (spec section 1)
    d_proj = int(model.config.projection_dim)
    d_hid = int(model.config.audio_config.hidden_size)
    return feat_ext, model, dtype, d_proj, d_hid


def smoke(args):
    """Pre-declared smoke gate (spec section 2.1). Reads NO labels, produces NO accuracy number."""
    decode_audio_pyav = _load_decode_fn()
    device = torch.device(args.device)
    feat_ext, model, dtype, d_proj, d_hid = build_model(device)
    print(f'[clap-smoke] model={MODEL} d_proj={d_proj} d_hidden={d_hid} device={device}', flush=True)
    print(f'[clap-smoke] FE sr={feat_ext.sampling_rate} nb_max_samples={feat_ext.nb_max_samples} '
          f'mels={feat_ext.feature_size} trunc={feat_ext.truncation} pad={feat_ext.padding}', flush=True)
    assert feat_ext.sampling_rate == TARGET_RATE, feat_ext.sampling_rate
    assert feat_ext.nb_max_samples == WINDOW_SAMPLES, feat_ext.nb_max_samples

    ids, _ = read_gt_ids(('train',))
    ids = ids[:args.n_smoke]
    vid_root = f'{REPO}/data/video/{DATASET}/All'
    recs, first_proj = [], []
    for vid in ids:
        audio, dur = decode_audio_pyav(f'{vid_root}/{vid}.mp4', target_rate=TARGET_RATE)
        if audio is None or len(audio) < MIN_SAMPLES:
            print(f'[clap-smoke]  !! {vid}: no_audio'); continue
        p1, h1, nw = encode_video(audio, feat_ext, model, device, dtype, args.batch_windows)
        p2, h2, _ = encode_video(audio, feat_ext, model, device, dtype, args.batch_windows)
        det = bool(np.array_equal(p1, p2) and np.array_equal(h1, h2))
        exp_nw = int(np.ceil(len(audio) / WINDOW_SAMPLES))
        recs.append({'id': vid, 'dur_s': round(float(dur), 2), 'n_win': nw, 'n_win_expected': exp_nw,
                     'D1_deterministic': det, 'proj_norm': round(float(np.linalg.norm(p1)), 4),
                     'hid_norm': round(float(np.linalg.norm(h1)), 4),
                     'nan': bool(np.isnan(p1).any() or np.isnan(h1).any()),
                     'allzero': bool(not p1.any())})
        first_proj.append(p1)
        print(f"[clap-smoke]  {vid}: dur={dur:.1f}s n_win={nw}(exp {exp_nw}) D1={det} "
              f"|proj|={np.linalg.norm(p1):.4f} |hid|={np.linalg.norm(h1):.4f}", flush=True)

    P = np.stack(first_proj)
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    cos = Pn @ Pn.T
    off = cos[~np.eye(len(P), dtype=bool)]
    d1 = all(r['D1_deterministic'] for r in recs)
    d2 = all((not r['nan']) and (not r['allzero']) and r['n_win'] == r['n_win_expected'] for r in recs)
    d3 = bool(off.min() < 0.99)
    out = {'model': MODEL, 'n_smoke': len(recs), 'd_proj': d_proj, 'd_hidden': d_hid,
           'fe': {'sr': feat_ext.sampling_rate, 'nb_max_samples': feat_ext.nb_max_samples,
                  'mels': feat_ext.feature_size, 'truncation': feat_ext.truncation,
                  'padding': feat_ext.padding},
           'D1_determinism_all': d1, 'D2_sanity_all': d2, 'D3_discriminative': d3,
           'proj_cos_offdiag': {'min': round(float(off.min()), 4), 'mean': round(float(off.mean()), 4),
                                'max': round(float(off.max()), 4)},
           'records': recs, 'SMOKE_PASS': bool(d1 and d2 and d3)}
    dst = f'{REPO}/refine-logs/CLAP_SMOKE_OUT.json'
    json.dump(out, open(dst, 'w'), indent=1)
    print(f"\n[clap-smoke] D-1 determinism={d1}  D-2 sanity={d2}  D-3 discriminative={d3} "
          f"(off-diag cos min={off.min():.4f} mean={off.mean():.4f} max={off.max():.4f})", flush=True)
    print(f"[clap-smoke] SMOKE_PASS={out['SMOKE_PASS']} -> {dst}", flush=True)
    if not out['SMOKE_PASS']:
        raise SystemExit('SMOKE FAILED — stop per spec section 2.1')


def extract(args):
    decode_audio_pyav = _load_decode_fn()
    device = torch.device(args.device)
    feat_ext, model, dtype, d_proj, d_hid = build_model(device)
    proj_dim, hid_dim = 2 * d_proj, 2 * d_hid
    tag = MODEL.split('/')[-1]
    print(f'[clap-extract] model={MODEL} tag={tag} device={device} dtype={dtype} '
          f'proj_pooled={proj_dim} hidden_pooled={hid_dim} batch_windows={args.batch_windows}', flush=True)

    vid_root = f'{REPO}/data/video/{DATASET}/All'
    shard_dir = f'{REPO}/data/audio/{DATASET}/clap_{tag}'
    os.makedirs(shard_dir, exist_ok=True)

    groups = [('trainval', ('train', 'val'))] if args.splits == 'trainval' else \
             [('test', ('test',))] if args.splits == 'test' else \
             [('trainval', ('train', 'val')), ('test', ('test',))]

    for gname, splits in groups:
        ids, labels = read_gt_ids(splits)
        cache = f'{REPO}/data/audio/{DATASET}/clap_{tag}_{gname}.pt'
        manifest = f'{REPO}/data/audio/{DATASET}/clap_{tag}_{gname}_manifest.json'
        missing = [i for i in ids if not os.path.exists(f'{vid_root}/{i}.mp4')]
        assert not missing, f'{gname}: missing raw videos {missing[:10]} (+{len(missing)})'
        todo = [i for i in ids if not os.path.exists(f'{shard_dir}/{i}.npz')]
        print(f'[clap-extract][{gname}] N={len(ids)} cached={len(ids)-len(todo)} todo={len(todo)}', flush=True)

        t0, status, nwin = time.time(), {}, {}
        for n, vid in enumerate(todo, 1):
            try:
                audio, _dur = decode_audio_pyav(f'{vid_root}/{vid}.mp4', target_rate=TARGET_RATE)
                if audio is None or len(audio) < MIN_SAMPLES:
                    pv = np.zeros(proj_dim, dtype=np.float32)
                    hv = np.zeros(hid_dim, dtype=np.float32)
                    st, nw = 'no_audio', 0
                else:
                    pv, hv, nw = encode_video(audio, feat_ext, model, device, dtype, args.batch_windows)
                    st = 'ok'
                    if np.isnan(pv).any() or np.isnan(hv).any():
                        pv = np.nan_to_num(pv, nan=0.0); hv = np.nan_to_num(hv, nan=0.0)
                        st = 'ok_nanfix'
            except Exception as e:  # noqa: BLE001
                pv = np.zeros(proj_dim, dtype=np.float32)
                hv = np.zeros(hid_dim, dtype=np.float32)
                st, nw = f'ERR:{type(e).__name__}:{e}', 0
                print(f'[clap-extract][{gname}]  !! {vid}: {st}', flush=True)
            np.savez(f'{shard_dir}/{vid}.npz', proj=pv, hidden=hv, n_win=np.int64(nw))
            status[vid], nwin[vid] = st, nw
            if st == 'no_audio':
                print(f'[clap-extract][{gname}]  !! {vid}: no_audio', flush=True)
            if n % 50 == 0 or n == len(todo):
                el = time.time() - t0
                print(f'[clap-extract][{gname}]  {n}/{len(todo)} [{el:.0f}s, {el/max(n,1):.2f}s/vid]', flush=True)

        proj = np.zeros((len(ids), proj_dim), dtype=np.float32)
        hid = np.zeros((len(ids), hid_dim), dtype=np.float32)
        wins = np.zeros(len(ids), dtype=np.int64)
        for i, vid in enumerate(ids):
            z = np.load(f'{shard_dir}/{vid}.npz')
            proj[i], hid[i], wins[i] = z['proj'], z['hidden'], int(z['n_win'])
        n_zero = int((~proj.any(axis=1)).sum())
        n_nan = int(np.isnan(proj).sum() + np.isnan(hid).sum())
        norms = np.linalg.norm(proj, axis=1)
        torch.save({'ids': [ids], 'proj': torch.tensor(proj), 'hidden': torch.tensor(hid),
                    'labels': torch.tensor(labels), 'n_windows': torch.tensor(wins),
                    'model': MODEL, 'model_tag': tag, 'd_proj': d_proj, 'd_hidden': d_hid,
                    'proj_dim': proj_dim, 'hidden_dim': hid_dim, 'pool': 'mean_cat_max_over_windows',
                    'window_s': WINDOW_SAMPLES / TARGET_RATE, 'sample_rate': TARGET_RATE,
                    'channels': 1, 'scope': gname}, cache)
        man = {'dataset': DATASET, 'group': gname, 'splits': list(splits), 'N': len(ids),
               'n_pos': int(labels.sum()), 'model': MODEL, 'model_tag': tag,
               'proj_dim': proj_dim, 'hidden_dim': hid_dim, 'pool': 'mean_cat_max_over_windows',
               'window_s': WINDOW_SAMPLES / TARGET_RATE, 'sample_rate': TARGET_RATE,
               'n_zero_vector_rows': n_zero, 'n_nan': n_nan,
               'status_counts': _count(status if todo else {'(all-cached)': len(ids)}),
               'n_windows': {'min': int(wins.min()), 'median': int(np.median(wins)),
                             'max': int(wins.max()), 'total': int(wins.sum())},
               'example_proj_norms': {ids[0]: float(norms[0]),
                                      ids[len(ids) // 2]: float(norms[len(ids) // 2]),
                                      ids[-1]: float(norms[-1])},
               'elapsed_s': round(time.time() - t0, 1), 'cache': cache}
        json.dump(man, open(manifest, 'w'), indent=1)
        print(f'[clap-extract][{gname}] DONE proj={proj.shape} hidden={hid.shape} n_zero={n_zero} '
              f'n_nan={n_nan} windows(total={wins.sum()}, med={int(np.median(wins))}) '
              f'status={man["status_counts"]} -> {cache}', flush=True)


def _count(status):
    c = {}
    for v in status.values():
        key = v if v in ('ok', 'ok_nanfix', 'no_audio', '(all-cached)') else 'ERR'
        c[key] = c.get(key, 0) + 1
    return c


def parse_args():
    ap = argparse.ArgumentParser(description='CLAP general-audio embedding extractor (Stage A).')
    ap.add_argument('--splits', type=str, default='all', choices=['all', 'trainval', 'test'])
    ap.add_argument('--batch_windows', type=int, default=8)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--smoke', action='store_true', help='run the pre-declared smoke gate only')
    ap.add_argument('--n_smoke', type=int, default=20)
    return ap.parse_args()


if __name__ == '__main__':
    t = time.time()
    a = parse_args()
    smoke(a) if a.smoke else extract(a)
    print(f'[clap] elapsed {time.time()-t:.0f}s', flush=True)
