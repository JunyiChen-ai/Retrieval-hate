"""Read fixed split IDs and window transcripts for label-free preparation."""
import json


def fixed_ids(root, corpus):
    splits = {s: [v.strip() for v in (root/'results/reproduction/splits'/f'{corpus}_{s}.txt').read_text().splitlines() if v.strip()]
              for s in ['train', 'val', 'test']}
    ids = sum(splits.values(), [])
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate ID or split overlap')
    return sorted(ids)


def window_transcripts(directory, k, ids):
    rows = {}
    for path in sorted(directory.glob(f'*_asrK{k}_whisper-large-v3.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            vid, texts = str(obj['id']), obj['window_text']
            if len(texts) != k or not all(isinstance(x, str) for x in texts):
                raise ValueError(f'invalid ASR: {path} {vid}')
            if vid in rows and rows[vid] != texts:
                raise ValueError(f'conflicting ASR: {vid}')
            rows[vid] = texts
    missing = sorted(set(ids)-rows.keys())
    rows.update({v: ['']*k for v in missing})
    return rows, missing
