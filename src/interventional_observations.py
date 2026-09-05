"""Validated raw observations shared by interventional evidence experiments."""
import json
import numpy as np

VERSION = '2026-09-05 four-input evidence v2 raw logits'


def load_observations(path, k, vid):
    obj = json.loads(path.read_text())
    if obj.get('version') != VERSION:
        raise ValueError(f'input version is not raw logits v2: {path}')
    if obj['id'] != vid or obj['order'] != ['av', 'v', 'a', 'empty']:
        raise ValueError(f'input identity/order: {path}')
    windows = obj['windows']
    if len(windows) != k or [w['index'] for w in windows] != list(range(k)):
        raise ValueError(f'window alignment: {path}')
    logits = np.asarray([w['log_odds'] for w in windows], dtype=np.float32)
    entropy = np.asarray([w['entropy'] for w in windows], dtype=np.float32)
    if logits.shape != (k, 4) or entropy.shape != (k, 4):
        raise ValueError(f'input shape: {path}')
    if not np.isfinite(logits).all() or not np.isfinite(entropy).all():
        raise ValueError(f'nonfinite input: {path}')
    return logits, entropy
