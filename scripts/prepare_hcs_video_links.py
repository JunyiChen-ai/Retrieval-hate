"""Recreate canonical .mp4 aliases without transcoding or overwriting media."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = Path.home() / 'data/HateClipSeg/videos'
dest = root / 'data/video/HateClipSeg/All'
if dest.is_symlink():
    if dest.resolve() != source.resolve():
        raise RuntimeError('unrecognized existing alias; inspect manually')
    dest.unlink()  # Only the preparer's directory alias, never media.
dest.mkdir(parents=True, exist_ok=True)
ids = sum([(root/'results/reproduction/splits'/f'hateclipseg_{s}.txt').read_text().split()
           for s in ['train', 'val', 'test']], [])
for vid in ids:
    matches = [p for p in source.glob(vid + '.*') if p.suffix in ['.mp4', '.webm', '.mkv', '.avi']]
    if len(matches) != 1:
        raise RuntimeError(f'expected one original for {vid}')
    alias = dest / f'{vid}.mp4'
    if alias.exists() or alias.is_symlink():
        if alias.resolve() != matches[0].resolve():
            raise RuntimeError(f'existing mapping differs: {vid}')
    else:
        alias.symlink_to(matches[0])
print(f'canonical aliases ready: {len(ids)}')
