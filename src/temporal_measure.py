"""Integration cells for the existing deterministic snippet subsampling grid."""
import numpy as np


def integration_cells(snippet_bounds, duration, count=None):
    bounds = np.asarray(snippet_bounds, dtype=np.float64)
    n = len(bounds)
    if n == 0 or duration <= 0 or not np.isfinite(bounds).all():
        raise ValueError('invalid snippet timeline')
    if count is None:
        count = n
    if not 1 <= count <= n:
        raise ValueError('invalid selected snippet count')
    # Same indices as macilsd.utils.uniform_extract, not a different sampler.
    if n > np.iinfo(np.uint16).max:
        raise ValueError('existing uint16 feature sampler cannot represent this video')
    selected = np.linspace(0, n-1, count, dtype=np.uint16) if count < n else np.arange(n)
    centers = bounds[selected].mean(-1)/float(duration)
    if (np.diff(centers) < 0).any():
        raise ValueError('nonmonotonic snippet timeline')
    edges = np.concatenate([[0.], ((centers[1:]+centers[:-1])/2).clip(0, 1), [1.]])
    cells = np.stack([edges[:-1], edges[1:]], -1).astype(np.float32)
    if (cells[:, 1] < cells[:, 0]).any() or not np.isclose((cells[:, 1]-cells[:, 0]).sum(), 1):
        raise ValueError('integration cells must partition [0,1]')
    return cells
