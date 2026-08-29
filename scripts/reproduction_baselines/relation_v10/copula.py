"""Label-free expert clustering and robust cluster-quality aggregation."""
import numpy as np


def dependence_clusters(values, threshold=.999):
    """Cluster duplicate/near-duplicate calibrated centered evidence streams."""
    x = np.asarray(values, float); n = x.shape[1]
    corr = np.nan_to_num(np.corrcoef(x, rowvar=False), nan=0., posinf=0., neginf=0.)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        i, j = find(i), find(j)
        if i != j: parent[j] = i
    for i in range(n):
        for j in range(i):
            if corr[i, j] >= threshold: union(i, j)
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)
    return list(groups.values()), corr


def cluster_weights(values, clusters, robust=True):
    """Give each cluster one unit of mass; optionally downweight outlier clusters."""
    x = np.asarray(values, float)
    centroids = np.stack([x[:, group].mean(1) for group in clusters], 1)
    if not robust or len(clusters) < 3:
        quality = np.ones(len(clusters))
    else:
        median = np.median(centroids, axis=1, keepdims=True)
        deviation = np.median(np.abs(centroids - median), axis=0)
        scale = max(float(np.median(deviation)), 1e-6)
        corr = np.nan_to_num(np.corrcoef(centroids, rowvar=False), nan=0.)
        agreement = np.empty(len(clusters))
        for i in range(len(clusters)):
            other = np.delete(corr[i], i)
            agreement[i] = np.clip((1 + np.median(other)) / 2, .05, 1.)
        quality = np.exp(-deviation / scale) * agreement
        quality = np.maximum(quality, 1e-6)
    mass = quality / quality.sum(); weights = np.zeros(x.shape[1])
    for value, group in zip(mass, clusters): weights[group] = value / len(group)
    return weights, quality, centroids


def fit(values, threshold=.999):
    clusters, corr = dependence_clusters(values, threshold)
    equal, _, _ = cluster_weights(values, clusters, robust=False)
    robust, quality, _ = cluster_weights(values, clusters, robust=True)
    return {"clusters": clusters, "correlation": corr, "cluster_equal": equal,
            "cluster_robust": robust, "cluster_quality": quality}
