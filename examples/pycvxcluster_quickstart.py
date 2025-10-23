

"""
Quick start example for the pycvxcluster package.

Run:
    pip install ./pycvxcluster_pkg   # or pip install -e ./pycvxcluster_pkg
    python pycvxcluster_quickstart.py
"""
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from pycvxcluster.pycvxcluster import SSNAL


def knn_weight_matrix(X, k=10, sigma=None):
    """Build a symmetric kNN weight matrix with Gaussian weights."""
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    dist, idx = nbrs.kneighbors(X)
    if sigma is None:
        # robust scale
        sigma = float(np.median(dist[:, 1:]))
        if sigma <= 0:
            sigma = 1.0
    rows, cols, data = [], [], []
    for i in range(n):
        for j, d in zip(idx[i, 1:], dist[i, 1:]):
            w = float(np.exp(-d**2 / (2*sigma*sigma)))
            rows += [i, j]; cols += [j, i]; data += [w, w]
    W = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return W.maximum(W.T)


def main():
    # 1) toy data (3 clusters)
    X, y_true = make_blobs(n_samples=600, centers=3, cluster_std=0.60, random_state=7)

    # 2a) auto-weights mode (builds kNN internally)
    ssnal_auto = SSNAL(gamma=5.0, k=10, phi=0.5, verbose=1)
    ssnal_auto.fit(X, save_labels=True, save_centers=True)
    print("[auto] clusters:", getattr(ssnal_auto, "n_clusters_", None))

    # 2b) precomputed weight matrix (from kNN)
    W = knn_weight_matrix(X, k=10)
    ssnal = SSNAL(gamma=5.0, verbose=1)
    ssnal.fit(X, weight_matrix=W, save_labels=True, save_centers=True, recalculate_weights=False)
    print("[precomputed W] clusters:", getattr(ssnal, "n_clusters_", None))

    # 3) sweep gamma with warm-starts (reusing W)
    import numpy as np
    grid = np.geomspace(1e-2, 10, 8)
    ssnal = SSNAL(gamma=grid[0], verbose=0)
    ssnal.fit(X, weight_matrix=W, save_labels=False, save_centers=True, recalculate_weights=False)
    ssnal.kwargs.update(x0=ssnal.centers_, y0=getattr(ssnal, "y_", None), z0=getattr(ssnal, "z_", None))
    for g in grid[1:]:
        ssnal.gamma = float(g)
        ssnal.fit(X, weight_matrix=W, save_labels=False, save_centers=True, recalculate_weights=False)
        ssnal.kwargs.update(x0=ssnal.centers_, y0=getattr(ssnal, "y_", None), z0=getattr(ssnal, "z_", None))
    print("Gamma sweep done; final centers shape:", ssnal.centers_.shape)


if __name__ == "__main__":
    main()
