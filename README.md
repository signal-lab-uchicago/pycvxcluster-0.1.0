
# pycvxcluster 

This is an implementation of the convex clustering method of Sun et al, 2021[1].

[1] Sun, Defeng, Kim-Chuan Toh, and Yancheng Yuan. "Convex clustering: Model, theoretical guarantee and efficient algorithm." Journal of Machine Learning Research 22.9 (2021): 1-32.


## Installation

```bash
# local install (from the parent folder that contains pycvxcluster_pkg/)
pip install ./pycvxcluster_pkg

# or editable install for dev
pip install -e ./pycvxcluster_pkg
```

### Optional / native dependencies

The ADMM / SSNAL backends can use **CHOLMOD** via `scikit-sparse` for faster sparse factorizations. Recommended:

```bash
# conda (recommended)
conda install -c conda-forge scikit-sparse

# or pip (requires SuiteSparse / CHOLMOD available to your compiler toolchain)
pip install scikit-sparse
```

> If `sksparse` isn’t present, importing the package still works; you’ll only need it when you call into the SSNAL solver path that requires CHOLMOD.

---

## Quick start

```python
import numpy as np
from sklearn.datasets import make_blobs
from pycvxcluster.pycvxcluster import SSNAL

# 1) toy data (3 clusters in 2D)
X, y_true = make_blobs(n_samples=600, centers=3, cluster_std=0.60, random_state=7)

# 2) run convex clustering; pycvxcluster will build a k-NN graph for you
ssnal = SSNAL(gamma=5.0, k=10, phi=0.5, verbose=1)
ssnal.fit(X, save_labels=True, save_centers=True)   # weight_matrix=None => auto k-NN

print("n_clusters:", getattr(ssnal, "n_clusters_", None))
print("labels shape:", ssnal.labels_.shape)
print("centers shape:", ssnal.centers_.shape)       # centers_ is (d x n) by convention
```

**What gets populated**

- `labels_` and `n_clusters_` (from `find_clusters`)  
- `centers_` (columns are the smoothed/clustered locations)  
- `primobj_`, `dualobj_`, `eta_`, `iter_`, `termination_`, `ssnal_runtime_`, `total_time_`

---

## Using a precomputed sample graph (weight matrix)

You can pass your own symmetric sparse weight matrix \(W\) (e.g., built from coordinates or a known graph). The estimator converts it to the node‑arc form internally and reuses it across calls.

```python
import numpy as np, scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from pycvxcluster.pycvxcluster import SSNAL

# Build W from kNN in feature space (Gaussian weights)
def knn_weight_matrix(X, k=10, sigma=None):
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    dist, idx = nbrs.kneighbors(X)
    if sigma is None:
        sigma = np.median(dist[:, 1:])
    rows, cols, data = [], [], []
    for i in range(n):
        for j, d in zip(idx[i, 1:], dist[i, 1:]):
            w = float(np.exp(-d**2 / (2*sigma*sigma)))
            rows += [i, j]; cols += [j, i]; data += [w, w]
    W = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return W.maximum(W.T)

X, _ = make_blobs(n_samples=600, centers=3, cluster_std=0.60, random_state=7)
W = knn_weight_matrix(X, k=10)

ssnal = SSNAL(gamma=5.0, verbose=1)
ssnal.fit(X, weight_matrix=W, save_labels=True, save_centers=True, recalculate_weights=False)
```

**Tip:** set `recalculate_weights=False` to reuse your `W` across parameter sweeps (e.g., trying different `gamma`).

---

## Warm‑starting across a λ/γ grid

When you scan over regularization values, warm‑starting can cut runtime significantly. After a fit, reuse the solver state as the initial point for the next run:

```python
import numpy as np
from pycvxcluster.pycvxcluster import SSNAL

ssnal = SSNAL(gamma=1e-2, k=10, verbose=0)
ssnal.fit(X, weight_matrix=W, save_centers=True, save_labels=False)

# Warm-start next values by seeding the internal kwargs
ssnal.kwargs.update(x0=ssnal.centers_, y0=getattr(ssnal, "y_", None), z0=getattr(ssnal, "z_", None))

for g in np.geomspace(1e-2, 10, 12)[1:]:
    ssnal.gamma = g
    ssnal.fit(X, weight_matrix=W, save_centers=True, save_labels=False, recalculate_weights=False)
    # update warm-starts again if desired
    ssnal.kwargs.update(x0=ssnal.centers_, y0=getattr(ssnal, "y_", None), z0=getattr(ssnal, "z_", None))
```

This mirrors how your GraphSVD code sweeps λ with warm‑starts. The estimator’s `kwargs` dict is forwarded to the underlying `ssnal_wrapper`, so `x0`, `y0`, `z0` are picked up automatically.

---

## API (SSNAL)

```python
from pycvxcluster.pycvxcluster import SSNAL

SSNAL(
  k=10,                 # k-NN graph size when building weights internally
  phi=0.5,              # weight-decay parameter used in compute_weight_matrix
  gamma=1.0,            # convex clustering penalty strength
  clustertol=1e-5,      # threshold when post-processing centers into clusters
  sigma=1.0,            # model parameter forwarded to solver
  maxiter=1000,         # SSNAL max iters
  admm_iter=100,        # ADMM iterations to warm-start SSNAL
  stoptol=1e-6,         # SSNAL stopping tolerance
  ncgtolconst=0.5,      # SSN-CG tolerance constant
  verbose=0,            # 0/1/2
  **kwargs              # forwarded to the solver (e.g., x0, y0, z0 for warm-start)
)
```

**Key attributes after `fit`:**  
`centers_`, `labels_`, `n_clusters_`, `weight_matrix_`, `node_arc_matrix_`, `weight_vec_`, `primobj_`, `dualobj_`, `eta_`, `iter_`, `termination_`, `ssnal_runtime_`, `total_time_`.

---

## Using with Graph‑GpLSI (GraphSVD)

The estimator integrates directly in your GraphSVD step—exactly how your code already does it during the λ‑grid search and the final pass; you update `ssnal.gamma`, reuse the internal warm‑starts, and feed back the smoothed centers as `U_tilde`. (See the `lambda_search` and `update_U_tilde` calls in your GraphSVD implementation.)

Minimal sketch:

```python
# inside your graphSVD loop
ssnal = SSNAL(verbose=0)
for lambd in lambd_grid:
    ssnal.gamma = lambd
    ssnal.fit(X=X_tilde @ V, weight_matrix=W, save_centers=True, save_labels=False,
              recalculate_weights=(lambd is lambd_grid[0]))
    # warm-starts for next iteration
    ssnal.kwargs.update(x0=ssnal.centers_, y0=ssnal.y_, z0=ssnal.z_)
U_tilde = ssnal.centers_.T
```

---

## Troubleshooting

- **`ModuleNotFoundError: sksparse`** – install `scikit-sparse` (see Installation).  
- **Large memory / long runs** – try smaller `k`, coarser λ/γ grid, or enable warm‑starts between λ values (above).  
- **Graph input** – ensure your `weight_matrix` is CSR or CSC, symmetric, and with non‑negative weights.

---

## License

MIT
