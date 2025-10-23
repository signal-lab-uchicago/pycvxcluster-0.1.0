
# pycvxcluster 

This is an implementation of the convex clustering method of Sun et al, 2021[1].

[1] Sun, Defeng, Kim-Chuan Toh, and Yancheng Yuan. "Convex clustering: Model, theoretical guarantee and efficient algorithm." Journal of Machine Learning Research 22.9 (2021): 1-32.

## Install

```bash
pip install ./pycvxcluster_pkg
# or editable:
pip install -e ./pycvxcluster_pkg
```

(Optional) For the alternative solver:
```bash
pip install 'pycvxcluster[alt]'
```

## Usage

```python
# The graphSVD code expects this path:
from pycvxcluster.pycvxcluster import SSNAL

# Or directly:
from pycvxcluster import SSNAL

ssnal = SSNAL(gamma=1.0, verbose=1)
ssnal.fit(X=features, weight_matrix=W, save_centers=True)
U_tilde = ssnal.centers_.T
```

The `SSNAL` class internally calls:
- `compute_weight_matrix` / `get_nam_wv_from_wm` (graph weights)
- `ssnal_wrapper` (solver)
- `find_clusters` (post-processing)
# pycvxcluster-0.1.0
