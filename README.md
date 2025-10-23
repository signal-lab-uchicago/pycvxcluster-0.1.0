
# pycvxcluster (local package)

This is a local, installable packaging of your `pycvxcluster` convex clustering utilities.

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
