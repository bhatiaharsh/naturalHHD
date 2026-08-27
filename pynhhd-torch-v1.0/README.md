## pynhhd (PyTorch)

#### PyTorch implementation of the natural Helmholtz-Hodge decomposition

Implementation of the TVCG paper
[doi:10.1109/TVCG.2014.2312012](http://dx.doi.org/10.1109/TVCG.2014.2312012).
For technical background, see the paper (`../docs/2014_TVCG_nHHD.pdf`) and
`../docs/PAPER_CRITIQUE.md`. The code is released under the BSD licence;
please cite the paper if you use this technique.

This is a from-scratch PyTorch port of `pynhhd-v1.1` (the numpy/scipy
implementation), built as a separate sibling package rather than a
drop-in replacement. It targets **GPU acceleration and batched
decomposition** (many fields/timesteps solved against one precomputed
Green's function) as the primary goal; it deliberately does not support
autograd (every public entry point runs under `torch.no_grad()`).
`pynhhd-v1.1` is unaffected by this package and remains the numpy
reference.

See `../docs/CODE_REVIEW_REPORT.md` for the bugs found in `pynhhd-v1.1`
that motivated parts of this port's design; each is fixed here by
construction rather than carried over (see each module's docstring for
specifics).

### Install

```
pip install -e .
```

Requires `torch`, `numpy`, and `scipy` (the last only for one-time,
construction-time work: Delaunay triangulation for unstructured meshes,
and the spherical solver's self-term integral).

### Usage

```python
import torch
from pynhhd import nHHD

# regular grid, 2D or 3D
h = nHHD(grid=(Y, X), spacings=(dy, dx))              # + device=, dtype= (default float32)
result = h.decompose(vfield)                          # vfield: (Y, X, 2) or (B, Y, X, 2)
result.d, result.r, result.h                           # divergent, rotational, harmonic components

# unstructured (2D triangular) mesh
h = nHHD(points=vertices)                              # + simplices= (Delaunay if omitted)
result = h.decompose(vfield)                            # vfield: (nsimplices, 2) or (B, nsimplices, 2)

# spherical (lat/lon) grid, 2D only
h = nHHD(sphericalgrid=(ny, nx), spacings=(dlat, dlon), lat=lat, lon=lon)
result = h.decompose(vfield)                            # vfield: (ny, nx, 2) or (B, ny, nx, 2)
```

`vfield` may be a numpy array or a torch tensor; the output type mirrors
the input. `decompose()` returns a `DecompositionResult(div, curl, d, r,
h, potentials)`; for a single (unbatched) field, `h.d`/`h.r`/`h.h`/etc.
are also set on the `nHHD` instance as a convenience.

### Status

| Grid type | Status |
|---|---|
| Regular grid (2D/3D) | done |
| Unstructured triangular mesh (2D) | done |
| Spherical / lat-lon grid (2D) | done |
| Unstructured tetrahedral mesh (3D) | not implemented (raises `NotImplementedError`) |
| Spherical grid (3D) | not implemented (raises `NotImplementedError`) |

Points-based (unstructured) inputs use a dense `N x N` Green's-function
matrix, which is `O(N^2)` memory on any device -- `PoissonSolver` refuses
to build it above `max_points` (default 20,000) rather than risk a
silent out-of-memory failure; pass a larger `max_points` explicitly if
you have the device memory.

### Tests

```
pip install -e ".[test]"
pytest tests/ -v
```

The suite includes: an FFT-convolution primitive check against
`scipy.signal.fftconvolve`; analytical-field checks and a 3D regression
test that isolates the exact term a known `pynhhd-v1.1` divergence bug
zeroed out; a hand-checkable boundary-vertex regression test for a known
`pynhhd-v1.1` unstructured-mesh curl bug; a from-scratch reimplementation
of `pynhhd-v1.1`'s spherical solver algorithm (used to independently
verify the FFT-based reformulation, since structural self-consistency
checks alone can't catch a wrong-but-self-consistent solver); batch-vs-
loop consistency; no-grad enforcement; numpy/torch round-tripping; device
parity (skipped without CUDA); and comparisons against `pynhhd-v1.1` on
the paths confirmed not to carry its known bugs.
