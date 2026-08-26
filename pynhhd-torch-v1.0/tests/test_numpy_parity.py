'''
Cross-validate against pynhhd-v1.1 (numpy) -- but ONLY on paths confirmed
NOT to be affected by the known numpy bugs (docs/CODE_REVIEW_REPORT.md):
the 2D structured-grid divcurl/gradient/decompose path. The 3D structured
path and the unstructured-mesh boundary path are deliberately excluded
here, since matching the (unfixed, as installed for this comparison) numpy
package there would mean matching a known bug rather than validating
correctness -- see test_structured_analytical.py for the 3D bug-regression
coverage instead.

Uses the `numpy_pynhhd` fixture (conftest.py) rather than a plain
`import pynhhd`, since this package now installs under the same `pynhhd`
name as the numpy package (see pyproject.toml) -- see conftest.py's
docstring.
'''

import numpy as np
import torch

from pynhhd import nHHD as nHHD_torch


def test_2d_divcurl_matches_numpy(numpy_pynhhd):
    rng = np.random.default_rng(0)
    Y, X = 20, 24
    dy, dx = 0.1, 0.15
    vfield = rng.standard_normal((Y, X, 2))

    np_mesh = numpy_pynhhd.StructuredGrid(grid=(Y, X), spacings=(dy, dx))
    div_np, curl_np = np_mesh.divcurl(vfield)

    h_t = nHHD_torch(grid=(Y, X), spacings=(dy, dx), dtype=torch.float64)
    div_t, curl_t = h_t.mesh.divcurl(torch.as_tensor(vfield).unsqueeze(0))

    np.testing.assert_allclose(div_t[0].numpy(), div_np, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(curl_t[0].numpy(), curl_np, rtol=1e-8, atol=1e-8)


def test_2d_decompose_matches_numpy(numpy_pynhhd):
    rng = np.random.default_rng(1)
    Y, X = 18, 22
    dy, dx = 0.1, 0.1
    vfield = rng.standard_normal((Y, X, 2))

    try:
        np_h = numpy_pynhhd.nHHD(grid=(Y, X), spacings=(dy, dx))
    except AttributeError as e:
        # pynhhd-v1.1's PoissonSolver sets `self.dtype = np.float`, an alias
        # removed in numpy>=1.24 -- see docs/CODE_REVIEW_REPORT.md sec 2.5.
        # If the installed numpy package hasn't had that bug fixed (e.g. this
        # branch was built off master, not fix/pynhhd-numpy-bugs), this
        # comparison simply cannot run against it -- skip with a clear reason
        # rather than reporting a spurious failure in the torch port.
        import pytest
        pytest.skip('pynhhd-v1.1 install hits the known np.float bug '
                     '(numpy>=1.24); see docs/CODE_REVIEW_REPORT.md sec 2.5: {}'.format(e))
    np_h.decompose(vfield.copy())

    t_h = nHHD_torch(grid=(Y, X), spacings=(dy, dx), dtype=torch.float64)
    res = t_h.decompose(vfield.copy())

    # FFT-based (torch) vs direct-convolution-via-fftconvolve (numpy) should
    # agree closely, but not bit-for-bit (different FFT libraries/algorithms) --
    # use a loose-ish but still meaningful tolerance.
    np.testing.assert_allclose(res.d, np_h.d, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(res.r, np_h.r, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(res.h, np_h.h, rtol=1e-4, atol=1e-6)
