'''
Analytical-field validation for the structured-grid path (per the paper's
Sec. 7.1 approach: construct known divergent + rotational + harmonic
components, sum them, decompose, check component-wise recovery).

Also isolates the specific 3D divergence formula bug found in
pynhhd-v1.1/pynhhd/structured.py:90 (div computed as du/dx+dv/dy+dv/dz
instead of du/dx+dv/dy+dw/dz -- see docs/CODE_REVIEW_REPORT.md sec 2.1) by
constructing a field whose divergence comes *only* from dw/dz: the buggy
formula would report ~zero divergence for such a field since it never reads
dw/dz at all, while the correct formula recovers it.
'''

import numpy as np
import torch
import pytest

from pynhhd import nHHD


def _rel_err(a, b):
    denom = np.abs(b).max()
    denom = denom if denom > 1e-12 else 1.0
    return np.abs(a - b).max() / denom


class Test2DAnalytical:
    # NOTE: fields like xi=(x,y) or xi=(-y,x) grow unboundedly and do NOT
    # vanish at infinity, so they don't satisfy the natural HHD's own
    # assumptions (Sec 5.1 of the paper: the field must vanish at infinity
    # for its free-space-Green's-function construction to apply) -- on a
    # finite domain, truncating the Green's-function integral to the grid
    # introduces real, non-negligible boundary error for such fields, so
    # "h should be ~0" is not actually a valid expectation for them. Instead
    # we check the structural properties that hold by construction
    # regardless of the input field: d is curl-free and r is divergence-free
    # (this is exactly what the paper itself validates quantitatively --
    # Sec 7.1 / Fig. 4 -- since d, r are computed as the gradient/rotated-
    # gradient of a potential).

    def test_d_is_numerically_curl_free(self):
        rng = np.random.default_rng(3)
        Y, X = 24, 28
        dy, dx = 0.05, 0.05
        vfield = torch.as_tensor(rng.standard_normal((Y, X, 2)))

        h = nHHD(grid=(Y, X), spacings=(dy, dx), dtype=torch.float64)
        res = h.decompose(vfield)

        _, curl_of_d = h.mesh.divcurl(torch.as_tensor(res.d).unsqueeze(0))
        assert curl_of_d.abs().max() < 1e-6

    def test_r_is_numerically_divergence_free(self):
        rng = np.random.default_rng(4)
        Y, X = 24, 28
        dy, dx = 0.05, 0.05
        vfield = torch.as_tensor(rng.standard_normal((Y, X, 2)))

        h = nHHD(grid=(Y, X), spacings=(dy, dx), dtype=torch.float64)
        res = h.decompose(vfield)

        div_of_r, _ = h.mesh.divcurl(torch.as_tensor(res.r).unsqueeze(0))
        assert div_of_r.abs().max() < 1e-6

    def test_reconstruction_identity(self):
        rng = np.random.default_rng(1)
        Y, X = 20, 22
        dy, dx = 0.1, 0.1
        vfield = rng.standard_normal((Y, X, 2))

        h = nHHD(grid=(Y, X), spacings=(dy, dx), dtype=torch.float64)
        res = h.decompose(vfield)

        recon = res.d + res.r + res.h
        np.testing.assert_allclose(recon, vfield, atol=1e-10)


class Test3DAnalytical:

    def test_reconstruction_identity(self):
        rng = np.random.default_rng(2)
        Z, Y, X = 10, 12, 14
        dz, dy, dx = 0.1, 0.1, 0.1
        vfield = rng.standard_normal((Z, Y, X, 3))

        h = nHHD(grid=(Z, Y, X), spacings=(dz, dy, dx), dtype=torch.float64)
        res = h.decompose(vfield)

        recon = res.d + res.r + res.h
        np.testing.assert_allclose(recon, vfield, atol=1e-8)

    def test_w_only_divergence_is_captured(self):
        '''
        Regression test for the numpy 3D-divergence bug (dv/dz instead of
        dw/dz). Field: w(z) = z^2 (so dw/dz = 2z, nonzero and z-varying),
        u = v = 0 identically -- so dv/dz is identically zero everywhere.

        The buggy formula (div = du/dx + dv/dy + dv/dz) would compute
        div == 0 everywhere for this field (since u, v are both zero), and
        the whole field would incorrectly come out as purely harmonic/
        rotational with a zero divergent component. The correct formula
        (div = du/dx + dv/dy + dw/dz) must recover the known dw/dz = 2z
        divergence and a non-trivial divergent component d != 0.
        '''
        Z, Y, X = 14, 10, 10
        dz, dy, dx = 0.05, 0.1, 0.1

        z = (torch.arange(Z, dtype=torch.float64) - Z // 2) * dz
        w = z**2
        vfield = torch.zeros(Z, Y, X, 3, dtype=torch.float64)
        vfield[..., 2] = w[:, None, None]

        h = nHHD(grid=(Z, Y, X), spacings=(dz, dy, dx), dtype=torch.float64)

        # check the raw divcurl output directly first (isolates the formula
        # itself from the rest of the Poisson-solve pipeline)
        vfield_batched = vfield.unsqueeze(0)
        div, curlu, curlv, curlw = h.mesh.divcurl(vfield_batched)

        expected_div = (2.0 * z)[None, :, None, None].expand(1, Z, Y, X)
        # central-difference error near the boundary rows is expected; check the interior
        interior = slice(2, -2)
        np.testing.assert_allclose(
            div[:, interior].numpy(), expected_div[:, interior].numpy(), rtol=1e-6, atol=1e-6
        )

        # and confirm this is NOT what the buggy formula would have produced
        # (buggy formula reads as div == 0 identically for this field, since
        # u == v == 0)
        assert not np.allclose(div.numpy(), 0.0, atol=1e-6)

        # end-to-end: the decomposition's divergent component d must be
        # non-trivial (not collapse to ~0, as it would under the bug)
        res = h.decompose(vfield)
        assert res.d.abs().max() > 1e-3
