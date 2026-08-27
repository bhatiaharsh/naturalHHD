'''
Validation for the spherical (lat/lon) grid path -- in particular, that the
new FFT-based solver (poisson.py's _solve_one_spherical +
greens.py's create_sphericalGrid_kernel) reproduces the same numerical
answer as pynhhd-v1.1's original per-latitude, per-target-point
roll+dblquad loop (poisson.py's _solve_spherical), for a small grid -- this
is the most direct regression check for a path with no simple closed-form
analytical ground truth (see the project plan).

`_reference_spherical_solve` below is a standalone reimplementation of
pynhhd-v1.1's exact roll-based algorithm (same Green's-function formula,
same dblquad self-term), rather than importing the pynhhd package directly
-- this deliberately avoids coupling this check to pynhhd-v1.1's
np.float/numpy>=1.24 bug (docs/CODE_REVIEW_REPORT.md sec 2.5), which is
unrelated to what's being validated here and would otherwise make this
specific (important) check silently skip depending on which numpy version
happens to be installed. It caught a real bug during development: an
earlier version of _solve_one_spherical implemented the contraction as a
circular CONVOLUTION, which coincidentally matches pynhhd-v1.1's actual
roll-based circular CORRELATION only for grids that are exactly global
(360-degree-spanning) in longitude -- structural checks alone (d curl-free,
r divergence-free, reconstruction identity) all still passed under that
bug, since those only check relationships *between* the solver's own
outputs, not that the outputs are correct; only this direct comparison
against an independent reference caught it. Tested here on both a
non-global (regional) and a genuinely global grid.

The divcurl finite-difference formula does not touch PoissonSolver at all,
so it's compared directly against pynhhd-v1.1's StructuredSphericalGrid
(unaffected by the np.float bug).
'''

import math
import numpy as np
from scipy import integrate
import torch
import pytest

from pynhhd import nHHD as nHHD_torch
from pynhhd.spherical import StructuredSphericalGrid
from pynhhd.poisson import PoissonSolver as TorchPoissonSolver


def _small_grid(ny=6, nx=8, dlat=20.0, dlon=25.0, lon0=-90.0, lat0=-50.0):
    lon = (np.arange(nx) * dlon) + lon0
    lat = (np.arange(ny) * dlat) + lat0
    lon2, lat2 = np.meshgrid(lon, lat)
    return lat2, lon2, (dlat, dlon)


def _reference_spherical_solve(div, curl, lat2, lon2, dlat, dlon, r=6371e3, tuning_param=0.0625):
    '''Standalone reimplementation of pynhhd-v1.1's roll-based spherical
    solve -- see module docstring for why this doesn't just import pynhhd.'''
    ny, nx = lat2.shape
    rgdx = np.radians([dlat, dlon])
    rlat = np.radians(lat2)
    rlon = np.radians(lon2)

    def greens_row(j):
        delta_lon = np.abs(rlon[j, 0] - rlon)
        cos_d, sin_d = np.cos(delta_lon), np.sin(delta_lon)
        coslat1, sinlat1 = np.cos(rlat[j, 0]), np.sin(rlat[j, 0])
        coslat2, sinlat2 = np.cos(rlat), np.sin(rlat)
        num = np.sqrt((coslat2 * sin_d) ** 2 + (coslat1 * sinlat2 - sinlat1 * coslat2 * cos_d) ** 2)
        den = sinlat1 * sinlat2 + coslat1 * coslat2 * cos_d
        gamma = np.arctan2(num, den)
        with np.errstate(divide='ignore'):
            return -np.log(np.sin(0.5 * gamma)) / (2 * np.pi)

    def self_term(j):
        def integrand(x, y):
            return -math.log(math.sin(math.sqrt(x * x + y * y) * 0.5)) / (2 * math.pi)
        da1 = tuning_param * rgdx[0]
        da2 = tuning_param * rgdx[1] * math.cos(rlat[j, 0])
        val, _ = integrate.dblquad(integrand, 0, da1, 0, da2)
        return val / (da1 * da2)

    dxy = (r * rgdx[0]) * (r * np.cos(rlat) * rgdx[1])

    phi = np.zeros((ny, nx))
    psi = np.zeros((ny, nx))
    for j in range(ny):
        G = greens_row(j)
        G[j, 0] = self_term(j)
        for i in range(nx):
            rolled = np.roll(G, i, axis=1)
            phi[j, i] = np.sum(div * rolled * dxy)
            psi[j, i] = np.sum(curl * rolled * dxy)
    return phi, psi


@pytest.mark.parametrize('grid_kind,grid_kwargs', [
    ('regional (non-global longitude span)', dict(ny=6, nx=8, dlat=20.0, dlon=25.0, lon0=-90.0)),
    ('global (360-degree longitude span)', dict(ny=6, nx=12, dlat=20.0, dlon=30.0, lon0=-180.0)),
])
def test_solve_matches_reference_roll_based_solver(grid_kind, grid_kwargs):
    lat2, lon2, spacings = _small_grid(**grid_kwargs)
    rng = np.random.default_rng(1)
    div = rng.standard_normal(lat2.shape)
    curl = rng.standard_normal(lat2.shape)

    phi_ref, psi_ref = _reference_spherical_solve(div, curl, lat2, lon2, *spacings, tuning_param=0.0625)

    t_solver = TorchPoissonSolver(sphericalgrid=lat2.shape, spacings=spacings, lat=lat2, lon=lon2,
                                   tuning_param=0.0625, dtype=torch.float64)
    t_solver.prepare()
    phi_t, psi_t = t_solver.solve([torch.as_tensor(div), torch.as_tensor(curl)])

    np.testing.assert_allclose(phi_t.numpy(), phi_ref, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(psi_t.numpy(), psi_ref, rtol=1e-9, atol=1e-6)


def test_divcurl_matches_numpy(numpy_pynhhd):
    lat2, lon2, spacings = _small_grid()
    rng = np.random.default_rng(0)
    vfield = rng.standard_normal((*lat2.shape, 2))

    np_mesh = numpy_pynhhd.StructuredSphericalGrid(sphericalgrid=lat2.shape, spacings=spacings, lat=lat2, lon=lon2)
    div_np, curl_np = np_mesh.divcurl(vfield, glob=True)

    t_mesh = StructuredSphericalGrid(sphericalgrid=lat2.shape, spacings=spacings, lat=lat2, lon=lon2,
                                      dtype=torch.float64)
    div_t, curl_t = t_mesh.divcurl(torch.as_tensor(vfield).unsqueeze(0))

    np.testing.assert_allclose(div_t[0].numpy(), div_np, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(curl_t[0].numpy(), curl_np, rtol=1e-6, atol=1e-8)


def test_reconstruction_identity_and_structure():
    lat2, lon2, spacings = _small_grid()
    rng = np.random.default_rng(2)
    vfield = torch.as_tensor(rng.standard_normal((*lat2.shape, 2)), dtype=torch.float64)

    h = nHHD_torch(sphericalgrid=lat2.shape, spacings=spacings, lat=lat2, lon=lon2, dtype=torch.float64)
    res = h.decompose(vfield)

    recon = res.d + res.r + res.h
    np.testing.assert_allclose(recon, vfield.numpy(), atol=1e-8)

    div_of_r, _ = h.mesh.divcurl(torch.as_tensor(res.r).unsqueeze(0))
    _, curl_of_d = h.mesh.divcurl(torch.as_tensor(res.d).unsqueeze(0))
    assert div_of_r.abs().max() < 1e-6
    assert curl_of_d.abs().max() < 1e-6


def test_batch_matches_loop():
    lat2, lon2, spacings = _small_grid()
    rng = np.random.default_rng(3)

    h = nHHD_torch(sphericalgrid=lat2.shape, spacings=spacings, lat=lat2, lon=lon2, dtype=torch.float64)
    B = 3
    batch = torch.as_tensor(rng.standard_normal((B, *lat2.shape, 2)), dtype=torch.float64)
    res_batch = h.decompose(batch)

    for b in range(B):
        res_one = h.decompose(batch[b])
        np.testing.assert_allclose(res_batch.d[b], res_one.d, atol=1e-10)
        np.testing.assert_allclose(res_batch.r[b], res_one.r, atol=1e-10)


def test_3d_spherical_not_implemented():
    with pytest.raises(NotImplementedError):
        StructuredSphericalGrid(sphericalgrid=(4, 4, 4), spacings=(1, 1, 1),
                                 lat=np.zeros((4, 4, 4)), lon=np.zeros((4, 4, 4)))
