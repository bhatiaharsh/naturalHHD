'''
Copyright (c) 2025, Harsh Bhatia (bhatia4@llnl.gov)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

'''
greens.py
    Free-space Green's function for the natural HHD, torch backend.

    NOTE: the spherical-grid Green's function (create_sphericalGrid) is
    planned but not yet ported from pynhhd-v1.1 -- see the project plan
    for delivery order.
'''

import math
import logging
LOGGER = logging.getLogger(__name__)

import torch
from scipy import integrate

from .utils.timer import Timer

_LOG2PI_INV = 0.5 / math.pi
_INV4PI = -0.25 / math.pi


class GreensFunction(object):

    def __init__(self, dim, dtype, device):
        self.dim = dim
        self.dtype = dtype
        self.device = device

    # compute the free-space Green's function value(s) for a tensor of
    # distances (elementwise). Mirrors pynhhd-v1.1/pynhhd/greens.py:compute().
    def compute(self, x):

        if self.dim == 1:
            return 0.5 * torch.abs(x)

        elif self.dim == 2:
            return _LOG2PI_INV * torch.log(x)

        elif self.dim == 3:
            return _INV4PI * torch.reciprocal(x)

        raise ValueError('GreensFunction only supports dim in {1, 2, 3}, got {}'.format(self.dim))

    # --------------------------------------------------------------------
    # build the full symmetric Green's-function kernel for a regular grid,
    # suitable for a 'same'-mode FFT convolution against a field of shape X.
    def create_grid(self, X, dx):

        LOGGER.info('Computing %dD Greens function on a regular grid', self.dim)
        gtimer = Timer(device=self.device)
        ltimer = Timer(device=self.device)

        dtype, device = self.dtype, self.device

        # ------------------------------------------------------------------
        # build a "half" radial-distance grid (one quadrant/octant), via
        # broadcasting rather than a python loop
        if self.dim == 1:
            n = X[0]
            dist = torch.arange(n, dtype=dtype, device=device) * dx[0]

        elif self.dim == 2:
            ny, nx = X
            dyv = torch.arange(ny, dtype=dtype, device=device) * dx[0]
            dxv = torch.arange(nx, dtype=dtype, device=device) * dx[1]
            yy, xx = torch.meshgrid(dyv, dxv, indexing='ij')
            dist = torch.hypot(yy, xx)

        elif self.dim == 3:
            nz, ny, nx = X
            dzv = torch.arange(nz, dtype=dtype, device=device) * dx[0]
            dyv = torch.arange(ny, dtype=dtype, device=device) * dx[1]
            dxv = torch.arange(nx, dtype=dtype, device=device) * dx[2]
            zz, yy, xx = torch.meshgrid(dzv, dyv, dxv, indexing='ij')
            dist = torch.hypot(torch.hypot(yy, xx), zz)

        else:
            raise ValueError('GreensFunction.create_grid only supports dim in {1, 2, 3}, got {}'.format(self.dim))

        # avoid log(0) / 1/0 at the self-distance cell -- same convention as
        # pynhhd-v1.1 (half the mean grid spacing, not zero)
        mean_dx = sum(dx) / len(dx)
        if self.dim == 1:
            dist[0] = 0.5 * mean_dx
        elif self.dim == 2:
            dist[0, 0] = 0.5 * mean_dx
        elif self.dim == 3:
            dist[0, 0, 0] = 0.5 * mean_dx

        ltimer.end()
        LOGGER.debug('creating half grid took %s', ltimer)

        # ------------------------------------------------------------------
        ltimer.start()
        half = self.compute(dist)
        ltimer.end()
        LOGGER.debug('creating green\'s function took %s', ltimer)

        # ------------------------------------------------------------------
        # mirror the half grid into the full symmetric kernel: full[i] =
        # half[|i|] along each axis, i.e. the kernel is even/symmetric since
        # the Green's function depends only on distance
        ltimer.start()

        full = half
        for axis in range(self.dim):
            mirrored = torch.flip(full, dims=[axis])
            # drop the duplicated center row/col/plane (index 0 along `axis`
            # in `full`, which is the boundary shared with the mirror)
            mirrored = mirrored.narrow(axis, 0, mirrored.shape[axis] - 1)
            full = torch.cat([mirrored, full], dim=axis)

        ltimer.end()
        gtimer.end()
        LOGGER.debug('creating full grid took %s', ltimer)
        LOGGER.info('Computing %dD Greens function took %s', self.dim, gtimer)

        return full

    # --------------------------------------------------------------------
    # build the full N x N Green's-function matrix for a set of unstructured
    # points (currently 2D only, matching UnstructuredGrid).
    def create_points(self, points, pvolumes):
        '''
        points:   (N, dim) tensor of point coordinates
        pvolumes: (N,) tensor of per-point cell areas (e.g. Voronoi-dual
                  areas), used only to derive a principled self-term below
                  -- NOT applied as a weight here (that happens in
                  PoissonSolver, matching pynhhd-v1.1's pre-scaling).

        Self-term: instead of pynhhd-v1.1's crude `+1e-10` epsilon on the
        diagonal (which, combined with float32, makes 1/eps or log(eps)
        blow up far more than it does at numpy's float64), each point's
        self-contribution is modeled as the analytic average of the
        free-space Green's function over a disk of area equal to that
        point's own cell (pvolumes[i]), rather than a point sample at a
        near-zero distance:
            mean_{disk radius R} log(r)   = log(R) - 1/2   (2D)
        which we fold back into `compute()` by using an *effective*
        self-distance d_eff = R * exp(-1/2), so that compute(d_eff) equals
        that analytic average exactly (for dim=3, the equivalent effective
        radius using the mean of 1/r over a ball is d_eff = (2/3) R, not
        currently used since unstructured 3D is unimplemented).
        '''
        if self.dim != 2:
            raise NotImplementedError('GreensFunction.create_points currently only supports dim=2')

        LOGGER.info('Computing %dD Greens function for %d unstructured points', self.dim, points.shape[0])
        gtimer = Timer(device=self.device)

        points = points.to(dtype=self.dtype, device=self.device)
        pvolumes = pvolumes.to(dtype=self.dtype, device=self.device)

        dist = torch.cdist(points, points, p=2)

        radii = torch.sqrt(pvolumes / math.pi)
        self_dist = radii * math.exp(-0.5)
        dist.diagonal().copy_(self_dist)  # cdist's true diagonal is 0

        G = self.compute(dist)

        gtimer.end()
        LOGGER.info('Computing %dD Greens function took %s', self.dim, gtimer)
        return G

    # --------------------------------------------------------------------
    # build the (ny, ny, nx) longitude-offset kernel tensor for a spherical
    # (lat/lon) grid: kernel[j, k, i] is the great-circle-distance Green's
    # function between a target point at (lat[j], lon[0]) and a source
    # point at (lat[k], lon[i]).
    #
    # This kernel is shift-invariant in longitude (only the offset
    # lon[i]-lon[0] matters, since great-circle distance depends only on
    # Δlongitude for two fixed latitudes) but NOT in latitude, hence the
    # explicit (j, k) pair rather than a single latitude-offset axis. This
    # is the key structural difference from pynhhd-v1.1's per-latitude,
    # per-target-point joblib loop (poisson.py:_solve_spherical): building
    # this tensor once lets every solve() call become a batched FFT
    # contraction (see poisson.py) instead of an O((ny*nx)^2) Python loop.
    #
    # Trade-off (documented, not hidden): O(ny^2 * nx) kernel memory,
    # precomputed once at construction and reused for every solve() call
    # and every batch element -- a net win under repeated/batched use of
    # the same grid geometry, which is this project's actual goal.
    def create_sphericalGrid_kernel(self, latv, lonv, r=6371e3, tuning_param=0.0625):
        '''
        latv, lonv: 1D tensors of per-row latitude / per-column longitude,
                    in DEGREES (matching pynhhd-v1.1's convention)
        r:          planet radius in meters (default: Earth)
        tuning_param: see _self_term() below

        Returns the (ny, ny, nx) kernel tensor.
        '''
        LOGGER.info('Computing spherical Greens function kernel (%d x %d x %d)',
                     latv.shape[0], latv.shape[0], lonv.shape[0])
        gtimer = Timer(device=self.device)

        ny, nx = latv.shape[0], lonv.shape[0]
        rlat = torch.deg2rad(latv.to(dtype=self.dtype, device=self.device))
        rlon = torch.deg2rad(lonv.to(dtype=self.dtype, device=self.device))

        delta_lon = rlon - rlon[0]                      # (nx,)
        cos_d, sin_d = torch.cos(delta_lon), torch.sin(delta_lon)

        coslat, sinlat = torch.cos(rlat), torch.sin(rlat)  # (ny,)

        coslat_j = coslat.view(ny, 1, 1)
        sinlat_j = sinlat.view(ny, 1, 1)
        coslat_k = coslat.view(1, ny, 1)
        sinlat_k = sinlat.view(1, ny, 1)
        cosD = cos_d.view(1, 1, nx)
        sinD = sin_d.view(1, 1, nx)

        num = torch.sqrt((coslat_k * sinD) ** 2 + (coslat_j * sinlat_k - sinlat_j * coslat_k * cosD) ** 2)
        den = sinlat_j * sinlat_k + coslat_j * coslat_k * cosD
        gamma = torch.atan2(num, den)

        kernel = -torch.log(torch.sin(0.5 * gamma)) / (2.0 * math.pi)

        # self-term: the (j, j, 0) entries are singular (gamma == 0). Computed
        # via a numeric double integral over the singularity's local cell --
        # O(ny) scipy calls at construction time (not O(ny*nx) per solve, as
        # in pynhhd-v1.1).
        rdlat = math.radians(float(latv[1] - latv[0])) if ny > 1 else 0.0
        rdlon = math.radians(float(lonv[1] - lonv[0])) if nx > 1 else 0.0

        self_vals = torch.empty(ny, dtype=self.dtype, device=self.device)
        for j in range(ny):
            self_vals[j] = self._self_term(float(rlat[j]), rdlat, rdlon, tuning_param)

        j_idx = torch.arange(ny, device=self.device)
        kernel[j_idx, j_idx, 0] = self_vals

        gtimer.end()
        LOGGER.info('Computing spherical Greens function kernel took %s', gtimer)
        return kernel

    @staticmethod
    def _self_term(rlat_j, rdlat, rdlon, tuning_param):
        '''
        Numeric double integral of the (singular) great-circle Green's
        function over the local cell around a point at latitude rlat_j,
        matching pynhhd-v1.1/pynhhd/poisson.py:_solve_spherical's
        `_solve_task` self-term treatment exactly (same integrand, same
        integration bounds convention).
        '''
        def integrand(x, y):
            return -math.log(math.sin(math.sqrt(x * x + y * y) * 0.5)) / (2.0 * math.pi)

        da1 = tuning_param * rdlat
        da2 = tuning_param * rdlon * math.cos(rlat_j)
        if da1 <= 0 or da2 <= 0:
            return 0.0

        val, _ = integrate.dblquad(integrand, 0, da1, 0, da2)
        return val / (da1 * da2)
