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
spherical.py
    Structured lat/lon (spherical) grid support for the natural HHD, torch
    backend. 2D only -- pynhhd-v1.1's dim==3 branch was an explicit,
    unimplemented stub that silently fell back to the (buggy) flat/
    Cartesian 3D divergence formula; it is deliberately not ported here
    (see docs/CODE_REVIEW_REPORT.md sec 2.1 and structured.py).

    divcurl/gradient/rotated_gradient faithfully port pynhhd-v1.1's
    spherical-metric finite-difference formulas (pynhhd-v1.1/pynhhd/
    spherical.py's `divcurl`/`gradient`/`rotated_gradient`, the "USED AT
    THE MOMENT" glob=True path -- divcurl2/divcurl3/wrap_xesmf were
    explicitly marked unused there and are not ported), including the
    longitude-seam periodic-wraparound patch (torch.gradient has no native
    periodic-boundary mode, so the seam columns are recomputed from an
    explicit 3-column wrapped window, matching the numpy approach).

    Known-bug note: pynhhd-v1.1's gradient() hardcoded `verbose = True`
    internally, ignoring the caller's argument and spamming stdout; this
    port routes everything through LOGGER instead (see utils/logger.py).

    This whole module is meant to run under torch.no_grad() -- see
    nhhd.py's decompose(). No differentiable path is supported or maintained.
'''

import math
import logging
LOGGER = logging.getLogger(__name__)

import torch

from .utils.timer import Timer
from .utils.device import resolve_dtype, resolve_device, as_tensor

EARTH_RADIUS_M = 6371e3


class StructuredSphericalGrid(object):

    def __init__(self, sphericalgrid, spacings, lat, lon, r=EARTH_RADIUS_M, dtype=None, device=None):
        '''
        sphericalgrid: tuple of grid dimensions (ny, nx)
        spacings:      tuple of grid spacings in DEGREES (dlat, dlon)
        lat, lon:      1D tensors/arrays of per-row latitude / per-column
                        longitude in degrees, OR 2D (ny, nx) meshgrid arrays
                        (reduced to 1D internally, assuming -- as
                        pynhhd-v1.1 does -- a regular/separable grid where
                        latitude is constant along each row and longitude
                        constant along each column)
        r:             planet radius in meters (default: Earth)
        '''
        self.dtype = resolve_dtype(dtype)
        self.device = resolve_device(device)

        self.dims = tuple(int(g) for g in sphericalgrid)
        self.dx = tuple(float(d) for d in spacings)
        self.dim = len(self.dims)
        self.r = float(r)

        if self.dim != 2:
            raise NotImplementedError(
                'StructuredSphericalGrid currently only supports 2D grids '
                '(pynhhd-v1.1\'s 3D path was an unimplemented stub that silently reused '
                'the flat/Cartesian formula -- see docs/CODE_REVIEW_REPORT.md)')
        if self.dim != len(self.dx):
            raise ValueError('Dimensions of spacings should match that of the grid')

        lat_t, _ = as_tensor(lat, self.dtype, self.device)
        lon_t, _ = as_tensor(lon, self.dtype, self.device)
        self.latv = lat_t[:, 0] if lat_t.dim() == 2 else lat_t
        self.lonv = lon_t[0, :] if lon_t.dim() == 2 else lon_t

        if self.latv.shape[0] != self.dims[0] or self.lonv.shape[0] != self.dims[1]:
            raise ValueError('lat/lon shapes {}/{} do not match sphericalgrid dims {}'.format(
                tuple(self.latv.shape), tuple(self.lonv.shape), self.dims))

        self._rlat = torch.deg2rad(self.latv)               # (ny,)
        self._coslat = torch.cos(self._rlat)                # (ny,)
        self._rdy = math.radians(self.dx[0])
        self._rdx = math.radians(self.dx[1])

        LOGGER.info('Initialized 2D spherical grid %s with spacings %s (degrees)', self.dims, self.dx)

    # --------------------------------------------------------------------
    def _check_vfield(self, vfield):
        if vfield.shape[-1] != self.dim or tuple(vfield.shape[1:3]) != self.dims:
            raise ValueError('Dimensions of vector field {} should match that of the grid {}'.format(
                tuple(vfield.shape), self.dims))

    def _check_sfield(self, sfield):
        if tuple(sfield.shape[1:3]) != self.dims:
            raise ValueError('Dimensions of scalar field {} should match that of the grid {}'.format(
                tuple(sfield.shape), self.dims))

    # --------------------------------------------------------------------
    def _lat_partial(self, f):
        '''(1/r) df/dphi -- plain (unweighted) latitude partial derivative.'''
        d = torch.gradient(f, spacing=self._rdy, dim=1)[0]
        return d / self.r

    def _lat_flux_partial(self, f):
        '''(1/(r cos(phi))) d(f cos(phi))/dphi -- the flux-form meridional
        derivative used by divcurl (matches pynhhd-v1.1's dudy/dvdy).'''
        weighted = f * self._coslat.view(1, -1, 1)
        d = torch.gradient(weighted, spacing=self._rdy, dim=1)[0]
        return d / (self.r * self._coslat.view(1, -1, 1))

    def _lon_partial(self, f):
        '''(1/(r cos(phi))) df/dlambda, with an explicit periodic-wraparound
        patch at the longitude seam (torch.gradient has no native periodic
        mode).'''
        d = torch.gradient(f, spacing=self._rdx, dim=2)[0]
        d = d / (self.r * self._coslat.view(1, -1, 1))

        left3 = torch.cat([f[:, :, -1:], f[:, :, :2]], dim=2)
        d_left = torch.gradient(left3, spacing=self._rdx, dim=2)[0][:, :, 1:2]
        d_left = d_left / (self.r * self._coslat.view(1, -1, 1))

        right3 = torch.cat([f[:, :, -2:], f[:, :, :1]], dim=2)
        d_right = torch.gradient(right3, spacing=self._rdx, dim=2)[0][:, :, 1:2]
        d_right = d_right / (self.r * self._coslat.view(1, -1, 1))

        return torch.cat([d_left, d[:, :, 1:-1], d_right], dim=2)

    # --------------------------------------------------------------------
    @torch.no_grad()
    def divcurl(self, vfield):
        '''vfield: (B, ny, nx, 2) -> (div, curl), each (B, ny, nx)'''

        self._check_vfield(vfield)
        LOGGER.debug('Computing divcurl')
        mtimer = Timer(device=self.device)

        u, v = vfield[..., 0], vfield[..., 1]

        dudy = self._lat_flux_partial(u)
        dudx = self._lon_partial(u)
        dvdy = self._lat_flux_partial(v)
        dvdx = self._lon_partial(v)

        div = dudx + dvdy
        curl = dvdx - dudy  # note the sign convention (matches pynhhd-v1.1)

        mtimer.end()
        LOGGER.debug('Computing divcurl done! took %s', mtimer)
        return div, curl

    # --------------------------------------------------------------------
    @torch.no_grad()
    def gradient(self, sfield):
        '''sfield: (B, ny, nx) -> (B, ny, nx, 2)'''

        self._check_sfield(sfield)
        LOGGER.debug('Computing gradient')
        mtimer = Timer(device=self.device)

        ddy = self._lat_partial(sfield)
        ddx = self._lon_partial(sfield)
        grad = torch.stack((ddx, ddy), dim=-1)

        mtimer.end()
        LOGGER.debug('Computing gradient done! took %s', mtimer)
        return grad

    @torch.no_grad()
    def rotated_gradient(self, sfield):
        '''sfield: (B, ny, nx) -> (B, ny, nx, 2), the pi/2-rotated gradient'''

        LOGGER.debug('Computing rotated gradient')
        mtimer = Timer(device=self.device)

        g = self.gradient(sfield)
        ddx, ddy = g[..., 0], g[..., 1]
        grad = torch.stack((-ddy, ddx), dim=-1)

        mtimer.end()
        LOGGER.debug('Computing rotated gradient done! took %s', mtimer)
        return grad
