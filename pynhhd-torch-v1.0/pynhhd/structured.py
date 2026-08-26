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
structured.py
    Regular/rectangular grid support for the natural HHD, torch backend.

    All methods operate on a mandatory leading batch axis (shape
    (B, *grid_dims, C) for vector fields, (B, *grid_dims) for scalar
    fields) -- batching is normalized once in nhhd.py's decompose(), not
    per grid method.

    Finite differences use torch.gradient (replacing numpy.gradient):
    2nd-order central differences in the interior, 1st-order one-sided at
    the grid boundary -- matching numpy's default edge_order=1 behavior.

    3D divergence bug fix: pynhhd-v1.1/pynhhd/structured.py:90 computed
    div = du/dx + dv/dy + dv/dz (reusing dv/dz where dw/dz was intended --
    dw/dz was computed there but never referenced). This is fixed here as
    div = du/dx + dv/dy + dw/dz from the start; see
    docs/CODE_REVIEW_REPORT.md section 2.1 and
    tests/test_structured_analytical.py / tests/test_structured_vs_cpp.py
    for the regression coverage that would catch this if reintroduced.

    This whole module is meant to run under torch.no_grad() -- see
    nhhd.py's decompose(). No differentiable path is supported or maintained.
'''

import logging
LOGGER = logging.getLogger(__name__)

import torch

from .utils.timer import Timer
from .utils.device import resolve_dtype, resolve_device


class StructuredGrid(object):

    def __init__(self, grid, spacings, dtype=None, device=None):
        '''
        grid:     tuple of grid dimensions (Y, X) or (Z, Y, X)
        spacings: tuple of grid spacings (dy, dx) or (dz, dy, dx)
        '''
        self.dims = tuple(int(g) for g in grid)
        self.dx = tuple(float(d) for d in spacings)
        self.dim = len(self.dims)

        if self.dim != 2 and self.dim != 3:
            raise ValueError('StructuredGrid works for 2D and 3D only')
        if self.dim != len(self.dx):
            raise ValueError('Dimensions of spacings should match that of the grid')

        self.dtype = resolve_dtype(dtype)
        self.device = resolve_device(device)

        # batch is always axis 0; grid axes follow it
        self._grid_axes = tuple(range(1, 1 + self.dim))

        LOGGER.info('Initialized %dD structured grid', self.dim)

    # --------------------------------------------------------------------
    def _check_vfield(self, vfield):
        if vfield.shape[-1] != self.dim or tuple(vfield.shape[1:1 + self.dim]) != self.dims:
            raise ValueError(
                'Dimensions of vector field {} should match that of the grid {} '
                '(expected shape (B, {}, {}))'.format(
                    tuple(vfield.shape), self.dims, ', '.join(map(str, self.dims)), self.dim))

    def _check_sfield(self, sfield):
        if tuple(sfield.shape[1:]) != self.dims:
            raise ValueError('Dimensions of scalar field {} should match that of the grid {}'.format(
                tuple(sfield.shape), self.dims))

    # --------------------------------------------------------------------
    @torch.no_grad()
    def divcurl(self, vfield):
        '''vfield: (B, *dims, dim) -> 2D: (div, curl); 3D: (div, curlu, curlv, curlw), each (B, *dims)'''

        self._check_vfield(vfield)

        LOGGER.debug('Computing divcurl')
        mtimer = Timer(device=self.device)

        u = vfield[..., 0]
        v = vfield[..., 1]

        if self.dim == 2:

            dudy, dudx = torch.gradient(u, spacing=self.dx, dim=self._grid_axes)
            dvdy, dvdx = torch.gradient(v, spacing=self.dx, dim=self._grid_axes)

            div = dudx + dvdy
            curl = dvdx - dudy

            mtimer.end()
            LOGGER.debug('Computing divcurl done! took %s', mtimer)
            return div, curl

        else:  # self.dim == 3

            w = vfield[..., 2]

            dudz, dudy, dudx = torch.gradient(u, spacing=self.dx, dim=self._grid_axes)
            dvdz, dvdy, dvdx = torch.gradient(v, spacing=self.dx, dim=self._grid_axes)
            dwdz, dwdy, dwdx = torch.gradient(w, spacing=self.dx, dim=self._grid_axes)

            # div = du/dx + dv/dy + dw/dz (the numpy version's `dvdz` here
            # instead of `dwdz` was bug #1 -- see module docstring)
            div = dudx + dvdy + dwdz

            curlu = dwdy - dvdz
            curlv = dudz - dwdx
            curlw = dvdx - dudy

            mtimer.end()
            LOGGER.debug('Computing divcurl done! took %s', mtimer)
            return div, curlu, curlv, curlw

    # --------------------------------------------------------------------
    @torch.no_grad()
    def curl3D(self, vfield):
        '''vfield: (B, *dims, 3) -> curl, (B, *dims, 3)'''

        self._check_vfield(vfield)
        if self.dim != 3:
            raise ValueError('curl3D works only for 3D')

        LOGGER.debug('Computing curl3D')
        mtimer = Timer(device=self.device)

        u, v, w = vfield[..., 0], vfield[..., 1], vfield[..., 2]

        dudz, dudy, dudx = torch.gradient(u, spacing=self.dx, dim=self._grid_axes)
        dvdz, dvdy, dvdx = torch.gradient(v, spacing=self.dx, dim=self._grid_axes)
        dwdz, dwdy, dwdx = torch.gradient(w, spacing=self.dx, dim=self._grid_axes)

        cu = dwdy - dvdz
        cv = dudz - dwdx
        cw = dvdx - dudy

        curl = torch.stack((cu, cv, cw), dim=-1)

        mtimer.end()
        LOGGER.debug('Computing curl3D done! took %s', mtimer)
        return curl

    # --------------------------------------------------------------------
    @torch.no_grad()
    def rotated_gradient(self, sfield):
        '''sfield: (B, *dims) -> (B, *dims, 2), the pi/2-rotated gradient (2D only)'''

        self._check_sfield(sfield)
        if self.dim != 2:
            raise ValueError('rotated_gradient works only for 2D')

        LOGGER.debug('Computing rotated gradient')
        mtimer = Timer(device=self.device)

        ddy, ddx = torch.gradient(sfield, spacing=self.dx, dim=self._grid_axes)
        ddy = -ddy

        grad = torch.stack((ddy, ddx), dim=-1)

        mtimer.end()
        LOGGER.debug('Computing rotated gradient done! took %s', mtimer)
        return grad

    # --------------------------------------------------------------------
    @torch.no_grad()
    def gradient(self, sfield):
        '''sfield: (B, *dims) -> (B, *dims, dim)'''

        self._check_sfield(sfield)

        LOGGER.debug('Computing gradient')
        mtimer = Timer(device=self.device)

        if self.dim == 2:
            ddy, ddx = torch.gradient(sfield, spacing=self.dx, dim=self._grid_axes)
            grad = torch.stack((ddx, ddy), dim=-1)

        else:  # self.dim == 3
            ddz, ddy, ddx = torch.gradient(sfield, spacing=self.dx, dim=self._grid_axes)
            grad = torch.stack((ddx, ddy, ddz), dim=-1)

        mtimer.end()
        LOGGER.debug('Computing gradient done! took %s', mtimer)
        return grad
