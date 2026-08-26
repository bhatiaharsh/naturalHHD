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

    NOTE: only the regular-grid kernel (create_grid) is implemented so far.
    The unstructured-point (create_points) and spherical-grid
    (create_sphericalGrid) Green's functions are planned but not yet ported
    from pynhhd-v1.1 -- see the project plan for delivery order.
'''

import math
import logging
LOGGER = logging.getLogger(__name__)

import torch

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
