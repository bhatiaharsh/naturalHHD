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
poisson.py
    Poisson solver for the natural HHD, torch backend.

    Solves the Poisson equation via convolution with the free-space Green's
    function (see greens.py), using an FFT-based convolution on a regular
    grid (torch.fft.rfftn/irfftn, replacing scipy.signal.fftconvolve).

    NOTE: only the regular-grid path is implemented so far. Unstructured-
    point and spherical-grid solvers are planned but not yet ported from
    pynhhd-v1.1 -- see the project plan for delivery order.

    Unlike pynhhd-v1.1's PoissonSolver, precompute (building and FFT-ing the
    Green's-function kernel) is mandatory constructor-time work, not lazy --
    this deliberately avoids the numpy version's bug where a lazily-invoked
    prepare() referenced an undefined `num_cores` (pynhhd-v1.1/pynhhd/
    poisson.py:349). solve() raises a clear RuntimeError if called before
    the solver is ready, instead of trying to silently self-heal.

    This whole module is meant to run under torch.no_grad() -- see
    nhhd.py's decompose(). No differentiable path is supported or maintained.
'''

import logging
LOGGER = logging.getLogger(__name__)

import torch

from .utils.timer import Timer
from .utils.device import resolve_dtype, resolve_device
from .greens import GreensFunction


class PoissonSolver(object):

    def __init__(self, grid, spacings, dtype=None, device=None):
        '''
        grid:     tuple of grid dimensions, e.g. (Y, X) or (Z, Y, X)
        spacings: tuple of grid spacings matching `grid`, e.g. (dy, dx)
        '''
        self.gdims = tuple(int(g) for g in grid)
        self.gdx = tuple(float(d) for d in spacings)
        self.dim = len(self.gdims)

        if self.dim != len(self.gdx):
            raise ValueError('Dimensions of spacings should match that of the grid')
        if self.dim not in (1, 2, 3):
            raise ValueError('PoissonSolver works only for 1, 2, or 3 dimensions')

        self.dtype = resolve_dtype(dtype)
        self.device = resolve_device(device)

        self.ready = False
        LOGGER.info('PoissonSolver: %dD regular grid = %s with spacings = %s',
                     self.dim, self.gdims, self.gdx)

    # --------------------------------------------------------------------
    @torch.no_grad()
    def prepare(self):
        '''Build the Green's-function kernel and cache its rFFT.'''

        gtimer = Timer(device=self.device)
        LOGGER.debug('Initializing Poisson solver (FFT convolution)')

        gcomp = GreensFunction(self.dim, self.dtype, self.device)
        G = gcomp.create_grid(self.gdims, self.gdx)

        # linear-convolution output size along each axis: M + N - 1
        conv_shape = tuple(g + k - 1 for g, k in zip(self.gdims, G.shape))
        self._G_shape = tuple(G.shape)
        self._conv_shape = conv_shape
        self._fft_dims = tuple(range(-self.dim, 0))

        Gpad = torch.zeros(conv_shape, dtype=self.dtype, device=self.device)
        Gpad[tuple(slice(0, s) for s in G.shape)] = G
        self._G_fft = torch.fft.rfftn(Gpad, dim=self._fft_dims)

        # 'same'-mode centering offset, matching scipy.signal.fftconvolve's
        # `_centered` helper: start = (kernel_size - 1) // 2, independent of
        # the field size.
        self._same_start = tuple((k - 1) // 2 for k in self._G_shape)

        self.ready = True
        gtimer.end()
        LOGGER.debug('Poisson solver initialized! took %s', gtimer)

    # --------------------------------------------------------------------
    @torch.no_grad()
    def _solve_one(self, f):

        if not self.ready:
            raise RuntimeError('PoissonSolver.prepare() must be called before solve()')

        if tuple(f.shape[-self.dim:]) != self.gdims:
            raise ValueError('Shape of function ({}) should match shape of grid ({}) '
                              'in its trailing {} dims'.format(tuple(f.shape), self.gdims, self.dim))

        f = f.to(dtype=self.dtype, device=self.device)
        batch_shape = f.shape[:-self.dim]

        gtimer = Timer(device=self.device)

        fpad = torch.zeros(batch_shape + self._conv_shape, dtype=self.dtype, device=self.device)
        fpad[(Ellipsis,) + tuple(slice(0, s) for s in self.gdims)] = f

        F_fft = torch.fft.rfftn(fpad, dim=self._fft_dims)
        P_fft = F_fft * self._G_fft
        p_full = torch.fft.irfftn(P_fft, s=self._conv_shape, dim=self._fft_dims)

        out_slices = (Ellipsis,) + tuple(
            slice(s, s + g) for s, g in zip(self._same_start, self.gdims)
        )
        p = p_full[out_slices]

        cell_volume = 1.0
        for d in self.gdx:
            cell_volume *= d
        p = p * cell_volume

        gtimer.end()
        LOGGER.debug('Poisson solver finished! took %s', gtimer)
        return p

    # --------------------------------------------------------------------
    @torch.no_grad()
    def solve(self, func):
        '''
        func: a single (batched) tensor, or a list/tuple of 2 or 4 such
        tensors (2D: [div, curl]; 3D: [div, curlu, curlv, curlw]).
        '''
        if isinstance(func, (list, tuple)):
            nfuncs = len(func)
            if nfuncs not in (2, 4):
                raise ValueError('solve() expects a single field or a list of 2 or 4 fields, got {}'.format(nfuncs))
            return tuple(self._solve_one(f) for f in func)

        return self._solve_one(func)
