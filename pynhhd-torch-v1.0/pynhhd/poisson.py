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

    Two modes:
      - regular grid ('grid'+'spacings'): FFT-based convolution
        (torch.fft.rfftn/irfftn, replacing scipy.signal.fftconvolve), the
        kernel's FFT cached once at construction.
      - unstructured points ('points'+'pvolumes'): dense N x N Green's
        matrix (torch.cdist) and a batched matmul solve. This is O(N^2)
        memory on ANY device (GPUs often have less memory than CPU RAM, so
        this is not merely a CPU-era limitation) -- `max_points` guards
        against silently attempting an allocation that will OOM.

    NOTE: the spherical-grid solver is planned but not yet ported from
    pynhhd-v1.1 -- see the project plan for delivery order.

    Precompute (building/caching the Green's-function kernel or matrix) is
    mandatory constructor-time work, not lazy -- this deliberately avoids
    the numpy version's bug where a lazily-invoked prepare() referenced an
    undefined `num_cores` (pynhhd-v1.1/pynhhd/poisson.py:349). solve()
    raises a clear RuntimeError if called before the solver is ready,
    instead of trying to silently self-heal.

    This whole module is meant to run under torch.no_grad() -- see
    nhhd.py's decompose(). No differentiable path is supported or maintained.
'''

import logging
LOGGER = logging.getLogger(__name__)

import torch

from .utils.timer import Timer
from .utils.device import resolve_dtype, resolve_device, as_tensor
from .greens import GreensFunction

DEFAULT_MAX_POINTS = 20000


class PoissonSolver(object):

    def __init__(self, grid=None, spacings=None, points=None, pvolumes=None,
                 dtype=None, device=None, max_points=DEFAULT_MAX_POINTS):
        '''
        Regular grid mode:
            grid:     tuple of grid dimensions, e.g. (Y, X) or (Z, Y, X)
            spacings: tuple of grid spacings matching `grid`, e.g. (dy, dx)

        Unstructured-points mode:
            points:   (N, dim) tensor/array of point coordinates
            pvolumes: (N,) tensor/array of per-point cell volumes/areas
                      (e.g. from UnstructuredGrid's Voronoi-dual areas)
            max_points: refuse to build the dense N x N Green's matrix
                      above this many points (O(N^2) memory) rather than
                      risk silently exhausting device memory.
        '''
        is_grid = grid is not None
        is_points = points is not None

        if is_grid == is_points:
            raise SyntaxError('PoissonSolver needs exactly one of (grid, spacings) or (points, pvolumes)')

        self.dtype = resolve_dtype(dtype)
        self.device = resolve_device(device)
        self.ready = False

        if is_grid:
            if spacings is None:
                raise SyntaxError('PoissonSolver needs spacings for a regular grid')

            self.ptype = 'G'
            self.gdims = tuple(int(g) for g in grid)
            self.gdx = tuple(float(d) for d in spacings)
            self.dim = len(self.gdims)

            if self.dim != len(self.gdx):
                raise ValueError('Dimensions of spacings should match that of the grid')
            if self.dim not in (1, 2, 3):
                raise ValueError('PoissonSolver works only for 1, 2, or 3 dimensions')

            LOGGER.info('PoissonSolver: %dD regular grid = %s with spacings = %s',
                         self.dim, self.gdims, self.gdx)

        else:
            if pvolumes is None:
                raise SyntaxError('PoissonSolver needs pvolumes for unstructured points')

            self.ptype = 'P'
            self.points, _ = as_tensor(points, self.dtype, self.device)
            self.pvolumes, _ = as_tensor(pvolumes, self.dtype, self.device)
            self.dim = self.points.shape[-1]
            self.max_points = max_points

            if self.pvolumes.shape[0] != self.points.shape[0]:
                raise ValueError('Number of pvolumes should match that of the points')
            if self.points.shape[0] > self.max_points:
                raise ValueError(
                    'PoissonSolver (unstructured points): {} points requested a dense {}x{} '
                    'Green\'s matrix ({:.2f} GB at this dtype), above max_points={}. This is an '
                    'O(N^2) memory algorithm on ANY device -- pass a larger max_points explicitly '
                    'if you have the memory, or reduce the point count.'.format(
                        self.points.shape[0], self.points.shape[0], self.points.shape[0],
                        self.points.shape[0] ** 2 * self.dtype.itemsize / 1e9, self.max_points))

            LOGGER.info('PoissonSolver: %dD unstructured points = %s', self.dim, tuple(self.points.shape))

    # --------------------------------------------------------------------
    @torch.no_grad()
    def prepare(self):
        '''Build and cache the Green's-function kernel/matrix.'''

        gtimer = Timer(device=self.device)
        LOGGER.debug('Initializing Poisson solver')

        gcomp = GreensFunction(self.dim, self.dtype, self.device)

        if self.ptype == 'G':
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
            # `_centered` helper: start = (kernel_size - 1) // 2, independent
            # of the field size.
            self._same_start = tuple((k - 1) // 2 for k in self._G_shape)

        else:  # self.ptype == 'P'
            self._G = gcomp.create_points(self.points, self.pvolumes)

        self.ready = True
        gtimer.end()
        LOGGER.debug('Poisson solver initialized! took %s', gtimer)

    # --------------------------------------------------------------------
    @torch.no_grad()
    def _solve_one_grid(self, f):

        if tuple(f.shape[-self.dim:]) != self.gdims:
            raise ValueError('Shape of function ({}) should match shape of grid ({}) '
                              'in its trailing {} dims'.format(tuple(f.shape), self.gdims, self.dim))

        f = f.to(dtype=self.dtype, device=self.device)
        batch_shape = f.shape[:-self.dim]

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
        return p

    @torch.no_grad()
    def _solve_one_points(self, f):

        n = self.points.shape[0]
        if f.shape[-1] != n:
            raise ValueError('Shape of function ({}) should have {} in its trailing dim '
                              '(number of points)'.format(tuple(f.shape), n))

        f = f.to(dtype=self.dtype, device=self.device)
        weighted = f * self.pvolumes  # broadcasts over any leading batch dim
        # p = G @ weighted, batched over any leading dims: (B, N) x (N, N)^T -> (B, N)
        return torch.matmul(weighted, self._G.T)

    def _solve_one(self, f):
        if not self.ready:
            raise RuntimeError('PoissonSolver.prepare() must be called before solve()')

        gtimer = Timer(device=self.device)
        p = self._solve_one_grid(f) if self.ptype == 'G' else self._solve_one_points(f)
        gtimer.end()
        LOGGER.debug('Poisson solver finished! took %s', gtimer)
        return p

    # --------------------------------------------------------------------
    @torch.no_grad()
    def solve(self, func):
        '''
        func: a single (batched) tensor, or a list/tuple of 2 or 4 such
        tensors (2D: [div, curl]; 3D: [div, curlu, curlv, curlw] -- 3D is
        grid-only, unstructured points are currently 2D-only).
        '''
        if isinstance(func, (list, tuple)):
            nfuncs = len(func)
            if nfuncs not in (2, 4):
                raise ValueError('solve() expects a single field or a list of 2 or 4 fields, got {}'.format(nfuncs))
            return tuple(self._solve_one(f) for f in func)

        return self._solve_one(func)
