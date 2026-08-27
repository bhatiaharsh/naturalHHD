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

    Three modes:
      - regular grid ('grid'+'spacings'): FFT-based convolution
        (torch.fft.rfftn/irfftn, replacing scipy.signal.fftconvolve), the
        kernel's FFT cached once at construction.
      - unstructured points ('points'+'pvolumes'): dense N x N Green's
        matrix (torch.cdist) and a batched matmul solve. This is O(N^2)
        memory on ANY device (GPUs often have less memory than CPU RAM, so
        this is not merely a CPU-era limitation) -- `max_points` guards
        against silently attempting an allocation that will OOM.
      - spherical grid ('sphericalgrid'+'spacings'+'lat'+'lon'): the
        longitude-offset kernel tensor (ny, ny, nx) from greens.py, FFT'd
        once along the longitude axis at construction. Each solve() is then
        an FFT + a batched einsum contraction over source latitude rows +
        an inverse FFT -- replacing pynhhd-v1.1's O((ny*nx)^2) per-latitude
        joblib-multiprocessing loop with an O(ny^2 * nx log nx),
        GPU/batch-friendly contraction. See greens.py's
        create_sphericalGrid_kernel for the reformulation's derivation.

    Precompute (building/caching the Green's-function kernel or matrix) is
    mandatory constructor-time work, not lazy -- this deliberately avoids
    the numpy version's bug where a lazily-invoked prepare() referenced an
    undefined `num_cores` (pynhhd-v1.1/pynhhd/poisson.py:349). solve()
    raises a clear RuntimeError if called before the solver is ready,
    instead of trying to silently self-heal.

    This whole module is meant to run under torch.no_grad() -- see
    nhhd.py's decompose(). No differentiable path is supported or maintained.
'''

import math
import logging
LOGGER = logging.getLogger(__name__)

import torch

from .utils.timer import Timer
from .utils.device import resolve_dtype, resolve_device, as_tensor
from .greens import GreensFunction

DEFAULT_MAX_POINTS = 20000


class PoissonSolver(object):

    def __init__(self, grid=None, spacings=None, points=None, pvolumes=None,
                 sphericalgrid=None, lat=None, lon=None, r=None, tuning_param=0.0625,
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

        Spherical-grid mode:
            sphericalgrid: tuple of grid dimensions (ny, nx)
            spacings:      tuple of grid spacings in DEGREES (dlat, dlon)
            lat, lon:      1D or 2D (ny, nx) lat/lon arrays in degrees
            r:             planet radius in meters (default: Earth)
            tuning_param:  self-term integration window size as a fraction
                      of one grid cell (see greens.py's create_sphericalGrid_kernel)
        '''
        modes = [grid is not None, points is not None, sphericalgrid is not None]
        if sum(modes) != 1:
            raise SyntaxError('PoissonSolver needs exactly one of (grid, spacings), '
                               '(points, pvolumes), or (sphericalgrid, spacings, lat, lon)')

        self.dtype = resolve_dtype(dtype)
        self.device = resolve_device(device)
        self.ready = False

        if grid is not None:
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

        elif points is not None:
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

        else:
            if spacings is None or lat is None or lon is None:
                raise SyntaxError('PoissonSolver needs spacings, lat, and lon for a spherical grid')

            self.ptype = 'S'
            self.gdims = tuple(int(g) for g in sphericalgrid)
            self.gdx = tuple(float(d) for d in spacings)
            self.dim = len(self.gdims)
            self.r = float(r) if r is not None else 6371e3
            self.tuning_param = tuning_param

            if self.dim != 2:
                raise ValueError('PoissonSolver (spherical) currently only supports 2D grids')
            if self.dim != len(self.gdx):
                raise ValueError('Dimensions of spacings should match that of the grid')

            lat_t, _ = as_tensor(lat, self.dtype, self.device)
            lon_t, _ = as_tensor(lon, self.dtype, self.device)
            self.latv = lat_t[:, 0] if lat_t.dim() == 2 else lat_t
            self.lonv = lon_t[0, :] if lon_t.dim() == 2 else lon_t

            LOGGER.info('PoissonSolver: %dD spherical grid = %s with spacings = %s (degrees)',
                         self.dim, self.gdims, self.gdx)

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

        elif self.ptype == 'P':
            self._G = gcomp.create_points(self.points, self.pvolumes)

        else:  # self.ptype == 'S'
            kernel = gcomp.create_sphericalGrid_kernel(self.latv, self.lonv, r=self.r,
                                                         tuning_param=self.tuning_param)
            self._kernel_fft = torch.fft.rfft(kernel, dim=-1)  # (ny, ny, nfreq)

            rdlat = math.radians(self.gdx[0])
            rdlon = math.radians(self.gdx[1])
            rlat = torch.deg2rad(self.latv)
            # per-source-row cell area weight (dxy in pynhhd-v1.1's _solve_spherical)
            self._dxy = (self.r ** 2) * rdlat * rdlon * torch.cos(rlat)  # (ny,)

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

    @torch.no_grad()
    def _solve_one_spherical(self, f):
        '''
        Reproduces pynhhd-v1.1's roll-based sum exactly:
            p[j,i] = sum_k sum_c func[k,c] * G_j(k, (c-i) mod nx) * dxy[k]
                   = sum_k sum_d func[k,(i+d) mod nx] * K[j,k,d] * dxy[k]
        (substituting d = (c-i) mod nx), which is a circular CORRELATION of
        func[k,:] with K[j,k,:] -- NOT a circular convolution -- summed over
        source-latitude rows k. The two coincide only when K[j,k,:] happens
        to be symmetric under the circular index reversal d -> (nx-d) mod nx,
        which holds for a genuinely global (360-degree-spanning) longitude
        grid but not in general (e.g. a regional grid spanning <360 degrees
        in longitude) -- so this must be implemented as a correlation to
        match pynhhd-v1.1 for any grid, not just global ones. Via the
        correlation theorem, circular correlation is
        IFFT(FFT(func) * conj(FFT(K))), i.e. the same contraction as a
        convolution but with the kernel's FFT conjugated.
        '''

        if tuple(f.shape[-2:]) != self.gdims:
            raise ValueError('Shape of function ({}) should match shape of grid ({}) '
                              'in its trailing 2 dims'.format(tuple(f.shape), self.gdims))

        f = f.to(dtype=self.dtype, device=self.device)
        batch_shape = f.shape[:-2]

        weighted = f * self._dxy.unsqueeze(-1)  # (*batch, ny, nx); broadcasts regardless of batch rank

        F_fft = torch.fft.rfft(weighted, dim=-1)  # (*batch, ny, nfreq)
        nfreq = F_fft.shape[-1]

        # merge any leading batch dims into one (so this also works correctly
        # for a fully unbatched call, matching _solve_one_grid/_solve_one_points'
        # handling of batch_shape == ()), contract over source latitude k, then
        # restore the original batch shape
        F_fft_flat = F_fft.reshape(-1, self.gdims[0], nfreq)
        P_fft_flat = torch.einsum('jkf,bkf->bjf', self._kernel_fft.conj(), F_fft_flat)
        P_fft = P_fft_flat.reshape(*batch_shape, self.gdims[0], nfreq)

        p = torch.fft.irfft(P_fft, n=self.gdims[1], dim=-1)  # (*batch, ny, nx)
        return p

    def _solve_one(self, f):
        if not self.ready:
            raise RuntimeError('PoissonSolver.prepare() must be called before solve()')

        gtimer = Timer(device=self.device)
        if self.ptype == 'G':
            p = self._solve_one_grid(f)
        elif self.ptype == 'P':
            p = self._solve_one_points(f)
        else:
            p = self._solve_one_spherical(f)
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
