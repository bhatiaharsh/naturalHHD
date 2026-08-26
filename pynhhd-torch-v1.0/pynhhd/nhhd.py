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
nhhd.py
    Top-level orchestrator for the natural HHD, torch backend.

    NOTE: only the regular-grid ('grid') path is implemented so far.
    Unstructured-point ('points') and spherical-grid ('sphericalgrid')
    inputs are planned but not yet ported from pynhhd-v1.1 -- see the
    project plan for delivery order.

    No differentiable path: every public entry point runs under
    torch.no_grad() (the user's decomposition never needs gradients through
    this library), so autograd bookkeeping is skipped entirely rather than
    merely unused.

    decompose() returns an explicit DecompositionResult instead of only
    mutating self.div/self.d/self.r/self.h in place (as pynhhd-v1.1 does) --
    with a batch of B fields as a first-class input, "which result does
    self.h refer to" would otherwise be ambiguous. The same attributes are
    still set on `self` for the unbatched, single-field case as a familiar
    convenience, but the returned object is the documented source of truth.
'''

import logging
from collections import namedtuple

LOGGER = logging.getLogger(__name__)

import torch

from .poisson import PoissonSolver
from .structured import StructuredGrid
from .utils.device import resolve_dtype, resolve_device, as_tensor, to_output, ensure_batched, restore_batch


DecompositionResult = namedtuple(
    'DecompositionResult', ['div', 'curl', 'd', 'r', 'h', 'potentials']
)


class nHHD(object):

    def __init__(self, grid=None, spacings=None, dtype=None, device=None):
        '''
        Currently supported: regular grids only.
            grid:     tuple of grid dimensions (Y, X) or (Z, Y, X)
            spacings: tuple of grid spacings (dy, dx) or (dz, dy, dx)
        '''

        if grid is None:
            raise SyntaxError(
                'nHHD Solver needs the shape of a regular grid (points/sphericalgrid '
                'inputs are not yet implemented in pynhhd)')
        if spacings is None:
            raise SyntaxError('nHHD needs spacings for regular grid')

        self.dtype = resolve_dtype(dtype)
        self.device = resolve_device(device)

        self.mesh = StructuredGrid(grid=grid, spacings=spacings, dtype=self.dtype, device=self.device)
        self.psolver = PoissonSolver(grid=self.mesh.dims, spacings=self.mesh.dx,
                                      dtype=self.dtype, device=self.device)

        self.dim = self.mesh.dim
        if self.dim != 2 and self.dim != 3:
            raise ValueError('nHHD Solver works only for 2 and 3 dimensions')

        self.psolver.prepare()

    # --------------------------------------------------------------------
    @torch.no_grad()
    def decompose(self, vfield, div=None, curlw=None):
        '''
        vfield: numpy array or torch tensor, shape (*grid_dims, dim) for a
        single field, or (B, *grid_dims, dim) for a batch of B fields.

        div, curlw: optionally supply the divergence/2D-curl directly
        instead of having them computed from vfield (e.g. if your data
        isn't on a Cartesian grid -- use the model-native divergence/curl).
        3D does not support this override (matching pynhhd-v1.1).

        Returns a DecompositionResult(div, curl, d, r, h, potentials). For
        an unbatched input, `self.div`/`self.d`/`self.r`/`self.h` are also
        set as a convenience, mirroring pynhhd-v1.1's API.
        '''

        vfield_t, was_numpy = as_tensor(vfield, self.dtype, self.device)
        vfield_t, was_batched = ensure_batched(vfield_t, unbatched_ndim=self.dim + 1)

        if vfield_t.shape[-1] != self.dim:
            raise ValueError('nHHD.decompose requires a {}-dimensional vector field'.format(self.dim))

        LOGGER.info('Decomposing a vector field: shape = %s', tuple(vfield_t.shape))

        if self.dim == 2:

            if div is None and curlw is None:
                div_t, curl_t = self.mesh.divcurl(vfield_t)
            else:
                div_t, was_numpy_div = as_tensor(div, self.dtype, self.device)
                curl_t, _ = as_tensor(curlw, self.dtype, self.device)
                div_t, _ = ensure_batched(div_t, unbatched_ndim=self.dim)
                curl_t, _ = ensure_batched(curl_t, unbatched_ndim=self.dim)

            nD, nRu = self.psolver.solve([div_t, curl_t])

            d = self.mesh.gradient(nD)
            r = self.mesh.rotated_gradient(nRu)
            potentials = (nD, nRu)

        else:  # self.dim == 3

            div_t, curlu_t, curlv_t, curlw_t = self.mesh.divcurl(vfield_t)

            nD, nRu, nRv, nRw = self.psolver.solve([div_t, curlu_t, curlv_t, curlw_t])
            # sign convention from the vector-Laplacian gauge condition
            nRu, nRv, nRw = -nRu, -nRv, -nRw

            d = self.mesh.gradient(nD)
            r = self.mesh.curl3D(torch.stack((nRu, nRv, nRw), dim=-1))
            potentials = (nD, nRu, nRv, nRw)
            curl_t = (curlu_t, curlv_t, curlw_t)

        h = vfield_t - d - r

        div_out = to_output(restore_batch(div_t, was_batched), was_numpy)
        curl_out = (to_output(restore_batch(curl_t, was_batched), was_numpy) if self.dim == 2
                    else tuple(to_output(restore_batch(c, was_batched), was_numpy) for c in curl_t))
        d_out = to_output(restore_batch(d, was_batched), was_numpy)
        r_out = to_output(restore_batch(r, was_batched), was_numpy)
        h_out = to_output(restore_batch(h, was_batched), was_numpy)
        pot_out = tuple(to_output(restore_batch(p, was_batched), was_numpy) for p in potentials)

        result = DecompositionResult(div=div_out, curl=curl_out, d=d_out, r=r_out, h=h_out, potentials=pot_out)

        if not was_batched:
            # convenience shim for single-field callers migrating from pynhhd-v1.1;
            # the returned DecompositionResult is the documented source of truth
            self.div, self.curl, self.d, self.r, self.h = div_out, curl_out, d_out, r_out, h_out

        return result
