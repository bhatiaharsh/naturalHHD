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
unstructured.py
    Unstructured (triangular-mesh) grid support for the natural HHD, torch
    backend. Currently 2D (triangulations) only -- 3D (tetrahedral) support
    raises NotImplementedError, matching pynhhd-v1.1's actual (never
    finished) state, rather than silently falling into a broken code path
    (pynhhd-v1.1/pynhhd/unstructured.py:96 referenced an undefined
    `self.sdim`; see docs/CODE_REVIEW_REPORT.md).

    Mesh topology (Delaunay triangulation via scipy, adjacency/boundary-
    edge extraction) is one-time, per-mesh preprocessing done in
    numpy/Python at construction -- dict/list-based edge matching is a
    natural fit there and this isn't a hot path. Everything that runs per
    decompose() call (per-triangle divergence/curl, gradient, and the
    Meyer's-algorithm Voronoi-dual point-area computation) is vectorized
    over all triangles at once via gather (points[simplices]) + scatter
    (index_add_) instead of pynhhd-v1.1's Python for-loops.

    Boundary-curl bug fix: pynhhd-v1.1/pynhhd/unstructured.py:328-329
    accumulated a stale `tn` left over from the unrelated main per-triangle
    loop into the boundary-edge curl correction, instead of the freshly
    computed `dt` for that edge (see docs/CODE_REVIEW_REPORT.md sec 2.2).
    Here, the boundary correction reuses the *already-correct* per-
    (face,corner) tangential/normal flux tensors computed for the interior
    pass (there is no second, separately-computed `dt`/`tn` pair and no
    shared mutable loop variable at all) -- this bug class is eliminated
    structurally, not just patched at one call site.

    This whole module is meant to run under torch.no_grad() -- see
    nhhd.py's decompose(). No differentiable path is supported or maintained.
'''

import logging
LOGGER = logging.getLogger(__name__)

import numpy as np
from scipy import spatial
import torch

from .utils.timer import Timer
from .utils.device import resolve_dtype, resolve_device, as_tensor


class UnstructuredGrid(object):

    def __init__(self, vertices, simplices=None, dtype=None, device=None):
        '''
        vertices:  (N, 2) array/tensor of vertex coordinates
        simplices: optional (T, 3) array/tensor of triangle vertex indices;
                   computed via scipy.spatial.Delaunay if not given
        '''
        self.dtype = resolve_dtype(dtype)
        self.device = resolve_device(device)

        self.vertices, _ = as_tensor(vertices, self.dtype, self.device)
        self.dim = self.vertices.shape[1]
        self.nvertices = self.vertices.shape[0]

        if self.dim != 2:
            raise NotImplementedError('UnstructuredGrid currently only supports 2D (triangulated) meshes')

        LOGGER.info('Initializing %dD mesh with %d vertices', self.dim, self.nvertices)
        mtimer = Timer(device=self.device)

        vertices_np = self.vertices.detach().cpu().numpy()

        if simplices is not None:
            simplices_np = np.asarray(simplices)
        else:
            LOGGER.debug('creating Delaunay mesh')
            simplices_np = spatial.Delaunay(vertices_np).simplices

        self.nsimplices = simplices_np.shape[0]
        if self.dim != simplices_np.shape[1] - 1:
            raise ValueError('Dimension mismatch: vertices imply dim={}, simplices imply dim={}'.format(
                self.dim, simplices_np.shape[1] - 1))

        self.simplices = torch.as_tensor(simplices_np, dtype=torch.long, device=self.device)

        # one-time topology preprocessing (numpy/Python -- see module docstring)
        bfaces_np, bedge_np = self._compute_boundary_topology(simplices_np)
        self._bfaces = torch.as_tensor(bfaces_np, dtype=torch.long, device=self.device)
        self._bedge_k = torch.as_tensor(bedge_np, dtype=torch.long, device=self.device)
        LOGGER.info('found %d boundary edges', self._bfaces.shape[0])

        # per-triangle geometry, precomputed once (used by every divcurl/
        # gradient/need_volumes call)
        self._edges, self._normals, self._area = self._compute_face_geometry()
        self.pvolumes = self._compute_pvolumes()

        mtimer.end()
        LOGGER.info('Initializing took %s', mtimer)

    # --------------------------------------------------------------------
    @staticmethod
    def _compute_boundary_topology(simplices_np):
        '''
        Find boundary edges as (face_idx, k) pairs, where k in {0,1,2}
        identifies the edge opposite vertex `simplices[face_idx, k]`
        (matching pynhhd-v1.1's edge convention). A boundary edge is one
        not shared by any other triangle.
        '''
        T = simplices_np.shape[0]
        edge_to_faces = {}

        for fidx in range(T):
            tri = simplices_np[fidx]
            for k in range(3):
                a, b = tri[(k + 1) % 3], tri[(k + 2) % 3]
                key = (a, b) if a < b else (b, a)
                edge_to_faces.setdefault(key, []).append((fidx, k))

        bfaces, bedge_k = [], []
        for key, occurrences in edge_to_faces.items():
            if len(occurrences) == 1:
                fidx, k = occurrences[0]
                bfaces.append(fidx)
                bedge_k.append(k)

        return np.asarray(bfaces, dtype=np.int64), np.asarray(bedge_k, dtype=np.int64)

    # --------------------------------------------------------------------
    def _compute_face_geometry(self):
        '''
        edges[:, k]   = the edge vector opposite vertex k, oriented
                        counterclockwise (matches pynhhd-v1.1's tvec)
        normals[:, k] = the corresponding inward normal (rotate +90deg)
        area          = triangle area
        '''
        v0 = self.vertices[self.simplices[:, 0]]
        v1 = self.vertices[self.simplices[:, 1]]
        v2 = self.vertices[self.simplices[:, 2]]

        e0 = v2 - v1  # opposite vertex 0
        e1 = v0 - v2  # opposite vertex 1
        e2 = v1 - v0  # opposite vertex 2
        edges = torch.stack([e0, e1, e2], dim=1)  # (T, 3, 2)

        normals = torch.stack([-edges[..., 1], edges[..., 0]], dim=-1)  # (T, 3, 2)

        cross = (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]) - (v2[:, 1] - v0[:, 1]) * (v1[:, 0] - v0[:, 0])
        area = 0.5 * torch.abs(cross)  # (T,)

        return edges, normals, area

    def _compute_pvolumes(self):
        '''
        Vertex Voronoi-dual areas via Meyer et al.'s mixed-area algorithm,
        vectorized over all triangles with torch.where in place of
        pynhhd-v1.1's per-triangle if/elif/else.
        '''
        edges, area = self._edges, self._area
        e0, e1, e2 = edges[:, 0], edges[:, 1], edges[:, 2]

        l0 = (e0 * e0).sum(-1)
        l1 = (e1 * e1).sum(-1)
        l2 = (e2 * e2).sum(-1)

        ew0 = l0 * (l1 + l2 - l0)
        ew1 = l1 * (l2 + l0 - l1)
        ew2 = l2 * (l0 + l1 - l2)

        dot01 = (e0 * e1).sum(-1)
        dot02 = (e0 * e2).sum(-1)
        dot12 = (e1 * e2).sum(-1)

        # case A: obtuse at vertex 0 (ew0 <= 0)
        cA1 = -0.25 * l2 * area / dot02
        cA2 = -0.25 * l1 * area / dot01
        cA0 = area - cA1 - cA2

        # case B: obtuse at vertex 1 (ew1 <= 0, and not case A)
        cB2 = -0.25 * l0 * area / dot01
        cB0 = -0.25 * l2 * area / dot12
        cB1 = area - cB2 - cB0

        # case C: obtuse at vertex 2 (ew2 <= 0, and not case A or B)
        cC0 = -0.25 * l1 * area / dot12
        cC1 = -0.25 * l0 * area / dot02
        cC2 = area - cC0 - cC1

        # case D: acute/right triangle (Voronoi-safe)
        ewsum = ew0 + ew1 + ew2
        ewscale = 0.5 * area / ewsum
        cD0 = ewscale * (ew1 + ew2)
        cD1 = ewscale * (ew2 + ew0)
        cD2 = ewscale * (ew0 + ew1)

        maskA = ew0 <= 0
        maskB = (~maskA) & (ew1 <= 0)
        maskC = (~maskA) & (~maskB) & (ew2 <= 0)

        c0 = torch.where(maskA, cA0, torch.where(maskB, cB0, torch.where(maskC, cC0, cD0)))
        c1 = torch.where(maskA, cA1, torch.where(maskB, cB1, torch.where(maskC, cC1, cD1)))
        c2 = torch.where(maskA, cA2, torch.where(maskB, cB2, torch.where(maskC, cC2, cD2)))

        pvolumes = torch.zeros(self.nvertices, dtype=self.dtype, device=self.device)
        pvolumes.index_add_(0, self.simplices[:, 0], c0)
        pvolumes.index_add_(0, self.simplices[:, 1], c1)
        pvolumes.index_add_(0, self.simplices[:, 2], c2)
        return pvolumes

    # --------------------------------------------------------------------
    def _check_vfield(self, vfield):
        if vfield.shape[-1] != self.dim or vfield.shape[1] != self.nsimplices:
            raise ValueError('UnstructuredGrid requires a vector field of shape (B, {}, {}), got {}'.format(
                self.nsimplices, self.dim, tuple(vfield.shape)))

    def _check_sfield(self, sfield):
        if sfield.shape[1] != self.nvertices:
            raise ValueError('UnstructuredGrid requires a scalar field of shape (B, {}), got {}'.format(
                self.nvertices, tuple(sfield.shape)))

    # --------------------------------------------------------------------
    @torch.no_grad()
    def divcurl(self, vfield):
        '''vfield: (B, nsimplices, 2) piecewise-constant field -> (div, curlw), each (B, nvertices)'''

        self._check_vfield(vfield)
        LOGGER.info('Computing divcurl')
        mtimer = Timer(device=self.device)

        B = vfield.shape[0]

        # per-(face, corner) normal/tangential flux -- (B, T, 3)
        dn = torch.einsum('tkc,btc->btk', self._normals, vfield)
        dt = torch.einsum('tkc,btc->btk', self._edges, vfield)

        div = torch.zeros(B, self.nvertices, dtype=self.dtype, device=self.device)
        curlw = torch.zeros(B, self.nvertices, dtype=self.dtype, device=self.device)

        for k in range(3):
            div.index_add_(1, self.simplices[:, k], dn[..., k])
            curlw.index_add_(1, self.simplices[:, k], dt[..., k])

        # boundary-edge correction: reuses the SAME dn/dt already computed
        # above for (bface, bedge_k) -- no second, independently-computed
        # value and no shared mutable loop variable, unlike pynhhd-v1.1's
        # stale-`tn` bug (see module docstring).
        if self._bfaces.numel() > 0:
            a_verts = self.simplices[self._bfaces, (self._bedge_k + 1) % 3]
            b_verts = self.simplices[self._bfaces, (self._bedge_k + 2) % 3]

            dn_b = dn[:, self._bfaces, self._bedge_k]  # (B, E)
            dt_b = dt[:, self._bfaces, self._bedge_k]  # (B, E)

            div.index_add_(1, a_verts, dn_b)
            div.index_add_(1, b_verts, dn_b)
            curlw.index_add_(1, a_verts, dt_b)
            curlw.index_add_(1, b_verts, dt_b)

        div = div * -0.5
        curlw = curlw * 0.5

        mtimer.end()
        LOGGER.info('Computing divcurl took %s', mtimer)
        return div, curlw

    # --------------------------------------------------------------------
    @torch.no_grad()
    def gradient(self, sfield):
        '''sfield: (B, nvertices) -> (B, nsimplices, 2)'''

        self._check_sfield(sfield)
        LOGGER.info('Computing gradient')
        mtimer = Timer(device=self.device)

        sf = sfield[:, self.simplices]  # (B, T, 3)
        contrib = sf.unsqueeze(-1) * self._normals.unsqueeze(0)  # (B, T, 3, 2)
        grad = contrib.sum(dim=2) * (0.5 / self._area).view(1, -1, 1)

        mtimer.end()
        LOGGER.info('Computing gradient took %s', mtimer)
        return grad

    @torch.no_grad()
    def rotated_gradient(self, sfield):
        '''sfield: (B, nvertices) -> (B, nsimplices, 2), the pi/2-rotated gradient'''
        g = self.gradient(sfield)
        gx, gy = g[..., 0], g[..., 1]
        return torch.stack([-gy, gx], dim=-1)
