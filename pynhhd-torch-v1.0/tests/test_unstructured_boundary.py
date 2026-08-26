'''
Validation for the unstructured (triangular-mesh) path, with a focus on the
boundary-vertex curl correction -- this is exactly where pynhhd-v1.1's
stale-`tn` bug lived (pynhhd-v1.1/pynhhd/unstructured.py:328-329, see
docs/CODE_REVIEW_REPORT.md sec 2.2).

Key regression check: a spatially CONSTANT vector field is trivially both
divergence- and curl-free everywhere, including at the mesh boundary. A
correct discrete scheme (where the boundary correction supplies exactly the
missing "closing" edge of a boundary vertex's control volume) reproduces
this exactly (to floating-point precision), for every vertex. The numpy
bug added a leftover value from an unrelated face/edge into the boundary
curl instead of the freshly computed tangential flux for the *current*
edge -- for a constant field, that leftover value is generically nonzero
and unrelated to the (zero) correct answer, so this test would fail under
the bug.
'''

import numpy as np
import torch
import pytest

from pynhhd import nHHD, UnstructuredGrid


def _make_mesh(n=6, dtype=torch.float64, jitter=0.01, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    pts += rng.uniform(-jitter, jitter, size=pts.shape)
    return UnstructuredGrid(vertices=torch.as_tensor(pts, dtype=dtype), dtype=dtype)


class TestConstantFieldIsExactlyHarmonic:

    def test_div_and_curl_are_zero_everywhere_including_boundary(self):
        mesh = _make_mesh()
        assert mesh._bfaces.numel() > 0, 'test mesh should actually have a boundary'

        const_field = torch.tensor([0.7, -1.3], dtype=torch.float64)
        vfield = const_field.expand(1, mesh.nsimplices, 2)

        div, curl = mesh.divcurl(vfield)

        assert div.abs().max() < 1e-10
        assert curl.abs().max() < 1e-10

    def test_boundary_vertices_specifically(self):
        '''Isolate just the boundary vertices (rather than the whole-mesh max),
        since that's precisely where the numpy bug lived.'''
        mesh = _make_mesh()

        boundary_vertices = torch.unique(torch.cat([
            mesh.simplices[mesh._bfaces, (mesh._bedge_k + 1) % 3],
            mesh.simplices[mesh._bfaces, (mesh._bedge_k + 2) % 3],
        ]))
        assert boundary_vertices.numel() > 0

        const_field = torch.tensor([1.0, 0.5], dtype=torch.float64)
        vfield = const_field.expand(1, mesh.nsimplices, 2)
        div, curl = mesh.divcurl(vfield)

        assert div[0, boundary_vertices].abs().max() < 1e-10
        assert curl[0, boundary_vertices].abs().max() < 1e-10


class TestReconstructionAndStructure:

    def test_reconstruction_identity(self):
        rng = np.random.default_rng(1)
        mesh = _make_mesh(seed=1)
        pts = mesh.vertices.numpy()

        h = nHHD(points=pts, dtype=torch.float64)
        vfield = torch.as_tensor(rng.standard_normal((h.mesh.nsimplices, 2)), dtype=torch.float64)
        res = h.decompose(vfield)

        recon = res.d + res.r + res.h
        np.testing.assert_allclose(recon, vfield.numpy(), atol=1e-10)

    def test_d_is_curl_free_and_r_is_divergence_free(self):
        rng = np.random.default_rng(2)
        mesh = _make_mesh(seed=2)
        pts = mesh.vertices.numpy()

        h = nHHD(points=pts, dtype=torch.float64)
        vfield = torch.as_tensor(rng.standard_normal((h.mesh.nsimplices, 2)), dtype=torch.float64)
        res = h.decompose(vfield)

        _, curl_of_d = h.mesh.divcurl(torch.as_tensor(res.d).unsqueeze(0))
        div_of_r, _ = h.mesh.divcurl(torch.as_tensor(res.r).unsqueeze(0))

        assert curl_of_d.abs().max() < 1e-6
        assert div_of_r.abs().max() < 1e-6

    def test_batch_matches_loop(self):
        rng = np.random.default_rng(3)
        mesh = _make_mesh(seed=3)
        pts = mesh.vertices.numpy()

        h = nHHD(points=pts, dtype=torch.float64)
        B = 4
        batch = torch.as_tensor(rng.standard_normal((B, h.mesh.nsimplices, 2)), dtype=torch.float64)
        res_batch = h.decompose(batch)

        for b in range(B):
            res_one = h.decompose(batch[b])
            np.testing.assert_allclose(res_batch.d[b], res_one.d, atol=1e-8)
            np.testing.assert_allclose(res_batch.r[b], res_one.r, atol=1e-8)


def test_max_points_guard():
    from pynhhd.poisson import PoissonSolver

    with pytest.raises(ValueError, match='max_points'):
        PoissonSolver(points=np.zeros((100, 2)), pvolumes=np.ones(100), max_points=50)


def test_3d_unstructured_not_implemented():
    with pytest.raises(NotImplementedError):
        UnstructuredGrid(vertices=torch.zeros(4, 3))
