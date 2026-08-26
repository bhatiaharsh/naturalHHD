'''
Decomposing a batch of B fields at once must match decomposing each field
individually in a loop, and no operation in the pipeline should build an
autograd graph (this library runs entirely under torch.no_grad()).
'''

import numpy as np
import torch
import pytest

from pynhhd import nHHD


@pytest.mark.parametrize('dim,grid,spacings', [
    (2, (18, 22), (0.1, 0.1)),
    (3, (8, 10, 12), (0.1, 0.1, 0.1)),
])
def test_batch_matches_loop(dim, grid, spacings):
    torch.manual_seed(0)
    B = 4
    shape = (B,) + grid + (dim,)
    batch = torch.randn(*shape, dtype=torch.float32)

    h = nHHD(grid=grid, spacings=spacings, dtype=torch.float32)

    res_batch = h.decompose(batch)

    for b in range(B):
        res_one = h.decompose(batch[b])
        np.testing.assert_allclose(res_batch.d[b], res_one.d, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(res_batch.r[b], res_one.r, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(res_batch.h[b], res_one.h, rtol=1e-4, atol=1e-5)


def test_decompose_runs_under_no_grad_even_if_input_requires_grad():
    grid, spacings = (16, 16), (0.1, 0.1)
    h = nHHD(grid=grid, spacings=spacings)

    x = torch.randn(*grid, 2, dtype=torch.float32, requires_grad=True)
    res = h.decompose(x)

    assert res.d.requires_grad is False
    assert res.r.requires_grad is False
    assert res.h.requires_grad is False
    assert res.d.grad_fn is None


def test_accepts_numpy_and_returns_numpy():
    grid, spacings = (12, 14), (0.1, 0.1)
    h = nHHD(grid=grid, spacings=spacings)

    x_np = np.random.default_rng(0).standard_normal((*grid, 2)).astype(np.float32)
    res = h.decompose(x_np)

    assert isinstance(res.d, np.ndarray)
    assert isinstance(res.r, np.ndarray)
    assert isinstance(res.h, np.ndarray)


def test_accepts_torch_and_returns_torch():
    grid, spacings = (12, 14), (0.1, 0.1)
    h = nHHD(grid=grid, spacings=spacings)

    x = torch.randn(*grid, 2, dtype=torch.float32)
    res = h.decompose(x)

    assert isinstance(res.d, torch.Tensor)
    assert isinstance(res.r, torch.Tensor)
    assert isinstance(res.h, torch.Tensor)
