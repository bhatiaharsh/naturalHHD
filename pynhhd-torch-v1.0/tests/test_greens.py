'''
Verify the torch FFT-convolution primitive (PoissonSolver's regular-grid
path) matches scipy.signal.fftconvolve(..., mode='same') directly -- i.e.
in isolation from the rest of the decomposition pipeline. This nails down
the kernel construction + 'same'-mode centering/padding logic specifically.
'''

import numpy as np
import torch
import pytest

scipy_signal = pytest.importorskip('scipy.signal')

from pynhhd.greens import GreensFunction
from pynhhd.poisson import PoissonSolver


def _numpy_reference_convolution(f_np, dims, dx, dim_):
    '''Mirror pynhhd-v1.1's approach: build the mirrored kernel with numpy
    (independently of pynhhd.greens), convolve via scipy, same as the
    numpy package does.'''
    if dim_ == 2:
        y, x = np.indices(dims, dtype=np.float64)
        y = y * dx[0]
        x = x * dx[1]
        d = np.hypot(y, x)
        d[0, 0] = 0.5 * ((dx[0] + dx[1]) / 2.0)
        g = (0.5 / np.pi) * np.log(d)

        fx = np.flip(g, 1)
        g = np.concatenate((fx[:, :-1], g), axis=1)
        fx = np.flip(g, 0)
        g = np.concatenate((fx[:-1, :], g), axis=0)

    else:
        raise NotImplementedError

    p = scipy_signal.fftconvolve(f_np, g, mode='same')
    p *= np.prod(dx)
    return p


@pytest.mark.parametrize('dims,dx', [
    ((17, 23), (0.3, 0.2)),   # odd sizes
    ((16, 24), (0.1, 0.15)),  # even sizes
])
def test_2d_convolution_matches_scipy(dims, dx):
    rng = np.random.default_rng(0)
    f_np = rng.standard_normal(dims).astype(np.float64)

    expected = _numpy_reference_convolution(f_np, dims, dx, dim_=2)

    solver = PoissonSolver(grid=dims, spacings=dx, dtype=torch.float64)
    solver.prepare()
    got = solver.solve(torch.as_tensor(f_np))  # PoissonSolver.solve works unbatched too (batch_shape=())
    got_np = got.cpu().numpy()

    np.testing.assert_allclose(got_np, expected, rtol=1e-8, atol=1e-8)


def test_greens_kernel_matches_numpy_mirroring():
    '''Sanity-check the torch kernel-mirroring logic (greens.py) directly
    against the numpy half-grid-then-flip/concatenate approach, for a small
    2D case, independent of the convolution/solve machinery.'''
    dims = (5, 7)
    dx = (0.3, 0.2)

    y, x = np.indices(dims, dtype=np.float64)
    y = y * dx[0]
    x = x * dx[1]
    d = np.hypot(y, x)
    d[0, 0] = 0.5 * ((dx[0] + dx[1]) / 2.0)
    g = (0.5 / np.pi) * np.log(d)
    fx = np.flip(g, 1)
    g = np.concatenate((fx[:, :-1], g), axis=1)
    fx = np.flip(g, 0)
    g_np = np.concatenate((fx[:-1, :], g), axis=0)

    gcomp = GreensFunction(dim=2, dtype=torch.float64, device=torch.device('cpu'))
    g_torch = gcomp.create_grid(dims, dx).numpy()

    np.testing.assert_allclose(g_torch, g_np, rtol=1e-10, atol=1e-10)
