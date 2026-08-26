'''
CPU vs CUDA parity, skipped automatically on machines without a GPU.
'''

import numpy as np
import torch
import pytest

from pynhhd import nHHD

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')


def test_cpu_cuda_parity_2d():
    torch.manual_seed(0)
    grid, spacings = (20, 24), (0.1, 0.1)
    x = torch.randn(*grid, 2, dtype=torch.float32)

    h_cpu = nHHD(grid=grid, spacings=spacings, dtype=torch.float32, device='cpu')
    res_cpu = h_cpu.decompose(x)

    h_cuda = nHHD(grid=grid, spacings=spacings, dtype=torch.float32, device='cuda')
    res_cuda = h_cuda.decompose(x.to('cuda'))

    np.testing.assert_allclose(res_cpu.d, res_cuda.d.cpu(), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(res_cpu.r, res_cuda.r.cpu(), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(res_cpu.h, res_cuda.h.cpu(), rtol=1e-3, atol=1e-4)
