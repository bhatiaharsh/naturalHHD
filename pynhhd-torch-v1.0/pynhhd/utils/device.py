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
device.py
    Shared dtype/device resolution and numpy<->torch/batch-shape helpers,
    reused by every grid type and the Poisson solver so this isn't
    reimplemented (with subtly different bugs) three times over.

    Design note: this whole library is meant to run under torch.no_grad()
    (see nhhd.py's decompose()) -- there is no supported differentiable path,
    so none of these helpers preserve or worry about autograd history.
'''

import numpy as np
import torch

# default precision: float32, favoring GPU throughput over the numpy
# reference's implicit float64. float64 remains available as an explicit
# opt-in (pass dtype=torch.float64) for accuracy-sensitive work -- see
# docs/CODE_REVIEW_REPORT.md and docs/PAPER_CRITIQUE.md for why the
# Green's-function self-term makes precision a real (not just cosmetic)
# consideration.
DEFAULT_DTYPE = torch.float32


def resolve_dtype(dtype=None):
    '''Return a concrete torch.dtype, defaulting to DEFAULT_DTYPE.'''
    if dtype is None:
        return DEFAULT_DTYPE
    if isinstance(dtype, torch.dtype):
        return dtype
    raise TypeError('dtype must be a torch.dtype (e.g. torch.float32), got {}'.format(type(dtype)))


def resolve_device(device=None, like=None):
    '''
    Return a concrete torch.device.
        device: explicit override (str or torch.device), takes precedence
        like:   a tensor to infer the device from, if device is not given
    Defaults to CPU if neither is given.
    '''
    if device is not None:
        return torch.device(device)
    if isinstance(like, torch.Tensor):
        return like.device
    return torch.device('cpu')


def as_tensor(x, dtype, device):
    '''
    Normalize a numpy array or torch tensor to a torch tensor with the given
    dtype/device. Returns (tensor, was_numpy) so callers can mirror the
    input type on output via to_output().
    '''
    was_numpy = isinstance(x, np.ndarray)
    if isinstance(x, torch.Tensor):
        t = x.to(dtype=dtype, device=device)
    else:
        t = torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    return t, was_numpy


def to_output(t, was_numpy):
    '''Mirror the input type on output: numpy in -> numpy out.'''
    if was_numpy:
        return t.detach().cpu().numpy()
    return t


def ensure_batched(x, unbatched_ndim):
    '''
    Add a leading batch dim of size 1 if `x` has exactly `unbatched_ndim`
    dims (a single, non-batched field). Returns (batched_tensor, was_batched)
    so the batch dim can be removed again via restore_batch().
    '''
    if x.dim() == unbatched_ndim:
        return x.unsqueeze(0), False
    if x.dim() == unbatched_ndim + 1:
        return x, True
    raise ValueError(
        'Expected a field with {} dims (unbatched) or {} dims (batched, '
        'leading batch axis), got shape {}'.format(
            unbatched_ndim, unbatched_ndim + 1, tuple(x.shape))
    )


def restore_batch(x, was_batched):
    '''Undo ensure_batched(): drop the leading batch dim if input wasn't batched.'''
    if was_batched:
        return x
    return x.squeeze(0)
