'''
Copyright (c) 2015, Harsh Bhatia (bhatia4@llnl.gov)
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

import timeit

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class Timer(object):
    '''
    Elapsed-time helper. Pass `device=` (a torch.device/str) to synchronize
    CUDA before start/end -- otherwise elapsed time on a CUDA device only
    reflects kernel-launch latency, not actual GPU compute time, since CUDA
    kernels are launched asynchronously.
    '''

    def __init__(self, device=None):
        self.device = device
        self.start()

    def _sync(self):
        if self.device is None or torch is None:
            return
        dev = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        if dev.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize(dev)

    def start(self):
        self._sync()
        self.stime = timeit.default_timer()

    def end(self, print_time=False):
        self._sync()
        self.etime = timeit.default_timer()

        if print_time:
            print(self)

    def __str__(self):

        tseconds = self.etime - self.stime

        s = ' [[ elapsed time: '
        if tseconds < 0.000001:
            s = s + ("%.3f micro-sec." % (tseconds*1000000))

        elif tseconds < 0.001:
            s = s + ("%.3f milli-sec." % (tseconds*1000))

        elif tseconds < 60.0:
            s = s + ("%.3f sec." % (tseconds))

        else:
            m = int(tseconds/ 60.0)
            s = s + ("%d min. %.3f sec." % (m, (tseconds - 60*m)))

        s = s + ' ]]'
        return s

    def __repr__(self):
        return self.__str__()
