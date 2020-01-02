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

import numpy
from scipy import spatial

import logging
LOGGER = logging.getLogger(__name__)

from .utils.timer import Timer

'''
    Module for the computation of Green's funcion
'''
class GreensFunction(object):

    def __init__(self, dim, dtype):
        self.dim = dim
        self.dtype = dtype

    def compute(self, x):

        if self.dim == 1:
            numpy.absolute(x,x)
            numpy.multiply(x, 0.5, x)

        elif self.dim == 2:
            numpy.log(x,x)
            numpy.multiply(x, (0.5 / numpy.pi), x)

        elif self.dim == 3:
            numpy.reciprocal(x, x)
            numpy.multiply(x, (-0.25 / numpy.pi), x)

        return x

    # create green's function on a grid
    def create_grid(self, X, dx):

        LOGGER.info('Computing {}D Greens function'.format(self.dim))
        gtimer = Timer()
        ltimer = Timer()

        # ----------------------------------------------------------------------
        # create a half radial-grid
        if self.dim == 1:
            x = numpy.indices([X[0]], dtype=self.dtype)[0]
            numpy.multiply(x, dx, x)

        elif self.dim == 2:
            y, x = numpy.indices(X, dtype=self.dtype)
            numpy.multiply(y, dx[0], y)
            numpy.multiply(x, dx[1], x)
            numpy.hypot(y, x, x)
            x[0,0] = 0.5 * ((dx[0]+dx[1])/2.0)

        elif self.dim == 3:
            z, y, x = numpy.indices(X, dtype=self.dtype)
            numpy.multiply(z, dx[0], z)
            numpy.multiply(y, dx[1], y)
            numpy.multiply(x, dx[2], x)
            numpy.hypot(y, x, x)
            numpy.hypot(z, x, x)
            x[0,0,0] = 0.5 * ((dx[0]+dx[1]+dx[2])/3.0)

        ltimer.end()
        LOGGER.debug('creating half grid took {}'.format(ltimer))
        # ----------------------------------------------------------------------
        # create a half Green's grid
        ltimer.start()

        x = self.compute(x)

        ltimer.end()
        LOGGER.debug('creating green\'s function took {}'.format(ltimer))

        # ----------------------------------------------------------------------
        # create the full grid
        ltimer.start()
        if self.dim == 1:
            fx = numpy.flip(x, 0)
            x = numpy.concatenate((fx[:-1],x), axis=0)

        elif self.dim == 2:
            fx = numpy.flip(x, 1)
            x = numpy.concatenate((fx[:,:-1],x), axis=1)

            fx = numpy.flip(x, 0)
            x = numpy.concatenate((fx[:-1,:],x), axis=0)

        elif self.dim == 3:
            fx = numpy.flip(x, 2)
            x = numpy.concatenate((fx[:,:,:-1],x), axis=2)

            fx = numpy.flip(x, 1)
            x = numpy.concatenate((fx[:,:-1,:],x), axis=1)

            fx = numpy.flip(x, 0)
            x = numpy.concatenate((fx[:-1,:,:],x), axis=0)

        ltimer.end()
        gtimer.end()
        LOGGER.debug('creating full grid took {}'.format(ltimer))
        LOGGER.info('Computing {}D Greens function took {}'.format(self.dim, gtimer))
        return x

    # create green's function on points
    def create_points(self, P):

        LOGGER.info('Computing {}D Greens function'.format(self.dim))
        gtimer = Timer()
        ltimer = Timer()
        # for 1D points, add an extra axis
        if (self.dim == 1) and (len(P.shape) == 1):
            P = P[:,numpy.newaxis]

        # create pairwise distance matrix now
        ltimer = Timer()
        x = spatial.distance.cdist(P,P)

        ltimer.end()
        LOGGER.debug('creating pairwise distances took {}'.format(ltimer))

        ltimer.start()
        # TODO: use a good approximation here
        # TODO: also avoid multiplying with identity
        zval = numpy.power(10.0, -10.0)
        x += zval * numpy.identity(x.shape[0])

        ltimer.end()
        LOGGER.debug('adjusting self-distances took {}'.format(ltimer))

        x = self.compute(x)
        gtimer.end()
        LOGGER.info('Computing {}D Greens function took {}'.format(self.dim, gtimer))
        return x
