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

import numpy as np
import logging
LOGGER = logging.getLogger(__name__)

from .utils.timer import Timer

# ------------------------------------------------------------------------------
class StructuredGrid(object):

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def __init__(self, **kwargs):
        '''
        kwargs:
            grid:        ndarray of grid dimensions (Y,X) or (Z,Y,X)
            spacings:    ndarray of grid spacings (dy, dx) or (dz, dy, dx)
        '''

        args = list(kwargs.keys())

        if ('grid' not in args) or ('spacings' not in args):
            raise SyntaxError("Dimensions and spacings of the grid are required")

        self.dims = kwargs['grid']
        self.dx = kwargs['spacings']
        self.dim = len(self.dims)

        if self.dim != 2 and self.dim != 3:
            raise ValueError("StructuredGrid works for 2D and 3D only")

        if self.dim != len(self.dx):
            raise ValueError("Dimensions of spacings should match that of the grid")

        LOGGER.info('Initialized {}D structured grid'.format(self.dim))

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def divcurl(self, vfield):

        if (vfield.shape[-1] != self.dim) or (vfield.shape[0:self.dim] != self.dims):
            raise ValueError("Dimensions of vector field should match that of the grid")

        LOGGER.debug('Computing divcurl')
        mtimer = Timer()

        if self.dim == 2:

            # self.dx = (dy,dx)
            dudy, dudx = np.gradient(vfield[:,:,0], self.dx[0], self.dx[1])
            dvdy, dvdx = np.gradient(vfield[:,:,1], self.dx[0], self.dx[1])

            np.add(dudx, dvdy, dudx)
            np.subtract(dvdx, dudy, dvdx)

            mtimer.end()
            LOGGER.debug('Computing divcurl done! took {}'.format(mtimer))
            return (dudx, dvdx)

        elif self.dim == 3:

            # self.dx = (dz,dy,dx)
            dudz, dudy, dudx = np.gradient(vfield[:,:,:,0], self.dx[0], self.dx[1], self.dx[2])
            dvdz, dvdy, dvdx = np.gradient(vfield[:,:,:,1], self.dx[0], self.dx[1], self.dx[2])
            dwdz, dwdy, dwdx = np.gradient(vfield[:,:,:,2], self.dx[0], self.dx[1], self.dx[2])

            np.add(dudx, dvdy, dudx)
            np.add(dudx, dvdz, dudx)

            np.subtract(dwdy, dvdz, dwdy)
            np.subtract(dudz, dwdx, dudz)
            np.subtract(dvdx, dudy, dvdx)

            mtimer.end()
            LOGGER.debug('Computing divcurl done! took {}'.format(mtimer))
            return (dudx, dwdy, dudz, dvdx)

    def curl3D(self, vfield):

        #if (vfield.shape[-1] != self.dim) or (vfield.shape[0:self.dim] - self.dims).any():
        if (vfield.shape[-1] != self.dim) or (vfield.shape[0:self.dim] != self.dims):
            raise ValueError("Dimensions of vector field should match that of the grid")

        if self.dim != 3:
            raise ValueError("curl3D works only for 2D")

        LOGGER.debug('Computing curl3D')
        mtimer = Timer()

        # self.dx = (dz,dy,dx)
        dudz, dudy, dudx = np.gradient(vfield[:,:,:,0], self.dx[0], self.dx[1], self.dx[2])
        dvdz, dvdy, dvdx = np.gradient(vfield[:,:,:,1], self.dx[0], self.dx[1], self.dx[2])
        dwdz, dwdy, dwdx = np.gradient(vfield[:,:,:,2], self.dx[0], self.dx[1], self.dx[2])

        np.subtract(dwdy, dvdz, dwdy)
        np.subtract(dudz, dwdx, dudz)
        np.subtract(dvdx, dudy, dvdx)

        mtimer.end()
        LOGGER.debug('Computing curl3D done! took {}'.format(mtimer))
        return (dwdy, dudz, dvdx)

    def rotated_gradient(self, sfield, verbose=False):

        if (sfield.shape != self.dims):
	    #if (sfield.shape - self.dims).any():
            raise ValueError("Dimensions of scalar field should match that of the grid")

        if self.dim != 2:
            raise ValueError("rotated_gradient works only for 2D")

        LOGGER.debug('Computing rotated gradient')
        mtimer = Timer()

        ddy, ddx = np.gradient(sfield, self.dx[0], self.dx[1])
        ddy *= -1.0

        grad = np.stack((ddy, ddx), axis=-1)

        mtimer.end()
        LOGGER.debug('Computing rotated gradient done! took {}'.format(mtimer))
        return grad

    def gradient(self, sfield, verbose=False):

        if (sfield.shape != self.dims):
	    #if (sfield.shape - self.dims).any():
            raise ValueError("Dimensions of scalar field should match that of the grid")

        LOGGER.debug('Computing gradient')
        mtimer = Timer()

        if self.dim == 2:

            # self.dx = (dy,dx)
            ddy, ddx = np.gradient(sfield, self.dx[0], self.dx[1])
            grad = np.stack((ddx, ddy), axis = -1)

        elif self.dim == 3:

            # self.dx = (dz,dy,dx)
            ddz, ddy, ddx = np.gradient(sfield, self.dx[0], self.dx[1], self.dx[2])
            grad = np.stack((ddx, ddy, ddz), axis = -1)

        mtimer.end()
        LOGGER.debug('Computing gradient done! took {}'.format(mtimer))
        return grad

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
