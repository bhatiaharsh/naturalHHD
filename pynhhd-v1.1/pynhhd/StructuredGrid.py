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

# ------------------------------------------------------------------------------
class StructuredGrid(object):

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def __init__(self, **kwargs):

        '''
        kwargs:
            grid:        ndarray of grid dimensions (Y,X) or (Z,Y,X)
            spacings:    ndarray of grid spacings (dy, dx) or (dz, dy, dx)
            verbose:     verbosity level
        '''

        args = kwargs.keys()

        if ('grid' not in args) or ('spacings' not in args):
            raise SyntaxError("Dimensions and spacings of the grid are required")

        verbose = 0
        if 'verbose' in args:
            verbose = kwargs['verbose']

        self.dims = kwargs['grid']
        self.dx = kwargs['spacings']
        self.dim = len(self.dims)

        if self.dim != 2 and self.dim != 3:
            raise ValueError("StructuredGrid works for 2D and 3D only")

        if self.dim != len(self.dx):
            raise ValueError("Dimensions of spacings should match that of the grid")

        if verbose > 0:
            print '     Initializing', self.dim, 'D structured grid...',
            sys.stdout.flush()
            mtimer = Timer()
            print ' Done!',
            mtimer.end()

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def divcurl(self, vfield, verbose=False):

        #if (vfield.shape[-1] != self.dim) or (vfield.shape[0:self.dim] - self.dims).any():
        if (vfield.shape[-1] != self.dim) or (vfield.shape[0:self.dim] != self.dims):
            raise ValueError("Dimensions of vector field should match that of the grid")

        if verbose:
            print '     Computing divcurl...',
            sys.stdout.flush()
            mtimer = Timer()

        if self.dim == 2:

            # self.dx = (dy,dx)
            dudy, dudx = numpy.gradient(vfield[:,:,0], self.dx[0], self.dx[1])
            dvdy, dvdx = numpy.gradient(vfield[:,:,1], self.dx[0], self.dx[1])

            numpy.add(dudx, dvdy, dudx)
            numpy.subtract(dvdx, dudy, dvdx)

            if verbose:
                print ' Done!',
                mtimer.end()

            return (dudx, dvdx)

        elif self.dim == 3:

            # self.dx = (dz,dy,dx)
            dudz, dudy, dudx = numpy.gradient(vfield[:,:,:,0], self.dx[0], self.dx[1], self.dx[2])
            dvdz, dvdy, dvdx = numpy.gradient(vfield[:,:,:,1], self.dx[0], self.dx[1], self.dx[2])
            dwdz, dwdy, dwdx = numpy.gradient(vfield[:,:,:,2], self.dx[0], self.dx[1], self.dx[2])

            numpy.add(dudx, dvdy, dudx)
            numpy.add(dudx, dvdz, dudx)

            numpy.subtract(dwdy, dvdz, dwdy)
            numpy.subtract(dudz, dwdx, dudz)
            numpy.subtract(dvdx, dudy, dvdx)

            if verbose:
                print ' Done!',
                mtimer.end()

            return (dudx, dwdy, dudz, dvdx)

    def curl3D(self, vfield, verbose=False):

        #if (vfield.shape[-1] != self.dim) or (vfield.shape[0:self.dim] - self.dims).any():
        if (vfield.shape[-1] != self.dim) or (vfield.shape[0:self.dim] != self.dims):
            raise ValueError("Dimensions of vector field should match that of the grid")

        if self.dim != 3:
            raise ValueError("curl3D works only for 2D")

        if verbose:
            print '     Computing curl...',
            sys.stdout.flush()
            mtimer = Timer()

        # self.dx = (dz,dy,dx)
        dudz, dudy, dudx = numpy.gradient(vfield[:,:,:,0], self.dx[0], self.dx[1], self.dx[2])
        dvdz, dvdy, dvdx = numpy.gradient(vfield[:,:,:,1], self.dx[0], self.dx[1], self.dx[2])
        dwdz, dwdy, dwdx = numpy.gradient(vfield[:,:,:,2], self.dx[0], self.dx[1], self.dx[2])

        numpy.subtract(dwdy, dvdz, dwdy)
        numpy.subtract(dudz, dwdx, dudz)
        numpy.subtract(dvdx, dudy, dvdx)

        if verbose:
            print ' Done!',
            mtimer.end()

        return (dwdy, dudz, dvdx)

    def rotated_gradient(self, sfield, verbose=False):

        if (sfield.shape != self.dims):
	    #if (sfield.shape - self.dims).any():
            raise ValueError("Dimensions of scalar field should match that of the grid")

        if self.dim != 2:
            raise ValueError("rotated_gradient works only for 2D")

        if verbose:
            print '     Computing rotated gradient...',
            sys.stdout.flush()
            mtimer = Timer()

        ddy, ddx = numpy.gradient(sfield, self.dx[0], self.dx[1])
        ddy *= -1.0

        grad = numpy.stack((ddy, ddx), axis=-1)

        if verbose:
            print ' Done!',
            mtimer.end()

        return grad

    def gradient(self, sfield, verbose=False):

        if (sfield.shape != self.dims):
	    #if (sfield.shape - self.dims).any():
            raise ValueError("Dimensions of scalar field should match that of the grid")

        if verbose:
            print '     Computing gradient...',
            sys.stdout.flush()
            mtimer = Timer()

        if self.dim == 2:

            # self.dx = (dy,dx)
            ddy, ddx = numpy.gradient(sfield, self.dx[0], self.dx[1])
            grad = numpy.stack((ddx, ddy), axis = -1)

        elif self.dim == 3:

            # self.dx = (dz,dy,dx)
            ddz, ddy, ddx = numpy.gradient(sfield, self.dx[0], self.dx[1], self.dx[2])
            grad = numpy.stack((ddx, ddy, ddz), axis = -1)

        if verbose:
            print ' Done!',
            mtimer.end()

        return grad

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
