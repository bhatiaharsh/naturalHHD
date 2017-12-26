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
from scipy import signal, spatial
from timer import Timer

'''
# ------------------------------------------------------------------------------
 Poisson Solver for 1D, 2D, and 3D scalar fields.
    Data may be defined on a rectlinear grid or unstructured points.

  1) create an object
        PoissonSolver(args)
            'solver'      : type of solver
                            'S' for spatial convolution
                                    only option for unstructured points
                            'F' for convolution in frequency space
                                    default option for regular grids

         --> for regular grid
            'grid'        : shape of the rectlinear grid
                                tuple of dimensions (Z,Y,X)
            'spacings'    : spacings of the rectlinear grid
                                tuple of spacings (dz,dy,dx)

         --> for unstrucutred points
            'points'      : unstructured points
                                numpy array of shape (N, dim)
            'pvolumes'    : point volumes for each point
                                numpy array of shape (N, 1)

  2) call prepare()
          this initializes internal datastructures depending upon the mesh.

          Once initialized, solve() can be called any number of times to
          solve the Poisson equation on the same domain. If the domain
          changes (e.g., a Lagrangian mesh), create a new object.

  3) call solve(f), where, f is a numpy array defining a scalar field

          for rectlinear grid,     f.shape = grid.shape
          for unstructured points, f.shape = (points.shape[0], 1)

# ------------------------------------------------------------------------------
'''

class PoissonSolver(object):

    # constructor
    def __init__(self, **kwargs):

        args = kwargs.keys()

        if ('grid' in args) == ('points' in args):
            raise SyntaxError("Poisson Solver needs either shape of a regular grid or the points in an rectilinear/unstructured grid")

        if ('grid' in args) != ('spacings' in args):
            raise SyntaxError("Poisson Solver needs spacings for regular grid")

        if ('points' in args) != ('pvolumes' in args):
            raise SyntaxError("Poisson Solver needs point volumes for unstructured points")

        # (default) data type
        self.dtype = numpy.float32
        if 'dtype' in args:
            self.dtype = kwargs['dtype']

        if 'grid' in args:
            self.ptype = 'G'
            self.stype = 'F'             # default
            self.gdims = kwargs['grid']
            self.gdx = kwargs['spacings']
            self.dim = len(self.gdims)
            if self.dim != len(self.gdx):
                raise ValueError("Dimensions of spacings should match that of the grid")

        elif 'points' in args:
            self.ptype = 'P'
            self.stype = 'S'            # only option
            self.points = kwargs['points']
            self.pvolumes = kwargs['pvolumes']
            self.dim = len(self.points.shape)
            if self.pvolumes.shape[0] != self.points.shape[0]:
                raise ValueError("Number of pvolumes should match that of the points")

        if 'solver' in args:
            if (kwargs['solver'] != 'S' and kwargs['solver'] != 'F'):
                raise SyntaxError('Solver can be only of type S(patial) or F(requency)')

            if (kwargs['solver'] == 'F') and ('points' in args):
                raise SyntaxError('Only Spatial solver supported for unstructured points')

            if (kwargs['solver'] == 'S') and ('grid' in args):
                raise SyntaxError('Frequncy solver strongly suggested for structured grids')

            self.stype = kwargs['solver']

        if (self.dim > 3):
            raise ValueError("Poisson Solver works only for 1, 2, or 3 dimensions")

        self.ready = False
        if (self.ptype == 'G'):
            print 'PoissonSolver:', self.dim, 'D grid =', self.gdims, 'with spacings', self.gdx
        elif (self.ptype == 'P'):
            print 'PoissonSolver', self.dim, 'D points =', self.points.shape

    # --------------------------------------------------------------------------
    # create the Green's function in appropriate representation
    # --------------------------------------------------------------------------

    # generate a grid -(N-1) to (N-1) to define distance function
    #     where N is the size of the original grid
    def _create_radialGrid(self, verbose):

        mtimer = Timer()
        if verbose:
            print '  - creating distance kernel:',

        rdims = tuple(2*d - 1 for d in self.gdims)

        R = numpy.indices(rdims, dtype=self.dtype)
        for d in xrange(self.dim):
            R[d] += 1-self.gdims[d]
            R[d] *= self.gdx[d]

        if self.dim == 1:
            numpy.absolute(R[0], R[0])
            R = R[0]

        elif self.dim == 2:
            numpy.hypot(R[1], R[0], R[0])
            R = R[0]

            # half of the average grid spacing
            zval = 0.5 * (self.gdx[0]+self.gdx[1]) / 2.0
            R[self.gdims[0]-1, self.gdims[1]-1] = zval

        elif self.dim == 3:
            numpy.hypot(R[2], R[1], R[1])
            numpy.hypot(R[1], R[0], R[0])
            R = R[0]

            # half of the average grid spacing
            zval = 0.5 * (self.gdx[0]+self.gdx[1]+self.gdx[2]) / 3.0
            R[self.gdims[0]-1, self.gdims[1]-1, self.gdims[2]-1] = zval

        if verbose:
            print R.shape, R.min(), R.max(),
            mtimer.end()

        return R

    # --------------------------------------------------------------------------
    # scale a function by point volumes for correct integration
    def _scale(self, func):

        if self.ptype == 'P':
            return numpy.multiply(func, self.pvolumes)

        elif self.ptype == 'G':
            sfunc = func * numpy.prod(self.gdx)

            '''
            # volume of the boundary elements is only half the voxel
            if self.dim == 1:
                sfunc[ 0] *= 0.5
                sfunc[-1] *= 0.5
            elif self.dim == 2:
                sfunc[ 0, :] *= 0.5
                sfunc[-1, :] *= 0.5
                sfunc[ :, 0] *= 0.5
                sfunc[ :,-1] *= 0.5
            elif self.dim == 3:
                sfunc[ 0, :, :] *= 0.5
                sfunc[-1, :, :] *= 0.5
                sfunc[ :, 0, :] *= 0.5
                sfunc[ :,-1, :] *= 0.5
                sfunc[ :, :, 0] *= 0.5
                sfunc[ :, :,-1] *= 0.5
            '''
        return sfunc

    def prepare(self, verbose=False):

        # ----------------------------------------------------------------------
        gtimer = Timer()
        if verbose:
            print '\nInitializing Poisson solver, type =', self.stype

        # ----------------------------------------------------------------------
        # generate a grid to store pairwise distances to store Green's function
        self.G = None                      # definition depends upon solver type

        # ----------------------------------------------------------------------
        # for Fourier solution,
        if self.stype == 'F':

            if self.ptype == 'G':
                self.G = self._create_radialGrid(verbose)

            elif self.ptype == 'P':
                raise SyntaxError('Frequncy solver not supported for unstructured points')
                #self.G = create_distArray()

        # ----------------------------------------------------------------------
        # for spatial solution,
            # use the pairwise distance matrix, N x N
            # where N is the number of points
        elif self.stype == 'S':

            if self.ptype == 'G':
                raise SyntaxError('Spatial solver not suggested for structured grids')
                #self.G = self.create_points(verbose)

            elif self.ptype == 'P':
                # for 1D points, add an extra axis
                if (self.dim == 1) and (len(self.points.shape) == 1):
                    self.points = self.points[:,numpy.newaxis]

                # create pairwise distance matrix now
                ltimer = Timer()
                self.G = spatial.distance.cdist(self.points, self.points)

                # TODO: use a good approximation here
                zval = numpy.power(10.0, -10.0)
                self.G += zval * numpy.identity(self.G.shape[0])

                if verbose:
                    print '  - created pairwise distance matrix:', self.G.shape, self.G.min(), self.G.max(),
                    ltimer.end()

        # ----------------------------------------------------------------------
        # compute the Green's function
        ltimer = Timer()
        if verbose:
            print '  - computing the Green\'s function:',

        if self.dim == 1:
            numpy.multiply(self.G, 0.5, self.G)

        elif self.dim == 2:
            numpy.log(self.G, self.G)
            numpy.multiply(self.G, (0.5 / numpy.pi), self.G)

        elif self.dim == 3:
            numpy.reciprocal(self.G, self.G)
            numpy.multiply(self.G, (-0.25 / numpy.pi), self.G)

        if verbose:
            print self.G.shape, self.G.min(), self.G.max(),
            ltimer.end()

        # ----------------------------------------------------------------------
        self.ready = True

        if verbose:
            print 'Poisson solver initialized',
            gtimer.end()

    # --------------------------------------------------------------------------
    # compute the integral solution
    # --------------------------------------------------------------------------
    def solve(self, f, verbose=False):

        if not self.ready:
            init(verbose)

        fshape = f.shape

        gtimer = Timer()
        if verbose:
            print '\nSolving Poisson Eq.'

        # ----------------------------------------------------------------------
        # regular grid
        if self.ptype == 'G':

            #if (self.gdims - fshape).any():
            if self.gdims != fshape:
                print self.gdims, fshape
                raise ValueError("Shape of function should match shape of grid")

            # convolution in frequency domain
            if self.stype == 'F':
                p = signal.fftconvolve(f, self.G, mode='same')
                numpy.multiply(p, numpy.prod(self.gdx), p)

            # convolution in spatial domain
            elif self.stype == 'S':
                raise SyntaxError('Spatial solver not suggested for structured grids')
                #p = (self.G).dot(f.reshape(-1)).reshape(fshape)

        # ----------------------------------------------------------------------
        # unstructured points
        elif self.ptype == 'P':

            if (self.points.shape[0] != fshape[0]) or (1 != len(fshape)):
                raise ValueError("Shape of function should be (N,1) with N as the number of points")

            # convolution in spatial domain
                # Dec 13, 2017 -- pre scaling works for unstructured grids!
            if self.stype == 'S':
                p = numpy.multiply(f, self.pvolumes)
                numpy.dot(self.G, p, p)
                #numpy.multiply(p, self.pvolumes, p)

            elif self.stype == 'F':
                raise SyntaxError('Frequncy solver not supported for unstructured points')

        # ----------------------------------------------------------------------
        if verbose:
            print 'Poisson solver finished',
            gtimer.end()

        return p
# ------------------------------------------------------------------------------
