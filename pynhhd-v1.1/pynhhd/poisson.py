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

import os
import numpy as np
from scipy import signal, spatial, integrate

import shutil, tempfile
import gsw
from joblib import Parallel, delayed

import logging
LOGGER = logging.getLogger(__name__)

from .utils.timer import Timer
from .greens import GreensFunction

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

         --> for spherical grid
            'sphericalgrid' : shape of the sphericalgrid grid
                                tuple of dimensions (Z,Y,X)
            'spacings'    : spacings of the rectlinear grid
                                tuple of spacings (dz,dy,dx)

        --> for unstrucutred points
            'points'      : unstructured points
                                np array of shape (N, dim)
            'pvolumes'    : point volumes for each point
                                np array of shape (N, 1)

  2) call prepare()
          this initializes internal datastructures depending upon the mesh.

          Once initialized, solve() can be called any number of times to
          solve the Poisson equation on the same domain. If the domain
          changes (e.g., a Lagrangian mesh), create a new object.

  3) call solve(f), where, f is a np array defining a scalar field

          for rectlinear grid,     f.shape = grid.shape
          for unstructured points, f.shape = (points.shape[0], 1)

# ------------------------------------------------------------------------------
'''

class PoissonSolver(object):

    # constructor
    def __init__(self, **kwargs):

        args = list(kwargs.keys())

        is_rgrid = 'grid' in args
        is_sgrid = 'sphericalgrid' in args
        is_upoints = 'points' in args

        # give either a "grid" or a "sphericalgrid" or "points"
        if (int(is_rgrid) + int(is_sgrid) + int(is_upoints)) != 1:
            raise SyntaxError("Poisson Solver needs either shape of a regular grid or the points in an rectilinear/unstructured grid")

        if (is_rgrid or is_sgrid) != ('spacings' in args):
            raise SyntaxError("Poisson Solver needs spacings for regular/spherical grid")

        if (is_upoints) != ('pvolumes' in args):
            raise SyntaxError("Poisson Solver needs point volumes for unstructured points")

        # (default) data type
        self.dtype = np.float #32
        if 'dtype' in args:
            self.dtype = kwargs['dtype']

        if is_rgrid:
            self.ptype = 'G'
            self.stype = 'F'             # default
            self.gdims = kwargs['grid']
            self.gdx   = kwargs['spacings']
            self.dim   = len(self.gdims)
            #self.lat = None
            #self.lon = None
            if self.dim != len(self.gdx):
                raise ValueError("Dimensions of spacings should match that of the grid")

        elif 'sphericalgrid' in args:
            self.ptype = 'S'
            self.stype = 'F'
            self.gdims = kwargs['sphericalgrid']
            self.lat   = kwargs['lat']
            self.lon   = kwargs['lon']
            self.gdx   = kwargs['spacings']
            self.dim   = len(self.gdims)
            if self.dim != len(self.gdx):
                raise ValueError("Dimensions of spacings should match that of the grid")

        elif 'points' in args:
            self.ptype = 'P'
            self.stype = 'S'            # only option
            self.points = kwargs['points']
            self.pvolumes = kwargs['pvolumes']
            #self.lat = None
            #self.lon = None
            self.dim = len(self.points.shape)
            if self.pvolumes.shape[0] != self.points.shape[0]:
                raise ValueError("Number of pvolumes should match that of the points")

        if 'solver' in args:
            if (kwargs['solver'] != 'S' and kwargs['solver'] != 'F'):
                raise SyntaxError('Solver can be only of type S(patial) or F(requency)')

            if (kwargs['solver'] == 'F') and is_upoints:
                raise SyntaxError('Only Spatial solver supported for unstructured points')

            if (kwargs['solver'] == 'S') and is_rgrid:
                raise SyntaxError('Frequncy solver strongly suggested for structured grids')

            self.stype = kwargs['solver']

        if (self.dim > 3):
            raise ValueError("Poisson Solver works only for 1, 2, or 3 dimensions")

        self.ready = False
        if (self.ptype == 'G'):
            LOGGER.info('PoissonSolver: {}D regular grid = {} with spacings {}'.format(self.dim, self.gdims, self.gdx))

        elif (self.ptype == 'S'):
            LOGGER.info('PoissonSolver: {}D spherical grid = {} with spacings {}'.format(self.dim, self.gdims, self.gdx))

        elif (self.ptype == 'P'):
            LOGGER.info('PoissonSolver: {}D points = {}'.format(self.dim, self.points.shape))

        else:
            raise SyntaxError('Unknown input type!')


    # --------------------------------------------------------------------------
    def prepare(self, num_cores=20):

        # ----------------------------------------------------------------------
        gtimer = Timer()
        LOGGER.debug('Initializing Poisson solver, type = {}'.format(self.stype))

        # ----------------------------------------------------------------------
        # generate a grid to store pairwise distances to store Green's function
        GComputer = GreensFunction(self.dim, self.dtype)
        self.G = None                      # definition depends upon solver type

        # ----------------------------------------------------------------------
        # for Fourier solution,
        if self.stype == 'F':

            if self.ptype == 'G':
                self.G = GComputer.create_grid(self.gdims, self.gdx)

            elif self.ptype == 'S':
                self.G = GComputer.create_grid(self.gdims, self.gdx)
                #self.G = self._create_radialGrid()

            elif self.ptype == 'P':
                raise SyntaxError('Frequncy solver not supported for unstructured points')
                #self.G = create_distArray()

        # ----------------------------------------------------------------------
        # for spatial solution,
            # use the pairwise distance matrix, N x N
            # where N is the number of points
        elif self.stype == 'S':

            if self.ptype == 'P':
                self.G = GComputer.create_points(self.points)

            else:
                raise SyntaxError('Spatial solver not suggested for structured grids')
                #self.G = self.create_points(verbose)

        # ----------------------------------------------------------------------
        self.ready = True
        gtimer.end()
        LOGGER.debug('Poisson solver initialized! took {}'.format(gtimer))

    # --------------------------------------------------------------------------
    # compute the integral solution
    # --------------------------------------------------------------------------
    def _solve_spherical(self, func, num_cores=20, r=6371E3, tuning_param=0.0625):
        '''
        Convolution with Green's function with divergence and curl.
        Note that on a spherical partially masked domain (i.e. one with land)
        the Green's function is unique for each point and thus we need to integrate
        each point separately. Here we use just a Green's function on a sphere
        (i.e. Green's function is not aware of the land) in which case the Green's function
        is unique for each latitude.

        The function is parallized with joblib, with the outher parallel loop iterating
        through latitudes while the inner loop goes through longitudes.

        Parameters
        ----------
        f            : a single np.array, or a list of two numpt arrays
        lat          : np.array (2D), latitude in degrees
        lon          : np.array (2D), longitude in degrees
        gdx          : list of integers or np.arrays specifying the grid size in lat/lon as [dy,dx]
        num_cores    : integer, number of cores to use, default is 10
        r            : float, radius of the planet in meters, default is earth i.e. R=6371E3 m
        tuning_param : float, tuning parameter for integrating over the Green's function singularity at
                       the point (j,i). Essentially a 1/weight for the central cell when convolving.
                       Default is 0.0625. See solve_parallel for detailed explanation.

        Returns
        --------
        phi, psi     : np.array (2D), estimates of the velocity potential (phi) and streamfunction (psi)
        '''

        assert self.ptype == 'S'

        # ----------------------------------------------------------------------
        ny, nx          = self.lat.shape

        # spacing in radians
        rgdx            = np.radians(np.array(self.gdx))
        # rlat = latitude
            # note that the definition for polar angle in Kimura
            # measures from earth's axis i.e., 90-latitude
        rlat            = np.radians(self.lat)
        # rlon = azimuthal angle
        rlon            = np.radians(self.lon)

        dy0             = r*rgdx[0]
        dx0             = r*np.cos(rlat)*rgdx[1]
        dxy             = (dy0*dx0)

        #
        pfolder         = tempfile.mkdtemp()
        ppath           = os.path.join(pfolder, 'p.mmap')
        p               = np.memmap(ppath, dtype=np.float, shape=(ny,nx,2), mode='w+')
        p[:]            = np.ones((ny,nx,2))*np.nan


        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        def _solve_intgrate(x, y):
            '''
            function to integrate over singularity
            '''
            return -np.log(np.sin(np.sqrt(x*x+y*y)*0.5))/(2.*np.pi)

        # ----------------------------------------------------------------------
        def _solve_task(j):
            '''
            This is the parallel loop called by the _solve_spherical function.

            The call from _solve_spherical happens inside a parallel loop over latitudes
            so here we first create a Green's function (~distance function)
            for each latitude, integrate over the singularity (central point
            where distance goes to 0 and Green's function to infinity), and
            finally loop over each point in longitude and integrate G*f

            rlat, rlon, and rgdx are all in radians.
            '''

            assert isinstance(func, list)
            assert len(func) == 2

            iinds = np.where(abs(func[0][j,:]) > 0)[0]
            if len(iinds) == 0:
                return

            G = GreensFunction.create_sphericalGrid(rlat, rlon, j, 0)

            #
            # Here are two integrals for the Green's function singularity
            #
            # This is a line integral of the greensfunction log(sin(x/2))/2*pi  over the central cell stacked up due to symmetry -
            # Note that this is not completely accurate because the symmetry is circular, while we are stacking up to form a rectangle
            # Note that the integral is from 0 to g (the cell boundary in lat) and use the symmetry i.e. multiply by 2
            # here it is important to integrate from 0 to g and multiply by 2 or integrate from -g to g
            #
            # this is basically Area*mean_value i.e. Area*1/l int^l_0 log(1-cos(x)) dx
            #
            # d_ang = tuning_param*(gdx2[0]+gdx2[1]*np.cos(lat[j,0]))
            # G[j,0] = (1/d_ang)*integrate.quad(lambda x: -np.log(np.sin(x/2))/(2*np.pi), 0,d_ang)[0]
            #
            # Direct double integral - this is more accurate, directly integrate over radial symmetry
            #
            da1 = tuning_param * rgdx[0]
            da2 = tuning_param * rgdx[1] * np.cos(rlat[j,0])
            G[j,0]  = (1/(da1*da2)) * integrate.dblquad(_solve_intgrate, 0, da1, 0, da2)[0]
            #
            for i in iinds:
                p[j,i,0] = np.sum(func[0][:,:] * np.roll(G,i,axis=1)*dxy) #roll G over all the points, dxy is symmetrix in lon so no need to roll it
                p[j,i,1] = np.sum(func[1][:,:] * np.roll(G,i,axis=1)*dxy) #roll G over all the points, dxy is symmetrix in lon so no need to roll it

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        Parallel(n_jobs = num_cores)(delayed(_solve_task)(j) for j in range(ny))
        p = np.array(p)

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        try:
            shutil.rmtree(pfolder)
        except OSError:
            pass
        #
        return p[:,:,0], p[:,:,1]

    # --------------------------------------------------------------------------
    def _solve(self, f):

        assert self.ptype == 'G' or self.ptype == 'P'

        if not self.ready:
            self.prepare(num_cores)

        fshape = f.shape

        gtimer = Timer()
        LOGGER.debug('Solving Poisson equation')

        # ----------------------------------------------------------------------
        # regular grid
        if self.ptype == 'G':

            if self.gdims != fshape:
                raise ValueError("Shape of function ({}) should match shape of grid ({})".format(fshape, self.gdims))

            # convolution in frequency domain
            if self.stype == 'F':
                p = signal.fftconvolve(f, self.G, mode='same')
                np.multiply(p, np.prod(self.gdx), p)

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
                p = np.multiply(f, self.pvolumes)
                np.dot(self.G, p, p)

            elif self.stype == 'F':
                raise SyntaxError('Frequncy solver not supported for unstructured points')

        # ----------------------------------------------------------------------
        gtimer.end()
        LOGGER.debug('Poisson solver finished! took {}'.format(gtimer))
        return p

    # --------------------------------------------------------------------------
    # solve can expect either a single function, or a list of functions
    def solve(self, func, num_cores=20, r=6371E3, tuning_param=0.0625):

        # spherical grid solver
        if self.ptype == 'S':
            assert isinstance(func, list)
            assert len(func) == 2
            return self._solve_spherical(func, num_cores, r, tuning_param)

        # regular grid or unstructured points
        elif isinstance(func, list):

            nfuncs = len(func)
            assert nfuncs == 2 or nfuncs == 4

            results = [self._solve(f) for f in func]

            if nfuncs == 2:
                return results[0], results[1]
            else:
                return results[0], results[1], results[2], results[3]

        # a single function
        else:
            return self._solve(func)

    # --------------------------------------------------------------------------
