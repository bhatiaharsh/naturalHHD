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
from scipy import signal, spatial
import logging
LOGGER = logging.getLogger(__name__)

from .utils.timer import Timer
from .greens import GreensFunction

import gsw
from joblib import Parallel, delayed
from joblib import load, dump
import tempfile
import shutil
import os

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

        if not ('grid' in args) and not ('points' in args) and not ('sphericalgrid' in args):
            raise SyntaxError("Poisson Solver needs either shape of a regular grid or the points in an rectilinear/unstructured grid")

        if (('grid' in args) != ('spacings' in args)) and (('sphericalgrid' in args) != ('spacings' in args)):
            raise SyntaxError("Poisson Solver needs spacings for regular/spherical grid")

        if ('points' in args) != ('pvolumes' in args):
            raise SyntaxError("Poisson Solver needs point volumes for unstructured points")

        # (default) data type
        self.dtype = numpy.float #32
        if 'dtype' in args:
            self.dtype = kwargs['dtype']

        if 'grid' in args:
            self.ptype = 'G'
            self.stype = 'F'             # default
            self.gdims = kwargs['grid']
            self.gdx = kwargs['spacings']
            self.dim = len(self.gdims)
            self.lat = None
            self.lon = None
            if self.dim != len(self.gdx):
                raise ValueError("Dimensions of spacings should match that of the grid")
        elif 'sphericalgrid' in args:
            self.ptype = 'G'
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
            self.lat = None
            self.lon = None
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
            LOGGER.info('PoissonSolver: {}D grid = {} with spacings {}'.format(self.dim, self.gdims, self.gdx))
        elif (self.ptype == 'P'):
            LOGGER.info('PoissonSolver: {}D points = {}'.format(self.dim, self.points.shape))

    # --------------------------------------------------------------------------
    # create the Green's function in appropriate representation
    # --------------------------------------------------------------------------

    # generate a grid -(N-1) to (N-1) to define distance function
    #     where N is the size of the original grid
    # ALEKSI: SO I THINK I UNDERSTAND WHAT IS GOING ON HERE
    #         THE GRID IS FIRST CREATED AND THE DISTANCE ARRAY
    #         IS CALCULATED AS A HYPOTENUSE OF THE X AND Y
    #         I DON'T QUITE UNDERSTAND WHY THE RESULTING ARRAY
    #         NEEDS TO HAVE (2nX-1,2nY-1) SHAPE THOUGH
    #         In any case, give dx,dy as 1d arrays and we will stack them up
    #         accordingly
    def _create_radialGrid(self, verbose):

        mtimer = Timer()
        if verbose:
            print('  - creating distance kernel:',)

        rdims = tuple(2*d - 1 for d in self.gdims)

        if numpy.isscalar(self.gdx[0]):
            R = numpy.indices(rdims, dtype=self.dtype)
            for d in range(self.dim):
                R[d] += 1-self.gdims[d]
                R[d] *= self.gdx[d]

        if self.dim == 1:
            numpy.absolute(R[0], R[0])
            R = R[0]

        elif self.dim == 2:
            numpy.hypot(R[1], R[0], R[0])
            R = R[0]

            # half of the average grid spacing
            if numpy.isscalar(self.gdx[0]):
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
            print(R.shape, R.min(), R.max(),)
            mtimer.end()

        return R


    def _create_radialGrid_2(self, verbose, j, i=0):
        # use the AzimuthalEquidistant from cartopy on a sphere
        '''
        Create a distance matrix on a sphere in a global domain. Chances are that it will work also
        when the domain is not global, but not quite sure.
        We assume self.lat, self.lon are 2D, i!=0 is not implemented.
        '''
        # make a new lat,lon so that the poin j,i in question is at the center, when lat,lon are dublicated
        # around this point
        # at the moment only works if the grid is equal in lon (i.e. no shifting of lon is taking place)
        lat=numpy.concatenate([self.lat[:,:nx//2],self.lat[:,nx//2:][::-1,::-1]],axis=0)
        lon=numpy.concatenate([self.lon[:,:nx//2],self.lon[:,nx//2:][::-1,::-1]],axis=0)
        if j>0:
            lat2=numpy.concatenate([self.lat[:-j,:][::-1,:],self.lat[:j,:],self.lat[j:,:],self.lat[-j:,:][::-1,:]],axis=0)
            lat2=numpy.concatenate([lat2,lat2],axis=1)
            lon2=numpy.concatenate([self.lon[:-j,:][::-1,:],self.lon[:j,:],self.lon[j:,:],self.lon[-j:,:][::-1,:]],axis=0)
            lon2=numpy.concatenate([lon2[:,::-1],lon2],axis=1)
        elif j==0:
            lat2=numpy.concatenate([self.lat[::-1,:],self.lat],axis=0)
            lon2=numpy.concatenate([self.lon[::-1,:],self.lon],axis=0)
            lat2=numpy.concatenate([lat2,lat2],axis=1)
            lon2=numpy.concatenate([lon2[:,::-1],lon2],axis=1)
        #
        ny,nx=lat2.shape
        lons=numpy.stack([self.lon[j,i]*numpy.ones(nx*ny),lon2.flatten()],axis=-1)
        lats=numpy.stack([self.lat[j,i]*numpy.ones(nx*ny),lat2.flatten()],axis=-1)
        R=numpy.reshape(gsw.distance(lons,lats),(ny,nx)) #GREAT CIRCLE DISTANCE
        # no zeros allowed - put half of the minimum distance where ever there are zeros - this is the same as originally done
        R[numpy.where(R==0)]=numpy.min(R[ny//2-1:ny//2+2,nx//2-1:nx//2+2][numpy.where(R[ny//2-1:ny//2+2,nx//2-1:nx//2+2]>0)])/2
        #
        if verbose:
            print(R.shape, R.min(), R.max(),)
        return R

    def _create_radialGrid_3(self, verbose, ny, nx, j, i):
        ''' '''
        # center in longitude
        lat2=numpy.roll(self.lat,nx//4-i-1,axis=1)
        lon2=numpy.roll(self.lon,nx//4-i-1,axis=1)
        # flip around so wraps around in latitude
        lat2=numpy.concatenate([lat2[:,:nx//2],lat2[:,nx//2:][::-1,::-1]],axis=0)
        lon2=numpy.concatenate([lon2[:,:nx//2],lon2[:,nx//2:][::-1,::-1]],axis=0)
        # roll in latitude so that the j,i location is at 'north pole' (middle of the array)
        lat2=numpy.roll(lat2,ny-j-1,axis=0)
        lon2=numpy.roll(lon2,ny-j-1,axis=0)
        # calculate the distance matrix
        lons=numpy.stack([self.lon[j,i]*numpy.ones(nx*ny),lon2.flatten()],axis=-1)
        lats=numpy.stack([self.lat[j,i]*numpy.ones(nx*ny),lat2.flatten()],axis=-1)
        R=numpy.reshape(gsw.distance(lons,lats),(nx,ny)) #note the reshaping, we flipped things over
        R[numpy.where(R==0)]=numpy.min(R[ny-1:ny+2,nx//4-1:nx//4+2][numpy.where(R[ny-1:ny+2,nx//4-1:nx//4+2]>0)])/2
        #
        return R


    #def _create_radilGrid_2(self, verbose, j, i=0)
    #    ny, nx      = self.lat.shape
    #    folder1     = tempfile.mkdtemp()
    #    path1       = os.path.join(folder1, 'G.mmap')
    #    Gm          = np.memmap(path1, dtype=self.dtype, shape=(ny,ny,nx), mode='w+')
    #    Gm[:]       = numpy.ones((ny,ny,nx))
    #    Parallel(n_jobs=num_cores)(delayed(_create_radialGrid2)(ii,f_out,u_mm,v_mm,Pden_u,Pden_v,n,ny,nx) for ii in range(nx))
    #
    #
    # --------------------------------------------------------------------------
    # scale a function by point volumes for correct integration
    # IT SEEMS THAT THIS IS NOT USED
    def _scale(self, func):

        if self.ptype == 'P':
            return np.multiply(func, self.pvolumes)

        elif self.ptype == 'G':
            sfunc = func * np.prod(self.gdx)

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

    def prepare(self, num_cores=18):

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
                if numpy.any(self.lat==None):
                    self.G = self._create_radialGrid(verbose)
                else:
                    self.G = GComputer.create_grid(self.gdims, self.gdx)
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
                self.G = GComputer.create_points(self.points)

        # ----------------------------------------------------------------------
        self.ready = True
        gtimer.end()
        LOGGER.debug('Poisson solver initialized! took {}'.format(gtimer))
/*        
=======
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
                    print('  - created pairwise distance matrix:', self.G.shape, self.G.min(), self.G.max(),)
                    ltimer.end()

        # ----------------------------------------------------------------------
        # compute the Green's function
        ltimer = Timer()
        if verbose:
            print('  - computing the Green\'s function:',)

        if self.dim == 1:
            numpy.multiply(self.G, 0.5, self.G)

        elif self.dim == 2:
            numpy.log(self.G, self.G)
            numpy.multiply(self.G, (0.5 / numpy.pi), self.G)

        elif self.dim == 3:
            numpy.reciprocal(self.G, self.G)
            numpy.multiply(self.G, (-0.25 / numpy.pi), self.G)

        if verbose:
            print(self.G.shape, self.G.min(), self.G.max(),)
            ltimer.end()

        # ----------------------------------------------------------------------
        self.ready = True

        if verbose:
            print('Poisson solver initialized',)
            gtimer.end()
>>>>>>> enambling spherical geometry:pynhhd-v1.1/pynhhd/nPoisson.py
*/
    # --------------------------------------------------------------------------
    # compute the integral solution
    # --------------------------------------------------------------------------

    def create_shift_inds(self,ny,nx):
        ''' '''
        iinds0 = numpy.tile(numpy.arange(nx),(nx,1))
        jinds0 = numpy.tile(numpy.arange(ny*2),(ny*2,1))
        for i in range(nx):
              iinds0[i,:] =  numpy.roll(iinds0[i,:],nx//4-i-1)

        for j in range(ny*2):
              jinds0[j,:] = numpy.roll(jinds0[j,:],ny-j-1)

        return jinds0, iinds0

    def solve_parallel(self,verbose,ny,nx,f,p,j,iinds0,jinds0):
        '''this function can be run in parallel from the solve
           NEW STRATEGY - WE WILL LOOP OVER BOTH J AND I
           AT EVERY I MAKE EARTH WRAP AROUND IN LATITUDE
           AND PLACE THE NORTH POLE AT YOUR CURRENT LOCATION
          '''
        if j%(ny//10)==0:
            print(str(100*j/ny)+'% ready')
        iinds = numpy.where(f[j,:]!=0)[0]
        if len(iinds)>0:
          #G = self._create_radialGrid_3(False,ny,nx, j, 0)
          G=numpy.load('/export/scratch/anummel1/HelmholtzR/R_at_'+str(self.lat[j,0])+'.npz')['R']
          G = 1/(4*numpy.pi*G)
          #maybe it would be possible to first reshape and shift in y and then shift things in place in y
          #f2=numpy.concatenate([f2[:,:nx//2],f2[:,nx//2:][::-1,::-1]],axis=0)
          #f2=numpy.take(f,numpy.take(jinds0,j,axis=0),axis=0)
          for i in iinds:
               if j%20==0:
                   print(j,i)
               #if abs(f[j,i])>0
               #print(j,i)
               #G = self._create_radialGrid_3(False,ny,nx, j, i)
               #G = 1/(4*numpy.pi*G) #numpy.log(G)
               #G = numpy.multiply(G, (0.5 / numpy.pi))
               #p0 = signal.fftconvolve(f, G, mode='same')
               #f2=numpy.roll(f,nx//4-i-1,axis=1)
               #
               f2=numpy.take(f,numpy.take(iinds0,i,axis=0),axis=1) #why is this so fast compared to f[:,iinds0[i,:]]
               f2=numpy.concatenate([f2[:,:nx//2],f2[:,nx//2:][::-1,::-1]],axis=0)
               f2=numpy.take(f2,numpy.take(jinds0,j,axis=0),axis=0) #f2[jinds0[j,:],:] #numpy.roll(f2,ny-j-1,axis=0)
               p[j,i] = signal.fftconvolve(f2, G, mode='valid')
               #p[j,i] = p0 #[ny,nx//4]

    def solve(self, f, verbose=False, num_cores=20):

        if numpy.any(self.lat==None) and not self.ready:
            init(verbose)

                fshape = f.shape

        gtimer = Timer()
        LOGGER.debug('Solving Poisson equation')

        # ----------------------------------------------------------------------
        # regular grid
        if self.ptype == 'G':

            #if (self.gdims - fshape).any():
            if self.gdims != fshape:
                raise ValueError("Shape of function ({}) should match shape of grid ({})".format(fshape, self.gdims))

            # convolution in frequency domain
            if self.stype == 'F':
                if numpy.any(self.lat==None):
                    p = signal.fftconvolve(f, self.G, mode='same')
                    np.multiply(p, numpy.prod(self.gdx), p)
                else:
                    ny, nx      = self.lat.shape
                    jinds0,iinds0 = self.create_shift_inds(ny,nx)
                    folder1     = tempfile.mkdtemp()
                    path1       = os.path.join(folder1, 'p.mmap')
                    p           = numpy.memmap(path1, dtype=self.dtype, shape=(ny,nx), mode='w+')
                    p[:]        = numpy.ones((ny,nx))*numpy.nan
                    Parallel(n_jobs = num_cores)(delayed(self.solve_parallel)(verbose,ny,nx, f, p, j,iinds0,jinds0) for j in range(ny))
                    p           = numpy.array(p)
                    try:
                        shutil.rmtree(folder1)
                    except OSError:
                        pass
                    #
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
                p = np.multiply(f, self.pvolumes)
                np.dot(self.G, p, p)
                #np.multiply(p, self.pvolumes, p)

            elif self.stype == 'F':
                raise SyntaxError('Frequncy solver not supported for unstructured points')

        # ----------------------------------------------------------------------
        gtimer.end()
        LOGGER.debug('Poisson solver finished! took {}'.format(gtimer))
        return p
# ------------------------------------------------------------------------------
