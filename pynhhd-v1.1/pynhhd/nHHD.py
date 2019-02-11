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
from nPoisson import PoissonSolver
from StructuredGrid import StructuredGrid
from UnstructuredGrid import UnstructuredGrid
from StructuredSphericalGrid import StructuredSphericalGrid
import nHHD_utils as nutils

# ---------------------------------------------------------------------------
class nHHD(object):

    # constructor
    def __init__(self, **kwargs):

        args = kwargs.keys()

        if not ('grid' in args) and not ('points' in args) and not ('sphericalgrid' in args):
            raise SyntaxError("nHHD Solver needs either shape of a regular grid or the points in an rectilinear/unstructured grid")
        if (('grid' in args) != ('spacings' in args)) and (('sphericalgrid' in args) != ('spacings' in args)):
            raise SyntaxError("nHHD needs spacings for regular grid")
        if 'grid' in args:
            self.mesh = StructuredGrid(grid=kwargs['grid'], spacings=kwargs['spacings'])
            self.psolver = PoissonSolver(grid=self.mesh.dims, spacings=self.mesh.dx)
        elif 'sphericalgrid' in args:
            self.mesh = StructuredSphericalGrid(sphericalgrid=kwargs['sphericalgrid'], spacings=kwargs['spacings'],lat=kwargs['lat'],lon=kwargs['lon'])
            self.psolver = PoissonSolver(sphericalgrid=self.mesh.dims, spacings=self.mesh.dx,lat=kwargs['lat'],lon=kwargs['lon'])
        elif 'points' in args:

            if 'simplices' in args:
                self.mesh = UnstructuredGrid(vertices=kwargs['points'], simplices=kwargs['simplices'])
            else:
                self.mesh = UnstructuredGrid(vertices=kwargs['points'])

            self.psolver = PoissonSolver(points=self.mesh.vertices, pvolumes=self.mesh.pvolumes)

        self.dim = self.mesh.dim
        if (self.dim != 2) and (self.dim != 3):
            raise ValueError("nHHD Solver works only for 2 and 3 dimensions")

        if not 'sphericalgrid' in args:
            #no need to do this if spherical grid - we need to calculate greens's function multiple times
            self.psolver.prepare(True)

    # create the 3-component decomposition
    def decompose(self, vfield, verbose=0, num_cores=32, tuning_param=0.0625, div=None, curlw=None):

        if vfield.shape[-1] != self.dim:
            raise ValueError("nHHD.decompose requires a valid-dimensional vector field")

        if verbose > 0:
            print('\n ==> nhhd.decompose', vfield.shape)

        if self.dim == 2:

            # compute div curl
            if numpy.all(div==None) and numpy.all(curlw==None):
                # calculate div and curl if they are not known
                (self.div, self.curlw) = self.mesh.divcurl(vfield)
                #(self.div, self.curlw) = self.mesh.divcurl3(vfield,use_xesmf=True, mode='bilinear')
            else:
                # here is a possibility to give div and curl directly
                # highly suggested if your data is not on cartesian grid
                # e.g. with model output, use the model definition of the div and curl
                self.div = div
                self.curlw = curlw 

            #if verbose > 0:
            print('div =', self.div.shape, self.div.min(), self.div.max())
            print('curlw =', self.curlw.shape, self.curlw.min(), self.curlw.max())

            # compute potentials
            if False: #numpy.any(self.psolver.lat==None):
                self.nD = self.psolver.solve(self.div, verbose > 0) #1)
                if verbose > 0:
                    print('nD =', self.nD.shape, self.nD.min(), self.nD.max())

                self.nRu = self.psolver.solve(self.curlw, verbose > 0) #1)
                if verbose > 0:
                    print('nR =', self.nRu.shape, self.nRu.min(), self.nRu.max())
            else:
                self.nD, self.nRu = nutils.solve(numpy.stack([self.div,self.curlw],axis=-1),self.psolver.lat,self.psolver.lon,self.psolver.gdx,num_cores=num_cores,tuning_param=tuning_param)
            # compute fields as gradients of potentials
            self.d = self.mesh.gradient(self.nD)
            if verbose > 0:
                print('d =', self.d.shape, self.d.min(), self.d.max())

            self.r = self.mesh.rotated_gradient(self.nRu)
            if verbose > 0:
                print('r =', self.r.shape, self.r.min(), self.r.max())
        else:

            # compute div curl
            (self.div, self.curlu, self.curlv, self.curlw) = self.mesh.divcurl(vfield)

            if verbose > 0:
                print('div =', self.div.shape, self.div.min(), self.div.max())
                print('curlu =', self.curlu.shape, self.curlu.min(), self.curlu.max())
                print('curlv =', self.curlv.shape, self.curlv.min(), self.curlv.max())
                print('curlw =', self.curlw.shape, self.curlw.min(), self.curlw.max())

            # compute potentials
            self.nD = self.psolver.solve(self.div, verbose > 0) #1)
            if verbose > 0:
                print('nD =', self.nD.shape, self.nD.min(), self.nD.max())

            self.nRu = self.psolver.solve(self.curlu, verbose > 0)
            self.nRu *= -1.0
            if verbose > 0:
                print('nRu =', self.nRu.shape, self.nRu.min(), self.nRu.max())

            self.nRv = self.psolver.solve(self.curlv, verbose > 0)
            self.nRv *= -1.0
            if verbose > 0:
                print('nRv =', self.nRv.shape, self.nRv.min(), self.nRv.max())

            self.nRw = self.psolver.solve(self.curlw, verbose > 0)
            self.nRw *= -1.0

            if verbose > 0:
                print('nRw =', self.nRw.shape, self.nRw.min(), self.nRw.max())

            # compute divergent field as gradient of D potential
            self.d = self.mesh.gradient(self.nD)
            if verbose > 0:
                print('d =', self.d.shape, self.d.min(), self.d.max())

            # temporary use of r
            self.r = numpy.stack((self.nRu, self.nRv, self.nRw), axis = -1)

            # compute rotational field as curl of R potential
            (cu, cv, cw) = self.mesh.curl3D(self.r)
            self.r = numpy.stack((cu, cv, cw), axis = -1)
            if verbose > 0:
                print('r =', self.r.shape, self.r.min(), self.r.max())

        self.h = numpy.add(self.d, self.r)
        numpy.subtract(vfield, self.h, self.h)

        if verbose > 0:
            print('h =', self.h.shape, self.h.min(), self.h.max())
# ---------------------------------------------------------------------------
