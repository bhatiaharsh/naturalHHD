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

import sys
import numpy as np
from scipy import spatial
import logging
LOGGER = logging.getLogger(__name__)

from .utils.timer import Timer

class UnstructuredGrid(object):

    '''Class to support nHHD on unstructured grids (triangular and tetrahedral)'''

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def need_volumes(self):

        ''' Compute volumes/areas for vertices, simplices, and corners'''

        if self.pvolumes.shape != (0,):
            return self.pvolumes

        def tri_area(o,a,b):
            return np.linalg.norm(np.cross(b-o,a-o)) / 2.0

        def tet_volume(o,a,b,c):
            return np.abs(np.dot(a-o,np.cross(b-o,c-o))) / 6.0

        LOGGER.info('Computing point areas/volumes')
        mtimer = Timer()

        self.svolumes = np.zeros(self.nsimplices,)
        self.pvolumes = np.zeros(self.nvertices,)
        self.cvolumes = np.zeros((self.nsimplices,self.dim+1))

        # triangulation
        if self.dim == 2:

            for sidx in range(self.nsimplices):

                simp = self.simplices[sidx]
                verts = self.vertices[simp]

                self.svolumes[sidx] = tri_area(verts[0], verts[1], verts[2])

                # edge weights
                e = [ verts[2]-verts[1], verts[0]-verts[2], verts[1]-verts[0] ]
                l2 = [ np.dot(e[0],e[0]), np.dot(e[1],e[1]), np.dot(e[2],e[2]) ]
                ew = [ l2[0]*(l2[1]+l2[2]-l2[0]), l2[1]*(l2[2]+l2[0]-l2[1]),l2[2]*(l2[0]+l2[1]-l2[2]) ]

                # corner areas
                if (ew[0] <= 0):
                    self.cvolumes[sidx,1] = -0.25 * l2[2] * self.svolumes[sidx] / np.dot(e[0], e[2])
                    self.cvolumes[sidx,2] = -0.25 * l2[1] * self.svolumes[sidx] / np.dot(e[0], e[1])
                    self.cvolumes[sidx,0] = self.svolumes[sidx] - self.cvolumes[sidx,1] - self.cvolumes[sidx,2]
                elif (ew[1] <= 0):
                    self.cvolumes[sidx,2] = -0.25 * l2[0] * self.svolumes[sidx] / np.dot(e[1], e[0]);
                    self.cvolumes[sidx,0] = -0.25 * l2[2] * self.svolumes[sidx] / np.dot(e[1], e[2]);
                    self.cvolumes[sidx,1] = self.svolumes[sidx] - self.cvolumes[sidx,2] - self.cvolumes[sidx,0];
                elif (ew[2] <= 0):
                    self.cvolumes[sidx,0] = -0.25 * l2[1] * self.svolumes[sidx] / np.dot(e[2], e[1]);
                    self.cvolumes[sidx,1] = -0.25 * l2[0] * self.svolumes[sidx] / np.dot(e[2], e[0]);
                    self.cvolumes[sidx,2] = self.svolumes[sidx] - self.cvolumes[sidx,0] - self.cvolumes[sidx,1];
                else:
                    ewscale = 0.5 * self.svolumes[sidx] / (ew[0] + ew[1] + ew[2])
                    for d in range(3):
                        self.cvolumes[sidx,d] = ewscale * (ew[(d+1)%3] + ew[(d+2)%3])

                self.pvolumes[simp[0]] += self.cvolumes[sidx,0]
                self.pvolumes[simp[1]] += self.cvolumes[sidx,1]
                self.pvolumes[simp[2]] += self.cvolumes[sidx,2]

        # tetrahedralization
        elif self.sdim == 3:

            raise ValueError('TODO: pvolumes for 3D')
            for sidx in range(self.nsimplices):

                simp = self.simplices[sidx]
                verts = self.vertices[simp]

                self.svolumes[sidx] = tet_volume(verts[0], verts[1], verts[2], verts[3])

                for v in simp:
                    self.pvolumes[v] += self.svolumes[sidx] / 4.0


        mtimer.end()
        LOGGER.info('Computing point areas/volume took {}'.format(mtimer))
        return self.pvolumes

    def need_adjacentfaces(self):

        '''
        Find adjacent faces for each vertex
            as list of lists
        '''

        if len(self.adjacent_faces) != 0:
            return self.adjacent_faces

        LOGGER.info('Computing adjacent_faces')
        mtimer = Timer()

        numadjacentfaces = np.zeros(self.nvertices, dtype=int)
        for f in self.simplices:
            for i in range(3):
                numadjacentfaces[f[i]] += 1

        # can be optimized further by avoiding "append"?
        self.adjacent_faces = [[] for _ in range(self.nvertices)]
        for fidx in range(self.nsimplices):
            for i in range(3):
                self.adjacent_faces[self.simplices[fidx, i]].append(fidx)

        mtimer.end()
        LOGGER.info('Computing adjacent_faces took {}'.format(mtimer))
        return self.adjacent_faces

    def need_acrossedge(self):

        '''
        Find adjacent faces for each face (across each edge)
            as ndarray of ints: shape (nsimplex, 3)
                 -1 denotes a face on the boundary (no face across edge)
        '''
        if self.across_edge.shape != (0,0):
            return self.across_edge

        self.need_adjacentfaces()

        LOGGER.info('Computing across_edge')
        mtimer = Timer()

        self.across_edge = -1 * np.ones((self.nsimplices, 3), dtype=np.int)

        for fidx in range(self.nsimplices):
            for i in range(3):

                if self.across_edge[fidx, i] != -1:
                    continue

                v1 = self.simplices[fidx, (i+1)%3]
                v2 = self.simplices[fidx, (i+2)%3]

                for other in self.adjacent_faces[v1]:

                    if other == fidx:
                        continue

                    if other not in self.adjacent_faces[v2]:
                        continue

                    oface = self.simplices[other]
                    j = np.where(oface == v1)[0]
                    j = (j+1)%3

                    if oface[(j+1)%3] != v2:
                        continue

                    self.across_edge[fidx, i] = other
                    self.across_edge[other, j] = fidx

        mtimer.end()
        LOGGER.info('Computing across_edge took {}'.format(mtimer))
        return self.across_edge

    def need_boundary(self):

        '''
        Find boundary of the triangulation
            as collection of boundary edeges as an array of [face, k]
        '''

        if self.bedges.shape != (0,0):
            return self.bedges

        self.need_acrossedge()

        LOGGER.info('Computing the boundary')
        mtimer = Timer()

        # find all boundary faces and edges
        bfaces = [fidx for fidx in range(self.nsimplices) if -1 in self.across_edge[fidx]]
        bedges = []

        for fidx in bfaces:
            face = self.simplices[fidx]
            nbrs = self.across_edge[fidx]
            bedges.extend( [[fidx, k] for k in range(3) if nbrs[k] == -1] )

        self.bedges = np.array(bedges)
        LOGGER.info('Computing the boundary found {} boundary edges'.format(self.bedges.shape[0]))
        mtimer.end()
        LOGGER.info('Computing the boundary took {}'.format(mtimer))
        return self.bedges

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def __init__(self, **kwargs):
        '''
        kwargs:
            vertices:    ndarray of shape (nverts, dim)      # dim = 2,3
            simplices:   ndarray of shape (nfaces, dim+1)
            verbose:     verbosity level
        '''

        args = list(kwargs.keys())

        if 'vertices' not in args:
            raise SyntaxError("Mesh object needs vertex data")

        self.vertices = kwargs['vertices']
        self.dim = self.vertices.shape[1]
        self.nvertices = self.vertices.shape[0]

        if self.dim != 2 and self.dim != 3:
            raise SyntaxError("Mesh object works for 2D and 3D only")

        LOGGER.info('Initializing {}D mesh with {} vertices'.format(self.dim, self.nvertices))
        mtimer = Timer()

        # create simplices if needed
        if 'simplices' in args:
            self.Delaunay = None
            self.simplices = kwargs['simplices']
            LOGGER.debug('got {} simplices'.format(self.simplices.shape[0]))

        else:
            LOGGER.debug('creating Delaunay mesh')

            self.Delaunay = spatial.Delaunay(self.vertices)
            self.simplices = self.Delaunay.simplices

            LOGGER.debug('created {} simplices'.format(self.simplices.shape[0]))

        self.nsimplices = self.simplices.shape[0]
        if self.dim != self.simplices.shape[1]-1:
            raise SyntaxError("Dimension mismatch! pdim = {} and sdim = {} do not match!".format(pdim, sdim))

        self.adjacent_faces = []
        self.across_edge = np.empty((0,0))
        self.bedges = np.empty((0,0))
        self.pvolumes = np.empty(0)

        self.need_volumes()
        self.need_boundary()
        #self.need_meshmatrices(verbose > 1)

        mtimer.end()
        LOGGER.info('Initializing took {}'.format(mtimer))

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def divcurl(self, vfield):

        if vfield.shape != (self.nsimplices, self.dim):
            LOGGER.error('vfield = {}, dim = {}, verts = {} simpls = {}'
                  .format(vfield.shape, self.dim, self.vertices.shape, self.simplices.shape))
            raise ValueError("UnstructuredGrid requires a valid-dimensional vector field")

        mtimer = Timer()
        LOGGER.info('Computing divcurl')

        div = np.zeros(self.nvertices)
        curlw = np.zeros(self.nvertices)

        # for each face
        for sidx in range(self.nsimplices):

            simp = self.simplices[sidx]

            for k in range(3):

                v = simp[k]
                a = simp[ (k+1)%3 ]
                b = simp[ (k+2)%3 ]

                # normal and tangent vectors
                tvec = self.vertices[b] - self.vertices[a]    # counterclockwise
                nvec = np.array([-tvec[1], tvec[0]])       # inward

                dn = np.dot(nvec, vfield[sidx])
                tn = np.dot(tvec, vfield[sidx])

                div[v]   += dn
                curlw[v] += tn

        # fix for boundary edges
        for bedge in self.bedges:

            sidx = bedge[0]
            eidx = bedge[1]

            a = self.simplices[sidx][(eidx+1)%3]
            b = self.simplices[sidx][(eidx+2)%3]

            tvec = self.vertices[b] - self.vertices[a]        # counterclockwise
            nvec = np.array([-tvec[1], tvec[0]])           # inward

            dn = np.dot(nvec, vfield[sidx])
            dt = np.dot(tvec, vfield[sidx])

            div[a]  += dn
            div[b]  += dn
            curlw[a] += tn
            curlw[b] += tn

        div   *= -0.5
        curlw *=  0.5

        mtimer.end()
        LOGGER.info('Computing divcurl took {}'.format(mtimer))
        return (div, curlw)

    def gradient(self, sfield):

        if sfield.shape[0] != self.nvertices:
            LOGGER.error('sfield = {}, dim = {}, verts = {} simpls = {}'
                  .format(sfield.shape, self.dim, self.vertices.shape, self.simplices.shape))
            raise ValueError("UnstructuredGrid requires a valid-dimensional scalar field")

        mtimer = Timer()
        LOGGER.info('Computing gradient')

        grad = np.zeros((self.nsimplices, self.dim))

        # for 2D
        for sidx in range(self.nsimplices):

            simp = self.simplices[sidx]
            f = 0.5 / self.svolumes[sidx]

            for k in range(3):

                v = simp[k]
                a = simp[ (k+1)%3 ]
                b = simp[ (k+2)%3 ]

                # normal and tangent vectors
                tvec = self.vertices[b] - self.vertices[a]    # counterclockwise
                nvec = np.array([-tvec[1], tvec[0]])       # inward

                grad[sidx] += f * sfield[v] * nvec

        mtimer.end()
        LOGGER.info('Computing gradient took {}'.format(mtimer))
        return grad

    def rotated_gradient(self, sfield):

        rgrad = self.gradient(sfield)

        rgrad[:,[0, 1]] = rgrad[:,[1, 0]]
        rgrad[:,0] *= -1.0

        return rgrad

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
