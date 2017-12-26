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
import numpy
from scipy import spatial
from timer import Timer

class UnstructuredGrid(object):

    '''Class to support nHHD on unstructured grids (triangular and tetrahedral)'''

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def need_volumes(self, verbose=False):

        ''' Compute volumes/areas for vertices, simplices, and corners'''

        if self.pvolumes.shape != (0,):
            return self.pvolumes

        def tri_area(o,a,b):
            return numpy.linalg.norm(numpy.cross(b-o,a-o)) / 2.0

        def tet_volume(o,a,b,c):
            return numpy.abs(numpy.dot(a-o,numpy.cross(b-o,c-o))) / 6.0

        if verbose:
            print '     Computing point areas/volumes...',
            sys.stdout.flush()
            mtimer = Timer()

        self.svolumes = numpy.zeros(self.nsimplices,)
        self.pvolumes = numpy.zeros(self.nvertices,)
        self.cvolumes = numpy.zeros((self.nsimplices,self.dim+1))

        # triangulation
        if self.dim == 2:

            for sidx in xrange(self.nsimplices):

                simp = self.simplices[sidx]
                verts = self.vertices[simp]

                self.svolumes[sidx] = tri_area(verts[0], verts[1], verts[2])

                # edge weights
                e = [ verts[2]-verts[1], verts[0]-verts[2], verts[1]-verts[0] ]
                l2 = [ numpy.dot(e[0],e[0]), numpy.dot(e[1],e[1]), numpy.dot(e[2],e[2]) ]
                ew = [ l2[0]*(l2[1]+l2[2]-l2[0]), l2[1]*(l2[2]+l2[0]-l2[1]),l2[2]*(l2[0]+l2[1]-l2[2]) ]

                # corner areas
                if (ew[0] <= 0):
                    self.cvolumes[sidx,1] = -0.25 * l2[2] * self.svolumes[sidx] / numpy.dot(e[0], e[2])
                    self.cvolumes[sidx,2] = -0.25 * l2[1] * self.svolumes[sidx] / numpy.dot(e[0], e[1])
                    self.cvolumes[sidx,0] = self.svolumes[sidx] - self.cvolumes[sidx,1] - self.cvolumes[sidx,2]
                elif (ew[1] <= 0):
                    self.cvolumes[sidx,2] = -0.25 * l2[0] * self.svolumes[sidx] / numpy.dot(e[1], e[0]);
                    self.cvolumes[sidx,0] = -0.25 * l2[2] * self.svolumes[sidx] / numpy.dot(e[1], e[2]);
                    self.cvolumes[sidx,1] = self.svolumes[sidx] - self.cvolumes[sidx,2] - self.cvolumes[sidx,0];
                elif (ew[2] <= 0):
                    self.cvolumes[sidx,0] = -0.25 * l2[1] * self.svolumes[sidx] / numpy.dot(e[2], e[1]);
                    self.cvolumes[sidx,1] = -0.25 * l2[0] * self.svolumes[sidx] / numpy.dot(e[2], e[0]);
                    self.cvolumes[sidx,2] = self.svolumes[sidx] - self.cvolumes[sidx,0] - self.cvolumes[sidx,1];
                else:
                    ewscale = 0.5 * self.svolumes[sidx] / (ew[0] + ew[1] + ew[2])
                    for d in xrange(3):
                        self.cvolumes[sidx,d] = ewscale * (ew[(d+1)%3] + ew[(d+2)%3])

                self.pvolumes[simp[0]] += self.cvolumes[sidx,0]
                self.pvolumes[simp[1]] += self.cvolumes[sidx,1]
                self.pvolumes[simp[2]] += self.cvolumes[sidx,2]

        # tetrahedralization
        elif self.sdim == 3:

            raise ValueError('TODO: pvolumes for 3D')
            for sidx in xrange(self.nsimplices):

                simp = self.simplices[sidx]
                verts = self.vertices[simp]

                self.svolumes[sidx] = tet_volume(verts[0], verts[1], verts[2], verts[3])

                for v in simp:
                    self.pvolumes[v] += self.svolumes[sidx] / 4.0


        if verbose:
            print ' Done!',
            mtimer.end()

        return self.pvolumes

    def need_adjacentfaces(self, verbose=False):

        '''
        Find adjacent faces for each vertex
            as list of lists
        '''

        if len(self.adjacent_faces) != 0:
            return self.adjacent_faces

        if verbose:
            print '     Computing adjacent_faces...',
            sys.stdout.flush()
            mtimer = Timer()

        numadjacentfaces = numpy.zeros(self.nvertices, dtype=int)
        for f in self.simplices:
            for i in xrange(3):
                numadjacentfaces[f[i]] += 1

        # can be optimized further by avoiding "append"?
        self.adjacent_faces = [[] for _ in xrange(self.nvertices)]
        for fidx in xrange(self.nsimplices):
            for i in xrange(3):
                self.adjacent_faces[self.simplices[fidx, i]].append(fidx)

        if verbose:
            print 'Done!',
            mtimer.end()

        return self.adjacent_faces

    def need_acrossedge(self, verbose=False):

        '''
        Find adjacent faces for each face (across each edge)
            as ndarray of ints: shape (nsimplex, 3)
                 -1 denotes a face on the boundary (no face across edge)
        '''
        if self.across_edge.shape != (0,0):
            return self.across_edge

        self.need_adjacentfaces(verbose)

        if verbose:
            print '     Computing across_edge...',
            sys.stdout.flush()
            mtimer = Timer()

        self.across_edge = -1 * numpy.ones((self.nsimplices, 3), dtype=numpy.int)

        for fidx in xrange(self.nsimplices):
            for i in xrange(3):

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
                    j = numpy.where(oface == v1)[0]
                    j = (j+1)%3

                    if oface[(j+1)%3] != v2:
                        continue

                    self.across_edge[fidx, i] = other
                    self.across_edge[other, j] = fidx

        if verbose:
            print ' Done!',
            mtimer.end()
        return self.across_edge

    def need_boundary(self, verbose=False):

        '''
        Find boundary of the triangulation
            as collection of boundary edeges as an array of [face, k]
        '''

        if self.bedges.shape != (0,0):
            return self.bedges

        self.need_acrossedge(verbose)

        if verbose:
            print '     Computing the boundary...',
            sys.stdout.flush()
            mtimer = Timer()

        # find all boundary faces and edges
        bfaces = [fidx for fidx in xrange(self.nsimplices) if -1 in self.across_edge[fidx]]
        bedges = []

        for fidx in bfaces:
            face = self.simplices[fidx]
            nbrs = self.across_edge[fidx]
            bedges.extend( [[fidx, k] for k in xrange(3) if nbrs[k] == -1] )

        self.bedges = numpy.array(bedges)
        if verbose:
            print ' Done! found', self.bedges.shape[0], 'boundary edges',
            mtimer.end()

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

        args = kwargs.keys()

        if 'vertices' not in args:
            raise SyntaxError("Mesh object needs vertex data")

        verbose = 2
        if 'verbose' in args:
            verbose = kwargs['verbose']

        self.vertices = kwargs['vertices']
        self.dim = self.vertices.shape[1]
        self.nvertices = self.vertices.shape[0]

        if self.dim != 2 and self.dim != 3:
            raise SyntaxError("Mesh object works for 2D and 3D only")

        if verbose > 0:
            print '     Initializing', self.dim, 'D mesh with', self.nvertices, 'vertices...',
            sys.stdout.flush()
            mtimer = Timer()

        if verbose > 1:
            print ''

        # create simplices if needed
        if 'simplices' in args:
            self.Delaunay = None
            self.simplices = kwargs['simplices']
            if verbose > 1:
                print '      got', self.simplices.shape[0], 'simplices'
        else:
            if verbose > 1:
                print '      creating Delaunay mesh...'
                sys.stdout.flush()

            self.Delaunay = spatial.Delaunay(self.vertices)
            self.simplices = self.Delaunay.simplices

            if verbose > 1:
                print ' Done! created', self.simplices.shape[0], 'simplices'
                sys.stdout.flush()

        self.nsimplices = self.simplices.shape[0]
        if self.dim != self.simplices.shape[1]-1:
            raise SyntaxError("Dimension mismatch! pdim = "+str(pdim)+" and sdim = "+str(sdim)+" do not match!")

        self.adjacent_faces = []
        self.across_edge = numpy.empty((0,0))
        self.bedges = numpy.empty((0,0))
        self.pvolumes = numpy.empty(0)

        self.need_volumes(verbose > 1)
        self.need_boundary(verbose > 1)
        #self.need_meshmatrices(verbose > 1)

        if verbose > 0:
            print ' Done!',
            mtimer.end()

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def divcurl(self, vfield, verbose=False):

        if vfield.shape != (self.nsimplices, self.dim):
            print 'UnstructuredGrid.divcurl', vfield.shape, self.dim, self.vertices.shape, self.simplices.shape
            raise ValueError("UnstructuredGrid requires a valid-dimensional vector field")

        if verbose:
            print '     Computing divcurl...',
            sys.stdout.flush()
            mtimer = Timer()

        div = numpy.zeros(self.nvertices)
        curlw = numpy.zeros(self.nvertices)

        # for each face
        for sidx in xrange(self.nsimplices):

            simp = self.simplices[sidx]

            for k in xrange(3):

                v = simp[k]
                a = simp[ (k+1)%3 ]
                b = simp[ (k+2)%3 ]

                # normal and tangent vectors
                tvec = self.vertices[b] - self.vertices[a]    # counterclockwise
                nvec = numpy.array([-tvec[1], tvec[0]])       # inward

                dn = numpy.dot(nvec, vfield[sidx])
                tn = numpy.dot(tvec, vfield[sidx])

                div[v]   += dn
                curlw[v] += tn

        # fix for boundary edges
        for bedge in self.bedges:

            sidx = bedge[0]
            eidx = bedge[1]

            a = self.simplices[sidx][(eidx+1)%3]
            b = self.simplices[sidx][(eidx+2)%3]

            tvec = self.vertices[b] - self.vertices[a]        # counterclockwise
            nvec = numpy.array([-tvec[1], tvec[0]])           # inward

            dn = numpy.dot(nvec, vfield[sidx])
            dt = numpy.dot(tvec, vfield[sidx])

            div[a]  += dn
            div[b]  += dn
            curlw[a] += tn
            curlw[b] += tn

        div = -0.5 * div
        curlw = 0.5 * curlw

        if verbose:
            print ' Done!',
            mtimer.end()
            #print '\tdiv =', div.shape, div.min(), div.max()
            #print '\tcurlw =', curlw.shape, curlw.min(), curlw.max()

        return (div, curlw)

    def gradient(self, sfield, verbose=False):

        if sfield.shape[0] != self.nvertices:
            print 'UnstructuredGrid.gradient', self.dim, sfield.shape, self.vertices.shape, self.simplices.shape
            raise ValueError("UnstructuredGrid requires a valid-dimensional vector field")

        if verbose:
            print '     Computing gradient...',
            sys.stdout.flush()
            mtimer = Timer()

        grad = numpy.zeros((self.nsimplices, self.dim))

        # for 2D
        for sidx in xrange(self.nsimplices):

            deb = sidx==0

            simp = self.simplices[sidx]
            f = 0.5 / self.svolumes[sidx]

            if deb:
                print sidx, simp, self.svolumes[sidx], f

            for k in xrange(3):

                v = simp[k]
                a = simp[ (k+1)%3 ]
                b = simp[ (k+2)%3 ]

                # normal and tangent vectors
                tvec = self.vertices[b] - self.vertices[a]    # counterclockwise
                nvec = numpy.array([-tvec[1], tvec[0]])       # inward

                grad[sidx] += f * sfield[v] * nvec

                if deb:
                    print '\t', k, nvec, sfield[v],grad[sidx]

        if verbose:
            print ' Done!',
            mtimer.end()
        return grad

    def rotated_gradient(self, sfield):

        rgrad = self.gradient(sfield)

        rgrad[:,[0, 1]] = rgrad[:,[1, 0]]
        rgrad[:,0] *= -1.0

        return rgrad

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
