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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# discretize a colormap into n levels
def discretize_colormap(cmap, n):

    # extract all colors from the cmap
    cmaplist = [cmap(i) for i in range(cmap.N)]

    for i in range(n):
        oidx = i * float(cmap.N)/float(n)
        cmaplist[i] = cmap( int(oidx) )

    # create the new map
    return cmap.from_list('Custom cmap', cmaplist, n)

# draw streamlines on a regular grid
def draw_slines(X, Y, u, v, vrng):

    mgn = np.sqrt(u*u + v*v)
    strm = plt.streamplot(X, Y, u, v, color=mgn, linewidth=2, cmap=plt.cm.autumn)
    plt.clim(vmin=vrng[0], vmax=vrng[1])

    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

    frame.axes.set_aspect('equal') #, 'datalim')
    plt.colorbar(strm.lines)

# draw streamlines on a triangulation
def draw_quivers(points, vfield, vrng, n=20):

    s = 350
    X = points[::n,0]
    Y = points[::n,1]
    u = vfield[::n,0]
    v = vfield[::n,1]

    mgn = np.linalg.norm(vfield, axis=1)
    strm = plt.quiver(X,Y,u,v,mgn,pivot='tail',cmap=plt.cm.autumn,scale=s,scale_units='width',width=0.005)
    plt.clim(vmin=vrng[0], vmax=vrng[1])

    #frame = plt.gca()
    #frame.axes.get_xaxis().set_visible(False)
    #frame.axes.get_yaxis().set_visible(False)

    #frame.axes.set_aspect('equal') #, 'datalim'))
    plt.colorbar()

# ------------------------------------------------------------------------------
def draw_scatter3D(positions, normals=None, color=None, alpha=1, ax=None):

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    if color == None:
        color = 'r'

    ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=color, marker='.', alpha=alpha)

    if (normals != None):
        ax.quiver( positions[:,0], positions[:,1], positions[:,2],
                   normals[:,0], normals[:,1], normals[:,2],
                   length=10,pivot='tail',color='cyan')
# ------------------------------------------------------------------------------
