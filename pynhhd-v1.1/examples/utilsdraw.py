
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# discretize a colormap into n levels
def discretize_colormap(cmap, n):

    # extract all colors from the cmap
    cmaplist = [cmap(i) for i in range(cmap.N)]

    for i in xrange(n):
        oidx = i * float(cmap.N)/float(n)
        cmaplist[i] = cmap( int(oidx) )

    # create the new map
    return cmap.from_list('Custom cmap', cmaplist, n)

# draw streamlines on a regular grid
def draw_slines(X, Y, u, v, vrng):

    mgn = numpy.sqrt(u*u + v*v)
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

    mgn = numpy.linalg.norm(vfield, axis=1)
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
