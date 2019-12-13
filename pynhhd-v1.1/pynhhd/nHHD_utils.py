import numpy
#from scipy import signal, spatial
from joblib import Parallel, delayed
from joblib import load, dump
import tempfile
import shutil
import os
from scipy import integrate #, special
#
# the following are not needed, unless wrap_xesmf will be called from outside
try:
    import xarray as xr
except ImportError:
    pass

try:
    import xesmf as xe
except ImportError:
    pass
#
def wrap_xesmf(sfield,lat,lon,lat2,lon2,dx,mode='bilinear',reuse_w=False,periodic=True):
    '''
    wrap xesmf
    '''
    lat2_b     = numpy.arange(lat2.min()-dx[0]/2, lat2.max()+dx[0], dx[0])
    lon2_b     = numpy.arange(lon2.min()-dx[1]/2, lon2.max()+dx[1], dx[1])
    lat_b      = numpy.arange(lat.min()-dx[0]/2, lat.max()+dx[0], dx[0])
    lon_b      = numpy.arange(lon.min()-dx[1]/2, lon.max()+dx[1], dx[1])
    ds_out     = xr.Dataset({'lat': (['lat'], lat[:,0]),'lon': (['lon'], lon[0,:]), 'lat_b':(['lat_b'], lat_b), 'lon_b': (['lon_b'], lon_b), })
    ds_in      = xr.Dataset({'lat': (['lat'], lat2[:,0]),'lon': (['lon'], lon2[0,:]), 'lat_b':(['lat_b'], lat2_b), 'lon_b': (['lon_b'], lon2_b), })
    regridder  = xe.Regridder(ds_in, ds_out, mode, filename=mode+'_'+str(lat.min())+str(lat.max())+str(lon.min())+str(lon.max())+'_to_'+str(lat2.min())+str(lat2.max())+str(lon2.min())+str(lon2.max())+'_periodic_'+str(periodic)+'.nc',reuse_weights=reuse_w,periodic=periodic)
    dum = sfield.copy()
    dum[numpy.where(numpy.isnan(dum))]=0
    mask=numpy.ones(dum.shape)
    mask[numpy.where(dum==0)]=0
    out = regridder(dum)
    mask  = regridder(mask)
    out = out/mask
    out[numpy.where(numpy.isnan(out))]=0
    #
    return out

def create_radialGrid_2(rlat,rlon,j,i):
    '''
    Should give the same answer as create_radialGrid_1
    but this version is more stable over all angles
    '''
    #
    Delta_lon    = abs(rlon[j,i] - rlon)
    cosDelta_lon = numpy.cos(Delta_lon)
    sinDelta_lon = numpy.sin(Delta_lon)
    coslat1      = numpy.cos(rlat[j,i])
    coslat2      = numpy.cos(rlat)
    sinlat1      = numpy.sin(rlat[j,i])
    sinlat2      = numpy.sin(rlat)
    num          = numpy.sqrt( (coslat2*sinDelta_lon)**2 + (coslat1*sinlat2 - sinlat1*coslat2*cosDelta_lon)**2)
    den          = sinlat1*sinlat2 + coslat1*coslat2*cosDelta_lon
    gamma        = numpy.arctan2(num,den)
    # Compute Green's function
    #G = -numpy.log(0.5*(1 - numpy.cos(gamma)))/(4*numpy.pi)
    G = -numpy.log(numpy.sin(0.5*gamma))/(2*numpy.pi)
    G[j,i] = 0
    #
    return G


def create_radialGrid_1(rlat2,rlon,j,i):
    '''
    NOTE THAT BECAUSE OF NUMERICAL REASONS IT IS !VERY! IMPORTANT TO USE AT LEAST
    NUMPY.FLOAT64 PRECISSION FOR RLAT AND RLON!!

    Returns

    G=-log(1-cos(y))/(4*np.pi)

    where y is the great circle angle

    Following Kimura 1999: Vortex motion on surfaces with constant curvature
    DOI: 10.1098/rspa.1999.0311

    and Kimura and Okamoto 1987: Vortex Motion on a Sphere, https://doi.org/10.1143/JPSJ.56.4203

    Note that the speacial case of Vincenty's formula is more accurate for great circle angle.
    '''
    #
    # This is directly following Kimura
    # NOTE THAT HERE RLAT SHOULD BE POLAR ANGLE - I.E PI/2-LAT
    rlat=numpy.pi/2 - rlat2
    cosg = numpy.cos(rlat)*numpy.cos(rlat[j,i])-numpy.sin(rlat)*numpy.sin(rlat[j,i])*numpy.cos(abs(rlon[j,i]-rlon))
    G = -numpy.log(0.5*(1-cosg))/(4*numpy.pi) #this is correct!!
    G[j,i] = 0 # set this to zero, will integrate the discontinuity elsewhere
    #
    return G

def int_fun(x,y):
    '''
    function to integrate over singularity
    '''
    return -numpy.log(numpy.sin(numpy.sqrt(x**2+y**2)/2))/(2*numpy.pi)

def solve_parallel(lat,lon,ny,nx,f,p,j,dxy,gdx2, tuning_param = 0.0625):
    '''
    This is the paralelel loop called by the solve function.

    The call from solve happens inside a parallel loop over latitudes
    so here we first create a Green's function (~distance function)
    for each latitude, integrate over the singularity (central point
    where distance goes to 0 and Green's function to infinity), and
    finally loop over each point in longitude and integrate G*f

    lat, lon and gdx2 are all in radians.
    '''
    if j%(ny//20)==0:
       print(str(100*j/ny)+'% ready')
    iinds = numpy.where(abs(f[j,:,0])>0)[0]
    if len(iinds)>0:
        G = create_radialGrid_2(lat,lon,j,0)
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
        # d_ang = tuning_param*(gdx2[0]+gdx2[1]*numpy.cos(lat[j,0]))
        # G[j,0] = (1/d_ang)*integrate.quad(lambda x: -numpy.log(numpy.sin(x/2))/(2*numpy.pi), 0,d_ang)[0]
        #
        # Direct double integral - this is more accurate, directly integrate over radial symmetry
        #
        da1 = tuning_param*gdx2[0]
        da2 = tuning_param*gdx2[1]*numpy.cos(lat[j,0])
        G[j,0]  = (1/(da1*da2))*integrate.dblquad(int_fun,0,da1,0,da2)[0]
        #
        for i in iinds:
            for k in range(2):
                p[j,i,k] = numpy.sum(f[:,:,k]*numpy.roll(G,i,axis=1)*dxy) #roll G over all the points, dxy is symmetrix in lon so no need to roll it

def solve(f,lat,lon,gdx,num_cores=20,r=6371E3, tuning_param=0.0625):
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
    f            : numpy.array, should be numpy.stack([self.div,self.curl],axis=-1)
    lat          : numpy.array (2D), latitude in degrees
    lon          : numpy.array (2D), longitude in degrees
    gdx          : list of integers or numpy.arrays specifying the grid size in lat/lon as [dy,dx]
    num_cores    : integer, number of cores to use, default is 10
    r            : float, radius of the planet in meters, default is earth i.e. R=6371E3 m
    tuning_param : float, tuning parameter for integrating over the Green's function singularity at
                   the point (j,i). Essentially a 1/weight for the central cell when convolving.
                   Default is 0.0625. See solve_parallel for detailed explanation.

    Returns
    --------
    phi, psi     : numpy.array (2D), estimates of the velocity potential (phi) and streamfunction (psi)

    '''
    print('solve:', lat.dtype,lon.dtype)
    ny, nx          = lat.shape
    gdx2            = numpy.radians(numpy.array(gdx))
    rlat            = numpy.radians(lat) # latitude - note that the definition for polar angle in Kimura measures from earth's axis i.e. 90-latitude
    rlon            = numpy.radians(lon) # azimuthal angle
    dy0             = r*numpy.radians(gdx[0])
    dx0             = r*numpy.cos(rlat)*numpy.radians(gdx[1])
    dxy             = (dy0*dx0)
    #
    folder1         = tempfile.mkdtemp()
    path1           = os.path.join(folder1, 'p.mmap')
    p               = numpy.memmap(path1, dtype=numpy.float, shape=(ny,nx,2), mode='w+')
    p[:]            = numpy.ones((ny,nx,2))*numpy.nan
    #
    Parallel(n_jobs = num_cores)(delayed(solve_parallel)(rlat,rlon,ny,nx,f,p,j,dxy,gdx2,tuning_param=tuning_param) for j in range(ny))
    p = numpy.array(p)
    try:
        shutil.rmtree(folder1)
    except OSError:
        pass
    #
    return p[:,:,0], p[:,:,1]
