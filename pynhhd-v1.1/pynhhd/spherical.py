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
from scipy import interpolate
import logging
LOGGER = logging.getLogger(__name__)

from .utils.timer import Timer

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

# ------------------------------------------------------------------------------
class StructuredSphericalGrid(object):

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def __init__(self, **kwargs):

        '''
        kwargs:
            sphericalgrid:      ndarray of grid dimensions (Y,X) or (Z,Y,X)
            spacings:           ndarray of grid spacings (dy, dx) or (dz, dy, dx) (dy,dx are in degrees)
            verbose:     verbosity level
            lat:         latitude (2D)
            lon:         longitude (2D)
        '''

        args = list(kwargs.keys())

        if not ('sphericalgrid' in args) or not ('spacings' in args):
            raise SyntaxError("Dimensions and spacings of the grid are required")

        self.dims = kwargs['sphericalgrid']
        self.dx   = kwargs['spacings']      # in degrees - needs to be regular
        self.lat  = kwargs['lat']           # needs to be 2D
        self.lon  = kwargs['lon']           # needs to be 2D
        self.dim  = len(self.dims)

        if self.dim != 2 and self.dim != 3:
            raise ValueError("SphericalGrid works for 2D and 3D only")

        if self.dim != len(self.dx):
            raise ValueError("Dimensions of spacings should match that of the grid")

        LOGGER.info('Initialized {}D spherical grid'.format(self.dim))


    def divcurl2(self, vfield, verbose=False, glob=True, r=6371E3):
        '''
        THIS IS NOT USED AT THE MOMENT - BUT FEEL FREE TO PLAY AROUND

        Here we use splines to get more accurate (?) gradients than just central differences

        '''
        dudx_spline = interpolate.RectBivariateSpline(np.radians(self.lat[:,0]), np.radians(self.lon[0,:]), vfield[:,:,0])
        dvdx_spline = interpolate.RectBivariateSpline(np.radians(self.lat[:,0]), np.radians(self.lon[0,:]), vfield[:,:,1])
        dudy_spline = interpolate.RectBivariateSpline(np.radians(self.lat[:,0]), np.radians(self.lon[0,:]), vfield[:,:,0]*np.cos(np.radians(self.lat)))
        dvdy_spline = interpolate.RectBivariateSpline(np.radians(self.lat[:,0]), np.radians(self.lon[0,:]), vfield[:,:,1]*np.cos(np.radians(self.lat)))
        #
        dudx = dudx_spline.ev(np.radians(self.lat),np.radians(self.lon),0,1)/(r*np.cos(np.radians(self.lat))) #note that our variables are [y,x] so
        dudy = dudy_spline.ev(np.radians(self.lat),np.radians(self.lon),1,0)/(r*np.cos(np.radians(self.lat))) #therefore the derivatives are flipped from
        dvdx = dvdx_spline.ev(np.radians(self.lat),np.radians(self.lon),0,1)/(r*np.cos(np.radians(self.lat))) #what the documentation says
        dvdy = dvdy_spline.ev(np.radians(self.lat),np.radians(self.lon),1,0)/(r*np.cos(np.radians(self.lat)))
        #
        div = np.add(dudx, dvdy)
        curl = np.subtract(dvdx, dudy)

        return (div, curl)

    def divcurl3(self, vfield, verbose=False, glob=True, r=6371E3, use_xesmf=True, mode='bilinear'):
        '''
        THIS IS NOT USED AT THE MOMENT -BUT FEEL FREE TO PLAY AROUND

        Define the derivatives as forward derivatives and then use the spline interpolation to center the derivatives
        this is closer to the original central difference approach - and wrapping around is easier to take care of
        note that could also use xesmf for the regridding
        '''

        dudy = np.zeros(vfield[:,:,0].shape)
        dudx = np.zeros(vfield[:,:,0].shape)
        dvdy = np.zeros(vfield[:,:,0].shape)
        dvdx = np.zeros(vfield[:,:,0].shape)
        lon2,lat2 = np.meshgrid(np.arange(self.lon[0,0]+self.dx[1]/2,self.lon[0,-1]+self.dx[1],self.dx[1])-self.dx[1],np.arange(self.lat[0,0]+self.dx[0]/2,self.lat[-1,0]+self.dx[0],self.dx[0])-self.dx[0])
        lon2x = np.radians(lon2[:,1:])
        lat2x = np.radians(lat2[:,1:])
        lon2y = np.radians(lon2[1:,:])
        lat2y = np.radians(lat2[1:,:])
        dudy[1:,:] = (vfield[1:,:,0]*np.cos(np.radians(self.lat[1:,:]))-vfield[:-1,:,0]*np.cos(np.radians(self.lat[:-1,:])))/(np.radians(self.dx[0])*r*np.cos(lat2y))
        dvdy[1:,:] = (vfield[1:,:,1]*np.cos(np.radians(self.lat[1:,:]))-vfield[:-1,:,1]*np.cos(np.radians(self.lat[:-1,:])))/(np.radians(self.dx[0])*r*np.cos(lat2y))
        dudx[:,1:] = (vfield[:,1:,0]-vfield[:,:-1,0])/(np.radians(self.dx[1])*r*np.cos(lat2x))
        dvdx[:,1:] = (vfield[:,1:,1]-vfield[:,:-1,1])/(np.radians(self.dx[1])*r*np.cos(lat2x))
        dudx[:,0]  = (vfield[:,0,0]-vfield[:,1,0])/(np.radians(self.dx[1])*r*np.cos(lat2x[:,0]))
        dvdx[:,0]  = (vfield[:,0,1]-vfield[:,1,1])/(np.radians(self.dx[1])*r*np.cos(lat2x[:,0]))
        #
        if not use_xesmf:
            dudx_spline = interpolate.RectBivariateSpline(np.radians(self.lat[:,0]), np.radians(lon2[0,:]), dudx) #centered in lat, shifted in lon
            dvdx_spline = interpolate.RectBivariateSpline(np.radians(self.lat[:,0]), np.radians(lon2[0,:]), dvdx)
            dudy_spline = interpolate.RectBivariateSpline(np.radians(lat2[:,0]), np.radians(self.lon[0,:]), dudy) #centered in lon, shifted in lat
            dvdy_spline = interpolate.RectBivariateSpline(np.radians(lat2[:,0]), np.radians(self.lon[0,:]), dvdy)
            #
            dudx2 = dudx_spline.ev(np.radians(self.lat),np.radians(self.lon))
            dudy2 = dudy_spline.ev(np.radians(self.lat),np.radians(self.lon))
            dvdx2 = dvdx_spline.ev(np.radians(self.lat),np.radians(self.lon))
            dvdy2 = dvdy_spline.ev(np.radians(self.lat),np.radians(self.lon))
        elif use_xesmf:
            dudx2 = nutils.wrap_xesmf(dudx, self.lat, self.lon, self.lat,  lon2, self.dx, reuse_w=True, mode=mode)
            dvdx2 = nutils.wrap_xesmf(dvdx, self.lat, self.lon, self.lat,  lon2, self.dx, reuse_w=True, mode=mode)
            dudy2 = nutils.wrap_xesmf(dudy, self.lat, self.lon, lat2, self.lon,  self.dx, reuse_w=True, mode=mode)
            dvdy2 = nutils.wrap_xesmf(dvdy, self.lat, self.lon, lat2, self.lon,  self.dx, reuse_w=True, mode=mode)
        #
        div2 = np.add(dudx2, dvdy2)
        curl2 = np.subtract(dvdx2, dudy2)
        #
        print(div2.min(),div2.max())
        print(curl2.min(),curl2.max())
        #
        return (div2, curl2)

    #
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def divcurl(self, vfield, verbose=False, glob=True, r=6371E3):
        '''
        THIS IS USED AT THE MOMENT - EXACTLY THE SAME APPROACH AS IN NON-SPHERICAL GRID,
        BUT GRADIENTS ARE DEFINED ON A SPHERE

        (discrete) Divergence and curl are calculated from a vector field.
        If glob=True a wrap around in longitude is assumed.
        This function uses central differences.
        '''

        if (vfield.shape[-1] != self.dim) or (vfield.shape[0:self.dim] != self.dims):
            raise ValueError("Dimensions of vector field should match that of the grid")

        LOGGER.debug('Computing divcurl')
        mtimer = Timer()

        if self.dim == 2:

            if glob:
                #second order derivative at the wrapping longitude boundary
                dudy = np.gradient(vfield[:,:,0]*np.cos(np.radians(self.lat)), np.radians(self.dx[0]), axis=0)/(r*np.cos(np.radians(self.lat)))
                #dudy2 = np.gradient(vfield[:,:,0], np.radians(self.dx[0]), axis=-2)/(r*np.cos(np.radians(self.lat))) - vfield[:,:,0]*np.tan(np.radians(self.lat))/r
                dudx = np.gradient(vfield[:,:,0], np.radians(self.dx[1]), axis=1)/(r*np.cos(np.radians(self.lat)))
                dudx[:,0] = np.gradient(np.concatenate([vfield[:,-1:,0],vfield[:,:2,0]],axis=1), np.radians(self.dx[1]), axis=1)[:,1]/(r*np.cos(np.radians(self.lat[:,0])))
                dudx[:,-1] = np.gradient(np.concatenate([vfield[:,-2:,0],vfield[:,:1,0]],axis=1), np.radians(self.dx[1]), axis=1)[:,1]/(r*np.cos(np.radians(self.lat[:,0])))
                dvdy = np.gradient(vfield[:,:,1]*np.cos(np.radians(self.lat)), np.radians(self.dx[0]), axis=0)/(r*np.cos(np.radians(self.lat)))
                #dvdy2 = np.gradient(vfield[:,:,1], np.radians(self.dx[0]), axis=-2)/(r*np.cos(np.radians(self.lat))) - vfield[:,:,1]*np.tan(np.radians(self.lat))/r
                dvdx = np.gradient(vfield[:,:,1], np.radians(self.dx[1]), axis=1)/(r*np.cos(np.radians(self.lat)))
                dvdx[:,0] = np.gradient(np.concatenate([vfield[:,-1:,1],vfield[:,:2,1]],axis=1), np.radians(self.dx[1]), axis=1)[:,1]/(r*np.cos(np.radians(self.lat[:,0])))
                dvdx[:,-1] = np.gradient(np.concatenate([vfield[:,-2:,1],vfield[:,:1,1]],axis=1), np.radians(self.dx[1]), axis=1)[:,1]/(r*np.cos(np.radians(self.lat[:,0])))
            else:

                # self.dx = (dy,dx)
                dudy = np.gradient(vfield[:,:,0]*np.cos(np.radians(self.lat)), np.radians(self.dx[0]), axis=0)/(r*np.cos(np.radians(self.lat)))
                dudx = np.gradient(vfield[:,:,0], np.radians(self.dx[1]), axis=1)/(r*np.cos(np.radians(self.lat)))
                dvdy = np.gradient(vfield[:,:,1]*np.cos(np.radians(self.lat)), np.radians(self.dx[0]), axis=0)/(r*np.cos(np.radians(self.lat)))
                dvdx = np.gradient(vfield[:,:,1], np.radians(self.dx[1]), axis=1)/(r*np.cos(np.radians(self.lat)))

            div = np.add(dudx, dvdy)
            curl = np.subtract(dvdx, dudy) #note the sign convention

            mtimer.end()
            LOGGER.debug('Computing divcurl done! took {}'.format(mtimer))

            return (div, curl)

        elif self.dim == 3:
            print('NOT IMPLEMENTED - WILL RETURN THE SAME AS CARTESIAN DIV')
            # self.dx = (dz,dy,dx)
            dudz, dudy, dudx = np.gradient(vfield[:,:,:,0], self.dx[0], self.dx[1], self.dx[2])
            dvdz, dvdy, dvdx = np.gradient(vfield[:,:,:,1], self.dx[0], self.dx[1], self.dx[2])
            dwdz, dwdy, dwdx = np.gradient(vfield[:,:,:,2], self.dx[0], self.dx[1], self.dx[2])

            np.add(dudx, dvdy, dudx)
            np.add(dudx, dvdz, dudx)

            np.subtract(dwdy, dvdz, dwdy)
            np.subtract(dudz, dwdx, dudz)
            np.subtract(dvdx, dudy, dvdx)

            mtimer.end()
            LOGGER.debug('Computing divcurl done! took {}'.format(mtimer))

            return (dudx, dwdy, dudz, dvdx)

    def curl3D(self, vfield, verbose=False):

        #if (vfield.shape[-1] != self.dim) or (vfield.shape[0:self.dim] - self.dims).any():
        if (vfield.shape[-1] != self.dim) or (vfield.shape[0:self.dim] != self.dims):
            raise ValueError("Dimensions of vector field should match that of the grid")

        if self.dim != 3:
            raise ValueError("curl3D works only for 2D")

        if verbose:
            print('     Computing curl...',)
            sys.stdout.flush()
            mtimer = Timer()

        # self.dx = (dz,dy,dx)
        dudz, dudy, dudx = np.gradient(vfield[:,:,:,0], self.dx[0], self.dx[1], self.dx[2])
        dvdz, dvdy, dvdx = np.gradient(vfield[:,:,:,1], self.dx[0], self.dx[1], self.dx[2])
        dwdz, dwdy, dwdx = np.gradient(vfield[:,:,:,2], self.dx[0], self.dx[1], self.dx[2])

        np.subtract(dwdy, dvdz, dwdy)
        np.subtract(dudz, dwdx, dudz)
        np.subtract(dvdx, dudy, dvdx)

        if verbose:
            print(' Done!',)
            mtimer.end()

        return (dwdy, dudz, dvdx)

    def rotated_gradient(self, sfield, verbose=False, r=6371E3, glob=True):
        '''Same as gradient but rotated 90 degrees '''
        if (sfield.shape != self.dims):
	    #if (sfield.shape - self.dims).any():
            raise ValueError("Dimensions of scalar field should match that of the grid")

        if self.dim != 2:
            raise ValueError("rotated_gradient works only for 2D")

        if verbose:
            print('     Computing rotated gradient...',)
            sys.stdout.flush()
            mtimer = Timer()

        if glob:
            # central difference also at the boundary
            ddy = np.gradient(sfield, np.radians(self.dx[0]), axis=0)/r
            ddx = np.gradient(sfield, np.radians(self.dx[1]), axis=1)/(r*np.cos(np.radians(self.lat)))
            ddx[:,0] = np.gradient(np.concatenate([sfield[:,-1:],sfield[:,:2]],axis=1), np.radians(self.dx[1]), axis=1)[:,1]/(r*np.cos(np.radians(self.lat[:,0])))
            ddx[:,-1] = np.gradient(np.concatenate([sfield[:,-2:],sfield[:,:1]],axis=1), np.radians(self.dx[1]), axis=1)[:,1]/(r*np.cos(np.radians(self.lat[:,0])))
        else:
            #ddy, ddx = np.gradient(sfield, self.dx[0], self.dx[1])
            ddy = np.gradient(sfield, np.radians(self.dx[0]), axis=0)/r
            ddx = np.gradient(sfield, np.radians(self.dx[1]), axis=1)/(r*np.cos(np.radians(self.lat)))
        #
        ddy = -1.0*ddy

        grad = np.stack((ddy, ddx), axis=-1)

        if verbose:
            print(' Done!',)
            mtimer.end()

        return grad

    def gradient(self, sfield, verbose=False, glob=True, r=6371E3):
        '''
        Calculate the discrete (central differences) gradient of a scalar field.

        if glob=True we assume a global doamin and wrapping around in longitude (axis=1)
        In this case also the end points will use central differences, else they will use
        forward (backward) differences at the left and right hand extremes of the matrix.
        '''

        verbose = True

        if (sfield.shape != self.dims):
	    #if (sfield.shape - self.dims).any():
            raise ValueError("Dimensions of scalar field should match that of the grid")

        if verbose:
            print('     Computing gradient...',)
            sys.stdout.flush()
            mtimer = Timer()

        if self.dim == 2:

            print (glob, self.dx, r)
            print (sfield.min(), sfield.max())

            if glob:
                ddy = np.gradient(sfield, np.radians(self.dx[0]), axis=0)/r
                ddx = np.gradient(sfield, np.radians(self.dx[1]), axis=1)/(r*np.cos(np.radians(self.lat)))
                ddx[:,0] = np.gradient(np.concatenate([sfield[:,-1:],sfield[:,:2]],axis=1), np.radians(self.dx[1]), axis=1)[:,1]/(r*np.cos(np.radians(self.lat[:,0])))
                ddx[:,-1] = np.gradient(np.concatenate([sfield[:,-2:],sfield[:,:1]],axis=1), np.radians(self.dx[1]), axis=1)[:,1]/(r*np.cos(np.radians(self.lat[:,0])))
            else:
                # self.dx = (dy,dx)
                ddy = np.gradient(sfield, np.radians(self.dx[0]), axis=0)/r
                ddx = np.gradient(sfield, np.radians(self.dx[1]), axis=1)/(r*np.cos(np.radians(self.lat)))

            grad = np.stack((ddx, ddy), axis = -1)

        elif self.dim == 3:
            print('not implemented yet! - will return the same as cartesian')
            # self.dx = (dz,dy,dx)
            ddz, ddy, ddx = np.gradient(sfield, self.dx[0], self.dx[1], self.dx[2])
            grad = np.stack((ddx, ddy, ddz), axis = -1)

        if verbose:
            print(' Done!',)
            mtimer.end()

        return grad

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
