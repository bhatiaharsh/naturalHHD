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
from scipy import interpolate
import nHHD_utils as nutils
from timer import Timer

# ------------------------------------------------------------------------------
class StructuredSphericalGrid(object):

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def __init__(self, **kwargs):

        '''
        kwargs:
            sphericalgrid:        ndarray of grid dimensions (Y,X) or (Z,Y,X)
            spacings:    ndarray of grid spacings (dy, dx) or (dz, dy, dx) (dy,dx are in degrees)
            verbose:     verbosity level
            lat:         latitude (2D)
            lon:         longitude (2D)
        '''

        args = kwargs.keys()

        if not ('sphericalgrid' in args) or not ('spacings' in args):
            raise SyntaxError("Dimensions and spacings of the grid are required")

        verbose = 0
        if 'verbose' in args:
            verbose = kwargs['verbose']

        self.dims = kwargs['sphericalgrid']
        self.dx   = kwargs['spacings'] #in degrees - needs to be regular
        self.lat  = kwargs['lat'] #needs to be 2D
        self.lon  = kwargs['lon'] #needs to be 2D
        self.dim  = len(self.dims)

        if self.dim != 2 and self.dim != 3:
            raise ValueError("StructuredGrid works for 2D and 3D only")

        if self.dim != len(self.dx):
            raise ValueError("Dimensions of spacings should match that of the grid")

        if verbose > 0:
            print('     Initializing', self.dim, 'D structured grid...',)
            sys.stdout.flush()
            mtimer = Timer()
            print(' Done!',)
            mtimer.end()



    def divcurl2(self, vfield, verbose=False, glob=True, r=6371E3):
        '''
        THIS IS NOT USED AT THE MOMENT - BUT FEEL FREE TO PLAY AROUND

        Here we use splines to get more accurate (?) gradients than just central differences

        '''
        dudx_spline = interpolate.RectBivariateSpline(numpy.radians(self.lat[:,0]), numpy.radians(self.lon[0,:]), vfield[:,:,0])
        dvdx_spline = interpolate.RectBivariateSpline(numpy.radians(self.lat[:,0]), numpy.radians(self.lon[0,:]), vfield[:,:,1])
        dudy_spline = interpolate.RectBivariateSpline(numpy.radians(self.lat[:,0]), numpy.radians(self.lon[0,:]), vfield[:,:,0]*numpy.cos(numpy.radians(self.lat)))
        dvdy_spline = interpolate.RectBivariateSpline(numpy.radians(self.lat[:,0]), numpy.radians(self.lon[0,:]), vfield[:,:,1]*numpy.cos(numpy.radians(self.lat)))
        #
        dudx = dudx_spline.ev(numpy.radians(self.lat),numpy.radians(self.lon),0,1)/(r*numpy.cos(numpy.radians(self.lat))) #note that our variables are [y,x] so
        dudy = dudy_spline.ev(numpy.radians(self.lat),numpy.radians(self.lon),1,0)/(r*numpy.cos(numpy.radians(self.lat))) #therefore the derivatives are flipped from
        dvdx = dvdx_spline.ev(numpy.radians(self.lat),numpy.radians(self.lon),0,1)/(r*numpy.cos(numpy.radians(self.lat))) #what the documentation says
        dvdy = dvdy_spline.ev(numpy.radians(self.lat),numpy.radians(self.lon),1,0)/(r*numpy.cos(numpy.radians(self.lat)))
        #
        div = numpy.add(dudx, dvdy)
        curl = numpy.subtract(dvdx, dudy)

        return (div, curl)

    def divcurl3(self, vfield, verbose=False, glob=True, r=6371E3, use_xesmf=True, mode='bilinear'):
        '''
        THIS IS NOT USED AT THE MOMENT -BUT FEEL FREE TO PLAY AROUND

        Define the derivatives as forward derivatives and then use the spline interpolation to center the derivatives
        this is closer to the original central difference approach - and wrapping around is easier to take care of
        note that could also use xesmf for the regridding
        '''

        dudy = numpy.zeros(vfield[:,:,0].shape)
        dudx = numpy.zeros(vfield[:,:,0].shape)
        dvdy = numpy.zeros(vfield[:,:,0].shape)
        dvdx = numpy.zeros(vfield[:,:,0].shape)
        lon2,lat2 = numpy.meshgrid(numpy.arange(self.lon[0,0]+self.dx[1]/2,self.lon[0,-1]+self.dx[1],self.dx[1])-self.dx[1],numpy.arange(self.lat[0,0]+self.dx[0]/2,self.lat[-1,0]+self.dx[0],self.dx[0])-self.dx[0])
        lon2x = numpy.radians(lon2[:,1:])
        lat2x = numpy.radians(lat2[:,1:])
        lon2y = numpy.radians(lon2[1:,:])
        lat2y = numpy.radians(lat2[1:,:])
        dudy[1:,:] = (vfield[1:,:,0]*numpy.cos(numpy.radians(self.lat[1:,:]))-vfield[:-1,:,0]*numpy.cos(numpy.radians(self.lat[:-1,:])))/(numpy.radians(self.dx[0])*r*numpy.cos(lat2y))
        dvdy[1:,:] = (vfield[1:,:,1]*numpy.cos(numpy.radians(self.lat[1:,:]))-vfield[:-1,:,1]*numpy.cos(numpy.radians(self.lat[:-1,:])))/(numpy.radians(self.dx[0])*r*numpy.cos(lat2y))
        dudx[:,1:] = (vfield[:,1:,0]-vfield[:,:-1,0])/(numpy.radians(self.dx[1])*r*numpy.cos(lat2x))
        dvdx[:,1:] = (vfield[:,1:,1]-vfield[:,:-1,1])/(numpy.radians(self.dx[1])*r*numpy.cos(lat2x))
        dudx[:,0]  = (vfield[:,0,0]-vfield[:,1,0])/(numpy.radians(self.dx[1])*r*numpy.cos(lat2x[:,0]))
        dvdx[:,0]  = (vfield[:,0,1]-vfield[:,1,1])/(numpy.radians(self.dx[1])*r*numpy.cos(lat2x[:,0]))
        #
        if not use_xesmf:
            dudx_spline = interpolate.RectBivariateSpline(numpy.radians(self.lat[:,0]), numpy.radians(lon2[0,:]), dudx) #centered in lat, shifted in lon
            dvdx_spline = interpolate.RectBivariateSpline(numpy.radians(self.lat[:,0]), numpy.radians(lon2[0,:]), dvdx)
            dudy_spline = interpolate.RectBivariateSpline(numpy.radians(lat2[:,0]), numpy.radians(self.lon[0,:]), dudy) #centered in lon, shifted in lat
            dvdy_spline = interpolate.RectBivariateSpline(numpy.radians(lat2[:,0]), numpy.radians(self.lon[0,:]), dvdy)
            #
            dudx2 = dudx_spline.ev(numpy.radians(self.lat),numpy.radians(self.lon))
            dudy2 = dudy_spline.ev(numpy.radians(self.lat),numpy.radians(self.lon))
            dvdx2 = dvdx_spline.ev(numpy.radians(self.lat),numpy.radians(self.lon))
            dvdy2 = dvdy_spline.ev(numpy.radians(self.lat),numpy.radians(self.lon))
        elif use_xesmf:
            dudx2 = nutils.wrap_xesmf(dudx, self.lat, self.lon, self.lat,  lon2, self.dx, reuse_w=True, mode=mode)
            dvdx2 = nutils.wrap_xesmf(dvdx, self.lat, self.lon, self.lat,  lon2, self.dx, reuse_w=True, mode=mode)
            dudy2 = nutils.wrap_xesmf(dudy, self.lat, self.lon, lat2, self.lon,  self.dx, reuse_w=True, mode=mode)
            dvdy2 = nutils.wrap_xesmf(dvdy, self.lat, self.lon, lat2, self.lon,  self.dx, reuse_w=True, mode=mode)
        #
        div2 = numpy.add(dudx2, dvdy2)
        curl2 = numpy.subtract(dvdx2, dudy2)
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

        #if (vfield.shape[-1] != self.dim) or (vfield.shape[0:self.dim] - self.dims).any():
        if (vfield.shape[-1] != self.dim) or (vfield.shape[0:self.dim] != self.dims):
            raise ValueError("Dimensions of vector field should match that of the grid")

        if verbose:
            print('     Computing divcurl...',)
            sys.stdout.flush()
            mtimer = Timer()

        if self.dim == 2:

            if glob:
                #second order derivative at the wrapping longitude boundary
                dudy = numpy.gradient(vfield[:,:,0]*numpy.cos(numpy.radians(self.lat)), numpy.radians(self.dx[0]), axis=0)/(r*numpy.cos(numpy.radians(self.lat)))
                #dudy2 = numpy.gradient(vfield[:,:,0], numpy.radians(self.dx[0]), axis=-2)/(r*numpy.cos(numpy.radians(self.lat))) - vfield[:,:,0]*numpy.tan(numpy.radians(self.lat))/r
                dudx = numpy.gradient(vfield[:,:,0], numpy.radians(self.dx[1]), axis=1)/(r*numpy.cos(numpy.radians(self.lat)))
                dudx[:,0] = numpy.gradient(numpy.concatenate([vfield[:,-1:,0],vfield[:,:2,0]],axis=1), numpy.radians(self.dx[1]), axis=1)[:,1]/(r*numpy.cos(numpy.radians(self.lat[:,0])))
                dudx[:,-1] = numpy.gradient(numpy.concatenate([vfield[:,-2:,0],vfield[:,:1,0]],axis=1), numpy.radians(self.dx[1]), axis=1)[:,1]/(r*numpy.cos(numpy.radians(self.lat[:,0])))
                dvdy = numpy.gradient(vfield[:,:,1]*numpy.cos(numpy.radians(self.lat)), numpy.radians(self.dx[0]), axis=0)/(r*numpy.cos(numpy.radians(self.lat)))
                #dvdy2 = numpy.gradient(vfield[:,:,1], numpy.radians(self.dx[0]), axis=-2)/(r*numpy.cos(numpy.radians(self.lat))) - vfield[:,:,1]*numpy.tan(numpy.radians(self.lat))/r
                dvdx = numpy.gradient(vfield[:,:,1], numpy.radians(self.dx[1]), axis=1)/(r*numpy.cos(numpy.radians(self.lat)))
                dvdx[:,0] = numpy.gradient(numpy.concatenate([vfield[:,-1:,1],vfield[:,:2,1]],axis=1), numpy.radians(self.dx[1]), axis=1)[:,1]/(r*numpy.cos(numpy.radians(self.lat[:,0])))
                dvdx[:,-1] = numpy.gradient(numpy.concatenate([vfield[:,-2:,1],vfield[:,:1,1]],axis=1), numpy.radians(self.dx[1]), axis=1)[:,1]/(r*numpy.cos(numpy.radians(self.lat[:,0])))
            else:
                # self.dx = (dy,dx)
                dudy = numpy.gradient(vfield[:,:,0]*numpy.cos(numpy.radians(self.lat)), numpy.radians(self.dx[0]), axis=0)/(r*numpy.cos(numpy.radians(self.lat)))
                dudx = numpy.gradient(vfield[:,:,0], numpy.radians(self.dx[1]), axis=1)/(r*numpy.cos(numpy.radians(self.lat)))
                dvdy = numpy.gradient(vfield[:,:,1]*numpy.cos(numpy.radians(self.lat)), numpy.radians(self.dx[0]), axis=0)/(r*numpy.cos(numpy.radians(self.lat)))
                dvdx = numpy.gradient(vfield[:,:,1], numpy.radians(self.dx[1]), axis=1)/(r*numpy.cos(numpy.radians(self.lat)))

            div = numpy.add(dudx, dvdy)
            curl = numpy.subtract(dvdx, dudy) #note the sign convention

            if verbose:
                print(' Done!',)
                mtimer.end()

            return (div, curl)

        elif self.dim == 3:
            print('NOT IMPLEMENTED - WILL RETURN THE SAME AS CARTESIAN DIV')
            # self.dx = (dz,dy,dx)
            dudz, dudy, dudx = numpy.gradient(vfield[:,:,:,0], self.dx[0], self.dx[1], self.dx[2])
            dvdz, dvdy, dvdx = numpy.gradient(vfield[:,:,:,1], self.dx[0], self.dx[1], self.dx[2])
            dwdz, dwdy, dwdx = numpy.gradient(vfield[:,:,:,2], self.dx[0], self.dx[1], self.dx[2])

            numpy.add(dudx, dvdy, dudx)
            numpy.add(dudx, dvdz, dudx)

            numpy.subtract(dwdy, dvdz, dwdy)
            numpy.subtract(dudz, dwdx, dudz)
            numpy.subtract(dvdx, dudy, dvdx)

            if verbose:
                print(' Done!',)
                mtimer.end()

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
        dudz, dudy, dudx = numpy.gradient(vfield[:,:,:,0], self.dx[0], self.dx[1], self.dx[2])
        dvdz, dvdy, dvdx = numpy.gradient(vfield[:,:,:,1], self.dx[0], self.dx[1], self.dx[2])
        dwdz, dwdy, dwdx = numpy.gradient(vfield[:,:,:,2], self.dx[0], self.dx[1], self.dx[2])

        numpy.subtract(dwdy, dvdz, dwdy)
        numpy.subtract(dudz, dwdx, dudz)
        numpy.subtract(dvdx, dudy, dvdx)

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
            ddy = numpy.gradient(sfield, numpy.radians(self.dx[0]), axis=0)/r
            ddx = numpy.gradient(sfield, numpy.radians(self.dx[1]), axis=1)/(r*numpy.cos(numpy.radians(self.lat)))
            ddx[:,0] = numpy.gradient(numpy.concatenate([sfield[:,-1:],sfield[:,:2]],axis=1), numpy.radians(self.dx[1]), axis=1)[:,1]/(r*numpy.cos(numpy.radians(self.lat[:,0])))
            ddx[:,-1] = numpy.gradient(numpy.concatenate([sfield[:,-2:],sfield[:,:1]],axis=1), numpy.radians(self.dx[1]), axis=1)[:,1]/(r*numpy.cos(numpy.radians(self.lat[:,0])))
        else:
            #ddy, ddx = numpy.gradient(sfield, self.dx[0], self.dx[1])
            ddy = numpy.gradient(sfield, numpy.radians(self.dx[0]), axis=0)/r
            ddx = numpy.gradient(sfield, numpy.radians(self.dx[1]), axis=1)/(r*numpy.cos(numpy.radians(self.lat)))
        #
        ddy = -1.0*ddy

        grad = numpy.stack((ddy, ddx), axis=-1)

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
                ddy = numpy.gradient(sfield, numpy.radians(self.dx[0]), axis=0)/r
                ddx = numpy.gradient(sfield, numpy.radians(self.dx[1]), axis=1)/(r*numpy.cos(numpy.radians(self.lat)))
                ddx[:,0] = numpy.gradient(numpy.concatenate([sfield[:,-1:],sfield[:,:2]],axis=1), numpy.radians(self.dx[1]), axis=1)[:,1]/(r*numpy.cos(numpy.radians(self.lat[:,0])))
                ddx[:,-1] = numpy.gradient(numpy.concatenate([sfield[:,-2:],sfield[:,:1]],axis=1), numpy.radians(self.dx[1]), axis=1)[:,1]/(r*numpy.cos(numpy.radians(self.lat[:,0])))
            else:
                # self.dx = (dy,dx)
                ddy = numpy.gradient(sfield, numpy.radians(self.dx[0]), axis=0)/r
                ddx = numpy.gradient(sfield, numpy.radians(self.dx[1]), axis=1)/(r*numpy.cos(numpy.radians(self.lat)))

            grad = numpy.stack((ddx, ddy), axis = -1)

        elif self.dim == 3:
            print('not implemented yet! - will return the same as cartesian')
            # self.dx = (dz,dy,dx)
            ddz, ddy, ddx = numpy.gradient(sfield, self.dx[0], self.dx[1], self.dx[2])
            grad = numpy.stack((ddx, ddy, ddz), axis = -1)

        if verbose:
            print(' Done!',)
            mtimer.end()

        return grad

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
