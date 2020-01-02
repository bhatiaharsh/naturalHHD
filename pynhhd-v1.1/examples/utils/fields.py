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
import logging
LOGGER = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
def subsample(vfield, factor):
    return vfield[::factor,::factor,:]

# ------------------------------------------------------------------------------
# add a critical point to a field on regular grid
def add_criticalPoint2D(vfield, spacings, gpos, type, direction, sigma, scaling, decay_type):

    (y, x, d) = vfield.shape
    (dy, dx) = spacings
    (cy, cx) = gpos

    LOGGER.info('adding critical point: vfield {} {} {}'.format(vfield.shape, spacings, gpos))

    #decay = 1.0 / sigma
    for iy in range(0, y):
        for ix in range(0, x):

            dx1 = (ix-cx)*dx
            dy1 = (iy-cy)*dy

            r2 = (dx1*dx1 + dy1*dy1)
            r = np.sqrt(dx1*dx1 + dy1*dy1)

            #if decay_type == 0:
            #e = np.exp(- decay * np.sqrt(r2))
            e = 1
            '''
            else:
                if r2 > 1.0:
                    e = 1.0 / ((r/2.0)*(r/2.0))
                else:
                    e = 2.0 - r/2.0
            '''
            # orbit
            if type == 1:
                vfield[iy,ix,0] += -1*direction * scaling * dy1 * e
                vfield[iy,ix,1] +=    direction * scaling * dx1 * e

            # source/sink
            elif type == 2:
                vfield[iy,ix,0] += direction * dx1
                vfield[iy,ix,1] += direction * dy1

            # saddle
            elif type == 3:
                vfield[iy,ix,0] += direction * dy1
                vfield[iy,ix,1] += direction * dx1

    return vfield

def add_criticalPoint3D(vfield, spacings, gpos, type, direction, decay, scaling):

    (z, y, x, d) = vfield.shape
    (dz, dy, dx) = spacings
    (cz, cy, cx) = gpos

    LOGGER.info('adding critical point: vfield {} {} {}'.format(vfield.shape, spacings, gpos))

    for iz in range(0, z):
        for iy in range(0, y):
            for ix in range(0, x):

                dx1 = (ix-cx)*dx
                dy1 = (iy-cy)*dy
                dz1 = (iz-cz)*dz

                #e = np.exp(-decay * np.sqrt( dx1*dx1 + dy1*dy1 ))
                vfield[iz,iy,ix,0] += direction * dx1
                vfield[iz,iy,ix,1] += direction * dy1
                vfield[iz,iy,ix,2] += direction * dz1
    return vfield

# ------------------------------------------------------------------------------
# create a vector field critical point on a set of points
def create_criticalPoint2D(points, gpos, type, direction, decay, scaling):

    vfield = points - gpos

    if type == 1:
        vfield[:,[0,1]] = vfield[:,[1,0]]
        vfield[:,1] *= -1.0


    #    return vfield
    for i in range(points.shape[0]):

        e = np.exp(-decay * np.linalg.norm(vfield[i]))

        # orbit
        if type == 1:
            #vfield[:,[0,1]] = vfield[:,[1,0]]
            #vfield[:,1] *= -1.0

            vfield[i,0] *= direction * e * scaling
            vfield[i,1] *= direction * e * scaling

            #vfield[i,0] += -1*direction * vfield[i,1] * e * scaling
            #vfield[i,1] +=    direction * vfield[i,0] * e * scaling

        # source/sink
        elif type == 2:
            vfield[i,0] += direction * vfield[i,0]
            vfield[i,1] += direction * vfield[i,1]

        # saddle
        elif type == 3:
            vfield[i,0] += direction * vfield[i,1]
            vfield[i,1] += direction * vfield[i,1]

    return vfield
    '''
    # source
    if type == 1:
        vfield[:,0], vfield[:,1] = vfield[:,1], -vfield[:,0]
        return direction * vfield

    # source
    elif type == 2:
        return direction * vfield
    '''

# ------------------------------------------------------------------------------
