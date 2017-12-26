
import numpy

# ------------------------------------------------------------------------------
def subsample(vfield, factor):

    return vfield[::factor,::factor,:]
    '''
    X = vfield.shape[1]
    Y = vfield.shape[0]

    vfield2 = numpy.zeros((Y/factor, X/factor, 2), dtype=numpy.float32)

    vfield2[:,:,0] = vfield[::factor,::factor,0]
    vfield2[:,:,1] = vfield[::factor,::factor,1]

    return vfield2
    '''

# ------------------------------------------------------------------------------
# add a critical point to a field on regular grid
def add_criticalPoint2D(vfield, spacings, gpos, type, direction, sigma, scaling, decay_type):

    (y, x, d) = vfield.shape
    (dy, dx) = spacings
    (cy, cx) = gpos

    print 'adding critical point: vfield', vfield.shape, spacings, gpos

    #decay = 1.0 / sigma
    for iy in range(0, y):
        for ix in range(0, x):

            dx1 = (ix-cx)*dx
            dy1 = (iy-cy)*dy

            r2 = (dx1*dx1 + dy1*dy1)
            r = numpy.sqrt(dx1*dx1 + dy1*dy1)

            #if decay_type == 0:
            #e = numpy.exp(- decay * numpy.sqrt(r2))
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

    print 'adding critical point: vfield', vfield.shape, spacings, gpos

    for iz in range(0, z):
        for iy in range(0, y):
            for ix in range(0, x):

                dx1 = (ix-cx)*dx
                dy1 = (iy-cy)*dy
                dz1 = (iz-cz)*dz

                #e = numpy.exp(-decay * numpy.sqrt( dx1*dx1 + dy1*dy1 ))
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


    print gpos, type, direction, decay, scaling

    for i in xrange(points.shape[0]):

        e = numpy.exp(-decay * numpy.linalg.norm(vfield[i]))

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
