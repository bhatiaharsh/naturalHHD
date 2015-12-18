/*
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
*/

#ifndef _RGRID_H_
#define _RGRID_H_

#include <math.h>
#include <cstddef>
#include <cstdint>

/** ---------------------------------------------------------------------------
 * RGrid.h
 *  This file contains a structure to represent regular grids.
 *      This has centralized functionalities to obtain neighbors etc.
 */
struct RGrid {

public:
    uint8_t dim;    /*!< 2D or 3D */
    size_t sz;      /*!< size of the grid*/

    size_t X, Y, Z;
    float dx, dy, dz;

private:

    // save these quantities to minimize multiplication on the fly
    size_t XY;

    float sq_dx, sq_dy, sq_dz;
    float one_over_dx;
    float one_over_dy;
    float one_over_dz;

public:

    /// default constructor
    RGrid(){
        init();
    }

    /// constructor for 1D grid
    RGrid(size_t X_, float dx_){
        init();

        dim = 1;
        X = X_;
        dx = dx_;
        sz = X;

        init_local();
    }

    /// constructor for 2D grid
    RGrid(size_t X_, size_t Y_, float dx_, float dy_){
        init();

        dim = 2;
        X = X_;         Y = Y_;
        dx = dx_;       dy = dy_;
        sz = X*Y;

        if(dim == 3)        printf(" Created 3D grid [%zd x %zd x %zd] with spacings %.3f %.3f %.3f\n", X, Y, Z, dx, dy, dz);
        else if(dim == 2)   printf(" Created 2D grid [%zd x %zd] with spacings %.3f %.3f\n", X, Y, dx, dy);

        init_local();
    }

    /// constructor for 3D grid
    RGrid(size_t X_, size_t Y_, size_t Z_, float dx_, float dy_, float dz_){
        init();

        dim = 3;
        X = X_;         Y = Y_;         Z = Z_;
        dx = dx_;       dy = dy_;       dz = dz_;
        sz = X*Y*Z;

        init_local();
    }

private:
    /// initialize the grid
    void init(){

        dim = 0;        sz = 0;
        X = 0;          Y = 0;          Z = 0;
        dx = 0.0;       dy = 0.0;       dz = 0.0;

        XY = 0;
        sq_dx = 0.0;    one_over_dx = 0.0;
        sq_dy = 0.0;    one_over_dy = 0.0;
        sq_dz = 0.0;    one_over_dz = 0.0;
    }
    void init_local(){

        sq_dx = dx*dx;      one_over_dx = 1.0/dx;
        if(dim == 1)        return;

        sq_dy = dy*dy;      one_over_dy = 1.0/dy;
        XY = X*Y;
        if(dim == 2)        return;

        sq_dz = dz*dz;      one_over_dz = 1.0/dz;
    }

    static void get_nbrs(size_t X_, float one_over_dx_, size_t x, size_t &xm, size_t &xp, float &dx_){
        xm = (x == 0) ? x : x-1;
        xp = (x == X_-1) ? x : x+1;
        dx_ = one_over_dx_ / (xp - xm);
    }

public:

    // -------------------------------------------------------------------
    /// convert between indices and vertex id
    inline void get_coords(size_t v, size_t &x, size_t &y) const{
        x = v%X;    y = v/X;
    }

    /// convert between indices and vertex id
    inline void get_coords(size_t v, size_t &x, size_t &y, size_t &z) const{
        z = v/XY;   get_coords(v%XY, x, y);
    }

    /// convert between indices and vertex id
    inline size_t get_idx(size_t x, size_t y) const{
        return y*X + x;
    }

    /// convert between indices and vertex id
    inline size_t get_idx(size_t x, size_t y, size_t z) const{
        return z*XY + y*X + x;
    }

    /// -------------------------------------------------------------------
    /// get neighbors of a vertex p
    /// pxp, pxm are x-neighbors to the right (plus) and left (minus)
    /// and so on for y and z dimensions

    /** @brief Get neighbors of a vertex p.
    *
    *  @param p is the input point
    *  @param pxp is the x-neighbor to the right (plus)
    *  @param pxm is the x-neighbor to the left (minus)
    *  @param pyp is the y-neighbor to the right (plus)
    *  @param pym is the y-neighbor to the left (minus)
    *  @param one_over_dx_ is the reciprocal of the dx corresponding to this neighbor
    *  @param one_over_dy_ is the reciprocal of the dy corresponding to this neighbor
    */
    void get_nbrs_idx_2D(size_t p, size_t &pxp, size_t &pxm, float &one_over_dx_,
                                   size_t &pyp, size_t &pym, float &one_over_dy_) const{

        size_t x, y;
        get_coords(p, x, y);

        // find nbring indices
        size_t xm, xp;
        size_t ym, yp;

        get_nbrs(X, one_over_dx, x, xm, xp, one_over_dx_);
        get_nbrs(Y, one_over_dy, y, ym, yp, one_over_dy_);

        // compute indices
        pxp = get_idx(xp, y);
        pxm = get_idx(xm, y);
        pyp = get_idx(x, yp);
        pym = get_idx(x, ym);
    }

    /** @brief Get neighbors of a vertex p.
    *
    *  @param p is the input point
    *  @param pxp is the x-neighbor to the right (plus)
    *  @param pxm is the x-neighbor to the left (minus)
    *  @param pyp is the y-neighbor to the right (plus)
    *  @param pym is the y-neighbor to the left (minus)
    *  @param pzp is the z-neighbor to the right (plus)
    *  @param pzm is the z-neighbor to the left (minus)
    *  @param one_over_dx_ is the reciprocal of the dx corresponding to this neighbor
    *  @param one_over_dx_ is the reciprocal of the dx corresponding to this neighbor
    *  @param one_over_dz_ is the reciprocal of the dz corresponding to this neighbor
    */
    void get_nbrs_idx_3D(size_t p, size_t &pxp, size_t &pxm, float &one_over_dx_,
                                   size_t &pyp, size_t &pym, float &one_over_dy_,
                                   size_t &pzp, size_t &pzm, float &one_over_dz_) const{

        size_t x, y, z;
        get_coords(p, x, y, z);

        // find nbring indices
        size_t xm, xp;
        size_t ym, yp;
        size_t zm, zp;

        get_nbrs(X, one_over_dx, x, xm, xp, one_over_dx_);
        get_nbrs(Y, one_over_dy, y, ym, yp, one_over_dy_);
        get_nbrs(Z, one_over_dz, z, zm, zp, one_over_dz_);

        // compute indices
        pxp = get_idx(xp, y, z);
        pxm = get_idx(xm, y, z);
        pyp = get_idx(x, yp, z);
        pym = get_idx(x, ym, z);
        pzp = get_idx(x, y, zp);
        pzm = get_idx(x, y, zm);
    }


    /// Compute square of the distance on the regular grid
    float get_dist_sq(size_t v1, size_t v2) const{

        if(dim == 1){
            long int dist = (v1-v2);
            return dist*dist*sq_dx;
        }
        if(dim == 2){

            size_t x1, y1, x2, y2;
            get_coords(v1, x1, y1);
            get_coords(v2, x2, y2);

            long int distx = (x1-x2);
            long int disty = (y1-y2);

            return (distx*distx*sq_dx) + (disty*disty*sq_dy);
        }
        if(dim == 3){

            size_t x1, y1, z1, x2, y2, z2;
            get_coords(v1, x1, y1, z1);
            get_coords(v2, x2, y2, z2);

            long int distx = (x1-x2);
            long int disty = (y1-y2);
            long int distz = (z1-z2);

            return (distx*distx*sq_dx) + (disty*disty*sq_dy) + (distz*distz*sq_dz);
        }

        return 0.0;
    }

    /// Compute the distance on the regular grid
    inline float get_dist(size_t v1, size_t v2) const{
        return sqrt(get_dist_sq(v1,v2));
    }
};

#endif
