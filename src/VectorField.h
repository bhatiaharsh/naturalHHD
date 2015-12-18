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

#ifndef VECTOR_FIELD_H
#define VECTOR_FIELD_H

#include <stdlib.h>
#include <string>
#include <iostream>
#include <algorithm>

#include "RW.h"
#include "RGrid.h"

typedef unsigned int uint;

/// =============================================================
/// MACROS for fundamental functionalities

// min and max elements in an array
#define minv(a,n) *std::min_element(a, a+n)
#define maxv(a,n) *std::max_element(a, a+n)

// 1D derivative
#define dvdx(vp,vm,one_over_dx) ((vp-vm)*one_over_dx)

// divergence
#define divg2D(uxp,uxm,one_over_dx,vyp,vym,one_over_dy) (dvdx(uxp,uxm,one_over_dx) + dvdx(vyp,vym,one_over_dy))
#define divg3D(uxp,uxm,one_over_dx,vyp,vym,one_over_dy,vzp,vzm,one_over_dz) (dvdx(uxp,uxm,one_over_dx) + dvdx(vyp,vym,one_over_dy) + dvdx(vzp,vzm,one_over_dz))

// rotation
#define rot2D(vxp,vxm,one_over_dx,uyp,uym,one_over_dy) (dvdx(vxp,vxm,one_over_dx) - dvdx(uyp,uym,one_over_dy))


using namespace std;

/** ---------------------------------------------------------------------------
 * VectorField.h
 * Class for storing a vector field.
 */
template <typename T = float>
class VectorField {

public:

    // --------------------------------------------------------------
    uint8_t dim;                /// dimensionality of the vector field
    size_t sz;                  /// number of vectors in the field

    T *u, *v, *w;               // vector components

    T *mgn;                     // magnitude
    T *div;                     // divergence
    T *curlu, *curlv, *curlw;   // curl components

    /// initialize a vector field
    void init() {
        sz = dim = 0;
        u = v = w = 0;
        mgn = div = 0;
        curlu = curlv = curlw = 0;
    }

    /// destructor
    ~VectorField(){

        //printf("~VectorField()\n");
        if(u)   delete []u;
        if(v)   delete []v;
        if(w)   delete []w;
        if(mgn)     delete []mgn;
        if(div)     delete []div;
        if(curlu)   delete []curlu;
        if(curlv)   delete []curlv;
        if(curlw)   delete []curlw;
        //printf("~VectorField() done!\n");
    }

    /// default constructor
    VectorField(){
        init();
    }

    /// read a vector field
    VectorField(string filename, uint dim_){

        if(dim_ != 2 & dim_ != 3){
            printf(" VectorField() -- only 2D and 3D vector fields accepted!\n");
            exit(1);
        }

        init();
        dim = dim_;                         // dimensionality of the data
        read_from_file(filename);
    }

    /// create a test vector field
    VectorField(const RGrid &rgrid){

        init();
        dim = rgrid.dim;
        sz = rgrid.sz;

        u = new T[sz];
        v = new T[sz];

        int cX = 0.5*rgrid.X;
        int cY = 0.5*rgrid.Y;

        if(dim == 2){
            uint C = rgrid.get_idx(cX, cY);

            // creating a field
            for(int y = 0; y < rgrid.Y; y++){
            for(int x = 0; x < rgrid.X; x++){

                uint p = rgrid.get_idx(x,y);

                T vx = (T)(x-cX) * rgrid.dx;
                T vy = (T)(y-cY) * rgrid.dy;

                // create a source!
                u[p] = vx;
                v[p] = vy;

                float wt = exp( (double)-2.0*rgrid.get_dist(p, C));
                if(p != C){
                    u[p] *= wt;
                    v[p] *= wt;
                }
            }
            }
        }

        else if(dim == 3){

            w = new T[sz];
            int cZ = 0.5*rgrid.Z;

            uint C = rgrid.get_idx(cX, cY, cZ);

            // creating a field
            for(int z = 0; z < rgrid.Z; z++){
            for(int y = 0; y < rgrid.Y; y++){
            for(int x = 0; x < rgrid.X; x++){

                uint p = rgrid.get_idx(x,y,z);

                T vx = (T)(x-cX) * rgrid.dx;
                T vy = (T)(y-cY) * rgrid.dy;
                T vz = (T)(z-cZ) * rgrid.dz;

                // create a source!
                u[p] = vx;
                v[p] = vy;
                w[p] = vz;

                float wt = exp( (double)-2.0*rgrid.get_dist(p, C));
                if(p != C){
                    u[p] *= wt;
                    v[p] *= wt;
                    w[p] *= wt;
                }
            }
            }
            }
        }
    }

    /// read from file
    void read_from_file(string filename){

        if(dim != 2 && dim != 3){
            printf(" VectorField::read_from_file() -- Need to know dimensionality before reading from file!\n");
            exit(1);
        }

        T* data;                            // data to be read
        size_t num_vals_read = 0;           // num of values read

        data = RW::read_binary<T>(filename, num_vals_read, true);
        if(!data){
            printf(" VectorField::read_from_file() -- failed to read the data!\n");
            exit(1);
        }

        /// ------ store the read data in (u, v, w) format
        sz = num_vals_read / dim;       // num of vectors

        printf(" Initializing a %zdD vector field of size %zd...", dim, sz);
        fflush(stdout);

        u = new T[sz];
        v = new T[sz];

        for(uint i = 0; i < sz; i++){
            u[i] = data[dim*i];
            v[i] = data[dim*i+1];
        }

        if(dim == 3){
            w = new T[sz];
            for(uint i = 0; i < sz; i++){
                w[i] = data[dim*i+2];
            }
        }
        printf(" Done!\n");
        delete[] data;
    }

    /// write to file
    void write_to_file(string filename) const{

        T* data = new T[dim*sz];

        if(dim == 2){
            for(uint i = 0; i < sz; i++){
                data[dim*i] = u[i];
                data[dim*i+1] = v[i];

                //printf(" %f %f\n", data[dim*i], data[dim*i+1]);
            }
        }
        else if(dim == 3){
            for(uint i = 0; i < sz; i++){
                data[dim*i] = u[i];
                data[dim*i+1] = v[i];
                data[dim*i+2] = w[i];
            }
        }

        RW::write_binary<T>(filename, data, dim*sz, true);
        delete[] data;
    }

private:

    bool match_grid(const RGrid &rgrid, std::string tag = "") const{

        if(rgrid.sz == this->sz && rgrid.dim == this->dim)
            return true;

        // else break!
        printf(" %s -- Incompatible grid. (size, dim) -- (%zd,%zd) != (%zd,%zd)\n", tag.c_str(),
               sz, dim, rgrid.sz, rgrid.dim);
        exit(1);
    }

public:

    /// compute magnitudes
    void need_magnitudes(const RGrid &rgrid){

        match_grid(rgrid, "VectorField::need_magnitudes()");

        printf(" -- Computing magnitudes...");
        fflush(stdout);
        mgn = new T[sz];

        if(dim == 2){

            #pragma omp parallel for
            for(size_t i = 0; i < sz; i++){
                mgn[i] = sqrt(u[i]*u[i] + v[i]*v[i]);
            }
        }
        else if(dim == 3){

            #pragma omp parallel for
            for(size_t i = 0; i < sz; i++){
                mgn[i] = sqrt(u[i]*u[i] + v[i]*v[i] + w[i]*w[i]);
            }
        }
        printf(" Done!\n");
    }

    /// compute divergence and curl
    void need_divcurl(const RGrid &rgrid){

        match_grid(rgrid, "VectorField::need_divcurl()");

        printf(" -- Computing %zdD divergence and curl...", dim);
        fflush(stdout);

        if(dim == 2){

            div = new T[sz];
            curlw = new T[sz];

            #pragma omp parallel for
            for(size_t p = 0; p < rgrid.sz; p++){

                size_t pxp, pxm, pyp, pym;
                float dx, dy;

                rgrid.get_nbrs_idx_2D(p, pxp, pxm, dx, pyp, pym, dy);

                div[p] = divg2D(u[pxp], u[pxm], dx, v[pyp], v[pym], dy);
                curlw[p] = rot2D(v[pxp], v[pxm], dx, u[pyp], u[pym], dy);
            }
        }

        else if(dim == 3){

            div = new T[sz];
            curlu = new T[sz];
            curlv = new T[sz];
            curlw = new T[sz];

            #pragma omp parallel for
            for(size_t p = 0; p < rgrid.sz; p++){

                size_t pxp, pxm, pyp, pym, pzp, pzm;
                float dx, dy, dz;

                rgrid.get_nbrs_idx_3D(p, pxp, pxm, dx, pyp, pym, dy, pzp, pzm, dz);

                div[p]  = divg3D(u[pxp], u[pxm], dx, v[pyp], v[pym], dy, w[pzp], w[pzm], dz);
                curlu[p] = rot2D(w[pyp], w[pym], dy, v[pzp], v[pzm], dz);
                curlv[p] = rot2D(u[pzp], u[pzm], dz, w[pxp], w[pxm], dx);
                curlw[p] = rot2D(v[pxp], v[pxm], dx, u[pyp], u[pym], dy);
            }
        }

        printf(" Done!\n");
    }

    /// rotate a vector field by 2pi
    void rotate_J(){

        if(dim != 2){
            printf(" VectorField::rotate_J() works for 2D fields only!\n");
            exit(1);
        }

        printf(" -- Rotating 2D vector field...");
        fflush(stdout);
        for(size_t i = 0; i < sz; i++){
            float t = u[i];
            u[i] = -v[i];
            v[i] = t;
        }
        printf(" Done!\n");
    }

    /// show stats
    void show_stats(const std::string &tag) const {
        /*printf("\n --%s--\n", tag.c_str());
        if(mgn != 0)    printf(" \t magn [%.3f, %.3f]\n", minv(mgn,sz), maxv(mgn,sz));
        if(div != 0)    printf(" \t div  [%.3f, %.3f]\n", minv(div,sz), maxv(div,sz));
        if(curlw != 0)  printf(" \t curl [%.3f, %.3f]\n", minv(curlw,sz), maxv(curlw,sz));*/
    }

    /** @brief Compute a vector field as a gradient of a scalar field.
     *
     *  @param f: A scalar field
     *  @param rgrid: Regular grid on which the fields are defined
     */
    void compute_as_gradient_field(const T* f, const RGrid &rgrid){

        printf(" -- Computing gradient ...");
        fflush(stdout);

        dim = rgrid.dim;
        sz = rgrid.sz;

        // ---------------------------------
        // now compute the gradient field
        if(dim == 2){

            u = new T[sz];
            v = new T[sz];

            #pragma omp parallel for
            for(size_t p = 0; p < sz; p++){

                size_t pxp, pxm, pyp, pym;
                float dx, dy;

                rgrid.get_nbrs_idx_2D(p, pxp, pxm, dx, pyp, pym, dy);

                u[p] = dvdx(f[pxp], f[pxm], dx);
                v[p] = dvdx(f[pyp], f[pym], dy);
            }
        }

        else if(dim == 3){

            u = new T[sz];
            v = new T[sz];
            w = new T[sz];

            #pragma omp parallel for
            for(size_t p = 0; p < sz; p++){

                size_t pxp, pxm, pyp, pym, pzp, pzm;
                float dx, dy, dz;

                rgrid.get_nbrs_idx_3D(p, pxp, pxm, dx, pyp, pym, dy, pzp, pzm, dz);

                u[p] = (f[pxp]-f[pxm]) * dx;
                v[p] = (f[pyp]-f[pym]) * dy;
                w[p] = (f[pzp]-f[pzm]) * dz;
            }
        }
        printf(" Done!\n");
    }

    /** @brief Compute a vector field as a curl of a scalar field.
     *
     *  @param v: three components of a vector field
     *  @param rgrid: 3D regular grid on which the fields are defined
     */
    void compute_as_curl_field(const T* vu, const T* vv, const T* vw, const RGrid &rgrid){

        if(rgrid.dim != 3){
            printf(" VectorField::compute_as_curl_field() works for 3D fields only!\n");
            exit(1);
        }

        dim = rgrid.dim;
        sz = rgrid.sz;

        printf(" -- Computing curl field...");
        fflush(stdout);

        u = new T[sz];
        v = new T[sz];
        w = new T[sz];

        #pragma omp parallel for
        for(size_t p = 0; p < sz; p++){

            size_t pxp, pxm, pyp, pym, pzp, pzm;
            float one_over_dx, one_over_dy, one_over_dz;

            rgrid.get_nbrs_idx_3D(p, pxp, pxm, one_over_dx, pyp, pym, one_over_dy, pzp, pzm, one_over_dz);

            u[p] = rot2D(vw[pyp], vw[pym], one_over_dy, vv[pzp], vv[pzm], one_over_dz);
            v[p] = rot2D(vu[pzp], vu[pzm], one_over_dz, vw[pxp], vw[pxm], one_over_dx);
            w[p] = rot2D(vv[pxp], vv[pxm], one_over_dx, vu[pyp], vu[pym], one_over_dy);
        }
        printf(" Done!\n");
    }

    /** @brief Compute a harmonic field.
     */
    void compute_as_harmonic_field(const VectorField<T> &vf, const VectorField<T> &d, const VectorField<T> &r){

        if(vf.dim != d.dim || vf.dim != r.dim){
            printf(" Mismatch in field dimensions. Cannot compute harmonic!\n");
            exit(1);
        }
        if(vf.sz != d.sz || vf.sz != r.sz){
            printf(" Mismatch in field sizes. Cannot compute harmonic!\n");
            exit(1);
        }

        printf(" -- Computing harmonic ...");
        fflush(stdout);

        dim = vf.dim;
        sz = vf.sz;

        if(dim == 2){
            u = new T[sz];
            v = new T[sz];

            for(size_t i = 0; i < sz; i++){
                u[i] = vf.u[i] - d.u[i] - r.u[i];
                v[i] = vf.v[i] - d.v[i] - d.v[i];
            }
        }

        else if(dim == 3){

            u = new T[sz];
            v = new T[sz];
            w = new T[sz];

            for(size_t i = 0; i < sz; i++){
                u[i] = vf.u[i] - d.u[i] - r.u[i];
                v[i] = vf.v[i] - d.v[i] - d.v[i];
                w[i] = vf.w[i] - d.w[i] - r.w[i];
            }
        }

        printf(" Done!\n");
    }

    /// ==================================================================
    /// gradient and curl operators
    /*static VectorField<T> compute_gradient(const T* f, const RGrid &rgrid){

        printf(" -- Computing gradient ...");
        fflush(stdout);

        VectorField<T> gf;

        gf.dim = rgrid.dim;
        gf.sz = rgrid.sz;

        /// ---------------------------------
        /// now compute the gradient field

        if(gf.dim == 2){

            gf.u = new T[gf.sz];
            gf.v = new T[gf.sz];

            #pragma omp parallel for
            for(uint p = 0; p < gf.sz; p++){

                size_t pxp, pxm, pyp, pym;
                float dx, dy;

                rgrid.get_nbrs_idx_2D(p, pxp, pxm, dx, pyp, pym, dy);

                gf.u[p] = dvdx(f[pxp], f[pxm], dx);
                gf.v[p] = dvdx(f[pyp], f[pym], dy);
            }
        }

        else if(gf.dim == 3){

            gf.u = new T[gf.sz];
            gf.v = new T[gf.sz];
            gf.w = new T[gf.sz];

            #pragma omp parallel for
            for(uint p = 0; p < gf.sz; p++){

                size_t pxp, pxm, pyp, pym, pzp, pzm;
                float dx, dy, dz;

                rgrid.get_nbrs_idx_3D(p, pxp, pxm, dx, pyp, pym, dy, pzp, pzm, dz);

                gf.u[p] = (f[pxp]-f[pxm]) * dx;
                gf.v[p] = (f[pyp]-f[pym]) * dy;
                gf.w[p] = (f[pzp]-f[pzm]) * dz;
            }
        }
        printf(" Done!\n");
        return gf;
    }

    static void compute_gradient(const T* f, const RGrid &rgrid, VectorField<T> &gf){

        printf(" -- Computing gradient ...");
        fflush(stdout);

        gf.dim = rgrid.dim;
        gf.sz = rgrid.sz;

        /// ---------------------------------
        /// now compute the gradient field

        if(gf.dim == 2){

            gf.u = new T[gf.sz];
            gf.v = new T[gf.sz];

            #pragma omp parallel for
            for(uint p = 0; p < gf.sz; p++){

                size_t pxp, pxm, pyp, pym;
                float dx, dy;

                rgrid.get_nbrs_idx_2D(p, pxp, pxm, dx, pyp, pym, dy);

                gf.u[p] = dvdx(f[pxp], f[pxm], dx);
                gf.v[p] = dvdx(f[pyp], f[pym], dy);
            }
        }

        else if(gf.dim == 3){

            gf.u = new T[gf.sz];
            gf.v = new T[gf.sz];
            gf.w = new T[gf.sz];

            #pragma omp parallel for
            for(uint p = 0; p < gf.sz; p++){

                size_t pxp, pxm, pyp, pym, pzp, pzm;
                float dx, dy, dz;

                rgrid.get_nbrs_idx_3D(p, pxp, pxm, dx, pyp, pym, dy, pzp, pzm, dz);

                gf.u[p] = (f[pxp]-f[pxm]) * dx;
                gf.v[p] = (f[pyp]-f[pym]) * dy;
                gf.w[p] = (f[pzp]-f[pzm]) * dz;
            }
        }
        printf(" Done!\n");
    }

    static VectorField<T> compute_curl(const T* vu, const T* vv, const T* vw, const RGrid &rgrid){

        if(rgrid.dim != 3){
            printf(" VectorField::compute_curl() works for 3D fields only!\n");
            exit(1);
        }

        VectorField<T> cf;

        cf.dim = rgrid.dim;
        cf.sz = rgrid.sz;

        printf(" -- Computing curl...");
        fflush(stdout);

        cf.u = new T[cf.sz];
        cf.v = new T[cf.sz];
        cf.w = new T[cf.sz];

        #pragma omp parallel for
        for(uint p = 0; p < cf.sz; p++){

            size_t pxp, pxm, pyp, pym, pzp, pzm;
            float one_over_dx, one_over_dy, one_over_dz;

            rgrid.get_nbrs_idx_3D(p, pxp, pxm, one_over_dx, pyp, pym, one_over_dy, pzp, pzm, one_over_dz);

            cf.u[p] = rot2D(vw[pyp], vw[pym], one_over_dy, vv[pzp], vv[pzm], one_over_dz);
            cf.v[p] = rot2D(vu[pzp], vu[pzm], one_over_dz, vw[pxp], vw[pxm], one_over_dx);
            cf.w[p] = rot2D(vv[pxp], vv[pxm], one_over_dx, vu[pyp], vu[pym], one_over_dy);
        }
        printf(" Done!\n");
        return cf;
    }

    static void compute_curl(const T* vu, const T* vv, const T* vw, const RGrid &rgrid, VectorField<T> &cf){

        if(rgrid.dim != 3){
            printf(" VectorField::compute_curl() works for 3D fields only!\n");
            exit(1);
        }

        cf.dim = rgrid.dim;
        cf.sz = rgrid.sz;

        printf(" -- Computing curl...");
        fflush(stdout);

        cf.u = new T[cf.sz];
        cf.v = new T[cf.sz];
        cf.w = new T[cf.sz];

        #pragma omp parallel for
        for(uint p = 0; p < cf.sz; p++){

            size_t pxp, pxm, pyp, pym, pzp, pzm;
            float one_over_dx, one_over_dy, one_over_dz;

            rgrid.get_nbrs_idx_3D(p, pxp, pxm, one_over_dx, pyp, pym, one_over_dy, pzp, pzm, one_over_dz);

            cf.u[p] = rot2D(vw[pyp], vw[pym], one_over_dy, vv[pzp], vv[pzm], one_over_dz);
            cf.v[p] = rot2D(vu[pzp], vu[pzm], one_over_dz, vw[pxp], vw[pxm], one_over_dx);
            cf.w[p] = rot2D(vv[pxp], vv[pxm], one_over_dx, vu[pyp], vu[pym], one_over_dy);
        }
        printf(" Done!\n");
    }*/
    /// ==================================================================
    /// add and subtract operators
    /*VectorField<T> operator+(const VectorField<T> &toadd) const{

        if(dim != toadd.dim || sz != toadd.sz){
            printf(" Mismatch in argument fields. Cannot add!\n");
            exit(1);
        }

        VectorField<T> c;
        c.dim = dim;
        c.sz = sz;

        c.u = new T[sz];
        c.v = new T[sz];

        if(dim == 2){

            for(uint i = 0; i < sz; i++){
                c.u[i] = u[i] + toadd.u[i];
                c.v[i] = v[i] + toadd.v[i];
            }
        }

        else if(dim == 3){

            c.w = new T[c.sz];

            for(uint i = 0; i < sz; i++){
                c.u[i] = u[i] + toadd.u[i];
                c.v[i] = v[i] + toadd.v[i];
                c.w[i] = w[i] + toadd.w[i];
            }
        }

        return c;
    }

    VectorField<T> operator-(const VectorField<T> &toadd) const{

        if(dim != toadd.dim || sz != toadd.sz){
            printf(" Mismatch in argument fields. Cannot add!\n");
            exit(1);
        }

        VectorField<T> c;
        c.dim = dim;
        c.sz = sz;

        c.u = new T[sz];
        c.v = new T[sz];

        if(dim == 2){

            for(uint i = 0; i < sz; i++){
                c.u[i] = u[i] - toadd.u[i];
                c.v[i] = v[i] - toadd.v[i];
            }
        }

        else if(dim == 3){

            c.w = new T[c.sz];

            for(uint i = 0; i < sz; i++){
                c.u[i] = u[i] - toadd.u[i];
                c.v[i] = v[i] - toadd.v[i];
                c.w[i] = w[i] - toadd.w[i];
            }
        }

        return c;
    }*/
};


#endif
