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

#ifndef NHHD_H
#define NHHD_H

#include <stdlib.h>
#include <math.h>

#include "RGrid.h"
#include "VectorField.h"
#include "trapezoidalIntegration.h"
#include "Poisson.h"


/**
 * HHD.h
 *  Library for computing the potential functions for the natural HHD.
 *  - Technical details can be found in the paper doi: 10.1109/TVCG.2014.2312012
 */
template <typename T>
class naturalHHD {

public:

    T *D;   /*!< Divegence potential */
    T *Ru;  /*!< u-component of Rotation potential */
    T *Rv;  /*!< v-component of Rotation potential */
    T *Rw;  /*!< w-component of Rotation potential */

    /// Destructor
    ~naturalHHD(){
        if(D)   delete []D;
        if(Ru)  delete []Ru;
        if(Rv)  delete []Rv;
        if(Rw)  delete []Rw;
    }

    // 1 -- compute all potentials in a single loop
    // 2 -- compute all potentials in separate loops
    // 3 -- compute all potentials in separate loops using Greens function map
    /** @brief Initialize the naturalHHD.
     *
     *  Create Green's function for a 2D or 3D regular grid.
     *
     *  @param vfield: Vector field.
     *  @param rgrid: Reference to the regular grid.
     *  @param COMPUTE_TYPE: 1 (compute all potentials in a single loop),
     *                       2 (compute all potentials in separate loops),
     *                       3 (compute all potentials in separate loops using Greens function map)
     *
     */
    naturalHHD(const VectorField<T> &vfield, const RGrid &rgrid, unsigned int COMPUTE_TYPE = 1){

        if(vfield.dim != rgrid.dim || vfield.sz != rgrid.sz) {
            printf(" HHD_potentials -- mismatch in size! vfield = (%zd, %ld), rgrid = (%zd, %ld)\n",
                       vfield.dim, vfield.sz, rgrid.dim, rgrid.sz);
            exit(1);
        }

        D = 0;  Ru = 0; Rv = 0; Rw = 0;

        if(COMPUTE_TYPE == 1){

            if(vfield.dim == 2)         compute_2D(vfield.div, vfield.curlw, rgrid);
            else if(vfield.dim == 3)    compute_3D(vfield.div, vfield.curlu, vfield.curlv, vfield.curlw, rgrid);
        }

        else if(COMPUTE_TYPE == 2){

            if(vfield.dim == 2){
                D = Poisson::solve_Poisson_2D(vfield.div, rgrid);
                Ru = Poisson::solve_Poisson_2D(vfield.curlw, rgrid);
            }

            else if(vfield.dim == 3){

                D = Poisson::solve_Poisson_3D(vfield.div, rgrid);
                Ru = Poisson::solve_Poisson_3D(vfield.curlu, rgrid);
                Rv = Poisson::solve_Poisson_3D(vfield.curlv, rgrid);
                Rw = Poisson::solve_Poisson_3D(vfield.curlw, rgrid);
            }
        }

        else if(COMPUTE_TYPE == 3){

            GreenFunc<float> map(rgrid);

            if(vfield.dim == 2){
                D = Poisson::solve_Poisson_2D(vfield.div, rgrid, map);
                Ru = Poisson::solve_Poisson_2D(vfield.curlw, rgrid, map);
            }

            else if(vfield.dim == 3){

                D = Poisson::solve_Poisson_3D(vfield.div, rgrid, map);
                Ru = Poisson::solve_Poisson_3D(vfield.curlu, rgrid, map);
                Rv = Poisson::solve_Poisson_3D(vfield.curlv, rgrid, map);
                Rw = Poisson::solve_Poisson_3D(vfield.curlw, rgrid, map);
            }
        }

        if(vfield.dim == 3){
            for(size_t i = 0; i < rgrid.sz; i++){
                Ru[i] *= -1;    Rv[i] *= -1;    Rw[i] *= -1;
            }
        }
    }

private:
    /** @brief Compute 2D potentials.
     *
     *  @param fd: Divergence of the vector field.
     *  @param fr: Rotation of the vector field.
     *  @param rgrid: Regular grid.
     */
    void compute_2D(const T* fd, const T* fr, const RGrid &rgrid){

        if(rgrid.dim != 2){
            printf(" HHD_potentials::compute_2D requires a 2D regular grid!\n");
            exit(1);
        }

        printf(" Solving 2D natural HHD...");
        fflush(stdout);

        clock_t tic = clock();

        // variables for showing progress
        float percent_factor = 100.0f / (float)rgrid.sz;
        float percent_unit = 0.1;                           // show progress at this percent work
        uint progress_unit = std::max((uint)1, (uint) (percent_unit / percent_factor));

        //printf(" %f %f %d\n", percent_factor, percent_unit, progress_unit);
        // -------------------------------------------------------------------
        // the 2D Green's function is ---- 1/(2pi) log( dist(x,x_0) )
            // instead, I use ------------ 1/(4pi) log( dist^2(x,x_0) )
        static T Greens_factor = 0.25/M_PI;

        // output and temporary memory

        D = new T[rgrid.sz];
        Ru = new T[rgrid.sz];

        T* fdt = new T[rgrid.sz];
        T* frt = new T[rgrid.sz];

        // -------------------------------------------------------------------
        // compute at v0
        for(size_t v0 = 0; v0 < rgrid.sz; v0++){

            // populate the temporary field for the entire grid!
            #pragma omp parallel for
            for(size_t v = 0; v < rgrid.sz; v++){

                // don't want to compute log (0)
                T dist = (v == v0) ? (0.01 * (rgrid.dx+rgrid.dy)) :
                                     rgrid.get_dist_sq(v, v0);

                fdt[v] = (log(dist) * fd[v]);
                frt[v] = (log(dist) * fr[v]);
            }

            // integrate
            D[v0] = Greens_factor * integ::trapezoidal_2D(fdt, rgrid.X, rgrid.Y, rgrid.dx, rgrid.dy);
            Ru[v0] = Greens_factor * integ::trapezoidal_2D(frt, rgrid.X, rgrid.Y, rgrid.dx, rgrid.dy);

            if(v0 % progress_unit == 0){
                printf("\r Solving 2D natural HHD... %.1f %%", (float)v0*percent_factor);
                fflush(stdout);
            }
        }

        delete []fdt;
        delete []frt;

        double sec = (double)(clock() - tic)/CLOCKS_PER_SEC;
        if(sec < 1)     printf("\r Solving 2D natural HHD... Done! in %.3f msec.\n", 1000.0*sec);
        else            printf("\r Solving 2D natural HHD... Done! in %.3f sec.\n", sec);
    }

    /** @brief Compute 3D potentials.
     *
     *  @param fd: Divergence of the vector field.
     *  @param fr: Rotation of the vector field.
     *  @param rgrid: Regular grid.
     */
    void compute_3D(const T* fd, const T* fru, const T* frv, const T* frw, const RGrid &rgrid){

        if(rgrid.dim != 3){
            printf(" HHD_potentials::compute_3D requires a 3D regular grid!\n");
            exit(1);
        }

        printf(" Solving 3D natural HHD...");
        fflush(stdout);

        clock_t tic = clock();

        // variables for showing progress
        float percent_factor = 100.0f / (float)rgrid.sz;
        float percent_unit = 0.1;                           // show progress at this percent work
        uint progress_unit = std::max((uint)1, (uint) (percent_unit / percent_factor));

        //printf(" %f %f %d\n", percent_factor, percent_unit, progress_unit);
        // -------------------------------------------------------------------
        // the 3D Green's function is ---- 1/(4pi) dist(x,x_0)
        T Greens_factor =  -0.25/M_PI;

        // output and temporary memory

        D = new T[rgrid.sz];
        Ru = new T[rgrid.sz];
        Rv = new T[rgrid.sz];
        Rw = new T[rgrid.sz];

        T* fdt = new T[rgrid.sz];
        T* frut = new T[rgrid.sz];
        T* frvt = new T[rgrid.sz];
        T* frwt = new T[rgrid.sz];

        /// -------------------------------------------------------------------
        // compute at v0
        for(size_t v0 = 0; v0 < rgrid.sz; v0++){

            // populate the temporary field for the entire grid!
            #pragma omp parallel for
            for(size_t v = 0; v < rgrid.sz; v++){

                // don't want to compute 1 / 0
                T dist = (v == v0) ? (0.01 * (rgrid.dx+rgrid.dy+rgrid.dz)) :
                                     rgrid.get_dist(v, v0);

                T ldist = 1.0 / dist;

                fdt[v] = (ldist * fd[v]);
                frut[v] = (ldist * fru[v]);
                frvt[v] = (ldist * frv[v]);
                frwt[v] = (ldist * frw[v]);
            }

            // integrate
            D[v0]  = Greens_factor * integ::trapezoidal_3D(fdt, rgrid.X, rgrid.Y, rgrid.Z, rgrid.dx, rgrid.dy, rgrid.dz);
            Ru[v0] = Greens_factor * integ::trapezoidal_3D(frut, rgrid.X, rgrid.Y, rgrid.Z, rgrid.dx, rgrid.dy, rgrid.dz);
            Rv[v0] = Greens_factor * integ::trapezoidal_3D(frvt, rgrid.X, rgrid.Y, rgrid.Z, rgrid.dx, rgrid.dy, rgrid.dz);
            Rw[v0] = Greens_factor * integ::trapezoidal_3D(frwt, rgrid.X, rgrid.Y, rgrid.Z, rgrid.dx, rgrid.dy, rgrid.dz);

            if(v0 % progress_unit == 0){
                printf("\r Solving 3D natural HHD... %.1f %%", (float)v0*percent_factor);
                fflush(stdout);
            }
        }

        delete []fdt;
        delete []frut;
        delete []frvt;
        delete []frwt;

        double sec = (double)(clock() - tic)/CLOCKS_PER_SEC;
        if(sec < 1)     printf("\r Solving 3D natural HHD... Done! in %.3f msec.\n", 1000.0*sec);
        else            printf("\r Solving 3D natural HHD... Done! in %.3f sec.\n", sec);
    }

};
#endif
