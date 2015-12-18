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

/** ---------------------------------------------------------------------------
 * Poisson.h
 *  Library for solving the Poisson equation using integral solution.
 *  - Technical details can be found in the paper doi: 10.1109/TVCG.2014.2312012
 *
 *  Author: Harsh Bhatia. bhatia4@llnl.gov
 */


#ifndef POISSON_H
#define POISSON_H

#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "RGrid.h"
#include "GreensFunction.h"
#include "trapezoidalIntegration.h"


/** functions for solving 2D and 3D Poisson equation on a regular grid
        f is called the source function that creates a potential (RHS)
        phi is the potential function (LHS)
*/
namespace Poisson {

    /// ---------------------------------------------------------------------------
    /// ----- serial implementation!
    ///    I use a map that computes the Green's function only once, and
    ///     uses it for solving all equations
    template <typename T>
    T* solve_Poisson_2D(const T* f, const RGrid &rgrid, const GreenFunc<T> &gFunc){

        if(rgrid.dim != 2){
            printf(" solve_Poisson_2D requires a 2D regular grid!\n");
            return 0;
        }

        printf(" Solving 2D Poisson Eqn...");
        fflush(stdout);

        clock_t tic = clock();

        // variables for showing progress
        float percent_factor = 100.0f / (float)rgrid.sz;
        float percent_unit = 0.1;                           // show progress at this percent work
        uint progress_unit = std::max((uint)1, (uint) (percent_unit / percent_factor));

        /// output and temporary memory
        T* phi = new T[rgrid.sz];
        T* fd = new T[rgrid.sz];

        /// -------------------------------------------------------------------
        /// compute at v0
        for(size_t v0 = 0; v0 < rgrid.sz; v0++){

            /// populate the temporary field for the entire grid!
            #pragma omp parallel for
            for(size_t v = 0; v < rgrid.sz; v++){

                fd[v] = gFunc.get_val(v, v0) * f[v];
            }

            /// integrate
            phi[v0] = integ::trapezoidal_2D(fd, rgrid.X, rgrid.Y, rgrid.dx, rgrid.dy);

            if(v0 % progress_unit == 0){
                printf("\r  Solving 2D Poisson Eqn... %.1f %%", (float)v0*percent_factor);
                fflush(stdout);
            }
        }

        delete []fd;

        double sec = (double)(clock() - tic)/CLOCKS_PER_SEC;
        if(sec < 1)     printf("\r  Solving 2D Poisson Eqn... Done! in %.3f msec.\n", 1000.0*sec);
        else            printf("\r  Solving 2D Poisson Eqn... Done! in %.3f sec.\n", sec);
        return phi;
    }


    template <typename T>
    T* solve_Poisson_3D(const T* f, const RGrid &rgrid, const GreenFunc<T> &rmap){

        if(rgrid.dim != 3){
            printf(" solve_Poisson_3D requires a 3D regular grid!\n");
            return 0;
        }

        printf(" Solving 3D Poisson Eqn...");
        fflush(stdout);

        clock_t tic = clock();

        // variables for showing progress
        float percent_factor = 100.0f / (float)rgrid.sz;
        float percent_unit = 0.1;                           // show progress at this percent work
        uint progress_unit = std::max((uint)1, (uint) (percent_unit / percent_factor));

        //printf(" %f %f %d\n", percent_factor, percent_unit, progress_unit);

        /// output and temporary memory
        T* phi = new T[rgrid.sz];
        T* fd = new T[rgrid.sz];

        /// -------------------------------------------------------------------
        /// compute at v0
        for(size_t v0 = 0; v0 < rgrid.sz; v0++){

            /// populate the temporary field for the entire grid!
            #pragma omp parallel for
            for(size_t v = 0; v < rgrid.sz; v++){

                fd[v] = rmap.get_val(v, v0) * f[v];
            }

            /// integrate
            phi[v0] = integ::trapezoidal_3D(fd, rgrid.X, rgrid.Y, rgrid.Z, rgrid.dx, rgrid.dy, rgrid.dz);

            if(v0 % progress_unit == 0){
                printf("\r Solving 3D Poisson Eqn... %.1f %%", (float)v0*percent_factor);
                fflush(stdout);
            }
        }

        delete []fd;
        double sec = (double)(clock() - tic)/CLOCKS_PER_SEC;
        if(sec < 1)     printf("\r Solving 3D Poisson Eqn... Done! in %.3f msec.\n", 1000.0*sec);
        else            printf("\r Solving 3D Poisson Eqn... Done! in %.3f sec.\n", sec);
        return phi;
    }



    /// ---------------------------------------------------------------------------
    /// ----- serial implementation!
    ///     compute the Green's function on the fly
    ///     too many computations of log and sqrt!

    template <typename T>
    T* solve_Poisson_2D(const T* f, const RGrid &rgrid){

        if(rgrid.dim != 2){
            printf(" solve_Poisson_2D requires a 2D regular grid!\n");
            return 0;
        }

        printf(" Solving 2D Poisson Eqn...");
        fflush(stdout);

        clock_t tic = clock();

        // variables for showing progress
        float percent_factor = 100.0f / (float)rgrid.sz;
        float percent_unit = 0.1;                           // show progress at this percent work
        uint progress_unit = std::max((uint)1, (uint) (percent_unit / percent_factor));

        //printf(" %f %f %d\n", percent_factor, percent_unit, progress_unit);
        /// -------------------------------------------------------------------
        /// the 2D Green's function is ---- 1/(2pi) log( dist(x,x_0) )
            /// instead, I use ------------ 1/(4pi) log( dist^2(x,x_0) )
        T Greens_factor = 0.25/M_PI;

        /// output and temporary memory
        T* phi = new T[rgrid.sz];
        T* fd = new T[rgrid.sz];

        /// -------------------------------------------------------------------
        /// compute at v0
        for(size_t v0 = 0; v0 < rgrid.sz; v0++){

            /// populate the temporary field for the entire grid!
            #pragma omp parallel for
            for(size_t v = 0; v < rgrid.sz; v++){

                /// don't want to compute log (0)
                T dist = (v == v0) ? (0.01 * (rgrid.dx+rgrid.dy)) :
                                     rgrid.get_dist_sq(v, v0);

                fd[v] = (log(dist) * f[v]);
            }

            /// integrate
            phi[v0] = Greens_factor * integ::trapezoidal_2D(fd, rgrid.X, rgrid.Y, rgrid.dx, rgrid.dy);

            if(v0 % progress_unit == 0){
                printf("\r Solving 2D Poisson Eqn... %.1f %%", (float)v0*percent_factor);
                fflush(stdout);
            }
        }

        delete []fd;
        double sec = (double)(clock() - tic)/CLOCKS_PER_SEC;
        if(sec < 1)     printf("\r Solving 2D Poisson Eqn... Done! in %.3f msec.\n", 1000.0*sec);
        else            printf("\r Solving 2D Poisson Eqn... Done! in %.3f sec.\n", sec);
        return phi;
    }


    template <typename T>
    T* solve_Poisson_3D(const T* f, const RGrid &rgrid){

        if(rgrid.dim != 3){
            printf(" solve_Poisson_3D requires a 3D regular grid!\n");
            return 0;
        }

        printf(" Solving 3D Poisson Eqn...");
        fflush(stdout);

        clock_t tic = clock();

        // variables for showing progress
        float percent_factor = 100.0f / (float)rgrid.sz;
        float percent_unit = 0.1;                           // show progress at this percent work
        uint progress_unit = std::max((uint)1, (uint) (percent_unit / percent_factor));

        //printf(" %f %f %d\n", percent_factor, percent_unit, progress_unit);
        /// -------------------------------------------------------------------
        /// the 3D Green's function is ---- 1/(4pi) dist(x,x_0)
        T Greens_factor = -0.25/M_PI;

        /// output and temporary memory
        T* phi = new T[rgrid.sz];
        T* fd = new T[rgrid.sz];

        /// -------------------------------------------------------------------
        /// compute at v0
        for(size_t v0 = 0; v0 < rgrid.sz; v0++){

            /// populate the temporary field for the entire grid!
            #pragma omp parallel for
            for(size_t v = 0; v < rgrid.sz; v++){

                /// don't want to compute 1 / 0
                T dist = (v == v0) ? (0.01 * (rgrid.dx+rgrid.dy+rgrid.dz)) :
                                     rgrid.get_dist(v, v0);

                fd[v] = (f[v] / dist);
            }

            /// integrate
            phi[v0] = Greens_factor * integ::trapezoidal_3D(fd, rgrid.X, rgrid.Y, rgrid.Z, rgrid.dx, rgrid.dy, rgrid.dz);

            if(v0 % progress_unit == 0){
                printf("\r Solving 3D Poisson Eqn... %.1f %%", (float)v0*percent_factor);
                fflush(stdout);
            }
        }

        delete []fd;
        double sec = (double)(clock() - tic)/CLOCKS_PER_SEC;
        if(sec < 1)     printf("\r Solving 3D Poisson Eqn... Done! in %.3f msec.\n", 1000.0*sec);
        else            printf("\r Solving 3D Poisson Eqn... Done! in %.3f sec.\n", sec);

        return phi;
    }

    /// ---------------------------------------------------------------------------
}
#endif
