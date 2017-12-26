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

#ifndef GREENFUNC_H_
#define GREENFUNC_H_

#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "RGrid.h"


// during integration, i need to use Green's function
// i dont want to compute log and sqrt everytime
// instead, just compute them once and store
// this map stores the Green's functiion (v, v0)
// this is a symmetrical map computed for regular grids
// i store it as a 1D array

/** ---------------------------------------------------------------------------
 *  GreensFunction.h
 * This class creates a map to represent Green's function G(v, v0),
 *      so that distance, log, and sqrt have to be computed only once.
 *      Since, G(v, v0) = G(v0, v), only half of the matrix is saved.
 *      It is laid out linearly as a 1D map.
 */
template <typename T = float>
class GreenFunc {

private:

    size_t sz;      /*!< Size of the grid */
    size_t onedsz;  /*!< Size of the 1D map (= sz*(sz-1) / 2) */
    T* onedmap;     /*!< The 1D map used to store the function */
    T zeroval;      /*!< Value corresponding to v = v0 */


    /// Compute the index in the 1D map corresponding to the two global indices.
    inline long int get_1didx(size_t v, size_t v0) const {

        return (v > v0) ? (long int)(v0*(sz-1)) - (long int)(v0-1)*v0/2 + (long int)(v - v0 - 1) :
                         (long int)(v*(sz-1)) - (long int)(v-1)*v/2 + (long int)(v0 - v - 1);
    }

public:

    /** @brief Constructor.
     *
     *  Create Green's function for a 2D or 3D regular grid.
     *
     *  @param rgrid: Reference to the regular grid.
     */
    GreenFunc(const RGrid &rgrid){

        sz = rgrid.sz;
        onedsz = sz*(sz-1) / 2;
        onedmap = new T[onedsz];

        printf(" Creating Green's function map... [%zd x %zd] = (%'zd) ", sz, sz, onedsz);
        fflush(stdout);

        clock_t tic = clock();

        // variables for showing progress
        float percent_factor = 100.0f / (float)rgrid.sz;
        float percent_unit = 0.1;                           // show progress at this percent work
        unsigned int progress_unit = std::max((unsigned int)1, (unsigned int) (percent_unit / percent_factor));

        // -------------------------------------------------------------------
        // the 2D Green's function is ---- 1/(2pi) log( dist(x,x_0) )
            // instead, I use ------------ 1/(4pi) log( dist^2(x,x_0) )
        // the 3D Green's function is ---- -1/(4pi) dist(x,x_0)


        if(rgrid.dim == 2){

            T Greens_factor = 0.25/M_PI;

            // Harsh fixed this on 12/11/2016
#if 1
            zeroval = Greens_factor * log( rgrid.get_dist_sq(0, 0) );
#else
            zeroval = Greens_factor * log((0.01 * (rgrid.dx+rgrid.dy)));
#endif

            for(size_t v0 = 0; v0 < sz; v0++){
            for(size_t v = v0+1; v < sz; v++){
                onedmap[get_1didx(v0, v)] = Greens_factor * log(rgrid.get_dist_sq(v0, v));
            }

            if(v0 % progress_unit == 0){
                printf("\r Creating Green's function map... [%zd x %zd] = (%zd)   %.1f %%", sz, sz, onedsz, (float)v0*percent_factor);
                fflush(stdout);
            }
            }
        }
        else {

            T Greens_factor = -0.25/M_PI;

            // Harsh fixed this on 12/11/2016
#if 1
            zeroval = Greens_factor * log( rgrid.get_dist(0, 0) );
#else
            zeroval = Greens_factor / (0.01 * (rgrid.dx+rgrid.dy+rgrid.dz));
#endif
            for(size_t v0 = 0; v0 < sz; v0++){
            for(size_t v = v0+1; v < sz; v++){
                onedmap[get_1didx(v0, v)] = Greens_factor / rgrid.get_dist(v0, v);
            }

            if(v0 % progress_unit == 0){
                printf("\r Creating Green's function map... [%zd x %zd] = (%zd)   %.1f %%", sz, sz, onedsz, (float)v0*percent_factor);
                fflush(stdout);
            }
            }
        }

        double sec = (double)(clock() - tic)/CLOCKS_PER_SEC;
        if(sec < 1)     printf("\r Creating Green's function map... [%zd x %zd] = (%zd)   Done! in %.3f msec.\n", sz, sz, onedsz, 1000.0*sec);
        else            printf("\r Creating Green's function map... [%zd x %zd] = (%zd)   Done! in %.3f sec.\n", sz, sz, onedsz, sec);
    }

    /// Destructor
    ~GreenFunc(){
        delete []onedmap;
    }

    /** @brief Get the value of Green's function: G(v, v0).
    */
    inline T get_val(size_t v, size_t v0) const {
        return (v == v0) ? zeroval : onedmap[get_1didx(v, v0)];
    }
};
#endif
