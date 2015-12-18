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
 * trapezoidalIntegration.h
 *  Library for approximating the integral of a function over 1D, 2D, and 3D
 *      domains using trapezoidal methjod.
 *  - The regular grid [X x Y x Z] has grid spacings (dx, dy, dz).
 *
 *  Author: Harsh Bhatia. bhatia4@llnl.gov
 */


#ifndef _TRAPEZOIDAL_H_
#define _TRAPEZOIDAL_H_

#include <vector>
#include <numeric>

namespace integ {

    /// ----------------------------------------------------------------------------------------------------
    /** LINE integral using 1D trapezoidal rule
        the corners of the line need to be counted once
        everything in the interior needs to be counted twice
    */
    template <typename T>
    T trapezoidal_1D(const std::vector<T> &f, size_t N, T h){

        if(f.size() != N){
            cerr <<" trapezoidal_1D -- Size mismatch ("<<f.size()<<" != "<<N<<").\n";
            return T(0);
        }

        T sum = 2.0*accumulate(f.begin(), f.end(), T(0));
        sum -= f[0];
        sum -= f[N-1];
        return 0.5*h*sum;
    }

    template <typename T>
    T trapezoidal_1D(const T *f, size_t N, T h){

        T sum = 2.0*accumulate(f, f+N, T(0));
        sum -= f[0];
        sum -= f[N-1];
        return 0.5*h*sum;
    }

    /// ----------------------------------------------------------------------------------------------------
    /** AREA integral using 2D trapezoidal rule
        the corners of the plane need to be counted once
        the edges of the plane need to be counted twice
        everything in the interior needs to be counted four times
    */
    template <typename T>
    T trapezoidal_2D(const std::vector<T> &f, size_t X, size_t Y, T dx, T dy){

        if(f.size() != (X*Y)){
            //throw std::invalid_argument(" trapezoidal_2D -- invalid dimensions!");
            cerr <<" trapezoidal_2D -- Size mismatch ("<<f.size()<<" != "<<X*Y<<").\n";
            return T(0);
        }

        // count everything 4 times
        T sum = 4.0*std::accumulate(f.begin(), f.end(), T(0));

        // the four boundary edges need to be counted only twice
        for(size_t x = 1; x < X-1; x++){
            size_t ymin = x;
            size_t ymax = (Y-1)*X + x;

            sum -= 2.0*f[ymin];
            sum -= 2.0*f[ymax];
        }
        for(size_t y = 1; y < Y-1; y++){
            size_t xmin = y*X;
            size_t xmax = (y+1)*X - 1;

            sum -= 2.0*f[xmin];
            sum -= 2.0*f[xmax];
        }

        // the four corners need to be counted only once
        sum -= 3.0*f[0];        sum -= 3.0*f[X-1];
        sum -= 3.0*f[(Y-1)*X];  sum -= 3.0*f[X*Y-1];

        return 0.25*dx*dy*sum;
    }

    template <typename T>
    T trapezoidal_2D(const T *f, size_t X, size_t Y, T dx, T dy){

        // count everything 4 times
        T sum = 4.0*std::accumulate(f, f+(X*Y), T(0));

        // the four boundary edges need to be counted only twice
        for(size_t x = 1; x < X-1; x++){
            size_t ymin = x;
            size_t ymax = (Y-1)*X + x;

            sum -= 2.0*f[ymin];
            sum -= 2.0*f[ymax];
        }
        for(size_t y = 1; y < Y-1; y++){
            size_t xmin = y*X;
            size_t xmax = (y+1)*X - 1;

            sum -= 2.0*f[xmin];
            sum -= 2.0*f[xmax];
        }

        // the four corners need to be counted only once
        sum -= 3.0*f[0];        sum -= 3.0*f[X-1];
        sum -= 3.0*f[(Y-1)*X];  sum -= 3.0*f[X*Y-1];

        return 0.25*dx*dy*sum;
    }

    /// ----------------------------------------------------------------------------------------------------
    /** VOLUME integral using 3D trapezoidal rule
        the corners of the cube need to be counted once
        the edges of the cube need to be counted twice
        the faces of the cube need to be counted four times
        everything in the interior needs to be counted eight times
    */
    template <typename T>
    T trapezoidal_3D(const std::vector<T> &f, size_t X, size_t Y, size_t Z, T dx, T dy, T dz){

        if(f.size() != (X*Y*Z)){
            //throw std::invalid_argument(" trapezoidal_3D -- invalid dimensions!");
            cerr <<" trapezoidal_3D -- Size mismatch ("<<f.size()<<" != "<<X*Y*Z<<").\n";
            return T(0);
        }

        size_t XY = X*Y;

        // -------------------------------------------------------
        // count everything 8 times
        T sum = 8.0*std::accumulate(f.begin(), f.end(), T(0));

        // -------------------------------------------------------
        // faces

        // top and bottom faces
        for(size_t y = 1; y < Y-1; y++){
        for(size_t x = 1; x < X-1; x++){

            size_t v_z0 = y*X + x;
            size_t v_zZ = XY*(Z-1) + y*X + x;

            //cout<<" y = "<<y<<", x = "<<x<<" bot = "<<v_z0<<" top = "<<v_zZ<<endl;
            sum -= 4.0*f[v_z0];
            sum -= 4.0*f[v_zZ];
        }
        }

        // left and right faces
        for(size_t z = 1; z < Z-1; z++){
        for(size_t y = 1; y < Y-1; y++){

            size_t v_x0 = XY*z + y*X;
            size_t v_xX = XY*z + y*X + (X-1);

            //cout<<" z = "<<z<<", y = "<<y<<" left = "<<v_x0<<", right = "<<v_xX<<endl;
            sum -= 4.0*f[v_x0];
            sum -= 4.0*f[v_xX];
        }
        }

        // front and back faces
        for(size_t z = 1; z < Z-1; z++){
        for(size_t x = 1; x < X-1; x++){

            size_t v_y0 = XY*z + x;
            size_t v_yY = XY*z + (Y-1)*X + x;

            //cout<<" z = "<<z<<", x = "<<x<<" front = "<<v_y0<<", back = "<<v_yY<<endl;
            sum -= 4.0*f[v_y0];
            sum -= 4.0*f[v_yY];
        }
        }

        // -------------------------------------------------------
        // now, the 12 edges

        // the x-edges
        for(size_t x = 1; x < X-1; x++){

            size_t v_yz = x;
            size_t v_yZ = XY*(Z-1) + x;
            size_t v_Yz = (Y-1)*X + x;
            size_t v_YZ = XY*(Z-1) + (Y-1)*X + x;

            //cout<<" x "<<x<<" 00 = "<<v_yz<<" 01 = "<<v_yZ<<" 10 = "<<v_Yz<<" 11 = "<<v_YZ<<endl;
            sum -= 6.0*f[v_yz];
            sum -= 6.0*f[v_yZ];
            sum -= 6.0*f[v_Yz];
            sum -= 6.0*f[v_YZ];
        }

        // the y-edges
        for(size_t y = 1; y < Y-1; y++){

            size_t v_xz = y*X;
            size_t v_xZ = XY*(Z-1) + y*X;
            size_t v_Xz = y*X + (X-1);
            size_t v_XZ = XY*(Z-1) + y*X + (X-1);

            //cout<<" y "<<y<<" 00 = "<<v_xz<<" 01 = "<<v_xZ<<" 10 = "<<v_Xz<<" 11 = "<<v_XZ<<endl;
            sum -= 6.0*f[v_xz];
            sum -= 6.0*f[v_xZ];
            sum -= 6.0*f[v_Xz];
            sum -= 6.0*f[v_XZ];
        }

        // the z-edges
        for(size_t z = 1; z < Z-1; z++){

            size_t v_xy = XY*z;
            size_t v_xY = XY*z + (Y-1)*X;
            size_t v_Xy = XY*z + (X-1);
            size_t v_XY = XY*z + (Y-1)*X + (X-1);

            //cout<<" z "<<z<<" 00 = "<<v_xy<<" 01 = "<<v_xY<<" 10 = "<<v_Xy<<" 11 = "<<v_XY<<endl;
            sum -= 6.0*f[v_xy];
            sum -= 6.0*f[v_xY];
            sum -= 6.0*f[v_Xy];
            sum -= 6.0*f[v_XY];
        }

        // -------------------------------------------------------
        // the eight corners
        size_t xyZ = XY*(Z-1);

        sum -= 7.0*f[0];    sum -= 7.0*f[X-1];      sum -= 7.0*f[(Y-1)*X];      sum -= 7.0*f[XY-1];
        sum -= 7.0*f[xyZ];  sum -= 7.0*f[xyZ+X-1];  sum -= 7.0*f[xyZ+(Y-1)*X];  sum -= 7.0*f[xyZ+XY-1];

        // -------------------------------------------------------
        return 0.125*dx*dy*dz*sum;
    }

    template <typename T>
    T trapezoidal_3D(const T *f, size_t X, size_t Y, size_t Z, T dx, T dy, T dz){

        size_t XY = X*Y;

        // -------------------------------------------------------
        // count everything 8 times
        T sum = 8.0*std::accumulate(f, f+(XY*Z), T(0));

        // -------------------------------------------------------
        // faces

        // top and bottom faces
        for(size_t y = 1; y < Y-1; y++){
        for(size_t x = 1; x < X-1; x++){

            size_t v_z0 = y*X + x;
            size_t v_zZ = XY*(Z-1) + y*X + x;

            //cout<<" y = "<<y<<", x = "<<x<<" bot = "<<v_z0<<" top = "<<v_zZ<<endl;
            sum -= 4.0*f[v_z0];
            sum -= 4.0*f[v_zZ];
        }
        }

        // left and right faces
        for(size_t z = 1; z < Z-1; z++){
        for(size_t y = 1; y < Y-1; y++){

            size_t v_x0 = XY*z + y*X;
            size_t v_xX = XY*z + y*X + (X-1);

            //cout<<" z = "<<z<<", y = "<<y<<" left = "<<v_x0<<", right = "<<v_xX<<endl;
            sum -= 4.0*f[v_x0];
            sum -= 4.0*f[v_xX];
        }
        }

        // front and back faces
        for(size_t z = 1; z < Z-1; z++){
        for(size_t x = 1; x < X-1; x++){

            size_t v_y0 = XY*z + x;
            size_t v_yY = XY*z + (Y-1)*X + x;

            //cout<<" z = "<<z<<", x = "<<x<<" front = "<<v_y0<<", back = "<<v_yY<<endl;
            sum -= 4.0*f[v_y0];
            sum -= 4.0*f[v_yY];
        }
        }

        // -------------------------------------------------------
        // now, the 12 edges

        // the x-edges
        for(size_t x = 1; x < X-1; x++){

            size_t v_yz = x;
            size_t v_yZ = XY*(Z-1) + x;
            size_t v_Yz = (Y-1)*X + x;
            size_t v_YZ = XY*(Z-1) + (Y-1)*X + x;

            //cout<<" x "<<x<<" 00 = "<<v_yz<<" 01 = "<<v_yZ<<" 10 = "<<v_Yz<<" 11 = "<<v_YZ<<endl;
            sum -= 6.0*f[v_yz];
            sum -= 6.0*f[v_yZ];
            sum -= 6.0*f[v_Yz];
            sum -= 6.0*f[v_YZ];
        }

        // the y-edges
        for(size_t y = 1; y < Y-1; y++){

            size_t v_xz = y*X;
            size_t v_xZ = XY*(Z-1) + y*X;
            size_t v_Xz = y*X + (X-1);
            size_t v_XZ = XY*(Z-1) + y*X + (X-1);

            //cout<<" y "<<y<<" 00 = "<<v_xz<<" 01 = "<<v_xZ<<" 10 = "<<v_Xz<<" 11 = "<<v_XZ<<endl;
            sum -= 6.0*f[v_xz];
            sum -= 6.0*f[v_xZ];
            sum -= 6.0*f[v_Xz];
            sum -= 6.0*f[v_XZ];
        }

        // the z-edges
        for(size_t z = 1; z < Z-1; z++){

            size_t v_xy = XY*z;
            size_t v_xY = XY*z + (Y-1)*X;
            size_t v_Xy = XY*z + (X-1);
            size_t v_XY = XY*z + (Y-1)*X + (X-1);

            //cout<<" z "<<z<<" 00 = "<<v_xy<<" 01 = "<<v_xY<<" 10 = "<<v_Xy<<" 11 = "<<v_XY<<endl;
            sum -= 6.0*f[v_xy];
            sum -= 6.0*f[v_xY];
            sum -= 6.0*f[v_Xy];
            sum -= 6.0*f[v_XY];
        }

        // -------------------------------------------------------
        // the eight corners
        size_t xyZ = XY*(Z-1);

        sum -= 7.0*f[0];    sum -= 7.0*f[X-1];      sum -= 7.0*f[(Y-1)*X];      sum -= 7.0*f[XY-1];
        sum -= 7.0*f[xyZ];  sum -= 7.0*f[xyZ+X-1];  sum -= 7.0*f[xyZ+(Y-1)*X];  sum -= 7.0*f[xyZ+XY-1];

        // -------------------------------------------------------
        return 0.125*dx*dy*dz*sum;
    }
}
#endif
