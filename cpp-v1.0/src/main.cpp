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
 * Simple application to compute the natural HHD
 * - Technical details can be found in the paper doi: 10.1109/TVCG.2014.2312012
 *
 * Author: Harsh Bhatia (bhatia4@llnl.gov)
 */

#include <iostream>
#include <string>
//#include <locale.h>


#include "RW.h"
#include "VectorField.h"
#include "GreensFunction.h"
#include "Poisson.h"
#include "HHD.h"

#define USE_MAP

using namespace std;

/*!
 * main function ..................!!
 */
int main (int argc, char* argv[]){

    //setlocale(LC_NUMERIC, "");  // enable thousand separator!
    // -------------------------------------------------------
    // create a regular grid based on the command line paramters!
    RGrid rgrid;

    if(argc == 6){
        rgrid = RGrid(atoi(argv[2]), atoi(argv[4]), atof(argv[3]), atof(argv[5]));
    }
    else if(argc == 8){
        rgrid = RGrid(atoi(argv[2]), atoi(argv[4]), atoi(argv[6]), atof(argv[3]), atof(argv[5]), atof(argv[7]));
    }
    else {
        printf(" Usage: %s <vfield_filename> <X> <dx> <Y> <dy> [Z] [dz]\n", argv[0]);
        exit(1);
    }

    // -------------------------------------------------------
    // read the filename
    std::string filename(argv[1]);
    VectorField<float> vfield(filename, rgrid.dim);
    //VectorField<float> vfield(rgrid);

    vfield.need_magnitudes(rgrid);
    vfield.need_divcurl(rgrid);
    vfield.show_stats("vfield");

    // -------------------------------------------------------
    // the three components!
    naturalHHD<float> nhhd(vfield, rgrid, 1);

    VectorField<float> d, r, h;

    // d
    d.compute_as_gradient_field(nhhd.D, rgrid);

    // r
    if(rgrid.dim == 2){
        r.compute_as_gradient_field(nhhd.Ru, rgrid);
        r.rotate_J();
    }
    else if(rgrid.dim == 3){
       r.compute_as_curl_field(nhhd.Ru, nhhd.Rv, nhhd.Rw, rgrid);
    }

    // h
    h.compute_as_harmonic_field(vfield, d, r);

    // -------------------------------------------------------
    d.need_magnitudes(rgrid);
    r.need_magnitudes(rgrid);
    h.need_magnitudes(rgrid);

    d.need_divcurl(rgrid);
    r.need_divcurl(rgrid);
    h.need_divcurl(rgrid);

    d.show_stats("d");
    r.show_stats("r");
    h.show_stats("h");

    // -------------------------------------------------------
    vfield.write_to_file("vfield.raw");
    d.write_to_file("gradfield.raw");
    r.write_to_file("curlfield.raw");
    h.write_to_file("harmfield.raw");

    // -------------------------------------------------------
    // write D
    RW::write_binary<float>("nD.raw", nhhd.D, rgrid.sz, true);


    return 0;
}
