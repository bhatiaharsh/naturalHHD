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
 * RW.h
 *  Read-write functionalities.
 *
 *  Author: Harsh Bhatia. bhatia4@llnl.gov
 */


#ifndef RWUTILS_H_
#define RWUTILS_H_

#include <vector>
#include <string>
#include <fstream>

namespace RW {

    /// ----------------------------------------------------------------------------------------------------
    /// Basic function -- Read a chunk of floats and doubles!

    template <typename T>
    T* read_binary(std::string filename, size_t &N, bool verbose){

        // open the file
        FILE *datafile = fopen(filename.c_str(), "rb");
        if(!datafile){
            printf(" Unable to open data file %s\n", filename.c_str());
            N = 0;
            return 0;
        }

        if(verbose){
            printf(" Reading binary values from file %s...", filename.c_str());
            fflush(stdout);
        }

        if(N == 0){
            // find the size of the file
            fseek(datafile, 0, SEEK_END);
            size_t sz = ftell(datafile);
            rewind(datafile);

            if(sz % sizeof(T) != 0){
                printf("\n\t - Invalid number of values in file %s. Size of file = %zu, Size of each value = %zu -- mod = %zu\n",
                       filename.c_str(), sz, sizeof(T), (sz%sizeof(T)));
                fclose(datafile);
                return 0;
            }
            N = sz / sizeof(T);
        }

        // read the data
        T *values = new T[N];
        size_t rd_sz = fread(values, sizeof(T), N, datafile);
        if(rd_sz != N){
            printf("\n\t - Expected %zd, but read %zd values!\n", N, rd_sz);
            N = rd_sz;
        }

        // return
        fclose(datafile);
        if(verbose)
            printf(" Done! Read %'zd values!\n", N);
        return values;
    }

    template <typename T>
    void write_binary(std::string filename, const T *data, size_t N, bool verbose){

        // open the file
        FILE *fp = fopen(filename.c_str(), "wb");
        if(!fp){
            printf(" Unable to open data file %s\n", filename.c_str());
            return;
        }

        if(verbose){
            printf(" Writing binary values to file %s...", filename.c_str());
            fflush(stdout);
        }

        fwrite(data, sizeof(T), N,fp);

        // return
        fclose(fp);
        if(verbose)
            printf(" Done! %'zd values!\n", N);
        return;
    }
}

#endif
