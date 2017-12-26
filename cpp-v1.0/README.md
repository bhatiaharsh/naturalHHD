## The Natural Helmholtz-Hodge Decomposition

#### Implementation of the TVCG paper [doi:10.1109/TVCG.2014.2312012](http://dx.doi.org/10.1109/TVCG.2014.2312012)
#### by: Harsh Bhatia (bhatia4@llnl.gov)

A C++ library to compute the natural HHD. For details, please see the aforementioned paper. The code is released under BSD licence. Please make appropriate citation if you use the technique/code.


- **Build:** use CMakeLists.txt 
- **Usage:** ./naturalHHD vfield_filename X dx Y dz [Z] [dz]
- vfield_filename: filename of the raw file containing the vector field as binary floats
- X, Y, Z: size of the grid
- dx, dy, dz: grid spacings
- Z and dz are optional parameters needed only for 3D vector fields
- program expectes (X * Y * 2) float values for 2D, and (X * Y * Z * 3) float values for 3d vector fields
- **Output:** four raw files containing the original field and the three components respectively


##### version 1.0 (Dec 18, 2015)
* Serial implementation on regular/rectangular grids

###### version 1.0.1 (Nov 14, 2016)

* Fixed a bug in the computation of harmonic field

###### version 1.0.2 (Dec 11, 2016)

* Using a better approximation of log(0) and 1/0 produces sharper results

