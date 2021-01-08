# bonzo

Grid-based MHD code for plasma modelling.

Based on the following papers with modifications:
1) Van-Leer integrator:   
"A simple unsplit Godunov method for multidimensional MHD" by J. Stone and  T. Gardiner (2009)
2) 3rd-order Runge-Kutta integrator:  
"Pluto: a Numerical Code for Computational Astrophysics" by A. Mignone et al. (2007)
3) Implementation of non-Cartesian geometries:  
"High-order conservative reconstruction schemes for finite volume methods in cylindrical and spherical coordinates" by A. Mignone (2014)
4) Diffusion solver based on super-time-stepping:  
"A stabilized Runge-Kutta-Legendre method for explicit super-time-stepping of parabolic and mixed equations" by C. Meyer et al. (2014)

## Prerequisites

- python 3
- cython 3
- optional (parallelization): MPI (tested with OpenMPI and Intel MPI) + mpi4py
- optional (output file format): HDF5 + h5py, necessary with MPI

## Compilation

1) See available compilation options:
```
python setup.py --help 
```
2) Compile:
```
python setup.py build_ext --inplace --problem=problem_name [options]
```
``problem_name`` corresponds to the name of the chosen problem generator found in the ``problem`` directory

## Running

1) Create a user directory in the root folder:
```
mkdir user_dir
```
2) Place a configuration file (example provided in the root folder) in the user folder:
```
cp config.cfg user_dir
```
3) Run: 
```
python main.py ./user_dir
```
4) Output is written to the ``out`` directory in the user folder, restart files saved in ``rst``.
