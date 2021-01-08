# bonzo

Grid-based MHD code for plasma modelling.

Based on the following papers with modifications:
1) Van-Leer integrator: 
"A simple unsplit Godunov method for multidimensional MHD" by J. Stone, T. Gardiner, 2009
2) 3rd-order Runge-Kutta integrator:
"
3) Implementation of non-Cartesian geometries:
4) Diffusion solver based on ... super-time-stepping:

## Prerequisites

1) Python 3
2) Cython 3
3) optional (parallelization): MPI (tested with OpenMPI and Intel MPI) + mpi4py
4) optional (output file format): HDF5 + h5py, necessary with MPI

## Compilation

1) see available compilation options
```
python setup.py --help 
```
2) compile
```
python setup.py build_ext --inplace --problem=[problem_name] [options]
```
"problem_name" corresponds to the name of the chosen problem generator found in the problem directory

## Running

1) create a user directory in the root folder
```
makedir [user_folder]
```
2) place a configuration file (example provided in the root folder) in the user folder
```
cp config.cfg [user_folder]
```
3) run 
```
python main.py ./[user_folder]
```
4) the output will be written to the "out" directory in the user folder, restart files saved in "rst"
