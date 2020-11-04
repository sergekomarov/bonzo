#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport M_PI, sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand

from src.coord cimport loc_ind_to_glob_crd

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64



cdef void set_user_problem(real4d W, real4d B, BnzSim sim):

  # dom: Domain object
  # W: array of primitive hydrodynamic variables [RHO,VX,VY,VZ,P,(PPD),(PE),PSC]
  # B: array of staggered magnetic field

  cdef:
    BnzGrid grid = sim.grid
    BnzPhysics phys = sim.phys

  cdef:
    ints i,j,m,K
    double x=0,y=0,z=0
    double lami = 0.02
    double p0 = 1./phys.gam

  #-------------------------------

  phys.nuiic0 = sqrt(p0) / lami
  # phys.mu = 0.33*sqrt(p0) * 0.02
  # phys.mu4=1e-10

  for i in range(grid.i1, grid.i2+1):
    for j in range(grid.j1, grid.j2+1):
      for m in range(grid.m1, grid.m2+1):

        loc_ind_to_glob_crd(&x,&y,&z, i,j,m, grid)

        W[i,j,m,RHO] = 1.
        W[i,j,m,P] = 1./phys.gam

        IF CGL: W[i,j,m,PPD] = W[i,j,m,P]
        IF TWOTEMP: W[i,j,m,PE] = W[i,j,m,P]

        W[i,j,m,PSC] = 1. + phys.tvar * cos(2*M_PI / grid.Lglob[1] * y)

        B[i,j,m,0] = 1e-2


# ============================================================

cdef void set_bc_ptrs_grid_user(BnzBC bc):
    # bc.bc_grid_funcs[0][0] = x1_bc_grid
  return

IF PIC or MHDPIC:
  cdef void set_bc_ptrs_prt_user(BnzBC bc):
    # bc.bc_prt_funcs[0][0] = x1_bc_prt
    return

cdef void set_phys_ptrs_user(BnzPhysics phys):
  #phys.grav_pot_func = grav_pot
  return


# Set user history variables and particle selection function.

cdef void set_output_user(BnzOutput output):
  # set up to NHST_U (8 by default) function pointers
  # output.hst_funcs_u[0] = hst_var1
  # output.hst_names_u[0] = "B2h"
  # IF PIC:
  #   output.prt_sel_func = select_particle
  return
