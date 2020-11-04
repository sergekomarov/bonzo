# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand

from bnz.coord cimport lind2gcrd, lind2gcrd_x,lind2gcrd_y,lind2gcrd_z
from bnz.read_config import read_user_param

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64


cdef void set_problem(BnzSim sim):

  cdef:
    BnzGrid grid = sim.grid
    GridParams gp = grid.params
    real4d W = grid.data.W     # array of primitive variables
    real4d B = grid.data.B     # array of face-centered magnetic field
    BnzPhysics phys = sim.phys

  cdef:
    ints i,j,k
    real x=0,y=0,z=0, R
    double r,th

  # turn off MHD solver, set Lx=Ly=2, gamma=2

  cdef np.ndarray[real, ndim=2] Az = np.zeros((gp.Ntot[1]+1,
                                               gp.Ntot[0]+1), dtype=np_real)
  # choose Lx=Ly=2

  for j in range(gp.Ntot[1]+1):
    for i in range(gp.Ntot[0]+1):

      lind2gcrd(&x,&y,&z, i,j,0, gp)
      x -= 0.5*(gp.Lglob[0] + gp.dl[0])
      y -= 0.5*(gp.Lglob[1] + gp.dl[1])
      r = sqrt(x**2 + y**2) + 1e-20

      Az[j,i] = r

  for k in range(gp.k1, gp.k2+1):
    for j in range(gp.j1, gp.j2+1):
      for i in range(gp.i1, gp.i2+1):

        lind2gcrd(&x,&y,&z, i,j,k, gp)

        x -= 0.5*gp.Lglob[0]
        y -= 0.5*gp.Lglob[1]
        r = sqrt(x**2 + y**2) + 1e-20
        th = np.arcsin(y/r)
        if x<0: th = M_PI-th

        if th>11*M_PI/12 and th<13*M_PI/12 and r>0.5 and r<0.7:
          W[PR,k,j,i] = 12.
        else:
          W[PR,k,j,i] = 10.

        W[RHO,k,j,i] = 1.

        B[0,k,j,i] =  (Az[j+1,i] - Az[j,i]) * gp.dli[1]
        B[1,k,j,i] = -(Az[j,i+1] - Az[j,i]) * gp.dli[0]



# ============================================================

cdef void do_user_work_cons(real4d U1, real4d B1, real4d U0, real4d B0,
                       ints lims[6], BnzSim sim, double dt):
  return


cdef void set_bc_grid_ptrs_user(BnzBC bc):
    # bc.bc_grid_funcs[0][0] = x1_bc_grid
  return

IF PIC or MHDPIC:
  cdef void set_bc_prt_ptrs_user(BnzBC bc):
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
