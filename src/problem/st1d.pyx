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
    real x=0,y=0,z=0

  cdef double beta = read_user_param('beta', 'f', sim.output.usr_dir)
  IF MPI: print mpi.COMM_WORLD.Get_rank(), beta

  #-----------------------------------------------------------------------------

  for k in range(gp.k1, gp.k2+1):
    for j in range(gp.j1, gp.j2+1):
      for i in range(gp.i1, gp.i2+1):

        lind2gcrd(&x,&y,&z, i,j,k, gp)

        # if x <= 0.5*gp.Lglob[0]:
        #   W[RHO,k,j,i] = 1.08
        #   W[VX,k,j,i] = 1.2
        #   W[VY,k,j,i] = 0.01
        #   W[VZ,k,j,i] = 0.5
        #   W[PR,k,j,i] = 0.95
        #   B[0,k,j,i] = 2./sqrt(4*M_PI)
        #   B[1,k,j,i] = 3.6/sqrt(4*M_PI)
        #   B[2,k,j,i] = 2./sqrt(4*M_PI)
        # else:
        #   W[RHO,k,j,i] = 1.
        #   W[VX,k,j,i] = 0
        #   W[VY,k,j,i] = 0
        #   W[VZ,k,j,i] = 0
        #   W[PR,k,j,i] = 1.
        #   B[0,k,j,i] = 2./sqrt(4*M_PI)
        #   B[1,k,j,i] = 4./sqrt(4*M_PI)
        #   B[2,k,j,i] = 2./sqrt(4*M_PI)

        if x <= 0.5*gp.Lglob[0]:
          W[RHO,k,j,i] = 1.
          W[VX,k,j,i] = 0
          W[VY,k,j,i] = 0
          W[VZ,k,j,i] =0
          W[PR,k,j,i] = 1
          B[0,k,j,i] = 0.75
          B[1,k,j,i] = 1
          B[2,k,j,i] = 0
        else:
          W[RHO,k,j,i] = 0.125
          W[VX,k,j,i] = 0
          W[VY,k,j,i] = 0
          W[VZ,k,j,i] = 0
          W[PR,k,j,i] = 0.1
          B[0,k,j,i] = 0.75
          B[1,k,j,i] = -1
          B[2,k,j,i] = 0

       # if x <= 0.5*gp.Lglob[0]:
       #   W[RHO,k,j,i] = 1.
       #   W[VX,k,j,i] = 10
       #   W[VY,k,j,i] = 0
       #   W[VZ,k,j,i] =0
       #   W[PR,k,j,i] = 20
       #   B[0,k,j,i] = 5./sqrt(4*M_PI)
       #   B[1,k,j,i] = 5./sqrt(4*M_PI)
       #   B[2,k,j,i] = 0
       # else:
       #   W[RHO,k,j,i] = 1.
       #   W[VX,k,j,i] = -10
       #   W[VY,k,j,i] = 0
       #   W[VZ,k,j,i] = 0
       #   W[PR,k,j,i] = 1
       #   B[0,k,j,i] = 5./sqrt(4*M_PI)
       #   B[1,k,j,i] = 5./sqrt(4*M_PI)
       #   B[2,k,j,i] = 0



# =====================================================================

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
