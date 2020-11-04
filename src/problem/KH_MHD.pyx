# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand

from bnz.coord cimport lind2gcrd, lind2gcrd_x,lind2gcrd_y,lind2gcrd_z

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
    double rho1,rho2

  rho1=1.
  rho2=2.

  cdef double beta = read_user_param('beta', 'f', sim.output.usr_dir)

  #-----------------------------------------------------------------------------

  for k in range(gp.k1, gp.k2+1):
    for j in range(gp.j1, gp.j2+1):
      for i in range(gp.i1, gp.i2+1):

        lind2gcrd(&x,&y,&z, i,j,k, gp)

        R = sqrt((y-0.5*gp.Lglob[1])**2)# + (z-0.5*gp.Lglob[2])**2)
        if  R > gp.Lglob[1]/4:
          W[VX,k,j,i] = 0.5
          W[RHO,k,j,i] = rho1
        else:
          W[VX,k,j,i] = -0.5
          W[RHO,k,j,i] = rho2

        W[VX,k,j,i] += 0.01* np.random.normal(0,1)
        W[PR,k,j,i] = 2.5

        IF CGL: W[PPD,k,j,i] = W[PR,k,j,i]

        B[0,k,j,i] = 1./sqrt(0.5*beta)


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
