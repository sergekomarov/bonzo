# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand

from bnz.coord cimport lind2gcrd

IF SPREC:
  np_real = np.float32
ELSE:
  np_real = np.float64



cdef void set_problem(BnzSim sim):

  cdef:
    BnzGrid grid = sim.grid
    GridParams gp = grid.params
    GridData gd = grid.data
    real4d W = gd.W
    BnzPhysics phys = sim.phys
  IF MFIELD:
    cdef real4d B = gd.B

  cdef:
    ints i,j,k
    real x=0,y=0,z=0, a,en
    double A=1e-2

  phys.gam = 5./3  # reset gas gamma

  #-----------------------------------------------------------------------------

  for k in range(gp.k1, gp.k2+1):
    for j in range(gp.j1, gp.j2+1):
      for i in range(gp.i1, gp.i2+1):

        lind2gcrd(&x,&y,&z, i,j,k, gp)

        W[RHO,k,j,i] = 1.
        W[PR,k,j,i]  = 0.6
        a = A * cos(2*M_PI*x)

        # ==== HYDRO ====

        # sound wave

        W[RHO,k,j,i] +=  a
        W[VX,k,j,i] += -a
        W[VY,k,j,i] +=  a
        W[VZ,k,j,i] +=  a
        W[PR,k,j,i] += (phys.gam-1) * 1.5*a

        # entropy wave

        #

        # IF MFIELD:

          # ==== MHD ====

          # Alfven, slow, fast

          # B[0,k,j,i] = 1.
          # B[1,k,j,i] = sqrt(2)
          # B[2,k,j,i] = 0.5

          # 1) Alfven

          # W[VY,k,j,i] += -1./3 * a
          # W[VZ,k,j,i] += 2*sqrt(2)/3 * a
          # B[1,k,j,i] += -1./3 * a
          # B[2,k,j,i] += 2*sqrt(2)/3 * a

          # 2) Slow

          # W[RHO,k,j,i] +=  2/sqrt(5) * a
          # W[VX,k,j,i] += -1/sqrt(5) * a
          # W[VY,k,j,i] += -4*sqrt(2)/3 / sqrt(5) * a
          # W[VZ,k,j,i] += -2./3 / sqrt(5) * a
          # B[1,k,j,i] += - 2*sqrt(2)/3 / sqrt(5) * a
          # B[2,k,j,i] += - 1./3 / sqrt(5) * a
          #
          # W[PR,k,j,i] += (phys.gam-1) * 3./(2*sqrt(5)) * a


          # 3) Fast

          # W[RHO,k,j,i] +=  1/sqrt(5) * a
          # W[VX,k,j,i] += -2/sqrt(5) * a
          # W[VY,k,j,i] += 2*sqrt(2)/3 / sqrt(5) * a
          # W[VZ,k,j,i] += 1./3 / sqrt(5) * a
          # B[1,k,j,i] += 4*sqrt(2)/3 / sqrt(5) * a
          # B[2,k,j,i] += 2./3 / sqrt(5) * a
          #
          # W[PR,k,j,i] += (phys.gam-1) * 9./(2*sqrt(5)) * a



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
