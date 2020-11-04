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
    real x=0,y=0,z=0, xe,ye,ze

  cdef double beta = read_user_param('beta', 'f', sim.output.usr_dir)

  cdef np.ndarray[real, ndim=2] Az = np.zeros((gp.Ntot[1]+1,gp.Ntot[0]+1), dtype=np_real)

  #-----------------------------------------------------------------------------

  for j in range(gp.j1, gp.j2+2):
    for i in range(gp.i1, gp.i2+2):

      lind2gcrd(&x,&y,&z, i,j,0, gp)
      # vector potential is set at cell corners
      xe = x - 0.5*gp.dl[0]
      ye = y - 0.5*gp.dl[1]
      Az[j,i] = cos(4*M_PI*xe)/(4*M_PI) + cos(2*M_PI*ye)/(2*M_PI)

  for k in range(gp.k1, gp.k2+1):
    for j in range(gp.j1, gp.j2+1):
      for i in range(gp.i1, gp.i2+1):

        W[RHO,k,j,i] = 25./9
        W[PR,k,j,i] = 5./3
        IF CGL: W[PPD,k,j,i] = W[PR,k,j,i]

        lind2gcrd(&x,&y,&z, i,j,k, gp)

        W[VX,k,j,i] = -sin(2*M_PI*y)
        W[VY,k,j,i] =  sin(2*M_PI*x)

        IF MFIELD:
          B[0,k,j,i] =  (Az[j+1,i] - Az[j,i]) / gp.dl[1] / sqrt(beta) + 1e-20
          B[1,k,j,i] = -(Az[j,i+1] - Az[j,i]) / gp.dl[0] / sqrt(beta) + 1e-20



# ============================================================

cdef void do_user_work_cons(real4d U1, real4d B1, real4d U0, real4d B0,
                       ints lims[6], BnzSim sim, double dt):
  return



# Set user grid boundary conditions.

cdef void set_bc_grid_ptrs_user(BnzBC bc):
  # bc.bc_grid_funcs[0,0] = x1_bc_grid
  return


# Set user particle boundary conditions.

IF PIC or MHDPIC:
  cdef void set_bc_prt_ptrs_user(BnzBC bc):
    # bc.bc_prt_funcs[0,0] = x1_bc_prt
    # bc.bc_exch_funcs[0,0] = x1_bc_exch
    return


# Set user physics function (gravitational potential, diffusivities etc.)

cdef void set_phys_ptrs_user(BnzPhysics phys):

  phys.grav_pot_func = grav_pot

  return


# Set user history variables and particle selection function.

cdef void set_output_user(BnzOutput output):

  # set up to NHST_U (8 by default) function pointers

  # output.hst_funcs_u[0] = hst_var1
  # output.hst_names_u[0] = "B2h"

  IF PIC: output.prt_sel_func = select_particle
  return



# ===========================================================


# cdef void x1_bc_grid(BnzGrid grid, ints[::1] bc_vars):
#   return
# cdef void x1_bc_exch(BnzGrid grid):
#  return
# cdef coid x1_bc_prt(BnzGrid grid):
#   return

# ===========================================================



cdef real grav_pot(real x, real y, real z, double g0, double Lglob[3]) nogil:

  # cdef:
  #   real Rx = x-0.5*Lglob[0]
  #   real Ry = y-0.5*Lglob[1]
  #   real Rz = z-0.5*Lglob[2]
  #   real R = sqrt(Rx**2+Ry**2+Rz**2)
  #   real rc=0.12*0.4
  #   double gam=5./3

  # if R < rc*sqrt(3):
  #     return 1./gam * log(1+(R/rc)**2)
  # else:
  #     return 1./gam * log(4)
  #     #        return (-1.5*sqrt(3)*rc/R + log(4)+1.5) / gam

  return g0*y



# =========================================================

cdef double hst_var1(BnzSim sim):

  # cdef:
  #   ints i,j,k
  #   real B2=0
  #
  # cdef:
  #   GridParams gp = sim.grid.params
  #   real4d U = sim.grid.data.U
  #
  # IF MPI:
  #   cdef:
  #     double[::1] var = np.empty(1, dtype='f8')
  #     double[::1] var_sum = np.empty(1, dtype='f8')
  #
  # for k in range(gp.k1, gp.k2+1):
  #   for j in range(gp.j1, gp.j2+1):
  #     for i in range(gp.i1, gp.i2+1):
  #       B2 += U[BX,k,j,i]**2 + U[BY,k,j,i]**2 + U[BZ,k,j,i]**2
  #
  # IF MPI:
  #   var[0] = B2
  #   gp.comm.Allreduce(var, var_sum, op=mpi.SUM)
  #   B2 = var_sum[0]

  return 0.#B2 / (2*gp.Nact_glob[0]*gp.Nact_glob[1]*gp.Nact_glob[2])


# -------------------------------------------------------

IF MHDPIC or PIC:

  cdef int select_particle(ParticleData *pd, ints n):

    if pd.id[n] % 100 == 0:
      return 1
    else:
      return 0
