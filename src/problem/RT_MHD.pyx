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

    real Lx=gp.Lglob[0], Ly=gp.Lglob[1], Lz=gp.Lglob[2]
    real Lxi, Lyi, Lzi

    real rho1, rho2

  cdef:
    ints i,j,k
    real x=0,y=0,z=0


  #-----------------------------------------------------------------------------

  Lxi, Lyi = 1./Lx, 1./Ly
  IF D3D: Lzi = 1./Lz
  ELSE: Lzi=0.

  rho1=1.
  rho2=2.

  cdef double beta = read_user_param('beta', 'f', sim.output.usr_dir)

  for k in range(gp.k1, gp.k2+1):
    for j in range(gp.j1, gp.j2+1):
      for i in range(gp.i1, gp.i2+1):

        lind2gcrd(&x,&y,&z, i,j,k, gp)
        x -= 0.5*Lx
        y -= 0.5*Ly
        z -= 0.5*Lz

        W[VY,k,j,i] += 0.01* (1+cos(2*M_PI*x*Lxi)) *\
                             (1+cos(2*M_PI*y*Lyi)) *\
                             (1+cos(2*M_PI*z*Lzi)) / 8

        if y < 0.:
          W[RHO,k,j,i] = rho1
        else:
          W[RHO,k,j,i] = rho2

        W[PR,k,j,i] = 2.5 - W[RHO,k,j,i] * phys.g0 * y

        IF CGL: W[PPD,k,j,i] = W[PR,k,j,i]

        B[0,k,j,i] = 1./sqrt(0.5*beta)


# ============================================================

cdef void do_user_work_cons(real4d U1, real4d B1, real4d U0, real4d B0,
                       ints lims[6], BnzSim sim, double dt):
  return



# Set user grid boundary conditions.=

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

  return


cdef real grav_pot(real x, real y, real z, double g0, double Lglob[3]) nogil:

  return g0*y
