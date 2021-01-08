# -*- coding: utf-8 -*-

IF MPI:
  from mpi4py import MPI as mpi
  from mpi4py cimport MPI as mpi

import numpy as np
cimport numpy as np

from libc.math cimport sqrt,floor,ceil,log,exp,sin,cos,pow,fabs,fmin,fmax
from libc.stdlib cimport rand, RAND_MAX, srand

from bnz.coord cimport lind2gcrd, lind2gind
from bnz.read_config import read_user_param

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
    real x=0,y=0,z=0, R
    real Lx=gp.Lglob[0], Ly=gp.Lglob[1], Lz=gp.Lglob[2]

  cdef:
    double beta = read_user_param('beta', 'f', sim.output.usr_dir)
    double bx0  = read_user_param('bpar', 'f', sim.output.usr_dir)
    real p0 = 1./phys.gam
    # double vrms = read_user_param('vrms', 'f', sim.output.usr_dir)

  #-------------------------

  for k in range(gp.k1, gp.k2+1):
    for j in range(gp.j1, gp.j2+1):
      for i in range(gp.i1, gp.i2+1):

        lind2gcrd(&x,&y,&z, i,j,k, gp)

        W[RHO,k,j,i] = 1.

        W[PSC,k,j,i]=1 + 0.1*cos(2*M_PI*x)
        W[PR,k,j,i] = p0
        IF CGL: W[PPD,k,j,i] = W[PR,k,j,i]
        IF TWOTEMP: W[PE,k,j,i] = W[PR,k,j,i]
        IF MFIELD: B[0,k,j,i] = bx0

  # if vrms != 0.:
    # v = gen_divfree3dvec(Nx,Ny,Nz,dx, rms=pm.vrms,
    #             s=pm.s, kmin=pm.kmin, kmax=pm.kmax,
    #             nt=dom.nt)

  # transport coefficients

  # viscosity

  # lami = 0.015
  # vthi = sqrt(Tbg)
  # pm.mu = 0.33 * lami * vthi

  # thermal conductivity

  # lame = 0.001 / 3#0.005
  # vthe = 42*sqrt(Tbg)
  # phys.kappa0 = 0.93 * lame * vthe


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
  # phys.grav_pot_func = grav_pot
  # phys.thcond_elec_func = thcond_elec
  return


cdef real grav_pot(real x, real y, real z, double g, double Lglob[3]) nogil:

  return 0


# Set user history variables and particle selection function.

cdef void set_output_user(BnzOutput output):
  # set up to NHST_U (8 by default) function pointers
  # output.hst_funcs_u[0] = hst_var1
  # output.hst_names_u[0] = "B2h"
  # IF PIC:
  #   output.prt_sel_func = select_particle
  return
