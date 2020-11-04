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
    ints i,j,m
    double x=0,y=0,z=0

  cdef:
    double th = 0.#3*M_PI/12  # mean magnetic field angle relative to X
    double cth = cos(th)
    double sth = sin(th)

    double rho0 = 1.
    double B0 = 1.
    double p0 = 0.5*B0**2 * phys.beta
    double va2 = B0**2 / rho0
    double alpha = 0.5 * phys.gam * phys.beta
    double D = sqrt((1+alpha)**2 - 4*alpha*cth**2)
    double cf = sqrt(0.5*va2 * (1+alpha+D))

    double k0 = 4*M_PI/grid.Lglob[0]
    double L0 = 2*M_PI/k0
    double kpl = k0*cth  # relative to B0
    double kpd = k0*sth

    double vpl1 = (-1 + alpha + D) * cth   # relative to B
    double vpd1 = ( 1 + alpha + D) * sth
    double v1 = sqrt(vpl1**2 + vpd1**2)

    double amp = 0.001 * cf
    double ph, kv_om,kB0_om, vpl,vpd, Bpl,Bpd

  vpl1 /= v1
  vpd1 /= v1

  # take ion m.f.p. = wavelength/10
  cdef:
    double vthi = sqrt(0.5*phys.beta*va2)
    double lami = 0.1*L0
  phys.nuiic0 = vthi / lami
  phys.kappa0 =  0. * 0.93 * lami * 42 * vthi
  # using vthi = sqrt(2*T/mi)

  # phys.nuiic0 = phys.nuiic0 * k0*cf



  #-----------------------------------------------------------------------------

  for i in range(grid.i1, grid.i2+1):
    for j in range(grid.j1, grid.j2+1):
      for m in range(grid.m1, grid.m2+1):

        loc_ind_to_glob_crd(&x,&y,&z, i,j,m, grid)

        ph = sin(k0*x)

        vpl = amp * vpl1 * ph   # relative to B0
        vpd = amp * vpd1 * ph

        kv_om = (vpl * cth + vpd * sth) / cf
        kB0_om = B0 * cth / cf

        Bpl = kv_om * B0 - kB0_om * vpl
        Bpd =            - kB0_om * vpd

        # v and B in coordinates relative to k||X

        W[i,j,m,VX] =  cth*vpl + sth*vpd #+ 0.001*sin(4*M_PI*x)
        W[i,j,m,VY] = -sth*vpl + cth*vpd #+ 0.001*sin(4*M_PI*x)

        W[i,j,m,RHO] = rho0 * (1 + kv_om)
        W[i,j,m,P] = p0 * (1 + phys.gam * kv_om)
        W[i,j,m,PPD] = W[i,j,m,P]

        # B[i,j,m,0] =  cth*(Bpl+B0) + sth*Bpd
        B[i,j,m,0] =  cth*B0
        B[i,j,m,1] = -sth*(Bpl+B0) + cth*Bpd



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
