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
    # real4d B = grid.data.B     # array of face-centered magnetic field
    BnzPhysics phys = sim.phys

    real Lx=gp.Lglob[0], Ly=gp.Lglob[1], Lz=gp.Lglob[2]

  cdef:
    ints i,j,k
    real x=0, y=0, z=0
    real x1, y1, z1, r
    real f

  cdef double beta = read_user_param('beta', 'f')

  # L0 = 100 pc
  # t0 = 3e5 yr
  # v0 = 320 km/s
  # T0 = 10^7 K
  # n0 = 1 cm^(-3)
  # E0 = 4.7 * 10^52 erg
  # M0 = 2.3 * 10^4 Mo

  # Lx = 2; Ly = Lz = 1

  cdef:
    real rho0 = 1.
    real T0 = 1e-3

    real h0 = 1.
    real h1 = 2*h0

    real costh0 = cos(M_PI/9)
    real sinth0 = sqrt(1.-costh0**2)

    real xc = 0.5*Lx
    real yc = 0.5*Ly
    real zc = 0.5*Lz

    real msn = 5. / 2.3e4   # 5 Mo
    real esn = 3. * 1. / 47      # 10^51 erg
    real vsn0 = sqrt(2*esn / msn)

    # calculate the initial SN radius as a fraction of the Sedov radius
    real rho0_sn = rho0 * exp(-h1/h0)
    real rsed = (3./(4*M_PI) * msn / rho0_sn)**0.33   # = 0.04
    real fr = 1.
    real rsn0 = fr * rsed

    real vsn0_ = sqrt(5./3) * fr**(-1.5) * vsn0 / rsn0

  #-----------------------------------------------------------------------------


  for k in range(gp.k1, gp.k2+1):
    lind2gcrd_z(&z, k, gp)
    for j in range(gp.j1, gp.j2+1):
      lind2gcrd_y(&y, j, gp)
      for i in range(gp.i1, gp.i2+1):
        lind2gcrd_x(&x, i, gp)

        x1 = x-xc
        y1 = y-yc
        z1 = z-zc

        f = exp(- (h1 - (x1 * costh0 + y1 * sinth0)) / h0)
        W[RHO,k,j,i] = rho0 * f
        W[PR,k,j,i]  = rho0 * T0# * f

        r = sqrt(x1**2 + y1**2 + z1**2)

        if r < rsn0:

          # v = vsn0 at r = rsn0
          W[RHO,k,j,i] = rho0_sn / fr**3
          W[PR,k,j,i] = 1e4 * W[PR,k,j,i]
          W[VX,k,j,i] = vsn0_ * x1
          W[VY,k,j,i] = vsn0_ * y1
          W[VZ,k,j,i] = vsn0_ * z1



# =============================================================================

cdef void do_user_work_cons(real4d U1, real4d B1, real4d U0, real4d B0,
                       ints lims[6], BnzSim sim, double dt):

  cdef:
    GridParams gp = sim.grid.params

    ints ig1,ig2, jg,kg
    ints il1,il2, jl,kl

    ints is_on1=0, is_on2=0

    double mdot, vjet, rhojet, Tjet, ejet, vx,vy,vz, ph_prec,t_prec, th_jet
    double gamm1i = 1./(sim.phys.gam-1)


  if sim.t > 0.01 and sim.t < 0.2:

    # set locations of forward and backward-propagating jets using global indices

    ig1 = <ints>(gp.Nact_glob[0] / 2)
    ig2 = <ints>(gp.Nact_glob[0] / 2) - 1

    jg = gp.Nact_glob[1] / 2
    kg = gp.Nact_glob[2] / 2

    # check if the selected pixels are on this processor

    IF MPI:

      if (jg / gp.Nact[1] == gp.pos[1]) and (kg / gp.Nact[2] == gp.pos[2]):

        if ig1 / gp.Nact[0] == gp.pos[0]:
          is_on1 = 1
          il1 = gp.ng + ig1 % gp.Nact[0]

        if ig2 / gp.Nact[0] == gp.pos[0]:
          is_on2 = 1
          il2 = gp.ng + ig2 % gp.Nact[0]

        if is_on1 or is_on2:
          jl = gp.ng + jg % gp.Nact[1]
          IF D3D:
            kl = gp.ng + kg % gp.Nact[2]
          ELSE:
            kl = 0

    ELSE:

      is_on1 = 1
      il1 = gp.ng + ig1
      is_on2 = 1
      il2 = gp.ng + ig2
      jl = gp.ng + jg
      IF D3D:
        kl = gp.ng + kg
      ELSE:
        kl = 0

    # set jet parameters

    if is_on1 or is_on2:

      mdot  = 1e-6 / 2.3e4 * 3e5   # 1e-6  M0/yr
      vjet = 250.                 # 0.26 c
      Tjet = 1000.
      rhojet = mdot / (2 * vjet * gp.dl[1]*gp.dl[2])

      t_prec = 1./2e3

      th_jet = M_PI / 9
      ph_prec = 2*M_PI / t_prec * sim.t

      vx = vjet * cos(th_jet)
      vy = vjet * sin(th_jet) * cos(ph_prec)
      vz = vjet * sin(th_jet) * sin(ph_prec)
      # vz=0

      ejet = 0.5 * rhojet * (vx**2 + vy**2 + vz**2) + gamm1i * rhojet * Tjet


    # inject jet mass, momentum, and energy

    if is_on1:

      U1[RHO,kl,jl,il1] = rhojet
      U1[MX,kl,jl,il1] = rhojet * vx
      U1[MY,kl,jl,il1] = rhojet * vy
      U1[MZ,kl,jl,il1] = rhojet * vz
      U1[EN,kl,jl,il1] = ejet

    if is_on2:

      U1[RHO,kl,jl,il2] = rhojet
      U1[MX,kl,jl,il2] = - rhojet * vx
      U1[MY,kl,jl,il2] = - rhojet * vy
      U1[MZ,kl,jl,il2] = - rhojet * vz
      U1[EN,kl,jl,il2] = ejet


  return



# ============================================================

# Set user grid boundary conditions.

cdef void set_bc_grid_ptrs_user(BnzBC bc):
  # bc.bc_grid_funcs[0,0] = x1_bc_grid
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

  # IF PIC: output.prt_sel_func = select_particle
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

  cdef:
    real x1 = x - 0.5*Lglob[0]
    real y1 = y - 0.5*Lglob[1]

    real h1=2., h0=1.

    real costh0 = cos(M_PI/9)
    real sinth0 = sqrt(1.-costh0**2)
    real T0 = 1e-3

  return (h1 - (x1 * costh0 + y1 * sinth0)) / h0 * T0
