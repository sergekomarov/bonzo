# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.grid cimport *


cdef extern from "reconstr.h" nogil:
  void reconstr_const(real**, real**, real***, real ***,
                      GridCoord*, int,
                      ints,ints,ints,ints,
                      int, real)
  void reconstr_linear(real**, real**, real***, real ***,
                       GridCoord*, int,
                       ints,ints,ints,ints,
                       int, real)
  # void reconstr_parab0(real**, real**, real***, real ***,
  #                      GridCoord*, int,
  #                      ints,ints,ints,ints,
  #                      int, real)
  void reconstr_parab(real**, real**, real***, real ***,
                       GridCoord*, int,
                       ints,ints,ints,ints,
                       int, real)
  void reconstr_weno(real**, real**, real***, real ***,
                     GridCoord*, int,
                     ints,ints,ints,ints,
                     int, real)

cdef extern from "fluxes.h" nogil:
  void hll_flux(real**, real**, real**, real*, ints,ints, real)
  void hllt_flux(real**, real**, real**, real*, ints,ints, real)

IF MFIELD:
  cdef extern from "fluxes.h" nogil:
    void hlld_flux(real**, real**, real**, real*, ints,ints, real)
ELSE:
  cdef extern from "fluxes.h" nogil:
    void hllc_flux(real**, real**, real**, real*, ints,ints, real)

IF CGL:
  cdef extern from "fluxes.h" nogil:
    void hlla_flux(real**, real**, real**, real*, ints,ints, real)


#==========================================================================

# Identifiers.

# time integrators
ctypedef enum TIntegr:
  TINT_VL
  TINT_RK3

# Riemann solvers
ctypedef enum RSolver:
  RS_HLL
  RS_HLLC
  RS_HLLD
  RS_HLLA
  RS_HLLT

# spacial reconstruction
ctypedef enum Reconstr:
  RCN_CONST
  RCN_LINEAR
  RCN_WENO
  RCN_PARAB

# limiters
# ctypedef enum Limiter:
#   LIM_MM
#   LIM_MC
#   LIM_VL
#   LIM_L2

# Function pointers.

# function pointer to Riemann solver
ctypedef void (*RSolverFunc)(
            real**, real**, real**, real*,   # &flux, wl, wr, bx
            ints, ints,                      # start/end x-indices
            real) nogil                      # gas gamma

# pointer to limiter function
# ctypedef real (*LimiterFunc)(real, real) nogil

# pointer to reconstruction function
ctypedef void (*ReconstrFunc)(
            real**, real**,     # return wR(i-1/2), wL(i+1/2)
            real***,            # array of variables along x-axis
            real***,            # scratch arrays
            int,                # orientation of cell interfaces
            ints,ints,          # start/end x-indices
            ints,ints,          # y- and z-indices of the slice along x
            int,                # characteristic projection on/off
            real                # gas gamma
            ) nogil

IF MHDPIC:
  # function pointer to particle interpolation kernel
  ctypedef void (*WeightFunc)(ints*, ints*, ints*,
                      real, real, real, real***,
                      real, int) nogil

# Array structures used by the integrator.

cdef class IntegrData:

  cdef:

    real4d ec           # cell-centered electric field
    real4d ee           # edge-centered electric field

    real4d flux_x, flux_y, flux_z   # Godunov fluxes

    real4d cons_s       # predictor-step arrays of cell-centered conserved variables
    real4d cons_ss

    real4d bf_s         # predictor-step arrays of face-centered magnetic field
    real4d bf_ss

    real4d bf_init      # initial magnetic field
    real3d phi          # static gravitational potential
    real4d fdriv        # driving force
    real3d nuii_eff     # effective ion collision rate

    IF MHDPIC:
      cdef real4d fcoup_tmp

# Scatch arrays used by reconstruction and Riemann solver C routines.

cdef class IntegrScratch:

  cdef:
    real ****w_rcn
    real ***wl
    real ***wr
    real ***wl_

# cdef class IntegrPhys:
#
#   cdef real gam


# Integrator class.

cdef class BnzIntegr:

  cdef:

    # Courant number
    real cour

    # time integrator
    TIntegr tintegr

    # Riemann solver
    RSolver rsolver
    RSolverFunc rsolver_func

    # reconstruction
    Reconstr reconstr
    ReconstrFunc reconstr_func
    int reconstr_order

    # limiter used for resonstruction
    # Limiter limiter
    # LimiterFunc limiter_func

    IntegrData data
    IntegrScratch scratch

    # limiting in characteristic variables on/off
    int charact_proj

    # pressure floor
    real pressure_floor
    # density floor
    real rho_floor

    # parabolic Courant number
    real cour_diff

    # super-time-stepping on(1)/off(0)
    int sts

    # gas gamma
    real gam

  IF MHDPIC:

    cdef:

      int Ninterp              # order of interpolation (for MHDPIC)
      WeightFunc weight_func   # pointer to kernel weight function

      real sol        # effective speed of light
      real q_mc       # charge-to-mass ratio of CRs relative to thermal ions
      real rho_cr     # CR density


  # Allocate space for integration arrays.
  cdef void init(self, GridCoord*)



# # function pointer to gravitational potential
# ctypedef real (*GravPotFunc)(real,real,real, real, real[3]) nogil
#
# # function pointer to electron thermal conductivity
# ctypedef real (*ThcondElecFunc)(real,real,real, real, real[3]) nogil
#
# # (an)isotropic electron thermal conduction
# ctypedef enum ThcondType:
#   TC_ISO
#   TC_ANISO
