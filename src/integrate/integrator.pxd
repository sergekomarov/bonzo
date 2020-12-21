# -*- coding: utf-8 -*-

from bnz.defs cimport *
from bnz.coordinates.coord cimport GridCoord


cdef extern from "reconstr_c.h" nogil:
  void reconstr_const(real**, real**, real***, real ***,
                      GridCoord*, int,
                      int,int,int,int,
                      int, real)
  void reconstr_linear(real**, real**, real***, real ***,
                       GridCoord*, int,
                       int,int,int,int,
                       int, real)
  # void reconstr_parab0(real**, real**, real***, real ***,
  #                      GridCoord*, int,
  #                      int,int,int,int,
  #                      int, real)
  void reconstr_parab(real**, real**, real***, real ***,
                       GridCoord*, int,
                       int,int,int,int,
                       int, real)
  void reconstr_weno(real**, real**, real***, real ***,
                     GridCoord*, int,
                     int,int,int,int,
                     int, real)

cdef extern from "fluxes_c.h" nogil:
  void hll_flux(real**, real**, real**, real*, int,int, real)
  void hllt_flux(real**, real**, real**, real*, int,int, real)

IF MFIELD:
  cdef extern from "fluxes_c.h" nogil:
    void hlld_flux(real**, real**, real**, real*, int,int, real)
ELSE:
  cdef extern from "fluxes_c.h" nogil:
    void hllc_flux(real**, real**, real**, real*, int,int, real)

IF CGL:
  cdef extern from "fluxes_c.h" nogil:
    void hlla_flux(real**, real**, real**, real*, int,int, real)


# -----------------------------------------------------------------

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


# Function pointers.

# function pointer to Riemann solver
ctypedef void (*RSolverFunc)(
            real**, real**, real**, real*,   # &flux, wl, wr, bx
            int, int,                        # start/end x-indices
            real) nogil                      # gas gamma

# pointer to limiter function
# ctypedef real (*LimiterFunc)(real, real) nogil

# pointer to reconstruction function
ctypedef void (*ReconstrFunc)(
            real**, real**,     # return wR(i-1/2), wL(i+1/2)
            real***,            # array of variables along x-axis
            real***,            # scratch arrays
            int,                # orientation of cell interfaces
            int,int,            # start/end x-indices
            int,int,            # y- and z-indices of the slice along x
            int,                # characteristic projection on/off
            real                # gas gamma
            ) nogil


# Array structures used by the integrator.

cdef class IntegrData:

  cdef:

    real4d efldc       # cell-centered electric field
    real4d eflde       # edge-centered electric field

    real4d flx_x    # Godunov fluxes
    real4d flx_y
    real4d flx_z

    real4d cons_s   # predictor-step arrays of cell-centered conserved variables
    real4d cons_ss

    real4d bfld_s     # predictor-step arrays of face-centered magnetic field
    real4d bfld_ss


# Scatch arrays used by reconstruction and Riemann solver.

cdef class IntegrScratch:

  cdef:
    real ****w_rcn
    real ***wl
    real ***wr
    real ***wl_


# Integrator class.

cdef class BnzIntegr:

  cdef IntegrData data
  cdef IntegrScratch scratch

  cdef BnzGravity gravity
  cdef BnzTurbDriv turb_driv

  cdef:

    # current time
    real time

    # current step number
    long step

    # current timestep
    real dt

    # maximum time
    real tmax

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

    # limiting in characteristic variables on/off
    int char_proj

    # pressure floor
    real pressure_floor
    # density floor
    real rho_floor

  cdef:
    real gam      # gas gamma
    real sol      # effective speed of light
    real q_mc     # charge-to-mass ratio of CRs relative to thermal ions
    real rho_cr   # CR density


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
