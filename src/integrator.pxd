# -*- coding: utf-8 -*-

from defs_cy cimport *

cdef extern from "reconstr.h" nogil:
  void reconstr_const(real**, real**, real***, size_t, size_t,
                    int,int, real***, double)
  void reconstr_linear(real**, real**, real***, size_t, size_t,
                    int,int, real***, double)
  void reconstr_parab0(real**, real**, real***, size_t, size_t,
                    int,int, real***, double)
  void reconstr_parab(real**, real**, real***, size_t, size_t,
                    int,int, real***, double)
  void reconstr_weno(real**, real**, real***, size_t, size_t,
                    int,int, real***, double)

cdef extern from "fluxes.h" nogil:
  void HLLflux(real**, real**, real**, real*, size_t, size_t, double)
  void HLLTflux(real**, real**, real**, real*, size_t, size_t, double)

IF not MFIELD:
  cdef extern from "fluxes.h" nogil:
    void HLLCflux(real**, real**, real**, real*, size_t, size_t, double)

IF MFIELD:
  cdef extern from "fluxes.h" nogil:
    void HLLDflux(real**, real**, real**, real*, size_t, size_t, double)

IF CGL:
  cdef extern from "fluxes.h" nogil:
    void HLLAflux(real**, real**, real**, real*, size_t, size_t, double)


#==========================================================================

IF not PIC:

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
              real**, real**, real**, real*,   # &F, WL, WR, Bx
              size_t, size_t,                  # start/end indices
              real) nogil                      # gam

  # pointer to limiter function
  ctypedef real (*LimiterFunc)(real, real) nogil

  # pointer to reconstruction function
  ctypedef void (*ReconstrFunc)(
              real**, real**,     # return: wR(i-1/2), wL(i+1/2)
              real***,            # array of variables along one axis
              size_t,size_t,      # start/end indices
              int,                # axis
              #LimiterFunc,       # limiter function
              int,                # characteristic projection on/off
              real***,            # scratch array
              real              # gas gamma
              ) nogil

IF MHDPIC:
  # function pointer to particle interpolation kernel
  ctypedef void (*WeightFunc)(ints*, ints*, ints*,
                      real, real, real, real***,
                      real, int) nogil


# Integrator class.

cdef class BnzIntegrator:

  # Attributes.

  # Courant number
  cdef real cour

  IF not PIC:

    cdef:

      # parabolic Courant number
      real cour_diff

      # time integrator
      TIntegr tintegr

      # Riemann solver
      RSolver rsolver
      RSolverFunc rsolver_func

      # reconstruction
      Reconstr reconstr
      ReconstrFunc reconstr_func
      ints reconstr_order

      # limiter used for resonstruction
      # Limiter limiter
      # LimiterFunc limiter_func

      # limiting in characteristic variables on/off
      int charact_proj

      # pressure floor
      real pressure_floor
      # density floor
      real rho_floor

      # super-time-stepping on(1)/off(0)
      int sts

  IF MHDPIC:
    # for pure PIC interpolation is trilinear
    cdef int Ninterp              # order of interpolation (for MHDPIC)
    cdef WeightFunc weight_func   # pointer to kernel weight function

  IF PIC:
    # number of passes of current filter
    cdef int Nfilt

  # Functions.

  cdef void init_data(self)
