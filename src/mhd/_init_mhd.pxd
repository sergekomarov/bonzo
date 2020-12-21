# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.data_struct cimport *

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


cdef void init(BnzSim, bytes)
