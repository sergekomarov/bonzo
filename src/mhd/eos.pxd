# -*- coding: utf-8 -*-

from bnz.defs cimport *

cdef extern from "eos_c.h" nogil:
  void cons2prim_1(real**, real**, int, int, real)
  void prim2cons_1(real**, real**, int, int, real)

cdef void cons2prim_3(real4d, real4d, int[6], real) nogil
cdef void prim2cons_3(real4d, real4d, int[6], real) nogil
