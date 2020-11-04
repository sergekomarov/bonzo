# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *

cdef extern from "eos.h" nogil:
  void cons2prim_1(real**, real**, ints, ints, real)
  void prim2cons_1(real**, real**, ints, ints, real)

cdef void cons2prim_3(real4d, real4d, ints[6], real) nogil
cdef void prim2cons_3(real4d, real4d, ints[6], real) nogil
