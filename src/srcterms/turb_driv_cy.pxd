# -*- coding: utf-8 -*-
from bnz.defs_cy cimport *
from bnz.coord.coord_cy cimport GridCoord

cdef extern from "turb_driv.h":
  void advance_driv_force_c(real****, GridCoord*, int*, real,real,int, real)

cdef class BnzTurbDriv:

  cdef:
    real f0
    real tau
    int nmod

  cdef real4d fdriv

  cdef:
    void advance_driv_force(self, GridCoord*, int*, real)
    void apply_turb_driv(self, real4d, real4d, int*, real) nogil
