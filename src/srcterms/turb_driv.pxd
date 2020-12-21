# -*- coding: utf-8 -*-
from bnz.defs cimport *
from bnz.coordinates.coord cimport GridCoord

cdef extern from "turb_driv_c.h":
  void advance_driv_force_i(real*, real*, real*,
                            real*, real*,
                            real*, real, real, int,int,
                            real, real, real,
                            real, real, int)

cdef class BnzTurbDriv:

  cdef:
    real f0
    real tau
    int nmod

  cdef real4d fdriv

  cdef:
    void advance_driv_force(self, GridCoord*, int*, real)
    void apply_driv_force(self, real4d, real4d, int*, real) nogil
