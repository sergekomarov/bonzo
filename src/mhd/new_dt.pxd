# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.coord_cy cimport GridCoord
from bnz.mhd.integrator cimport BnzIntegr

cdef extern from "eos.h" nogil:
  real fms(real*, real, real)

cdef real new_dt(real4d, GridCoord*, BnzIntegr)
