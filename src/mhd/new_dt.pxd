# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.grid cimport *
from integrator cimport *

cdef extern from "eos.h" nogil:
  real fms(real*, real, real)

cdef double new_dt(real4d, GridCoord*, BnzIntegr)
