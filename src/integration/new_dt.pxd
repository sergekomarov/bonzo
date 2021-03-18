# -*- coding: utf-8 -*-

from bnz.defs cimport *
from bnz.coordinates.coord cimport GridCoord
from bnz.integrate.integrator cimport BnzIntegr

cdef extern from "eos_c.h" nogil:
  real fms(real*, real, real)

cdef real new_dt(real4d, GridCoord*, BnzIntegr)
