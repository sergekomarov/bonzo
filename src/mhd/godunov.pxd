# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.grid cimport *
from bnz.mhd.integrator cimport *

cdef void godunov_fluxes(real4d, real4d, real4d,
                         real4d, real4d, ints*, GridCoord*,
                         BnzIntegr, ints)

cdef void advance_hydro(real4d, real4d, real4d,real4d,real4d,
                        GridCoord*, ints*, real) nogil

# cdef void apply_pressure_floor(real4d, ints[6], double,double) nogil
