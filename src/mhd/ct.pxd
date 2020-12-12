# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.grid cimport GridCoord


cdef void advance_b(real4d,real4d,real4d, GridCoord*, int*, real) nogil

cdef void interp_bc(real4d, real4d, GridCoord*, int*) nogil

cdef void ec_from_prim(real4d, real4d, int*) nogil

cdef void interp_ee1(real4d, real4d, real4d,real4d,real4d, int*) nogil

cdef void interp_ee2(real4d, real4d, real4d,real4d,real4d, int*,
                     real3d, real) nogil
