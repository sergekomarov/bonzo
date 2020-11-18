# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.grid cimport *


cdef void advance_b_field(real4d,real4d,real4d, GridCoord*, ints*, real) nogil

cdef void interp_b_field(real4d, real4d, GridCoord*, ints*) nogil

cdef void e_field_cntr(real4d, real4d, ints*) nogil

cdef void interp_e_field_1(real4d, real4d, real4d,real4d,real4d, ints*) nogil

cdef void interp_e_field_2(real4d, real4d, real4d,real4d,real4d, ints*,
                           real3d, real) nogil
