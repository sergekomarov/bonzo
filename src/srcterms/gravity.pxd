# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.coord_cy cimport GridCoord

ctypedef real (*GravPotFunc)(real,real,real, real, real[3]) nogil

cdef class BnzGravity:

  cdef:
    real g0
    real3d grav_pot
    GravPotFunc grav_pot_func

  cdef void apply_gravity(self, real4d, real4d, real4d,real4d,real4d,
                          GridCoord*, int *lims, real) nogil
