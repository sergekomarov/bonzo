# -*- coding: utf-8 -*-

from bnz.defs cimport *
from bnz.coordinates.coord cimport GridCoord

ctypedef real (*GravPotFunc)(real,real,real, real) nogil

cdef class BnzGravity:

  cdef:
    real g0[3]
    int const_g
    real3d grav_pot
    GravPotFunc grav_pot_func

  cdef post_user_init(self, GridCoord*)

  cdef void apply_gravity(self, real4d, real4d, real4d,real4d,real4d,
                          GridCoord*, int*, real) nogil
