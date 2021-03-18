# -*- coding: utf-8 -*-

from bnz.defs cimport *

cdef class BnzDiffusion

# cdef void set_nuiic(real3d, real4d, int*, real,real) nogil
# cdef void collide_cons(real4d, int*, BnzDiffusion, real) nogil
cdef void collide(BnzDiffusion, real4d, int*,real) nogil
