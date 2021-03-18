# -*- coding: utf-8 -*-

from bnz.defs cimport *
from bnz.coordinates.coord cimport GridCoord
from bnz.coordinates.grid cimport BnzGrid,GridData

cdef class BnzDiffusion

cdef void diffuse(BnzDiffusion, BnzGrid, real)
