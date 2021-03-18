# -*- coding: utf-8 -*-
from bnz.defs cimport *
from bnz.coordinates.grid cimport BnzGrid
from bnz.coordinates.coord cimport GridCoord
from integrator cimport BnzIntegr
from bnz.io cimport BnzIO

cdef void init_sim(BnzGrid, BnzIntegr, BnzIO, str)
cdef void advance(BnzGrid, BnzIntegr, BnzIO, real)
