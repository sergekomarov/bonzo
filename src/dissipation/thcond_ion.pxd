# -*- coding: utf-8 -*-

from bnz.defs cimport *
from bnz.coordinates.coord cimport GridCoord
from bnz.coordinates.grid cimport BnzGrid,GridData
from bnz.integrate.integrator cimport BnzIntegr

cdef void diffuse(BnzGrid,BnzIntegr, real)
