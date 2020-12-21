# -*- coding: utf-8 -*-

from bnz.defs cimport *
from bnz.coordinates.coord cimport GridCoord
from bnz.coordinates.grid cimport GridData, BnzGrid
from bnz.particles.particle cimport BnzParticles
from bnz.integrate.integrator cimport BnzIntegr

cdef enum VarType:
  VAR_PRIM
  VAR_CONS

cdef class BnzIO

cdef void write_history(BnzIO, BnzGrid, BnzIntegr)
cdef void write_grid(BnzIO, BnzGrid, BnzIntegr, int)
cdef void write_slice(BnzIO, BnzGrid, BnzIntegr)
cdef void write_particles(BnzIO, BnzGrid, BnzIntegr, int)
