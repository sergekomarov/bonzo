# -*- coding: utf-8 -*-

from bnz.defs cimport *
from bnz.coordinates.coord cimport GridCoord
from bnz.coordinates.grid cimport GridData, BnzGrid
from bnz.particles.particle cimport BnzParticles
from bnz.integrate.integrator cimport BnzIntegr

cdef class BnzIO

cdef void set_restart_grid(BnzIO, BnzGrid, BnzIntegr)
IF MHDPIC: cdef void set_restart_particles(BnzIO, BnzGrid, BnzIntegr)
