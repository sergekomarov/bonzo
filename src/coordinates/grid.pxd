# -*- coding: utf-8 -*-

from bnz.defs cimport *
from coord cimport GridCoord

cdef class GridData:

  cdef:
    real4d cons         # cell-centered conserved variables
    real4d prim         # cell-centered primitive variables
    real4d bfld         # face-centered magnetic field
    real4d bfld_init    # initial magnetic field
    real4d fcoup        # particle feedback array


cdef class GridBC
cdef class BnzIntegr
IF MHDPIC: cdef class BnzParticles

cdef class BnzGrid:

  cdef:
    GridCoord *coord       # grid coordinates
    GridData data          # grid data
    GridBC bc              # boundary conditions
  IF MHDPIC:
    cdef BnzParticles prts    # particles

  cdef str usr_dir       # user directory, contains config file

  cdef void cons2prim(self, int*, real)
  cdef void prim2cons(self, int*, real)

  cdef void apply_grid_bc(self, BnzIntegr, int1d)
  IF MHDPIC: cdef void apply_prt_bc(self, BnzIntegr)
