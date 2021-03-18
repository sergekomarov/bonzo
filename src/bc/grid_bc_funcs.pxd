# -*- coding: utf-8 -*-

from bnz.defs cimport *
from bnz.coordinates.grid cimport GridData
from bnz.coordinates.coord cimport GridCoord
cdef class BnzIntegr

cdef void x1_grid_bc_periodic(GridData,GridCoord*, BnzIntegr, int1d)
cdef void x2_grid_bc_periodic(GridData,GridCoord*, BnzIntegr, int1d)
cdef void y1_grid_bc_periodic(GridData,GridCoord*, BnzIntegr, int1d)
cdef void y2_grid_bc_periodic(GridData,GridCoord*, BnzIntegr, int1d)
cdef void z1_grid_bc_periodic(GridData,GridCoord*, BnzIntegr, int1d)
cdef void z2_grid_bc_periodic(GridData,GridCoord*, BnzIntegr, int1d)

cdef void x1_grid_bc_outflow(GridData,GridCoord*, BnzIntegr, int1d)
cdef void x2_grid_bc_outflow(GridData,GridCoord*, BnzIntegr, int1d)
cdef void y1_grid_bc_outflow(GridData,GridCoord*, BnzIntegr, int1d)
cdef void y2_grid_bc_outflow(GridData,GridCoord*, BnzIntegr, int1d)
cdef void z1_grid_bc_outflow(GridData,GridCoord*, BnzIntegr, int1d)
cdef void z2_grid_bc_outflow(GridData,GridCoord*, BnzIntegr, int1d)

cdef void x1_grid_bc_reflect(GridData,GridCoord*, BnzIntegr, int1d)
cdef void x2_grid_bc_reflect(GridData,GridCoord*, BnzIntegr, int1d)
cdef void y1_grid_bc_reflect(GridData,GridCoord*, BnzIntegr, int1d)
cdef void y2_grid_bc_reflect(GridData,GridCoord*, BnzIntegr, int1d)
cdef void z1_grid_bc_reflect(GridData,GridCoord*, BnzIntegr, int1d)
cdef void z2_grid_bc_reflect(GridData,GridCoord*, BnzIntegr, int1d)

cdef void r1_grid_bc_sph(GridData,GridCoord*, BnzIntegr, int1d)
cdef void r1_grid_bc_cyl(GridData,GridCoord*, BnzIntegr, int1d)
cdef void th1_grid_bc_sph(GridData,GridCoord*, BnzIntegr, int1d)
cdef void th2_grid_bc_sph(GridData,GridCoord*, BnzIntegr, int1d)

IF MPI:
  cdef void pack_grid_all(GridData,GridCoord*, int1d, real1d, int,int)
  cdef void unpack_grid_all(GridData,GridCoord*, int1d, real1d, int,int)
