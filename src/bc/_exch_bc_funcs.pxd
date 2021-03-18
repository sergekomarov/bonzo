# -*- coding: utf-8 -*-

from bnz.defs cimport *
from bnz.coordinates.grid cimport GridData
from bnz.coordinates.coord cimport GridCoord
cdef class BnzIntegr

cdef void x1_exch_bc_periodic(GridData,GridCoord*, BnzIntegr, int1d)
cdef void x2_exch_bc_periodic(GridData,GridCoord*, BnzIntegr, int1d)
cdef void y1_exch_bc_periodic(GridData,GridCoord*, BnzIntegr, int1d)
cdef void y2_exch_bc_periodic(GridData,GridCoord*, BnzIntegr, int1d)
cdef void z1_exch_bc_periodic(GridData,GridCoord*, BnzIntegr, int1d)
cdef void z2_exch_bc_periodic(GridData,GridCoord*, BnzIntegr, int1d)

cdef void x1_exch_bc_outflow(GridData,GridCoord*, BnzIntegr, int1d)
cdef void x2_exch_bc_outflow(GridData,GridCoord*, BnzIntegr, int1d)
cdef void y1_exch_bc_outflow(GridData,GridCoord*, BnzIntegr, int1d)
cdef void y2_exch_bc_outflow(GridData,GridCoord*, BnzIntegr, int1d)
cdef void z1_exch_bc_outflow(GridData,GridCoord*, BnzIntegr, int1d)
cdef void z2_exch_bc_outflow(GridData,GridCoord*, BnzIntegr, int1d)

cdef void x1_exch_bc_reflect(GridData,GridCoord*, BnzIntegr, int1d)
cdef void x2_exch_bc_reflect(GridData,GridCoord*, BnzIntegr, int1d)
cdef void y1_exch_bc_reflect(GridData,GridCoord*, BnzIntegr, int1d)
cdef void y2_exch_bc_reflect(GridData,GridCoord*, BnzIntegr, int1d)
cdef void z1_exch_bc_reflect(GridData,GridCoord*, BnzIntegr, int1d)
cdef void z2_exch_bc_reflect(GridData,GridCoord*, BnzIntegr, int1d)

cdef void r1_exch_bc_sph(GridData,GridCoord*, BnzIntegr, int1d)
cdef void r1_exch_bc_cyl(GridData,GridCoord*, BnzIntegr, int1d)
cdef void th1_exch_bc_sph(GridData,GridCoord*, BnzIntegr, int1d)
cdef void th2_exch_bc_sph(GridData,GridCoord*, BnzIntegr, int1d)

IF MPI:
  cdef void pack_exch_all(GridData,GridCoord*, int1d, real1d, int,int)
  cdef void unpack_exch_all(GridData,GridCoord*, int1d, real1d, int,int)
