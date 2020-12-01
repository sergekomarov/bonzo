# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.grid cimport *

cdef class BnzSim

cdef void x1_grid_bc_periodic(BnzSim, ints1d)
cdef void x2_grid_bc_periodic(BnzSim, ints1d)
cdef void y1_grid_bc_periodic(BnzSim, ints1d)
cdef void y2_grid_bc_periodic(BnzSim, ints1d)
cdef void z1_grid_bc_periodic(BnzSim, ints1d)
cdef void z2_grid_bc_periodic(BnzSim, ints1d)

cdef void x1_grid_bc_outflow(BnzSim, ints1d)
cdef void x2_grid_bc_outflow(BnzSim, ints1d)
cdef void y1_grid_bc_outflow(BnzSim, ints1d)
cdef void y2_grid_bc_outflow(BnzSim, ints1d)
cdef void z1_grid_bc_outflow(BnzSim, ints1d)
cdef void z2_grid_bc_outflow(BnzSim, ints1d)

cdef void x1_grid_bc_reflect(BnzSim, ints1d)
cdef void x2_grid_bc_reflect(BnzSim, ints1d)
cdef void y1_grid_bc_reflect(BnzSim, ints1d)
cdef void y2_grid_bc_reflect(BnzSim, ints1d)
cdef void z1_grid_bc_reflect(BnzSim, ints1d)
cdef void z2_grid_bc_reflect(BnzSim, ints1d)

cdef void r1_grid_bc_sph(BnzSim, ints1d)
cdef void r1_grid_bc_cyl(BnzSim, ints1d)
cdef void th1_grid_bc_sph(BnzSim, ints1d)
cdef void th2_grid_bc_sph(BnzSim, ints1d)

IF MPI:
  cdef void pack_grid_all(BnzGrid, ints1d, real1d, int,int)
  cdef void unpack_grid_all(BnzGrid, ints1d, real1d, int,int)
