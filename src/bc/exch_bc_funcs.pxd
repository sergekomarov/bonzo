# -*- coding: utf-8 -*-

from bnz.defs_cy cimport *
from bnz.coord.grid cimport *

cdef class BnzSim

cdef void x1_exch_bc_periodic(BnzSim, ints1d)
cdef void x2_exch_bc_periodic(BnzSim, ints1d)
cdef void y1_exch_bc_periodic(BnzSim, ints1d)
cdef void y2_exch_bc_periodic(BnzSim, ints1d)
cdef void z1_exch_bc_periodic(BnzSim, ints1d)
cdef void z2_exch_bc_periodic(BnzSim, ints1d)

cdef void x1_exch_bc_outflow(BnzSim, ints1d)
cdef void x2_exch_bc_outflow(BnzSim, ints1d)
cdef void y1_exch_bc_outflow(BnzSim, ints1d)
cdef void y2_exch_bc_outflow(BnzSim, ints1d)
cdef void z1_exch_bc_outflow(BnzSim, ints1d)
cdef void z2_exch_bc_outflow(BnzSim, ints1d)

cdef void x1_exch_bc_reflect(BnzSim, ints1d)
cdef void x2_exch_bc_reflect(BnzSim, ints1d)
cdef void y1_exch_bc_reflect(BnzSim, ints1d)
cdef void y2_exch_bc_reflect(BnzSim, ints1d)
cdef void z1_exch_bc_reflect(BnzSim, ints1d)
cdef void z2_exch_bc_reflect(BnzSim, ints1d)

cdef void r1_exch_bc_sph(BnzSim, ints1d)
cdef void r1_exch_bc_cyl(BnzSim, ints1d)
cdef void th1_exch_bc_sph(BnzSim, ints1d)
cdef void th2_exch_bc_sph(BnzSim, ints1d)

IF MPI:
  cdef void pack_exch_all(BnzGrid, ints1d, real1d, int,int)
  cdef void unpack_exch_all(BnzGrid, ints1d, real1d, int,int)
